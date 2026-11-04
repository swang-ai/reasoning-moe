from __future__ import annotations

import pandas as pd

_MOE_BLOCK_CLASSES = {
    "Qwen3MoeSparseMoeBlock",
    "Qwen2MoeSparseMoeBlock",
    "Qwen3NextSparseMoeBlock",
    "DeepseekV2MoE", "DeepseekV3MoE",
    "MixtralMoE",
    "MLPBlock",
}


def register_vllm_models():
    return


def _worker_set_steer(self, num_layers, num_experts, pos_le, neg_le, moe_classes,
                      prune_le=()):
    import torch

    model = self.model_runner.model
    model = getattr(model, "language_model", model)
    cfg = getattr(model, "config", None)
    top_k = int(getattr(cfg, "num_experts_per_tok", 8)) if cfg else 8

    import re
    _ROUTER_CLASSES = {"Gemma4Router"}
    blocks = []
    for name, mod in model.named_modules():
        cn = mod.__class__.__name__
        g = None
        if cn in moe_classes and (hasattr(mod, "gate") or hasattr(mod, "router")):
            g = getattr(mod, "gate", None)
            if g is None:
                g = mod.router
        elif cn in _ROUTER_CLASSES:
            g = mod
        if g is None:
            continue
        m = re.search(r"layers\.(\d+)", name)
        blocks.append((int(m.group(1)) if m else len(blocks), g))
    blocks.sort(key=lambda x: x[0])

    mat = torch.zeros((num_layers, num_experts), dtype=torch.float32)
    for (l, e) in neg_le:
        mat[l, e] = -1.0
    for (l, e) in prune_le:
        mat[l, e] = -2.0
    per_layer_pos = {}
    for (l, e) in pos_le:
        c = per_layer_pos.get(l, 0)
        if c < top_k:
            mat[l, e] = 1.0
            per_layer_pos[l] = c + 1

    import os as _os
    eps = float(_os.environ.get("STEER_EPS", "0.01"))

    def _make_hook():
        def hook(module, inp, out):
            vec = getattr(module, "_steer_vec", None)
            if vec is None:
                return out
            lg = out[0] if isinstance(out, tuple) else out
            v = vec.to(lg.device)
            pos = v > 0
            neg = (v < 0) & (v > -1.5)
            prune = v <= -1.5
            if pos.any() or neg.any() or prune.any():
                lg = torch.log_softmax(lg.float(), dim=-1)
                mx = lg.max(dim=-1, keepdim=True).values
                mn = lg.min(dim=-1, keepdim=True).values
                if pos.any():
                    lg[:, pos] = mx + eps
                if neg.any():
                    lg[:, neg] = mn - eps
                if prune.any():
                    lg[:, prune] = float("-inf")
            return (lg,) + tuple(out[1:]) if isinstance(out, tuple) else lg
        return hook

    n_set = 0
    for compact_idx, (li, g) in enumerate(blocks):
        if not hasattr(g, "_steer_hook_handle"):
            g._steer_hook_handle = g.register_forward_hook(_make_hook())
        row = mat[compact_idx] if compact_idx < num_layers else None
        g._steer_vec = row if (row is not None and bool((row != 0).any())) else None
        n_set += 1
    n_boost = int((mat > 0).sum())
    n_supp = int(((mat < 0) & (mat > -1.5)).sum())
    n_prune = int((mat <= -1.5).sum())
    return (f"blocks={n_set} boosted={n_boost} suppressed={n_supp} "
            f"pruned={n_prune} top_k_cap={top_k}")


def steer_moe(llm, activations_path, num_pos_experts, num_neg_experts,
              reverse_effect: bool = False, strategy: str = "risk_diff",
              steering_magnitude=1000, prune: bool = False):
    df = pd.read_pickle(activations_path)
    num_layers = int(df["layer"].max()) + 1
    num_experts = int(df["expert"].max()) + 1
    metric = "risk_diff"
    df = df.sort_values("risk_diff_abs", ascending=False).reset_index(drop=True)
    if reverse_effect:
        df[metric] = -df[metric]

    pos = df[df[metric] > 0].head(num_pos_experts)
    neg = df[df[metric] < 0].head(num_neg_experts)
    pos_le = [(int(r.layer), int(r.expert)) for r in pos.itertuples()]
    neg_le = [(int(r.layer), int(r.expert)) for r in neg.itertuples()]
    prune_le = neg_le if prune else []
    if prune:
        neg_le = []

    out = llm.collective_rpc(_worker_set_steer,
                             args=(num_layers, num_experts, pos_le, neg_le,
                                   list(_MOE_BLOCK_CLASSES), prune_le))
    print(f"[steer_native] {out[0] if out else 'applied'}  "
          f"(pos={len(pos_le)} neg={len(neg_le)} prune={len(prune_le)})")
    return df
