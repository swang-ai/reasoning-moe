from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moe_steer.data_utils import PHASE_LABELS, LABEL_ALIASES
from moe_steer.select_experts import write_class_pkls
from moe_steer.models import get_model_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-layers", type=int)
    ap.add_argument("--num-experts", type=int)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--min-tokens", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    try:
        s = get_model_spec(args.model)
        L = args.num_layers or s.num_layers
        E = args.num_experts or s.num_experts
        K = args.top_k or s.top_k
    except Exception:
        L, E, K = args.num_layers, args.num_experts, args.top_k
        assert L and E and K, "pass --num-layers/--num-experts/--top-k"

    dev = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    labels = list(PHASE_LABELS)
    l2i = {l: i for i, l in enumerate(labels)}
    C = len(labels)
    keep = set(labels)

    count = torch.zeros(L * E, C, dtype=torch.float64, device=dev)
    Nc = torch.zeros(C, dtype=torch.float64, device=dev)
    layer_off = (torch.arange(L, device=dev).view(L, 1) * E)

    traces_dir, labels_dir = Path(args.traces), Path(args.labels)
    lps = sorted(labels_dir.glob("*.json"))
    t0 = time.time()
    n_tr = 0
    for i, lp in enumerate(lps):
        pt = traces_dir / f"{lp.stem}.pt"
        if not pt.exists():
            continue
        nodes = json.loads(lp.read_text()).get("nodes", [])
        if not nodes:
            continue
        d = torch.load(pt, weights_only=False)
        ids = d["topk_ids"]
        T = ids.shape[1]
        tk = ids.to(dev, dtype=torch.long).permute(1, 0, 2).contiguous()
        glob_idx = (tk + layer_off.unsqueeze(0)).reshape(T, -1)
        for node in nodes:
            a, b = node["token_start"], min(node["token_end"], T)
            lbl = LABEL_ALIASES.get(str(node["label"]).strip().lower(),
                                    str(node["label"]).strip().lower())
            if a >= b or lbl not in keep:
                continue
            c = l2i[lbl]
            flat = glob_idx[a:b].reshape(-1)
            count[:, c] += torch.bincount(flat, minlength=L * E).double()
            Nc[c] += (b - a)
        n_tr += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(lps)} traces  {time.time() - t0:.1f}s  "
                  f"tokens={int(Nc.sum())}", flush=True)

    count = count.cpu().numpy().reshape(L, E, C)
    Nc = Nc.cpu().numpy()
    N_total = float(Nc.sum())
    total_active = count.sum(axis=2)

    used = [l for l in labels if Nc[l2i[l]] >= max(1, args.min_tokens)]
    dropped = [(l, int(Nc[l2i[l]])) for l in labels if l not in used]
    print(f"loaded {int(N_total):,} tokens from {n_tr} traces  ({time.time()-t0:.1f}s)",
          flush=True)
    print(f"classes kept: {[(l, int(Nc[l2i[l]])) for l in used]}", flush=True)
    if dropped:
        print(f"classes dropped (<{args.min_tokens} tok): {dropped}", flush=True)

    risk = np.zeros((L, E, len(used)), dtype=np.float32)
    for j, l in enumerate(used):
        c = l2i[l]
        nc = Nc[c]
        nr = N_total - nc
        p_c = count[:, :, c] / max(nc, 1.0)
        p_r = (total_active - count[:, :, c]) / max(nr, 1.0)
        risk[:, :, j] = (p_c - p_r).astype(np.float32)

    write_class_pkls(args.out_dir, used, risk_table=risk, top_k=K)
    print(f"model={args.model}  geometry={L}x{E} top-{K}", flush=True)
    print(f"wrote pickles to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
