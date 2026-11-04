from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from moe_steer.data_utils import PHASE_LABELS, LABEL_ALIASES
from moe_steer.select_experts import write_class_pkls
from moe_steer.models import get_model_spec


def load_labeled_topk(traces_dir, labels_dir, num_layers, num_experts,
                      labels=None, min_tokens=0, limit=None):
    labels = list(labels or PHASE_LABELS)
    keep = set(labels)
    traces_dir, labels_dir = Path(traces_dir), Path(labels_dir)
    topks, lbls = [], []
    n_traces = 0
    for lp in sorted(labels_dir.glob("*.json")):
        if limit is not None and n_traces >= limit:
            break
        pt = traces_dir / f"{lp.stem}.pt"
        if not pt.exists():
            continue
        nodes = json.loads(lp.read_text()).get("nodes", [])
        if not nodes:
            continue
        d = torch.load(pt, weights_only=False)
        ids = d["topk_ids"].numpy().astype(np.int64)
        T = ids.shape[1]
        ids_tlk = ids.transpose(1, 0, 2)
        for node in nodes:
            t0, t1 = node["token_start"], min(node["token_end"], T)
            lbl = str(node["label"]).strip().lower()
            lbl = LABEL_ALIASES.get(lbl, lbl)
            if t0 >= t1 or lbl not in keep:
                continue
            topks.append(ids_tlk[t0:t1])
            lbls.append([lbl] * (t1 - t0))
        n_traces += 1
    if not topks:
        raise SystemExit(f"No labeled tokens under {traces_dir} / {labels_dir}")
    topk = np.concatenate(topks).astype(np.int64).reshape(-1, num_layers, ids.shape[2])
    flat = np.array([x for seg in lbls for x in seg])

    counts = Counter(flat)
    used = [l for l in labels if counts.get(l, 0) >= max(1, min_tokens)]
    dropped = [(l, counts.get(l, 0)) for l in labels if l not in used]
    l2i = {l: i for i, l in enumerate(used)}
    mask = np.array([l in l2i for l in flat])
    topk = topk[mask]
    y = np.array([l2i[l] for l in flat[mask]], dtype=np.int64)
    print(f"loaded {len(y):,} tokens from {n_traces} traces  (topk {topk.shape})", flush=True)
    print(f"classes kept: {[(l, int(counts[l])) for l in used]}", flush=True)
    if dropped:
        print(f"classes dropped (<{min_tokens} tok): {dropped}", flush=True)
    return topk, y, used


def risk_diff_bincount(topk, y, num_layers, num_experts, num_labels, device="cpu"):
    L, E, C = num_layers, num_experts, num_labels
    tk = torch.as_tensor(topk, device=device)
    yt = torch.as_tensor(y, device=device)
    N = tk.shape[0]
    layer_off = (torch.arange(L, device=device).view(1, L, 1) * E)
    def counts_for(mask):
        sub = tk[mask]
        flat = (sub + layer_off).reshape(-1)
        return torch.bincount(flat, minlength=L * E).float().reshape(L, E)
    risk = torch.zeros(L, E, C)
    for c in range(C):
        m = (yt == c)
        nc = int(m.sum())
        nr = N - nc
        p_c = counts_for(m) / max(nc, 1)
        p_r = counts_for(~m) / max(nr, 1)
        risk[..., c] = (p_c - p_r).cpu()
    return risk.numpy()


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
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--self-check", type=int, default=0,
                    help="compare against numpy reference on first N traces (min_tokens=0)")
    args = ap.parse_args()

    try:
        s = get_model_spec(args.model)
        L = args.num_layers or s.num_layers
        E = args.num_experts or s.num_experts
        K = args.top_k or s.top_k
    except Exception:
        L, E, K = args.num_layers, args.num_experts, args.top_k
        assert L and E and K, "pass --num-layers/--num-experts/--top-k"

    if args.self_check:
        from moe_steer.data_utils import load_labeled_traces
        from moe_steer.select_experts import compute_risk_diff
        print(f"== self-check on first {args.self_check} traces (min_tokens=0) ==", flush=True)
        router, y_ref, used_ref = load_labeled_traces(
            args.traces, args.labels, L, E, min_tokens=0)
        topk, y, used = load_labeled_topk(args.traces, args.labels, L, E,
                                          min_tokens=0, limit=args.self_check)
        ref = compute_risk_diff(
            _dense_from_topk(topk, E), y, len(used), K)
        fast = risk_diff_bincount(topk, y, L, E, len(used), device=args.device)
        d = np.abs(ref - fast).max()
        print(f"used(fast)={used}", flush=True)
        print(f"max|ref-fast| = {d:.3e}  ->  {'IDENTICAL ✅' if d < 1e-5 else 'MISMATCH ❌'}", flush=True)
        return

    topk, y, used = load_labeled_topk(args.traces, args.labels, L, E,
                                      min_tokens=args.min_tokens)
    print(f"model={args.model}  geometry={L}x{E} top-{K}", flush=True)
    risk = risk_diff_bincount(topk, y, L, E, len(used), device=args.device)
    write_class_pkls(args.out_dir, used, risk_table=risk, top_k=K)
    print(f"wrote pickles to {args.out_dir}/", flush=True)


def _dense_from_topk(topk, num_experts):
    N, L, K = topk.shape
    dense = np.zeros((N, L, num_experts), dtype=np.float32)
    np.put_along_axis(dense, topk, 1.0, axis=2)
    return dense


if __name__ == "__main__":
    main()
