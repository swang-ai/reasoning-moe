from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

REASONINGFLOW_LABELS = [
    "planning", "fact", "reasoning", "restatement",
    "assumption", "example", "reflection", "conclusion",
]
LABEL_ALIASES = {"verification": "reflection"}

REASONINGFLOW_CORE = [
    "planning", "fact", "reasoning", "restatement", "reflection", "conclusion",
]
CONDENSED_LABELS = ["planning", "reasoning", "restatement", "fact", "conclusion"]

PHASE_LABELS = REASONINGFLOW_CORE
LABEL2IDX = {l: i for i, l in enumerate(PHASE_LABELS)}


def load_labeled_traces(traces_dir, labels_dir, num_layers, num_experts,
                        labels=None, min_tokens=0):
    labels = list(labels or PHASE_LABELS)
    keep = set(labels)
    traces_dir, labels_dir = Path(traces_dir), Path(labels_dir)
    routers, lbls = [], []
    n_traces = 0
    for lp in sorted(labels_dir.glob("*.json")):
        pt = traces_dir / f"{lp.stem}.pt"
        if not pt.exists():
            continue
        nodes = json.loads(lp.read_text()).get("nodes", [])
        if not nodes:
            continue
        d = torch.load(pt, weights_only=False)
        if "router_logits" in d:
            rl = d["router_logits"].float().numpy()
        else:
            ids = d["topk_ids"].numpy().astype(np.int64)
            rl = np.zeros((ids.shape[0], ids.shape[1], num_experts), dtype=np.float32)
            np.put_along_axis(rl, ids, 1.0, axis=2)
        T = rl.shape[1]
        rl_te = rl.transpose(1, 0, 2)
        for node in nodes:
            t0, t1 = node["token_start"], min(node["token_end"], T)
            lbl = str(node["label"]).strip().lower()
            lbl = LABEL_ALIASES.get(lbl, lbl)
            if t0 >= t1 or lbl not in keep:
                continue
            routers.append(rl_te[t0:t1])
            lbls.append([lbl] * (t1 - t0))
        n_traces += 1
    if not routers:
        raise SystemExit(f"No labeled tokens found under {traces_dir} / {labels_dir}")
    router = np.concatenate(routers).astype(np.float32).reshape(-1, num_layers, num_experts)
    flat = np.array([x for seg in lbls for x in seg])

    from collections import Counter
    counts = Counter(flat)
    used = [l for l in labels if counts.get(l, 0) >= max(1, min_tokens)]
    dropped = [(l, counts.get(l, 0)) for l in labels if l not in used]
    l2i = {l: i for i, l in enumerate(used)}
    mask = np.array([l in l2i for l in flat])
    router = router[mask]
    y = np.array([l2i[l] for l in flat[mask]], dtype=np.int64)
    print(f"loaded {len(y):,} tokens from {n_traces} traces  ({router.shape})")
    print(f"classes kept: {[(l, int(counts[l])) for l in used]}")
    if dropped:
        print(f"classes dropped (<{min_tokens} tok): {dropped}")
    return router, y, used
