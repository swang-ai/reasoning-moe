from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def hard_activation(router: np.ndarray, top_k: int) -> np.ndarray:
    idx = np.argpartition(-router, top_k - 1, axis=-1)[..., :top_k]
    hard = np.zeros_like(router, dtype=np.float32)
    np.put_along_axis(hard, idx, 1.0, axis=-1)
    return hard


def compute_risk_diff(router, y, num_labels, top_k):
    y = np.asarray(y)
    hard = hard_activation(router, top_k)
    L, E = hard.shape[1], hard.shape[2]
    risk = np.zeros((L, E, num_labels), dtype=np.float32)
    for c in range(num_labels):
        risk[..., c] = hard[y == c].mean(axis=0) - hard[y != c].mean(axis=0)
    return risk


def _table_to_df(table_2d):
    L, E = table_2d.shape
    rows = [{"layer": l, "expert": e, "risk_diff": float(table_2d[l, e]),
             "risk_diff_abs": abs(float(table_2d[l, e]))}
            for l in range(L) for e in range(E)]
    return pd.DataFrame(rows)


def write_class_pkls(out_dir, labels, *, router=None, y=None, top_k=None,
                     metric="risk_diff", risk_table=None, seed=0, verbose=True,
                     **_legacy):
    if metric != "risk_diff":
        raise ValueError(f"metric {metric!r} removed — only 'risk_diff' is supported "
                         "(see module docstring)")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    C = len(labels)
    if risk_table is None:
        risk_table = compute_risk_diff(router, y, C, top_k)
    paths = {}
    for c, name in enumerate(labels):
        df = _table_to_df(risk_table[..., c])
        p = out / f"riskdiff_{name}.pkl"
        df.to_pickle(p); paths[name] = p
        if verbose:
            top = df[df.risk_diff > 0].sort_values("risk_diff_abs", ascending=False).head(3)
            tops = ", ".join(f"L{int(r.layer)}E{int(r.expert)}({r.risk_diff:+.2f})"
                             for r in top.itertuples())
            print(f"  [{name:12s}] risk_diff -> {p.name}   top: {tops}")
    return paths
