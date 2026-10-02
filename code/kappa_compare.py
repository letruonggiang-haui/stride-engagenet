"""Is the EngageNet-versus-DAiSEE adjacent-clip kappa comparison like-for-like?

The DAiSEE label audit computes adjacent-clip agreement only over pairs where BOTH clips carry Engagement >= 2,
i.e. over two of the four levels. The EngageNet figures in the facts packet are computed over all four. Restricting
a kappa to fewer categories changes it mechanically, so the two numbers cannot be set beside each other as "the
same calculation" until EngageNet is recomputed under the same restriction. This script does that.

usage: kappa_compare.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import engagenet_labels
from sklearn.metrics import cohen_kappa_score

en = engagenet_labels()


def adjacent(d, floor=None):
    """Pairs of consecutive clips within one (subject, session, video); optionally only pairs with both >= floor."""
    a, b = [], []
    for _, g in d.sort_values(["subject", "session", "video", "idx"]).groupby(["subject", "session", "video"]):
        v = g.y4.values
        for x, y in zip(v[:-1], v[1:]):
            if floor is not None and not (x >= floor and y >= floor):
                continue
            a.append(int(x)); b.append(int(y))
    return np.array(a), np.array(b)


print("EngageNet adjacent-clip agreement, unweighted Cohen kappa")
print(f"{'split':8s} {'all four levels':>34s} {'restricted to levels >= 2':>36s}")
for sp in ("train", "val", "test"):
    d = en[en.split == sp]
    a, b = adjacent(d)
    a2, b2 = adjacent(d, floor=2)
    left = f"n={len(a):5d} agree {(a == b).mean():.4f} kappa {cohen_kappa_score(a, b):.4f}"
    if len(a2) > 1 and len(set(a2.tolist() + b2.tolist())) > 1:
        right = f"n={len(a2):5d} agree {(a2 == b2).mean():.4f} kappa {cohen_kappa_score(a2, b2):.4f}"
    else:
        right = "degenerate"
    print(f"{sp:8s} {left:>34s} {right:>36s}")
print()
print("DAiSEE: see results/LABEL_STABILITY.json, produced by code/daisee_label_audit.py.")
print("The figure this project cited before 29/09/2026 (agreement 0.58, kappa about 0.09) does NOT reproduce;")
print("the public label files give kappa 0.1351 under the same restriction.")
