"""Adjacent-clip label stability on DAiSEE, and the like-for-like EngageNet comparison.

Three reviewers flagged that the DAiSEE figure quoted in the manuscript -- agreement 0.58 against 0.54 expected
under independence, kappa about 0.09 -- was a constant typed into a comparison script with no artefact behind it.
This script computes it from the public label files and writes results/LABEL_STABILITY.json.

What the quantity is, and is not: it measures whether two consecutive clips of the same session carry the same
label. That is TEMPORAL STABILITY of the labels, not agreement between annotators on one clip. A low value is
equally consistent with noisy labels and with engagement that genuinely changes within a session, and this measure
cannot separate the two. The manuscript must not read it as a noise ceiling.

The DAiSEE audit restricts to pairs where BOTH clips are at level 2 or above, i.e. two of the four levels. EngageNet
is therefore computed twice: over all four levels, and under the same restriction, because restricting the level set
changes kappa mechanically and only the restricted figure is comparable.

usage: daisee_label_audit.py
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels
from sklearn.metrics import cohen_kappa_score

LAB = os.path.expanduser("~/daisee/DAiSEE/Labels")


def stats(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(set(a.tolist() + b.tolist())) < 2:
        return dict(n_pairs=int(len(a)), agreement=None, kappa=None, independent_expectation=None)
    p = np.bincount(a, minlength=4) / len(a)
    q = np.bincount(b, minlength=4) / len(b)
    return dict(n_pairs=int(len(a)), agreement=float((a == b).mean()),
                kappa=float(cohen_kappa_score(a, b)),          # UNWEIGHTED, deliberately
                independent_expectation=float((p * q).sum()))


out = {"quantity": "adjacent-clip label stability within one recording session",
       "kappa_weighting": "unweighted",
       "caveat": ("This measures temporal stability of labels, not inter-annotator agreement on one clip. "
                  "A low value is equally consistent with noisy labels and with genuinely fast-varying engagement; "
                  "the measure cannot separate them.")}

# ---- DAiSEE: session and order from the clip id, as the group's audit did
rows = []
for split, fn in (("train", "TrainLabels.csv"), ("val", "ValidationLabels.csv"), ("test", "TestLabels.csv")):
    p = os.path.join(LAB, fn)
    if not os.path.exists(p): continue
    d = pd.read_csv(p, sep=None, engine="python"); d.columns = [c.strip() for c in d.columns]
    d["stem"] = d["ClipID"].astype(str).str.strip().str.replace(r"\.(avi|mp4)$", "", regex=True)
    d = d[d.stem.str.len() >= 8].copy()
    d["session"] = d.stem.str[:7]; d["order"] = d.stem.str[7:].astype(int)
    d["y"] = d["Engagement"].astype(int); d["split"] = split
    rows.append(d[["session", "order", "y", "split"]])
if rows:
    da = pd.concat(rows, ignore_index=True).sort_values(["session", "order"])
    a_all, b_all, a_r, b_r = [], [], [], []
    for _, g in da.groupby("session"):
        v = g.y.values
        for x, y in zip(v[:-1], v[1:]):
            a_all.append(int(x)); b_all.append(int(y))
            if x >= 2 and y >= 2: a_r.append(int(x)); b_r.append(int(y))
    out["DAiSEE_all_four_levels"] = stats(a_all, b_all)
    out["DAiSEE_restricted_to_levels_ge_2"] = stats(a_r, b_r)
else:
    out["DAiSEE"] = "label files not found on this machine"

# ---- EngageNet, both ways, grouped by (subject, session, video)
en = engagenet_labels()
for sp in ("train", "val", "test"):
    d = en[en.split == sp].sort_values(["subject", "session", "video", "idx"])
    a_all, b_all, a_r, b_r = [], [], [], []
    for _, g in d.groupby(["subject", "session", "video"]):
        v = g.y4.values
        for x, y in zip(v[:-1], v[1:]):
            a_all.append(int(x)); b_all.append(int(y))
            if x >= 2 and y >= 2: a_r.append(int(x)); b_r.append(int(y))
    out[f"EngageNet_{sp}_all_four_levels"] = stats(a_all, b_all)
    out[f"EngageNet_{sp}_restricted_to_levels_ge_2"] = stats(a_r, b_r)

for k, v in out.items():
    if isinstance(v, dict) and "kappa" in v:
        kk = v["kappa"]; ag = v["agreement"]
        print(f"{k:44s} n={v['n_pairs']:6d}  agreement {ag:.4f}  kappa {kk:.4f}  independent {v['independent_expectation']:.4f}"
              if kk is not None else f"{k:44s} degenerate")
json.dump(out, open(os.path.join(ROOT, "results", "LABEL_STABILITY.json"), "w"), indent=1)
print("\nLABEL_STABILITY_DONE")
