"""Paired subject-level bootstrap intervals for the two comparisons of the third campaign that the manuscript
leads with: CMS against the 61-frame window, and segment consensus against CMS (macro-F1).

Estimand: exactly the reported difference, the mean over the nine (fold, seed) pairs of
macro-F1(arm A) - macro-F1(arm B), each run scored on its own inner fold. The point estimate on the original
sample is therefore +0.0865 and -0.0513, as in Table 14.

Resampling: within each inner fold its subjects are drawn with replacement (the folds are subject-disjoint and
each fold's models are scored on that fold alone), B = 2000, numpy default_rng(0), the same resample serving both
arms and all three seeds, so the interval belongs to the paired difference. Percentile interval, 2.5 and 97.5.
This is the method of Section 3.6 applied to inner-fold predictions. No model is trained and no official split is
read.

usage: paired_ci_v3.py            writes results/PAIRED_CI_V3.json (write-once)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT

B, NC = 2000, 4
OUT = os.path.join(ROOT, "results", "PAIRED_CI_V3.json")
if os.path.exists(OUT): sys.exit(f"refusing to overwrite {OUT}: write-once")


def macro_f1(C):
    """C [..., 4, 4], rows truth; sklearn's macro-F1 over labels present in truth or prediction"""
    C = C.astype(np.float64); tp = np.diagonal(C, axis1=-2, axis2=-1); rows = C.sum(-1); cols = C.sum(-2)
    present = (rows + cols) > 0
    f1 = np.where(present, 2 * tp / np.maximum(rows + cols, 1e-12), 0.0)
    return (f1 * present).sum(-1) / present.sum(-1)


def load(arm, f, s):
    z = np.load(os.path.join(ROOT, "preds", f"ft_v3_{arm}_f{f}_s{s}_train.npz"), allow_pickle=True)
    return z["P"].argmax(1), z["y"].astype(int), z["keys"], z["subject"]


def per_subject_conf(yh, y, subj, subjects):
    """confusion matrix of each subject, [S, 4, 4]"""
    idx = {s: i for i, s in enumerate(subjects)}
    C = np.zeros((len(subjects), NC, NC), np.int64)
    np.add.at(C, (np.array([idx[s] for s in subj]), y, yh), 1)
    return C


def selftest():
    from sklearn.metrics import f1_score
    import warnings
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = rng.integers(3, 50); y = rng.integers(0, 4, n); yh = rng.integers(0, rng.integers(1, 5), n)
        C = np.zeros((NC, NC), np.int64); np.add.at(C, (y, yh), 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); ref = f1_score(y, yh, average="macro")
        assert abs(macro_f1(C) - ref) < 1e-12


selftest()
rng = np.random.default_rng(0)
folds = []
for f in range(3):
    arms = {}
    for arm in ("R", "S", "C"):
        confs = []
        for s in range(3):
            yh, y, keys, subj = load(arm, f, s)
            if arm == "R" and s == 0:
                ref_keys, subjects = keys, np.unique(subj)
            assert (keys == ref_keys).all(), "clips differ across arms or seeds"
            confs.append(per_subject_conf(yh, y, subj, subjects))
        arms[arm] = np.stack(confs)                  # [3 seeds, S, 4, 4]
    folds.append(dict(subjects=subjects, arms=arms))

draws = [rng.integers(0, len(F["subjects"]), size=(B, len(F["subjects"]))) for F in folds]


def stat(a, b):
    """point estimate and bootstrap distribution of mean over folds and seeds of macro-F1(a) - macro-F1(b)"""
    point, boot = [], []
    for F, d in zip(folds, draws):
        A, Bm = F["arms"][a], F["arms"][b]
        point.append(np.mean(macro_f1(A.sum(1)) - macro_f1(Bm.sum(1))))
        w = np.stack([np.bincount(row, minlength=len(F["subjects"])) for row in d]).astype(np.float64)   # [B, S]
        fa = macro_f1(np.einsum("bs,ksij->bkij", w, A)); fb = macro_f1(np.einsum("bs,ksij->bkij", w, Bm))   # [B, 3]
        boot.append((fa - fb).mean(1))
    point = float(np.mean(point)); boot = np.mean(np.stack(boot), 0)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(point=point, ci95=[float(lo), float(hi)], share_of_resamples_above_zero=float((boot > 0).mean()))


out = {"method": "paired subject-level bootstrap within inner folds, B = 2000, default_rng(0), percentile; "
                 "estimand = mean over nine (fold, seed) pairs of the macro-F1 difference",
       "subjects_per_fold": [int(len(F["subjects"])) for F in folds],
       "CMS_minus_61_frame_window": stat("C", "R"),
       "segment_consensus_minus_CMS": stat("S", "C")}
for k in ("CMS_minus_61_frame_window", "segment_consensus_minus_CMS"):
    r = out[k]; print("%-30s %+.4f  [%+.4f, %+.4f]  share > 0: %.4f" % (k, r["point"], r["ci95"][0], r["ci95"][1], r["share_of_resamples_above_zero"]))
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
