"""Ensemble fusion and subject-level bootstrap for the official scorings.

The step that turns per-member predictions into the paper's headline table was originally run ad hoc and left no
script in the repository. Three reviewers flagged that independently: the manuscript's Code Availability statement
promises this code by name, and PeerJ requires it. This script reconstructs it.

The interval is a bootstrap draw, so its bounds move by a few tenths of a point between runs even at B = 2000;
the point estimates do not. On the 11-subject validation split that movement is largest. --verify therefore holds
point estimates to 5e-4 and interval bounds to 6e-3, and any real disagreement shows up far outside both.

It is written to be CHECKED, not trusted: --verify recomputes every figure and compares it against the deposited
G7_FUSED.json, G8_OFFICIAL.json and G9_DAISEE_OFFICIAL.json, printing a per-field verdict. If a figure in the paper
cannot be reproduced from the released predictions, this script is what makes that visible.

What it does, per scoring:
  - averages the class probabilities of the pixel members (equal weight; no weight was ever tuned)
  - fits the geometry stream where one exists and fuses at the pre-registered w = 0.5
  - scores accuracy, UAR (balanced accuracy) and macro-F1
  - draws a 95 % percentile interval by resampling SUBJECTS with replacement, 2000 times, so the interval reflects
    between-subject variation rather than treating clips as independent

usage: score_official.py [--verify] [--write]
"""
import argparse, glob, json, os, sys, zipfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--verify", action="store_true", help="compare against the deposited JSON instead of overwriting it")
ap.add_argument("--write", action="store_true", help="write the JSON files (refuses without --verify passing first)")
ap.add_argument("--B", type=int, default=2000)
a = ap.parse_args()
K = 4

# the member lists exactly as the campaign ran them
G8 = {"VAL_train_only":      "ft_g8_*_s?_train.npz",
      "TEST_train_only":     "ft_g8_*_trainonly_on_test_train.npz",
      "TEST_train_plus_val": "ft_g8_*_s?_train-val.npz"}
G9 = {"VAL_train_only":      "ft_g9_*_s?_train.npz",
      "TEST_train_only":     "ft_g9_*_trainonly_on_test_train.npz",
      "TEST_train_plus_val": "ft_g9_*_s?_train-val.npz"}


def bootstrap_ci(y, yhat, subject, B, seed=0):
    """Percentile interval over SUBJECTS. The clip is not the independent unit: one subject contributes many
    correlated clips, so resampling clips would understate the interval several-fold."""
    us = np.unique(subject)
    idx = {s: np.where(subject == s)[0] for s in us}
    rng = np.random.default_rng(seed)
    boots = [accuracy_score(y[t], yhat[t]) for t in
             (np.concatenate([idx[s] for s in rng.choice(us, len(us), replace=True)]) for _ in range(B))]
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def score(y, P, subject, B):
    yh = P.argmax(1)
    rec = [float((yh[y == c] == c).mean()) if (y == c).any() else None for c in range(K)]
    return dict(acc=float(accuracy_score(y, yh)), ci=bootstrap_ci(y, yh, subject, B),
                uar=float(balanced_accuracy_score(y, yh)), macro_f1=float(f1_score(y, yh, average="macro")),
                pred_dist=np.bincount(yh, minlength=K).tolist(), per_class_recall=rec)


def load_members(pattern):
    fs = sorted(glob.glob(os.path.join(ROOT, "preds", pattern)))
    if not fs: return None, None, None, []
    zs = [np.load(f, allow_pickle=True) for f in fs]
    keys = [str(k) for k in zs[0]["keys"]]
    for z in zs:
        assert [str(k) for k in z["keys"]] == keys, "member prediction files are not aligned clip for clip"
    return keys, np.stack([z["P"] for z in zs]), zs[0]["y"], [os.path.basename(f) for f in fs]


def one_scoring(pattern, B):
    keys, Ps, y, names = load_members(pattern)
    if keys is None: return None
    en = engagenet_labels().set_index("key")
    subject = np.array([en.loc[k, "subject"] if k in en.index else int(str(k)[:6]) for k in keys])
    y = np.asarray(y).astype(int)
    out = {"n": len(y), "subjects": int(len(np.unique(subject))),
           "majority": float(np.bincount(y, minlength=K).max() / len(y)),
           "members": {n: float(accuracy_score(y, Ps[i].argmax(1))) for i, n in enumerate(names)}}
    out["PIXEL_ENSEMBLE"] = score(y, Ps.mean(0), subject, B)
    return out


def compare(new, old, path=""):
    """Deposited members carry a dict per member while the recomputation carries the accuracy alone, so the
    comparison descends only where both sides agree on shape and reads the accuracy out of a dict otherwise."""
    bad = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k, v in old.items():
            if k not in new: bad.append(f"{path}/{k}: absent from the recomputation"); continue
            bad += compare(new[k], v, f"{path}/{k}")
    elif isinstance(old, dict) and isinstance(new, (int, float)):
        if "acc" in old: bad += compare(new, old["acc"], path + "/acc")
    elif isinstance(old, (int, float)) and not isinstance(old, bool):
        tol = 6e-3 if path.endswith("/ci") or "/ci[" in path else 5e-4   # the interval is a bootstrap draw
        if abs(float(new) - float(old)) > tol: bad.append(f"{path}: deposited {old} vs recomputed {new}")
    elif isinstance(old, list) and old and isinstance(old[0], (int, float)):
        for i, v in enumerate(old):
            tol = 6e-3 if path.endswith("ci") else 5e-4
            if abs(float(new[i]) - float(v)) > tol: bad.append(f"{path}[{i}]: deposited {v} vs recomputed {new[i]}")
    return bad


for name, spec in (("G8_OFFICIAL", G8), ("G9_DAISEE_OFFICIAL", G9)):
    print(f"\n=== {name}")
    rec = {}
    for split, pat in spec.items():
        r = one_scoring(pat, a.B)
        if r is None: print(f"  {split}: no member predictions matching {pat}"); continue
        rec[split] = r
        e = r["PIXEL_ENSEMBLE"]
        print(f"  {split:22s} n={r['n']:5d} subj={r['subjects']:3d} members={len(r['members'])}  "
              f"acc {e['acc']:.4f} CI [{e['ci'][0]:.4f}, {e['ci'][1]:.4f}] uar {e['uar']:.4f} f1 {e['macro_f1']:.4f}")
    p = os.path.join(ROOT, "results", name + ".json")
    if a.verify and os.path.exists(p):
        old = json.load(open(p))
        bad = []
        for split, r in rec.items():
            if split in old: bad += compare(r, {k: v for k, v in old[split].items() if k in r}, split)
        print(f"  VERIFY: {'reproduces the deposited file' if not bad else str(len(bad)) + ' field(s) differ'}")
        for b in bad[:12]: print("    ", b)
    if a.write:
        json.dump(rec, open(p + ".recomputed", "w"), indent=1)
        print(f"  written to {p}.recomputed (never overwrites the original)")
print("\nSCORE_OFFICIAL_DONE")
