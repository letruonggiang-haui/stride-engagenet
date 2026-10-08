"""Fusion and scoring of the official EngageNet runs: the pixel ensemble, the geometry stream and the fused prediction
at w = 0.5, each with its subject-level bootstrap interval. This is the step that wrote results/G8_OFFICIAL.json on
25 September 2026. It ran inline at the end of the official queue (state/g8.sh, after the twelve fine-tuning runs)
and is deposited here on 8 October 2026; everything below the marker line is that inline code, byte for byte.

It refits the geometry stream (HistGradientBoostingClassifier, 400 trees, learning rate 0.06, random_state 0) on the
cached 1,338-dimensional descriptors and averages it with the pixel ensemble at w = 0.5. The bootstrap draws from one
generator shared across the systems of a scoring, so its intervals are reproduced only by running exactly this code
in this order; code/paired_official_ci.py reuses the same geometry fit and reproduces every point accuracy.

Run from ~/engagenet_p2_crossdomain with PYTHONPATH=pylib:code and the run interpreter. It rewrites
results/G8_OFFICIAL.json, so it refuses to run without --rewrite; use it on a scratch copy of the tree.
"""
import sys
if "--rewrite" not in sys.argv:
    sys.exit("refusing to rewrite results/G8_OFFICIAL.json without --rewrite; run it on a scratch copy")
# ---- verbatim from state/g8.sh below this line ----
import os, sys, glob, json, numpy as np
sys.path.insert(0, "code")
from p2x_lib import engagenet_labels
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
C = os.path.expanduser("~/engagenet/experiments/cache")
en = engagenet_labels()
def geom(split):
    d = en[en.split==split].sort_values(["subject","session","video","idx"]).reset_index(drop=True)
    return d, np.nan_to_num(np.load(f"{C}/geom_{split}.npy"))
out = {}
for name, pat, gtrain, gsplit in (("VAL_train_only", "preds/ft_g8_*_s?_train.npz", ["train"], "val"),
                                  ("TEST_train_only", "preds/ft_g8_*_trainonly_on_test_train.npz", ["train"], "test"),
                                  ("TEST_train_plus_val", "preds/ft_g8_*_s?_train-val.npz", ["train","val"], "test")):
    fs = sorted(glob.glob(pat)); zs = [np.load(f, allow_pickle=True) for f in fs]
    if not zs: out[name] = "no predictions found"; continue
    keys = zs[0]["keys"]; y = zs[0]["y"].astype(int)
    for z in zs: assert (z["keys"] == keys).all()
    d, G = geom(gsplit); gi = dict(zip(d.key, range(len(d)))); subj = np.array([int(d.subject.iloc[gi[k]]) for k in keys])
    Ppix = np.mean([z["P"] for z in zs], 0)
    dt = [geom(s) for s in gtrain]; Xtr = np.concatenate([g for _, g in dt]); ytr = np.concatenate([dd.y4.values for dd, _ in dt])
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=0).fit(Xtr, ytr)
    Pg = m.predict_proba(G[[gi[k] for k in keys]]); Pf = 0.5 * Ppix + 0.5 * Pg
    us = np.unique(subj); idx = {s: np.where(subj == s)[0] for s in us}; rng = np.random.default_rng(0)
    def sc(P):
        yh = P.argmax(1); b = [accuracy_score(y[i], yh[i]) for i in (np.concatenate([idx[s] for s in rng.choice(us, len(us), replace=True)]) for _ in range(2000))]
        return dict(acc=accuracy_score(y, yh), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], uar=balanced_accuracy_score(y, yh), macro_f1=f1_score(y, yh, average="macro"), pred_dist=np.bincount(yh, minlength=4).tolist())
    out[name] = dict(n=int(len(y)), subjects=int(len(us)), majority=float(np.bincount(y, minlength=4).max() / len(y)), members={os.path.basename(f): sc(z["P"]) for f, z in zip(fs, zs)},
                     PIXEL_ENSEMBLE=sc(Ppix), geometry=sc(Pg), FUSED_w05=sc(Pf))
    r = out[name]; print(f"\n=== {name}: n {r['n']}  subjects {r['subjects']}  majority {r['majority']:.4f}")
    for k, v in r["members"].items(): print(f"   {k:44s} acc {v['acc']:.4f}")
    for k in ("PIXEL_ENSEMBLE", "geometry", "FUSED_w05"): v = r[k]; print(f"   {k:20s} acc {v['acc']:.4f}  CI [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]  uar {v['uar']:.4f}  macroF1 {v['macro_f1']:.4f}  {v['pred_dist']}")
json.dump(out, open("results/G8_OFFICIAL.json", "w"), indent=1, default=float)
print("\nG8_OFFICIAL_DONE")
