"""Paired subject-bootstrap intervals for the fused-minus-pixel difference on the three inner folds.

The manuscript reports these three intervals, but no deposited artefact held them: G7_FUSED.json carries only a
marginal interval around the fused score itself. The formula audit flagged that as an assertion a reader could not
check. This script recomputes them from the saved fold predictions and writes results/PAIRED_FUSION_CI.json, so the
claim rests on a released file.

Paired means the SAME bootstrap resample of subjects is scored under both systems and the difference taken inside
the resample. That is the interval the comparison actually needs; a marginal interval around either system alone is
a different and much wider quantity.

usage: paired_fusion_ci.py
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels
from sklearn.metrics import accuracy_score, f1_score

B = 2000
# exactly the member lists the fold results were built from: four members exist only on fold 0
PIX = {0: ["ft_g6_ep12_ema999_train.npz", "ft_g7_svfap_f0_s1_train.npz",
           "ft_g7_svfap_f0_s2_train.npz", "ft_g7_maedfer_f0_s0_train.npz"],
       1: ["ft_g7_svfap_f1_s0_train.npz"],
       2: ["ft_g7_svfap_f2_s0_train.npz"]}

en = engagenet_labels().set_index("key")
out = {"B": B, "unit": "subject", "interval": "percentile",
       "note": ("Paired: one resample of subjects is scored under both systems and the difference taken inside it. "
                "Fold 0 uses four pixel members; folds 1 and 2 use a single SVFAP model, which is all that exists.")}

for fold in (0, 1, 2):
    fs = [os.path.join(ROOT, "preds", f) for f in PIX[fold]]
    fs = [f for f in fs if os.path.exists(f)]
    gp = os.path.join(ROOT, "preds", f"geom_gbdt_fold{fold}.npz")
    if not fs or not os.path.exists(gp):
        out[f"fold{fold}"] = {"status": "missing predictions", "pixel_files_found": len(fs),
                              "geometry_cached": os.path.exists(gp)}
        print(f"fold {fold}: missing inputs", flush=True)
        continue
    zs = [np.load(f, allow_pickle=True) for f in fs]
    keys = [str(k) for k in zs[0]["keys"]]
    common = set(keys)
    for z in zs: common &= {str(k) for k in z["keys"]}
    gz = np.load(gp, allow_pickle=True); gk = {str(k): i for i, k in enumerate(gz["keys"])}
    common &= set(gk)
    keys = [k for k in keys if k in common]
    def align(z):
        pos = {str(k): i for i, k in enumerate(z["keys"])}
        return np.stack([z["P"][pos[k]] for k in keys])
    P_pix = np.mean([align(z) for z in zs], 0)
    P_geo = np.stack([gz["P"][gk[k]] for k in keys])
    P_fus = 0.5 * P_pix + 0.5 * P_geo
    y = en.loc[keys, "y4"].values.astype(int)
    subj = en.loc[keys, "subject"].values
    us = np.unique(subj); idx = {s: np.where(subj == s)[0] for s in us}
    rng = np.random.default_rng(0)
    d_acc, d_f1 = [], []
    for _ in range(B):
        take = np.concatenate([idx[s] for s in rng.choice(us, len(us), replace=True)])
        d_acc.append(accuracy_score(y[take], P_fus[take].argmax(1)) - accuracy_score(y[take], P_pix[take].argmax(1)))
        d_f1.append(f1_score(y[take], P_fus[take].argmax(1), average="macro")
                    - f1_score(y[take], P_pix[take].argmax(1), average="macro"))
    def summarise(point, boots):
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return dict(point=float(point), ci=[float(lo), float(hi)], width_points=float(100 * (hi - lo)))
    acc = summarise(accuracy_score(y, P_fus.argmax(1)) - accuracy_score(y, P_pix.argmax(1)), d_acc)
    f1d = summarise(f1_score(y, P_fus.argmax(1), average="macro") - f1_score(y, P_pix.argmax(1), average="macro"), d_f1)
    out[f"fold{fold}"] = {"n_clips": len(keys), "n_subjects": int(len(us)), "pixel_members": len(fs),
                          "accuracy_difference": acc, "macro_f1_difference": f1d}
    print(f"fold {fold}: n={len(keys)} subjects={len(us)} members={len(fs)}  "
          f"acc diff {acc['point']:+.4f} CI [{acc['ci'][0]:+.4f}, {acc['ci'][1]:+.4f}] width {acc['width_points']:.1f} pts  |  "
          f"macro-F1 diff {f1d['point']:+.4f} CI [{f1d['ci'][0]:+.4f}, {f1d['ci'][1]:+.4f}] width {f1d['width_points']:.1f} pts", flush=True)

json.dump(out, open(os.path.join(ROOT, "results", "PAIRED_FUSION_CI.json"), "w"), indent=1, default=float)
print("PAIRED_FUSION_CI_DONE", flush=True)
