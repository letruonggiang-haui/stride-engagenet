"""Where is the remaining headroom? Three probes on INNER CV fold predictions only (never the opened official splits).
 1. error correlation between ensemble members -> is diversity the lever, or are seeds redundant?
 2. per-subject accuracy spread and an ORACLE per-subject prior reweighting -> how much is subject effect?
 3. label-free temporal smoothing over adjacent clips of the same recording -> is sequence context usable?"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.expanduser("~/engagenet_p2_crossdomain/code")); os.chdir(os.path.expanduser("~/engagenet_p2_crossdomain"))
from p2x_lib import engagenet_labels
from sklearn.metrics import accuracy_score
en = engagenet_labels(); meta = en.set_index("key")
FOLDS = {0: ["preds/ft_g6_ep12_ema999_train.npz","preds/ft_g7_svfap_f0_s1_train.npz","preds/ft_g7_svfap_f0_s2_train.npz","preds/ft_g7_maedfer_f0_s0_train.npz"],
         1: ["preds/ft_g7_svfap_f1_s0_train.npz"], 2: ["preds/ft_g7_svfap_f2_s0_train.npz"]}
for fold, fs in FOLDS.items():
    zs = [np.load(f, allow_pickle=True) for f in fs]; keys = zs[0]["keys"]; common = set(keys.tolist())
    for z in zs: common &= set(z["keys"].tolist())
    keys = np.array([k for k in keys if k in common]); y = meta.loc[keys, "y4"].values.astype(int)
    def al(z): pos = {k: i for i, k in enumerate(z["keys"])}; return z["P"][[pos[k] for k in keys]]
    Ps = [al(z) for z in zs]; P = np.mean(Ps, 0); yh = P.argmax(1); base = accuracy_score(y, yh)
    print(f"\n===== fold {fold}: {len(y)} clips, {len(fs)} member(s), ensemble acc {base:.4f}")
    if len(Ps) > 1:   # 1. error correlation
        errs = np.stack([(p.argmax(1) != y).astype(float) for p in Ps]); C = np.corrcoef(errs)
        off = C[np.triu_indices(len(Ps), 1)]; anyright = (errs.min(0) == 0).mean(); allwrong = (errs.min(0) == 1).mean()
        print(f"  [1] member error correlation: mean {off.mean():.3f} (min {off.min():.3f}, max {off.max():.3f}); at least one member right on {anyright:.3f}; all wrong on {allwrong:.3f}")
        print(f"      -> oracle 'pick the right member when any is right' = {anyright:.4f}  (ensemble gets {base:.4f})")
    # 2. subject effect and oracle per-subject prior
    subj = meta.loc[keys, "subject"].values; us = np.unique(subj)
    per = np.array([accuracy_score(y[subj == s], yh[subj == s]) for s in us]); n_per = np.array([(subj == s).sum() for s in us])
    print(f"  [2] per-subject accuracy: min {per.min():.2f}  p25 {np.percentile(per,25):.2f}  median {np.median(per):.2f}  p75 {np.percentile(per,75):.2f}  max {per.max():.2f}   (subjects {len(us)})")
    src_prior = np.bincount(y, minlength=4) / len(y)
    P_or = P.copy()
    for s in us:
        m = subj == s; tp = np.bincount(y[m], minlength=4) / m.sum() + 1e-6
        P_or[m] = P[m] * (tp / src_prior)[None]
    print(f"      oracle reweight by the TRUE subject label prior: {accuracy_score(y, P_or.argmax(1)):.4f}  (+{accuracy_score(y, P_or.argmax(1)) - base:.4f}) -- ceiling for any transductive per-subject method")
    # label-free proxy: reweight by the subject's own MEAN PREDICTED distribution (no labels)
    P_lf = P.copy()
    for s in us:
        m = subj == s; pp = P[m].mean(0) + 1e-6
        P_lf[m] = P[m] * (pp / src_prior)[None]
    print(f"      label-free reweight by the subject's mean predicted distribution: {accuracy_score(y, P_lf.argmax(1)):.4f}  ({accuracy_score(y, P_lf.argmax(1)) - base:+.4f})")
    # 3. temporal smoothing over adjacent clips in the same recording (label-free, uses only clip order)
    df = pd.DataFrame(dict(key=keys, rec=meta.loc[keys, "recording"].values, idx=meta.loc[keys, "idx"].values, i=np.arange(len(keys))))
    for k in (1, 2, 4):
        Ps_ = P.copy()
        for rec, g in df.sort_values("idx").groupby("rec"):
            ii = g.i.values
            for j, t in enumerate(ii):
                lo, hi = max(0, j - k), min(len(ii), j + k + 1); Ps_[t] = P[ii[lo:hi]].mean(0)
        a = accuracy_score(y, Ps_.argmax(1)); print(f"  [3] smooth +-{k} adjacent clips in the recording: {a:.4f}  ({a - base:+.4f})")
