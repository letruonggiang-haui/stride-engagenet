"""Paired subject-level bootstrap intervals between the systems scored on the official EngageNet splits.

Written on 8 October 2026, after the official scorings of 25 September and after every campaign; it is a post-hoc
analysis and is reported as one. No model is trained, no official split is re-opened: it reads the per-clip
predictions the official runs already wrote (preds/ft_g8_*.npz) and refits the geometry stream exactly as the
official scoring did (state/g8.sh, deposited verbatim as code/g8_official_fusion.py).

Why: the marginal interval around one system's score includes the between-subject variation that two systems scored
on the same subjects share, so it is the wrong quantity for ranking two systems. The interval on their paired
difference is the right one. These are the paired widths the reviewers asked for.

Fixed before the script was first run:
  metric      accuracy, the metric of the published EngageNet tables
  scorings    VAL_train_only (11 subjects), TEST_train_only (26), TEST_train_plus_val (26), as in G8_OFFICIAL.json
  pairs       primary: pixel ensemble - geometry stream (two streams with different inputs and model families, the
              nearest thing in this study to two independently built entries);
              then fused (w = 0.5) - pixel ensemble; MAE-DFER-B s0 - each SVFAP-B seed; each pair of SVFAP-B seeds
  bootstrap   subjects drawn with replacement within the scoring, B = 2000, numpy default_rng(0), one set of draws
              per scoring shared by every pair, percentile interval 2.5 / 97.5
  check       before any interval is computed, the recomputed point accuracies of every member, the ensemble, the
              geometry stream and the fused prediction must equal G8_OFFICIAL.json to 1e-9; if the geometry refit does
              not reproduce, the geometry and fused pairs are dropped and the JSON says so
  reading     a gap between two published validation entries is read as resolvable on 11 subjects only if it is
              larger than half the width of the primary pair's paired interval on VAL_train_only. The text reports how
              many of the adjacent gaps in Table 1 pass, whatever the count. Our systems share training data and a
              pipeline, so their errors correlate more than two independently built entries' would; their paired
              intervals are therefore a narrow, optimistic reference, and the text says so.

usage (on the experiment server, from ~/engagenet_p2_crossdomain, PYTHONPATH=pylib:code, the run interpreter
~/venvs/engagenet/bin/python): paired_official_ci.py      writes results/PAIRED_OFFICIAL_CI.json (write-once)
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels
from sklearn.ensemble import HistGradientBoostingClassifier

B = 2000
OUT = os.path.join(ROOT, "results", "PAIRED_OFFICIAL_CI.json")
if os.path.exists(OUT): sys.exit(f"refusing to overwrite {OUT}: write-once")
C = os.path.expanduser("~/engagenet/experiments/cache")
G8 = json.load(open(os.path.join(ROOT, "results", "G8_OFFICIAL.json")))
en = engagenet_labels()

SCORINGS = (("VAL_train_only", "preds/ft_g8_*_s?_train.npz", ["train"], "val"),
            ("TEST_train_only", "preds/ft_g8_*_trainonly_on_test_train.npz", ["train"], "test"),
            ("TEST_train_plus_val", "preds/ft_g8_*_s?_train-val.npz", ["train", "val"], "test"))
PAIRS = [("PIXEL_ENSEMBLE", "geometry"), ("FUSED_w05", "PIXEL_ENSEMBLE"),
         ("maedfer_s0", "svfap_s0"), ("maedfer_s0", "svfap_s1"), ("maedfer_s0", "svfap_s2"),
         ("svfap_s0", "svfap_s1"), ("svfap_s0", "svfap_s2"), ("svfap_s1", "svfap_s2")]


def geom(split):   # verbatim from state/g8.sh
    d = en[en.split == split].sort_values(["subject", "session", "video", "idx"]).reset_index(drop=True)
    return d, np.nan_to_num(np.load(f"{C}/geom_{split}.npy"))


def short(f):      # ft_g8_maedfer_s0_train.npz -> maedfer_s0
    b = os.path.basename(f).split("_")
    return b[2] + "_" + b[3]


out = {"method": "paired subject-level bootstrap of the accuracy difference, B = 2000, default_rng(0), one set of "
                 "subject draws per scoring shared by all pairs, percentile 2.5 / 97.5; post hoc, written 2026-10-08",
       "primary_pair": "PIXEL_ENSEMBLE - geometry", "scorings": {}}
os.chdir(ROOT)
for name, pat, gtrain, gsplit in SCORINGS:
    fs = sorted(glob.glob(pat)); zs = [np.load(f, allow_pickle=True) for f in fs]
    keys = zs[0]["keys"]; y = zs[0]["y"].astype(int)
    for z in zs: assert (z["keys"] == keys).all()
    d, G = geom(gsplit); gi = dict(zip(d.key, range(len(d))))
    subj = np.array([int(d.subject.iloc[gi[k]]) for k in keys])
    Ppix = np.mean([z["P"] for z in zs], 0)
    dt = [geom(s) for s in gtrain]
    Xtr = np.concatenate([g for _, g in dt]); ytr = np.concatenate([dd.y4.values for dd, _ in dt])
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=0).fit(Xtr, ytr)
    Pg = m.predict_proba(G[[gi[k] for k in keys]]); Pf = 0.5 * Ppix + 0.5 * Pg
    pred = {short(f): z["P"].argmax(1) for f, z in zip(fs, zs)}
    pred.update(PIXEL_ENSEMBLE=Ppix.argmax(1), geometry=Pg.argmax(1), FUSED_w05=Pf.argmax(1))

    # reproduction check against the deposited official scoring
    dep = G8[name]; check = {}
    for f in fs: check[short(f)] = (float((pred[short(f)] == y).mean()), dep["members"][os.path.basename(f)]["acc"])
    for k in ("PIXEL_ENSEMBLE", "geometry", "FUSED_w05"): check[k] = (float((pred[k] == y).mean()), dep[k]["acc"])
    ok = {k: abs(a - b) < 1e-9 for k, (a, b) in check.items()}
    pix_ok = all(v for k, v in ok.items() if k not in ("geometry", "FUSED_w05"))
    assert pix_ok, f"{name}: pixel predictions do not reproduce G8_OFFICIAL.json: {check}"
    geom_ok = ok["geometry"] and ok["FUSED_w05"]

    us = np.unique(subj); S = len(us); si = np.searchsorted(us, subj)
    n_s = np.bincount(si, minlength=S).astype(float)
    corr = {k: np.bincount(si, weights=(v == y).astype(float), minlength=S) for k, v in pred.items()}
    rng = np.random.default_rng(0)
    W = np.stack([np.bincount(r, minlength=S) for r in rng.integers(0, S, size=(B, S))]).astype(float)   # [B, S]
    bacc = {k: (W @ c) / (W @ n_s) for k, c in corr.items()}
    rec = {"n": int(len(y)), "subjects": int(S), "reproduces_G8_OFFICIAL": {k: bool(v) for k, v in ok.items()},
           "recomputed_vs_deposited_acc": check, "pairs": {}}
    for a_, b_ in PAIRS:
        if not geom_ok and ("geometry" in (a_, b_) or "FUSED_w05" in (a_, b_)):
            rec["pairs"][f"{a_} - {b_}"] = "dropped: geometry refit did not reproduce the official scoring"; continue
        diff = bacc[a_] - bacc[b_]; lo, hi = np.percentile(diff, [2.5, 97.5])
        pt = float((pred[a_] == y).mean() - (pred[b_] == y).mean())
        rec["pairs"][f"{a_} - {b_}"] = dict(point_pts=100 * pt, ci_pts=[100 * float(lo), 100 * float(hi)],
                                           width_pts=100 * float(hi - lo), share_above_zero=float((diff > 0).mean()))
    lo, hi = np.percentile(bacc["PIXEL_ENSEMBLE"], [2.5, 97.5])
    rec["ensemble_marginal_same_draws"] = dict(ci_pts=[100 * float(lo), 100 * float(hi)], width_pts=100 * float(hi - lo))
    out["scorings"][name] = rec
    print(f"=== {name}  n {len(y)}  subjects {S}  reproduces: {ok}")
    for k, v in rec["pairs"].items():
        print(f"   {k:34s} " + (v if isinstance(v, str) else
              f"{v['point_pts']:+6.2f}  [{v['ci_pts'][0]:+6.2f}, {v['ci_pts'][1]:+6.2f}]  width {v['width_pts']:5.2f}"))
    print(f"   ensemble marginal, same draws: width {rec['ensemble_marginal_same_draws']['width_pts']:.2f}")
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
