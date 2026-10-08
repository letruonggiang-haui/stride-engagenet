"""Gate G-S of MODEL_V3_DESIGN_2026-10-02 (sha256 6843bb9613b99ebb7310657e95cd46faf290c6ce69939241651aa12e2705ec70).

Three arms, three subject-level inner folds of the EngageNet training split, seeds 0, 1 and 2 in every cell:

    R  inherited recipe      16 frames, stride 4, one random window per clip per epoch
    C  clip-matched stride   16 frames, stride 20
    S  segment consensus     16 frames, stride 4, five windows per clip trained jointly (--segments 5)

R and S are scored on the same five windows with the same probability averaging.

Pre-registered gate G-S, the only gate that decides whether a model paper exists. S passes when all three hold:
  1. the three-fold mean of (seed-mean macro-F1 of S - seed-mean macro-F1 of C) is >= +0.015;
  2. on at least two folds that seed-mean difference is positive AND larger than the larger of the two arms'
     seed ranges (max - min over the three seeds) on that fold;
  3. the three-fold mean UAR difference (S - C) is non-negative.
Macro-F1 selects, UAR is secondary, accuracy is reported and never used to select. R has no gate.

The gate refuses to give a verdict until all 18 S and C runs exist and each was run with the pre-registered
arguments. results/GATE_V3.json is write-once.

usage: gate_v3.py            (verdict, once complete)
       gate_v3.py --status   (lists what exists; computes nothing)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT

FOLDS, SEEDS = (0, 1, 2), (0, 1, 2)
THRESH, MIN_FOLDS = 0.015, 2
COMMON = dict(backbone="svfap", dataset="engagenet", train_splits="train", cv_folds=3, epochs=12, warmup=2, lr=0.001,
              layer_decay=0.75, lora=16, frames=16, smoothing=0.1, ldl=0.0, placeholder=False, balanced=False,
              drop_path=0.1, ema=0.999, tta_windows=5, tta_flip=False, max_train=0, open_official=False)
ARMS = {"R": dict(stride=4, segments=0), "C": dict(stride=20, segments=0), "S": dict(stride=4, segments=5)}
OUT = os.path.join(ROOT, "results", "GATE_V3.json")


def path(arm, fold, seed):
    return os.path.join(ROOT, "results", f"FT_v3_{arm}_f{fold}_s{seed}_train.json")


def read(arm, fold, seed):
    p = path(arm, fold, seed)
    if not os.path.exists(p): return None
    d = json.load(open(p)); A = d["args"]
    want = dict(COMMON, cv_fold=fold, seed=seed, **ARMS[arm])
    bad = {k: (A.get(k, 0 if k == "segments" else None), v) for k, v in want.items() if A.get(k, 0 if k == "segments" else None) != v}
    if bad: sys.exit(f"refusing: {os.path.basename(p)} was not run with the pre-registered arguments: {bad}")
    if A["batch"] * A["accum"] != 32: sys.exit(f"refusing: {os.path.basename(p)} effective batch {A['batch']}x{A['accum']} is not 32 clips")
    if d.get("eval_set") != f"inner_cv_fold_{fold}_of_3": sys.exit(f"refusing: {os.path.basename(p)} eval_set {d.get('eval_set')}")
    f = d["final"]
    return dict(acc=f["acc"], uar=f["uar"], macro_f1=f["macro_f1"], pred_dist=f.get("pred_dist"), seconds=d.get("seconds"),
                majority=d.get("eval_majority_baseline"), n_eval=d.get("n_eval"))


runs = {arm: {f: {s: read(arm, f, s) for s in SEEDS} for f in FOLDS} for arm in ARMS}
have = {arm: sum(r is not None for f in FOLDS for r in runs[arm][f].values()) for arm in ARMS}
print("runs present:", {k: f"{v}/9" for k, v in have.items()}, flush=True)
if "--status" in sys.argv:
    for arm in ARMS:
        for f in FOLDS:
            print(arm, f"fold {f}:", ["done" if runs[arm][f][s] else "-" for s in SEEDS])
    sys.exit(0)
if os.path.exists(OUT):
    sys.exit(f"refusing to overwrite {OUT}: the gate verdict is write-once")
if have["S"] < 9 or have["C"] < 9:
    sys.exit("INCOMPLETE: G-S needs all nine S runs and all nine C runs; no verdict computed, nothing written")


def cell(arm, f, m):
    v = [runs[arm][f][s][m] for s in SEEDS]
    return float(np.mean(v)), float(max(v) - min(v)), v


out = {"design": "MODEL_V3_DESIGN_2026-10-02, gate G-S",
       "design_sha256": "6843bb9613b99ebb7310657e95cd46faf290c6ce69939241651aa12e2705ec70",
       "threshold_mean_macro_f1": THRESH, "min_folds_beyond_seed_range": MIN_FOLDS,
       "selection_metric": "macro_f1 primary, uar secondary; accuracy never used to select", "per_fold": {}}
d_f1, d_uar, d_acc, beyond = [], [], [], 0
for f in FOLDS:
    sm, sr, sv = cell("S", f, "macro_f1"); cm, cr, cv = cell("C", f, "macro_f1")
    su, _, _ = cell("S", f, "uar"); cu, _, _ = cell("C", f, "uar")
    sa, _, _ = cell("S", f, "acc"); ca, _, _ = cell("C", f, "acc")
    diff = sm - cm; ok = bool(diff > 0 and diff > max(sr, cr)); beyond += ok
    d_f1.append(diff); d_uar.append(su - cu); d_acc.append(sa - ca)
    out["per_fold"][f] = dict(S_macro_f1_seeds=sv, C_macro_f1_seeds=cv, S_mean=sm, C_mean=cm, S_seed_range=sr, C_seed_range=cr,
                              diff_macro_f1=diff, beyond_seed_range=ok, diff_uar=su - cu, diff_acc=sa - ca)
    print(f"fold {f}: S {sm:.4f} (range {sr:.4f})  C {cm:.4f} (range {cr:.4f})  diff {diff:+.4f}  beyond seed range: {ok}  "
          f"UAR diff {su - cu:+.4f}", flush=True)
c1 = bool(np.mean(d_f1) >= THRESH); c2 = bool(beyond >= MIN_FOLDS); c3 = bool(np.mean(d_uar) >= 0)
passed = c1 and c2 and c3
out.update(mean_diff_macro_f1=float(np.mean(d_f1)), folds_beyond_seed_range=int(beyond), mean_diff_uar=float(np.mean(d_uar)),
           mean_diff_acc=float(np.mean(d_acc)), condition_1=c1, condition_2=c2, condition_3=c3, gate_pass=passed,
           verdict=("PASS" if passed else "FAIL") + f": mean macro-F1 S - C {np.mean(d_f1):+.4f} (needs >= +{THRESH}), "
                   f"{beyond}/3 folds positive and beyond the seed range (needs >= {MIN_FOLDS}), mean UAR S - C {np.mean(d_uar):+.4f} (needs >= 0)")

# R, the diagnostic: reported whatever the gate says, never part of it
if have["R"] == 9:
    rd = {}
    for arm in ("S", "C"):
        per = [cell(arm, f, "macro_f1")[0] - cell("R", f, "macro_f1")[0] for f in FOLDS]
        rd[f"{arm}_minus_R_macro_f1_per_fold"] = per; rd[f"{arm}_minus_R_macro_f1_mean"] = float(np.mean(per))
        rd[f"{arm}_minus_R_uar_mean"] = float(np.mean([cell(arm, f, "uar")[0] - cell("R", f, "uar")[0] for f in FOLDS]))
    out["diagnostic_R"] = rd
else:
    out["diagnostic_R"] = f"R incomplete ({have['R']}/9); reported when complete"

# environment check: C rerun now against the v1 runs of the same fold and seed (forecast: within 0.010)
V1 = {(0, 0): "FT_g6_ep12_ema999_train.json", (0, 1): "FT_g7_svfap_f0_s1_train.json", (0, 2): "FT_g7_svfap_f0_s2_train.json",
      (1, 0): "FT_g7_svfap_f1_s0_train.json", (2, 0): "FT_g7_svfap_f2_s0_train.json"}
env = {}
for (f, s), fn in V1.items():
    p = os.path.join(ROOT, "results", fn)
    if os.path.exists(p):
        old = json.load(open(p))["final"]["macro_f1"]; env[f"fold{f}_seed{s}"] = runs["C"][f][s]["macro_f1"] - old
out["environment_check_C_new_minus_v1"] = env
if env: out["environment_check_max_abs"] = float(max(abs(v) for v in env.values()))

print(out["verdict"], flush=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("wrote", OUT)
