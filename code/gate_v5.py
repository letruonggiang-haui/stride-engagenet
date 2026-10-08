"""Gate G-D of MODEL_V5_DESIGN_2026-10-05 (sha256 c13e32416f4df296273040b4692d73d1f91695761bf3c942a2a804de90103994).

Arm P: the arm-C recipe of v3 started from the encoder adapted by pretrain_v5.py on the training clips of that fold
(--init-ckpt weights/svfap_v5/v5_pre_f<k>.pth). Control: the nine arm-C runs of v3, not rerun. The rule below is the v3 rule
with S replaced by P; the v3 text that follows is kept for the definitions.

Original v3 header:

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
ARMS = {"C": dict(stride=20, segments=0), "P": dict(stride=20, segments=0)}
OUT = os.path.join(ROOT, "results", "GATE_V5.json")


def path(arm, fold, seed):
    return os.path.join(ROOT, "results", f"FT_v{3 if arm == 'C' else 5}_{arm}_f{fold}_s{seed}_train.json")


def _tail(p):
    return "/".join(os.path.normpath(p).replace("\\", "/").split("/")[-3:]) if p else ""


def read(arm, fold, seed):
    p = path(arm, fold, seed)
    if not os.path.exists(p): return None
    d = json.load(open(p)); A = d["args"]
    want = dict(COMMON, cv_fold=fold, seed=seed, **ARMS[arm])
    bad = {k: (A.get(k, 0 if k == "segments" else None), v) for k, v in want.items() if A.get(k, 0 if k == "segments" else None) != v}
    if bad: sys.exit(f"refusing: {os.path.basename(p)} was not run with the pre-registered arguments: {bad}")
    want_init = os.path.join(ROOT, "weights", "svfap_v5", f"v5_pre_f{fold}.pth") if arm == "P" else ""
    # the last three path components are compared, so the deposited files, whose home directory is masked
    # as $HOME, check against the same weights file as the server copies did
    if _tail(A.get("init_ckpt", "")) != _tail(want_init): sys.exit(f"refusing: {os.path.basename(p)} init_ckpt {A.get('init_ckpt')!r}, expected {want_init!r}")
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
if have["P"] < 9 or have["C"] < 9:
    sys.exit("INCOMPLETE: G-D needs all nine P runs and all nine C runs; no verdict computed, nothing written")


def cell(arm, f, m):
    v = [runs[arm][f][s][m] for s in SEEDS]
    return float(np.mean(v)), float(max(v) - min(v)), v


out = {"design": "MODEL_V5_DESIGN_2026-10-05, gate G-D",
       "design_sha256": "c13e32416f4df296273040b4692d73d1f91695761bf3c942a2a804de90103994",
       "threshold_mean_macro_f1": THRESH, "min_folds_beyond_seed_range": MIN_FOLDS,
       "selection_metric": "macro_f1 primary, uar secondary; accuracy never used to select", "per_fold": {}}
d_f1, d_uar, d_acc, beyond = [], [], [], 0
for f in FOLDS:
    sm, sr, sv = cell("P", f, "macro_f1"); cm, cr, cv = cell("C", f, "macro_f1")
    su, _, _ = cell("P", f, "uar"); cu, _, _ = cell("C", f, "uar")
    sa, _, _ = cell("P", f, "acc"); ca, _, _ = cell("C", f, "acc")
    diff = sm - cm; ok = bool(diff > 0 and diff > max(sr, cr)); beyond += ok
    d_f1.append(diff); d_uar.append(su - cu); d_acc.append(sa - ca)
    out["per_fold"][f] = dict(P_macro_f1_seeds=sv, C_macro_f1_seeds=cv, P_mean=sm, C_mean=cm, P_seed_range=sr, C_seed_range=cr,
                              diff_macro_f1=diff, beyond_seed_range=ok, diff_uar=su - cu, diff_acc=sa - ca)
    print(f"fold {f}: P {sm:.4f} (range {sr:.4f})  C {cm:.4f} (range {cr:.4f})  diff {diff:+.4f}  beyond seed range: {ok}  "
          f"UAR diff {su - cu:+.4f}", flush=True)
c1 = bool(np.mean(d_f1) >= THRESH); c2 = bool(beyond >= MIN_FOLDS); c3 = bool(np.mean(d_uar) >= 0)
passed = c1 and c2 and c3
out.update(mean_diff_macro_f1=float(np.mean(d_f1)), folds_beyond_seed_range=int(beyond), mean_diff_uar=float(np.mean(d_uar)),
           mean_diff_acc=float(np.mean(d_acc)), condition_1=c1, condition_2=c2, condition_3=c3, gate_pass=passed,
           verdict=("PASS" if passed else "FAIL") + f": mean macro-F1 P - C {np.mean(d_f1):+.4f} (needs >= +{THRESH}), "
                   f"{beyond}/3 folds positive and beyond the seed range (needs >= {MIN_FOLDS}), mean UAR P - C {np.mean(d_uar):+.4f} (needs >= 0)")

print(out["verdict"], flush=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("wrote", OUT)
