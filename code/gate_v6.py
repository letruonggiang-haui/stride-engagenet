"""Gate G-D6 of MODEL_V6_DESIGN_2026-10-06 (sha256 0b65307dc848ebe704000662c1081ebf9f8d74df1e0412a16d8788115899307c),
the confirmation round of v5.

New runs only: per inner fold, one new continued pretraining (pretrain_v5.py, seed 1, tag v6_pre_f<k>) and fine-tuning seeds
3, 4, 5 for both arms:
    P  CMS fine-tuning recipe from weights/svfap_v5/v6_pre_f<k>.pth   results/FT_v6_P_f<k>_s<s>_train.json
    C  CMS fine-tuning recipe from the released checkpoint             results/FT_v6_C_f<k>_s<s>_train.json

The rule is G-D of v5 unchanged. P passes when all three hold:
  1. the three-fold mean of (seed-mean macro-F1 of P - seed-mean macro-F1 of C) is >= +0.015;
  2. on at least two folds that seed-mean difference is positive AND larger than the larger of the two arms' seed ranges;
  3. the three-fold mean UAR difference (P - C) is non-negative.
The v5 runs are never part of the decision; a pooled v5 + v6 difference is reported beside it and marked as such.

The gate refuses a verdict until all 18 runs and all three pretrainings exist with the pre-registered arguments.
results/GATE_V6.json is write-once.

usage: gate_v6.py            (verdict, once complete)
       gate_v6.py --status   (lists what exists; computes nothing)
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT

DESIGN_SHA = "0b65307dc848ebe704000662c1081ebf9f8d74df1e0412a16d8788115899307c"
FOLDS, SEEDS = (0, 1, 2), (3, 4, 5)
THRESH, MIN_FOLDS = 0.015, 2
COMMON = dict(backbone="svfap", dataset="engagenet", train_splits="train", cv_folds=3, epochs=12, warmup=2, lr=0.001,
              layer_decay=0.75, lora=16, frames=16, stride=20, segments=0, smoothing=0.1, ldl=0.0, placeholder=False,
              balanced=False, drop_path=0.1, ema=0.999, tta_windows=5, tta_flip=False, max_train=0, open_official=False)
PRE = dict(dataset="engagenet", train_splits="train", cv_folds=3, frames=16, stride=20, epochs=30, warmup=2, lr=0.0001,
           min_lr=1e-06, wd=0.05, mask_ratio=0.9, seed=1, max_train=0)
OUT = os.path.join(ROOT, "results", "GATE_V6.json")
res = lambda *p: os.path.join(ROOT, "results", *p)


def init_for(arm, fold):
    return os.path.join(ROOT, "weights", "svfap_v5", f"v6_pre_f{fold}.pth") if arm == "P" else ""


def _tail(p):
    return "/".join(os.path.normpath(p).replace("\\", "/").split("/")[-3:]) if p else ""


def read(arm, fold, seed, tagver="v6"):
    p = res(f"FT_{tagver}_{arm}_f{fold}_s{seed}_train.json")
    if not os.path.exists(p): return None
    d = json.load(open(p)); A = d["args"]
    want = dict(COMMON, cv_fold=fold, seed=seed)
    bad = {k: (A.get(k, 0 if k == "segments" else None), v) for k, v in want.items() if A.get(k, 0 if k == "segments" else None) != v}
    if bad: sys.exit(f"refusing: {os.path.basename(p)} was not run with the pre-registered arguments: {bad}")
    want_init = init_for(arm, fold) if tagver == "v6" else (os.path.join(ROOT, "weights", "svfap_v5", f"v5_pre_f{fold}.pth") if arm == "P" else "")
    # the last three path components are compared, so the deposited files, whose home directory is masked
    # as $HOME, check against the same weights file as the server copies did
    if _tail(A.get("init_ckpt", "")) != _tail(want_init):
        sys.exit(f"refusing: {os.path.basename(p)} init_ckpt {A.get('init_ckpt')!r}, expected {want_init!r}")
    if A["batch"] * A["accum"] != 32: sys.exit(f"refusing: {os.path.basename(p)} effective batch {A['batch']}x{A['accum']} is not 32 clips")
    if d.get("eval_set") != f"inner_cv_fold_{fold}_of_3": sys.exit(f"refusing: {os.path.basename(p)} eval_set {d.get('eval_set')}")
    f = d["final"]
    return dict(acc=f["acc"], uar=f["uar"], macro_f1=f["macro_f1"], pred_dist=f.get("pred_dist"), seconds=d.get("seconds"),
                resumed_at_epochs=d.get("resumed_at_epochs", []))


def read_pre(fold):
    p = res(f"PRE_v6_pre_f{fold}.json")
    if not os.path.exists(p): return None
    A = json.load(open(p))["args"]
    bad = {k: (A.get(k), v) for k, v in dict(PRE, cv_fold=fold).items() if A.get(k) != v}
    if bad: sys.exit(f"refusing: {os.path.basename(p)} was not run with the pre-registered arguments: {bad}")
    return True


runs = {arm: {f: {s: read(arm, f, s) for s in SEEDS} for f in FOLDS} for arm in ("C", "P")}
pre = {f: read_pre(f) for f in FOLDS}
have = {arm: sum(r is not None for f in FOLDS for r in runs[arm][f].values()) for arm in runs}
print("pretraining present:", f"{sum(bool(v) for v in pre.values())}/3", " runs present:", {k: f"{v}/9" for k, v in have.items()}, flush=True)
if "--status" in sys.argv:
    for f in FOLDS:
        print(f"fold {f}: pretraining {'done' if pre[f] else '-'}", *[f"{arm} {['done' if runs[arm][f][s] else '-' for s in SEEDS]}" for arm in runs])
    sys.exit(0)
if os.path.exists(OUT): sys.exit(f"refusing to overwrite {OUT}: the gate verdict is write-once")
if not all(pre.values()) or have["P"] < 9 or have["C"] < 9:
    sys.exit("INCOMPLETE: G-D6 needs the three pretrainings and all 18 runs; no verdict computed, nothing written")


def cell(R, arm, f, m, seeds):
    v = [R[arm][f][s][m] for s in seeds]
    return float(np.mean(v)), float(max(v) - min(v)), v


out = {"design": "MODEL_V6_DESIGN_2026-10-06, gate G-D6 (the G-D rule of v5 on new runs only)", "design_sha256": DESIGN_SHA,
       "threshold_mean_macro_f1": THRESH, "min_folds_beyond_seed_range": MIN_FOLDS, "seeds": list(SEEDS), "pretraining_seed": 1,
       "selection_metric": "macro_f1 primary, uar secondary; accuracy never used to select", "per_fold": {}}
d_f1, d_uar, d_acc, beyond = [], [], [], 0
for f in FOLDS:
    pm, pr, pv = cell(runs, "P", f, "macro_f1", SEEDS); cm, cr, cv = cell(runs, "C", f, "macro_f1", SEEDS)
    pu = cell(runs, "P", f, "uar", SEEDS)[0]; cu = cell(runs, "C", f, "uar", SEEDS)[0]
    pa = cell(runs, "P", f, "acc", SEEDS)[0]; ca = cell(runs, "C", f, "acc", SEEDS)[0]
    diff = pm - cm; ok = bool(diff > 0 and diff > max(pr, cr)); beyond += ok
    paired = [runs["P"][f][s]["macro_f1"] - runs["C"][f][s]["macro_f1"] for s in SEEDS]
    d_f1.append(diff); d_uar.append(pu - cu); d_acc.append(pa - ca)
    out["per_fold"][f] = dict(P_macro_f1_seeds=pv, C_macro_f1_seeds=cv, P_mean=pm, C_mean=cm, P_seed_range=pr, C_seed_range=cr,
                              diff_macro_f1=diff, beyond_seed_range=ok, diff_uar=pu - cu, diff_acc=pa - ca,
                              same_seed_differences_macro_f1=paired)
    print(f"fold {f}: P {pm:.4f} (range {pr:.4f})  C {cm:.4f} (range {cr:.4f})  diff {diff:+.4f}  beyond seed range: {ok}  "
          f"UAR diff {pu - cu:+.4f}  same-seed diffs {np.round(paired, 4).tolist()}", flush=True)
c1 = bool(np.mean(d_f1) >= THRESH); c2 = bool(beyond >= MIN_FOLDS); c3 = bool(np.mean(d_uar) >= 0)
passed = c1 and c2 and c3
out.update(mean_diff_macro_f1=float(np.mean(d_f1)), folds_beyond_seed_range=int(beyond), mean_diff_uar=float(np.mean(d_uar)),
           mean_diff_acc=float(np.mean(d_acc)), condition_1=c1, condition_2=c2, condition_3=c3, gate_pass=passed,
           verdict=("PASS" if passed else "FAIL") + f": mean macro-F1 P - C {np.mean(d_f1):+.4f} (needs >= +{THRESH}), "
                   f"{beyond}/3 folds positive and beyond the seed range (needs >= {MIN_FOLDS}), mean UAR P - C {np.mean(d_uar):+.4f} (needs >= 0)")

# reported beside the verdict, never part of it: v5 and v6 pooled, six seeds per arm per fold
try:
    v5 = {"C": {f: {s: read("C", f, s, "v3") for s in (0, 1, 2)} for f in FOLDS},
          "P": {f: {s: read("P", f, s, "v5") for s in (0, 1, 2)} for f in FOLDS}}
    if all(v5[a][f][s] for a in v5 for f in FOLDS for s in (0, 1, 2)):
        pooled = []
        for f in FOLDS:
            P6 = [runs["P"][f][s]["macro_f1"] for s in SEEDS] + [v5["P"][f][s]["macro_f1"] for s in (0, 1, 2)]
            C6 = [runs["C"][f][s]["macro_f1"] for s in SEEDS] + [v5["C"][f][s]["macro_f1"] for s in (0, 1, 2)]
            pooled.append(float(np.mean(P6) - np.mean(C6)))
        out["pooled_v5_v6_not_used_for_the_decision"] = dict(diff_macro_f1_per_fold=pooled, mean=float(np.mean(pooled)))
except SystemExit as e:
    out["pooled_v5_v6_not_used_for_the_decision"] = f"not computed: {e}"

print(out["verdict"], flush=True)
json.dump(out, open(OUT, "w"), indent=1, default=float)
print("wrote", OUT)
