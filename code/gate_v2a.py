"""Gate G-A of MODEL_V2_DESIGN_2026-09-27: does raising the frame budget at fixed temporal extent help?

Three configurations, all with span in 280-290 frames so the temporal EXTENT is held ~constant and only the
frame BUDGET varies, and all on the long-clip branch of the sampler (span <= 300) so positional jitter and the
five-window test-time averaging are genuinely active:

    control  T=16 s=19  span 286
    B32      T=32 s= 9  span 280
    B48      T=48 s= 6  span 283

Pre-registered gate: the best budget must beat the control by >= +0.015 macro-F1 averaged over the three inner
folds AND be non-negative on at least two folds. Selection metric is macro-F1 (primary) with UAR (secondary);
if the two disagree on which configuration wins, nothing is selected and the disagreement is reported. Accuracy
is recorded for leaderboard comparability only and is NEVER used to select. This last rule exists because the v1
campaign pre-registered it and then broke it (FACTS_MODEL_PAPER.md 11.1).

Reads results/FT_v2_T<T>_f<fold>_train.json, writes results/GATE_V2A.json.

usage: gate_v2a.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT

CFG = {16: 19, 32: 9, 48: 6}
FOLDS = (0, 1, 2)
THRESH, MIN_NONNEG = 0.015, 2


def read(T, fold):
    p = os.path.join(ROOT, "results", f"FT_v2_T{T}_f{fold}_train.json")
    if not os.path.exists(p): return None
    d = json.load(open(p)); f = d["final"]
    return dict(acc=f["acc"], uar=f["uar"], macro_f1=f["macro_f1"], pred_dist=f.get("pred_dist"),
                majority=d.get("eval_majority_baseline"), span=d.get("span_frames"), epoch=f.get("epoch"))


out = {"design": "MODEL_V2_DESIGN_2026-09-27 gate G-A", "configs": {}, "threshold": THRESH, "min_nonneg_folds": MIN_NONNEG,
       "selection_metric": "macro_f1 primary, uar secondary; accuracy never used to select"}
for T, s in CFG.items():
    rows = {f: read(T, f) for f in FOLDS}
    have = {f: r for f, r in rows.items() if r}
    out["configs"][f"T{T}"] = dict(frames=T, stride=s, span=(T - 1) * s + 1, folds=have,
                                   n_folds=len(have),
                                   macro_f1_mean=float(np.mean([r["macro_f1"] for r in have.values()])) if have else None,
                                   uar_mean=float(np.mean([r["uar"] for r in have.values()])) if have else None,
                                   acc_mean=float(np.mean([r["acc"] for r in have.values()])) if have else None)

ctl = out["configs"]["T16"]
print(f"control T16 s19: {ctl['n_folds']}/3 folds, macro-F1 {ctl['macro_f1_mean']}, UAR {ctl['uar_mean']}, acc {ctl['acc_mean']}", flush=True)
if ctl["n_folds"] < 3:
    out["verdict"] = "INCOMPLETE: the control needs all three folds before any comparison"
    json.dump(out, open(os.path.join(ROOT, "results", "GATE_V2A.json"), "w"), indent=1, default=float)
    sys.exit(print(out["verdict"]))

cands = {}
for name, c in out["configs"].items():
    if name == "T16" or c["n_folds"] < 3: continue
    per_fold_f1 = [c["folds"][f]["macro_f1"] - ctl["folds"][f]["macro_f1"] for f in FOLDS]
    per_fold_uar = [c["folds"][f]["uar"] - ctl["folds"][f]["uar"] for f in FOLDS]
    cands[name] = dict(gain_macro_f1_per_fold=per_fold_f1, gain_macro_f1_mean=float(np.mean(per_fold_f1)),
                       folds_nonneg=int(sum(x >= 0 for x in per_fold_f1)),
                       gain_uar_per_fold=per_fold_uar, gain_uar_mean=float(np.mean(per_fold_uar)),
                       gain_acc_mean=float(np.mean([c["folds"][f]["acc"] - ctl["folds"][f]["acc"] for f in FOLDS])))
    g = cands[name]
    print(f"{name} vs control: macro-F1 {np.round(per_fold_f1,4).tolist()} mean {g['gain_macro_f1_mean']:+.4f} "
          f"(non-neg {g['folds_nonneg']}/3) | UAR mean {g['gain_uar_mean']:+.4f} | acc mean {g['gain_acc_mean']:+.4f}", flush=True)
out["gains_vs_control"] = cands

if not cands:
    out["verdict"] = "INCOMPLETE: no budget configuration has all three folds yet"
else:
    best_f1 = max(cands, key=lambda k: cands[k]["gain_macro_f1_mean"])
    best_uar = max(cands, key=lambda k: cands[k]["gain_uar_mean"])
    agree = best_f1 == best_uar
    b = cands[best_f1]
    passed = bool(b["gain_macro_f1_mean"] >= THRESH and b["folds_nonneg"] >= MIN_NONNEG and agree)
    out.update(best_by_macro_f1=best_f1, best_by_uar=best_uar, metrics_agree=agree, gate_pass=passed,
               verdict=("PASS" if passed else "FAIL") + f": best by macro-F1 is {best_f1} at {b['gain_macro_f1_mean']:+.4f} "
                       f"(needs >= +{THRESH} and non-negative on >= {MIN_NONNEG}/3 folds); "
                       + ("the two selection metrics agree" if agree else
                          f"the two selection metrics DISAGREE (macro-F1 picks {best_f1}, UAR picks {best_uar}), so nothing is selected"))
    print("\n" + out["verdict"], flush=True)
    if passed:
        print(f"-> lock {best_f1} (T={out['configs'][best_f1]['frames']}, stride={out['configs'][best_f1]['stride']}) and run gate G-B; "
              f"the official splits stay closed until G-B passes too.", flush=True)
    else:
        print("-> the frame-budget axis goes into the paper as a controlled negative; the official splits stay closed.", flush=True)

json.dump(out, open(os.path.join(ROOT, "results", "GATE_V2A.json"), "w"), indent=1, default=float)
print("GATE_V2A_DONE", flush=True)
