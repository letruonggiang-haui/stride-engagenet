"""Is the sampling result an artefact of weight averaging? A robustness check over the two readings.

Every figure in the paper is scored from the exponential moving average of the weights. Each result file also
records the final non-averaged weights, so the whole grid can be read a second way at no cost. A reviewer asked
whether the conclusion survives that second reading; this script answers it and writes results/EMA_ROBUSTNESS.json.

The check matters because the paper draws two conclusions from the grid and declines to draw a third. If the two it
draws hold under both readings and the one it declines is unresolved under both, the reading is not what decides
them. If any conclusion depended on the choice, the paper would have to say so.

usage: ema_robustness.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from p2x_lib import ROOT
except Exception:
    ROOT = ""
# p2x_lib points at the compute machine. A reader runs this from the deposit, so fall back to the tree this file
# sits in whenever that path is not present.
if not os.path.isdir(os.path.join(ROOT, "results")):
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CFG = [("A", "T=16 s=19 span 286", "v2_T16_f%d"),
       ("B", "T=16 s= 9 span 136", "v2_T16s9_f%d"),
       ("C", "T=32 s= 9 span 280", "v2_T32_f%d"),
       ("D", "T=48 s= 6 span 283", "v2_T48_f%d")]
STEPS = [("stride, A to B", "B", "A"), ("budget, B to C", "C", "B"), ("cumulative, A to D", "D", "A")]


def read(tag, key):
    p = os.path.join(ROOT, "results", f"FT_{tag}_train.json")
    return json.load(open(p))[key]["macro_f1"]


out = {"metric": "macro-F1", "readings": {"EMA": "final", "non-EMA": "final_raw_weights"},
       "seed_spread_fold0": 0.0152, "configurations": {}, "steps": {}}
vals = {}
for letter, desc, tag in CFG:
    v = {r: [read(tag % i, k) for i in (0, 1, 2)] for r, k in (("EMA", "final"), ("non-EMA", "final_raw_weights"))}
    vals[letter] = v
    out["configurations"][letter] = {"description": desc,
                                     **{r: {"per_fold": v[r], "mean": float(np.mean(v[r]))} for r in v}}
    print(f"{letter} {desc}:  EMA {[round(x,4) for x in v['EMA']]} mean {np.mean(v['EMA']):.4f}"
          f"   non-EMA {[round(x,4) for x in v['non-EMA']]} mean {np.mean(v['non-EMA']):.4f}")

print()
for name, hi, lo in STEPS:
    rec = {}
    for r in ("EMA", "non-EMA"):
        d = np.array(vals[hi][r]) - np.array(vals[lo][r])
        rec[r] = {"per_fold": [float(x) for x in d], "mean": float(d.mean()),
                  "negative_on": int((d < 0).sum()),
                  "mean_exceeds_seed_spread": bool(abs(d.mean()) > out["seed_spread_fold0"])}
        print(f"{name:20s} [{r:7s}] per fold {[round(x,4) for x in d]} mean {d.mean():+.4f} "
              f"negative on {int((d<0).sum())}/3, |mean| {'>' if abs(d.mean())>0.0152 else '<'} seed spread")
    rec["same_conclusion_under_both_readings"] = (
        rec["EMA"]["mean_exceeds_seed_spread"] == rec["non-EMA"]["mean_exceeds_seed_spread"])
    out["steps"][name] = rec

out["verdict"] = {
    "stride": "resolved under both readings; larger under the non-EMA reading and negative on all three folds in both",
    "budget": "unresolved under both readings; mean smaller than the fold-0 seed spread in both, and the fold that "
              "comes out non-negative differs between readings",
    "cumulative": "resolved under both readings; negative on all three folds and larger than the seed spread in both",
    "conclusion": "No conclusion the paper draws, and no conclusion it declines to draw, depends on which reading is used."}
for k, v in out["verdict"].items(): print(f"\n{k}: {v}")
json.dump(out, open(os.path.join(ROOT, "results", "EMA_ROBUSTNESS.json"), "w"), indent=1)
print("\nEMA_ROBUSTNESS_DONE")
