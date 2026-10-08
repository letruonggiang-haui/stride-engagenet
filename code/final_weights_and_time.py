"""Final weights against the EMA weight average, and wall-clock time per run, for every EngageNet inner-fold run
under the locked schedule (12 epochs, EMA 0.999, lr 1e-3). Read-only over the deposited result files; it trains
nothing and opens no official split. Written on 8 October 2026, after every campaign, as a post-hoc reading.

Every result file of ft_video.py stores two scorings of its last epoch: `final`, at the EMA-averaged weights the
paper reports, and `final_raw_weights`, at the last non-averaged weights. `seconds` is the wall-clock time of the
run, training plus the per-epoch scoring of its inner fold, summed across resumed sessions, on one L4 shared with
other users; it is an upper bound on the cost of the run, not a benchmark.

The official scorings (FT_g8_*, FT_g9_*) also stored both readings; they are excluded here and read nowhere in the
paper, because each official split is opened once for the locked configuration, at the averaged weights.

usage: final_weights_and_time.py [results_dir]      prints Table S11 and writes FINAL_WEIGHTS_AND_TIME.json there
                                                    (write-once)
"""
import glob, json, os, sys
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT = os.path.join(RES, "FINAL_WEIGHTS_AND_TIME.json")

# (row label, campaign, result-file glob relative to RES)
ARMS = [("Locked recipe, 16 frames at stride 20", "first",
         ["FT_g6_ep12_ema999_train.json", "FT_g7_*_train.json"]),
        ("A, 16 frames at stride 19", "second", ["FT_v2_T16_f?_train.json"]),
        ("B, 16 frames at stride 9", "second", ["FT_v2_T16s9_f?_train.json"]),
        ("C, 32 frames at stride 9", "second", ["FT_v2_T32_f?_train.json"]),
        ("D, 48 frames at stride 6", "second", ["FT_v2_T48_f?_train.json"]),
        ("61-frame window, 16 frames at stride 4", "third", ["FT_v3_R_f?_s?_train.json"]),
        ("Segment consensus", "third", ["FT_v3_S_f?_s?_train.json"]),
        ("CMS (proposed)", "third", ["FT_v3_C_f?_s?_train.json"]),
        ("CMS after continued pretraining", "fifth", ["FT_v5_P_f?_s?_train.json"]),
        ("CMS", "sixth", ["FT_v6_C_f?_s?_train.json"]),
        ("CMS after continued pretraining", "sixth", ["FT_v6_P_f?_s?_train.json"])]

rows, every = [], []
for label, camp, pats in ARMS:
    fs = sorted({f for p in pats for f in glob.glob(os.path.join(RES, p))})
    js = [json.load(open(f)) for f in fs]
    for f, j in zip(fs, js):
        a = j["args"]
        assert (a["epochs"], a["ema"], a["lr"]) == (12, 0.999, 1e-3), f"{f}: not the locked schedule"
        assert a.get("cv_fold") is not None and not a.get("open_official"), f"{f}: not an inner-fold run"
    d = np.array([[j["final_raw_weights"][m] - j["final"][m] for m in ("macro_f1", "uar", "acc")] for j in js])
    sec = np.array([j["seconds"] for j in js])
    rows.append(dict(arm=label, campaign=camp, runs=len(js), final_ahead_macro_f1=int((d[:, 0] > 0).sum()),
                     mean_diff_macro_f1=float(d[:, 0].mean()), mean_diff_uar=float(d[:, 1].mean()),
                     mean_diff_acc=float(d[:, 2].mean()), seconds_median=float(np.median(sec)),
                     seconds_min=float(sec.min()), seconds_max=float(sec.max()),
                     files=[os.path.basename(f) for f in fs]))
    every += list(d)
every = np.array(every)
summary = dict(runs=len(every), final_ahead_macro_f1=int((every[:, 0] > 0).sum()),
               final_ahead_uar=int((every[:, 1] > 0).sum()), final_ahead_acc=int((every[:, 2] > 0).sum()))

print("%-40s %-7s %4s %5s %8s %8s %8s  %s" % ("arm", "camp", "runs", "ahead", "dF1", "dUAR", "dAcc", "seconds median (range)"))
for r in rows:
    print("%-40s %-7s %4d %5d %+8.4f %+8.4f %+8.4f  %s (%s to %s)" % (
        r["arm"], r["campaign"], r["runs"], r["final_ahead_macro_f1"], r["mean_diff_macro_f1"], r["mean_diff_uar"],
        r["mean_diff_acc"], "{:,.0f}".format(r["seconds_median"]), "{:,.0f}".format(r["seconds_min"]),
        "{:,.0f}".format(r["seconds_max"])))
print("all runs:", summary)
if os.path.exists(OUT):
    old = json.load(open(OUT))
    print("deposited file", "matches" if (old["rows"], old["summary"]) == (rows, summary) else "DIFFERS", "- not overwritten")
else:
    json.dump(dict(note=__doc__.strip().split("\n\n")[0], rows=rows, summary=summary), open(OUT, "w"), indent=1)
    print("wrote", OUT)
