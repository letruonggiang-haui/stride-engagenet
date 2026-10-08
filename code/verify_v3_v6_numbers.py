"""Regenerate, at the printed rounding, every value the manuscript reports from the four campaigns that followed
the official scoring (Section 4.4, Table 14, Article S1 Section C.8, Tables S8 and S9) and from the full count of
clip lengths (Sections 3.3 and 4.1.1). Reads only deposited result files; prints "printed / recomputed / verdict"
for each value and exits non-zero on any mismatch.

usage: verify_v3_v6_numbers.py [results_dir]      (default: results/ beside this code directory)
"""
import json, os, sys
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
J = lambda f: json.load(open(os.path.join(RES, f)))
F = lambda tag: J(f"FT_{tag}_train.json")["final"]


def arm(prefix, seeds=(0, 1, 2)):
    return {m: np.array([[F(f"{prefix}_f{f}_s{s}")[m] for s in seeds] for f in range(3)]) for m in ("macro_f1", "uar", "acc")}


R, S, C = arm("v3_R"), arm("v3_S"), arm("v3_C")
P5, P6, C6 = arm("v5_P"), arm("v6_P", (3, 4, 5)), arm("v6_C", (3, 4, 5))
K = J("KSHOT_V4_INNER.json"); L = J("CLIP_LENGTHS_V3.json"); G5 = J("GATE_V5.json"); G6 = J("GATE_V6.json")
pct = lambda x: "%.2f" % (100 * x)
r4 = lambda x: "%+.4f" % x
d = lambda a, b, m="macro_f1": a[m].mean(1) - b[m].mean(1)

checks = [
    # clip lengths, Sections 3.3, 4.1.1, C.5
    ("clips counted", "11,311", "{:,}".format(L["n_clips"])),
    ("reach 301 frames", "12.8", "%.1f" % (100 * L["frac_ge_301"])),
    ("short branch", "87.2", "%.1f" % (100 * (1 - L["frac_ge_301"]))),
    ("shorter than 281", "12.1", "%.1f" % (100 * L["frac_lt_281"])),
    ("maximum length", "330", str(L["max"])),
    # Table 14
    ("61-frame window mean", "49.59", pct(R["macro_f1"].mean())),
    ("segment consensus mean", "53.11", pct(S["macro_f1"].mean())),
    ("CMS mean, third", "58.25", pct(C["macro_f1"].mean())),
    ("continued pretraining mean, fifth", "59.88", pct(P5["macro_f1"].mean())),
    ("continued pretraining mean, sixth", "59.11", pct(P6["macro_f1"].mean())),
    ("CMS mean, sixth", "58.00", pct(C6["macro_f1"].mean())),
    ("61-frame window minus CMS", "-0.0865", r4(-d(C, R).mean())),
    ("segment consensus minus CMS", "-0.0513", r4(d(S, C).mean())),
    ("fifth minus CMS", "+0.0164", r4(d(P5, C).mean())),
    ("sixth minus its CMS", "+0.0111", r4(d(P6, C6).mean())),
    ("folds positive, 61-frame vs CMS", "0", str(int((d(R, C) > 0).sum()))),
    ("folds positive, consensus vs CMS", "0", str(int((d(S, C) > 0).sum()))),
    ("folds positive, fifth", "3", str(int((d(P5, C) > 0).sum()))),
    ("folds positive, sixth", "3", str(int((d(P6, C6) > 0).sum()))),
    # Section 4.4 prose
    ("CMS ahead of 61-frame window, paired runs", "9", str(int(((C["macro_f1"] - R["macro_f1"]) > 0).sum()))),
    ("CMS-61 beyond either seed range on every fold", "3", str(int(sum(
        d(C, R)[f] > max(np.ptp(R["macro_f1"][f]), np.ptp(C["macro_f1"][f])) for f in range(3))))),
    ("consensus minus 61-frame window", "+0.0352", r4(d(S, R).mean())),
    ("consensus ahead, paired runs", "9", str(int(((S["macro_f1"] - R["macro_f1"]) > 0).sum()))),
    ("fifth UAR", "+0.0190", r4(d(P5, C, "uar").mean())),
    ("sixth UAR", "+0.0127", r4(d(P6, C6, "uar").mean())),
    ("fifth folds beyond seed range", "1", str(G5["folds_beyond_seed_range"])),
    ("sixth folds beyond seed range", "1", str(G6["folds_beyond_seed_range"])),
    ("same-seed pairs favouring pretraining, macro-F1", "16", str(int(((P5["macro_f1"] - C["macro_f1"]) > 0).sum() + ((P6["macro_f1"] - C6["macro_f1"]) > 0).sum()))),
    ("same-seed pairs favouring pretraining, accuracy", "14", str(int(((P5["acc"] - C["acc"]) > 0).sum() + ((P6["acc"] - C6["acc"]) > 0).sum()))),
    ("pooled gain", "+0.0137", r4(np.mean(np.r_[(P5["macro_f1"] - C["macro_f1"]).ravel(), (P6["macro_f1"] - C6["macro_f1"]).ravel()]))),
    ("consensus UAR", "-0.0438", r4(d(S, C, "uar").mean())),
    # calibration, Section 4.4 and Table S9
    ("calibration gain, k = 5", "+0.0063", r4(K["mean_diff_macro_f1_M1_M0"])),
    ("calibration folds with CI above 0", "0", str(K["folds_ci_lower_above_zero"])),
    ("calibration gain, k = 20", "+0.0119", r4(np.mean([K["folds"][f]["k_curve"]["20"]["M1"]["macro_f1"] - K["folds"][f]["k_curve"]["20"]["M0"]["macro_f1"] for f in "012"]))),
    ("majority below uncalibrated", "0.19", "%.2f" % np.mean([K["folds"][f]["M0"]["macro_f1"] - K["folds"][f]["B1"]["macro_f1"] for f in "012"])),
    ("calibration accuracy gain", "+0.0060", r4(np.mean([K["folds"][f]["diff_acc_M1_M0"] for f in "012"]))),
]
for k in ("1", "3", "10"):
    v = np.mean([K["folds"][f]["k_curve"][k]["M1"]["macro_f1"] - K["folds"][f]["k_curve"][k]["M0"]["macro_f1"] for f in "012"])
    checks.append(("calibration gain, k = " + k, {"1": "+0.0028", "3": "+0.0043", "10": "+0.0110"}[k], r4(v)))
# Table S8, by fold
for name, A, printed in (("61-frame", R, ["53.60 (0.84)", "42.54 (3.53)", "52.65 (2.20)"]),
                         ("consensus", S, ["56.52 (1.50)", "47.65 (0.91)", "55.17 (2.26)"]),
                         ("CMS third", C, ["58.97 (1.52)", "56.31 (0.59)", "59.46 (2.12)"]),
                         ("pretraining fifth", P5, ["59.76 (0.77)", "58.80 (1.29)", "61.09 (1.81)"]),
                         ("CMS sixth", C6, ["58.63 (0.92)", "56.73 (2.45)", "58.63 (2.05)"]),
                         ("pretraining sixth", P6, ["60.06 (1.36)", "56.88 (2.88)", "60.38 (1.27)"])):
    for f in range(3):
        v = A["macro_f1"][f]
        checks.append(("S8 %s fold %d" % (name, f), printed[f], "%s (%.2f)" % (pct(v.mean()), 100 * np.ptp(v))))
# Table S9, by fold
S9 = {"0": ("24", "1,668", "59.24", "59.30", "35.80", "+0.0006 [-0.0097, +0.0104]"),
      "1": ("26", "1,756", "57.73", "58.82", "38.89", "+0.0109 [-0.0083, +0.0321]"),
      "2": ("28", "1,924", "61.91", "62.65", "47.99", "+0.0074 [-0.0045, +0.0219]")}
for f, row in S9.items():
    Kf = K["folds"][f]
    got = (str(Kf["eligible_students"]), "{:,}".format(Kf["query_clips"]), pct(Kf["M0"]["macro_f1"]), pct(Kf["M1"]["macro_f1"]),
           pct(Kf["B1"]["macro_f1"]), "%s [%s, %s]" % (r4(Kf["diff_macro_f1_M1_M0"]), r4(Kf["ci95_macro_f1_M1_M0"][0]), r4(Kf["ci95_macro_f1_M1_M0"][1])))
    for i, (p, g) in enumerate(zip(row, got)):
        checks.append(("S9 fold %s column %d" % (f, i + 2), p, g))
pre = [J(f"PRE_{r}_pre_f{f}.json")["log"] for r in ("v5", "v6") for f in range(3)]
checks.append(("pretraining loss, first epoch", "0.57 to 0.59", "%.2f to %.2f" % (min(l[0]["loss"] for l in pre), max(l[0]["loss"] for l in pre))))
checks.append(("pretraining loss, last epoch", "0.37 to 0.38", "%.2f to %.2f" % (min(l[-1]["loss"] for l in pre), max(l[-1]["loss"] for l in pre))))
checks.append(("pretraining epochs", "30", str(len(pre[0]))))
# the optimiser constants Article S1, Section C.8 prints, read from the args of all six pretraining runs
pargs = [J(f"PRE_{r}_pre_f{f}.json")["args"] for r in ("v5", "v6") for f in range(3)]
for name, printed, key, fmt in (("pretraining learning rate", "1e-4", "lr", "%.0e"), ("pretraining final learning rate", "1e-6", "min_lr", "%.0e"),
                                ("pretraining weight decay", "0.05", "wd", "%.2f"), ("pretraining warm-up epochs", "2", "warmup", "%d"),
                                ("pretraining batch, clips", "32", "batch", "%d"), ("pretraining mask ratio", "0.9", "mask_ratio", "%.1f")):
    vals = {(fmt % a[key]).replace("e-0", "e-") for a in pargs}
    checks.append((name, printed, vals.pop() if len(vals) == 1 else "differs across runs: %s" % sorted(vals)))
checks.append(("pretraining accumulation", "1", str({a["accum"] for a in pargs}.pop())))
PCI = J("PAIRED_CI_V3.json")
checks.append(("paired CI, CMS - 61-frame", "+0.0688, +0.1046", "%+.4f, %+.4f" % tuple(PCI["CMS_minus_61_frame_window"]["ci95"])))
checks.append(("paired CI point, CMS - 61-frame", "+0.0865", r4(PCI["CMS_minus_61_frame_window"]["point"])))
checks.append(("paired CI, consensus - CMS", "-0.0656, -0.0356", "%+.4f, %+.4f" % tuple(PCI["segment_consensus_minus_CMS"]["ci95"])))
checks.append(("CMS seed ranges across folds, min to max", "0.0059 to 0.0212", "%.4f to %.4f" % (min(np.ptp(C["macro_f1"], 1)), max(np.ptp(C["macro_f1"], 1)))))
checks.append(("S9 excluded students", "7, 4, 2", ", ".join(str(len(K["folds"][f]["excluded_students"])) for f in "012")))
checks.append(("S9 smoothing strength", "50, 20, 50", ", ".join(str(K["folds"][f]["alpha_chosen_on_other_folds"]) for f in "012")))

bad = 0
for name, printed, got in checks:
    ok = printed.replace("-", "−") == got.replace("-", "−")
    bad += not ok
    print("%-50s %-28s %-28s %s" % (name, printed, got, "OK" if ok else "MISMATCH"))
print("\n%d values, %d mismatches" % (len(checks), bad))
sys.exit(1 if bad else 0)
