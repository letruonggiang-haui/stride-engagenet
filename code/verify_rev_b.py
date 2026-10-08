"""Regenerate, at the printed rounding, every value that entered the manuscript on 8 October 2026 with the second
review round (stage 59): the last-weight readings and run times of Article S1 Table S11 and Sections 3.7, 4.1.3 and
4.4; the paired official-split intervals of Table S12 and Sections 4.1.6 and 5.4; the ensemble's stop rule in
Section 3.5 and Table 13; and the member seed spreads of Section 4.1.4. Reads deposited files only; prints
"printed / recomputed / verdict" and exits non-zero on any mismatch.

Table S11 is recomputed here from the result files themselves, not read back from FINAL_WEIGHTS_AND_TIME.json.

usage: verify_rev_b.py [results_dir]      (default: results/ beside this code directory)
"""
import glob, json, os, re, sys
import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
PRE = os.path.join(os.path.dirname(RES), "preregistration")
J = lambda f: json.load(open(os.path.join(RES, f)))
FT = lambda pat: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, pat)))]
r4 = lambda x: "%+.4f" % x
checks = []

# ---------------------------------------------------------------- Table S11, recomputed from the result files
ARMS = [("first", ["FT_g6_ep12_ema999_train.json", "FT_g7_*_train.json"], "6 4 +0.0091 +0.0217 -0.0129 6,202 4,247 7,909"),
        ("A", ["FT_v2_T16_f?_train.json"], "3 3 +0.0340 +0.0420 +0.0094 3,144 3,131 3,218"),
        ("B", ["FT_v2_T16s9_f?_train.json"], "3 2 +0.0176 +0.0273 -0.0015 5,018 4,483 5,319"),
        ("C", ["FT_v2_T32_f?_train.json"], "3 2 +0.0196 +0.0262 -0.0081 9,717 9,591 9,971"),
        ("D", ["FT_v2_T48_f?_train.json"], "3 3 +0.0327 +0.0369 +0.0026 18,950 18,736 20,492"),
        ("61-frame", ["FT_v3_R_f?_s?_train.json"], "9 7 +0.0232 +0.0266 -0.0053 5,490 3,102 5,630"),
        ("consensus", ["FT_v3_S_f?_s?_train.json"], "9 9 +0.0296 +0.0324 +0.0056 9,340 9,090 16,688"),
        ("CMS third", ["FT_v3_C_f?_s?_train.json"], "9 7 +0.0169 +0.0252 +0.0017 3,147 3,104 3,238"),
        ("pretraining fifth", ["FT_v5_P_f?_s?_train.json"], "9 6 +0.0143 +0.0225 -0.0039 3,151 3,107 3,230"),
        ("CMS sixth", ["FT_v6_C_f?_s?_train.json"], "9 8 +0.0193 +0.0287 +0.0023 3,142 3,106 5,669"),
        ("pretraining sixth", ["FT_v6_P_f?_s?_train.json"], "9 6 +0.0182 +0.0273 -0.0011 3,145 3,105 5,679")]
allD = []
for name, pats, printed in ARMS:
    js = [j for p in pats for j in FT(p)]
    for j in js:
        assert (j["args"]["epochs"], j["args"]["ema"], j["args"]["lr"]) == (12, 0.999, 1e-3) and j["args"].get("cv_fold") is not None
    d = np.array([[j["final_raw_weights"][m] - j["final"][m] for m in ("macro_f1", "uar", "acc")] for j in js]); allD += list(d)
    s = np.array([j["seconds"] for j in js])
    got = "%d %d %s %s %s %s %s %s" % (len(js), (d[:, 0] > 0).sum(), r4(d[:, 0].mean()), r4(d[:, 1].mean()), r4(d[:, 2].mean()),
                                       "{:,.0f}".format(np.median(s)), "{:,.0f}".format(s.min()), "{:,.0f}".format(s.max()))
    checks.append(("S11 " + name, printed, got))
allD = np.array(allD)
checks.append(("runs under the locked schedule", "72", str(len(allD))))
checks.append(("last weights ahead, macro-F1", "57", str(int((allD[:, 0] > 0).sum()))))
checks.append(("last weights ahead, UAR", "67", str(int((allD[:, 1] > 0).sum()))))
sel = FT("FT_g6_ep12_ema999_train.json")[0]
checks.append(("schedule run, average ahead on macro-F1", "0.0112", "%.4f" % (sel["final"]["macro_f1"] - sel["final_raw_weights"]["macro_f1"])))
checks.append(("schedule run, average ahead on accuracy", "0.0266", "%.4f" % (sel["final"]["acc"] - sel["final_raw_weights"]["acc"])))
C = np.array([[j["final_raw_weights"]["macro_f1"] for j in FT("FT_v3_C_f%d_s?_train.json" % f)] for f in range(3)])
R = np.array([[j["final_raw_weights"]["macro_f1"] for j in FT("FT_v3_R_f%d_s?_train.json" % f)] for f in range(3)])
checks.append(("CMS - 61-frame at last weights", "+0.0802", r4((C - R).mean())))
checks.append(("pairs positive at last weights", "9", str(int(((C - R) > 0).sum()))))

# ---------------------------------------------------------------- Table S12 and Section 4.1.6
P = J("PAIRED_OFFICIAL_CI.json")["scorings"]
S12 = {"PIXEL_ENSEMBLE - geometry": ("+4.39 [-0.93, +11.46]", "+2.04 [-1.41, +5.20]", "+1.55 [-1.85, +4.77]"),
       "FUSED_w05 - PIXEL_ENSEMBLE": ("-2.71 [-8.89, +1.89]", "-0.31 [-2.91, +2.71]", "+0.22 [-2.60, +3.29]"),
       "maedfer_s0 - svfap_s0": ("-3.08 [-7.88, +0.09]", "+0.09 [-1.89, +2.08]", "+0.53 [-2.53, +3.50]"),
       "maedfer_s0 - svfap_s1": ("-2.15 [-5.81, +0.56]", "-0.44 [-2.56, +1.34]", "-1.24 [-3.54, +0.85]"),
       "maedfer_s0 - svfap_s2": ("-2.61 [-5.87, -0.17]", "-0.04 [-2.22, +1.86]", "-0.40 [-2.91, +1.85]"),
       "svfap_s0 - svfap_s1": ("+0.93 [-0.48, +2.57]", "-0.53 [-3.36, +1.85]", "-1.77 [-3.66, -0.09]"),
       "svfap_s0 - svfap_s2": ("+0.47 [-1.13, +2.34]", "-0.13 [-2.55, +2.11]", "-0.93 [-2.92, +1.04]"),
       "svfap_s1 - svfap_s2": ("-0.47 [-2.24, +1.18]", "+0.40 [-0.86, +1.66]", "+0.84 [-0.54, +2.33]")}
SC = ("VAL_train_only", "TEST_train_only", "TEST_train_plus_val")
for pair, row in S12.items():
    for s, printed in zip(SC, row):
        v = P[s]["pairs"][pair]
        checks.append(("S12 %s %s" % (pair, s), printed, "%+.2f [%+.2f, %+.2f]" % (v["point_pts"], *v["ci_pts"])))
for s in SC:
    checks.append(("every point accuracy reproduces G8_OFFICIAL, " + s, "True", str(all(P[s]["reproduces_G8_OFFICIAL"].values()))))
W = [P[s]["pairs"]["PIXEL_ENSEMBLE - geometry"]["width_pts"] for s in SC]
checks.append(("primary pair width, validation", "12.4", "%.1f" % W[0]))
checks.append(("primary pair width, test, both protocols", "6.6 6.6", "%.1f %.1f" % (W[1], W[2])))
checks.append(("half the validation width", "6.2", "%.1f" % (W[0] / 2)))
seedw = [P[s]["pairs"][k]["width_pts"] for s in SC for k in ("svfap_s0 - svfap_s1", "svfap_s0 - svfap_s2", "svfap_s1 - svfap_s2")]
checks.append(("seed-pair widths, range", "2.5 to 5.2", "%.1f to %.1f" % (min(seedw), max(seedw))))
# the published validation figures of Table 1 (PriorNet, Ordinal ST-GCN, ICMI, VE-MD, TCCT-Net, EmotiW baseline)
pub = sorted([73.58, 71.24, 69.10, 68.98, 68.91, 65.54], reverse=True)
gaps = np.round(-np.diff(pub), 2)
checks.append(("adjacent gaps, range", "0.07 to 3.37", "%.2f to %.2f" % (gaps.min(), gaps.max())))
checks.append(("adjacent gaps above half the validation width", "0", str(int((gaps > W[0] / 2).sum()))))
checks.append(("validation width more than twice every gap", "True", str(bool(W[0] > 2 * gaps.max()))))

# ---------------------------------------------------------------- Section 3.5, the ensemble's stop rule
g7 = J("G7_FUSED.json")["fold0"]
best = max(g7["members"].items(), key=lambda kv: kv[1]["acc"])
checks.append(("fold-0 ensemble accuracy at the lock", "70.47", "%.2f" % (100 * g7["pixel_ensemble"]["acc"])))
checks.append(("fold-0 best single backbone", "70.87 edfer_f0_s0", "%.2f %s" % (100 * best[1]["acc"], best[0])))
checks.append(("ensemble below it, points, from unrounded values", "0.41", "%.2f" % (100 * (best[1]["acc"] - g7["pixel_ensemble"]["acc"]))))
f1 = {k: v["macro_f1"] for k, v in g7["members"].items()}
sv = [v for k, v in f1.items() if "fap" in k or k == "12_ema999"]
ens = g7["pixel_ensemble"]["macro_f1"]
checks.append(("macro-F1: verdict depends on best seed vs seed mean", "below best seed, above mean by > 0.5",
               "%s best seed, %s mean by > 0.5" % ("below" if ens < max(f1.values()) else "above",
                                                    "above" if 100 * (ens - np.mean(sv)) > 0.5 else "not above")))
G8 = J("G8_OFFICIAL.json")
mae = [100 * G8[s]["members"][[k for k in G8[s]["members"] if "maedfer" in k][0]]["acc"] for s in SC]
checks.append(("MAE-DFER-B alone on the official splits", "69.09 69.73 69.33", "%.2f %.2f %.2f" % tuple(mae)))

# ---------------------------------------------------------------- Section 4.1.4, member seed spreads
for s, printed in (("TEST_train_only", "0.53"), ("VAL_train_only", "0.93"), ("TEST_train_plus_val", "1.77")):
    sv = [v["acc"] for k, v in G8[s]["members"].items() if "svfap" in k]
    checks.append(("SVFAP seed spread, " + s, printed, "%.2f" % (100 * (max(sv) - min(sv)))))

# ---------------------------------------------------------------- Article S1 A.6, the fusion-weight sweep
_recs = [os.path.join(d, "MODEL_PAPER_DESIGN_2026-09-21.md") for d in (PRE, os.path.dirname(RES))]   # deposit layout, then the working tree
rec = next((open(f, encoding="utf-8").read() for f in _recs if os.path.exists(f)), "")
m = re.search(r"\(0,(\d{4}) → 0,(\d{4}) → 0,(\d{4})\)", rec)
checks.append(("w sweep around 0.5, from the record of 22 September", "0.6846 0.6993 0.6895",
               " ".join("0." + x for x in m.groups()) if m else "record not found beside results/"))
checks.append(("forecasts re-anchored to", "0.685", "0.685" if "khoảng 0,685" in rec else "not found"))

# ---------------------------------------------------------------- final audit, 8 October 2026 (stages 66 and 67)
f1 = lambda tag: FT("FT_%s_train.json" % tag)[0]["final"]["macro_f1"]
checks.append(("LoRA row of Table 6, from unrounded values", "+0.0060", r4(f1("g2_svfap_lora16") - f1("g1_svfap_full"))))
checks.append(("schedule correction, macro-F1, from unrounded values", "+0.0080", r4(f1("g6_ep12_ema999") - f1("g3_svfap_locked_s0"))))
lr = lambda tag: FT("FT_%s_train.json" % tag)[0]["args"]["lr"]
checks.append(("learning rates: reference, LoRA row, 30-epoch row, locked", "0.0005 0.001 0.0005 0.001",
               "%g %g %g %g" % (lr("g1_svfap_full"), lr("g2_svfap_lora16"), lr("g3_svfap_locked_s0"), lr("g6_ep12_ema999"))))
s0 = lambda pat: [j["seconds"] for j in FT(pat)]
checks.append(("fold-0 run times, 61-frame arm and CMS, third campaign", "3,102 to 3,112; 3,104 to 3,112",
               "{:,.0f} to {:,.0f}; {:,.0f} to {:,.0f}".format(min(s0("FT_v3_R_f0_s?_train.json")), max(s0("FT_v3_R_f0_s?_train.json")),
                                                              min(s0("FT_v3_C_f0_s?_train.json")), max(s0("FT_v3_C_f0_s?_train.json")))))
CL = J("CLIP_LENGTHS_V3.json"); n10 = int(CL["percentiles"]["10"])
base = list(range(0, n10, 20)); win = (base + [n10 - 1] * 16)[:16]
checks.append(("10th-percentile clip: length, distinct frames, positions of the last", "167 10 7",
               "%d %d %d" % (n10, len(set(win)), win.count(n10 - 1))))
checks.append(("coverage of the 301-frame window on the longest clip", "91", "%d" % int(100 * 301 / CL["max"])))
g8 = J("G8_OFFICIAL.json")
used = sum(g8[s]["n"] for s in ("VAL_train_only", "TEST_train_only")) + FT("FT_g8_svfap_s0_train.json")[0].get("n_train", 0)
checks.append(("clips the runs use, and the cache excess", "11,206 105", "{:,} {}".format(used, CL["n_clips"] - used)))
PC = {r["backbone"]: r for r in J("PARAM_COUNTS.json")["counts"]}
checks.append(("parameter counts, SVFAP-B total / trainable / adapters", "77,633,812 1,058,820 39",
               "{:,} {:,} {}".format(PC["svfap"]["total"], PC["svfap"]["trainable_rank16"], PC["svfap"]["lora_modules"])))
checks.append(("parameter counts, MAE-DFER-B total / trainable / adapters", "84,876,804 1,968,132 96",
               "{:,} {:,} {}".format(PC["maedfer"]["total"], PC["maedfer"]["trainable_rank16"], PC["maedfer"]["lora_modules"])))

CB = J("CLIP_BRANCH_SHARES.json")["shares_shorter_than_span"]
checks.append(("short-branch shares at spans 61, 136, 280, 283, 286, 301 (face cache)", "0.4 4.0 12.1 12.1 12.4 87.2",
               " ".join("%.1f" % (100 * CB[k]["cache"]) for k in ("inherited recipe, 16 at 4", "B, 16 at 9", "C, 32 at 9",
                                                                  "D, 48 at 6", "A, 16 at 19", "CMS, 16 at 20"))))

bad = 0
for name, printed, got in checks:
    ok = printed.replace("-", "−") == got.replace("-", "−")
    bad += not ok
    print("%-58s %-26s %-26s %s" % (name[:58], printed, got, "OK" if ok else "MISMATCH"))
print("\n%d values, %d mismatches" % (len(checks), bad))
sys.exit(1 if bad else 0)
