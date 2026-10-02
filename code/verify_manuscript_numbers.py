"""Level-3 formula audit: recompute every experimental number quoted in the manuscript
from the results JSON files and print paper value / recomputed value / verdict.

usage: verify_manuscript_numbers.py <manuscript.md> <results dir>
"""
import json, os, re, sys

MS, R = sys.argv[1], sys.argv[2]
text = open(MS, encoding="utf-8").read()
load = lambda n: json.load(open(os.path.join(R, n), encoding="utf-8"))
rq = load("RQ12_marlin_face_1080p_rec_group.json")
rq8 = load("RQ12_marlin_face_1080p_vis8.json")
rq63 = load("RQ12_marlin_face_1080p_vis63.json")
rqall = load("RQ12_marlin_face_all_rec_group.json")
rq3 = load("RQ3_marlin_val.json")
nc = load("NEGCTRL_marlin_val.json")
e1 = load("E1_backbone.json")
e7 = load("E7_labels_val.json")
dg = load("DIAG_shift.json")
e0 = load(os.path.join("E0", "E0_report.json"))

arm = lambda d, k, m="qwk", dec="argmax": d[k][dec][m]
ci = lambda d, k, m="qwk", dec="argmax": d[k][dec]["ci"][m]

checks = [
    ("ceiling grouped QWK", 0.401, arm(rq, "ceiling_group")),
    ("ceiling grouped CI low", 0.244, ci(rq, "ceiling_group")[0]),
    ("ceiling grouped CI high", 0.541, ci(rq, "ceiling_group")[1]),
    ("ceiling grouped macro-F1", 0.545, arm(rq, "ceiling_group", "macro_f1")),
    ("ceiling clip QWK", 0.843, arm(rq, "ceiling_clip")),
    ("ceiling clip macro-F1", 0.882, arm(rq, "ceiling_clip", "macro_f1")),
    ("ceiling vis8 QWK", 0.252, arm(rq8, "ceiling_group")),
    ("ceiling vis63 QWK", 0.508, arm(rq63, "ceiling_group")),
    ("ceiling all-clips grouped QWK", 0.342, arm(rqall, "ceiling_group")),
    ("ceiling all-clips clip QWK", 0.871, arm(rqall, "ceiling_clip")),
    ("zero-shot QWK", 0.000, arm(rq, "zeroshot/f1_softmax/mean")),
    ("zero-shot macro-F1", 0.178, arm(rq, "zeroshot/f1_softmax/mean", "macro_f1")),
    ("majority floor macro-F1", 0.178, arm(rq, "floor_majority", "macro_f1")),
    ("label-prior floor macro-F1", 0.326, rq["floor_label_prior"]["macro_f1"]),
    ("label-prior floor QWK", 0.001, rq["floor_label_prior"]["qwk"]),
    ("dann QWK", -0.010, arm(rq, "dann")),
    ("dann macro-F1", 0.224, arm(rq, "dann", "macro_f1")),
    ("dann_bbse QWK", 0.000, arm(rq, "dann_bbse")),
    ("tent QWK", 0.000, arm(rq, "tent")),
    ("fewshot3 LDL QWK", 0.154, arm(rq, "fewshot3/f2_ldl_table")),
    ("fewshot3 LDL CI low", 0.024, ci(rq, "fewshot3/f2_ldl_table")[0]),
    ("fewshot3 adapter QWK", 0.131, arm(rq, "fewshot3/f1_softmax")),
    ("fewshot1 adapter QWK", -0.024, arm(rq, "fewshot1/f1_softmax")),
    ("fewshot3 vis8 adapter", 0.247, arm(rq8, "fewshot3/f1_softmax")),
    ("fewshot3 vis8 LDL", 0.263, arm(rq8, "fewshot3/f2_ldl_table")),
    ("fewshot3 vis63 adapter", -0.033, arm(rq63, "fewshot3/f1_softmax")),
    ("reverse QWK", 0.084, rq["reverse_ouc_to_engagenet_val"]["qwk"]),
    ("reverse macro-F1", 0.310, rq["reverse_ouc_to_engagenet_val"]["macro_f1"]),
    ("target clips", 6710, rq["target"]["n_clips"]),
    ("target clusters", 21, rq["target"]["n_groups"]),
    ("all-clips set size", 7525, rqall["target"]["n_clips"]),
    ("face failure rate", 0.187, rq["target"]["face_failure_rate"]),
    ("E1 marlin", 0.630, e1["encoders"]["marlin"]["mean_val_macro_f1"]),
    ("E1 videomae", 0.610, e1["encoders"]["videomae"]["mean_val_macro_f1"]),
    ("RQ3 face cc", 0.422, rq3["U3/ceiling/face"]["cc_bacc"]),
    ("RQ3 person mismatch", 0.267, rq3["U3/mismatch/person"]["cc_bacc"]),
    ("RQ3 person ceiling", 0.296, rq3["U3/ceiling/person"]["cc_bacc"]),
    ("RQ3 full mismatch", 0.054, rq3["U3/mismatch/full"]["cc_bacc"]),
    ("RQ3 full ceiling", 0.285, rq3["U3/ceiling/full"]["cc_bacc"]),
    ("RQ3 face_lr mismatch", 0.257, rq3["U3/mismatch/face_lr"]["cc_bacc"]),
    ("RQ3 face_lr ceiling", 0.370, rq3["U3/ceiling/face_lr"]["cc_bacc"]),
    ("RQ3 binary", 0.543, rq3["binary/ceiling/face"]["cc_bacc"]),
    ("component resolution", 0.164, rq3["components"]["resolution_mismatch"]["estimate"]),
    ("component framing person", 0.154, rq3["components"]["framing_person_mismatch"]["estimate"]),
    ("component framing full", 0.368, rq3["components"]["framing_full_mismatch"]["estimate"]),
    ("component label U4-U3", -0.020, rq3["components"]["label_U4_minus_U3"]["estimate"]),
    ("component label U3-binary", -0.121, rq3["components"]["label_U3_minus_binary"]["estimate"]),
    ("resolution + framing sum", 0.319, rq3["components"]["resolution_mismatch"]["estimate"] + rq3["components"]["framing_person_mismatch"]["estimate"]),
    ("residual 0.422 - 0.319", 0.103, rq3["U3/ceiling/face"]["cc_bacc"] - (rq3["components"]["resolution_mismatch"]["estimate"] + rq3["components"]["framing_person_mismatch"]["estimate"])),
    ("negctrl EngageNet in-domain", 0.691, nc["ceiling/EngageNet->EngageNet"]["qwk"]),
    ("negctrl DAiSEE in-domain", 0.123, nc["ceiling/DAiSEE->DAiSEE"]["qwk"]),
    ("negctrl EngageNet->DAiSEE", 0.116, nc["zeroshot/EngageNet->DAiSEE"]["qwk"]),
    ("negctrl DAiSEE->EngageNet", 0.306, nc["zeroshot/DAiSEE->EngageNet"]["qwk"]),
    ("E7 mean tau .5", 0.634, e7["seed0_tau0.50_mean-argmax_human_vs_rule"]["qwk"]),
    ("E7 median tau .5", 0.817, e7["seed0_tau0.50_median-argmax_human_vs_rule"]["qwk"]),
    ("E7 mean tau .67", 0.183, e7["seed0_tau0.67_mean-argmax_human_vs_rule"]["qwk"]),
    ("diag target persons", 37038, dg["n_target_persons"]),
    ("diag persons predicted Low", 37032, dg["target_persons_pred_hist"][0]),
    ("diag mean maxprob target", 0.999, dg["target_persons_mean_maxprob"]),
    ("diag mean maxprob source", 0.739, dg["source_val_mean_maxprob"]),
    ("diag domain AUC", 1.000, dg["domain_classifier_auc"]),
    ("deposit files", 7706, e0["n_clips"]),
    ("duplicate sets", 128, e0["exact_duplicate_groups"]),
    ("duplicate files", 299, e0["exact_duplicate_files"]),
]

bad = 0
for name, paper, computed in checks:
    tol = 0.5 if isinstance(paper, int) else (0.0006 if abs(paper) < 10 else 0.5)
    ok = abs(paper - computed) <= tol
    bad += not ok
    print(f"{'OK ' if ok else 'DIFF'} {name:34s} paper {paper!s:>8} computed {computed:.4f}" if isinstance(computed, float)
          else f"{'OK ' if ok else 'DIFF'} {name:34s} paper {paper!s:>8} computed {computed}")

# numbers quoted in the text that must also appear in the tables
for pat in (r"6,710", r"0\.401", r"0\.843", r"0\.154", r"37,032", r"1\.000", r"0\.116", r"0\.123"):
    print(("OK  quoted in text: " if re.search(pat, text) else "MISSING in text: ") + pat)
print("\nmismatches:", bad)
