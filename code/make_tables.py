"""Collect every results JSON into the tables the manuscript needs.

Reads results/{E0,E1,RQ12_*,RQ3_*,NEGCTRL_*,E7_*,POWER_*}.json and writes
results/TABLES.md (markdown, IEEE table order) and results/SUMMARY.json.
Numbers are printed exactly as computed; nothing is rounded away silently and
nothing is filled in where a file is missing.
"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/engagenet_p2_crossdomain")
R = os.path.join(ROOT, "results")
load = lambda p: json.load(open(p)) if os.path.exists(p) else None
MISS = "[CẦN BỔ SUNG]"


def fmt(m, ci=None, k="qwk", nd=3):
    if m is None:
        return MISS
    v = m.get(k) if isinstance(m, dict) else m
    if v is None:
        return MISS
    s = f"{v:.{nd}f}"
    if ci and k in ci:
        s += f" [{ci[k][0]:.{nd}f}, {ci[k][1]:.{nd}f}]"
    return s


def triple(node, dec="argmax"):
    """macro-F1, QWK, MAE with intervals from an evaluate() record"""
    if not node or dec not in node:
        return [MISS] * 3
    d = node[dec]; ci = d.get("ci", {})
    return [fmt(d, ci, "macro_f1"), fmt(d, ci, "qwk"), fmt(d, ci, "mae")]


out = ["# Result tables (auto-generated)", ""]
S = {}

e1 = load(os.path.join(R, "E1_backbone.json"))
if e1:
    out += ["## Backbone selection (EngageNet validation only)", "",
            "| Encoder | val macro-F1 (mean of 3 seeds) | trainable params | dim |", "|---|---|---|---|"]
    for enc, d in e1["encoders"].items():
        out.append(f"| {enc} | {d['mean_val_macro_f1']:.4f} | {d['runs'][0]['trainable_params']:,} | {d['dim']} |")
    out += ["", f"Chosen: **{e1['chosen']}**, decided {e1['date'][:10]} on {e1['criterion']}.", ""]
    S["backbone"] = e1["chosen"]

e0 = load(os.path.join(R, "E0/E0_report.json")); e0g = load(os.path.join(R, "E0/E0_groups_final.json"))
if e0:
    S["deposit"] = dict(files=e0["n_clips"], duplicate_sets=e0["exact_duplicate_groups"], duplicate_files=e0["exact_duplicate_files"],
                        joins=e0["n_joins"], chains_ge5=e0["n_chains_ge5"], share_in_chains_ge5=e0["share_clips_in_chains_ge5"])
    out += ["## Deposit audit (OUC-CGE)", "",
            f"- files {e0['n_clips']}, duplicate sets {e0['exact_duplicate_groups']} covering {e0['exact_duplicate_files']} files, none spanning two levels",
            f"- boundary-frame links {e0['n_joins']}, chains of five or more {e0['n_chains_ge5']}, share of clips in such chains {e0['share_clips_in_chains_ge5']:.3f}",
            f"- label purity within those chains {e0['chain_label_purity_quantiles']['0.0']:.2f} (minimum over chains)", ""]
    if e0g:
        out += ["| Grouping | groups | weighted label purity | groups with all three levels |", "|---|---|---|---|"]
        for k, d in e0g.items():
            out.append(f"| {k} | {len(d['sizes'])} | {d['weighted_label_purity']:.3f} | {d['groups_containing_all_three_labels']} |")
        out.append("")

for f in sorted(glob.glob(os.path.join(R, "RQ12_*.json"))):
    rq = load(f); tag = os.path.basename(f)[5:-5]
    t = rq.get("target", {})
    out += [f"## RQ1 and RQ2 — {tag}", "",
            f"Clips {t.get('n_clips', MISS)}, groups {t.get('n_groups', MISS)}, label counts {t.get('label_counts', MISS)}, "
            f"persons per clip {t.get('persons_per_clip_mean', 0):.2f}, face-detection failure rate {t.get('face_failure_rate', 0):.3f}.", "",
            "**TABLE II. ANCHORS AND ZERO-SHOT TRANSFER**", "",
            "| Arm | macro-F1 [95% CI] | QWK [95% CI] | MAE [95% CI] | predicted class counts |", "|---|---|---|---|---|"]
    order = ["ceiling_group", "ceiling_clip", "floor_majority"] + sorted(k for k in rq if k.startswith("zeroshot/"))
    for k in order:
        if k in rq:
            a, b, c = triple(rq[k]); out.append(f"| {k} | {a} | {b} | {c} | {rq[k]['argmax'].get('pred_dist')} |")
    if "floor_label_prior" in rq:
        d = rq["floor_label_prior"]; out.append(f"| floor_label_prior (expectation over draws) | {d['macro_f1']:.3f} | {d['qwk']:.3f} | {d['mae']:.3f} | - |")
    out += ["", "**TABLE III. ADAPTATION FAMILIES WITHIN ACCESS REGIME**", "",
            "| Access regime | Method | macro-F1 [95% CI] | QWK [95% CI] | MAE [95% CI] | gap closure (QWK) |", "|---|---|---|---|---|---|"]
    zs = rq.get("zeroshot/f1_softmax/mean", {}).get("argmax", {}).get("qwk")
    ce = rq.get("ceiling_group", {}).get("argmax", {}).get("qwk")
    def closure(k):
        v = rq.get(k, {}).get("argmax", {}).get("qwk")
        if v is None or zs is None or ce is None or abs(ce - zs) < 1e-9:
            return MISS
        return f"{(v - zs) / (ce - zs):.3f}"
    rows = [("Zero-target", "zeroshot/f1_softmax/mean"), ("Zero-target", "zeroshot/f1_coral/mean"), ("Zero-target", "zeroshot/f2_ldl_table/mean"),
            ("Unlabelled-target", "dann"), ("Unlabelled-target", "dann_bbse"),
            ("Inference-time target", "tent"), ("Inference-time target", "tent_continual"),
            ("Few-shot (1 cluster)", "fewshot1/f1_softmax"), ("Few-shot (3 clusters)", "fewshot3/f1_softmax"),
            ("Few-shot (1 cluster)", "fewshot1/f2_ldl_table"), ("Few-shot (3 clusters)", "fewshot3/f2_ldl_table")]
    for reg, k in rows:
        if k in rq:
            a, b, c = triple(rq[k]); out.append(f"| {reg} | {k} | {a} | {b} | {c} | {closure(k)} |")
    sens = {k: rq[k]["argmax"]["qwk"] for k in rq if k.startswith("zeroshot/") and isinstance(rq[k], dict) and "argmax" in rq[k]}
    if sens:
        out += ["", "Aggregation operator and LDL sensitivity (QWK, arg max decoding): " + ", ".join(f"{k.split('zeroshot/')[1]} {v:.3f}" for k, v in sorted(sens.items())), ""]
    if "reverse_ouc_to_engagenet_val" in rq:
        d = rq["reverse_ouc_to_engagenet_val"]
        out += ["", f"Reverse direction, OUC-CGE to EngageNet validation (n = 1 case): macro-F1 {d['macro_f1']:.3f}, QWK {d['qwk']:.3f}, MAE {d['mae']:.3f}, predicted classes {d['pred_dist']}.", ""]
    S[f"RQ12_{tag}"] = {k: rq[k]["argmax"]["qwk"] for k in rq if isinstance(rq.get(k), dict) and "argmax" in rq[k]}

for f in sorted(glob.glob(os.path.join(R, "RQ3_*.json"))):
    rq = load(f)
    out += [f"## RQ3 — {os.path.basename(f)[4:-5]}", "",
            f"Evaluated on {rq['n_eval']} EngageNet clips with all four views present ({rq['dropped_eval_missing_view']} dropped).", "",
            "| Reading | chance-corrected balanced accuracy [95% CI] |", "|---|---|"]
    for k in sorted(k for k in rq if "/" in k):
        out.append(f"| {k} | {rq[k]['cc_bacc']:.3f} [{rq[k]['ci'][0]:.3f}, {rq[k]['ci'][1]:.3f}] |")
    out += ["", "**TABLE IV. GAP DECOMPOSITION (source domain, common unit)**", "", "| Component | estimate [95% CI] |", "|---|---|"]
    for k, d in rq["components"].items():
        out.append(f"| {k} | {d['estimate']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] |")
    out.append("")
    S["RQ3"] = {k: d["estimate"] for k, d in rq["components"].items()}

for f in sorted(glob.glob(os.path.join(R, "NEGCTRL_*.json"))):
    nc = load(f)
    out += [f"## Negative control — {os.path.basename(f)[8:-5]}", "", "| Arm | macro-F1 | QWK | MAE | predicted | true |", "|---|---|---|---|---|---|"]
    for k, d in nc.items():
        if isinstance(d, dict) and "qwk" in d:
            out.append(f"| {k} | {d['macro_f1']:.3f} | {d['qwk']:.3f} | {d['mae']:.3f} | {d.get('pred_dist')} | {d.get('true_dist')} |")
    out.append("")
    S["negctrl"] = {k: d["qwk"] for k, d in nc.items() if isinstance(d, dict) and "qwk" in d}

e7 = load(os.path.join(R, "E7_labels_val.json")); e7p = load(os.path.join(R, "E7_preds_val_s0.json"))
if e7:
    out += ["## What the aggregation step costs (pseudo-groups on EngageNet)", "",
            f"Split {e7['split']}, {e7['n_clips']} clips, {e7['n_subjects']} subjects, U3 prior {e7['prior_u3']}.", "",
            "| tau | operator | operator on human labels vs counting rule (QWK) | model vs counting rule | model vs operator on human labels |", "|---|---|---|---|---|"]
    for tau in ("0.50", "0.60", "0.67"):
        for op in ("mean-argmax", "mean-expect", "majority-argmax", "median-argmax"):
            a = e7.get(f"seed0_tau{tau}_{op}_human_vs_rule", {}).get("qwk")
            b = (e7p or {}).get(f"seed0_tau{tau}_{op}_pred_vs_rule", {}).get("qwk")
            c = (e7p or {}).get(f"seed0_tau{tau}_{op}_pred_vs_humanop", {}).get("qwk")
            out.append(f"| {tau} | {op} | {fmt(a)} | {fmt(b)} | {fmt(c)} |")
    out.append("")

for f in sorted(glob.glob(os.path.join(R, "POWER_qwk_*.json"))):
    pw = load(f)
    out += [f"## Resolvable QWK difference — {os.path.basename(f)[10:-5]}", "",
            f"{pw['n_groups']} groups, {pw['n_clips']} clips, {pw['nsim']} simulations per cell, {pw['bootstrap']} bootstrap resamples.", "",
            "| between-group spread | arm-A QWK | smallest resolvable QWK difference at 80% power | type-I rate at zero difference |", "|---|---|---|---|"]
    for k, d in pw["mde"].items():
        v = d["min_dqwk_at_power80"]
        out.append(f"| {k} | {d['qwk_a']:.3f} | {v:.3f} |" if v is not None else f"| {k} | {d['qwk_a']:.3f} | not reached in this grid |")
        out[-1] += f" {d['type1_rate']:.3f} |"
    out.append("")

open(os.path.join(R, "TABLES.md"), "w", encoding="utf-8").write("\n".join(out))
json.dump(S, open(os.path.join(R, "SUMMARY.json"), "w"), indent=1, default=float)
print("\n".join(out))
