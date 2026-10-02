"""Read-only view of whatever v2a runs have finished. Does NOT write results/GATE_V2A.json -- that file is
reserved for the final verdict over all nine runs, so an interim look cannot be mistaken for the verdict.

usage: peek_v2a.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT

CFG = {16: 19, 32: 9, 48: 6}


def rd(tag):
    p = os.path.join(ROOT, "results", f"FT_{tag}_train.json")
    if not os.path.exists(p): return None
    d = json.load(open(p)); f = d["final"]
    return dict(acc=f["acc"], uar=f["uar"], f1=f["macro_f1"], pd=f["pred_dist"],
                maj=d.get("eval_majority_baseline"), span=d.get("span_frames"), ep=f.get("epoch"))


rows = {}
print(f"{'cfg':>5} {'fold':>4} {'major':>7} {'acc':>7} {'UAR':>7} {'macroF1':>8}   pred_dist")
for T in CFG:
    rr = {}
    for fold in (0, 1, 2):
        r = rd(f"v2_T{T}_f{fold}")
        if r:
            rr[fold] = r
            print(f"T{T:>4} {fold:>4} {r['maj']:>7.4f} {r['acc']:>7.4f} {r['uar']:>7.4f} {r['f1']:>8.4f}   {r['pd']}")
    if rr: rows[T] = rr

print()
for T, rr in rows.items():
    if len(rr) == 3:
        print(f"T{T} mean over 3 folds: acc {np.mean([r['acc'] for r in rr.values()]):.4f}  "
              f"UAR {np.mean([r['uar'] for r in rr.values()]):.4f}  "
              f"macro-F1 {np.mean([r['f1'] for r in rr.values()]):.4f}   (span {list(rr.values())[0]['span']})")

print()
if 16 in rows and len(rows[16]) == 3:
    c = rows[16]
    for T, rr in rows.items():
        if T == 16 or len(rr) != 3: continue
        gf = [rr[f]["f1"] - c[f]["f1"] for f in (0, 1, 2)]
        gu = [rr[f]["uar"] - c[f]["uar"] for f in (0, 1, 2)]
        ga = [rr[f]["acc"] - c[f]["acc"] for f in (0, 1, 2)]
        ok = np.mean(gf) >= 0.015 and sum(x >= 0 for x in gf) >= 2
        print(f"T{T} vs control T16/s19:")
        print(f"   macro-F1 {np.round(gf,4).tolist()} mean {np.mean(gf):+.4f}  (non-negative on {sum(x>=0 for x in gf)}/3)")
        print(f"   UAR      {np.round(gu,4).tolist()} mean {np.mean(gu):+.4f}")
        print(f"   accuracy {np.round(ga,4).tolist()} mean {np.mean(ga):+.4f}")
        print(f"   pre-registered gate (macro-F1 mean >= +0.0150 and non-negative on >= 2/3): "
              f"{'would PASS' if ok else 'would FAIL'}   [INTERIM -- T48 still outstanding]")
else:
    print("control not complete yet")

print("\n--- v1, same 16-frame budget but stride 20 (span 301 -> short branch), single seed: the C8 branch measurement ---")
for tag, fold in (("g6_ep12_ema999", 0), ("g7_svfap_f1_s0", 1), ("g7_svfap_f2_s0", 2)):
    r = rd(tag)
    if r: print(f"   fold {fold}  v1 s=20: acc {r['acc']:.4f}  UAR {r['uar']:.4f}  macro-F1 {r['f1']:.4f}")
if 16 in rows and len(rows[16]) == 3:
    v1 = {0: rd("g6_ep12_ema999"), 1: rd("g7_svfap_f1_s0"), 2: rd("g7_svfap_f2_s0")}
    if all(v1.values()):
        d = [rows[16][f]["f1"] - v1[f]["f1"] for f in (0, 1, 2)]
        da = [rows[16][f]["acc"] - v1[f]["acc"] for f in (0, 1, 2)]
        print(f"   long branch (s=19) minus short branch (s=20): macro-F1 {np.round(d,4).tolist()} mean {np.mean(d):+.4f}; "
              f"accuracy {np.round(da,4).tolist()} mean {np.mean(da):+.4f}")
        print("   caveat: seeds are not matched pairwise and the v1 runs are single-seed, so read this as a coarse estimate.")
