"""Late fusion of the LoRA-adapted person-box model with the frozen multi-encoder model.
Both are out-of-fold over the same grouped folds; clips are aligned by key and the class
probabilities averaged with equal weight (no weight is tuned on test data)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/engagenet_p2_crossdomain/code"))
from p2x_lib import ROOT, metrics, cluster_bootstrap
from sklearn.metrics import cohen_kappa_score

def within(y, yh, g):
    ks, ws = [], []
    for c in np.unique(g):
        m = g == c
        if len(np.unique(y[m])) < 2: continue
        ks.append(cohen_kappa_score(y[m], yh[m], labels=[0, 1, 2], weights="quadratic")); ws.append(m.sum())
    return float(np.average(ks, weights=ws))

G = sys.argv[1] if len(sys.argv) > 1 else "rec_group"
L = np.load(os.path.join(ROOT, f"preds/lora_{G}_full_lora.npz"), allow_pickle=True)
Fz = np.load(os.path.join(ROOT, f"preds/lora_{G}_full_frozen.npz"), allow_pickle=True) if os.path.exists(os.path.join(ROOT, f"preds/lora_{G}_full_frozen.npz")) else None
res = {}
def report(name, y, P, g):
    yh = P.argmax(1)
    r = dict(**metrics(y, yh, 3), ci=cluster_bootstrap(y, yh, g, 3, B=1000), qwk_within=within(y, yh, g), n=int(len(y)),
             pred_dist=np.bincount(yh, minlength=3).tolist())
    res[name] = r
    print(f"{name:34s} n={len(y)} qwk {r['qwk']:.3f} [{r['ci']['qwk'][0]:.3f},{r['ci']['qwk'][1]:.3f}] f1 {r['macro_f1']:.3f} acc {r['acc']:.3f} within {r['qwk_within']:.3f}")

ok = ~np.isnan(L["P"][:, 0])
report("lora_videomae_person", L["y"][ok], L["P"][ok], L["groups"][ok])
if Fz is not None:
    okf = ~np.isnan(Fz["P"][:, 0]); report("frozen_videomae_person (control)", Fz["y"][okf], Fz["P"][okf], Fz["groups"][okf])
kl = {k: i for i, k in enumerate(L["keys"])}
for arm in ("marlin", "all"):
    f = os.path.join(ROOT, f"preds/multienc_{G}_{arm}.npz")
    if not os.path.exists(f): continue
    M = np.load(f, allow_pickle=True)
    if "keys" not in M.files: print("no keys in", f); continue
    idx = [(i, kl[k]) for i, k in enumerate(M["keys"]) if k in kl and ok[kl[k]]]
    im = np.array([i for i, _ in idx]); il = np.array([j for _, j in idx])
    y, g = M["y"][im], M["groups"][im]
    report(f"frozen_{arm} (same clips)", y, M["P"][im], g)
    report(f"lora (same clips)", y, L["P"][il], g)
    report(f"FUSION lora + frozen_{arm}", y, (M["P"][im] + L["P"][il]) / 2, g)
    # paired difference: fusion vs frozen multi-encoder
    d = cluster_bootstrap(y, ((M["P"][im] + L["P"][il]) / 2).argmax(1), g, 3, B=1000, yhat_b=M["P"][im].argmax(1))
    res[f"paired_fusion_minus_frozen_{arm}"] = d; print("   paired dQWK fusion - frozen:", [round(v, 3) for v in d["qwk"]])
json.dump(res, open(os.path.join(ROOT, f"results/FUSION_{G}.json"), "w"), indent=1, default=float)
print("FUSE_DONE")
