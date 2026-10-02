"""SECOND official opening for the landmark-sequence stream -- ONLY if the pre-registered gate in results/SEQ_CV.json
passed (MODEL_PAPER_DESIGN amendment 10). This script refuses to run otherwise, so the gate is enforced mechanically,
not by intention.

Trains the winning temporal architecture at the CV-selected epoch count on (a) train -> scores val once and test once,
(b) train+val -> scores test once; fuses each with the already-saved G8 pixel-ensemble predictions of the same split
at the pre-registered w = 0.5. Clips without a landmark sequence (no face found in any frame) fall back to the pixel
probabilities alone, and their count is reported. Writes results/G10_SEQ_OFFICIAL.json.

usage: seq_official.py --gate-passed [--arch gru|tfm]   (arch defaults to the one that passed with the larger gain)
"""
import argparse, glob, json, math, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

ap = argparse.ArgumentParser(); ap.add_argument("--gate-passed", action="store_true"); ap.add_argument("--arch", default="")
ap.add_argument("--seqdir", default=os.path.join(ROOT, "results/geom300/seq300")); ap.add_argument("--L", type=int, default=300); ap.add_argument("--batch", type=int, default=64)
a = ap.parse_args()
cv = json.load(open(os.path.join(ROOT, "results/SEQ_CV.json")))
passed = {k: v for k, v in cv["arch"].items() if v.get("gate", {}).get("gate2_pass")}
if not a.gate_passed or not passed:
    sys.exit(f"refusing: --gate-passed={a.gate_passed}, architectures that passed the pre-registered gate: {list(passed) or 'none'}. The official splits stay closed.")
arch = a.arch or max(passed, key=lambda k: passed[k]["gate"]["fused_gain_mean"]); e_star = cv["arch"][arch]["epoch_star"]; EPOCHS = e_star + 1
print(f"gate passed for {list(passed)}; running {arch} for {EPOCHS} epochs (CV-selected epoch {e_star})", flush=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0); K, D = 4, 223

import seqmodel_defs as M   # architectures and augmentation shared with seqmodel.py

en = engagenet_labels()
def load(keys):
    X = np.zeros((len(keys), a.L, D), np.float32); Mk = np.zeros((len(keys), a.L), bool); has = np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        p = os.path.join(a.seqdir, k + ".npz")
        if not os.path.exists(p): continue
        s = np.load(p)["seq"]; n = min(len(s), a.L)
        if len(s) > a.L: s = s[np.linspace(0, len(s) - 1, a.L).astype(int)]; n = a.L
        X[i, :n] = s[:n]; Mk[i, :n] = True; has[i] = True
    return X, Mk, has

def train_predict(train_splits, eval_split):
    trd = en[en.split.isin(train_splits)].reset_index(drop=True); evd = en[en.split == eval_split].reset_index(drop=True)
    Xt, Mt, ht = load(trd.key.values); Xe, Me, he = load(evd.key.values); yt = trd.y4.values.astype(int)
    Xt, Mt, yt = Xt[ht], Mt[ht], yt[ht]; mu = Xt[Mt].mean(0); sd = Xt[Mt].std(0) + 1e-6
    Xt = torch.tensor((Xt - mu) / sd); Mt = torch.tensor(Mt); Xe = torch.tensor((Xe - mu) / sd); Me = torch.tensor(Me); ytt = torch.tensor(yt)
    model = (M.GRUNet(a.L) if arch == "gru" else M.TfmNet(a.L)).to(DEV); ema = {n: p.detach().clone() for n, p in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if arch == "gru" else 5e-4, weight_decay=0.05)
    steps = math.ceil(len(yt) / a.batch) * EPOCHS; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=max(steps, 1), pct_start=0.1)
    for ep in range(EPOCHS):
        model.train(); perm = np.random.permutation(len(yt))
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]; x, m = M.augment(Xt[idx].to(DEV), Mt[idx].to(DEV)); y = ytt[idx].to(DEV)
            loss = F.cross_entropy(model(x, m), y, label_smoothing=0.1); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            with torch.no_grad():
                for n, p in model.state_dict().items():
                    if p.dtype.is_floating_point: ema[n].mul_(0.999).add_(p, alpha=0.001)
    model.load_state_dict(ema); model.eval(); P = np.full((len(evd), K), np.nan, np.float32)
    with torch.no_grad():
        for s in range(0, len(evd), 256):
            idx = np.arange(s, min(s + 256, len(evd))); P[idx] = torch.softmax(model(Xe[idx].to(DEV), Me[idx].to(DEV)), 1).cpu().numpy()
    P[~he] = np.nan
    return evd, P, int((~he).sum())

def pixel(pattern):
    fs = sorted(glob.glob(os.path.join(ROOT, "preds", pattern))); zs = [np.load(f, allow_pickle=True) for f in fs]
    keys = zs[0]["keys"]
    for z in zs: assert (z["keys"] == keys).all()
    return dict(zip(keys, np.mean([z["P"] for z in zs], 0))), len(zs)

def score(evd, Pseq, pix, tag):
    y = evd.y4.values.astype(int); subj = evd.subject.values; Pp = np.stack([pix[k] for k in evd.key])
    miss = np.isnan(Pseq[:, 0]); Pf = 0.5 * Pp + 0.5 * np.where(miss[:, None], Pp, Pseq); Ps = np.where(miss[:, None], Pp, Pseq)
    us = np.unique(subj); idx = {s: np.where(subj == s)[0] for s in us}; rng = np.random.default_rng(0)
    def sc(P):
        yh = P.argmax(1); b = [accuracy_score(y[i], yh[i]) for i in (np.concatenate([idx[s] for s in rng.choice(us, len(us), replace=True)]) for _ in range(2000))]
        return dict(acc=accuracy_score(y, yh), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], uar=balanced_accuracy_score(y, yh), macro_f1=f1_score(y, yh, average="macro"), pred_dist=np.bincount(yh, minlength=K).tolist())
    r = dict(n=int(len(y)), subjects=int(len(us)), clips_without_sequence=int(miss.sum()), PIXEL_ENSEMBLE=sc(Pp), SEQ_alone=sc(Ps), FUSED_w05=sc(Pf))
    print(f"\n=== {tag}: n {r['n']} subjects {r['subjects']} (no-sequence fallback on {r['clips_without_sequence']})")
    for k in ("PIXEL_ENSEMBLE", "SEQ_alone", "FUSED_w05"): v = r[k]; print(f"   {k:16s} acc {v['acc']:.4f}  CI [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}]  uar {v['uar']:.4f}  macroF1 {v['macro_f1']:.4f}  {v['pred_dist']}", flush=True)
    return r

out = {"arch": arch, "epochs": EPOCHS}
evd, P, _ = train_predict(["train"], "val"); pix, n = pixel("ft_g8_*_s?_train.npz"); out["VAL_train_only"] = score(evd, P, pix, "VAL_train_only")
evd, P, _ = train_predict(["train"], "test"); pix, n = pixel("ft_g8_*_trainonly_on_test_train.npz"); out["TEST_train_only"] = score(evd, P, pix, "TEST_train_only")
evd, P, _ = train_predict(["train", "val"], "test"); pix, n = pixel("ft_g8_*_s?_train-val.npz"); out["TEST_train_plus_val"] = score(evd, P, pix, "TEST_train_plus_val")
json.dump(out, open(os.path.join(ROOT, "results/G10_SEQ_OFFICIAL.json"), "w"), indent=1, default=float); print("\nG10_SEQ_OFFICIAL_DONE")
