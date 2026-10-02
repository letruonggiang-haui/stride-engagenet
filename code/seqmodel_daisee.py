"""Landmark-sequence stream on DAiSEE (MODEL_PAPER_DESIGN amendment 14): the same two temporal architectures as
seqmodel.py, trained on the SAME three subject-level inner folds of the DAiSEE training split that ft_video.py uses
(--dataset daisee --cv-folds 3), then fused at the pre-registered w = 0.5 with the locked pixel recipe's predictions
on those folds (G11 reference runs) and with the G11 sweep winner's. Never touches the official DAiSEE splits.

Two modes, because the pixel fold predictions arrive later than the sequences:
  (default)  train both architectures on the three folds, pick the epoch from the mean curve, save per-fold
             probabilities to preds/seqd_<arch>_fold<k>.npz and the curves to results/SEQD_CV.json
  --fuse     load those, fuse with whatever pixel fold predictions exist, write the gate verdict into SEQD_CV.json

usage: seqmodel_daisee.py [--arch gru,tfm] [--epochs 40] [--smoke] | seqmodel_daisee.py --fuse
"""
import argparse, glob, json, math, os, sys, time, zipfile
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import seqmodel_defs as M

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="gru,tfm"); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--L", type=int, default=300); ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--seqdir", default=os.path.join(ROOT, "results/geom300_daisee/seq300")); ap.add_argument("--fuse", action="store_true")
a = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
K, D = M.K, M.D
OUT = os.path.join(ROOT, "results", "SEQD_CV.json")

# ---------------------------------------------------------------- folds: identical to ft_video.py --dataset daisee --cv-folds 3
d = pd.read_csv(os.path.expanduser("~/daisee/DAiSEE/Labels/TrainLabels.csv"), sep=None, engine="python"); d.columns = [c.strip() for c in d.columns]
d["key"] = d["ClipID"].astype(str).str.strip().str.replace(r"\.(avi|mp4)$", "", regex=True); d["y"] = d["Engagement"].astype(int); d["subject"] = d["key"].str[:6].astype(int)
CR = os.path.join(ROOT, "crops_face/daisee")
tr = d[d.key.apply(lambda k: zipfile.is_zipfile(os.path.join(CR, k + ".npz")) if os.path.exists(os.path.join(CR, k + ".npz")) else False)].reset_index(drop=True)
subs = np.sort(tr.subject.unique()); rng = np.random.default_rng(12345); rng.shuffle(subs); fold_of = {s: i % 3 for i, s in enumerate(subs)}
tr["fold"] = tr.subject.map(fold_of)
have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.seqdir, "*.npz"))}
n_crop = len(tr); tr = tr[tr.key.isin(have)].reset_index(drop=True)
if a.smoke: tr = tr.sample(min(len(tr), 120), random_state=0).reset_index(drop=True); a.epochs = 1
print(f"daisee train: {n_crop} clips with crops, sequences for {len(tr)}; folds {tr.fold.value_counts().sort_index().tolist()}, subjects/fold {tr.groupby('fold').subject.nunique().tolist()}", flush=True)
y_all = tr.y.values.astype(int)


def pixel_probs(pattern):
    """mean probability per clip over every preds file matching the pattern (one file per fold run)"""
    fs = sorted(glob.glob(os.path.join(ROOT, "preds", pattern)))
    if not fs: return None, []
    out = {}
    for f in fs:
        z = np.load(f, allow_pickle=True)
        for k, p in zip(z["keys"], z["P"]): out.setdefault(str(k), []).append(p)
    return {k: np.mean(v, 0) for k, v in out.items()}, [os.path.basename(f) for f in fs]


def score(y, P):
    yh = P.argmax(1); return dict(acc=accuracy_score(y, yh), uar=balanced_accuracy_score(y, yh), macro_f1=f1_score(y, yh, average="macro"), pred_dist=np.bincount(yh, minlength=K).tolist())


if a.fuse:
    # ------------------------------------------------------------ fusion with pixel fold predictions, gate verdict
    cv = json.load(open(OUT)); PIXSETS = {"locked": {0: "ft_g11_ref_locked_train.npz", 1: "ft_g11_ref_locked_f1_train.npz", 2: "ft_g11_ref_locked_f2_train.npz"}}
    sw = os.path.join(ROOT, "results", "G11_DAISEE_SWEEP.json")
    if os.path.exists(sw):
        w = json.load(open(sw))["winner"]; PIXSETS["sweep_winner"] = {0: f"ft_{w}_train.npz", 1: f"ft_{w}_f1_train.npz", 2: f"ft_{w}_f2_train.npz"}
    cv["fusion"] = {}
    for arch in cv["arch"]:
        for name, files in PIXSETS.items():
            rows = {}
            for fold in (0, 1, 2):
                sp = os.path.join(ROOT, "preds", f"seqd_{arch}_fold{fold}.npz")
                if not os.path.exists(sp): continue
                z = np.load(sp, allow_pickle=True); Ps, keys, y = z["P"], [str(k) for k in z["keys"]], z["y"]
                pix, used = pixel_probs(files[fold])
                if pix is None or not all(k in pix for k in keys): continue
                Pp = np.stack([pix[k] for k in keys]); Pf = 0.5 * Pp + 0.5 * Ps
                errs = np.stack([(Pp.argmax(1) != y).astype(float), (Ps.argmax(1) != y).astype(float)])
                rows[fold] = dict(n=int(len(y)), pixel=score(y, Pp), seq=score(y, Ps), fused=score(y, Pf), gain=float(accuracy_score(y, Pf.argmax(1)) - accuracy_score(y, Pp.argmax(1))),
                                  error_corr_pixel_vs_seq=float(np.corrcoef(errs)[0, 1]), pixel_files=used)
                print(f"[{arch} + {name}] fold {fold}: pixel {rows[fold]['pixel']['acc']:.4f}  seq {rows[fold]['seq']['acc']:.4f}  fused {rows[fold]['fused']['acc']:.4f}  gain {rows[fold]['gain']:+.4f}  err-corr {rows[fold]['error_corr_pixel_vs_seq']:.3f}", flush=True)
            if not rows: continue
            g = [r["gain"] for r in rows.values()]; fm = float(np.mean([r["fused"]["acc"] for r in rows.values()])); pm = float(np.mean([r["pixel"]["acc"] for r in rows.values()]))
            # the gate is amendment 12's, because a fused configuration is a candidate for the same single second opening
            gate = dict(folds_done=len(rows), fused_mean=fm, pixel_mean=pm, gain_mean=float(np.mean(g)), folds_nonneg=int(sum(x >= 0 for x in g)),
                        gate_pass=bool(len(rows) == 3 and fm >= 0.60 and (fm - pm) >= 0.03))
            print(f"[{arch} + {name}] GATE ({len(rows)}/3 folds): fused mean {fm:.4f} vs pixel {pm:.4f}, gain {fm-pm:+.4f} -> {'PASS' if gate['gate_pass'] else 'FAIL'} (needs 3 folds, fused >= 0.60 and gain >= +0.03)", flush=True)
            cv["fusion"][f"{arch}+{name}"] = dict(folds=rows, gate=gate)
    json.dump(cv, open(OUT, "w"), indent=1, default=float); print("SEQD_FUSE_DONE", flush=True); sys.exit(0)


# ---------------------------------------------------------------- sequences
def load_all(keys):
    X = np.zeros((len(keys), a.L, D), np.float32); Mk = np.zeros((len(keys), a.L), bool)
    for i, k in enumerate(keys):
        z = np.load(os.path.join(a.seqdir, k + ".npz")); s = z["seq"]; n = min(len(s), a.L)
        if len(s) > a.L: s = s[np.linspace(0, len(s) - 1, a.L).astype(int)]; n = a.L
        X[i, :n] = s[:n]; Mk[i, :n] = True
    return X, Mk


X_all, M_all = load_all(tr.key.values)
print(f"loaded {X_all.shape}, mean valid frames {M_all.sum(1).mean():.0f}", flush=True)


def run_fold(arch, fold, mu, sd):
    trn, ev = np.where(tr.fold.values != fold)[0], np.where(tr.fold.values == fold)[0]
    if len(trn) == 0 or len(ev) == 0: return None
    Xt = torch.tensor((X_all - mu) / sd); Mt = torch.tensor(M_all); yt = torch.tensor(y_all)
    model = (M.GRUNet(a.L) if arch == "gru" else M.TfmNet(a.L)).to(DEV); ema = {n: p.detach().clone() for n, p in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if arch == "gru" else 5e-4, weight_decay=0.05)
    steps = math.ceil(len(trn) / a.batch) * a.epochs; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=max(steps, 1), pct_start=0.1)
    curve, P_by_epoch = [], []
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(trn)
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]; x, m = M.augment(Xt[idx].to(DEV), Mt[idx].to(DEV)); y = yt[idx].to(DEV)
            loss = F.cross_entropy(model(x, m), y, label_smoothing=0.1); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            with torch.no_grad():
                for n, p in model.state_dict().items():
                    if p.dtype.is_floating_point: ema[n].mul_(a.ema).add_(p, alpha=1 - a.ema)
        live = {n: p.detach().clone() for n, p in model.state_dict().items()}; model.load_state_dict(ema); model.eval(); P = []
        with torch.no_grad():
            for s in range(0, len(ev), 256):
                idx = ev[s:s + 256]; P.append(torch.softmax(model(Xt[idx].to(DEV), Mt[idx].to(DEV)), 1).cpu().numpy())
        model.load_state_dict(live); P = np.concatenate(P); P_by_epoch.append(P)
        curve.append(dict(epoch=ep, acc=accuracy_score(y_all[ev], P.argmax(1)), uar=balanced_accuracy_score(y_all[ev], P.argmax(1)), macro_f1=f1_score(y_all[ev], P.argmax(1), average="macro")))
    return dict(ev=ev, curve=curve, P_by_epoch=P_by_epoch)


out = {"arch": {}, "folds_majority": {int(f): float((tr[tr.fold == f].y == tr[tr.fold == f].y.mode()[0]).mean()) for f in (0, 1, 2)}}
for arch in a.arch.split(","):
    t0 = time.time(); res = {}
    for fold in (0, 1, 2):
        trn = tr.fold.values != fold; mu = X_all[trn][M_all[trn]].mean(0); sd = X_all[trn][M_all[trn]].std(0) + 1e-6
        r = run_fold(arch, fold, mu, sd)
        if r: res[fold] = r; print(f"[{arch}] fold {fold}: final-epoch acc {r['curve'][-1]['acc']:.4f}, best-epoch acc {max(c['acc'] for c in r['curve']):.4f} ({time.time()-t0:.0f}s)", flush=True)
    if not res: continue
    E = min(len(r["curve"]) for r in res.values()); mean_curve = np.mean([[c["acc"] for c in r["curve"][:E]] for r in res.values()], 0); e_star = int(mean_curve.argmax())
    summary = {"epoch_star": e_star, "mean_curve_acc": mean_curve.tolist(), "folds": {}}
    for fold, r in res.items():
        ev = r["ev"]; keys = tr.key.values[ev]; y = y_all[ev]; Ps = r["P_by_epoch"][e_star]
        summary["folds"][fold] = dict(n=int(len(y)), majority=out["folds_majority"][fold], seq=score(y, Ps))
        np.savez(os.path.join(ROOT, "preds", f"seqd_{arch}_fold{fold}.npz"), P=Ps, keys=keys, y=y)
        print(f"[{arch}] fold {fold} @epoch {e_star}: seq acc {summary['folds'][fold]['seq']['acc']:.4f} (majority {out['folds_majority'][fold]:.4f}) uar {summary['folds'][fold]['seq']['uar']:.4f} pred {summary['folds'][fold]['seq']['pred_dist']}", flush=True)
    summary["seq_mean"] = float(np.mean([f["seq"]["acc"] for f in summary["folds"].values()]))
    out["arch"][arch] = summary
json.dump(out, open(OUT, "w"), indent=1, default=float); print("SEQD_CV_DONE", flush=True)
