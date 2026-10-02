"""Landmark-sequence stream (MODEL_PAPER_DESIGN amendment 10): a temporal model on per-frame facial geometry over
the full 300-frame clip, evaluated on the SAME three inner CV folds as the pixel campaign, then fused with the
pixel ensemble at the pre-registered w = 0.5. Never touches the official splits.

Two architectures, chosen by the mean accuracy curve across the three folds (no per-run epoch cherry-picking):
  gru   two-layer bidirectional GRU, hidden 256, attention pooling
  tfm   four-layer temporal transformer, d 256, 4 heads, learned positions, masked mean pooling
Input per frame: 204 shape-normalised landmark coordinates + 19 scalars (EAR, MAR, brow, pose ...) = 223.
Writes results/SEQ_CV.json with the pre-registered gate verdict.

usage: seqmodel.py [--arch gru,tfm] [--epochs 40] [--smoke]
"""
import argparse, glob, json, math, os, sys, time, zipfile
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="gru,tfm"); ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--L", type=int, default=300); ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--seqdir", default=os.path.join(ROOT, "results/geom300/seq300"))
a = ap.parse_args()
DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
K, D = 4, 223

# ---------------------------------------------------------------- folds: identical to ft_video.py
en = engagenet_labels(); tr = en[en.split == "train"].reset_index(drop=True)
CR = os.path.join(ROOT, "crops_face/engagenet")
tr = tr[tr.key.apply(lambda k: zipfile.is_zipfile(os.path.join(CR, k + ".npz")) if os.path.exists(os.path.join(CR, k + ".npz")) else False)].reset_index(drop=True)
subs = np.sort(tr.subject.unique()); rng = np.random.default_rng(12345); rng.shuffle(subs); fold_of = {s: i % 3 for i, s in enumerate(subs)}
tr["fold"] = tr.subject.map(fold_of)
have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.seqdir, "*.npz"))}
tr = tr[tr.key.isin(have)].reset_index(drop=True)
if a.smoke: tr = tr.sample(min(len(tr), 120), random_state=0).reset_index(drop=True); a.epochs = 1
print(f"sequences available for {len(tr)} of the cached train clips; folds {tr.fold.value_counts().sort_index().tolist()}", flush=True)


def load_all(keys):
    X = np.zeros((len(keys), a.L, D), np.float32); M = np.zeros((len(keys), a.L), bool)
    for i, k in enumerate(keys):
        z = np.load(os.path.join(a.seqdir, k + ".npz")); s = z["seq"]; n = min(len(s), a.L)
        if len(s) > a.L: s = s[np.linspace(0, len(s) - 1, a.L).astype(int)]; n = a.L   # longer clips (1000-fps subjects) are resampled
        X[i, :n] = s[:n]; M[i, :n] = True
    return X, M


X_all, M_all = load_all(tr.key.values); y_all = tr.y4.values.astype(int)
print(f"loaded {X_all.shape}, mean valid frames {M_all.sum(1).mean():.0f}", flush=True)


class AttnPool(nn.Module):
    def __init__(self, d): super().__init__(); self.w = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
    def forward(self, h, m):
        s = self.w(h).squeeze(-1).masked_fill(~m, -1e4); w = torch.softmax(s, 1); return (h * w.unsqueeze(-1)).sum(1)


class GRUNet(nn.Module):
    def __init__(self):
        super().__init__(); self.inp = nn.Sequential(nn.Linear(D, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3))
        self.gru = nn.GRU(256, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3); self.pool = AttnPool(512); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, K))
    def forward(self, x, m):
        h = self.inp(x); lens = m.sum(1).clamp_min(1).cpu()
        p = nn.utils.rnn.pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False); o, _ = self.gru(p)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1]); return self.head(self.pool(o, m))


class TfmNet(nn.Module):
    def __init__(self):
        super().__init__(); self.inp = nn.Sequential(nn.Linear(D, 256), nn.LayerNorm(256)); self.pos = nn.Parameter(torch.zeros(1, a.L, 256)); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(256, 4, 512, dropout=0.3, batch_first=True, norm_first=True); self.enc = nn.TransformerEncoder(layer, 4)
        self.norm = nn.LayerNorm(256); self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, K))
    def forward(self, x, m):
        h = self.enc(self.inp(x) + self.pos[:, :x.shape[1]], src_key_padding_mask=~m); h = self.norm(h)
        return self.head((h * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp_min(1))


def augment(x, m):
    """random contiguous temporal crop (80-100 %), frame dropout 10 %, small coordinate noise; all mask-aware."""
    B, L, _ = x.shape; x = x.clone(); m = m.clone()
    for i in range(B):
        n = int(m[i].sum());
        if n < 20: continue
        keep = int(n * np.random.uniform(0.8, 1.0)); s = np.random.randint(0, n - keep + 1)
        seg = x[i, s:s + keep].clone(); x[i] = 0; m[i] = False; x[i, :keep] = seg; m[i, :keep] = True
        drop = torch.rand(keep, device=x.device) < 0.1; m[i, :keep][drop] = False
    return x + 0.02 * torch.randn_like(x) * m.unsqueeze(-1), m


def run_fold(arch, fold, mu, sd):
    trn, ev = np.where(tr.fold.values != fold)[0], np.where(tr.fold.values == fold)[0]
    if len(trn) == 0 or len(ev) == 0: return None
    Xt = torch.tensor((X_all - mu) / sd); Mt = torch.tensor(M_all); yt = torch.tensor(y_all)
    model = (GRUNet() if arch == "gru" else TfmNet()).to(DEV); ema = {n: p.detach().clone() for n, p in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if arch == "gru" else 5e-4, weight_decay=0.05)
    steps = math.ceil(len(trn) / a.batch) * a.epochs; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=max(steps, 1), pct_start=0.1)
    curve, P_by_epoch = [], []
    for ep in range(a.epochs):
        model.train(); perm = np.random.permutation(trn)
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]; x, m = augment(Xt[idx].to(DEV), Mt[idx].to(DEV)); y = yt[idx].to(DEV)
            loss = F.cross_entropy(model(x, m), y, label_smoothing=0.1); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); sched.step()
            with torch.no_grad():
                for n, p in model.state_dict().items():
                    if p.dtype.is_floating_point: ema[n].mul_(a.ema).add_(p, alpha=1 - a.ema)
        live = {n: p.detach().clone() for n, p in model.state_dict().items()}; model.load_state_dict(ema); model.eval(); P = []
        with torch.no_grad():
            for s in range(0, len(ev), 256):
                idx = ev[s:s + 256]; P.append(torch.softmax(model(Xt[idx].to(DEV), Mt[idx].to(DEV)), 1).cpu().numpy())
        model.load_state_dict(live); P = np.concatenate(P); P_by_epoch.append(P)
        curve.append(dict(epoch=ep, acc=accuracy_score(y_all[ev], P.argmax(1)), macro_f1=f1_score(y_all[ev], P.argmax(1), average="macro")))
    return dict(ev=ev, curve=curve, P_by_epoch=P_by_epoch)


# ---------------------------------------------------------------- pixel predictions on the same folds (from G6/G7)
PIX = {0: ["ft_g6_ep12_ema999_train.npz", "ft_g7_svfap_f0_s1_train.npz", "ft_g7_svfap_f0_s2_train.npz", "ft_g7_maedfer_f0_s0_train.npz"],
       1: ["ft_g7_svfap_f1_s0_train.npz"], 2: ["ft_g7_svfap_f2_s0_train.npz"]}
def pixel_probs(fold):
    zs = [np.load(os.path.join(ROOT, "preds", f), allow_pickle=True) for f in PIX[fold] if os.path.exists(os.path.join(ROOT, "preds", f))]
    if not zs: return None
    out = {}
    for z in zs:
        for k, p in zip(z["keys"], z["P"]): out.setdefault(k, []).append(p)
    return {k: np.mean(v, 0) for k, v in out.items()}


out = {"arch": {}, "gate": {}}
for arch in a.arch.split(","):
    t0 = time.time(); res = {}
    for fold in (0, 1, 2):
        trn = tr.fold.values != fold; mu = X_all[trn][M_all[trn]].mean(0); sd = X_all[trn][M_all[trn]].std(0) + 1e-6
        r = run_fold(arch, fold, mu, sd)
        if r: res[fold] = r; print(f"[{arch}] fold {fold}: final-epoch acc {r['curve'][-1]['acc']:.4f}, best-epoch acc {max(c['acc'] for c in r['curve']):.4f} ({time.time()-t0:.0f}s)", flush=True)
    if not res: continue
    # epoch chosen from the MEAN curve across folds, then fixed for every fold
    E = min(len(r["curve"]) for r in res.values()); mean_curve = np.mean([[c["acc"] for c in r["curve"][:E]] for r in res.values()], 0); e_star = int(mean_curve.argmax())
    summary = {"epoch_star": e_star, "mean_curve_acc": mean_curve.tolist(), "folds": {}}
    for fold, r in res.items():
        ev = r["ev"]; keys = tr.key.values[ev]; y = y_all[ev]; Ps = r["P_by_epoch"][e_star]; pix = pixel_probs(fold)
        d = dict(n=int(len(y)), seq_acc=accuracy_score(y, Ps.argmax(1)), seq_macro_f1=f1_score(y, Ps.argmax(1), average="macro"))
        if pix and all(k in pix for k in keys):
            Pp = np.stack([pix[k] for k in keys]); Pf = 0.5 * Pp + 0.5 * Ps
            errs = np.stack([(Pp.argmax(1) != y).astype(float), (Ps.argmax(1) != y).astype(float)])
            d.update(pixel_acc=accuracy_score(y, Pp.argmax(1)), fused_acc=accuracy_score(y, Pf.argmax(1)), fused_macro_f1=f1_score(y, Pf.argmax(1), average="macro"),
                     gain=accuracy_score(y, Pf.argmax(1)) - accuracy_score(y, Pp.argmax(1)), error_corr_pixel_vs_seq=float(np.corrcoef(errs)[0, 1]))
            np.savez(os.path.join(ROOT, "preds", f"seq_{arch}_fold{fold}.npz"), P=Ps, keys=keys, y=y)
        summary["folds"][fold] = d
        print(f"[{arch}] fold {fold} @epoch {e_star}: seq {d['seq_acc']:.4f}" + (f"  pixel {d['pixel_acc']:.4f}  fused {d['fused_acc']:.4f}  gain {d['gain']:+.4f}  err-corr {d['error_corr_pixel_vs_seq']:.3f}" if "gain" in d else ""), flush=True)
    g = [d["gain"] for d in summary["folds"].values() if "gain" in d]; s_acc = [d["seq_acc"] for d in summary["folds"].values()]
    if g:
        summary["gate"] = dict(seq_mean=float(np.mean(s_acc)), gate1_seq_ge_0p68=bool(np.mean(s_acc) >= 0.68), fused_gain_mean=float(np.mean(g)), folds_nonneg=int(sum(x >= 0 for x in g)),
                               gate2_pass=bool(np.mean(g) >= 0.015 and sum(x >= 0 for x in g) >= 2))
        print(f"[{arch}] GATE: seq mean {np.mean(s_acc):.4f} (>=0.68: {summary['gate']['gate1_seq_ge_0p68']}) | fused gain mean {np.mean(g):+.4f}, non-negative on {summary['gate']['folds_nonneg']}/3 folds -> gate2 {'PASS' if summary['gate']['gate2_pass'] else 'FAIL'}", flush=True)
    out["arch"][arch] = summary
json.dump(out, open(os.path.join(ROOT, "results", "SEQ_CV.json"), "w"), indent=1, default=float); print("SEQ_CV_DONE", flush=True)
