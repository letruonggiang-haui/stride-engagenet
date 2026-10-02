"""Shared data loading, models, aggregation, metrics and cluster bootstrap."""
import glob, json, os, re
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, balanced_accuracy_score

ROOT = os.path.expanduser("~/engagenet_p2_crossdomain")
EN_LAB = os.path.expanduser("~/engagenet/data/EmotiW_2023_EngageNet")
U4 = ["Not-Engaged", "Barely-engaged", "Engaged", "Highly-Engaged"]
U4_TO_U3 = {0: 0, 1: 0, 2: 1, 3: 2}
# Table I, LDL target over U3 for each native EngageNet level
EN_LDL = np.array([[1.00, 0.00, 0.00], [0.80, 0.20, 0.00], [0.10, 0.80, 0.10], [0.00, 0.15, 0.85]], np.float32)
OUC_LAB = {"Low": 0, "Medium": 1, "High": 2}


def gauss_ldl(k, sigma, n=3):
    j = np.arange(n); d = np.exp(-(j - k) ** 2 / (2 * sigma ** 2)); return (d / d.sum()).astype(np.float32)


# ------------------------------------------------------------------ labels
def engagenet_labels():
    rows = []
    for split, fn in (("train", "train_engagement_labels.xlsx"), ("val", "validation_engagement_labels.xlsx"), ("test", "test_engagement_labels.csv")):
        p = os.path.join(EN_LAB, fn)
        df = pd.read_excel(p) if fn.endswith("xlsx") else pd.read_csv(p)
        df = df[["chunk", "label"]].copy(); df["split"] = split; rows.append(df)
    df = pd.concat(rows, ignore_index=True)
    df = df[df["label"].isin(U4)].copy()  # drop SNP
    df["y4"] = df["label"].map({l: i for i, l in enumerate(U4)})
    df["y3"] = df["y4"].map(U4_TO_U3)
    m = df["chunk"].str.extract(r"subject_(\d+)_([a-z0-9]+)_vid_(\d+)_(\d+)\.mp4")
    df["subject"] = m[0].astype(int); df["session"] = m[1]; df["video"] = m[2].astype(int); df["idx"] = m[3].astype(int)
    df["recording"] = df["subject"].astype(str) + "_" + df["session"] + "_" + df["video"].astype(str)
    df["key"] = df["chunk"].str[:-4]
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ features
def load_feats(featdir, keys, enc, view, pool_windows=True):
    """returns list of arrays [n_inst, D] (windows averaged over found faces for face views), and mask of loaded"""
    out, ok, nface, ninst = [], [], [], []
    for k in keys:
        p = os.path.join(featdir, k + ".npz")
        if not os.path.exists(p):
            out.append(None); ok.append(False); nface.append(0); ninst.append(0); continue
        z = np.load(p); a = z[f"{enc}__{view}"].astype(np.float32)  # n,3,D
        ff = z["face_found"]
        if view.startswith("face"):
            w = ff[..., None].astype(np.float32)
            keep = ff.any(1)
            feat = (a * w).sum(1) / np.maximum(w.sum(1), 1)
            feat = feat[keep]
        else:
            feat = a.mean(1)
        out.append(feat); ok.append(len(feat) > 0); nface.append(int(ff.any(1).sum())); ninst.append(int(ff.shape[0]))
    return out, np.array(ok), np.array(nface), np.array(ninst)


# ------------------------------------------------------------------ model
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class AdapterNet(nn.Module):
    """LayerNorm -> residual bottleneck adapter -> head. head: 'softmax' (K logits) or 'coral' (K-1 logits)."""

    def __init__(self, d, k=3, r=256, head="softmax", p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.down = nn.Linear(d, r); self.up = nn.Linear(r, d); nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        self.norm2 = nn.LayerNorm(d); self.drop = nn.Dropout(p)
        self.head_type = head; self.k = k
        if head == "coral":
            self.fc = nn.Linear(d, 1, bias=False); self.bias = nn.Parameter(torch.zeros(k - 1))
        else:
            self.fc = nn.Linear(d, k)
        self.disc = None

    def embed(self, x):
        h = self.norm(x); h = h + self.up(self.drop(F.gelu(self.down(h)))); return self.norm2(h)

    def logits(self, h):
        return self.fc(self.drop(h)) + self.bias if self.head_type == "coral" else self.fc(self.drop(h))

    def forward(self, x):
        return self.logits(self.embed(x))

    def probs(self, x):
        z = self.forward(x)
        if self.head_type == "coral":
            s = torch.sigmoid(z)  # P(y>k)
            s = torch.cat([torch.ones_like(s[:, :1]), s, torch.zeros_like(s[:, :1])], 1)
            p = (s[:, :-1] - s[:, 1:]).clamp_min(1e-6)
            return p / p.sum(1, keepdim=True)
        return F.softmax(z, 1)

    def add_discriminator(self, d):
        self.disc = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1))


def n_params(m, exclude_disc=True):
    return sum(p.numel() for n, p in m.named_parameters() if not (exclude_disc and n.startswith("disc")))


def loss_fn(model, x, y_hard, y_dist, kind, w=None):
    z = model(x)
    if kind == "ldl":
        lp = torch.log(model.probs(x)) if model.head_type == "coral" else F.log_softmax(z, 1)
        l = F.kl_div(lp, y_dist, reduction="none").sum(1)
    elif model.head_type == "coral":
        lev = torch.stack([(y_hard > j).float() for j in range(model.k - 1)], 1)
        l = F.binary_cross_entropy_with_logits(z, lev, reduction="none").sum(1)
    else:
        l = F.cross_entropy(z, y_hard, reduction="none")
    if w is not None:
        l = l * w
    return l.mean()


# ------------------------------------------------------------------ aggregation over persons
def aggregate(p_list, op):
    """p_list: list of arrays [n_i, K] -> group distribution [K]"""
    out = []
    for p in p_list:
        K = p.shape[1]
        if len(p) == 0:
            out.append(np.full(K, 1 / K)); continue
        if op == "mean":
            out.append(p.mean(0))
        elif op == "majority":
            c = np.bincount(p.argmax(1), minlength=K) / len(p); out.append(c)
        elif op == "median":
            e = (p * np.arange(K)).sum(1); m = int(np.clip(np.round(np.median(e)), 0, K - 1)); v = np.zeros(K); v[m] = 1; out.append(v)
        else:
            raise ValueError(op)
    return np.stack(out)


def decode(P, rule):
    K = P.shape[1]
    if rule == "argmax":
        return P.argmax(1)
    return np.clip(np.round((P * np.arange(K)).sum(1)), 0, K - 1).astype(int)  # expectation


# ------------------------------------------------------------------ metrics
def metrics(y, yhat, K):
    labels = list(range(K))
    return dict(macro_f1=f1_score(y, yhat, labels=labels, average="macro", zero_division=0),
                qwk=cohen_kappa_score(y, yhat, labels=labels, weights="quadratic") if len(set(y)) > 1 or len(set(yhat)) > 1 else 0.0,
                mae=float(np.abs(np.asarray(y) - np.asarray(yhat)).mean()),
                acc=accuracy_score(y, yhat), bacc=balanced_accuracy_score(y, yhat))


def _safe_qwk(y, yh, K):
    if len(np.unique(np.concatenate([y, yh]))) < 2:
        return 0.0
    return cohen_kappa_score(y, yh, labels=list(range(K)), weights="quadratic")


def cluster_bootstrap(y, yhat, clusters, K, B=1000, seed=0, yhat_b=None, alpha=0.05):
    """percentile CI by resampling clusters. If yhat_b given, returns CI of difference (yhat - yhat_b)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); yhat = np.asarray(yhat); cl = np.asarray(clusters)
    uc = np.unique(cl); idx = {c: np.where(cl == c)[0] for c in uc}
    res = {m: [] for m in ("macro_f1", "qwk", "mae")}
    for _ in range(B):
        s = np.concatenate([idx[c] for c in rng.choice(uc, len(uc), replace=True)])
        def mm(yh):
            return dict(macro_f1=f1_score(y[s], yh[s], labels=list(range(K)), average="macro", zero_division=0),
                        qwk=_safe_qwk(y[s], yh[s], K), mae=float(np.abs(y[s] - yh[s]).mean()))
        a = mm(yhat)
        if yhat_b is not None:
            b = mm(np.asarray(yhat_b)); a = {k: a[k] - b[k] for k in a}
        for k in res:
            res[k].append(a[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in res.items()}
