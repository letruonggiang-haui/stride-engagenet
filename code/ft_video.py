"""Fine-tune a facial-video backbone on cached face crops (cache_faces.py) for EngageNet or DAiSEE.
Design and pre-registered thresholds: 04_experiments/MODEL_PAPER_DESIGN_2026-09-21.md.

Backbones: svfap (SVFAP-B, 160 px, VoxCeleb2), maedfer (MAE-DFER-B, 160 px, VoxCeleb2), marlin (MARLIN ViT-L, 224 px).
Adaptation: full fine-tuning with layer-wise lr decay, or LoRA on the attention qkv projections (--lora R).
Training: one random 16-frame window (stride 4) per clip per epoch, random resized crop + flip, AdamW, cosine, warmup,
EMA weights. With --segments K (design v3, MODEL_V3_DESIGN_2026-10-02.md), K windows spread over the clip are trained
jointly: one shared crop and flip per clip, logits averaged over the K windows before the loss. Evaluation is unchanged.
Loss: cross-entropy with label smoothing, or ordinal label-distribution learning (--ldl SIGMA).
Missing faces: frames without a detection are zeroed after normalisation (--placeholder), or left as the held crop.
Evaluation: average of class probabilities over --tta-windows uniformly spaced windows (and their flips if --tta-flip).

Protocol: --eval-split val for selection runs (never test); test is opened once by an explicit --eval-split test run.
Inner subject-level CV on the training split: --cv-folds 3 --cv-fold k.
Writes results/FT_<tag>.json and preds/ft_<tag>.npz.
"""
import argparse, json, math, os, sys, time, io as _io
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, cv2
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels

ap = argparse.ArgumentParser()
ap.add_argument("--backbone", default="svfap", choices=["svfap", "maedfer", "marlin"])
ap.add_argument("--dataset", default="engagenet", choices=["engagenet", "daisee"])
ap.add_argument("--train-splits", default="train"); ap.add_argument("--eval-split", default="val")
ap.add_argument("--cv-folds", type=int, default=0); ap.add_argument("--cv-fold", type=int, default=-1)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--layer-decay", type=float, default=0.75); ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--lora", type=int, default=0, help="rank; 0 = full fine-tuning")
ap.add_argument("--frames", type=int, default=16); ap.add_argument("--stride", type=int, default=4)
ap.add_argument("--smoothing", type=float, default=0.1); ap.add_argument("--ldl", type=float, default=0.0, help="sigma of the ordinal Gaussian label distribution; 0 = CE")
ap.add_argument("--placeholder", action="store_true"); ap.add_argument("--balanced", action="store_true", help="class-balanced sampling of the training clips"); ap.add_argument("--drop-path", type=float, default=0.1)
ap.add_argument("--segments", type=int, default=0, help="K>0: segment consensus in training (v3); 0 = one window per clip")
ap.add_argument("--ema", type=float, default=0.999); ap.add_argument("--tta-windows", type=int, default=5); ap.add_argument("--tta-flip", action="store_true")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default=""); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--max-train", type=int, default=0, help="debug: cap the number of training clips")
ap.add_argument("--eval-only", default="", help="tag of a saved model: skip training, load models/ft_<tag>.pt and score the eval split once")
ap.add_argument("--init-ckpt", default="", help="v5: SVFAP checkpoint to start from (default: the released VoxCeleb2 checkpoint)")
ap.add_argument("--resume", action="store_true", help="v3: checkpoint after every epoch to ckpt/ft_<tag>.pt and continue from it if present (survives a reboot; a resumed run is not bit-identical)")
ap.add_argument("--open-official", action="store_true", help="required to score an official val/test split; it is then scored ONCE, after the last epoch")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); DEV = "cuda"
SIZE = 224 if a.backbone == "marlin" else 160
NORM = a.backbone != "marlin"   # MARLIN expects [0,1]; the two VideoMAE-style backbones expect ImageNet normalisation
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
CROPS = os.path.join(ROOT, "crops_face", a.dataset)


# ---------------------------------------------------------------- clip index
def index():
    if a.dataset == "engagenet":
        en = engagenet_labels(); en["y"] = en["y4"]; en["split"] = en["split"]; K = 4
        df = en[["key", "y", "subject", "split"]].copy()
    else:
        rows = []
        for split, fn in (("train", "TrainLabels.csv"), ("val", "ValidationLabels.csv"), ("test", "TestLabels.csv")):
            d = pd.read_csv(os.path.expanduser(f"~/daisee/DAiSEE/Labels/{fn}"), sep=None, engine="python"); d.columns = [c.strip() for c in d.columns]
            d["key"] = d["ClipID"].astype(str).str.strip().str.replace(r"\.(avi|mp4)$", "", regex=True); d["y"] = d["Engagement"].astype(int)
            d["subject"] = d["key"].str[:6].astype(int); d["split"] = split; rows.append(d[["key", "y", "subject", "split"]])
        df = pd.concat(rows, ignore_index=True); K = 4
    import zipfile
    df = df[df["key"].apply(lambda k: zipfile.is_zipfile(os.path.join(CROPS, k + ".npz")) if os.path.exists(os.path.join(CROPS, k + ".npz")) else False)].reset_index(drop=True)
    return df, K


df, K = index()
OFFICIAL = a.cv_folds == 0          # an official split is being scored, not an inner CV fold of the training split
if OFFICIAL and not a.open_official:
    sys.exit(f"refusing to score the official '{a.eval_split}' split without --open-official (protocol: official splits are scored once, per locked configuration)")
if OFFICIAL and a.eval_split in set(a.train_splits.split(",")):
    sys.exit(f"refusing: eval split '{a.eval_split}' is also in --train-splits '{a.train_splits}'")
tr = df[df.split.isin(a.train_splits.split(","))].reset_index(drop=True)
if a.cv_folds > 0:   # subject-level inner folds of the training split; the eval split is then a held-out slice of train
    subs = np.sort(tr.subject.unique()); rng = np.random.default_rng(12345); rng.shuffle(subs)
    fold_of = {s: i % a.cv_folds for i, s in enumerate(subs)}
    ev = tr[tr.subject.map(fold_of) == a.cv_fold].reset_index(drop=True); tr = tr[tr.subject.map(fold_of) != a.cv_fold].reset_index(drop=True)
else:
    ev = df[df.split == a.eval_split].reset_index(drop=True)
if a.max_train: tr = tr.sample(min(a.max_train, len(tr)), random_state=0).reset_index(drop=True)
print(f"{a.dataset} K={K} train {len(tr)} clips/{tr.subject.nunique()} subjects, eval {len(ev)}/{ev.subject.nunique()} ({'cv fold %d' % a.cv_fold if a.cv_folds else a.eval_split})", flush=True)


# ---------------------------------------------------------------- data
def decode(z, idxs, flip=False, crop=None):
    jpg, off = z["jpg"], z["off"]; out = []
    for i in idxs:
        im = cv2.imdecode(jpg[off[i]:off[i + 1]], cv2.IMREAD_COLOR)
        if crop is not None:
            x0, y0, s = crop; im = im[y0:y0 + s, x0:x0 + s]
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if flip: im = im[:, ::-1]
        out.append(im[:, :, ::-1])   # BGR -> RGB
    x = torch.from_numpy(np.ascontiguousarray(np.stack(out))).permute(3, 0, 1, 2).float() / 255.0   # 3,T,H,W
    if NORM: x = (x - MEAN) / STD
    return x


class Clips(Dataset):
    def __init__(self, d, train):
        self.d = d.reset_index(drop=True); self.train = train
    def __len__(self): return len(self.d)
    def windows(self, n):
        span = (a.frames - 1) * a.stride + 1
        if n < span:   # short clip: repeat the last frame; evaluation still returns tta_windows copies so batches collate
            base = list(range(0, n, a.stride)); base = (base + [n - 1] * a.frames)[:a.frames]; return [base] * max(a.segments, 1) if self.train else [base] * a.tta_windows
        if self.train and a.segments:   # v3: K windows at the evaluation starts, each moved by up to half the gap between starts
            j = max(a.stride // 2, 1); jr = (n - span) // (2 * max(a.segments - 1, 1)); out = []
            for s0 in np.linspace(0, n - span, a.segments).round().astype(int):
                s = int(np.clip(s0 + np.random.randint(-jr, jr + 1), 0, n - span))
                out.append([int(np.clip(s + t * a.stride + np.random.randint(-j, j + 1), 0, n - 1)) for t in range(a.frames)])
            return out
        if self.train:
            s = np.random.randint(0, n - span + 1); j = max(a.stride // 2, 1)
            # per-frame jitter: at large stride the window covers the whole clip, so the start offset alone
            # gives almost no augmentation left
            return [[int(np.clip(s + t * a.stride + np.random.randint(-j, j + 1), 0, n - 1)) for t in range(a.frames)]]
        starts = np.linspace(0, n - span, a.tta_windows).round().astype(int); return [[s + t * a.stride for t in range(a.frames)] for s in starts]
    def __getitem__(self, i):
        r = self.d.iloc[i]; z = np.load(os.path.join(CROPS, r.key + ".npz")); n = len(z["off"]) - 1; found = z["found"]
        xs = []; shared = None
        if self.train and a.segments:   # v3: the K windows of a clip share one crop and one flip
            sc = np.random.uniform(0.7, 1.0); side = int(224 * sc); shared = ((np.random.randint(0, 224 - side + 1), np.random.randint(0, 224 - side + 1), side), np.random.rand() < 0.5)
        for w in self.windows(n):
            if self.train and shared is not None:
                x = decode(z, w, flip=shared[1], crop=shared[0])
            elif self.train:
                sc = np.random.uniform(0.7, 1.0); side = int(224 * sc); x0 = np.random.randint(0, 224 - side + 1); y0 = np.random.randint(0, 224 - side + 1)
                x = decode(z, w, flip=np.random.rand() < 0.5, crop=(x0, y0, side))
            else:
                x = decode(z, w)
            if a.placeholder:
                miss = torch.from_numpy(~found[np.array(w)]).view(1, -1, 1, 1); x = x.masked_fill(miss, 0.0)
            xs.append(x)
        return torch.stack(xs), int(r.y), i


# ---------------------------------------------------------------- model
class LoRAQKV(nn.Module):
    """Low-rank update exposed through .weight/.bias, because both repos' attention modules read qkv.weight directly."""
    def __init__(self, base, r, alpha=None):
        super().__init__(); self.base = base
        for p in base.parameters(): p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.02); self.B = nn.Parameter(torch.zeros(base.out_features, r)); self.scale = (alpha or 2 * r) / r
        self.in_features, self.out_features = base.in_features, base.out_features
    @property
    def weight(self): return self.base.weight + self.scale * (self.B @ self.A)
    @property
    def bias(self): return self.base.bias
    def forward(self, x): return F.linear(x, self.weight, self.bias)


def strip(sd):
    out = {}
    for k, v in sd.items():
        k2 = k
        for p in ("module.", "encoder.", "backbone."):
            if k2.startswith(p): k2 = k2[len(p):]
        if k2.startswith(("decoder", "encoder_to_decoder", "mask_token", "head.")): continue
        out[k2] = v
    return out


def build():
    repos = os.path.join(ROOT, "repos")
    if a.backbone == "svfap":
        sys.path.insert(0, os.path.join(repos, "SVFAP")); import modeling_finetune as mf
        m = mf.svfap_base_patch16_160(num_classes=K, all_frames=a.frames, drop_path_rate=a.drop_path, stage_depths=(12, 6, 3), temporal_kernel_size=2, num_bottleneck_tokens=(8,))
        ck = torch.load(a.init_ckpt or os.path.join(ROOT, "weights/svfap/svfap_base_voxceleb2_checkpoint-99.pth"), map_location="cpu", weights_only=False)
    elif a.backbone == "maedfer":
        sys.path.insert(0, os.path.join(repos, "MAE-DFER")); import modeling_finetune as mf
        m = mf.vit_base_dim512_no_depth_patch16_160(num_classes=K, all_frames=a.frames, drop_path_rate=a.drop_path, depth=16, attn_type="local_global", lg_region_size=(2, 5, 10),
                                                    lg_first_attn_type="self", lg_third_attn_type="cross", lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
                                                    lg_classify_token_type="region", lg_no_second=False, lg_no_third=False)
        ck = torch.load(os.path.join(ROOT, "weights/maedfer/maedfer_base_voxceleb2.pth"), map_location="cpu", weights_only=False)
    else:
        import extract_feats as EF; enc = EF.load_marlin()
        class M(nn.Module):
            def __init__(self):
                super().__init__(); self.enc = enc; self.head = nn.Linear(1024, K)
            def forward(self, x): return self.head(self.enc(x, keep_seq=False))
        return M(), None
    sd = strip(ck.get("model", ck.get("module", ck))); res = m.load_state_dict(sd, strict=False)
    print(f"loaded {a.backbone}: missing {len(res.missing_keys)} {res.missing_keys[:6]} unexpected {len(res.unexpected_keys)} {res.unexpected_keys[:6]}", flush=True)
    return m, res


model, _ = build()
if a.lora:
    n_l = 0
    for name, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cn in ("qkv", "q", "kv", "v", "query", "value"):
                setattr(mod, cn, LoRAQKV(child, a.lora)); n_l += 1
    for n, p in model.named_parameters():
        if not (".A" in n or ".B" in n or n.startswith("head") or "head." in n): p.requires_grad = False
    print("lora modules", n_l, "trainable", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
model = model.to(DEV)


def param_groups():
    """layer-wise lr decay for full fine-tuning (VideoMAE convention); one group for LoRA."""
    if a.lora: return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr, "weight_decay": 0.0}]
    blocks = [n for n, _ in model.named_modules() if n.startswith("blocks.") and n.count(".") == 1]; nb = len(blocks)
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("blocks."): lid = int(n.split(".")[1]) + 1
        elif n.startswith(("patch_embed", "pos_embed", "cls_token", "enc.")): lid = 0
        else: lid = nb + 1
        scale = a.layer_decay ** (nb + 1 - lid); wd = 0.0 if (p.ndim == 1 or n.endswith(".bias")) else a.wd
        g = groups.setdefault((lid, wd), {"params": [], "lr": a.lr * scale, "weight_decay": wd}); g["params"].append(p)
    return list(groups.values())


opt = torch.optim.AdamW(param_groups(), betas=(0.9, 0.999))
ema = {n: p.detach().clone() for n, p in model.state_dict().items()}
train_ds, eval_ds = Clips(tr, True), Clips(ev, False)
if a.balanced:
    cnt = np.bincount(tr.y.values, minlength=K); w = (1.0 / np.maximum(cnt, 1))[tr.y.values]
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(tr), replacement=True)
    tl = DataLoader(train_ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True)
else:
    tl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True)
el = DataLoader(eval_ds, batch_size=max(1, a.batch // 2), shuffle=False, num_workers=a.workers, pin_memory=True)
steps_per_ep = len(tl) // a.accum; total = steps_per_ep * a.epochs; warm = steps_per_ep * a.warmup
base_lrs = [g["lr"] for g in opt.param_groups]


def set_lr(step):
    f = step / max(warm, 1) if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
    for g, b in zip(opt.param_groups, base_lrs): g["lr"] = b * f


def target(y):
    if a.ldl > 0:
        d = torch.exp(-0.5 * ((torch.arange(K, device=y.device)[None] - y[:, None]) / a.ldl) ** 2); return d / d.sum(1, keepdim=True)
    t = torch.full((len(y), K), a.smoothing / K, device=y.device); t[torch.arange(len(y)), y] += 1 - a.smoothing; return t


def evaluate(use_ema=True):
    if use_ema:
        backup = {n: p.detach().clone() for n, p in model.state_dict().items()}; model.load_state_dict(ema, strict=True)
    model.eval(); P = np.zeros((len(eval_ds), K), np.float32); Y = np.zeros(len(eval_ds), int)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for x, y, i in el:
            b, w = x.shape[:2]; x = x.view(b * w, *x.shape[2:]).to(DEV, non_blocking=True)
            pr = F.softmax(model(x).float(), 1)
            if a.tta_flip: pr = 0.5 * (pr + F.softmax(model(x.flip(-1)).float(), 1))
            P[i.numpy()] = pr.view(b, w, K).mean(1).cpu().numpy(); Y[i.numpy()] = y.numpy()
    if use_ema: model.load_state_dict(backup, strict=True)
    yh = P.argmax(1); from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    return dict(acc=accuracy_score(Y, yh), uar=balanced_accuracy_score(Y, yh), macro_f1=f1_score(Y, yh, average="macro"), pred_dist=np.bincount(yh, minlength=K).tolist()), P, Y


tag = a.tag or f"{a.dataset}_{a.backbone}_{'lora%d' % a.lora if a.lora else 'full'}_s{a.seed}" + (f"_cv{a.cv_fold}" if a.cv_folds else f"_{a.eval_split}")
tag = tag + "_" + "-".join(sorted(a.train_splits.split(",")))   # train-only and train+val runs must not share a filename
if a.eval_only:
    _sd = torch.load(os.path.join(ROOT, "models", f"ft_{a.eval_only}.pt"), map_location=DEV)
    _res = model.load_state_dict(_sd, strict=False); print(f"eval-only: loaded {len(_sd)} tensors from ft_{a.eval_only}.pt, missing {len(_res.missing_keys)} unexpected {len(_res.unexpected_keys)}", flush=True)
    ema = {n: p.detach().clone() for n, p in model.state_dict().items()}; a.epochs = 0
_rp = os.path.join(ROOT, "results", f"FT_{tag}.json")
if OFFICIAL and os.path.exists(_rp):
    sys.exit(f"refusing to overwrite {_rp}: an official-split result is write-once")
log = []; step = 0; t0 = time.time(); scaler = torch.amp.GradScaler("cuda")
start_ep, t_prev, RESUMED = 0, 0.0, []
CK = os.path.join(ROOT, "ckpt", f"ft_{tag}.pt")
if a.resume and os.path.exists(CK):   # v3: continue an interrupted run from its last completed epoch
    _ck = torch.load(CK, map_location=DEV, weights_only=False)
    model.load_state_dict(_ck["model"], strict=False)
    for _n, _v in _ck["ema"].items(): ema[_n].copy_(_v)
    opt.load_state_dict(_ck["opt"]); scaler.load_state_dict(_ck["scaler"])
    step, start_ep, log, t_prev = _ck["step"], _ck["epoch"] + 1, _ck["log"], _ck["seconds"]; RESUMED = _ck["resumed"] + [start_ep]
    torch.set_rng_state(_ck["rng_torch"].cpu()); torch.cuda.set_rng_state(_ck["rng_cuda"].cpu());   # RNG states must be CPU byte tensors
    np.random.set_state(_ck["rng_np"]); del _ck
    print(f"resumed from {CK}: continuing at epoch {start_ep} of {a.epochs}, step {step}", flush=True)


def save_ckpt(ep):
    if not a.resume: return
    os.makedirs(os.path.dirname(CK), exist_ok=True); tmp = CK + ".tmp"
    torch.save(dict(model={n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, ema={n: v.cpu() for n, v in ema.items()},
                    opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, epoch=ep, log=log, seconds=t_prev + time.time() - t0, resumed=RESUMED,
                    rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(), rng_np=np.random.get_state(), args=vars(a)), tmp)
    os.replace(tmp, CK)   # atomic: an interrupted save leaves the previous checkpoint intact
    print(f"ckpt saved epoch {ep} peak_mem_gb {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}", flush=True)


for ep in range(start_ep, a.epochs):
    model.train(); tot, nb = 0.0, 0
    for k, (x, y, _) in enumerate(tl):
        nw = x.shape[1] if a.segments else 1
        x = (x.reshape(x.shape[0] * nw, *x.shape[2:]) if a.segments else x[:, 0]).to(DEV, non_blocking=True); y = y.to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x).float()
            if a.segments: logits = logits.view(-1, nw, logits.shape[-1]).mean(1)   # segment consensus on logits
        loss = -(target(y) * F.log_softmax(logits, 1)).sum(1).mean() / a.accum
        scaler.scale(loss).backward(); tot += loss.item() * a.accum; nb += 1
        if (k + 1) % a.accum == 0:
            set_lr(step); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            with torch.no_grad():
                d = 0.0 if step < warm else a.ema   # d=0 copies the live weights; d=1 would freeze the EMA at init
                for n, p in model.state_dict().items():
                    if p.dtype.is_floating_point: ema[n].mul_(d).add_(p.detach(), alpha=1 - d)
                    else: ema[n].copy_(p)
    if OFFICIAL and ep < a.epochs - 1:
        print(f"ep {ep} loss {tot / max(nb, 1):.3f} (official split not scored until the final epoch) {time.time() - t0:.0f}s", flush=True); log.append(dict(epoch=ep, loss=tot / max(nb, 1))); save_ckpt(ep); continue
    r, P, Y = evaluate()
    log.append(dict(epoch=ep, loss=tot / max(nb, 1), **r)); print(f"ep {ep} loss {tot / max(nb, 1):.3f} acc {r['acc']:.4f} uar {r['uar']:.4f} f1 {r['macro_f1']:.4f} {r['pred_dist']} {time.time() - t0:.0f}s", flush=True)
    save_ckpt(ep)
if a.eval_only:
    r, P, Y = evaluate(); log.append(dict(epoch=-1, loss=float("nan"), **r)); print(f"eval-only acc {r['acc']:.4f} uar {r['uar']:.4f} f1 {r['macro_f1']:.4f} {r['pred_dist']}", flush=True)
raw, Praw, _ = evaluate(use_ema=False)   # the EMA is the reported model; the raw weights are a diagnostic on EMA lag
print(f"final raw-weight (non-EMA) score: acc {raw['acc']:.4f} uar {raw['uar']:.4f} f1 {raw['macro_f1']:.4f}", flush=True)
final = log[-1]
ev_counts = np.bincount(ev.y.values, minlength=K)
out = dict(args=vars(a), K=K, n_train=len(tr), n_eval=len(ev), subjects_eval=int(ev.subject.nunique()),
           eval_class_counts=ev_counts.tolist(), eval_majority_baseline=float(ev_counts.max() / max(len(ev), 1)),
           train_class_counts=np.bincount(tr.y.values, minlength=K).tolist(),
           span_frames=(a.frames - 1) * a.stride + 1, final=final, final_raw_weights=raw, log=log, seconds=t_prev + time.time() - t0,
           eval_set=("official_" + a.eval_split) if OFFICIAL else f"inner_cv_fold_{a.cv_fold}_of_{a.cv_folds}",
           eval_is_official_split=bool(OFFICIAL), tta_windows=a.tta_windows, tta_flip=bool(a.tta_flip))
if a.resume: out["resumed_at_epochs"] = RESUMED
print(f"majority baseline on this eval set: {out['eval_majority_baseline']:.4f} (accuracy is only comparable within the same split)", flush=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True); os.makedirs(os.path.join(ROOT, "preds"), exist_ok=True)
json.dump(out, open(os.path.join(ROOT, "results", f"FT_{tag}.json"), "w"), indent=1, default=float)
np.savez(os.path.join(ROOT, "preds", f"ft_{tag}.npz"), P=P, y=Y, keys=ev.key.values, subject=ev.subject.values)
torch.save({n: v.cpu() for n, v in ema.items() if a.lora and (".A" in n or ".B" in n or "head" in n)} if a.lora else {n: v.cpu() for n, v in ema.items()}, os.path.join(ROOT, "models", f"ft_{tag}.pt")) if os.path.isdir(os.path.join(ROOT, "models")) else None
if a.resume and os.path.exists(CK): os.remove(CK)   # the run is complete; its result, predictions and weights are saved above
print("FT_DONE", tag, json.dumps(final), flush=True)
