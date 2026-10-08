"""Continued MAE pretraining of SVFAP-B at the CMS window, design MODEL_V5_DESIGN_2026-10-05
(sha256 c13e32416f4df296273040b4692d73d1f91695761bf3c942a2a804de90103994).

Starts from the released SVFAP pretraining checkpoint (encoder and decoder), trains on the unlabelled clips of exactly the
training set the later fine-tuning uses (same index and the same subject-level folds as ft_video.py), samples 16 frames at
stride 20 with ft_video.py's training sampler and augmentation, tube masking at 0.9, and the per-patch normalised pixel MSE of
SVFAP's engine_for_pretraining.py. Checkpoints after every epoch and continues from the last one after an interruption.

Writes weights/svfap_v5/<tag>.pth  ({"model": state_dict} like the released checkpoint, so ft_video.py --init-ckpt loads it)
and results/PRE_<tag>.json (write-once).

usage: pretrain_v5.py --dataset engagenet --train-splits train --cv-folds 3 --cv-fold 0 --tag v5_pre_f0
"""
import argparse, json, math, os, sys, time, zipfile
from functools import partial
import numpy as np, pandas as pd, torch, torch.nn as nn, cv2
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels

DESIGN_SHA = "c13e32416f4df296273040b4692d73d1f91695761bf3c942a2a804de90103994"
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="engagenet", choices=["engagenet", "daisee"])
ap.add_argument("--train-splits", default="train")
ap.add_argument("--cv-folds", type=int, default=0); ap.add_argument("--cv-fold", type=int, default=-1)
ap.add_argument("--frames", type=int, default=16); ap.add_argument("--stride", type=int, default=20)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--min-lr", type=float, default=1e-6); ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--mask-ratio", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--max-train", type=int, default=0, help="smoke test only")
ap.add_argument("--tag", required=True)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); DEV = "cuda"; SIZE = 160
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
CROPS = os.path.join(ROOT, "crops_face", a.dataset)
OUT_W = os.path.join(ROOT, "weights", "svfap_v5", f"{a.tag}.pth"); OUT_R = os.path.join(ROOT, "results", f"PRE_{a.tag}.json")
CK = os.path.join(ROOT, "ckpt", f"pre_{a.tag}.pt")
if os.path.exists(OUT_R): sys.exit(f"refusing: {OUT_R} exists (write-once); the adapted weights are at {OUT_W}")


# ---------------------------------------------------------------- the training clips, built exactly as in ft_video.py
def index():
    if a.dataset == "engagenet":
        en = engagenet_labels(); en["y"] = en["y4"]; df = en[["key", "y", "subject", "split"]].copy()
    else:
        rows = []
        for split, fn in (("train", "TrainLabels.csv"), ("val", "ValidationLabels.csv"), ("test", "TestLabels.csv")):
            d = pd.read_csv(os.path.expanduser(f"~/daisee/DAiSEE/Labels/{fn}"), sep=None, engine="python"); d.columns = [c.strip() for c in d.columns]
            d["key"] = d["ClipID"].astype(str).str.strip().str.replace(r"\.(avi|mp4)$", "", regex=True); d["y"] = d["Engagement"].astype(int)
            d["subject"] = d["key"].str[:6].astype(int); d["split"] = split; rows.append(d[["key", "y", "subject", "split"]])
        df = pd.concat(rows, ignore_index=True)
    df = df[df["key"].apply(lambda k: zipfile.is_zipfile(os.path.join(CROPS, k + ".npz")) if os.path.exists(os.path.join(CROPS, k + ".npz")) else False)].reset_index(drop=True)
    return df


df = index()
tr = df[df.split.isin(a.train_splits.split(","))].reset_index(drop=True)
if a.cv_folds > 0:
    subs = np.sort(tr.subject.unique()); rng = np.random.default_rng(12345); rng.shuffle(subs)
    fold_of = {s: i % a.cv_folds for i, s in enumerate(subs)}
    held = tr[tr.subject.map(fold_of) == a.cv_fold]; tr = tr[tr.subject.map(fold_of) != a.cv_fold].reset_index(drop=True)
    assert not set(held.subject) & set(tr.subject)
if a.max_train: tr = tr.sample(min(a.max_train, len(tr)), random_state=0).reset_index(drop=True)
print(f"{a.dataset} pretraining clips {len(tr)} from {tr.subject.nunique()} subjects "
      f"({'inner fold %d held out' % a.cv_fold if a.cv_folds else 'splits ' + a.train_splits}); labels are not used", flush=True)

sys.path.insert(0, os.path.join(ROOT, "repos", "SVFAP"))
import modeling_pretrain as mp
from masking_generator import TubeMaskingGenerator
from einops import rearrange


def decode(z, idxs, flip=False, crop=None):
    jpg, off = z["jpg"], z["off"]; out = []
    for i in idxs:
        im = cv2.imdecode(jpg[off[i]:off[i + 1]], cv2.IMREAD_COLOR)
        if crop is not None:
            x0, y0, s = crop; im = im[y0:y0 + s, x0:x0 + s]
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        if flip: im = im[:, ::-1]
        out.append(im[:, :, ::-1])
    x = torch.from_numpy(np.ascontiguousarray(np.stack(out))).permute(3, 0, 1, 2).float() / 255.0
    return (x - MEAN) / STD


class Clips(Dataset):   # ft_video.py's training sampler and augmentation, one window per clip
    def __init__(self, d):
        self.d = d.reset_index(drop=True); self.mask = TubeMaskingGenerator((a.frames // 2, SIZE // 16, SIZE // 16), a.mask_ratio)
    def __len__(self): return len(self.d)
    def window(self, n):
        span = (a.frames - 1) * a.stride + 1
        if n < span:
            base = list(range(0, n, a.stride)); return (base + [n - 1] * a.frames)[:a.frames]
        s = np.random.randint(0, n - span + 1); j = max(a.stride // 2, 1)
        return [int(np.clip(s + t * a.stride + np.random.randint(-j, j + 1), 0, n - 1)) for t in range(a.frames)]
    def __getitem__(self, i):
        z = np.load(os.path.join(CROPS, self.d.key.iloc[i] + ".npz")); n = len(z["off"]) - 1
        w = self.window(n)
        sc = np.random.uniform(0.7, 1.0); side = int(224 * sc); x0 = np.random.randint(0, 224 - side + 1); y0 = np.random.randint(0, 224 - side + 1)
        x = decode(z, w, flip=np.random.rand() < 0.5, crop=(x0, y0, side))
        return x, torch.from_numpy(self.mask()).bool()


# ---------------------------------------------------------------- model: the released pretraining checkpoint, encoder and decoder
model = mp.PretrainVisionTransformer(img_size=SIZE, patch_size=16, encoder_embed_dim=512, encoder_num_heads=8, encoder_num_classes=0,
                                     decoder_num_classes=1536, decoder_embed_dim=384, decoder_num_heads=6, mlp_ratio=4, qkv_bias=True,
                                     norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_depth=4, stage_depths=(12, 6, 3),
                                     num_bottleneck_tokens=(8,), temporal_kernel_size=2)
src = torch.load(os.path.join(ROOT, "weights/svfap/svfap_base_voxceleb2_checkpoint-99.pth"), map_location="cpu", weights_only=False)
print("released checkpoint:", model.load_state_dict(src["model"], strict=True), flush=True); del src
model = model.to(DEV)
decay, no_decay = [], []
for n, p in model.named_parameters():
    (no_decay if (p.ndim == 1 or n.endswith(".bias") or "pos_embed" in n or n == "mask_token") else decay).append(p)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler("cuda")
tl = DataLoader(Clips(tr), batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True)
steps_per_ep = len(tl) // a.accum; total = steps_per_ep * a.epochs; warm = steps_per_ep * a.warmup


def lr_at(step):
    if step < warm: return a.lr * step / max(warm, 1)
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))


def target(videos):   # engine_for_pretraining.py, normlize_target=True
    with torch.no_grad():
        mean = MEAN.to(DEV)[None]; std = STD.to(DEV)[None]
        un = videos * std + mean
        sq = rearrange(un, "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c", p0=2, p1=16, p2=16)
        nm = (sq - sq.mean(dim=-2, keepdim=True)) / (sq.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        return rearrange(nm, "b n p c -> b n (p c)")


step, start_ep, log, t_prev = 0, 0, [], 0.0
if os.path.exists(CK):
    ck = torch.load(CK, map_location=DEV, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    step, start_ep, log, t_prev = ck["step"], ck["epoch"] + 1, ck["log"], ck["seconds"]
    torch.set_rng_state(ck["rng_torch"].cpu()); torch.cuda.set_rng_state(ck["rng_cuda"].cpu()); np.random.set_state(ck["rng_np"]); del ck
    print(f"resumed from {CK}: continuing at epoch {start_ep} of {a.epochs}, step {step}", flush=True)
RESUMED = [start_ep] if start_ep else []
loss_fn = nn.MSELoss(); t0 = time.time()
for ep in range(start_ep, a.epochs):
    model.train(); tot, nb = 0.0, 0
    for k, (x, m) in enumerate(tl):
        x = x.to(DEV, non_blocking=True); m = m.to(DEV, non_blocking=True).flatten(1)
        lab = target(x); B, _, C = lab.shape; lab = lab[m].reshape(B, -1, C)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, m)
        loss = loss_fn(out.float(), lab) / a.accum
        if not math.isfinite(loss.item()): sys.exit(f"loss is {loss.item()}, stopping")
        scaler.scale(loss).backward(); tot += loss.item() * a.accum; nb += 1
        if (k + 1) % a.accum == 0:
            for g in opt.param_groups: g["lr"] = lr_at(step)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
    log.append(dict(epoch=ep, loss=tot / max(nb, 1), lr=lr_at(step)))
    print(f"pre ep {ep} loss {tot / max(nb, 1):.4f} lr {lr_at(step):.2e} {t_prev + time.time() - t0:.0f}s", flush=True)
    os.makedirs(os.path.dirname(CK), exist_ok=True); tmp = CK + ".tmp"
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), step=step, epoch=ep, log=log,
                    seconds=t_prev + time.time() - t0, rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(),
                    rng_np=np.random.get_state()), tmp)
    os.replace(tmp, CK)
    print(f"ckpt saved epoch {ep} peak_mem_gb {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}", flush=True)

os.makedirs(os.path.dirname(OUT_W), exist_ok=True)
torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "args": vars(a), "design_sha256": DESIGN_SHA}, OUT_W + ".tmp")
os.replace(OUT_W + ".tmp", OUT_W)
json.dump(dict(args=vars(a), design_sha256=DESIGN_SHA, n_train=len(tr), subjects=int(tr.subject.nunique()), log=log,
               seconds=t_prev + time.time() - t0, resumed_at_epochs=RESUMED, weights=OUT_W), open(OUT_R, "w"), indent=1, default=float)
if os.path.exists(CK): os.remove(CK)
print("PRE_DONE", a.tag, OUT_W, flush=True)
