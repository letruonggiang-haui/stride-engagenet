"""Unified frozen-backbone feature extraction for all domains.

One pipeline for EngageNet, OUC-CGE and DAiSEE so that the source and target
representations differ only in what the camera saw.

Per clip: 3 windows of 16 frames (frame step 2) centred at 1/6, 3/6, 5/6 of the clip.
Views (crops) produced per instance:
  face      square crop 1.3x the detected face box, 224x224
  person    person box (YOLO11s, COCO weights), padded to square, 224x224
  full      whole frame letterboxed to 224x224 (EngageNet only; one instance)
  face_lr   face crop downsampled to a target face height then upsampled (EngageNet only)
EngageNet/DAiSEE: one instance per clip (largest face / largest person).
OUC-CGE: one instance per person detected on the middle frame; the box is held
static over the clip (students are seated) and the face is searched inside it.

Encoders: MARLIN ViT-L (YTF) and VideoMAE-B (Kinetics), mean-pooled tokens.
Output: one .npz per clip with arrays  <encoder>__<view>  of shape [n_inst, 3, D]
plus face_found [n_inst, 3], boxes, face_heights.

usage: extract_feats.py DOMAIN LISTFILE OUTDIR [--views face,person,full,face_lr] [--lr-heights file.npy]
"""
import argparse, os, sys, time, traceback, zlib
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
from multiprocessing import Pool
import numpy as np, cv2, torch

ROOT = os.path.expanduser("~/engagenet_p2_crossdomain")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
T, STEP, S = 16, 2, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

_det = None; _yolo = None; _args = None; _lr_h = None


def init_worker(args):
    global _det, _yolo, _args, _lr_h
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    import onnxruntime as ort
    from insightface.model_zoo.scrfd import SCRFD
    ort.set_default_logger_severity(3)  # the detector warns on every call about a benign output-shape mismatch
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1  # one thread per worker, no oversubscription
    so.log_severity_level = 3
    onnx = os.path.expanduser("~/.insightface/models/buffalo_l/det_10g.onnx")
    _det = SCRFD(model_file=onnx, session=ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"]))
    _det.prepare(ctx_id=-1, input_size=(320, 320), det_thresh=0.4)
    from ultralytics import YOLO
    _yolo = YOLO(os.path.join(ROOT, "weights/yolo11s.pt"))  # COCO-pretrained weights only
    _args = args
    _lr_h = np.load(args.lr_heights) if args.lr_heights else None


def open_video(path):
    # single decoder thread: FFmpeg otherwise starts one thread per core in every worker
    return cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])


def read_frames(path, idxs):
    cap = open_video(path); out = {}; want = sorted(set(idxs)); k = 0; i = 0
    while k < len(want):
        ok = cap.grab()
        if not ok:
            break
        if i == want[k]:
            ok, fr = cap.retrieve()
            if ok:
                out[i] = fr
            k += 1
        i += 1
    cap.release()
    return out


def sq_crop(img, box, scale, size=S):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * scale
    h, w = img.shape[:2]
    a = int(round(cx - side / 2)); b = int(round(cy - side / 2)); s = max(int(round(side)), 2)
    pad = np.zeros((s, s, 3), np.uint8)
    xa, ya, xb, yb = max(a, 0), max(b, 0), min(a + s, w), min(b + s, h)
    if xb > xa and yb > ya:
        pad[ya - b:yb - b, xa - a:xb - a] = img[ya:yb, xa:xb]
    return cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA if s > size else cv2.INTER_LINEAR)


def letterbox(img, size=S):
    h, w = img.shape[:2]; r = size / max(h, w)
    im = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8); y = (size - im.shape[0]) // 2; x = (size - im.shape[1]) // 2
    out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


def faces_in(img, offset=(0, 0), upscale=1.0):
    if upscale != 1.0:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_LINEAR)
    # webcam faces fill the frame, so 320 is enough there; classroom crops keep 640 when large
    size = (320, 320) if (max(img.shape[:2]) <= 480 or _args.domain != "ouc") else (640, 640)
    bb, _ = _det.detect(img, input_size=size, max_num=0)
    res = []
    for b in bb:
        x1, y1, x2, y2, sc = b
        res.append((x1 / upscale + offset[0], y1 / upscale + offset[1], x2 / upscale + offset[0], y2 / upscale + offset[1], sc))
    return res


def fill_boxes(boxes):
    """nearest-neighbour fill of missing per-frame boxes; None if none found"""
    found = [i for i, b in enumerate(boxes) if b is not None]
    if not found:
        return None
    return [boxes[min(found, key=lambda j: abs(j - i))] for i in range(len(boxes))]


def process(item):
    path, key = item
    try:
        cap = open_video(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        n = max(n, T * STEP + 1)
        centres = [int(n * c) for c in (1 / 6, 3 / 6, 5 / 6)]
        win = [[min(max(c - T * STEP // 2 + t * STEP, 0), n - 1) for t in range(T)] for c in centres]
        frames = read_frames(path, [i for w in win for i in w] + [n // 2])
        if not frames:
            return key, None, "no frames"
        get = lambda i: frames.get(i, frames[min(frames, key=lambda j: abs(j - i))])
        mid = get(n // 2); H, W = mid.shape[:2]
        views = _args.views
        # person boxes on the middle frame
        yr = _yolo.predict(mid, classes=[0], conf=0.35, imgsz=960, verbose=False, device="cpu")[0]
        torch.set_num_threads(1)  # ultralytics raises the torch thread count on first predict
        pboxes = [tuple(map(float, b)) for b in yr.boxes.xyxy.cpu().numpy()] if len(yr.boxes) else []
        pboxes = [b for b in pboxes if b[3] - b[1] >= 64]  # drop boxes too small to hold a usable face
        if _args.domain == "ouc":
            # groups are 7 to 9 students; keep the largest boxes and cap the memory a clip can cost
            pboxes = sorted(pboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:_args.max_persons]
        else:
            pboxes = sorted(pboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:1]
            if not pboxes:
                pboxes = [(0.0, 0.0, float(W), float(H))]
        n_inst = len(pboxes)
        crops = {v: np.zeros((n_inst, 3, T, S, S, 3), np.uint8) for v in views}
        face_found = np.zeros((n_inst, 3), bool); face_h = np.full((n_inst, 3), np.nan, np.float32)
        for wi, w in enumerate(win):
            fr = [get(i) for i in w]
            for pi, pb in enumerate(pboxes):
                fboxes = []
                for img in fr[::4]:  # every domain: detect on every 4th frame and hold the box
                    if _args.domain == "ouc":
                        x1, y1, x2, y2 = [int(v) for v in pb]
                        sub = img[max(y1, 0):y2, max(x1, 0):x2]
                        up = 2.0 if sub.shape[0] < 240 else 1.0
                        fs = faces_in(sub, (max(x1, 0), max(y1, 0)), up) if sub.size else []
                    else:
                        fs = faces_in(img)
                    fs = sorted(fs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) * b[4], reverse=True)
                    fboxes.append(fs[0][:4] if fs else None)
                fboxes = [fboxes[t // 4] for t in range(T)]
                fb = fill_boxes(fboxes)
                if fb is not None:
                    face_found[pi, wi] = True
                    med = np.median(np.array(fb), 0)  # smooth: median box over the window
                    face_h[pi, wi] = med[3] - med[1]
                for t, img in enumerate(fr):
                    if "face" in views and fb is not None:
                        crops["face"][pi, wi, t] = sq_crop(img, med, 1.3)
                    if "person" in views:
                        crops["person"][pi, wi, t] = sq_crop(img, pb, 1.05)
                    if "full" in views:
                        crops["full"][pi, wi, t] = letterbox(img)
                    if "face_lr" in views and fb is not None:
                        c = sq_crop(img, med, 1.3, size=S)
                        tgt = float(_lr_h[zlib.crc32(key.encode()) % len(_lr_h)])  # target face height (px) drawn from OUC-CGE
                        side = max(int(round(tgt * 1.3)), 4)
                        c = cv2.resize(cv2.resize(c, (side, side), interpolation=cv2.INTER_AREA), (S, S), interpolation=cv2.INTER_LINEAR)
                        crops["face_lr"][pi, wi, t] = c
        return key, dict(crops=crops, face_found=face_found, face_h=face_h, pboxes=np.array(pboxes, np.float32).reshape(-1, 4), frame_hw=(H, W)), None
    except Exception:
        return key, None, traceback.format_exc()


def load_marlin():
    """MARLIN ViT-L (YTF) from the HF mirror files, loaded without remote-code execution.
    Attention is swapped for torch SDPA, which computes the same softmax attention faster."""
    import json
    import torch.nn.functional as F
    sys.path.insert(0, os.path.join(ROOT, "weights"))
    from marlin_hf import modules as mm
    from marlin_hf.marlin import MarlinModel
    from marlin_hf.config import MarlinConfig
    from safetensors.torch import load_file

    def sdpa_forward(self, x):
        B, N, C = x.shape
        bias = None
        if self.q_bias is not None:
            bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=self.scale)
        return self.proj_drop(self.proj(y.transpose(1, 2).reshape(B, N, -1)))

    mm.Attention.forward = sdpa_forward
    cfg = json.load(open(os.path.join(ROOT, "weights/marlin_hf/config.json")))
    cfg = MarlinConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "auto_map", "torch_dtype", "transformers_version", "model_type")})
    m = MarlinModel(cfg)
    m.load_state_dict(load_file(os.path.join(ROOT, "weights/marlin_hf/model.safetensors")), strict=True)
    return m


class Encoders:
    def __init__(self, names):
        self.m = {}
        if "marlin" in names:
            m = load_marlin()
            self.m["marlin"] = m.to("cuda").eval().half()
        if "videomae" in names:
            os.environ["HF_HOME"] = os.path.expanduser("~/p2/hf")
            from transformers import VideoMAEModel
            self.m["videomae"] = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics").to("cuda").eval().half()

    @torch.no_grad()
    def __call__(self, name, clips):  # clips uint8 [B,T,H,W,3] BGR
        x = torch.from_numpy(clips).to("cuda").flip(-1).float() / 255.0
        if name == "marlin":
            x = x.permute(0, 4, 1, 2, 3).half()  # B,C,T,H,W, MARLIN expects [0,1]
            f = self.m[name](x, keep_seq=False)  # token mean then final LayerNorm, as in MARLIN
            return f.float().cpu().numpy()
        x = (x - torch.tensor(MEAN, device="cuda")) / torch.tensor(STD, device="cuda")
        x = x.permute(0, 1, 4, 2, 3).half()  # B,T,C,H,W
        return self.m[name](pixel_values=x).last_hidden_state.float().mean(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=["engagenet", "ouc", "daisee"])
    ap.add_argument("listfile"); ap.add_argument("outdir")
    ap.add_argument("--views", default="face,person")
    ap.add_argument("--encoders", default="marlin,videomae")
    ap.add_argument("--lr-heights", default=None)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--max-persons", type=int, default=12)
    a = ap.parse_args(); a.views = a.views.split(","); encs = a.encoders.split(",")
    os.makedirs(a.outdir, exist_ok=True)
    items = []
    for line in open(a.listfile):
        p = line.strip()
        if p:
            key = os.path.splitext(os.path.basename(p))[0] if a.domain != "ouc" else p.replace("/", "__")[:-4]
            if not os.path.exists(os.path.join(a.outdir, key + ".npz")):
                items.append((p, key))
    print("todo", len(items), flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")  # workers must not inherit a CUDA context
    with ctx.Pool(a.workers, initializer=init_worker, initargs=(a,)) as pool:
        E = Encoders(encs); t0 = time.time(); done = 0; errs = 0
        tw = te = 0.0; tl = time.time()
        buf = []  # (key, res) waiting for a GPU batch

        def flush():
            nonlocal done, te
            if not buf:
                return
            t1 = time.time()
            for v in a.views:
                arrs = [r["crops"][v].reshape(-1, T, S, S, 3) for _, r in buf]
                flat = np.concatenate(arrs) if arrs else np.zeros((0, T, S, S, 3), np.uint8)
                for e in encs:
                    feats = np.concatenate([E(e, flat[s:s + a.bs]) for s in range(0, len(flat), a.bs)]) if len(flat) else np.zeros((0, 1))
                    off = 0
                    for (_, r), arr in zip(buf, arrs):
                        r.setdefault("feats", {})[f"{e}__{v}"] = feats[off:off + len(arr)].reshape(arr.shape[0] // 3, 3, -1).astype(np.float16); off += len(arr)
            te += time.time() - t1
            for key, r in buf:
                out = dict(face_found=r["face_found"], face_h=r["face_h"], pboxes=r["pboxes"], frame_hw=np.array(r["frame_hw"]), **r["feats"])
                np.savez(os.path.join(a.outdir, key + ".npz"), **out)
                done += 1
                if done % 100 == 0:
                    print(f"done {done}/{len(items)} err {errs} {done / (time.time() - t0):.2f} clip/s wait {tw:.0f}s gpu {te:.0f}s", flush=True)
            buf.clear()

        import threading
        sem = threading.Semaphore(a.workers * 2)  # results in flight; without this the pool queue grows unbounded

        def feed():
            for it in items:
                sem.acquire()
                yield it

        for key, res, err in pool.imap_unordered(process, feed(), chunksize=1):
            sem.release()
            tw += time.time() - tl
            if res is None:
                errs += 1; print("ERR", key, err.strip().splitlines()[-1] if err else "", flush=True)
            else:
                buf.append((key, res))
                if sum(r["crops"][a.views[0]].shape[0] * 3 for _, r in buf) >= a.bs:
                    flush()
            tl = time.time()
        flush()
    print("EXTRACT_DONE", done, "errors", errs, flush=True)


if __name__ == "__main__":
    main()
