"""Full-temporal facial geometry: the 1338-d descriptor of the sibling project recomputed at 96 sampled frames
(~9.6 Hz instead of 1.6 Hz), plus a dynamics block of statistics that are UNDEFINED at 16 frames.

Why: the pixel stream gained +5.1 macro-F1 from whole-clip coverage, and the strongest non-PriorNet entry on this
benchmark (Ordinal ST-GCN, 71.24) is landmark-only over 300 frames. The existing geometry cache samples exactly 16
frames, so a blink (200-400 ms) falls between samples and pose variance is estimated from 16 points. Simply raising
the frame count while keeping the same mean/std/min/max summarisation would only de-noise the same quantities; the
gain has to come from the dynamics block below. Pre-registered keep threshold: +0.015 accuracy over the 16-frame
geometry baseline of 0.6712 on inner CV fold 0 (see MODEL_PAPER_DESIGN amendment 5).

Base features are produced by importing the sibling project's own frame_descriptor / temporal_stats, so the first
1338 dimensions are directly comparable to the cached 16-frame vector.

usage: geom96.py SPLIT OUTDIR [--frames 96] [--workers 20] [--pilot-keys FILE]
"""
import argparse, os, sys, time
import numpy as np, cv2

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # the GPU is held by training; landmarks run on CPU
# the geometry descriptor now ships with this deposit as geom_extract_min.py, a
# verbatim extract of the sibling project's module, so no sibling path is needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("split", choices=["train", "val", "test"]); ap.add_argument("outdir")
ap.add_argument("--frames", type=int, default=96); ap.add_argument("--workers", type=int, default=20)
ap.add_argument("--detect-every", type=int, default=8)
ap.add_argument("--pilot-keys", default="", help="file of clip keys, one per line; only these are extracted")
ap.add_argument("--save-seq", action="store_true", help="also write the raw per-frame sequence [T, 204+19] and a found mask, for temporal models")
a = ap.parse_args()
VIDEO_DIR = {"train": "Train", "val": "Validation", "test": "Test"}
FPS = 30.0

_G = None


def helpers():
    """Import the geometry module once per worker and force CPU inference."""
    global _G
    if _G is None:
        import onnxruntime as ort
        ort.get_available_providers  # noqa
        import geom_extract_min as G
        G.PROVIDERS = ["CPUExecutionProvider"]
        G.THREADS = 1
        _G = G
    return _G


def dynamics(scalars, frames_sampled, clip_frames):
    """Statistics of the per-frame scalar channels that 16 samples cannot express.

    scalars: [T, C] per-frame channels (16 interpretable scalars + yaw/pitch/roll).
    Sampling rate of the series, in Hz, is frames_sampled / (clip_frames / FPS).
    """
    x = np.asarray(scalars, dtype=np.float64)
    T, C = x.shape
    dur = max(clip_frames / FPS, 1e-3)
    fs = frames_sampled / dur                      # effective sampling rate of the series
    if T < 8:
        return np.zeros(8 * C, dtype=np.float32)
    mu, sd = x.mean(0), x.std(0) + 1e-8
    z = (x - mu) / sd
    d = np.diff(z, axis=0)
    ad = np.abs(d)
    med = np.median(ad, axis=0) + 1e-8

    # 1-2. dip rate and mean dip depth: a local minimum below -0.5 sd. On the eye-aspect-ratio channels this is a
    # blink; on pose channels it is a downward excursion. Only resolvable once samples are ~0.1 s apart.
    dip_rate = np.zeros(C); dip_depth = np.zeros(C)
    for c in range(C):
        s = z[:, c]
        loc = (s[1:-1] < s[:-2]) & (s[1:-1] < s[2:]) & (s[1:-1] < -0.5)
        n = int(loc.sum()); dip_rate[c] = n / dur
        dip_depth[c] = float(-s[1:-1][loc].mean()) if n else 0.0

    # 3-4. spectral shape: low-band (<1 Hz, posture) against high-band (>1 Hz, micro-movement), and centroid
    W = np.hanning(T)[:, None]
    P = np.abs(np.fft.rfft(z * W, axis=0)) ** 2
    f = np.fft.rfftfreq(T, d=1.0 / fs)
    lo = (f > 0) & (f <= 1.0); hi = f > 1.0
    plo = P[lo].sum(0) + 1e-8; phi = P[hi].sum(0) + 1e-8
    band_ratio = phi / plo
    centroid = (P[1:] * f[1:, None]).sum(0) / (P[1:].sum(0) + 1e-8)

    # 5-6. burstiness of motion and the longest quiet stretch, in seconds
    frac_fast = (ad > 2.0 * med).mean(0)
    still = np.zeros(C)
    for c in range(C):
        q = ad[:, c] < med[c]
        best = cur = 0
        for v in q:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        still[c] = best / fs

    # 7-8. autocorrelation at about 0.5 s and 1.0 s: how persistent the state is
    def ac(lag):
        if lag < 1 or lag >= T: return np.zeros(C)
        A, B = z[:-lag], z[lag:]
        return (A * B).mean(0) / (A.std(0) * B.std(0) + 1e-8)
    ac05, ac10 = ac(int(round(0.5 * fs))), ac(int(round(1.0 * fs)))
    return np.concatenate([dip_rate, dip_depth, band_ratio, centroid, frac_fast, still, ac05, ac10]).astype(np.float32)


def clip_vector(path):
    """Base 1338-d descriptor at `--frames` samples, concatenated with the dynamics block."""
    G = helpers()
    from insightface.app.common import Face
    det, lmk = _MODELS
    cap = cv2.VideoCapture(path); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None, 0, 0, None
    want = set(np.linspace(0, total - 1, a.frames).astype(int).tolist())
    frames, i = [], 0
    while cap.grab():
        if i in want:
            ok, f = cap.retrieve()
            if ok: frames.append(f)
        i += 1
    cap.release()
    shapes, scalars = [], []
    bbox, kps, h_img, w_img = None, None, None, None
    for fi, f in enumerate(frames):
        if h_img is None: h_img, w_img = f.shape[:2]
        if bbox is None or (fi % a.detect_every == 0):
            bb, kk = det.detect(f, input_size=G.DET_SIZE)
            if bb is not None and len(bb):
                j = int(np.argmax((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])))
                bbox, kps = bb[j, :4].copy(), kk[j]
            elif bbox is None:
                continue
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        loose = np.array([max(0, bbox[0] - 0.15 * bw), max(0, bbox[1] - 0.15 * bh),
                          min(w_img - 1, bbox[2] + 0.15 * bw), min(h_img - 1, bbox[3] + 0.15 * bh)])
        face = Face(bbox=loose, kps=kps, det_score=1.0)
        try:
            lmk.get(f, face)
        except Exception:
            bbox = None; continue
        pts = face.landmark_3d_68
        if pts is None: bbox = None; continue
        io = np.linalg.norm(pts[G.EYE_R].mean(0) - pts[G.EYE_L].mean(0))
        if not np.isfinite(io) or io < 4.0: bbox = None; continue
        sh, sc = G.frame_descriptor(pts, loose, f.shape)
        shapes.append(sh); scalars.append(sc)
        bbox = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    if not shapes:
        return None, 0, total, None
    base = np.concatenate([G.temporal_stats(shapes), G.temporal_stats(scalars)])
    dyn = dynamics(np.stack(scalars), len(shapes), total)
    seq = np.concatenate([np.stack(shapes), np.stack(scalars)], 1).astype(np.float32) if a.save_seq else None
    return np.concatenate([base, dyn]).astype(np.float32), len(shapes), total, seq


_MODELS = None
SEQDIR = os.path.join(a.outdir, f"seq{a.frames}")
if a.save_seq: os.makedirs(SEQDIR, exist_ok=True)


def init():
    global _MODELS
    cv2.setNumThreads(1)
    G = helpers()
    _MODELS = G.build_models()


def work(item):
    key, path = item
    try:
        v, n, total, seq = clip_vector(path)
        if seq is not None:   # one file per clip, so a killed run never leaves a corrupt monolith
            np.savez_compressed(os.path.join(SEQDIR, key + ".npz"), seq=seq, n=np.int16(n), clip_frames=np.int32(total))
        return key, v, n, total, None
    except Exception as ex:
        return key, None, 0, 0, repr(ex)[:140]


if __name__ == "__main__":
    from multiprocessing import get_context
    import pandas as pd
    sys.path.insert(0, os.path.expanduser("~/engagenet_p2_crossdomain/code"))
    from p2x_lib import engagenet_labels
    en = engagenet_labels()
    d = en[en.split == a.split].sort_values(["subject", "session", "video", "idx"]).reset_index(drop=True)
    if a.pilot_keys:
        want = {l.strip() for l in open(a.pilot_keys) if l.strip()}
        d = d[d.key.isin(want)].reset_index(drop=True)
    vdir = os.path.expanduser(f"~/engagenet/videos")
    tdir = os.path.expanduser("~/engagenet_p2_crossdomain/data/engagenet_test_videos")
    def pathof(k):
        p = os.path.join(vdir, k + ".mp4")
        return p if os.path.exists(p) else os.path.join(tdir, k + ".mp4")
    items = [(k, pathof(k)) for k in d.key]
    os.makedirs(a.outdir, exist_ok=True)
    print(f"{a.split}: {len(items)} clips at {a.frames} frames, detect every {a.detect_every}", flush=True)
    V, NF, TOT, KEYS = [], [], [], []
    t0 = time.time(); bad = 0; dim = None
    with get_context("spawn").Pool(a.workers, initializer=init) as pool:
        for j, (key, v, n, total, err) in enumerate(pool.imap(work, items, chunksize=2), 1):
            if err: bad += 1; print("ERR", key, err, flush=True)
            if v is None:
                v = np.zeros(dim, np.float32) if dim else None
            elif dim is None:
                dim = len(v)
            V.append(v); NF.append(n); TOT.append(total); KEYS.append(key)
            if j % 100 == 0: print(f"{j}/{len(items)} bad {bad} {time.time()-t0:.0f}s", flush=True)
    V = [np.zeros(dim, np.float32) if v is None else v for v in V]
    suff = f"{a.frames}" + ("_pilot" if a.pilot_keys else "")
    np.savez_compressed(os.path.join(a.outdir, f"geom{suff}_{a.split}.npz"),
                        X=np.stack(V), nframes=np.array(NF, np.int16), clip_frames=np.array(TOT, np.int32), keys=np.array(KEYS))
    print(f"GEOM96_DONE {a.split} n={len(V)} dim={dim} bad={bad} {time.time()-t0:.0f}s", flush=True)
