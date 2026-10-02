"""G0 of MODEL_PAPER_DESIGN_2026-09-21.md: per-frame face crops for fine-tuning video backbones.

For every clip of EngageNet (train/val/test) and DAiSEE, the largest face is detected with SCRFD on every fourth frame,
boxes are interpolated between detections and held at the ends, and a square crop of 1.3 times the box is written for
every frame at 224x224 as JPEG. Frames with no usable detection get found=False and a crop from the last known box
(or a black frame when the clip has no detection at all), so the training code can implement missing-face placeholders.

Layout: <outdir>/<domain>/<key>.npz with jpg (uint8, concatenated JPEG bytes), off (int64 offsets, len n+1),
found (bool, n), box (float32, n x 4), fps, hw.
usage: cache_faces.py {engagenet,daisee} OUTDIR [--workers 40] [--size 224] [--every 4]
"""
import argparse, glob, os, sys, time, zipfile
import numpy as np, cv2
from multiprocessing import get_context
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_feats as E

ap = argparse.ArgumentParser()
ap.add_argument("domain", choices=["engagenet", "daisee"]); ap.add_argument("outdir")
ap.add_argument("--workers", type=int, default=24); ap.add_argument("--size", type=int, default=224); ap.add_argument("--every", type=int, default=4)
a = ap.parse_args()


class Args:  # what extract_feats.init_worker expects
    views = ["face"]; domain = "engagenet"; lr_heights = None; max_persons = 1


def interpolate(boxes):
    """boxes: list (one per frame) of 4-tuples or None -> float array n x 4 and found mask; linear between detections, held at ends."""
    n = len(boxes); idx = [i for i, b in enumerate(boxes) if b is not None]
    if not idx:
        return None, np.zeros(n, bool)
    out = np.zeros((n, 4), np.float32); found = np.zeros(n, bool)
    for i in range(n):
        j = max([k for k in idx if k <= i], default=None); k = min([k for k in idx if k >= i], default=None)
        if j is None: out[i] = boxes[k]
        elif k is None: out[i] = boxes[j]
        elif j == k: out[i] = boxes[j]
        else:
            w = (i - j) / (k - j); out[i] = (1 - w) * np.array(boxes[j]) + w * np.array(boxes[k])
        found[i] = (j is not None and i - j < a.every) or (k is not None and k - i < a.every)
    return out, found


def openv(path):   # one decoder thread and a 20 s timeout, so a broken file cannot hold a worker for ten minutes
    return cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1, cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 20000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 20000])


def work(item):
    """Two passes over the clip so that no worker ever holds more than one decoded frame: pass 1 detects on every
    fourth frame and keeps only boxes, pass 2 decodes again and writes the crops."""
    path, key, out = item
    dst = os.path.join(out, key + ".npz")
    if os.path.exists(dst) and zipfile.is_zipfile(dst):
        return key, -1, None
    try:
        cap = openv(path); fps = cap.get(cv2.CAP_PROP_FPS); nf = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 14 EngageNet subjects (1,032 clips) are encoded at a nominal 1,000 fps with every real frame repeated ~33 times;
        # keep every k-th frame so that all clips are handled at about 30 fps (same rule as the sibling project's decimate_1000fps.py)
        k = max(1, int(round(fps / 30.0))) if fps > 100 else (max(1, int(round(nf / 300.0))) if nf > 1500 else 1)
        dets = []; H = W = 0; i = 0
        while cap.grab():
            if i % k == 0:
                ok, fr = cap.retrieve()
                if ok:
                    j = len(dets); H, W = fr.shape[:2]
                    if j % a.every == 0:
                        fs = E.faces_in(fr); fs = sorted(fs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]) * b[4], reverse=True)
                        dets.append(tuple(fs[0][:4]) if fs else None)
                    else:
                        dets.append(None)
            i += 1
        cap.release()
        if not dets:
            return key, 0, "no frames"
        box, found = interpolate(dets)
        cap = openv(path); bufs = []; i = 0; j = 0
        while cap.grab() and j < len(dets):
            if i % k == 0:
                ok, fr = cap.retrieve()
                if ok:
                    c = E.sq_crop(fr, box[j], 1.3, size=a.size) if box is not None else np.zeros((a.size, a.size, 3), np.uint8)
                    ok2, enc = cv2.imencode(".jpg", c, [cv2.IMWRITE_JPEG_QUALITY, 90]); bufs.append(enc.reshape(-1)); j += 1
            i += 1
        cap.release()
        fps = fps / k
        n = len(bufs); box = box[:n] if box is not None else np.zeros((n, 4), np.float32); found = found[:n]
        off = np.zeros(n + 1, np.int64); off[1:] = np.cumsum([len(b) for b in bufs])
        tmp = dst + ".tmp.npz"; np.savez(tmp, jpg=np.concatenate(bufs), off=off, found=found, box=box, fps=np.float32(fps), hw=np.array([H, W], np.int32)); os.replace(tmp, dst)
        return key, int(found.sum()), None
    except Exception as ex:
        return key, 0, repr(ex)[:160]


if __name__ == "__main__":
    out = os.path.join(a.outdir, a.domain); os.makedirs(out, exist_ok=True)
    if a.domain == "engagenet":
        paths = sorted(glob.glob(os.path.expanduser("~/engagenet/videos/*.mp4")) + glob.glob(os.path.join(E.ROOT, "data/engagenet_test_videos/*.mp4")))
        items = [(p, os.path.basename(p)[:-4], out) for p in paths]
    else:
        # 652 DAiSEE clips (506 train, 146 test) ship as .mp4, not .avi; both containers are used
        paths = sorted(glob.glob(os.path.expanduser("~/daisee/DAiSEE/DataSet/*/*/*/*.avi")) + glob.glob(os.path.expanduser("~/daisee/DAiSEE/DataSet/*/*/*/*.mp4")))
        items = [(p, os.path.basename(p)[:-4], out) for p in paths]
    print(a.domain, len(items), "clips", flush=True)
    t0 = time.time(); done = 0; bad = 0
    with get_context("spawn").Pool(a.workers, initializer=E.init_worker, initargs=(Args(),)) as pool:
        for key, nf, err in pool.imap_unordered(work, items, chunksize=4):
            done += 1
            if done % 50 == 0:   # memory guard: the first run of this script took the machine down at 36 workers
                mi = dict(l.split(":")[:2] for l in open("/proc/meminfo")); avail = int(mi["MemAvailable"].split()[0]) // 1024
                if avail < 30000: print(f"LOW MEMORY {avail} MB available, sleeping", flush=True); time.sleep(120)
            if err: bad += 1; print("ERR", key, err, flush=True)
            if done % 200 == 0: print(f"{done}/{len(items)} bad {bad} {time.time() - t0:.0f}s", flush=True)
    print(f"CACHE_FACES_DONE {a.domain} {done} bad {bad}", flush=True)
