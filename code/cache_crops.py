"""Cache person-box crops for end-to-end fine-tuning.

The frozen-feature pipeline cannot adapt the encoder, which every review flagged as the
binding constraint. This writes, for each classroom clip in the primary analysis set, one
window of 16 frames per detected person, as 112x112 uint8 crops, so a LoRA fine-tune can
read them without decoding video again.

Layout: crops/<clip key>.npz with crops [n_persons, 16, 112, 112, 3] and boxes [n_persons, 4].
usage: cache_crops.py LISTFILE OUTDIR [--workers 16] [--size 112]
"""
import argparse, os, sys
import numpy as np, cv2
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import extract_feats as E

ap = argparse.ArgumentParser()
ap.add_argument("listfile"); ap.add_argument("outdir")
ap.add_argument("--workers", type=int, default=16); ap.add_argument("--size", type=int, default=112)
ap.add_argument("--max-persons", type=int, default=12)
a = ap.parse_args()
T, STEP = 16, 2


class Args:  # what extract_feats' worker initialiser expects
    views = ["person"]; domain = "ouc"; lr_heights = None; max_persons = 12


def work(item):
    path, key, out = item
    try:
        cap = E.open_video(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        n = max(n, T * STEP + 1)
        centre = n // 2
        idxs = [min(max(centre - T * STEP // 2 + t * STEP, 0), n - 1) for t in range(T)]
        frames = E.read_frames(path, idxs + [centre])
        if not frames:
            return key, 0, "no frames"
        get = lambda i: frames.get(i, frames[min(frames, key=lambda j: abs(j - i))])
        mid = get(centre)
        yr = E._yolo.predict(mid, classes=[0], conf=0.35, imgsz=960, verbose=False, device="cpu")[0]
        boxes = [tuple(map(float, b)) for b in yr.boxes.xyxy.cpu().numpy()] if len(yr.boxes) else []
        boxes = [b for b in boxes if b[3] - b[1] >= 64]
        boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)[:a.max_persons]
        if not boxes:
            return key, 0, "no persons"
        crops = np.zeros((len(boxes), T, a.size, a.size, 3), np.uint8)
        for t, i in enumerate(idxs):
            img = get(i)
            for pi, b in enumerate(boxes):
                crops[pi, t] = E.sq_crop(img, b, 1.05, size=a.size)
        np.savez_compressed(os.path.join(out, key + ".npz"), crops=crops, boxes=np.array(boxes, np.float32))
        return key, len(boxes), None
    except Exception as ex:
        return key, 0, repr(ex)[:120]


if __name__ == "__main__":
    os.makedirs(a.outdir, exist_ok=True)
    items = []
    for line in open(a.listfile):
        p = line.strip()
        if not p:
            continue
        key = p.replace("/", "__")[:-4]
        if not os.path.exists(os.path.join(a.outdir, key + ".npz")):
            items.append((p, key, a.outdir))
    print("todo", len(items), flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    done = err = 0
    with ctx.Pool(a.workers, initializer=E.init_worker, initargs=(Args(),)) as pool:
        for key, n, e in pool.imap_unordered(work, items, chunksize=4):
            done += 1; err += e is not None
            if e is not None and err < 20:
                print("ERR", key, e, flush=True)
            if done % 500 == 0:
                print(f"cached {done}/{len(items)} errors {err}", flush=True)
    print("CACHE_DONE", done, "errors", err, flush=True)
