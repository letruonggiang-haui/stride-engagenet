"""300-frame landmark sequences for DAiSEE, reusing geom96.py's worker. A real file (not stdin) because the spawn
context re-imports the main module in every worker. usage: seq300_daisee.py"""
import os, sys, glob, time
sys.argv = ["geom96.py", "train", os.path.expanduser("~/engagenet_p2_crossdomain/results/geom300_daisee"), "--frames", "300", "--workers", "20", "--detect-every", "8", "--save-seq"]
sys.path.insert(0, os.path.expanduser("~/engagenet_p2_crossdomain/code"))
import geom96 as G

if __name__ == "__main__":
    from multiprocessing import get_context
    paths = sorted(glob.glob(os.path.expanduser("~/daisee/DAiSEE/DataSet/*/*/*/*.avi")) + glob.glob(os.path.expanduser("~/daisee/DAiSEE/DataSet/*/*/*/*.mp4")))
    done = {os.path.basename(p)[:-4] for p in glob.glob(G.SEQDIR + "/*.npz")}
    items = [(os.path.basename(p)[:-4], p) for p in paths if os.path.basename(p)[:-4] not in done]
    print(f"daisee: {len(items)} clips to extract at 300 frames ({len(done)} already done)", flush=True)
    t0 = time.time(); bad = 0
    with get_context("spawn").Pool(20, initializer=G.init) as pool:
        for j, (key, v, n, total, err) in enumerate(pool.imap_unordered(G.work, items, chunksize=2), 1):
            bad += err is not None
            if j % 200 == 0: print(f"{j}/{len(items)} bad {bad} {time.time()-t0:.0f}s", flush=True)
    print(f"SEQ300_DAISEE_DONE {len(items)} bad {bad}", flush=True)
