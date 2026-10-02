import numpy as np, pandas as pd, os, cv2, json
from multiprocessing import Pool
R = os.path.expanduser("~/engagenet_p2_crossdomain"); st = pd.read_csv(R + "/results/E0/ouc_structure.csv")
def seek_ok(rel):
    cap = cv2.VideoCapture(os.path.join(R, "data/OUC-CGE/clips", rel)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); bad = 0
    for i in (0, max(n // 2, 0), max(n - 2, 0)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, _ = cap.read(); bad += (not ok)
    cap.release(); return bad
if __name__ == "__main__":
    with Pool(24) as p: bad = np.array(p.map(seek_ok, st["path"], chunksize=16))
    st["seek_bad"] = bad; a = st[(st.dup_keep == 1) & (st.width == 1920)]
    out = dict(all_files_with_failed_seek=int((bad > 0).sum()), analysis_set=int((a.seek_bad > 0).sum()), frames_failed_hist=np.bincount(bad, minlength=4).tolist(),
               vis63_of_affected=a[a.seek_bad > 0]["vis63"].value_counts().to_dict(), labels_of_affected=a[a.seek_bad > 0]["label"].value_counts().to_dict())
    json.dump(out, open(R + "/results/REV_seek_failures.json", "w"), indent=1, default=int); print(out)
