"""Frame count of every EngageNet clip in the face cache, and the share of clips shorter than each window span the
paper uses, which is the share that takes the short-clip branch of the sampler (ft_video.py: `if n < span`).

The cache holds 11,311 clips, 105 more than the 11,206 the runs use; both counts are reported. Read-only.

usage (on the experiment server, PYTHONPATH=pylib:code): clip_lengths.py   prints JSON; results/CLIP_BRANCH_SHARES.json
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels

files = sorted(glob.glob(os.path.join(ROOT, "crops_face", "engagenet", "*.npz")))
n = {os.path.basename(f)[:-4]: len(np.load(f)["off"]) - 1 for f in files}
en = engagenet_labels()
used = set(en.key[en.split.isin(["train", "val", "test"])]) & set(n)
SPANS = {"inherited recipe, 16 at 4": 61, "B, 16 at 9": 136, "C, 32 at 9": 280, "D, 48 at 6": 283, "A, 16 at 19": 286,
         "CMS, 16 at 20": 301}
out = {"clips_in_cache": len(n), "clips_used": len(used), "shares_shorter_than_span": {}}
for name, s in SPANS.items():
    out["shares_shorter_than_span"][name] = {
        "span": s, "cache": float(np.mean([v < s for v in n.values()])),
        "used": float(np.mean([n[k] < s for k in used]))}
print(json.dumps(out, indent=1))
