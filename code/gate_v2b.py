"""Gate G-B of MODEL_V2_DESIGN_2026-09-27: does combining streams CONDITIONALLY on observable, label-free
signals beat combining them with a fixed weight?

Motivation, all measured in v1: the fixed equal-weight fusion gained -0.0139 / -0.0310 / +0.0340 across the three
inner folds with every paired interval containing zero; the landmark-sequence stream's errors correlate with the
pixel stream's at only 0.38-0.55 against 0.79 among pixel members; and 13% of EngageNet clips contain frames with
no detected face. A constant weight cannot be right both for a clean clip and for one where the sensor failed.

Three streams, all out-of-fold on the same three subject-disjoint inner folds:
  pixel      the v2 control, SVFAP-B + LoRA-16, 16 frames stride 19 (preds/ft_v2_T16_f<k>_train.npz)
  geometry   the 1338-d full-clip descriptor + HistGradientBoosting, refitted here per fold
  sequence   the 300-frame landmark transformer (preds/seq_tfm_fold<k>.npz)

Router: softmax over the three stream weights from a small MLP on a reliability vector. Two pre-declared variants:
  per-clip      10 features describing this clip's observability only            <- HEADLINE
  transductive  + 6 subject-level aggregates of the same signals                 <- declared secondary
The per-clip variant is the headline because the published comparators score each clip independently; the
transductive variant uses other (unlabelled) clips of the same subject and is therefore a different task, which is
why it is reported separately rather than mixed in.

Pre-registered gate: the router must beat the best single stream by >= +0.015 macro-F1 averaged over the three folds
AND beat the best fixed-weight fusion by >= +0.010. Selection metric is macro-F1 (primary) with UAR (secondary).

Leakage, stated plainly: to score fold k the router is fitted on the out-of-fold predictions of the other two folds.
Those predictions come from stream models that were themselves trained on data including fold k, so the inner-fold
estimate here is slightly optimistic. The official protocol has no such path -- the router would be fitted on
out-of-fold training predictions, frozen, and only then applied to a split no model or router has seen.

usage: gate_v2b.py [--epochs 300] [--seed 0]
"""
import argparse, glob, json, os, sys, zipfile
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT, engagenet_labels
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--geom-cache", default=os.path.expanduser("~/engagenet/experiments/cache"))
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
K = 4
CR = os.path.join(ROOT, "crops_face/engagenet")
THRESH_SINGLE, THRESH_FIXED = 0.015, 0.010

# ---------------------------------------------------------------- clips, folds (identical to ft_video.py)
en = engagenet_labels()
tr = en[en.split == "train"].sort_values(["subject", "session", "video", "idx"]).reset_index(drop=True)
tr["gi"] = np.arange(len(tr))                                   # row index into the sibling project's geometry cache
Gall = np.load(os.path.join(a.geom_cache, "geom_train.npy"))
assert len(Gall) == len(tr), f"geometry cache {len(Gall)} rows vs {len(tr)} label rows"
tr = tr[tr.key.apply(lambda k: zipfile.is_zipfile(os.path.join(CR, k + ".npz")) if os.path.exists(os.path.join(CR, k + ".npz")) else False)].reset_index(drop=True)
subs = np.sort(tr.subject.unique()); rng = np.random.default_rng(12345); rng.shuffle(subs)
tr["fold"] = tr.subject.map({s: i % 3 for i, s in enumerate(subs)})
print(f"{len(tr)} cached train clips, folds {tr.fold.value_counts().sort_index().tolist()}", flush=True)


# ---------------------------------------------------------------- reliability features, from the face cache only
def reliability(keys):
    """Ten label-free per-clip signals describing how well the face was observed. All already in the cache."""
    out = np.zeros((len(keys), 10), np.float32)
    for i, k in enumerate(keys):
        z = np.load(os.path.join(CR, k + ".npz"))
        found = z["found"].astype(np.float32); box = z["box"].astype(np.float32)
        hw = z["hw"]; fps = float(z["fps"]); n = len(found)
        frame_area = float(hw[0]) * float(hw[1]) + 1e-6
        area = np.clip((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]), 0, None) / frame_area
        cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
        jit = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2) / (np.sqrt(frame_area) + 1e-6) if n > 1 else np.zeros(1)
        miss = 1.0 - found
        runs, cur = 0, 0
        for v in miss:
            cur = cur + 1 if v else 0; runs = max(runs, cur)
        edge = ((box[:, 0] <= 1) | (box[:, 1] <= 1) | (box[:, 2] >= hw[1] - 1) | (box[:, 3] >= hw[0] - 1)).mean()
        med = np.median(area[found > 0]) if found.any() else 0.0
        out[i] = [found.mean(), runs / max(n, 1), area.mean(), area.std(), jit.mean(),
                  float(np.percentile(jit, 95)), edge, fps / 1000.0, n / 300.0,
                  float((area < 0.5 * med).mean()) if med > 0 else 0.0]
    return out


RCACHE = os.path.join(ROOT, "results", "reliability_train.npz")
if os.path.exists(RCACHE):
    z = np.load(RCACHE, allow_pickle=True); rmap = dict(zip([str(k) for k in z["keys"]], z["R"]))
    if all(k in rmap for k in tr.key): R = np.stack([rmap[k] for k in tr.key]); print("reliability features from cache", flush=True)
    else: R = None
else: R = None
if R is None:
    print("computing reliability features from the face cache ...", flush=True)
    R = reliability(tr.key.values)
    np.savez_compressed(RCACHE, R=R, keys=tr.key.values); print(f"reliability features {R.shape} cached", flush=True)

# subject aggregates of the same signals: mean and std over the subject's clips, plus the subject's clip count
sdf = pd.DataFrame(R, columns=[f"r{i}" for i in range(10)]); sdf["subject"] = tr.subject.values
agg = sdf.groupby("subject").agg(["mean", "std"]).fillna(0.0)
SUB = np.stack([np.concatenate([agg.loc[s].values[:6], [np.log1p((tr.subject == s).sum())]]) for s in tr.subject.values]).astype(np.float32)
print(f"per-clip reliability {R.shape}, subject aggregates {SUB.shape}", flush=True)

y_all = tr.y4.values.astype(int)


# ---------------------------------------------------------------- the three streams, out-of-fold per fold
def load_npz(path, keys):
    if not os.path.exists(path): return None
    z = np.load(path, allow_pickle=True); pos = {str(k): i for i, k in enumerate(z["keys"])}
    P = np.full((len(keys), K), np.nan, np.float32)
    for i, k in enumerate(keys):
        if k in pos: P[i] = z["P"][pos[k]]
    return P


streams, missing_report = {}, {}
for fold in (0, 1, 2):
    m = tr.fold.values == fold; keys = tr.key.values[m]
    px = load_npz(os.path.join(ROOT, "preds", f"ft_v2_T16_f{fold}_train.npz"), keys)
    sq = load_npz(os.path.join(ROOT, "preds", f"seq_tfm_fold{fold}.npz"), keys)
    # geometry: refit per fold on the other two folds, exactly the locked recipe. Cached, because the fit is slow
    # (1338 features, 400 iterations, four classes) and it competes for cores with whatever is on the GPU.
    gcache = os.path.join(ROOT, "preds", f"geom_gbdt_fold{fold}.npz")
    if os.path.exists(gcache):
        z = np.load(gcache, allow_pickle=True); pos = {str(k): i for i, k in enumerate(z["keys"])}
        gm = np.stack([z["P"][pos[k]] for k in keys]).astype(np.float32); print(f"  fold {fold} geometry from cache", flush=True)
    else:
        trn = tr.fold.values != fold
        gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, random_state=0)
        gb.fit(np.nan_to_num(Gall[tr.gi.values[trn]]), y_all[trn])
        gm = gb.predict_proba(np.nan_to_num(Gall[tr.gi.values[m]])).astype(np.float32)
        np.savez_compressed(gcache, P=gm, keys=keys, y=y_all[m]); print(f"  fold {fold} geometry fitted and cached", flush=True)
    if px is None: sys.exit(f"missing pixel predictions for fold {fold}")
    if sq is None: print(f"  fold {fold}: no sequence predictions, that stream falls back to pixel", flush=True)
    miss_sq = np.isnan(sq[:, 0]) if sq is not None else np.ones(len(keys), bool)
    if sq is None: sq = px.copy()
    sq = np.where(miss_sq[:, None], px, sq)          # clips without a landmark sequence fall back to the pixel head
    missing_report[fold] = int(miss_sq.sum())
    streams[fold] = dict(keys=keys, idx=np.where(m)[0], y=y_all[m], pixel=px, geom=gm, seq=sq)
    for nm in ("pixel", "geom", "seq"):
        P = streams[fold][nm]
        print(f"  fold {fold} {nm:6s}: acc {accuracy_score(streams[fold]['y'], P.argmax(1)):.4f} "
              f"macroF1 {f1_score(streams[fold]['y'], P.argmax(1), average='macro'):.4f}", flush=True)
print(f"clips without a landmark sequence per fold: {missing_report}", flush=True)


# ---------------------------------------------------------------- router
class Router(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, 3))
    def forward(self, x): return torch.softmax(self.net(x), -1)


def score(y, P):
    yh = P.argmax(1)
    return dict(acc=accuracy_score(y, yh), uar=balanced_accuracy_score(y, yh), macro_f1=f1_score(y, yh, average="macro"))


def fit_router(feat_idx, train_folds, eval_fold):
    """feat_idx: which feature block to use. Fitted on the out-of-fold predictions of train_folds only."""
    def block(rows): return np.concatenate([R[rows]] + ([SUB[rows]] if feat_idx == "trans" else []), 1)
    Xtr = np.concatenate([block(streams[f]["idx"]) for f in train_folds])
    Ptr = np.stack([np.concatenate([streams[f][n] for f in train_folds]) for n in ("pixel", "geom", "seq")], 1)
    ytr = np.concatenate([streams[f]["y"] for f in train_folds])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xt = torch.tensor((Xtr - mu) / sd); Pt = torch.tensor(Ptr); yt = torch.tensor(ytr)
    net = Router(Xt.shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    for _ in range(a.epochs):
        w = net(Xt)                                        # [N,3]
        mix = (w.unsqueeze(-1) * Pt).sum(1).clamp_min(1e-8)
        loss = F.nll_loss(mix.log(), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    ev = streams[eval_fold]
    Xe = torch.tensor((block(ev["idx"]) - mu) / sd)
    Pe = torch.tensor(np.stack([ev[n] for n in ("pixel", "geom", "seq")], 1))
    with torch.no_grad():
        we = net(Xe); mixe = (we.unsqueeze(-1) * Pe).sum(1).numpy()
    return mixe, we.numpy(), float(loss.item())


out = {"design": "MODEL_V2_DESIGN_2026-09-27 gate G-B", "threshold_vs_best_single": THRESH_SINGLE,
       "threshold_vs_fixed": THRESH_FIXED, "clips_without_sequence": missing_report, "folds": {}}
for fold in (0, 1, 2):
    ev = streams[fold]; y = ev["y"]; others = [f for f in (0, 1, 2) if f != fold]
    row = {"n": int(len(y)), "single": {n: score(y, ev[n]) for n in ("pixel", "geom", "seq")}}
    row["fixed_thirds"] = score(y, (ev["pixel"] + ev["geom"] + ev["seq"]) / 3)
    row["fixed_w05_pixel_geom"] = score(y, 0.5 * ev["pixel"] + 0.5 * ev["geom"])
    for variant in ("perclip", "trans"):
        P, w, l = fit_router("perclip" if variant == "perclip" else "trans", others, fold)
        row[f"router_{variant}"] = score(y, P)
        row[f"router_{variant}_mean_weights"] = w.mean(0).tolist()
        row[f"router_{variant}_weight_std"] = w.std(0).tolist()
    out["folds"][fold] = row
    best_single = max(row["single"], key=lambda n: row["single"][n]["macro_f1"])
    print(f"\nfold {fold} (n={len(y)}):")
    for n in ("pixel", "geom", "seq"): print(f"   {n:20s} macroF1 {row['single'][n]['macro_f1']:.4f}  acc {row['single'][n]['acc']:.4f}  uar {row['single'][n]['uar']:.4f}")
    for n in ("fixed_thirds", "fixed_w05_pixel_geom", "router_perclip", "router_trans"):
        print(f"   {n:20s} macroF1 {row[n]['macro_f1']:.4f}  acc {row[n]['acc']:.4f}  uar {row[n]['uar']:.4f}")
    print(f"   best single stream on macro-F1: {best_single}; router mean weights (pixel,geom,seq) "
          f"per-clip {np.round(row['router_perclip_mean_weights'],3).tolist()} sd {np.round(row['router_perclip_weight_std'],3).tolist()}")

# ---------------------------------------------------------------- gate
def mean_of(key, sub=None):
    return float(np.mean([(out["folds"][f][key][sub] if sub else out["folds"][f][key]) for f in (0, 1, 2)]))


best_single_f1 = float(np.mean([max(out["folds"][f]["single"][n]["macro_f1"] for n in ("pixel", "geom", "seq")) for f in (0, 1, 2)]))
best_fixed_f1 = max(mean_of("fixed_thirds", "macro_f1"), mean_of("fixed_w05_pixel_geom", "macro_f1"))
out["summary"] = {"best_single_macro_f1_mean": best_single_f1, "best_fixed_macro_f1_mean": best_fixed_f1}
for variant in ("perclip", "trans"):
    r = mean_of(f"router_{variant}", "macro_f1")
    per_fold = [out["folds"][f][f"router_{variant}"]["macro_f1"] - max(out["folds"][f]["single"][n]["macro_f1"] for n in ("pixel", "geom", "seq")) for f in (0, 1, 2)]
    passed = bool(r - best_single_f1 >= THRESH_SINGLE and r - best_fixed_f1 >= THRESH_FIXED and sum(x >= 0 for x in per_fold) >= 2)
    out["summary"][f"router_{variant}"] = dict(macro_f1_mean=r, uar_mean=mean_of(f"router_{variant}", "uar"),
                                               acc_mean=mean_of(f"router_{variant}", "acc"),
                                               gain_vs_best_single=r - best_single_f1, gain_vs_best_fixed=r - best_fixed_f1,
                                               per_fold_gain_vs_best_single=per_fold, folds_nonneg=int(sum(x >= 0 for x in per_fold)),
                                               gate_pass=passed)
    print(f"\nrouter_{variant}: macro-F1 mean {r:.4f} | vs best single {r-best_single_f1:+.4f} (needs >= +{THRESH_SINGLE}) "
          f"| vs best fixed fusion {r-best_fixed_f1:+.4f} (needs >= +{THRESH_FIXED}) | non-negative on {sum(x>=0 for x in per_fold)}/3 -> "
          f"{'PASS' if passed else 'FAIL'}")

out["headline_variant"] = "perclip"
out["gate_pass"] = out["summary"]["router_perclip"]["gate_pass"]
out["note"] = ("The per-clip variant is the headline because published comparators score clips independently. "
               "The transductive variant uses other unlabelled clips of the same subject and is reported separately. "
               "Inner-fold estimates are slightly optimistic: the router is fitted on out-of-fold predictions from "
               "stream models that were trained on data including the evaluation fold.")
json.dump(out, open(os.path.join(ROOT, "results", "GATE_V2B.json"), "w"), indent=1, default=float)
print(f"\nHEADLINE (per-clip router) gate: {'PASS' if out['gate_pass'] else 'FAIL'}")
print("GATE_V2B_DONE", flush=True)
