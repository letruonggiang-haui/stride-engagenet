"""Subject calibration from k labelled clips, design MODEL_V4_DESIGN_2026-10-04
(sha256 02e22447d6479a843b461a083b2c753cec6f8dc51e6fc3baf13bf4d2e9fdff58).

Setting: a teacher labels k clips of one enrolment recording of a student; the model is scored on the clips of that
student's OTHER recordings. Students with a single recording are excluded and counted.

    M0  uncalibrated: argmax p(y|x)
    B1  support majority: the most frequent support label for every query clip; ties go to the larger training prior
    M1  proposed: p'(y|x) ~ p(y|x) * pihat_s(y) / pi_train(y),  pihat_s = (n_s + alpha * pi_train) / (k + alpha)

Gate G-P (inner folds, k = 5), all four needed:
    1. three-fold mean macro-F1 (M1 - M0) >= +0.015
    2. on >= 2 folds the lower bound of the subject-level bootstrap interval of macro-F1 (M1 - M0) is > 0
    3. three-fold mean macro-F1 (M1 - B1) > 0
    4. three-fold mean UAR (M1 - M0) >= 0
alpha for fold f is chosen on the other two folds (k = 5); alpha for test on all three folds. The k curve uses that alpha.

Base predictions: inner folds, arm C of v3 (three SVFAP seeds, probabilities averaged); test, the three v1 SVFAP seeds.
Draws: R = 200 support draws per student from numpy default_rng(0), re-seeded for every (evaluation set, k), outer loop over
draws, inner loop over students in ascending id. Bootstrap: B = 2000 over students, default_rng(0), draws held fixed.
Metrics follow sklearn: macro-F1 over the labels present in truth or prediction, UAR over the labels present in truth.

usage: kshot_v4.py inner   writes results/KSHOT_V4_INNER.json (write-once)
       kshot_v4.py test    writes results/KSHOT_V4_TEST.json  (write-once; refuses unless the inner gate passed)
"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT

DESIGN_SHA = "02e22447d6479a843b461a083b2c753cec6f8dc51e6fc3baf13bf4d2e9fdff58"
K_GATE, KS, R, B = 5, (1, 3, 5, 10, 20), 200, 2000
ALPHAS = (1, 2, 5, 10, 20, 50, 100)
NC = 4
RX_EN = re.compile(r"subject_(\d+)_([a-z0-9]+)_vid_(\d+)_(\d+)$")
res = lambda *p: os.path.join(ROOT, "results", *p)
prd = lambda *p: os.path.join(ROOT, "preds", *p)


# ------------------------------------------------------------------ metrics from confusion matrices (rows truth)
def metrics(C):
    """C [..., 4, 4] -> dict of arrays [...]: acc, macro_f1 (sklearn default labels), uar (sklearn balanced accuracy)"""
    C = C.astype(np.float64); tp = np.diagonal(C, axis1=-2, axis2=-1); rows = C.sum(-1); cols = C.sum(-2)
    acc = tp.sum(-1) / C.sum((-2, -1))
    present = (rows + cols) > 0
    f1 = np.where(present, 2 * tp / np.maximum(rows + cols, 1e-12), 0.0)
    macro_f1 = (f1 * present).sum(-1) / present.sum(-1)
    has_true = rows > 0
    rec = np.where(has_true, tp / np.maximum(rows, 1e-12), 0.0)
    uar = (rec * has_true).sum(-1) / has_true.sum(-1)
    return dict(acc=acc, macro_f1=macro_f1, uar=uar)


def conf(y, yh):
    C = np.zeros((NC, NC), np.int64); np.add.at(C, (y, yh), 1); return C


def selftest():
    from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
    rng = np.random.default_rng(1)
    for _ in range(300):
        n = rng.integers(3, 60); y = rng.integers(0, rng.integers(1, 5), n); yh = rng.integers(0, rng.integers(1, 5), n)
        m = metrics(conf(y, yh))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = (accuracy_score(y, yh), f1_score(y, yh, average="macro"), balanced_accuracy_score(y, yh))
        assert np.allclose([m["acc"], m["macro_f1"], m["uar"]], ref), (y, yh, m, ref)


# ------------------------------------------------------------------ data
def load(paths):
    zs = [np.load(p, allow_pickle=True) for p in paths]
    keys, y = zs[0]["keys"], zs[0]["y"]
    for z in zs[1:]:
        assert (z["keys"] == keys).all() and (z["y"] == y).all(), "seed files disagree on clips"
    return np.mean([z["P"].astype(np.float64) for z in zs], 0), y.astype(int), keys


def prior(json_path):
    c = np.array(json.load(open(json_path))["train_class_counts"], np.float64); return c / c.sum()


def structure(keys, dataset):
    """eligible students: list of (subject, enrolment clip indices, query clip indices); plus excluded subjects"""
    rec = {}
    for i, k in enumerate(keys):
        k = str(k)
        if dataset == "engagenet":
            m = RX_EN.match(k); assert m, k
            s, order = int(m.group(1)), (m.group(2), int(m.group(3)))
        else:   # DAiSEE: subject = first six characters, recording = first seven (FACTS / project CLAUDE.md)
            s, order = int(k[:6]), (k[:7],)
        rec.setdefault(s, {}).setdefault(order, []).append(i)
    elig, excl = [], []
    for s in sorted(rec):
        recs = [rec[s][o] for o in sorted(rec[s])]
        if len(recs) < 2: excl.append(s); continue
        elig.append((s, np.array(recs[0]), np.concatenate([np.array(r) for r in recs[1:]])))
    return elig, excl


# ------------------------------------------------------------------ the three predictors over all draws
def draws(elig, y, k):
    """support label counts [R, S, 4], re-seeded for every (evaluation set, k)"""
    rng = np.random.default_rng(0); out = np.zeros((R, len(elig), NC), np.int64)
    for r in range(R):
        for j, (_, enr, _) in enumerate(elig):
            sup = rng.choice(enr, size=min(k, len(enr)), replace=False)
            out[r, j] = np.bincount(y[sup], minlength=NC)
    return out


def confusions(P, y, pi, elig, counts, alpha):
    """per draw and student: confusion of M1 [R,S,4,4], of B1 [R,S,4,4]; of M0 [S,4,4]"""
    Rn, S = counts.shape[:2]
    c1 = np.zeros((Rn, S, NC, NC), np.int64); cb = np.zeros_like(c1); c0 = np.zeros((S, NC, NC), np.int64)
    for j, (_, _, q) in enumerate(elig):
        Pq, yq = P[q], y[q]
        c0[j] = conf(yq, Pq.argmax(1))
        n = counts[:, j, :].astype(np.float64); keff = n.sum(1, keepdims=True)
        pihat = (n + alpha * pi) / (keff + alpha)                                   # [R,4]
        yh1 = (Pq[None] * (pihat / pi)[:, None, :]).argmax(2)                       # [R, nq]
        for r in range(Rn):
            c1[r, j] = conf(yq, yh1[r])
        mx = n.max(1, keepdims=True); cand = np.where(n == mx, pi[None], -1.0); yb = cand.argmax(1)   # ties -> larger prior
        for r in range(Rn):
            cb[r, j] = conf(yq, np.full(len(q), yb[r]))
    return c1, cb, c0


def summarise(c1, cb, c0):
    m1 = {k: float(v.mean()) for k, v in metrics(c1.sum(1)).items()}
    mb = {k: float(v.mean()) for k, v in metrics(cb.sum(1)).items()}
    m0 = {k: float(v) for k, v in metrics(c0.sum(0)).items()}
    return m0, m1, mb


def boot(c1, c0, metric):
    """subject-level bootstrap of mean-over-draws (M1 - M0); returns (low, high)"""
    rng = np.random.default_rng(0); S = c0.shape[0]; d = np.empty(B)
    for b in range(B):
        w = np.bincount(rng.integers(0, S, S), minlength=S).astype(np.float64)
        p1 = np.einsum("s,rsij->rij", w, c1); p0 = np.einsum("s,sij->ij", w, c0)
        d[b] = metrics(p1)[metric].mean() - metrics(p0)[metric]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ------------------------------------------------------------------ inner folds: alpha, gate, k curve
def inner():
    out_p = res("KSHOT_V4_INNER.json")
    if os.path.exists(out_p): sys.exit(f"refusing to overwrite {out_p}: write-once")
    selftest()
    folds = {}
    for f in range(3):
        P, y, keys = load([prd(f"ft_v3_C_f{f}_s{s}_train.npz") for s in range(3)])
        pis = [prior(res(f"FT_v3_C_f{f}_s{s}_train.json")) for s in range(3)]
        assert all(np.allclose(pis[0], p) for p in pis); pi = pis[0]
        elig, excl = structure(keys, "engagenet")
        cnt5 = draws(elig, y, K_GATE)
        grid = {}
        for a in ALPHAS:
            c1, cb, c0 = confusions(P, y, pi, elig, cnt5, a)
            grid[a] = summarise(c1, cb, c0)[1]["macro_f1"]
        folds[f] = dict(P=P, y=y, pi=pi, elig=elig, excl=excl, cnt5=cnt5, grid=grid)
        print(f"fold {f}: {len(elig)} eligible students ({len(excl)} excluded), {sum(len(q) for *_, q in elig)} query clips; "
              f"pi_train {np.round(pi, 3).tolist()}", flush=True)
    out = dict(design_sha256=DESIGN_SHA, k_gate=K_GATE, draws=R, bootstrap=B, alphas=list(ALPHAS), folds={})
    d_f1, d_b, d_uar, lows = [], [], [], 0
    for f in range(3):
        F = folds[f]
        others = [g for g in range(3) if g != f]
        a_f = max(ALPHAS, key=lambda a: np.mean([folds[g]["grid"][a] for g in others]))
        c1, cb, c0 = confusions(F["P"], F["y"], F["pi"], F["elig"], F["cnt5"], a_f)
        m0, m1, mb = summarise(c1, cb, c0)
        lo, hi = boot(c1, c0, "macro_f1"); lo_a, hi_a = boot(c1, c0, "acc")
        curve = {}
        for k in KS:
            ck = F["cnt5"] if k == K_GATE else draws(F["elig"], F["y"], k)
            curve[k] = dict(zip(("M0", "M1", "B1"), summarise(*confusions(F["P"], F["y"], F["pi"], F["elig"], ck, a_f))))
        df1 = m1["macro_f1"] - m0["macro_f1"]; d_f1.append(df1); d_b.append(m1["macro_f1"] - mb["macro_f1"])
        d_uar.append(m1["uar"] - m0["uar"]); lows += lo > 0
        out["folds"][f] = dict(eligible_students=len(F["elig"]), excluded_students=F["excl"],
                               query_clips=int(sum(len(q) for *_, q in F["elig"])), pi_train=F["pi"].tolist(),
                               alpha_grid_macro_f1_k5=F["grid"], alpha_chosen_on_other_folds=a_f, M0=m0, M1=m1, B1=mb,
                               diff_macro_f1_M1_M0=df1, ci95_macro_f1_M1_M0=[lo, hi], diff_acc_M1_M0=m1["acc"] - m0["acc"],
                               ci95_acc_M1_M0=[lo_a, hi_a], diff_macro_f1_M1_B1=m1["macro_f1"] - mb["macro_f1"],
                               diff_uar_M1_M0=m1["uar"] - m0["uar"], k_curve=curve)
        print(f"fold {f}: alpha {a_f}  M0 F1 {m0['macro_f1']:.4f}  M1 F1 {m1['macro_f1']:.4f}  B1 F1 {mb['macro_f1']:.4f}  "
              f"M1-M0 {df1:+.4f} [{lo:+.4f}, {hi:+.4f}]  acc M1-M0 {m1['acc'] - m0['acc']:+.4f}  UAR M1-M0 {m1['uar'] - m0['uar']:+.4f}", flush=True)
    c = [bool(np.mean(d_f1) >= 0.015), bool(lows >= 2), bool(np.mean(d_b) > 0), bool(np.mean(d_uar) >= 0)]
    passed = all(c)
    a_test = max(ALPHAS, key=lambda a: np.mean([folds[g]["grid"][a] for g in range(3)]))
    out.update(mean_diff_macro_f1_M1_M0=float(np.mean(d_f1)), folds_ci_lower_above_zero=int(lows),
               mean_diff_macro_f1_M1_B1=float(np.mean(d_b)), mean_diff_uar_M1_M0=float(np.mean(d_uar)),
               conditions=c, gate_pass=passed, alpha_for_test=a_test,
               verdict=("PASS" if passed else "FAIL") + f": mean macro-F1 M1-M0 {np.mean(d_f1):+.4f} (needs >= +0.015); "
                       f"{lows}/3 folds with CI lower bound > 0 (needs 2); M1-B1 {np.mean(d_b):+.4f} (needs > 0); "
                       f"UAR M1-M0 {np.mean(d_uar):+.4f} (needs >= 0)")
    print(out["verdict"], flush=True)
    json.dump(out, open(out_p, "w"), indent=1, default=float); print("wrote", out_p)


# ------------------------------------------------------------------ test, only after a passed gate
def test():
    out_p = res("KSHOT_V4_TEST.json")
    if os.path.exists(out_p): sys.exit(f"refusing to overwrite {out_p}: write-once")
    g = json.load(open(res("KSHOT_V4_INNER.json")))
    if not g.get("gate_pass"): sys.exit("refusing: gate G-P did not pass, so the test labels are not read (design section 9)")
    a = g["alpha_for_test"]; out = dict(design_sha256=DESIGN_SHA, alpha=a, k_gate=K_GATE, draws=R, bootstrap=B, sets={})
    SETS = {"engagenet_test_train": ("engagenet", "g8", "trainonly_on_test_train"), "engagenet_test_train-val": ("engagenet", "g8", "train-val"),
            "daisee_test_train": ("daisee", "g9", "trainonly_on_test_train"), "daisee_test_train-val": ("daisee", "g9", "train-val")}
    for name, (ds, g_, suf) in SETS.items():
        P, y, keys = load([prd(f"ft_{g_}_svfap_s{s}_{suf}.npz") for s in range(3)])
        pi = prior(res(f"FT_{g_}_svfap_s0_{suf}.json"))
        elig, excl = structure(keys, ds)
        cnt = draws(elig, y, K_GATE); c1, cb, c0 = confusions(P, y, pi, elig, cnt, a); m0, m1, mb = summarise(c1, cb, c0)
        ci = {m: boot(c1, c0, m) for m in ("acc", "macro_f1")}
        curve = {k: dict(zip(("M0", "M1", "B1"), summarise(*confusions(P, y, pi, elig, cnt if k == K_GATE else draws(elig, y, k), a)))) for k in KS}
        out["sets"][name] = dict(eligible_students=len(elig), excluded_students=excl, query_clips=int(sum(len(q) for *_, q in elig)),
                                 M0=m0, M1=m1, B1=mb, diff_acc_M1_M0=m1["acc"] - m0["acc"], ci95_acc_M1_M0=ci["acc"],
                                 diff_macro_f1_M1_M0=m1["macro_f1"] - m0["macro_f1"], ci95_macro_f1_M1_M0=ci["macro_f1"],
                                 diff_macro_f1_M1_B1=m1["macro_f1"] - mb["macro_f1"], k_curve=curve,
                                 role="primary" if ds == "engagenet" else "transfer, no gate")
        print(f"{name}: {len(elig)} students, M0 acc {m0['acc']:.4f} F1 {m0['macro_f1']:.4f} | M1 acc {m1['acc']:.4f} F1 {m1['macro_f1']:.4f} | "
              f"B1 F1 {mb['macro_f1']:.4f} | acc M1-M0 {m1['acc'] - m0['acc']:+.4f} {np.round(ci['acc'], 4).tolist()} | "
              f"F1 M1-M0 {m1['macro_f1'] - m0['macro_f1']:+.4f} {np.round(ci['macro_f1'], 4).tolist()}", flush=True)
    json.dump(out, open(out_p, "w"), indent=1, default=float); print("wrote", out_p)


if __name__ == "__main__":
    {"inner": inner, "test": test}[sys.argv[1]]()
