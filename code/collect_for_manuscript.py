#!/usr/bin/env python3
"""Collect, on the machine the experiments ran on, the figures the manuscript still
marks for the author.

    cd ~/engagenet_p2_crossdomain/code
    python3 collect_for_manuscript.py

Writes results/ENV_AND_CI.json and prints the same content. Three blocks run
independently, so one failure still leaves the other two reported.

Block A  the run environment: nothing in the deposited logs or result files records
         a version string, a GPU name or a timing, so these can only be read here.
Block B  total parameter counts. ft_video.py prints the TRAINABLE count after the
         adapters go in, which the logs kept (1,058,820 for SVFAP-B and 1,968,132
         for MAE-DFER-B), but never the total. The model is built exactly as
         ft_video.py builds it, so the trainable figure printed here should
         reproduce the log and confirm the build is the same one.
Block C  the one interval the manuscript quotes with no deposited source: the
         95 % subject-level interval around the single SVFAP seed on inner fold 0,
         printed in Section 4.1.6 as 69.16 with [62.39, 75.93], a width of 13.5
         points. The bootstrap here is score_official.py's own function, with its
         own defaults (B = 2000, seed 0, percentiles 2.5 and 97.5), so a match is
         a reproduction rather than a new estimate.

NOT recoverable, and not attempted here: the 21.3-point median interval width that
Section 4.1.6 reports for 11-subject subsets of fold 0. No script in the deposit
performs that exercise and no record fixes how many subsets were drawn or under
which seed, so re-running it would be a new analysis under a procedure chosen after
the number is known. That one is a decision, not a measurement: fix the procedure
and report it as new, or drop the figure.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
out = {}


def section(name):
    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)


def rec(block, key, value):
    out.setdefault(block, {})[key] = value
    print("  %-34s %s" % (key, value))


# ----------------------------------------------------------------- A. environment
section("A. Run environment")
try:
    import platform
    rec("environment", "python", platform.python_version())
    rec("environment", "platform", platform.platform())
except Exception as e:
    rec("environment", "platform_error", repr(e))

for mod, attr in (("torch", "__version__"), ("numpy", "__version__"),
                  ("sklearn", "__version__"), ("cv2", "__version__"),
                  ("onnxruntime", "__version__"), ("insightface", "__version__"),
                  ("pandas", "__version__")):
    try:
        m = __import__(mod)
        rec("environment", mod, getattr(m, attr))
    except Exception as e:
        rec("environment", mod, "NOT IMPORTABLE: %r" % (e,))

try:
    import torch
    rec("environment", "torch.version.cuda", torch.version.cuda)
    rec("environment", "cudnn", torch.backends.cudnn.version())
    rec("environment", "cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rec("environment", "gpu_name", p.name)
        rec("environment", "gpu_memory_MiB", round(p.total_memory / 1024 ** 2))
        rec("environment", "gpu_capability", "%d.%d" % (p.major, p.minor))
except Exception as e:
    rec("environment", "torch_cuda_error", repr(e))

try:
    q = subprocess.run(["nvidia-smi",
                        "--query-gpu=name,driver_version,memory.total",
                        "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=30)
    rec("environment", "nvidia_smi", q.stdout.strip() or q.stderr.strip())
except Exception as e:
    rec("environment", "nvidia_smi", "not available: %r" % (e,))

# ------------------------------------------------------- B. parameter counts
section("B. Parameter counts, built as ft_video.py builds them")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from p2x_lib import ROOT

    class LoRAQKV(nn.Module):
        """Copied from ft_video.py so the counted model is the trained one."""
        def __init__(self, base, r, alpha=None):
            super().__init__()
            self.base = base
            for p in base.parameters():
                p.requires_grad = False
            self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
            self.B = nn.Parameter(torch.zeros(base.out_features, r))
            self.scale = (alpha or 2 * r) / r
            self.in_features, self.out_features = base.in_features, base.out_features

        @property
        def weight(self):
            return self.base.weight + self.scale * (self.B @ self.A)

        @property
        def bias(self):
            return self.base.bias

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    def build(backbone, frames=16, K=4):
        repos = os.path.join(ROOT, "repos")
        if backbone == "svfap":
            sys.path.insert(0, os.path.join(repos, "SVFAP"))
            import modeling_finetune as mf
            return mf.svfap_base_patch16_160(
                num_classes=K, all_frames=frames, drop_path_rate=0.0,
                stage_depths=(12, 6, 3), temporal_kernel_size=2,
                num_bottleneck_tokens=(8,))
        sys.path.insert(0, os.path.join(repos, "MAE-DFER"))
        import modeling_finetune as mf
        return mf.vit_base_dim512_no_depth_patch16_160(
            num_classes=K, all_frames=frames, drop_path_rate=0.0, depth=16,
            attn_type="local_global", lg_region_size=(2, 5, 10),
            lg_first_attn_type="self", lg_third_attn_type="cross",
            lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
            lg_classify_token_type="region", lg_no_second=False, lg_no_third=False)

    EXPECTED = {"svfap": 1058820, "maedfer": 1968132}
    for backbone in ("svfap", "maedfer"):
        try:
            # drop_path_rate changes no parameter count, so the weights are not needed
            model = build(backbone)
            total = sum(p.numel() for p in model.parameters())
            rec("parameters", backbone + "_total", total)

            n_l = 0
            for name, mod in list(model.named_modules()):
                for cn, child in list(mod.named_children()):
                    if isinstance(child, nn.Linear) and cn in ("qkv", "q", "kv", "v", "query", "value"):
                        setattr(mod, cn, LoRAQKV(child, 16))
                        n_l += 1
            for n, p in model.named_parameters():
                if not (".A" in n or ".B" in n or n.startswith("head") or "head." in n):
                    p.requires_grad = False
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            rec("parameters", backbone + "_lora_modules", n_l)
            rec("parameters", backbone + "_trainable_rank16", trainable)
            rec("parameters", backbone + "_matches_log",
                trainable == EXPECTED[backbone])
            rec("parameters", backbone + "_total_with_adapters",
                sum(p.numel() for p in model.parameters()))
            del model
        except Exception as e:
            rec("parameters", backbone + "_error", repr(e))
except Exception as e:
    rec("parameters", "setup_error", repr(e))

# ------------------------------------------------- C. the undeposited interval
section("C. The fold-0 single-seed interval of Section 4.1.6")
try:
    import numpy as np
    from sklearn.metrics import accuracy_score
    from p2x_lib import ROOT

    def bootstrap_ci(y, yhat, subject, B=2000, seed=0):
        """score_official.py's own function, unchanged, so this reproduces rather
        than re-estimates."""
        us = np.unique(subject)
        idx = {s: np.where(subject == s)[0] for s in us}
        rng = np.random.default_rng(seed)
        boots = [accuracy_score(y[t], yhat[t]) for t in
                 (np.concatenate([idx[s] for s in rng.choice(us, len(us), replace=True)])
                  for _ in range(B))]
        return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # 69.16 is FT_g7_svfap_f0_s2_train.json /final/acc = 0.6915849...,
    # one SVFAP member on inner fold 0, not the ensemble
    name = "ft_g7_svfap_f0_s2_train.npz"
    path = os.path.join(ROOT, "preds", name)
    rec("interval", "predictions_file", path)
    if not os.path.exists(path):
        rec("interval", "status", "NOT FOUND; list ~/engagenet_p2_crossdomain/preds "
                                 "and pick the fold-0 SVFAP seed-2 file")
    else:
        z = np.load(path, allow_pickle=True)
        y, P, subject = z["y"], z["P"], z["subject"]
        yh = P.argmax(1)
        acc = float(accuracy_score(y, yh))
        lo, hi = bootstrap_ci(y, yh, subject)
        rec("interval", "n_clips", int(len(y)))
        rec("interval", "n_subjects", int(len(np.unique(subject))))
        rec("interval", "accuracy_pct", round(100 * acc, 2))
        rec("interval", "ci_pct", [round(100 * lo, 2), round(100 * hi, 2)])
        rec("interval", "width_pct", round(100 * (hi - lo), 2))
        rec("interval", "manuscript_says", {"accuracy": 69.16,
                                            "ci": [62.39, 75.93], "width": 13.5})
        rec("interval", "reproduces",
            abs(100 * acc - 69.16) < 0.01
            and abs(100 * lo - 62.39) < 0.01 and abs(100 * hi - 75.93) < 0.01)
except Exception as e:
    rec("interval", "error", repr(e))

# ------------------------------------------------------------------- write
section("Written")
try:
    dest = os.path.join(os.path.dirname(HERE), "results", "ENV_AND_CI.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True, default=str)
    print("  " + dest)
except Exception as e:
    print("  could not write: %r" % (e,))
print("\nThe 21.3-point median width of Section 4.1.6 is deliberately NOT computed "
      "here; see this file's docstring.")
