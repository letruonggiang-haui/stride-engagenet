"""Total and trainable parameter counts of one backbone, built as ft_video.py builds it, in a fresh process.

collect_for_manuscript.py builds SVFAP-B and MAE-DFER-B in one process. Both repositories name their model file
modeling_finetune.py, so the second import returns the first module and MAE-DFER-B fails with an AttributeError
(results/ENV_AND_CI.json, parameters.maedfer_error). Run once per backbone instead; the model code below is the
build() and LoRA wrapping of collect_for_manuscript.py, unchanged.

usage (on the experiment server, PYTHONPATH=pylib:code, the run interpreter): count_params.py svfap|maedfer
prints one JSON line; results/PARAM_COUNTS.json collects the two lines.
"""
import json, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p2x_lib import ROOT

backbone = sys.argv[1]
assert backbone in ("svfap", "maedfer")


class LoRAQKV(nn.Module):
    def __init__(self, base, r, alpha=None):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = (alpha or 2 * r) / r


repos = os.path.join(ROOT, "repos")
if backbone == "svfap":
    sys.path.insert(0, os.path.join(repos, "SVFAP"))
    import modeling_finetune as mf
    model = mf.svfap_base_patch16_160(num_classes=4, all_frames=16, drop_path_rate=0.0, stage_depths=(12, 6, 3),
                                      temporal_kernel_size=2, num_bottleneck_tokens=(8,))
else:
    sys.path.insert(0, os.path.join(repos, "MAE-DFER"))
    import modeling_finetune as mf
    model = mf.vit_base_dim512_no_depth_patch16_160(
        num_classes=4, all_frames=16, drop_path_rate=0.0, depth=16, attn_type="local_global",
        lg_region_size=(2, 5, 10), lg_first_attn_type="self", lg_third_attn_type="cross",
        lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False, lg_classify_token_type="region",
        lg_no_second=False, lg_no_third=False)
total = sum(p.numel() for p in model.parameters())
n_l = 0
for name, mod in list(model.named_modules()):
    for cn, child in list(mod.named_children()):
        if isinstance(child, nn.Linear) and cn in ("qkv", "q", "kv", "v", "query", "value"):
            setattr(mod, cn, LoRAQKV(child, 16)); n_l += 1
for n, p in model.named_parameters():
    if not (".A" in n or ".B" in n or n.startswith("head") or "head." in n):
        p.requires_grad = False
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(json.dumps({"backbone": backbone, "total": total, "lora_modules": n_l, "trainable_rank16": trainable,
                  "python": sys.version.split()[0], "torch": torch.__version__}))
