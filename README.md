# Clip-matched stride: temporal sampling of a self-supervised facial-video backbone for engagement recognition

Code, pre-registration records and result files for the manuscript submitted to
PeerJ Computer Science.

The paper measures the two free parameters of the temporal sampling window used
by appearance models on EngageNet: the frame budget `T` and the stride `s`. The
span, the coverage and the within-window rate all follow from those two. The
stride is the parameter the design resolves; the single budget step from 16 to 32
frames is reported unresolved.

## Layout

    code/              experiment, evaluation and analysis code
    results/           result files behind every reported number, plus the gate records
    logs/              training logs, one per campaign
    preregistration/   the dated design records that fix every threshold before the run it governs

## The four campaigns after the official scoring

Four campaigns were pre-registered after the official splits had been scored and ran on the inner folds
only: segment consensus against clip-matched stride, per-student calibration from labelled enrolment
clips, continued self-supervised pretraining at the clip-matched-stride window, and a confirmation round
of the last. Each has its dated record in `preregistration/` (MODEL_V3 to MODEL_V6, with two addenda),
its gate script in `code/` (`gate_v3.py`, `kshot_v4.py`, `gate_v5.py`, `gate_v6.py`), its queue script
(`run_v3.sh`, `run_v5.sh`, `run_v6.sh`), its result files in `results/` and its log in `logs/`. All four
gates were missed and no official split was opened for any of them.

`code/verify_v3_v6_numbers.py` regenerates, at the printed rounding, every value the manuscript reports
from those campaigns and from the full count of clip lengths, and exits non-zero on any mismatch. Run on
the deposited `results/`, it reports every value checked and no mismatch, and the three gate scripts reproduce
`GATE_V3.json`, `GATE_V5.json` and `GATE_V6.json` exactly.

The design records are byte-identical to the files whose sha256 the gate scripts carry. In the result
files and logs the server's home directory is written `$HOME`; the gate scripts compare the last three
components of a weights path, so the masking does not affect them. `code/ft_video.py` gained three options
for these campaigns (`--segments`, `--resume`, `--init-ckpt`); without them it takes the same path as the
version that produced the official runs, and a rerun of the official inner-fold configuration reproduced
the earlier macro-F1 to every digit on five fold-and-seed pairs.

## What is here, and what is not

The result files hold aggregate metrics only: no clip identifiers and no per-clip
labels. Every number printed in the paper can be read back from them without
access to the video.

No video-derived artefact is redistributed — no frames, face crops, landmark
caches, backbone features or per-clip predictions. The dataset paper states a
CC BY 4.0 licence for EngageNet, but the download agreement that governs derived
features has not been re-read, so those are withheld.

Pretrained backbone weights are not redistributed. SVFAP and MAE-DFER weights
come from their own authors under their own terms, and the two InsightFace ONNX
models come from the `buffalo_l` pack under that project's terms.

## Datasets

EngageNet video and labels are granted by the dataset authors by the route their
paper sets out. DAiSEE labels are public and its video comes from its authors.

One condition on reproducing the paper: our copy of the EngageNet test labels
came in the same download as the training and validation labels, and we have not
confirmed with the dataset authors whether a grant made today carries them. A
reader given only the training and validation partitions can reproduce the
inner-fold numbers but not the scores on the official test split.

## Added on 8 October 2026, after a second review round

Three analyses that train nothing and open no split, each with a write-once output and checked, with the values the
manuscript prints, by `code/verify_rev_b.py`:

- `code/paired_official_ci.py` -> `results/PAIRED_OFFICIAL_CI.json`: paired subject-level intervals on the accuracy
  difference between systems of this work on the official splits, from the per-clip predictions of the official runs
  (Article S1, Table S12). Every point accuracy reproduces `G8_OFFICIAL.json`.
- `code/final_weights_and_time.py` -> `results/FINAL_WEIGHTS_AND_TIME.json`: the last-weight score against the EMA
  average, and the wall-clock time, of every inner-fold run under the locked schedule (Article S1, Table S11).
- `code/g8_official_fusion.py`: the fusion and scoring step that wrote `G8_OFFICIAL.json` on 25 September, which ran
  inline in the official queue; deposited byte for byte below a guard line, and rerun on a scratch copy it reproduces
  `G8_OFFICIAL.json` field for field.

- `code/count_params.py` -> `results/PARAM_COUNTS.json`: total and trainable parameter counts of SVFAP-B and
  MAE-DFER-B, one fresh process per backbone (in one process the two repositories' `modeling_finetune.py`
  collide, which is why `ENV_AND_CI.json` records an error for MAE-DFER-B).
- `code/clip_lengths.py` -> `results/CLIP_BRANCH_SHARES.json`: the share of clips shorter than each window span,
  i.e. the share that takes the short-clip branch of the sampler, over the face cache and over the clips used.

The environment block of `results/ENV_AND_CI.json` was read by a different interpreter (Python 3.11.16) from the one
the runs used (Python 3.13.13, named in `logs/v3.log`, `v5.log` and `v6.log`), so it does not describe the runs;
Article S1, Section B.2 gives the run interpreter's versions. The interval block of the same file is unaffected.

## Running it

The preprocessing, the window sampler, both streams, the pre-registered gates and
the subject-level bootstrap are all here, so a reader with the same dataset access
can regenerate every number in the paper.

`collect_for_manuscript.py` collects, on the machine the experiments ran on, the
run environment, the parameter counts and the one subject-level interval that the
deposit does not already carry.

## Status

Private during peer review. It will be made public on acceptance, and archived
for a DOI at that point.
