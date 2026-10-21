# README of Supplemental Code and Data S1

## Title

Clip-matched stride: temporal sampling of a self-supervised facial-video backbone for engagement recognition on
EngageNet.

Supplemental Code and Data S1 of the article of the same title, submitted to PeerJ Computer Science as an AI
Application article by Truong Giang Le, Dang Hung Phan, Viet Tuyen Nguyen and Trung Viet Nguyen. Corresponding
authors: Dang Hung Phan (hungpd@haui.edu.vn) and Trung Viet Nguyen (vietit@utc.edu.vn).

## Description

The code fine-tunes a self-supervised facial-video backbone (SVFAP-B, with MAE-DFER-B as a second ensemble member)
on four-level clip engagement labels, and measures the two free parameters of the temporal sampling window: the frame
budget `T` and the stride `s`. The span, the coverage of the clip and the within-window rate all follow from those
two. The article reports that, on EngageNet with SVFAP-B, keeping `T` = 16 and lengthening the stride until the window
spans the labelled clip (clip-matched stride, `s` = 20) gives the largest gain measured, and that the single
frame-budget step from 16 to 32 frames is not resolved.

This file holds the code, the dated pre-registration records that fixed every threshold before the run it governed,
the result files behind every number in the article, and the training logs. It holds no video and no video-derived
data (see Dataset Information).

## Dataset Information

Both datasets are third-party data and are not redistributed here.

- **EngageNet** (Singh et al., 2023, DOI 10.1145/3577190.3614164). Face videos of learners with one of four
  engagement labels per 10 s clip: Not-Engaged, Barely-engaged, Engaged, Highly-Engaged. Videos and labels are
  requested from the dataset authors by the route given at https://github.com/engagenet/engagenet_baselines. The
  code expects the label files `train_engagement_labels.xlsx`, `validation_engagement_labels.xlsx` and
  `test_engagement_labels.csv`. Clips labelled SNP are dropped.
- **DAiSEE**, the Dataset for Affective States in E-Environments (Gupta et al., 2016, arXiv:1609.01885). Downloaded
  after a registration form at https://people.iith.ac.in/vineethnb/resources/daisee/index.html. The code uses the
  `Engagement` column (0 to 3) of `TrainLabels.csv`, `ValidationLabels.csv` and `TestLabels.csv`; the files do not
  share one separator, so they are read with separator detection. The subject code is the first six characters of
  the clip identifier.

One condition on reproducing the article: our copy of the EngageNet test labels came in the same download as the
training and validation labels, and we have not confirmed with the dataset authors whether a grant made today
carries them. A reader given only the training and validation partitions can reproduce the inner-fold numbers but
not the scores on the official test split.

What is here and what is not. The result files hold aggregate metrics only: no clip identifiers and no per-clip
labels. No video-derived artefact is redistributed: no frames, face crops, landmark caches, backbone features or
per-clip predictions. The dataset paper states a CC BY 4.0 licence for EngageNet, but the download agreement that
governs derived features has not been re-read, so those are withheld. In the result files and logs the server's home
directory is written `$HOME`.

## Code Information

    code/                experiment, evaluation and analysis code (Python, three bash queue scripts)
    results/             result files behind every reported number, and the gate records
    logs/                training logs, one per campaign
    preregistration/     the dated design records, in Vietnamese as written
    preregistration/en/  English translations of those records, made on 15 October 2026; numbers keep the
                         original's Vietnamese format (decimal comma, space as thousands separator)

The scripts used for the article, by stage:

| Stage | Scripts |
|---|---|
| Preprocessing | `cache_faces.py` (face detection, tracking and 224 x 224 crops; uses the SCRFD loader of `extract_feats.py`), `geom_extract_min.py` and `geom96.py` (68-point 3D landmarks and the geometry descriptor), `seq300_daisee.py` |
| Shared library | `p2x_lib.py` (label loading, data paths, metrics, subject-level bootstrap) |
| Appearance stream | `ft_video.py` (window sampler, LoRA fine-tuning of SVFAP-B or MAE-DFER-B, inner folds and official scoring) |
| Geometry and sequence streams | `seqmodel.py`, `seqmodel_daisee.py`, `seqmodel_defs.py`, `seq_official.py` |
| Official scoring and fusion | `score_official.py`, `g8_official_fusion.py` |
| Gates of the pre-registered components | `gate_v2a.py`, `gate_v2b.py`, `peek_v2a.py` (read-only view), `gate_v3.py`, `kshot_v4.py`, `pretrain_v5.py`, `gate_v5.py`, `gate_v6.py`, queue scripts `run_v3.sh`, `run_v5.sh`, `run_v6.sh` |
| Analyses | `paired_ci_v3.py`, `paired_fusion_ci.py`, `paired_official_ci.py`, `final_weights_and_time.py`, `ema_robustness.py`, `headroom.py`, `kappa_compare.py`, `daisee_label_audit.py`, `clip_lengths.py`, `count_params.py`, `collect_for_manuscript.py` |
| Checks of the printed numbers | `verify_v3_v6_numbers.py`, `verify_rev_b.py` |

Five scripts come from the earlier, classroom stage of the same project and produce no number in this article:
`cache_crops.py`, `chk_seek.py`, `fuse_eval.py`, `make_tables.py` and `verify_manuscript_numbers.py`. They are kept
because the deposit is a copy of the project's code folder. `extract_feats.py` belongs to the same stage, but its
face-detector loader is used by `cache_faces.py`.

The four campaigns after the official scoring were pre-registered after the official splits had been scored and ran
on the inner folds only: segment consensus against clip-matched stride, per-student calibration from labelled
enrolment clips, continued self-supervised pretraining at the clip-matched-stride window, and a confirmation round of
the last. Each has its dated record in `preregistration/` (MODEL_V3 to MODEL_V6, with two addenda), its gate script,
its queue script, its result files and its log. All four gates were missed and no official split was opened for any
of them. The design records are byte-identical to the files whose SHA-256 the gate scripts carry. `ft_video.py` gained
three options for these campaigns (`--segments`, `--resume`, `--init-ckpt`); without them it takes the same path as
the version that produced the official runs, and a rerun of the official inner-fold configuration reproduced the
earlier macro-F1 to every digit on five fold-and-seed pairs.

Analyses added on 8 October 2026, which train nothing and open no split:

- `paired_official_ci.py` -> `results/PAIRED_OFFICIAL_CI.json`: paired subject-level intervals on the accuracy
  difference between systems of this work on the official splits (Article S1, Table S12).
- `final_weights_and_time.py` -> `results/FINAL_WEIGHTS_AND_TIME.json`: last-weight scores against the EMA average,
  and wall-clock time, of every inner-fold run under the locked schedule (Article S1, Table S11).
- `g8_official_fusion.py`: the fusion and scoring step that wrote `G8_OFFICIAL.json` on 25 September, which ran
  inline in the official queue; rerun on a scratch copy it reproduces `G8_OFFICIAL.json` field for field.
- `count_params.py` -> `results/PARAM_COUNTS.json`: total and trainable parameter counts of SVFAP-B and MAE-DFER-B,
  one fresh process per backbone (in one process the two upstream `modeling_finetune.py` modules collide, which is
  why `results/ENV_AND_CI.json` records an error for MAE-DFER-B).
- `clip_lengths.py` -> `results/CLIP_BRANCH_SHARES.json`: the share of clips shorter than each window span, which is
  the share that takes the short-clip branch of the sampler.

The environment block of `results/ENV_AND_CI.json` was read by a different interpreter (Python 3.11.16) from the one
the runs used (Python 3.13.13, named in `logs/v3.log`, `v5.log` and `v6.log`), so it does not describe the runs. The
interval block of the same file is unaffected.

## Usage Instructions

The paths are those of the server the experiments ran on; `ROOT` is set in `code/p2x_lib.py`. All commands are run
from the top folder of this file (the folder that holds `code/`). Steps 1 and 7 need no dataset: a reader can install
the environment and run the two verification scripts against the deposited result files without any video.

1. Create a Python 3.13 environment with a CUDA build of PyTorch and install the other packages:

       python3.13 -m venv ~/venvs/engagenet
       source ~/venvs/engagenet/bin/activate
       pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
       pip install -r requirements.txt

2. Place the data. EngageNet labels in `~/engagenet/data/EmotiW_2023_EngageNet/`, EngageNet videos in
   `~/engagenet/videos/` (test videos may sit in `ROOT/data/engagenet_test_videos/`), DAiSEE under
   `~/daisee/DAiSEE/` with its `DataSet/` and `Labels/` folders. To check that the EngageNet labels load, run

       PYTHONPATH=code python -c "import p2x_lib; df = p2x_lib.engagenet_labels(); print(df.groupby(['split', 'label']).size())"

   `engagenet_labels()` reads the three label files, drops the SNP clips and returns one row per clip with the
   columns `chunk`, `label`, `split`, `y4` (level 0 to 3), `y3`, `subject`, `session`, `video`, `idx`, `recording`
   and `key`.
3. Place the third-party code and weights, which are not redistributed: the SVFAP repository
   (https://github.com/sunlicai/SVFAP) in `ROOT/repos/SVFAP` and its VoxCeleb2 checkpoint as
   `ROOT/weights/svfap/svfap_base_voxceleb2_checkpoint-99.pth`; the MAE-DFER repository
   (https://github.com/sunlicai/MAE-DFER) in `ROOT/repos/MAE-DFER` and its VoxCeleb2 checkpoint as
   `ROOT/weights/maedfer/maedfer_base_voxceleb2.pth`; the InsightFace `buffalo_l` pack
   (https://github.com/deepinsight/insightface), whose `det_10g.onnx` and `1k3d68.onnx` the code reads from
   `~/.insightface/models/buffalo_l/`.
4. Cache the face crops:

       python code/cache_faces.py engagenet ROOT/crops_face/engagenet
       python code/cache_faces.py daisee ROOT/crops_face/daisee

5. Train and score one configuration. An inner-fold run of clip-matched stride, fold 0, seed 0:

       python code/ft_video.py --backbone svfap --lora 16 --frames 16 --stride 20 --epochs 12 --warmup 2 \
           --lr 1e-3 --accum 2 --ema 0.999 --seed 0 --cv-folds 3 --cv-fold 0 --tag cms_f0_s0

   The official runs used the same settings with `--train-splits train` or `--train-splits train,val`,
   `--eval-split val` or `test`, and `--open-official`; without that flag the script refuses to score an official
   split. The full argument list of every official run is stored in its `results/FT_g8_*.json` (EngageNet) and
   `results/FT_g9_*.json` (DAiSEE, `--stride 19`).
6. Fuse the ensemble members and draw the subject-level intervals with `code/score_official.py`. It needs the
   per-clip predictions written by the runs of step 5, which are not deposited.
7. Check the printed numbers against the deposited results, with no data needed:

       python code/verify_v3_v6_numbers.py
       python code/verify_rev_b.py

## Requirements

A CUDA GPU; the runs used one NVIDIA L4 with 23,034 MiB of memory. The training logs record Python 3.13.13, PyTorch
2.6.0 and CUDA 12.4. The other versions were not logged at run time; `requirements.txt` lists the versions installed
on the same server on 15 October 2026 (Ubuntu 25.10).

## Methodology

1. Labels are read from the official label files; EngageNet SNP clips are dropped.
2. Frame rates are normalised to about 30 fps. A clip encoded above 100 fps is decimated by round(fps / 30);
   otherwise, a clip with more than 1,500 frames is decimated by round(frames / 300).
3. SCRFD detects the face on every fourth frame. The face with the largest product of box area and detection score
   is kept, boxes are interpolated between detections and held at the clip ends, and a square crop of 1.3 times the
   box is stored for every frame at 224 x 224 pixels.
4. A window of `T` frames at stride `s` is sampled (`ft_video.py`, `Clips.windows`). When the clip is shorter than the
   span `(T - 1)s + 1`, the window is `range(0, N, s)` padded with the last frame. Otherwise training uses a random
   start with per-frame jitter, and scoring averages five evenly spaced windows.
5. Training cuts a random square of 0.7 to 1.0 of the crop side, flips horizontally with probability 0.5, resizes to
   160 x 160 and normalises with the ImageNet channel statistics. The backbone is fine-tuned with rank-16 LoRA for 12
   epochs, with an exponential moving average of the weights at 0.999.
6. Every choice is made on three subject-disjoint inner folds of the training split, and each official split is
   opened once per training protocol. Accuracy, UAR and macro-F1 are reported, with 95 % intervals from 2,000
   resamples of subjects.

## Citations

If you use this code, please cite the article (Le, Phan, Nguyen and Nguyen, "Clip-matched stride: temporal sampling
of a self-supervised facial-video backbone for engagement recognition on EngageNet", PeerJ Computer Science,
submitted) and the works it builds on:

- Singh M, Hoque X, Zeng D, Wang Y, Ikeda K, Dhall A. 2023. Do I have your attention: a large scale engagement
  prediction dataset and baselines. ICMI 2023, 174–182. DOI 10.1145/3577190.3614164.
- Gupta A, D'Cunha A, Awasthi K, Balasubramanian V. 2016. DAiSEE: towards user engagement recognition in the wild.
  arXiv:1609.01885.
- Sun L, Lian Z, Wang K, He Y, Xu M, Sun H, Liu B, Tao J. 2025. SVFAP: self-supervised video facial affect perceiver.
  IEEE Transactions on Affective Computing 16(1):405–422. DOI 10.1109/TAFFC.2024.3436913.
- Sun L, Lian Z, Liu B, Tao J. 2023. MAE-DFER: efficient masked autoencoder for self-supervised dynamic facial
  expression recognition. ACM MM 2023, 6110–6121. DOI 10.1145/3581783.3612365.
- Guo J, Deng J, Lattas A, Zafeiriou S. 2022. Sample and computation redistribution for efficient face detection.
  ICLR 2022.
- Hu EJ, Shen Y, Wallis P, Allen-Zhu Z, Li Y, Wang S, Wang L, Chen W. 2022. LoRA: low-rank adaptation of large
  language models. ICLR 2022.

## License & Contribution Guidelines

The code is released under version 3 of the GNU General Public License (`LICENSE`). `NOTICE.md` lists what the
licence does not cover: pretrained backbone weights, the InsightFace models and all dataset content.

Questions, reports of errors and proposed changes are welcome by e-mail to the corresponding authors. A report should
name the file, the command that was run and the output. A proposed change to code that produces a reported number
should come with the output of the two verification scripts of Usage Instructions step 7. The dated records in
`preregistration/` are not edited: they are kept byte for byte because the gate scripts carry their SHA-256 digests.
