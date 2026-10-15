> English translation of `preregistration/MODEL_PAPER_DESIGN_2026-09-21.md` (SHA-256 `6f7eeae5384d7a0111e7c761fb6937ed3c890cd3bd250bb4f4229b2d7fca1762`), made on 15 October 2026 for publication with the article. The Vietnamese original is the dated record and is unchanged; where the two differ, the original holds.

# Model paper design, version 2 — single goal: beat SOTA on EngageNet and DAiSEE

Written on 21/09/2026, replacing version 1 of the same day (version 1 took "robustness to framing" as its axis; the author changed the goal to beating SOTA, with no limit on backbone or training time). Every prediction and threshold in section 5 is recorded before running. Every result is reported, including when it does not beat.

## 1. Marks to beat (verified)

**EngageNet, 4 levels, official validation (11 people, 1 071 clips; test 26 people, 2 257 clips, labels provided to the group by the dataset authors):**

| Method | Val | Test | Notes |
|---|---|---|---|
| PriorNet (arXiv:2605.03615, 5/2026, preprint, no code) | **73,58** | — | SVFAP + Prior-LoRA rank 16 in every even block + missing-face placeholder + evidential loss; 16 frames 224. Their ablation: SVFAP + CE 68,91; + placeholder 71,15; + LoRA 70,03; + objective 69,56; full 73,58 |
| Ordinal ST-GCN (arXiv:2403.17175) | 71,24 | — | landmark |
| ICMI 2023 (dataset authors) | 69,10 | 67,61 | OpenFace; train+val → test: 69,29 |
| VE-MD (arXiv:2604.02397) | 68,98 | — | |
| TCCT-Net | 68,91 | — | |
| Baseline MARLIN→Transformer (EmotiW 2023) | 65,54 | 62,00 | |
| Group: frozen MARLIN-L + adapter (`LADDER_U4`) | 63,2 | — | 3 seeds, per-person CI [0,539; 0,731] |
| Group (paper P1): frozen MARLIN + temporal pooling, ensemble | 67,51 | 68,3 | |

**DAiSEE, 4 engagement levels, official test 1 784 clips** (`P2-Identity-Invariant-Engagement/SOTA_DAISEE_KIEM_CHUNG.md`):

| Method | Test | Protocol |
|---|---|---|
| ViBED-Net (arXiv:2510.18016, preprint) | 73,43 | official split; lowest-class recall 0,25 |
| PriorNet | 69,06 | train+val pooled |
| Selim 2022 (IEEE Access, peer-reviewed) | 67,48 | official split |
| Abedi & Khan ordinal TCN | 67,4 | |
| ResNet+TCN (with code) | 63,9 | train+val pooled |
| Majority-class floor | 49,4 | |

How to read the marks: on EngageNet, the published figures are validation figures; for test there are only the dataset authors' figures. The paper will report both, and claim "beats" on validation only if > 73,58, and on test only if > 69,29. On DAiSEE, report both protocols; claim "beats" when > 69,06 (pooled) and > 67,48 (official); ViBED-Net's 73,43 is listed in the table with a preprint flag.

## 2. Why the current model is 10 points behind and what can close that gap

The group's encoder is fully frozen (0,632). PriorNet shows on the same dataset: fine-tuning with LoRA + missing-face handling + a better loss = +4,7 points on their own backbone. Their backbone (SVFAP, self-supervised pretraining on VoxCeleb2, 160×160, 16 frames, stride 4) is a backbone specialised for face video, stronger than MARLIN at dynamic expression recognition (DFEW WR 74,8 versus MARLIN not reported on DFEW; MAE-DFER from the same group 74,4). Three levers, in order of expectation:

1. **Fine-tune backbone** (not only the head): LoRA rank 16 in every block, or full fine-tuning with layer-wise LR decay, EMA, drop-path; SVFAP-B is the first candidate, MAE-DFER-B (VoxCeleb2, weights on Google Drive) second, MARLIN-L third, VideoMAEv2-giant (HF, CC-BY-NC) fourth if the first three are not enough.
2. **Full-clip coverage**: each clip has 300 frames; SOTA uses 16 frames. Train with random windows, score with the mean of 5 windows × 2 crops (TTA), plus a small temporal pooling layer over the window embeddings.
3. **Ensemble** of several fine-tuned backbones, probability averaging. Ensembling usually gives +1–3 points; this is the main reason to do three backbones rather than one.

In addition, two cheap components with precedent: a placeholder for frames where no face is detected (PriorNet +2,2 on EngageNet; the group rewrites it, not using their code) and ordinal soft labels (LDL Gaussian σ = 0,5, already in `train_core.py`).

On DAiSEE: the group's audit shows the labels are close to noise (adjacent agreement 0,58 versus 0,54 independent). High scores on DAiSEE are partly a fit to class bias; ViBED-Net 73,43 has lowest-class recall 0,25. The paper must report UAR and macro-F1 alongside accuracy and state this clearly, but still optimises accuracy because that is the table's metric.

## 3. Protocol, unchanged throughout the campaign

- EngageNet: train with 91 people to select everything; validation with 11 people scored on finalised configurations (each configuration once); **test with 26 people opened once at the very end** for (a) the model trained on train and (b) the model trained on train + val, as the dataset authors did. Hyperparameters are selected with 3 person-independent folds inside train, validation is not used for selection.
- DAiSEE: same approach, two protocols (official; train+val pooled), test opened once.
- Metrics: accuracy (to compare with the table), UAR, macro-F1, 95 % per-person bootstrap CI, 3 seeds, paired differences between configurations.
- Test-set labels or statistics are not used at any step, including feature normalisation and threshold selection.
- The full configuration is written into the result file; every run is listed in the appendix, including failed ones.

## 4. Run plan (by stage; a later stage runs only if the earlier stage gives a reason)

**G0. Data.** Cache 224×224 JPEG face crops for every frame of every EngageNet clip (11 311) and DAiSEE clip (8 232): SCRFD on one frame in every 4, box interpolation, 1,3× enlargement; frames without a face are flagged (used for the placeholder). Estimated 50 GB, 4–6 CPU hours with 40 processes.

**G1. Reproduce the mark on the strongest downloadable backbone.** SVFAP-B (if downloadable; the SharePoint link currently returns 401, trying via the browser) or MAE-DFER-B: fine-tune with their own recipe for DFEW (AdamW, lr 1e-3, 100 epochs, batch 16–32, 160×160, 16×4, mixup/label smoothing as in the original code) on EngageNet train, score val. Expected: 67–71 (PriorNet reports SVFAP + CE = 68,91). This is the starting point; if below 66 the pipeline has a bug, do not proceed.

**G2. Each lever, one 3-seed run each:** (a) missing-face placeholder; (b) ordinal LDL instead of CE; (c) 5-window TTA; (d) LoRA rank 16 versus full fine-tuning; (e) temporal pooling layer over 5 windows. Keep whichever gives > +0,5 points on CV within train.

**G3. Three backbones with the finalised recipe, then ensemble.** SVFAP-B, MAE-DFER-B, MARLIN-L (LoRA, 224). Ensemble by probability averaging, weights selected on CV within train.

**G4. Validation once, then test once** (train-only and train+val). At the same time run DAiSEE with the recipe finalised on EngageNet (no reselection), plus one separate hyperparameter selection on DAiSEE train if the EngageNet recipe gives a result below the with-code mark of 63,9.

Budget: G1 about 6–10 hours/backbone/seed on L4 (SVFAP-B 160² is much lighter than MARLIN-L 224²); the whole campaign 150–250 GPU hours, i.e. 1–2 weeks of continuous running. The GPU is currently free.

## 5. Predictions and thresholds, recorded in advance

| | My prediction (EngageNet val) | Stop / rejection threshold |
|---|---|---|
| G1 fine-tune one backbone | 67–71 | < 66: fix the pipeline before proceeding |
| G2 placeholder | +1 to +2 | < +0,5 on CV: drop |
| G2 ordinal LDL | 0 to +1 | < +0,5: drop |
| G2 5-window TTA | +0,5 to +1,5 | < +0,5: drop |
| G2 LoRA versus full | on par, LoRA more stable | choose the higher one on CV |
| G3 three-backbone ensemble | +1,5 to +3 over the best backbone | < +0,5: report single backbone |
| **Final result, val** | **72–75; I estimate the probability of beating 73,58 at 40–50 %** | if < 73,58: the paper reports test as the main figure, val in the table, no val SOTA claim |
| Final result, test (train+val) | 69–72 | versus 69,29 |
| DAiSEE, pooled | 67–71 | versus 69,06 |
| DAiSEE, official | 66–70 | versus 67,48 |

Largest risk, recorded in advance: EngageNet validation has 11 people, the CI is ±5 points wide; a 72 model and a 74 model cannot be distinguished statistically on that set. The paper will say this and put the focus on the 26-person test. If SOTA is beaten on only one of the two datasets, the title and abstract must say exactly that.

## 6. Honesty boundaries

- There is no PriorNet code; every component is rewritten from the description; their figures are self-reported in a preprint.
- The SVFAP/MAE-DFER weights belong to the group of Sun et al.; cite the source and licence.
- Version v3 (the calibrated protocol on OUC-CGE) is used as the out-of-domain evaluation chapter, without changing its conclusions.
- Two simulated review panels and code written by AI under the author's direction: declare accurately.

## Addendum of 21/09/2026, after G1 had run 5 epochs and before choosing the G2 lever set

The three points below were found when looking at the first 5 epochs of `g1_svfap_full`; recorded before any lever runs.

**1. The temporal window covers only 20 % of the clip. This is the most important lever and no one has tried it on this benchmark.** The recipe `16 frames × stride 4` is taken from the SVFAP group's own DFEW recipe, and DFEW clips are short, so 16×4 covers nearly all of the clip. EngageNet clips are 10 seconds long, after downsampling ~304 frames remain, so span = (16−1)×4+1 = 61 frames, i.e. **20 % of the clip**. At scoring time, the 5 TTA windows do cover the whole clip, but they are five independent 2-second judgements that are then averaged, not one 10-second judgement; for a slowly varying state such as engagement the two are not equivalent. PriorNet states "16 frames 224×224" for 10-second clips, most likely sampled uniformly over the whole clip. **Lever 1 of G2 is `--stride 20`** (span 301 ≈ 99 % of the clip), run before placeholder / balanced / LDL / LoRA. In addition: at stride 20 augmentation by window shifting almost disappears, so per-frame position jitter of ±stride/2 has been added to `ft_video.py`.

**2. The comparison trap between CV and validation. Every threshold recorded in advance in section 5 must be re-read in the correct units.** The majority-class floor of CV fold 0 is **0,5792**, that of official validation is **0,5313**. The G1 threshold "67–71, stop if < 66" is written in validation units, but every G1/G2 run reports figures on CV. A figure of 0,62 on CV fold 0 is **not** "5 points below SOTA"; it is 4 points above the floor of that fold, which is quite different. Fix: **select configurations by UAR and macro-F1 on CV, not by accuracy**, because those two metrics are robust to class imbalance and comparable across folds. `ft_video.py` now writes `eval_majority_baseline`, `eval_class_counts`, `train_class_counts` and `span_frames` into every `FT_*.json` file so that no one (reviewers included) misreads a CV accuracy as a validation accuracy. Accuracy is comparable with 73,58 only in the single validation run for each finalised configuration.

**3. Choose the model trunk first, then run the levers.** The initial plan hard-wired the G2 levers to SVFAP before G1 said which backbone wins. Fixed: `state/g2.sh` waits for both runs of G1, chooses the trunk by macro-F1, and only then runs the levers on that trunk.

**4. Add a class-balanced sampling lever.** The class distribution of the training set is [1213, 721, 1175, 2322]; in the first three epochs the model predicts a single class. For optimising accuracy CE is usually right, but balanced sampling will likely shorten that phase and raise UAR. Added the flag `--balanced` and put it in the lever list, **without** changing the running configuration.

**Note on G1 progress (not an incident):** epochs 0–2 run in the warm-up phase (`--warmup 3`), so the learning rate has not yet reached its base level and the model predicts one class; epoch 3 is the first epoch at the full learning rate. Loss decreases monotonically 1,154 → 0,868 and the prediction distribution gradually opens up (1 class → 2 → 3 → 4). Epoch 4: acc 0,6254, UAR 0,3743, macro-F1 0,3613. The next check points are epochs 12 and 18; a real sign of failure would be UAR still ≤ 0,28 with predictions in ≤ 2 classes from epoch 12 onwards, or loss plateauing above 0,80.

## Addendum 5, 22/09/2026 — ARCHITECTURE CHANGE: two-stream model. Recorded before G3 finishes and before any validation run.

The author gave full authority to change the model, architecture and backbone, as long as a model or procedure capable of beating SOTA can be proposed. This section records the decision and, more importantly, **fixes the fusion parameters in advance** so that the one-shot validation figure is not contaminated by selection.

### Why change: I had tied my own hands to a single stream

Re-reading the SOTA table by input modality, not by score: **every entry in the table except PriorNet is geometry, not pixels.** Ordinal ST-GCN 71,24 uses 78 landmarks over 300 frames; the ICMI mark 69,10 uses OpenFace gaze + head pose + AU; TCCT-Net 68,91 uses exactly two head-pose signals. I, by contrast, trained a 16-frame pixel backbone and reached 0,6544 on the CV fold. Meanwhile the group's P1 paper already has 1 338-dimensional geometric features for all 11 206 clips, sorted in the deterministic order `sort_values(["subject","session","vid","chunk_idx"])`, matching exactly 7 879 / 1 071 / 2 256 rows — I have checked and was able to align them.

Measured on **exactly the CV fold 0** that the pixel campaign is reporting:

| stream | acc | UAR | macro-F1 |
|---|---|---|---|
| pixel (SVFAP fine-tune, full-clip coverage) | 0,6544 | 0,5814 | 0,5863 |
| geometry (1 338 dimensions, GBDT, trains in a few seconds) | 0,6712 | 0,5588 | 0,5554 |
| fusion, equal weights | 0,6993 | 0,6042 | 0,6072 |

Geometry **alone already beats** the pixel model on accuracy, at a cost of a few seconds instead of 11 GPU hours; pixel is better on macro-F1. The two complement each other, and fusion gains 4,5 accuracy points over pixel.

### Fixed in advance, binding, not to be reversed after seeing results

1. **The fusion weight `w_pixel = 0,5` is fixed in advance** as an unweighted average. The sweep w ∈ {0,2 … 0,8} on fold 0 is **exploratory** and is **not** used to choose w. The reason must be stated explicitly: the peak sits exactly in the middle of the grid and is 1,5 points above both neighbouring points (0,6846 → 0,6993 → 0,6895) in a curve that is otherwise very smooth, so it is almost certainly noise of fold 0 and not structure. Therefore:
2. **The honest fold 0 figure for the fusion stream is about 0,685, not 0,6993.** Every forecast and every threshold must be built on 0,685. The figure 0,6993 may only be cited as an exploratory maximum, together with this warning verbatim.
3. Fusion is probability averaging. Rank fusion was tried and **clearly lost** (0,6144), excluded.
4. No stacking, no learning of fusion weights on the evaluation set at any level.

### Proposed model

**Two-stream engagement recogniser with full-clip coverage.**
- *Stream A, appearance:* SVFAP-B fine-tuned end-to-end, 16-frame window at stride 20 covering 99 % of the clip, EMA, layer-wise lr decay. Contribution: facial appearance dynamics.
- *Stream B, geometry:* 3D landmark and head pose statistics over the whole clip → gradient-boosted trees. Contribution: pose and gaze-direction geometry that pixels at 160 px lose.
- *Fusion:* equal-weight probability averaging, fixed in advance.
- *Ensemble:* multiple seeds and multiple backbones in stream A, and only then fusion with stream B at equal weights.

### Next lever and a cheap test to decide

The current geometry uses only **16 frames** sampled uniformly (`N_FRAMES = 16`, max nframes = 16), i.e. 1,6 fps: blink rate, micro-movements and pose variance are all invisible. Since full-clip coverage in the pixel stream gave +5,1 points, full-clip coverage in the geometry stream is the largest remaining lever.

Decision: **96 frames, not 300.** 300 frames cost 19 times the CPU cost of calling the landmark model, while the extra information above about 10 fps is small. 96 frames cost 6 times, give about 10 fps, enough to see the blink rate (200–400 ms).

**What decides the success or failure of this lever is not the number of frames but the set of statistics.** The current 1 338-dimensional vector is mean/std/min/max over 16 samples; keeping that summarisation and increasing to 96 frames only yields less noisy estimates of the **same** quantities, adding almost nothing. The gain lies in statistics that are **not definable** at 16 frames, so the following must be added: blink rate and inter-blink interval (from zero crossings of the eye aspect ratio); low-band versus high-band energy of yaw/pitch/roll; the fraction of frames with head angular velocity above a threshold and the length of the longest still run; autocorrelation of the pose signal at 0,5 s and 1 s.

**Threshold recorded in advance:** test at 96 frames on about 800 clips spanning both the train part and the eval part of fold 0, using exactly that GBDT, against the geometry mark 0,6712. **Below +0,015**, conclude that 16-frame geometry is already near its ceiling, stop this lever and put the remaining time into seeds and backbones. From +0,015 upwards, run in full, and consider 192 frames in the next round.

### A third stream, almost free

P1 has already cached frozen features in the same deterministic order: MAE-DFER 513 dimensions, MARLIN 1 024 dimensions, ArcFace 512 dimensions. Adding a stream costs only one GBDT fit. P1 found that they complement geometry. Try in the same exploration; take it if it adds +0,01 or more.

### Updated forecast, stated as ranges

Fold 0 fusion after removing the noise peak: about 0,685. Training on 7 879 clips instead of 5 431: +1 to +3. Seed and backbone ensemble: +1 to +2. Full-clip-coverage geometry: unknown, 0 to +3. In total **0,70–0,74, with 73,58 at the top of the range rather than in the middle.** This is a real chance, but the top of the range depends on whether the geometry lever delivers or not. Today I have already lowered the forecast once based on data; raising it just because of one peak on one fold would be exactly the same kind of mistake in the opposite direction.

## Addendum 6, 22/09/2026 — multi-stream results, and one of my levers has failed

### Exploration of every cached stream, on exactly CV fold 0

| stream, alone | dimensions | acc | UAR | macro-F1 |
|---|---|---|---|---|
| 16-frame geometry | 1 338 | 0,6712 | 0,5588 | 0,5554 |
| OpenFace (gaze, head pose, AU — exactly the feature set of the ICMI mark 69,10) | 295 | 0,6548 | 0,5526 | 0,5222 |
| pixel (SVFAP fine-tune, full-clip coverage) | — | 0,6544 | 0,5814 | 0,5863 |
| OpenFace as sequence | 9 280 | 0,6197 | 0,4940 | 0,4343 |
| frozen MAE-DFER | 2 560 | 0,6160 | 0,4742 | 0,4405 |

Probability-averaging fusion: `pixel+geom` 0,6993 acc and macro-F1 **0,6072** (highest); `pixel+geom+ofseq` 0,7042 acc but macro-F1 only 0,5866. OpenFace and MAE-DFER added to `pixel+geom` do not improve it.

**Selection warning, applied to myself:** I have just looked at the maximum of 26 combinations on **one** fold. The difference between 0,6993 and 0,7042 is 0,5 points, within the noise of 2 448 clips. Choosing `pixel+geom+ofseq` because it tops this table would be **exactly the same kind of mistake** that I criticised in `03_integrity/PROTOCOL_AUDIT_2026-09-21.md` when I pointed out that the figure 63,2 is the maximum of a 16-cell grid on validation. Therefore: **the model remains the two-stream `pixel+geom` as fixed in addendum 5**; the 26-combination table is exploratory; to add a third stream it must be selected on fold 1 and fold 2, the two folds not yet used, and only then confirmed.

### Full-clip-coverage geometry lever: my hypothesis was wrong, and my test was underpowered

Wrote `code/geom96.py`: it rebuilds exactly the 1 338-dimensional vector at 96 frames (≈9,6 Hz instead of 1,6 Hz) by importing the P1 paper's own `frame_descriptor` and `temporal_stats`, so the first 1 338 dimensions are directly comparable, plus a 152-dimensional dynamics block comprising signal dip rate and dip depth, the ratio of high-band to low-band energy, spectral centroid, the fraction of fast-motion frames, the longest still run, and autocorrelation at 0,5 s and 1,0 s.

Test results on 919 clips:

| feature set | dimensions | acc | versus (a) |
|---|---|---|---|
| (a) 16 frames, cached | 1 338 | 0,5174 | — |
| (b) 96 frames, same summarisation | 1 338 | 0,5283 | +0,0109 |
| (c) 96 frames, plus dynamics block | 1 490 | 0,5196 | **+0,0022** |

The threshold recorded in advance is +0,015 for (c) versus (a). **(c) reaches +0,0022, missing by far.** More notable: the dynamics block makes things **worse** than only increasing the number of frames, i.e. the 152 dimensions I added are noise rather than signal. My hypothesis that blink rate and the pose frequency spectrum carry engagement information, in the form I computed them, is wrong.

**But my test also has a design flaw, which must be stated clearly:** I split 919 clips into 459 for fitting and 460 for scoring, so the standard error of accuracy is about 0,023 — **larger than the 0,015 threshold I set**. That test never had enough power to distinguish +0,015 from 0. The honest conclusion is **no conclusion yet**, not "negative": it only rules out the possibility that this lever wins big, by something like +5 points as the pixel lever did.

Since the CPU is idle while the GPU runs training, answering it properly is **free**: extraction of all 11 206 clips at 96 frames is running (about one hour), followed by scoring on the **whole** of fold 0 with 2 448 clips, where the standard error is about 0,010 and the +0,015 threshold becomes resolvable. The final verdict is written to `results/G4_FUSED.json` against exactly the threshold recorded in advance, and if it misses, the lever is dropped and may not be rescued by changing the threshold.

### Current queue

`p2x_g2` LDL lever then LoRA → `p2x_g3` finalises the winning recipe, runs three backbones and three seeds, then ensembles → `p2x_g4` fuses pixel with geometry at w = 0,5 fixed in advance, and scores the verdict of the 96-frame lever. `p2x_geom96f` extracts geometry on the CPU in parallel. The G4 figure is the figure I will submit.

## Addendum 7, 24/09/2026 — locking the official model BEFORE opening validation and test

Recorded before any official split is scored. After this section, the model, the number of epochs, the EMA coefficient, the ensemble members and the fusion method may not be changed.

### Evidence from three internal folds (G7, `results/G7_FUSED.json`)

| fold | majority floor | pixel (ensemble/single) | geometry | fusion w=0,5 | fusion − pixel, per-person CI |
|---|---|---|---|---|---|
| 0 | 0,5792 | 0,7047 (4 members) | 0,6712 | 0,6908 | −0,0139 [−0,0380; +0,0077] |
| 1 | 0,4391 | 0,6362 (single) | 0,5865 | 0,6051 | −0,0310 [−0,0723; +0,0067] |
| 2 | 0,4171 | 0,6555 (single) | 0,6716 | 0,6894 | +0,0340 [−0,0130; +0,0888] |

Three points drawn from this:

1. **Fusion with geometry at w = 0,5 no longer has a measurable effect.** All three confidence intervals of the paired difference contain 0, mean −0,004. The reason is visible: addendum 5 fixed w = 0,5 when the pixel stream was still at 0,6544 and geometry at 0,6712, i.e. geometry was stronger; after the number of epochs and the EMA were corrected, the pixel stream rose to 0,6985–0,7047 while geometry stayed at its ceiling of ~0,67 on every fold, so equal weights pull the combination towards the weaker stream. This is exactly what internal CV exists to do: check a decision recorded in advance and correct it **before** touching the official split.
2. **The three folds differ greatly in class distribution** (floor 0,58 / 0,44 / 0,42) because the split is by person and many people have labels heavily skewed towards one class. All lever screening in G2–G6 was done on fold 0, which is the easiest fold; the fold 0 figure of 0,70 is optimistic relative to the three-fold mean of 0,66. Validation (floor 0,53) and test (floor 0,50) lie in between.
3. **MAE-DFER at 12 epochs becomes the best single member on fold 0** (0,7087), the reverse of the ranking at 30 epochs (0,6058), i.e. the long schedule had made it overfit more heavily than SVFAP. MARLIN is dropped: both failures were caused by running out of GPU memory when another user held 10 GB, not by a model fault; two backbones are enough.

### Official model, fixed

- **Members (4):** SVFAP-B LoRA-16 seed 0, 1, 2; MAE-DFER-B LoRA-16 seed 0. Same recipe: 16-frame window at stride 20, 12 epochs, warm-up 2, EMA 0,999, lr 1e-3, batch 16 × accumulation 2, layer-wise decay 0,75, 5-window TTA.
- **Main figure (headline): pixel ensemble**, probability average of the four members.
- **Secondary figure, reported next to the main figure exactly as in addendum 5: fusion w = 0,5** with the 16-frame geometry stream (GBDT), taken from **the same forward passes**, so no split is opened any additional time.
- Both come with 95 % per-person bootstrap confidence intervals, UAR and macro-F1. No tuning of w. No epoch selection on the official split.

### Two openings, each split exactly once

1. Train on **train** (7 879 clips) → score **validation** once. Purpose: to place next to the published table, which consists entirely of validation figures.
2. Train on **train + validation** (8 950 clips) → score **test** once. Purpose: the main claim, compared with the dataset authors' 69,29 under exactly the pooled protocol they used; and 67,61 for the train-only variant, taken from the very members of opening 1 scored on test, also exactly once.

### Commitments on reading the results, written in advance

- All four figures (val ensemble, val fusion, test ensemble, test fusion) are reported intact, even if the secondary figure is higher than the main figure. No swapping of roles after seeing the figures.
- A "beats" claim is made only on test, only if the pixel ensemble > 69,29 **and** 69,29 lies below the confidence interval or the paper states clearly that it lies inside it.
- Forecast, stated as ranges, based on the three-fold mean of 0,66 and the difficulty of each split: validation 0,68–0,71; **test 0,66–0,70, with 69,29 in the middle of the range rather than at the bottom**. I estimate the probability of beating 69,29 at 40–55 %. If it does not beat it, the paper makes no SOTA claim; the remaining contribution is the full-clip-coverage model plus the finding that the 11-person validation table cannot distinguish methods, and the decision whether to publish or not is the author's.

## Addendum 8, 25/09/2026 — OFFICIAL RESULTS. Each split opened exactly once, with exactly the model locked in addendum 7.

Source: `results/G8_OFFICIAL.json`, log `results/g8_official.log`. Four members, 12 epochs, EMA 0,999, stride 20, LoRA-16. 95 % per-person bootstrap confidence intervals, 2 000 resamples.

| Split | Trained on | n / people | floor | **Pixel ensemble (main)** | Fusion w=0,5 (secondary) | Geometry |
|---|---|---|---|---|---|---|
| Validation | train | 1 071 / 11 | 0,5313 | **0,7134** [0,6206; 0,7924] | 0,6863 [0,5933; 0,7744] | 0,6695 |
| Test | train | 2 256 / 26 | 0,5027 | **0,7021** [0,6210; 0,7765] | 0,6990 [0,6122; 0,7777] | 0,6817 |
| Test | train + val | 2 256 / 26 | 0,5027 | **0,7004** [0,6219; 0,7714] | **0,7026** [0,6174; 0,7823] | 0,6848 |

UAR / macro-F1 of the pixel ensemble: val 0,6076 / 0,6289; test train-only 0,5625 / 0,5799; test train+val 0,5604 / 0,5766.

### Comparison with the verified marks

| Comparison | Mark | Ours | Difference |
|---|---|---|---|
| Validation, versus the dataset authors (ICMI) | 69,10 | 71,34 | **+2,24** |
| Validation, versus Ordinal ST-GCN | 71,24 | 71,34 | +0,10 |
| Validation, versus PriorNet (preprint v1, no code) | 73,58 | 71,34 | −2,24 |
| **Test, train only, versus ICMI** | **67,61** | **70,21** | **+2,60** |
| **Test, train+val pooled, versus ICMI under the same protocol** | **69,29** | **70,04** (pixel) / **70,26** (fusion) | **+0,75 / +0,97** |

### How to read, following exactly the commitments written in advance in addendum 7

1. **On test, the point estimate beats every comparison mark under the same protocol**: +2,60 in the train-only variant and +0,75 to +0,97 in the pooled variant. These are the only two figures in the field compared on the 26-person test set apart from the dataset authors themselves.
2. **69,29 lies inside our confidence interval**, and vice versa. The paper must say exactly that: the point estimate is higher, but the benchmark cannot separate the two methods statistically. The confidence interval on test is 15,5 points wide; on validation 17,2 points wide.
3. **On validation, we rank second** in the published table, on par with ST-GCN, 2,24 points below PriorNet. That entire gap fits within a 17-point confidence interval. No claim of beating PriorNet; nor is one needed: that figure exists only in preprint version v1 on 11 people.
4. **The secondary figure is 0,22 points higher than the main figure on pooled test.** Reported intact, no swapping of roles; 0,22 points is noise and the three internal folds have shown that fusion has no measurable effect.
5. Pooling validation into train does **not** improve test (0,7004 versus 0,7021): 1 071 extra clips from 11 people change nothing within the noise. Worth recording because ICMI gained +1,68 from the same operation.

### Forecast versus reality

Forecast recorded in advance: validation 0,68–0,71, test 0,66–0,70, probability of beating 69,29 was 40–55 %. Actual: validation 0,7134 (slightly above the top of the range), test 0,7004–0,7026 (top of the range), beating 69,29 on the point estimate. The forecast was somewhat conservative but not wrong in direction.

### Publication condition self-set in `02_draft/DINH_VI_BAI_MO_HINH.md`: "beat 69,29 on test with a confidence interval". Achieved on the point estimate. The correctly scoped claim is: *higher than every mark under the same protocol on the point estimate, and the first paper to report per-person confidence intervals, thereby demonstrating that this benchmark cannot distinguish the methods ranked on it.*

## Addendum 9, 25/09/2026 — DAiSEE, recorded before opening any of its official splits

**The denominator now matches the official one.** The 146 test clips and 506 train clips that were "missing" turned out to all be on disk as `.mp4`; the cache script only scanned `.avi`. Fixed, and the rest cached. There is no longer any reason to report DAiSEE figures on a mismatched denominator. Subjects taken from `ClipID[:6]` give 109 IDs; they are used only for bootstrap grouping, so the discrepancy with the figure 112 in the documentation does not affect scores; this will be noted.

**Model: transfer unchanged the recipe locked in addendum 7, reselecting nothing on DAiSEE**, exactly as in G4 of the original plan. Four members, 12 epochs, EMA 0,999, LoRA-16. One technical difference to keep the behaviour equivalent: DAiSEE clips are exactly 300 frames, so stride 20 (span 301) falls into the short-clip branch of `windows()` and loses the position jitter during training; use **stride 19** (span 286, 95 % coverage) to go through the same branch as EngageNet. This is keeping the recipe, not tuning parameters. There is no geometry stream for DAiSEE (the P1 cache has only EngageNet), so there is only the pixel ensemble figure.

**Three openings, each split once per configuration:** train → validation; those same models → test (official split protocol, Selim mark 67,48 "as published by the authors, not yet verified", and the with-code mark ResNet+TCN 63,9); train + validation → test (pooled protocol, PriorNet mark 69,06). With per-person CI, UAR, macro-F1, and the prediction distribution to show minority-class recall — the weakness noted for ViBED-Net 73,43 (lowest-class recall 0,25).

**Forecast recorded in advance:** the group's DAiSEE label audit shows the labels are close to noise (adjacent agreement 0,58 versus 0,54 independent), so I expect lower than EngageNet: official test 0,63–0,68, pooled test 0,64–0,69. I estimate the probability of beating 69,06 at 25–35 %; of beating 67,48 at about 45 %. Only if the EngageNet recipe gives below 63,9 is it permitted to select separate hyperparameters on DAiSEE train with internal CV, as the original plan allows.

## Addendum 10, 25/09/2026 — further improvement: landmark-sequence stream. Recorded before extracting data and before training.

### Evidence on where headroom remains (`state/headroom.py`, internal folds only)

| Direction | Measurement | Conclusion |
|---|---|---|
| Smoothing over adjacent clips, window ±1/±2/±4 | −1,2 … −6,3 points on all three folds | **Excluded.** Per-clip predictions are better than the smoothed version, even though adjacent labels agree 68 % |
| Per-person adjustment, label-free version | −3,2 / +0,1 / −2,5 | **Exclude** this approach; the oracle ceiling that knows the true label distribution is +3,3 / +5,9 / +8,3, i.e. the headroom is real but no label-free method reaches it yet. Record in the paper as a measured ceiling, not used |
| Ensemble diversity | error correlation among the 4 members **0,79**; oracle "pick the right member" 0,783 versus mean 0,705 | Adding seeds with the same recipe is useless; members with **different inputs** are needed |

### Chosen direction and why

Ordinal ST-GCN reaches 71,24 on validation **using only sequences of 78 landmarks over 300 frames** — on par with my pixel ensemble, with an entirely different input. My existing geometry stream is 16-frame summary statistics fed into trees (0,67); the failed 96-frame test is a failure of the **summarisation**, not of the modality. A sequence model on per-frame landmarks is something not yet tried, has a published precedent at a level of 71, and is expected to have errors less correlated with pixel than seed with seed.

### Design, fixed in advance

- **Data:** per-frame landmark sequences, **all 300 frames** (no sparse sampling), each frame 204 normalised coordinates + 19 quantities including yaw/pitch/roll; same detector and same `frame_descriptor` as the existing geometry stream. Frames without a face are flagged and filled with the nearest frame.
- **Two architectures to try, selected by three-fold internal CV:** (i) two-layer bidirectional GRU, hidden 256, attention pooling; (ii) 4-layer temporal transformer, d = 256, 4 heads, learned positional encoding, mean pooling. Both: dropout 0,3, AdamW, 40 epochs, EMA, epoch selection on internal folds. Small, trains in a few minutes, can run on CPU if the GPU is busy with DAiSEE.
- **Fusion:** probability averaging with the four-member pixel ensemble, **w = 0,5 fixed in advance** (no tuning, same reason as in addendum 5).

### Gates, recorded in advance — no official split is opened unless they are passed

1. The sequence model alone must reach ≥ 0,68 mean over the three internal folds (old geometry mark 0,64–0,67).
2. Fusion with the pixel ensemble must give **≥ +0,015 accuracy averaged over three folds and be non-negative on at least two folds** relative to the pixel ensemble alone. This is a higher threshold than the earlier geometry lever (+0,015 on one fold) because this time each opening is a cost in claim strength.
3. Only if the gates are passed are validation and test opened **one more time**, for exactly the locked configuration. The paper will report **both configurations** opened (addendum 7 is the "base model", this configuration is the "full model"), stating clearly that each split is scored twice, each time with a configuration locked in advance. There is no third opening.

### Forecast, recorded in advance

Sequence model alone: 0,66–0,70 over three folds (ST-GCN suggests the top of the range is feasible). Fusion: +0,01 to +0,025 if the error correlation with pixel is below 0,6; ≈ 0 if above 0,7. I estimate the probability of passing gate 2 at 40 %. If it does not pass, the official results in addendum 8 are the final results, and this section goes into the paper as a controlled negative experiment.

## Addendum 11, 25/09/2026 — GATE VERDICT for the landmark-sequence stream: NOT PASSED. The official results in addendum 8 are the final EngageNet results.

Source: `results/SEQ_CV.json`, log `results/seq_cv.log`. Three internal folds, epoch chosen from the mean curve (epoch 18 for both architectures), fusion w = 0,5 fixed.

| Architecture | Sequence alone (3 folds) | Mean | Gate 1 ≥ 0,68 | Fusion gain over pixel ensemble (3 folds) | Mean | ≥ 0 | Gate 2: Mean ≥ +0,015 and ≥ 2 folds ≥ 0 |
|---|---|---|---|---|---|---|---|
| Bidirectional GRU | 0,6356 / 0,6106 / 0,6392 | 0,6284 | no | −0,0004 / +0,0151 / +0,0212 | +0,0120 | 2/3 | **no** |
| Temporal transformer | 0,6761 / 0,6001 / 0,6660 | 0,6474 | no | +0,0004 / +0,0043 / +0,0360 | **+0,0136** | **3/3** | **no, short by 0,0014** |

Error correlation between the sequence stream and the pixel ensemble: 0,38–0,55, far lower than the 0,79 among the pixel members. The diversity that addendum 10 expected **is real**.

### How to read, following exactly the recorded rules

1. **The gate is not passed, so the official split is not opened a second time.** `seq_official.py` reads this file itself and refuses to run; checked. The official results in addendum 8 — test 70,04 / 70,26 versus 69,29 — are the final EngageNet results.
2. **The transformer misses gate 2 by exactly 0,0014.** The gain is positive on all three folds, largest on the hardest fold (+3,6 points on fold 2). This is *suggestive* evidence, not sufficient evidence under the standard I set myself before running — and the standard was set precisely so that a suggestive result does not consume a split opening. Lowering the threshold to 0,013 after seeing 0,0136 is exactly what I prohibited in addendum 6: "if it misses, it is dropped, not rescued by changing the threshold".
3. **The sequence stream alone reaches only 0,65**, much lower than the 71,24 of Ordinal ST-GCN. Three possible explanations, not yet distinguishable: their graph-convolution architecture on the landmark structure carries information that a flat GRU/transformer does not have; their 78 MediaPipe landmarks differ from my 68 InsightFace 3D landmarks; or their figure is optimistic because of selection on the 11-person validation. No conclusion on which one.
4. **Goes into the paper as a controlled negative experiment**, with all three figures above: combining the two strongest input families of the leaderboard gives a gain of +0,012 to +0,014 over three internal folds, below the level the design can resolve, even though the two families make clearly different errors.

### If the author wants to continue this direction after submission

Only with a new pre-registration record, and the paper must list every round tried. Well-founded candidates: graph convolution exactly as in ST-GCN instead of a flat sequence model, and fewer epochs (the mean curve peaks at 18/40, i.e. 40 epochs already overfit). Each additional round is one more "try until it passes", which weakens the claim; it must be weighed whether it is worth it when the expected gain still lies within the ±7,5-point confidence interval of the test set.

## Addendum 12, 25/09/2026 — DAiSEE: the EngageNet recipe failed, triggering the fallback clause of addendum 9. Recorded before the sweep.

### Signal from the official opening in progress (G9, ensemble not yet available)

Per member, train → test: acc 0,530 / 0,570 / 0,536 / 0,517; UAR 0,27–0,30 (chance 0,25); predicts only classes 2 and 3; test floor 0,4944. Train → validation: 0,596–0,600 versus floor 0,5689. Training loss is still decreasing at epoch 11 (0,905 → 0,896): **not finished learning**, quite unlike EngageNet with the same recipe (loss 0,63 at epoch 11). Data checked: faces found 100 %, 300 frames/clip, faces 167 px, crops correct. Train labels [34, 213, 2 617, 2 494]: 95 % of the data in two classes, so accuracy is almost a 2-vs-3 problem with prior 51/49.

Clause of addendum 9: "only if the EngageNet recipe gives below 63,9 is it permitted to select separate hyperparameters on DAiSEE train with internal CV". It is below. Further note: the group's old pipeline (frozen features, `NEGCTRL`) also scores DAiSEE at 0,488, **below even the floor** — DAiSEE is where both generations of pipeline fail, and the group's label audit (adjacent agreement 0,58 versus 0,54 chance, κ ≈ 0,09) says the labels are close to noise. Expectations must be low.

### Sweep, fixed in advance

Three person-wise folds (`ClipID[:6]`) on DAiSEE train, screening on fold 0 then confirming the best configuration on folds 1 and 2. Four hypotheses, one run each:

| | Hypothesis | Configuration |
|---|---|---|
| A | not finished learning | locked recipe but **36 epochs** |
| B | LoRA-16 too tight for a harder domain | **full fine-tuning**, 12 epochs, lr 5e-4, layer-decay 0,75 |
| C | both | full fine-tuning, 24 epochs |
| D | the 2-vs-3 signal lies in fine-grained dynamics | locked recipe but **stride 4** (2 s window) |

Everything else kept unchanged: EMA 0,999, batch 16 × 2, 5-window TTA, soft labels 0,1.

### Gate for the second DAiSEE opening, recorded in advance

The winning configuration must reach **≥ 0,60 accuracy averaged over the three internal folds** (train floor 0,488, i.e. a lift of +11, equivalent to the lift of the published marks) **and beat the locked recipe by ≥ 0,03 on the same three folds**. If it does not pass, DAiSEE goes into the paper with exactly the figures of G9, with confidence intervals and prediction distributions, as a fully reported negative result; the paper claims nothing on DAiSEE. If it passes, open exactly once more and report both configurations.

In parallel, extract 300-frame landmark sequences for DAiSEE on the idle CPU: on EngageNet this stream missed the gate by 0,0014, but on DAiSEE the pixel stream is much weaker, so its role may differ; used only if it passes the same kind of gate as in addendum 10.

## Addendum 13, 25/09/2026 — OFFICIAL DAiSEE RESULTS, configuration locked in addendum 7, each split opened exactly once. The figures are recorded as they are, with no explanation beyond what was recorded in advance in addendum 12.

Source: `results/G9_DAISEE_OFFICIAL.json`, log `logs/g9.log` on the server. Four members (SVFAP seed 0/1/2, MAE-DFER seed 0), probability-averaging ensemble, 5-window TTA. 95 % per-person bootstrap confidence intervals, 2 000 resamples.

| Configuration | n / people | Floor | Ensemble | 95 % CI | UAR | macro-F1 | Prediction distribution [0,1,2,3] | Members |
|---|---|---|---|---|---|---|---|---|
| train → validation | 1 429 / 19 | 0,5689 | **0,6004** | [0,5626; 0,6339] | 0,296 | 0,277 | [0, 0, 1 197, 232] | 0,600 / 0,600 / 0,598 / 0,596 |
| train → test | 1 784 / 21 | 0,4944 | **0,5331** | [0,4911; 0,5757] | 0,279 | 0,271 | [0, 0, 1 064, 720] | 0,530 / 0,570 / 0,536 / 0,517 |
| train + validation → test | 1 784 / 21 | 0,4944 | **0,5723** | [0,5319; 0,6122] | 0,297 | 0,282 | [0, 0, 1 299, 485] | 0,572 / 0,578 / 0,563 / 0,548 |

Verified marks on DAiSEE test: ResNet + TCN 63,9; Selim 67,48 (split as stated by the authors, not yet verified); PriorNet 69,06 (pooled train + val). **The locked configuration loses to every mark**, including the lowest, under both protocols: −6,7 points versus 63,9 and −11,8 versus 69,06. The confidence interval does not reach 63,9.

Three things that can be read from the figures themselves, and no more:

1. The two low classes are never predicted in any of the three openings; UAR 0,28–0,30 is close to the chance level of 0,25. The model in effect learns a threshold between class 2 and class 3.
2. The four-member ensemble on test is **not higher** than the best member (0,5723 versus 0,5779; 0,5331 versus 0,5701): the members make the same errors, quite unlike EngageNet, where the ensemble adds.
3. Pooling validation into train adds +3,9 points on test, in the same direction as and larger than the +1,7 of the ICMI mark on EngageNet; here it adds 1 429 clips, i.e. +32 % data, for a model that has not yet finished learning (addendum 12).

The forecast in addendum 9 ("EngageNet recipe, nothing adjusted") did not record a specific figure; the early signal in addendum 12 (0,52–0,57 per member) was right. The fallback clause is running (G11). If G11 does not pass the gate, these are the DAiSEE figures that go into the paper.

### First signal from G11, recorded at 09:33 UTC (hypothesis A, fold 0, not finished)

The locked recipe extended to 36 epochs on fold 0: accuracy peaks at 0,5782 at epoch 20 then decreases monotonically to 0,5568 at epoch 33 while training loss keeps decreasing 0,896 → 0,838. **Longer training does not fix it**: the model overfits before the two low classes are learned. The screening rule takes the last epoch, so A will lose to the 12-epoch reference. Recorded so that no one reads the 0,5782 peak as the result of A.

## Addendum 14, 25/09/2026 — landmark-sequence stream on DAiSEE. Recorded before training, after finishing extraction of 9 068 sequences (0 errors).

Why still try although this stream missed the gate on EngageNet: on EngageNet it had to add to a pixel stream that had already reached 0,69; on DAiSEE the pixel stream is only 0,57 and the four members make the same errors (the ensemble adds nothing, addendum 13), so a stream with a different kind of error has more headroom. The data are already available, the model is small, and it costs no additional opening.

**Design, fixed:** `code/seqmodel_daisee.py`. Same two architectures, same hyperparameters, same augmentation, same way of choosing the epoch from the mean curve as in addendum 10 — nothing adjusted for DAiSEE. Three person-wise folds that **match exactly** the folds of `ft_video.py --dataset daisee --cv-folds 3` (same seed 12345, same way of filtering clips by crop). Fusion w = 0,5 fixed with (i) the locked recipe on the same folds (the `g11_ref_locked*` runs) and (ii) the winning configuration of the G11 sweep, if any.

**Gate, recorded in advance:** use **the gate of addendum 12**, not addendum 10, because any configuration that wants to open DAiSEE a second time must pass through the same door: fusion over three folds reaches **a mean ≥ 0,60 and beats the pixel stream on the same folds by ≥ 0,03**. There is a single second opening for DAiSEE; if both the G11 sweep and the sequence fusion pass, choose the one with the higher three-fold mean and record both clearly. If not passed, the DAiSEE sequence stream goes into the paper as a controlled negative result, in the same way as in addendum 11.

**Forecast, recorded in advance:** sequence alone 0,55–0,60 over three folds (train floor 0,488). Fusion gain +0,01 to +0,04. Probability of passing the gate: 20 %.

## Addendum 15, 26/09/2026 — DAiSEE VERDICT: both the G11 sweep and the sequence stream did NOT PASS the gate. DAiSEE is not opened a second time. The figures in addendum 13 are the paper's final DAiSEE figures. The experimental campaign ends.

Source: `results/G11_DAISEE_SWEEP.json`, `results/SEQD_CV.json`, log `logs/g11.log`, `logs/seqd_cv.log`, `logs/seqd_fuse.log`. Three person-wise folds of DAiSEE train (1 823 / 1 665 / 1 870 clips, 23 people per fold), per-fold floors 0,4931 / 0,5327 / 0,4872.

### Sweep G11 (addendum 12), last epoch, no peak picking

Fold 0 screening: B full fine-tuning 12 epochs **0,5897**; C full 24 epochs 0,5826; locked reference 0,5754; D 2 s window 0,5639; A locked 36 epochs 0,5535. Of the four hypotheses, exactly one beats the reference by more than 1 point. A is lowest: extending training hurts (peak 0,578 at epoch 20, dropping to 0,554). D is below the reference: the signal does not lie in fine-grained dynamics.

Confirmation of B on three folds: **0,5897 / 0,5772 / 0,5251, mean 0,5640**; locked reference 0,5754 / 0,5682 / 0,5144, mean 0,5527; difference **+0,0113**. The gate requires ≥ 0,60 and ≥ +0,03: **fails** both conditions. B still predicts only classes 2 and 3, UAR 0,30.

### Landmark-sequence stream (addendum 14)

| | Sequence alone, 3 folds | Mean | Fusion with locked: gain over 3 folds | Mean | Fusion with B: gain over 3 folds | Mean |
|---|---|---|---|---|---|---|
| GRU | 0,5496 / 0,5357 / 0,5332 | 0,5395 | −0,0022 / −0,0162 / +0,0091 | **−0,0031** | −0,0154 / −0,0132 / +0,0171 | **−0,0038** |
| Transformer | 0,5568 / 0,5333 / 0,5380 | 0,5427 | +0,0044 / −0,0150 / +0,0086 | **−0,0007** | +0,0077 / −0,0060 / +0,0070 | **+0,0029** |

Sequence alone is only 3,5–4 points above the floor, and also predicts only classes 2 and 3. Error correlation with pixel is 0,29–0,50, i.e. the two streams genuinely err in different ways, but fusion adds nothing: the four combinations give −0,004 to +0,003. The forecast recorded in advance (sequence 0,55–0,60, gain +0,01 to +0,04) **was wrong on the optimistic side** in both parts.

### How to read, following exactly the recorded rules

1. DAiSEE is not opened a second time. The figures that go into the paper are those of addendum 13: test 0,533 (train) and 0,572 (train + val), CI [0,532; 0,612], below every verified mark. The paper claims nothing on DAiSEE.
2. The paper must list everything tried on DAiSEE so that no one reads 0,572 as "did not try hard enough": the locked recipe; the four sweep hypotheses; the two sequence-stream architectures and the four fusion combinations. All on internal folds of train, without touching test.
3. One observation consistent across all configurations, worth recording in the discussion at exactly the right strength: **no configuration, pixel or geometry, LoRA or full, learns the two low classes**, and every configuration stops at 0,55–0,59 on internal folds. Combined with the group's label audit (κ ≈ 0,09 between adjacent clips), this is indirect evidence, not a conclusion, that most of the gap to 63,9–69,06 does not lie in the model. It must not be written more strongly than that, and it must be stated clearly that we could not reproduce those marks, so an error on our side is not ruled out.
4. The experimental campaign ends. No further configurations are tried on either dataset. What remains is the writing.
