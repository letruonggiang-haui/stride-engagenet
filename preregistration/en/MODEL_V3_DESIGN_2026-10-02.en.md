> English translation of `preregistration/MODEL_V3_DESIGN_2026-10-02.md` (SHA-256 `6843bb9613b99ebb7310657e95cd46faf290c6ce69939241651aa12e2705ec70`), made on 15 October 2026 for publication with the article. The Vietnamese original is the dated record and is unchanged; where the two differ, the original holds.

# Model design v3, pre-registration, written on 02/10/2026 before running anything

Status: draft awaiting the author's approval. Not run even once. The server 192.168.207.98 could not be reached at the time of writing (SSH timed out). When the author approves, record the sha256 of this file in `FACTS_MODEL_PAPER.md`; every change after that must be a dated addendum, not an edit to the text above.

This is the third model proposal on EngageNet. The two previous ones are campaign v1 (`MODEL_PAPER_DESIGN_2026-09-21.md`) and v2 (`MODEL_V2_DESIGN_2026-09-27.md`), with twelve pre-registered components, none of which passed its gate. A paper written from v3 must list all three rounds.

The model has not been named yet. In this file the model is "branch S". It is named only if branch S passes the gate.

## 1. Thesis

EngageNet labels describe the whole 10-second clip. The two recipes already run each meet only half of that requirement.

The inherited recipe (16 frames, stride 4) keeps exactly the pretraining configuration of SVFAP-B, checked from `scripts/voxceleb2/pretrain_svfap_base.sh` of the SVFAP repository: `--num_frames 16 --sampling_rate 4`. But each training sample is one 61-frame window carrying the label of the whole clip. At scoring, five windows spread across the clip have their probabilities averaged (`linspace(0, n − 61, 5)` in `ft_video.py`; `tta_windows` = 5 in `FT_g1_svfap_full_train.json`). The clip is covered only at the scoring step, never at the training step.

The CMS recipe (16 frames, stride 20) puts the whole clip into one computation pass, so the supervision matches the label. But it stretches the window to five times the pretraining configuration.

Branch S does both: it splits the clip into K = 5 windows of 16 frames at stride 4, runs the same backbone on each window, averages the logits of the five windows and then computes the loss against the clip label. This is the segmental consensus principle of TSN (Wang et al., 2016) applied to a 3D video backbone pretrained MAE-style. The novelty lies not in the principle but in the question it answers on engagement.

Sentence that must not be written: branch S "keeps the same time scale in seconds" as pretraining. The frame rate of VoxCeleb2 has not been checked. Only write "keeps the pretraining configuration, 16 frames stride 4".

## 2. Three branches, fixed in advance

Common recipe for all three branches, keeping the locked recipe unchanged: SVFAP-B, LoRA rank 16 on q, k, v, 12 epochs of which 2 warm-up, EMA 0,999, lr 1e-3, soft labels 0,1, random crop 0,7–1,0 and horizontal flip, effective batch 32 clips (not 32 windows). Three internal folds split by person as in every previous campaign (`default_rng(12345)`, `fold = i % 3`). Three seeds 0, 1, 2 for each branch in each fold. All branches run in the same environment (`venv/bin/python`), including branch C, which already has old numbers.

| Branch | Training | Scoring |
|---|---|---|
| R, inherited | one 16 × 4 window per clip per epoch, random start point in [0, n − 61], jitter ±2 frames | five windows `linspace(0, n − 61, 5)`, probability averaging |
| C, CMS | 16 frames stride 20, exactly as in the v1 runs | as in v1 |
| S, segmental consensus | five 16 × 4 windows per clip; window k starts at `linspace(0, n − 61, 5)[k]` plus a uniform integer in ±⌊(n − 61)/8⌋, clamped to [0, n − 61], jitter ±2 frames; one crop region and one flip shared by all five windows of a clip; the logits of the five windows are averaged before the loss is computed | exactly the five windows of R, probability averaging |

R and S are scored on the same windows with the same pooling, so the two branches differ only in how they are supervised during training. This is the cleanest single-factor comparison the project has ever had.

Clips with n < 61 frames go through the existing short branch of `ft_video.py` for both R and S.

Smoke test before the real run: `--max-train 64`, one epoch, only to check memory and tensor shapes. The scores of the smoke test are not read.

## 3. Gate, pre-registered

The selection metric is macro-F1, the secondary metric is UAR. If the two metrics do not point in the same direction, the configuration is not selected and the paper reports that disagreement. Accuracy is only for placing next to the leaderboard.

Gate G-S is the only gate that decides whether there is a model paper. Branch S passes when all three conditions hold:

1. The three-fold mean of the difference (seed mean of S minus seed mean of C) ≥ +0,015 macro-F1.
2. In at least two folds, the seed-mean difference is positive and larger than the larger seed range of the two branches in that fold (seed range = largest minus smallest of the three seeds). The known seed deviation is 0,0152, close to the threshold in condition 1; this condition is kept so that a "pass" is not noise.
3. The three-fold mean UAR difference is not negative.

Branch R has no gate. It is a diagnostic and is always reported.

## 4. Readings, pre-registered

| Outcome | Read as | Consequence |
|---|---|---|
| S passes G-S | supervising the whole clip at the pretraining configuration is better than stretching the window | proceed to Section 6, open the official split |
| S ≈ C (does not pass G-S), both better than R | what decides is clip-level supervision, not the temporal configuration of each computation pass | no model paper; report as a mechanism finding |
| S loses to C | one sparse window covering the whole clip in one computation pass is better than consensus over standard windows; long-range interaction within the same computation pass matters more than the pretraining configuration | CMS stands; no model paper |

The case where S is not better than R must also be reported: then clip-level supervision during training adds nothing over the five-window average at scoring.

## 5. Forecasts, pre-registered

| Quantity | Forecast | Probability |
|---|---|---|
| S − C, macro-F1 three-fold mean | −0,010 to +0,020 | passes G-S: 25 % |
| S − R, macro-F1 three-fold mean | +0,010 to +0,040 | S > R: 65 % |
| C rerun in the new environment compared with the v1 numbers for the same fold, same seed | difference below 0,010 | 70 % |

Recorded alongside: the v2 forecast for the 32-frame budget was wrong even in direction (+0,010 to +0,030, actual −0,0437), and twelve of twelve previous components did not pass their gates. The most likely outcome is no model paper.

## 6. If G-S passes: official evaluation

Conditional secondary component, decided before opening the split: replace logit averaging with an attention pooling layer over the five window embeddings (`forward_features` of SVFAP, with a learned window-order embedding). Gate: ≥ +0,010 macro-F1 over logit averaging, the same conditions 2 and 3 of G-S, three seeds three folds. If it fails, the model is logit averaging. Prior work for this kind of pooling must be found and checked before writing.

Train the locked branch S with three seeds on train and three seeds on train + val. Score validation once (train protocol) and test once for each protocol, with `--open-official`. The main number is the three-seed ensemble of S, probability averaging.

Main inferential claim: person-level bootstrap interval, paired, for the difference (three-seed ensemble of S minus three-seed SVFAP ensemble of v1), on the same 2 256 clips and 26 persons, B = 2 000, seed 0, for accuracy and macro-F1. This is the only paired comparison possible on this dataset, because only the team's own per-clip predictions are available. Must be checked when the server is back: the files `preds/ft_g8_svfap_s{0,1,2}*.npz` are still all present. The differences from 67,61, 69,29 and 73,58 remain differences between point estimates, not tested. Branch C of v1 ran in the old environment; the environment deviation measured in row three of Section 5 must be reported next to the paired interval.

The paper must declare: the EngageNet test set has already been scored by the team for v1, once per protocol; the v3 scoring is the second time for each protocol, with a model locked before scoring.

DAiSEE: pure transfer, locked S recipe, three seeds, two protocols, no gate, no expectation. Branch S does not need a stride change for DAiSEE because the 16 × 4 window does not depend on clip length, so the sampling-branch discrepancy between the two datasets in v1 no longer exists.

## 7. If G-S fails

No official split is opened. No model paper. The results of the three branches are recorded in full and reported, including negative ones. No other variant of branch S is tried on this dataset without a new pre-registration.

## 8. Cost, read from the `seconds` field of the result files

One internal-fold run of 12 epochs at T = 16 takes 3 131–3 144 s when the machine is idle (v2) and 6 176–7 909 s when the machine is shared (g6, g7). Branch S is estimated at about five times that, i.e. 4,4–11 hours per run. This is an unmeasured assumption; the smoke test will give the first number.

| Stage | Number of runs | Estimate |
|---|---|---|
| Internal folds: R 9, C 9, S 9 | 27 | 2,5–6 days |
| Attention pooling, only if G-S passes | 9 | an additional 1,7–4 days |
| Official EngageNet, S: 3 seeds × 2 protocols (one official v1 run takes 4 596–4 831 s) | 6 | 1,5–2 days |
| DAiSEE, S: 3 seeds × 2 protocols | 6 | 1–2 days |

Total if everything is run: about one to two GPU weeks on one L4. If G-S fails, stop after the first stage.

## 9. Things to do before the first run

1. The author approves this file; record the sha256 in FACTS.
2. The server reconnects; check `preds/ft_g8_*` and measure the fraction of EngageNet clips under 281 frames (affects the "uniform windows" sentence about the short branch in the PeerJ paper).
3. Add the flag `--segments K` to `ft_video.py` exactly as in Section 2, with a smoke test. Do not change behaviour when `--segments` is absent, so that all old runs remain reproducible.
