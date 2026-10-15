> English translation of `preregistration/MODEL_V6_DESIGN_2026-10-06.md` (SHA-256 `0b65307dc848ebe704000662c1081ebf9f8d74df1e0412a16d8788115899307c`), made on 15 October 2026 for publication with the article. The Vietnamese original is the dated record and is unchanged; where the two differ, the original holds.

# Design v6, confirmation round of v5, pre-registration, written on 06/10/2026 before running anything

The author requested on 06/10/2026: "Run a v6 confirmation round because I want to have a model paper". This file fixes every choice before the first run; its sha256 is recorded in FACTS before running. Every change after that is a dated addendum file.

## 0. Why there is a v6, and what must be declared

v5 (`MODEL_V5_DESIGN_2026-10-05.md`, FACTS section 45) failed gate G-D close to the threshold: P − C = +0,0164 macro-F1 three-fold mean (threshold +0,015), positive in all three folds, UAR +0,0190, but only 1/3 folds had a difference exceeding the seed range (2 needed). v6 is done because v5 was close to the threshold, and the paper must say exactly that. This is the sixth campaign; the five previous campaigns have 15 pre-registered components, none of which passed its gate. A paper written from v6 must list all six rounds.

Three constraints so that v6 does not become "try until it passes":
1. The decision is based only on the new runs of v6. Do not pool the v5 numbers into the gate, do not choose between v5 and v6.
2. The gate keeps exactly the G-D rule of v5, no loosening, no change of metric.
3. If v6 fails then stop this direction on EngageNet for good. There is no v7 for the same model.

## 1. Model, unchanged from v5

The proposed model (branch P) is exactly as in section 2 of pre-registration v5: continued MAE-style pretraining of SVFAP-B at the CMS window (16 frames, stride 20, tube mask 0,9, 30 epochs, lr 1e-4, batch 32) on the training clips of the fold, without using labels, then fine-tuning with the CMS recipe (LoRA 16, 12 epochs, EMA 0,999). The code is unchanged from v5: `pretrain_v5.py` sha256 cef6ff9b…, `ft_video.py` sha256 65da774c….

## 2. New runs

- Pretraining: one new run for each fold with seed 1 (v5 used seed 0), label `v6_pre_f<k>`.
- Fine-tuning: seeds 3, 4, 5 for both branches in each fold (v1, v3, v5 used 0, 1, 2).
  - P: `v6_P_f<k>_s<s>`, initialised from `weights/svfap_v5/v6_pre_f<k>.pth`.
  - C (CMS control, new run): `v6_C_f<k>_s<s>`, initialised from the released SVFAP checkpoint.
  - P and C have the same fine-tuning seed so they share the data order, augmentation and LoRA initialisation.
- Total: 3 pretraining runs, 18 fine-tuning runs. No run opens an official split.

## 3. Gate G-D6

Exactly the G-D rule of v5, applied to the 9 new P runs and 9 new C runs:
1. The three-fold mean of the difference (seed mean of P minus seed mean of C) ≥ +0,015 macro-F1.
2. In at least two folds, that difference is positive and larger than the larger seed range of the two branches in that fold.
3. The three-fold mean UAR difference is not negative.

## 4. Readings, pre-registered

| Outcome | Read as | Consequence |
|---|---|---|
| G-D6 passes | the effect of v5 replicates on new pretraining seeds and fine-tuning seeds, at a level beyond seed noise | official evaluation according to section 5 of pre-registration v5, unchanged |
| G-D6 fails, mean difference positive | continued pretraining may help a little but this design cannot separate it from seed noise | stop; report v5 and v6 as an unconfirmed positive result; no model paper |
| G-D6 fails, mean difference ≤ 0 | the +0,0164 of v5 is noise | stop; no model paper |

Secondary report, not part of the gate: the P − C difference pooling v5 and v6 (6 seeds per branch per fold), with a sentence stating that the pooled number is not used for the decision.

## 5. Forecasts, pre-registered

| Quantity | Forecast | Probability |
|---|---|---|
| P − C, macro-F1 three-fold mean | 0,000 to +0,025 | passes G-D6: 30 % |
| mean difference positive | | 75 % |

## 6. Operation

As in v5: checkpoint every epoch, resume after restart, lock, wait for GPU memory, `@reboot` entry removed by itself when done. No smoke test needed because the code is unchanged from v5 and v5 has already passed the smoke test. Order: for each fold, pretraining first, then C and P in turn for each seed. Estimate when the GPU is not shared: about 18 hours.
