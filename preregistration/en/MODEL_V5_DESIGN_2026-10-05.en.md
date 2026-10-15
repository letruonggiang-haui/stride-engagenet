> English translation of `preregistration/MODEL_V5_DESIGN_2026-10-05.md` (SHA-256 `c13e32416f4df296273040b4692d73d1f91695761bf3c942a2a804de90103994`), made on 15 October 2026 for publication with the article. The Vietnamese original is the dated record and is unchanged; where the two differ, the original holds.

# Design v5, pre-registration, written on 05/10/2026 before running anything

The author requested on 05/10/2026: "I still want the model paper, run the remaining direction". The remaining direction is continued MAE-style pretraining of the backbone at the CMS window. This file fixes every choice and threshold before the first run; its sha256 is recorded in FACTS before running. Every change after that is a dated addendum file.

This is the fifth campaign. The four previous campaigns (v1, v2, v3, v4) have 14 pre-registered components, none of which passed its gate. A paper written from v5 must list all five rounds.

## 1. Thesis

SVFAP-B was pretrained on VoxCeleb2 with 16 frames at stride 4. The best configuration on EngageNet, however, is 16 frames at stride 20 (CMS), i.e. the backbone is fine-tuned at a window five times as long as the window it learned, on a different domain (learning webcams versus celebrity interviews). v3 showed that keeping the pretraining window while splitting the clip into segments loses to CMS in all three folds, so the long window is what needs to be kept. The question of v5: if the backbone relearns self-supervised representations at the CMS window itself and on the target data itself (without using labels), is the model fine-tuned afterwards better or not.

This direction is exactly the third future step that the PeerJ paper wrote down in advance in Article S1, section C.7: "Continue the self-supervised pretraining of the backbone on the target corpus at the frame budget the downstream task will use."

## 2. Model, fixed in advance

Continued pretraining (branch P):
- Initialisation: the released SVFAP pretraining checkpoint, including both the encoder and the decoder (`weights/svfap/svfap_base_voxceleb2_checkpoint-99.pth`; loads with a full match into `PretrainVisionTransformer` with a 512-dimensional encoder, 8 heads, a 384-dimensional decoder, 6 heads, depth 4, checked on 05/10).
- Data: exactly the training clip set of the corresponding fine-tuning run, without using labels. Internal fold f: the clips of the two remaining folds, built with exactly the fold-splitting code of `ft_video.py`. No clip of the fold being scored, of validation or of test goes into pretraining.
- Sampling: 16 frames at stride 20, exactly the training sampler of `ft_video.py` (long branch: random start point and noise of ±10 frames; short branch: `range(0, n, 20)` padded with the last frame); random crop 0,7–1,0, horizontal flip, ImageNet normalisation.
- Tube mask, ratio 0,9; MSE loss on per-patch normalised pixels of the masked tokens, exactly as in SVFAP's `engine_for_pretraining.py`.
- AdamW β = (0,9; 0,95), weight decay 0,05, lr 1e-4, warm-up 2 epochs, cosine to 1e-6, batch 32 clips (may be 16 × accumulation of 2 if memory is insufficient), 30 epochs, seed 0, autocast fp16.
- Output: the adapted encoder, saved in exactly the old checkpoint format so that `ft_video.py` loads it through the new flag `--init-ckpt`.

Fine-tuning: exactly the recipe of branch C of v3 (16 frames at stride 20, LoRA rank 16 on q, k, v, 12 epochs, warm-up 2, EMA 0,999, lr 1e-3, batch 16 × 2, soft labels 0,1, five scoring windows), differing only in the initialisation point. Seeds 0, 1, 2 in each fold.

Control: branch C of v3, nine runs already available, same environment, same code (v3 showed that rerunning branch C matches v1 number for number). Not rerun.

## 3. Gate G-D, the only gate that decides whether there is a model paper

Branch P passes when all three conditions hold, macro-F1 is the selection metric:
1. The three-fold mean of the difference (seed mean of P minus seed mean of C) ≥ +0,015.
2. In at least two folds, that difference is positive and larger than the larger seed range of the two branches in that fold.
3. The three-fold mean UAR difference is not negative.

## 4. Readings, pre-registered

| Outcome | Read as | Consequence |
|---|---|---|
| P passes G-D | adapting the self-supervised representation to the target domain and window improves the fine-tuned model | official evaluation, section 5 |
| P does not differ from C | domain mismatch or window mismatch between pretraining and fine-tuning is not the bottleneck on this dataset | no model paper |
| P loses to C | continued pretraining on about 5 400 clips damages the representation learned from VoxCeleb2 | no model paper |

## 5. If G-D passes: official evaluation

Continued pretraining on the whole of train (for the train protocol) and on train + val (for the train + val protocol), same recipe. Fine-tune three seeds for each protocol. Score validation once (train protocol) and test once for each protocol with `--open-official`. Main number: the three-seed ensemble of P, probability averaging.

Main inferential claim: person-level bootstrap interval, paired, 95 %, for accuracy and macro-F1 of (three-seed ensemble of P − three-seed ensemble of SVFAP of v1, `preds/ft_g8_svfap_s{0,1,2}_*`), on the same 2 256 clips and 26 persons, B = 2 000, seed 0. The differences from 67,61, 69,29 and 73,58 are still differences between point estimates, not tested.

Must be declared: the EngageNet test set was already scored by the team for v1, once per protocol; the v5 scoring is the second time for each protocol, with the model locked before scoring.

DAiSEE: pure transfer, no gate, no expectation; continued pretraining on the train (and train + val) of DAiSEE, window of 16 frames at stride 19 as in the DAiSEE configuration of v1, then fine-tune three seeds per protocol.

## 6. If G-D fails

No official split is opened. No model paper. Results are recorded in full, including negative ones. No other variant (number of epochs, lr, mask ratio, window) is tried without a new pre-registration.

## 7. Forecasts, pre-registered

| Quantity | Forecast | Probability |
|---|---|---|
| P − C, macro-F1 three-fold mean | −0,010 to +0,020 | passes G-D: 15 % |
| P − C negative | | 40 % |

Recorded alongside: 14 of 14 previous components failed their gates; the forecasts of v2 and v3 were skewed toward the optimistic side.

## 8. Operation

Every pretraining and fine-tuning epoch saves a checkpoint and can resume after a restart; the queue has a lock, waits for enough GPU memory, restarts itself through an `@reboot` entry and removes that entry itself when done. Smoke test first: pretraining 2 epochs on 64 clips, interrupt after the first checkpoint then resume; fine-tuning 1 epoch on 64 clips from the checkpoint just created. The scores of the smoke test are not read.
