> English translation of `preregistration/MODEL_V2_DESIGN_2026-09-27.md` (SHA-256 `56b4988b757fd4ec45f8cadb401efa3218ccfb3c7e0d9217554807751cdc82d2`), made on 15 October 2026 for publication with the article. The Vietnamese original is the dated record and is unchanged; where the two differ, the original holds.

# Model design v2 — pre-registration, written 27/09/2026 BEFORE running anything

The author requires: the paper must propose a **model**, not be a measurement paper. This file redesigns the model so that the mechanism is the contribution, and pre-registers every threshold before running. The v1 campaign was closed in addendum 15 of `MODEL_PAPER_DESIGN_2026-09-21.md`; this is a new pre-registration, exactly per the condition set in addendum 11 ("only with a new pre-registration, and the paper must list every round that was tried").

Baseline figures: `FACTS_MODEL_PAPER.md`. Every number in this file is taken from there or from the two new measurements recorded in section 2.

---

## 1. The model's thesis, and why it is a mechanism and not a trick

**Engagement is a slowly varying state, observed through an unreliable sensor.** Two consequences, one component per consequence, and each consequence already has measured evidence from the v1 campaign.

**Consequence 1 — the whole episode must be sampled, and there is a breakpoint between extent and resolution.** The v1 campaign measured: keeping 16 frames while widening the window extent from 61 frames (20 % of the clip) to the whole clip gives **+0,0514 macro-F1**, about eight times larger than the next single lever. That is the **extent** axis at a fixed frame budget. The other axis — **frame budget at a fixed extent** — nobody has measured, including the team. The two axes belong to the same design parameter; measuring both gives a statement that transfers to other papers: for a slowly varying state, covering the whole episode matters more than sampling densely, and the benefit of dense sampling switches off after a breakpoint.

**Consequence 2 — the usefulness of each modality changes with observation quality, so the fusion must be conditional.** v1 evidence: the fusion gain at fixed weight w = 0,5 fluctuates **−0,0139 / −0,0310 / +0,0340** across the three folds, all three paired confidence intervals contain 0; the error correlation between the landmark-sequence stream and the pixel ensemble is 0,38–0,55, compared with 0,79 between the pixel members; and **13 % of EngageNet clips have frames in which no face was found** (mean rate 0,919 per clip, only 87,1 % of clips reach 100 %). Read correctly: **the second modality is not wrong, the combination is wrong.** A constant weight cannot be right both for clean-face clips and for clips with partial face loss.

This is the point I must say plainly: v1 reported fusion and ensembling as two components that **do not pay their cost**. Model v2 does not erase that result; it takes that very result as the motivation, and if v2 wins then the story is the *combination mechanism*, not *adding a modality*.

---

## 2. Two new measurements, made 27/09/2026, that decide feasibility

**SVFAP's positional embedding is a fixed sin-cos, not a learned parameter** (`get_sinusoid_encoding_table` at `repos/SVFAP/modeling_finetune.py:289`, used at line 343, and `detach()` at line 463; the table is generated from `num_patches` derived from `all_frames`). Consequence: **changing the frame budget is changing a constructor parameter**, the position table is regenerated analytically, no checkpoint mismatch, no interpolation needed. The frame-budget axis costs exactly the compute cost, not implementation effort.

**The face cache already stores enough reliability signal; nothing more needs to be extracted.** Each file `crops_face/engagenet/<key>.npz` has `found` (300, bool per frame), `box` (300×4, float32 per frame), `fps`, `hw`. The entire reliability vector in section 3.3 can be computed from this at zero cost.

Server: L4 23 GB, 0 MiB in use, 0 users logged in.

---

## 3. The model, fixed in advance

### 3.1. Appearance stream and **two-axis temporal grid**

SVFAP-B, LoRA rank 16 on q, k, v — kept unchanged because it has been measured (+0,0233 accuracy at the 30-epoch schedule). Change: the frame budget `T` and the stride `s` are chosen so that **the span lies within 280–290 frames in all three configurations**, i.e. the extent stays nearly constant at about 95 % of the clip, and **only the frame budget changes**:

| Configuration | T | s | span = (T−1)s+1 | coverage | frequency within the window |
|---|---|---|---|---|---|
| control | 16 | 19 | 286 | ~95 % | 1,6 Hz |
| B32 | 32 | 9 | 280 | ~93 % | 3,3 Hz |
| B48 | 48 | 6 | 283 | ~94 % | 5,1 Hz |

**Choosing the stride this way is a deliberate decision, not a default.** In v1, stride 20 gives span 301 > 300 frames so **more than 92 % of EngageNet clips fall into the short-clip branch**: one deterministic window, no position jitter, and five identical TTA windows so TTA has no effect. The three configurations above have span ≤ 286 ≤ 300 so **all of them go through the long-clip branch**: random start point, position jitter ±s/2, and five TTA windows that genuinely differ. This also brings EngageNet onto **the same branch as DAiSEE** (span 286 at stride 19), i.e. it also removes the C8 deviation that v1 had to declare.

Free by-product: comparing the control T=16/s=19 with the v1 figure T=16/s=20 gives **a measurement of the effect of the sampling branch**, i.e. it answers exactly the C8 question that was left open.

Everything else keeps the locked recipe: 12 epochs, warm-up 2, EMA 0,999, lr 1e-3, layer-decay 0,75, soft labels 0,1, TTA 5 windows. Batch is reduced and accumulation increased when T is large to keep the effective batch at 32.

### 3.2. The two remaining streams, already available, zero cost

1 338-dimensional geometry + GBDT (0,6712 alone on fold 0) and the landmark-sequence stream of 300 frames × 223 dimensions with a temporal transformer (0,6474 alone, mean over three folds, error correlation with pixel 0,38–0,55). Both have already been extracted for all EngageNet and DAiSEE clips.

### 3.3. **Reliability gate** — the second main component

Per-clip reliability vector `r`, ten-dimensional, computed from the cached `found` and `box`:

1. fraction of frames in which a face was found; 2. length of the longest run of face loss divided by 300; 3. mean face area divided by frame area; 4. standard deviation of face area; 5. mean displacement of the box centre between two consecutive frames; 6. the 95th percentile of that displacement; 7. fraction of frames whose box touches the image border; 8. fps of the clip (the 14 people with 1 000 fps clips are a real group); 9. number of frames decoded; 10. fraction of frames in which the face area is below half of the clip's median.

Gate `g`: MLP 10 → 16 → 3, softmax for the three stream weights. The final probability is a convex combination of the three streams with weights `g(r)`.

**Anti-leakage constraint, and a change from an earlier commitment that must be declared.** Addendum 5 of v1 recorded "no stacking, no learning of fusion weights on the evaluation set at any level". The reliability gate **is** a learned combiner, so this is a change from that commitment and I declare it explicitly. Reason it is acceptable: what the old rule protected is *not tuning weights on the evaluation set*, and that constraint is kept absolutely. The gate is fitted only on **out-of-fold predictions of the train set**, using a person-independent nested scheme: for each internal fold k, the three streams are trained on the two remaining folds and generate predictions for fold k, and the gate is fitted on the union of those out-of-fold predictions. Not a single parameter of the gate sees validation or test. In the official run, the gate is fitted on the out-of-fold predictions of the whole train set and then frozen.

Alongside it, **a simpler variant must be reported next to it**: hard routing by a single threshold (`if face-found rate < τ then use geometry, otherwise use appearance`), τ chosen on the internal folds. If hard routing comes close to the learned gate, then the mechanism is *conditioning*, not *the capacity of the gate*, and the paper must say so.

### 3.4. Two secondary levers, **run only if the two main components pass the gate**

- **CORAL ordinal output** instead of a four-class softmax. Measured motivation: the gap between UAR 0,5625 and accuracy 0,7021 on EngageNet test is large, and on DAiSEE no configuration predicts the two low classes, with UAR stuck at 0,28–0,30. Gaussian LDL soft labels were tried and failed (−0,0016 macro-F1), but LDL is **not** rank-consistent ordinal regression; this is something different and not yet tried.
- **Small temporal pooler** over the segment embeddings, two layers, to model the persistence of the state instead of mean pooling.

---

## 4. Gates, pre-registered. Selection by **UAR and macro-F1**, not by accuracy

Three person-independent internal folds of train, exactly the v1 fold construction (`rng = default_rng(12345)`, `fold = i % 3`). Mean over three folds. Epoch fixed at 12, no epoch selection from the curve.

**This point corrects a v1 deviation.** In v1 I pre-registered the rule of selecting by UAR and macro-F1 but when choosing the training schedule I selected by accuracy instead (addendum C7). This time the rule is the rule: **every configuration-selection decision uses macro-F1 as the primary metric and UAR as the secondary metric; if those two metrics disagree then the configuration is not selected and the paper reports that disagreement.** Accuracy is used only for placing alongside the leaderboard, not for selection.

| Gate | Condition | If it fails |
|---|---|---|
| **G-A, frame budget** | the best budget configuration beats the control T=16/s=19 by **≥ +0,015 macro-F1** mean over three folds **and** is non-negative in ≥ 2 folds | the budget axis enters the paper as a negative experiment; model v2 retains only the reliability gate |
| **G-B, reliability gate** | gated fusion beats **the best single stream by ≥ +0,015 macro-F1** mean over three folds **and** beats **fixed w = 0,5 fusion by ≥ +0,010** | the gate enters the paper as a negative experiment |
| **G-C, opening the official split** | **both G-A and G-B pass** | no third opening; the v1 results are the final results and the paper keeps the old axis |
| **G-D, secondary levers** | each ≥ +0,010 macro-F1 mean and non-negative in ≥ 2 folds, run after G-A and G-B are done | dropped, full numbers reported |

**DAiSEE: pure transfer, no claims, no gate.** Run exactly the locked configuration of v2, report the number with confidence interval and prediction distribution. DAiSEE labels have κ ≈ 0,09 and v1 lost to the lowest baseline by 6,7 points; no expectations are set.

---

## 5. Forecasts, pre-registered

| | My forecast | Probability of passing the threshold |
|---|---|---|
| Budget 16 → 32 frames at fixed extent | +0,010 to +0,030 macro-F1 | passes +0,015: **50 %** |
| Budget 48 frames versus 32 | ≤ 32 frames (the benefit switches off, and 7 879 clips overfit easily) | 48 > 32: **30 %** |
| Reliability gate versus fixed w = 0,5 | +0,010 to +0,030 | passes both thresholds: **45 %** |
| Sampling branch, s=19 versus s=20 at T=16 | −0,005 to +0,015 (the long branch has real augmentation) | — |
| CORAL, if run | accuracy 0 to +0,010; **UAR +0,02 to +0,05** | passes +0,010 macro-F1: 40 % |
| **EngageNet test if both G-A and G-B pass** | **71 to 73** | exceeds 69,29 on the point estimate: **70 %** |

**One thing must be said in advance and must not be forgotten when writing the paper: no model change makes the margin statistically significant on this benchmark.** The width of the confidence interval is determined by the 26 people in the test set, not by the model; it will remain about 15 points. I estimate the probability that the lower bound of the confidence interval exceeds 69,29 at **below 5 %**. This means the paper can make the model the main claim, but the confidence-interval part **must not be dropped** — it moves from the main claim to the limitations section, and the sentence "the point estimate is higher but the benchmark cannot separate them statistically" must still be in the paper.

---

## 6. Cost and the price to pay

Estimate on the idle L4: budget grid 3 configurations × 3 folds ≈ 45 hours; the reliability gate reuses saved predictions so it is nearly free; the two secondary levers ≈ 30 hours if run; the official opening 4 members × 2 protocols ≈ 40 hours; DAiSEE ≈ 20 hours. Total about **115–135 GPU hours, i.e. five to seven days**.

**The integrity price, and this is the author's decision.** The EngageNet test set has been scored twice (once for each training protocol) with the v1 model. Opening it a third time for v2 means the paper **must list all three openings** and the reader must see clearly that model v2 was locked before that opening. This weakens the sentence "each split opened exactly once" and adds a model-development-history section to the paper. In exchange, the paper has a model to propose. There is no way to avoid this price other than not opening a third time.

## 7. If the gates do not pass

If G-A and G-B both fail then **the split is not opened a third time**, the v1 results are the final results, and the paper is forced to return to the measurement axis — because then we have no mechanism we can demonstrate. I record this in advance so that there is no case of looking at the numbers first and only then searching for a way to tell the story.

---

## 8. Interim results, recorded 27/09/2026 at 14:00 UTC — six of nine runs finished. **The frame-budget axis FAILED, and failed clearly.**

Source: `results/FT_v2_T{16,32}_f{0,1,2}_train.json`, log `logs/v2a.log`, viewed with `code/peek_v2a.py` (this script does **not** write `GATE_V2A.json`, so that an interim look cannot be mistaken for a verdict).

| Configuration | fold 0 | fold 1 | fold 2 | Mean macro-F1 | Mean UAR | Mean acc |
|---|---|---|---|---|---|---|
| control T=16, s=19, span 286 | 0,5939 | 0,5535 | 0,6008 | **0,5827** | 0,5658 | 0,6696 |
| B32, T=32, s=9, span 280 | 0,5740 | 0,4721 | 0,5711 | **0,5391** | 0,5321 | 0,6437 |

Difference B32 versus control: macro-F1 **−0,0199 / −0,0814 / −0,0297, mean −0,0437, non-negative in 0/3 folds**. UAR −0,0337. Accuracy −0,0259. The pre-registered threshold is ≥ +0,015 and non-negative in ≥ 2 folds: **failed, and failed by about three times the threshold in the wrong direction**.

**My forecast had the direction entirely wrong.** I pre-registered "+0,010 to +0,030, probability of passing the gate 50 %". Actual −0,0437. This is my most severe forecast error in both campaigns and it must be recorded exactly as such in the paper.

### A trivial explanation was ruled out before interpreting

The first suspicion to check is **whether the pre-training weights were silently dropped when the number of frames changed**. The SVFAP architecture has `temporal_seq_len = num_frames / tubelet_size` (8 at 16 frames, 16 at 32 frames), three stages with `Conv1d` temporal downsampling by a factor of 2 between stages, and a fixed bottleneck summariser of 8 tokens. If any parameter has a shape that depends on the temporal length, loading would produce a shape mismatch.

Check: `ft_video.py:169` uses `load_state_dict(sd, strict=False)` **without filtering by shape**, and PyTorch still raises an error on a shape mismatch even with `strict=False`. The 32-frame run raised no error and reported exactly 4 missing keys (`fc_norm`, `head`), identical to 16 frames. **Conclusion: all pre-training weights were loaded in full; the result is not an artifact of loading.**

### Three explanations, not yet distinguishable, none chosen

1. **Mismatch between pre-training and fine-tuning in temporal scale.** SVFAP was pre-trained at 16 frames, so the downsampling `Conv1d` layers and the bottleneck learned to work on 8 → 4 → 2 temporal tokens. At 32 frames they see 16 → 8 → 4. The weights are shape-compatible but the scale of temporal structure they learned no longer matches. Testable, but requires pre-training again.
2. **Fixed adaptation budget.** LoRA rank 16 on 39 blocks gives 1 058 820 trainable parameters, unchanged at every frame budget, but they must adapt over twice the number of tokens within the same 12 epochs.
3. **Redundancy plus overfitting.** For a slowly varying state, the frames added at 3,3 Hz carry nearly the same information as the adjacent frames at 1,6 Hz, so they consume capacity without adding information.

### What this says about the model's thesis

The way to read these numbers **without** exaggerating: consequence 1 of the thesis in section 1 says the whole episode must be sampled. v1 measured that **extent** has large value (+0,0514 macro-F1). v2 has just measured that **resolution** within the same extent has **no** value, and at 3,3 Hz is harmful. The two halves together form a statement sharper than either: *for engagement, covering the whole episode is what matters, while sampling denser than 1,6 Hz is not, and beyond some level it consumes capacity without adding information.* This is a more complete finding than version v1, and it was obtained precisely because the second axis was measured and failed.

But it is **a finding, not a model component**. The frame-budget axis gives us nothing to propose. So the weight of the model paper now falls entirely on the reliability gate (G-B).

### Two secondary measurements, free, both of value

**Sampling branch (the C8 question that v1 left open).** Control T=16/s=19 (span 286, long-clip branch, with position jitter, five genuinely different TTA windows) versus v1 T=16/s=20 (span 301, short-clip branch, no position jitter, TTA ineffective), same three folds: macro-F1 −0,0022 / −0,0072 / +0,0172, **mean +0,0026**; mean accuracy +0,0062. That is, **the sampling branch has almost no effect**. The C8 limitation that the v1 draft had to declare now has an accompanying measurement, and the measurement says it is small. Note when reporting: seeds are not paired and the v1 runs are a single seed, so this is a rough estimate.

**Reproduction.** The new control reaches mean macro-F1 0,5827 versus 0,5801 for v1 on the same three folds, with a different stride and a different code path. The pipeline reproduces itself.

### Decision on T=48: **run it to completion**

Running fold 0, about 1 500 seconds per epoch, the three folds finish around 04:30 UTC on the 28th. Stopping now because it is already "clearly visible" is exactly arbitrary stopping — precisely the kind of error the pre-registration exists to prevent, and I wrote "no early stopping" into the script itself. Moreover, a three-point grid showing a monotonic decline with frame budget is much stronger evidence than two points.

### A flaw in my own gate design, which must be stated

Section 4 records that gate G-C (opening the official split) requires **both G-A and G-B to pass**. Writing it that way merges two independent components into one condition, and it now also locks out the possibility of using a component that is still alive. I do **not** amend the condition myself after seeing G-A fail — that would be moving the threshold. The right thing to do: run G-B according to exactly its own pre-registered threshold, then present it to the author to decide whether to amend the linking condition, and if it is amended the paper must declare clearly that this is an amendment **made after seeing G-A fail**.

---

## 9. GATE G-B VERDICT: **DOES NOT PASS**. Both main components of v2 failed. The official split is not opened a third time.

Source: `results/GATE_V2B.json`, log `logs/v2b.log`. Three out-of-fold streams on the same three internal folds: pixel (v2 control T=16/s=19), 1 338-dimensional geometry + GBDT refitted per fold, 300-frame landmark sequence (transformer). The router is an MLP 10 → 16 → 3 on the reliability vector, fitted on the out-of-fold predictions of the two remaining folds.

| | fold 0 | fold 1 | fold 2 | Mean macro-F1 |
|---|---|---|---|---|
| pixel alone | 0,5939 | 0,5535 | 0,6008 | **0,5828** |
| geometry alone | — | 0,5293 | 0,5862 | — |
| sequence alone | — | 0,5461 | 0,5658 | — |
| equal fusion of three streams | — | 0,5447 | 0,6044 | — |
| w=0,5 fusion pixel+geometry | — | 0,5419 | 0,6059 | — |
| **per-clip routing** | 0,5852 | 0,5494 | 0,6111 | **0,5819** |
| routing + per-person aggregation | 0,5874 | 0,5548 | 0,5987 | 0,5803 |

Per-clip routing versus **the best single stream: −0,0009**, requires ≥ +0,015. Versus **the best fixed-weight fusion: +0,0033**, requires ≥ +0,010. Non-negative in **1/3 folds**. The variant with per-person aggregation: −0,0024. **Both fail.**

The router is **not** degenerate: the mean weights (pixel, geometry, sequence) are 0,571 / 0,057 / 0,372 in fold 0 and 0,699 / 0,071 / 0,230 in fold 1, with standard deviation 0,10–0,24 so it really does vary from clip to clip. It learned the right thing — nearly dropping geometry, relying on pixel and partly on sequence — and that still does not beat pixel alone.

### How to read it, exactly per the rule recorded in section 4 and section 7

G-A failed and G-B failed, so under gate G-C **the official split is not opened a third time**. The v1 result is the final EngageNet result. No amendment is proposed: the clause in section 4 that I criticised myself for as "merging two independent components" now has no practical meaning, because the second component has also died under its own threshold.

### The most important thing from both campaigns, said in one sentence

**Eleven components were tried and none of them paid its cost**, all with thresholds pre-registered before running: face-loss frame placeholder (+0,0035), ordinal soft labels (−0,0016), class-balanced sampling (−0,0077), 96-frame geometry with a kinematics block (−0,0004), landmark-sequence stream on EngageNet (+0,0136, short by 0,0014), geometry fusion at w = 0,5 (−0,004), four-member ensemble (better than the average member, not better than the best member), 32-frame budget (−0,0437), per-clip reliability routing (−0,0009) and with per-person aggregation (−0,0024), plus two DAiSEE fallback branches.

**Three things worked, and none of the three is adding capacity:** temporal window extent (+0,0514 macro-F1), the 12-epoch training schedule with EMA 0,999 (+0,0208 accuracy), and LoRA rank 16 (+0,0233 accuracy, measured at the 30-epoch schedule). All three are a matter of *what is observed* and *how it is optimised*, not *how much larger the model gets*.

And **three out of three label-free estimators are negative**: smoothing over adjacent clips, a per-person prior from the mean prediction distribution, and the reliability router. The per-person oracle gives +0,058 and the member-selection oracle gives +0,078, but no real estimator reaches them. This is the base rate for judging every further proposal in this space.

### Two things that do not stop

48 frames continues to completion (around 04:30 UTC on the 28th) so that the budget curve has three points. The per-fold geometry predictions are now cached at `preds/geom_gbdt_fold{0,1,2}.npz` so every rerun of G-B takes a few minutes instead of two hours.

---

## 10. CLOSING CAMPAIGN v2, 28/09/2026 by the author's decision

The author chose: accept the test results already obtained, run no further campaign, focus on writing. The official split is not opened a third time. The official results of v1 (addenda 8 and 13 of `MODEL_PAPER_DESIGN_2026-09-21.md`) are the final results for both datasets.

**But the v2 campaign was not useless, and this must go into the paper.** It turns the paper's mechanism claim from one-sided into two-sided:

- v1 measured window **extent** at a fixed frame budget: from 20 % of the clip to the whole clip gives **+0,0514 macro-F1**, about eight times the next single lever.
- v2 measured **frame budget** at a fixed extent (span 280–286 for all three configurations): 16 → 32 → 48 frames gives a **monotonic decline**, fold 0 being 0,5939 → 0,5740 → 0,5426 and fold 1 being 0,5535 → 0,4721 → 0,4450.

Together these form a design principle with a measured breakpoint: *for a slowly varying state, covering the whole episode is what matters; sampling denser than about 1,6 Hz consumes capacity without adding information.* Nobody has measured either of these two axes. This is the paper's mechanism contribution, and it is a principle and not a module.

v2 also yielded three more things, all three of which go into the paper:

1. **Reliability routing fails its gate** (−0,0009 versus the best single stream): conditional combination edges past a constant weight (+0,0033) but does not beat a single stream. The router is not degenerate — mean weights pixel 0,57–0,70, sequence 0,23–0,37, geometry 0,06–0,08, standard deviation 0,10–0,24 — so it learns what it needs to learn and it is still not enough.
2. **The sampling-branch measurement, resolving v1's C8 limitation**: the difference between the long-clip branch (s=19) and the short-clip branch (s=20) is only **+0,0026 macro-F1** mean over three folds. That limitation now has a number attached and the number says it is small.
3. **Reproduction check**: the new 16-frame control reaches mean macro-F1 0,5827 versus 0,5801 for v1 on the same three folds, with a different stride and a different code path.

One numeric cell is still missing: **48 frames fold 2**, running, finishing around 05:50 UTC on the 28th. The gate script computes and writes `GATE_V2A.json` by itself when it finishes. The verdict cannot change: 32 frames already failed at −0,0437 with 0/3 folds non-negative, and 48 frames is even lower in both folds available so far.

**Summary of the two campaigns, in one sentence.** Twelve components tried with pre-registered thresholds, none of them paid its cost. Three things worked — window extent, training schedule, LoRA — none of them is adding capacity. Three out of three label-free estimators are negative, even though the per-person oracle gives +0,058 and the member-selection oracle gives +0,078. The paper must tell all three of these lines in full.

---

## 11. Addendum of 29/09/2026 — where the pre-registration of the variable-separating control is stored

The data-audit round pointed out a defect in **where it is stored**: the draft says the three readings of the control experiment's result were pre-registered, but a reader searching the two design files does not find them — because I recorded them in `FACTS_MODEL_PAPER.md` section 13, which is a working document edited continuously, and not in a dated pre-registration. Acknowledged exactly as such, and copied here so that the record is where the reader will look for it.

**Pre-registered on 28/09/2026, before running T=16/s=9, three readings:**

| Outcome | Read as | The paper's claim becomes |
|---|---|---|
| T=16/s=9 ≈ T=32/s=9 (about 0,54) | frame budget does not matter at a fixed stride; **stride is the deciding variable** | "sparse sampling"; stride is the only substantial lever |
| T=16/s=9 ≈ T=16/s=19 (about 0,58) | stride does not matter at a fixed budget; **frame budget is the deciding variable** | "keep the frame budget small"; stride is not a lever |
| between the two levels | both contribute | the paper decomposes the two parts and states the proportion |

Accompanying forecast: 60 % falls on row one, 20 % row two, 20 % row three. **Actual outcome: row three** — B = 0,5517 lies in between, closer to C than to A. Right direction, wrong row.

The first version of this table labelled the first two rows with "frequency" and "coverage" and **row two was inverted**; the error is recorded at `FACTS_MODEL_PAPER.md` section 13.3. It affects no decision because the outcome fell on row three, and row three was written correctly from the start.
