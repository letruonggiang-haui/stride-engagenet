> English translation of `preregistration/MODEL_V4_DESIGN_2026-10-04.md` (SHA-256 `02e22447d6479a843b461a083b2c753cec6f8dc51e6fc3baf13bf4d2e9fdff58`), made on 15 October 2026 for publication with the article. The Vietnamese original is the dated record and is unchanged; where the two differ, the original holds.

# Design v4, draft pre-registration, written on 04/10/2026, not approved, no number computed yet

Status: draft. No k-shot number has been computed on the internal folds, because that is exactly the gate's data. When the author approves, rename the file removing the suffix `_DRAFT`, record the sha256 in FACTS, and from then on do not edit.

This is the fourth campaign. The three previous campaigns on the standard EngageNet task (v1 21/09, v2 27/09, v3 02/10) have 13 pre-registered components, none of which passed its gate. A paper written from v4 must list all four rounds.

## 0. Paper framing: this is a different task, not a better model on the standard task

v4 assumes that at deployment there are k clips of that same student labelled by the teacher (an enrollment session). The standard EngageNet task does not allow this. Therefore:

- Not comparable with 67,61, 69,29, 73,58 or any number on the leaderboard. No claim about the leaderboard.
- The only comparison is with the team's own unadjusted model, on exactly the same query clips. This comparison can be paired and tested, because the team has per-clip predictions.
- Prior work that must be mentioned: Churaev & Savchenko (AIED 2024, DOI 10.1007/978-3-031-64315-6_29) fine-tune a binary classifier with a few labelled videos of each user, on their own dataset, and report a 10–15 % gain in accuracy. Alarefah, Kammoun Jarraya & Abuzinadah (Computers 2025, DOI 10.3390/computers14030109) use prototypical-network-style few-shot learning on EngageNet and self-report reaching the highest level; the full text could not be read (the MDPI page blocks automated access), so the protocol and the numbers are not yet known.
- Label-prior adjustment is an existing technique in the literature on label distribution shift; the original reference must be found and checked before writing.

## 1. Thesis and prior evidence

The label distribution differs very strongly between students: the majority-class floor of the three internal folds is 57,92 %, 43,91 % and 41,71 % because students whose labels are concentrated almost entirely on one level were placed in the same fold; per-student accuracy ranges from 0,00 to 1,00 (FACTS section 12.3, recorded 28/09, before v3). An oracle using each student's true label prior gives +0,058 accuracy; the label-free estimator (the student's mean predicted distribution) gives a negative result. Information about each student's prior must therefore come from labels. The cheapest source is a few clips labelled by the teacher.

The +0,058 oracle uses all of the student's clips, including clips from the same recording as the clip being scored, in which adjacent labels coincide 68 % of the time (0,6834 / 0,6513 / 0,6923). Part of the oracle may therefore be only temporal label correlation within the same recording. The protocol below separates the enrollment recording from the query recordings to remove that part.

## 2. Protocol

Data structure, counted on 04/10 from the clip names `subject_<id>_<session>_vid_<video>_<idx>`: train 91 students, 247 recordings, 78 students with two or more recordings; test 26 students, 75 recordings, 24 students with two or more recordings.

- Eligible students: have two or more recordings. Students with only one recording are excluded, and are counted and reported.
- Enrollment recording: the first recording in (session, video) order. Support set: k clips drawn at random without replacement from the enrollment recording (if fewer than k clips, take all). Query set: all clips belonging to the remaining recordings of that student.
- R = 200 support-set draws for each student, `numpy.random.default_rng(0)`, fixed before computing anything.
- Gated k: k = 5. The curve k ∈ {1, 3, 5, 10, 20} is reported, without a gate.

Base predictions:
- Internal folds: branch C of v3 (CMS), three-seed probability mean for each clip, `preds/ft_v3_C_f<k>_s<s>_train.npz`.
- Test: three-seed SVFAP ensemble of v1, `preds/ft_g8_svfap_s{0,1,2}_trainonly_on_test_train.npz` for the train protocol and `preds/ft_g8_svfap_s{0,1,2}_train-val.npz` for the train + val protocol. Do not use the four-member ensemble (with MAE-DFER-B) of the main number of v1, so that the base model on test is of the same kind as the base model used for development on the internal folds.
- π_train: the label distribution of the training set that produced the base model (the other two folds for an internal fold; train, or train + val, for test).

## 3. Methods, fixed in advance

- M0, no adjustment: argmax p(y | x).
- B1, support-set majority: assign the label that occurs most often among the k support clips to every query clip; on a tie, choose the label with the larger π_train.
- M1, proposed model, per-student prior adjustment: p'(y | x) ∝ p(y | x) · π̂_s(y) / π_train(y), with π̂_s(y) = (n_s(y) + α · π_train(y)) / (k + α), n_s(y) is the number of support clips carrying label y. α → ∞ returns M0; α → 0 trusts the support set completely.
- Choose α from the grid {1, 2, 5, 10, 20, 50, 100}: to score fold f, choose the α giving the largest mean macro-F1 at k = 5 on the other two folds (same draw protocol); to score test, choose on all three internal folds. Do not use the labels of the fold being scored.

## 4. Metrics

For each draw: macro-F1, UAR and accuracy pooled over all query clips of the eligible students in the fold; averaged over the 200 draws. Differences: M1 − M0 and M1 − B1. Confidence interval within a fold: student-level bootstrap, B = 2 000, seed 0, the draws held fixed, each bootstrap sample recomputes the mean over draws.

## 5. Gate G-P, on the internal folds, k = 5

M1 passes when all four conditions hold:

1. The three-fold mean of macro-F1 (M1 − M0) ≥ +0,015.
2. In at least two folds, the lower bound of the student-level bootstrap interval for macro-F1 (M1 − M0) > 0.
3. The three-fold mean of macro-F1 (M1 − B1) > 0.
4. The three-fold mean of UAR (M1 − M0) ≥ 0.

## 6. Readings, pre-registered

| Outcome | Read as | Consequence |
|---|---|---|
| M1 passes G-P | the student's prior transfers between recordings, and k = 5 clips are enough to estimate it | score test once for each protocol, Section 7 |
| M1 not better than M0 | the per-student prior does not transfer between recordings; most of the +0,058 oracle is within-recording correlation | no model paper |
| B1 equal to or better than M1 | the teacher's labels carry the information by themselves, the model adds nothing | no model paper |

## 7. If G-P passes: evaluation on test

Apply the locked M1 (α chosen on the three internal folds) to the v1 test predictions of both protocols, same draw protocol, k = 5, R = 200. Main inferential claim: student-level bootstrap interval, paired, 95 %, for accuracy and macro-F1 of (M1 − M0) on the eligible test students, B = 2 000, seed 0. Also report M1 − B1 and the k curve.

Must be declared: this step reads the test labels under a new protocol (k support clips for each student), after the test labels have already been used to score v1. The method and all parameters are locked on the internal folds before reading.

Secondary comparison, only if G-P passes, in the style of Churaev & Savchenko: fine-tune the classification head on the embeddings of the k support clips. This requires extracting embeddings with a few hours of GPU; reported as a comparison model, without a gate.

DAiSEE: pure transfer, no gate, no expectation; recordings taken from `stem[:7]`.

## 8. Forecasts, pre-registered

| Quantity | Forecast | Probability |
|---|---|---|
| macro-F1 (M1 − M0), k = 5, three-fold mean | 0,000 to +0,030 | passes G-P: 30 % |
| M1 > B1 | | 75 % |

Recorded alongside: 13 of 13 previous components failed their gates; the forecasts of v2 and v3 were both skewed toward the optimistic side.

## 9. If G-P fails

Do not read the test labels. No model paper. The internal-fold results are recorded in full, including negative ones. No other variant of M1 is tried on this dataset without a new pre-registration.

## 10. Cost

The entire internal-fold part runs on CPU from the existing prediction files, about one hour. The test part also runs on CPU. Only the secondary comparison in Section 7 needs a GPU.
