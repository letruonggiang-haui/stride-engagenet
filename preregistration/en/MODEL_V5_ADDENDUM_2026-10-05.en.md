> English translation of `preregistration/MODEL_V5_ADDENDUM_2026-10-05.md` (SHA-256 `2b2e5633ecf16a142da1e3d34a6146b95a762d974860177991981022ed5e7ebb`), made on 15 October 2026 for publication with the article. The Vietnamese original is the dated record and is unchanged; where the two differ, the original holds.

# Addendum to pre-registration v5, dated 05/10/2026, before the first real run

Pre-registration: `MODEL_V5_DESIGN_2026-10-05.md`, sha256 c13e32416f4df296273040b4692d73d1f91695761bf3c942a2a804de90103994, not modified.

The first smoke test (06:29 UTC) stopped the queue because of a bug in the test itself: with 64 clips, two pretraining epochs finished in 5 seconds, before the step "interrupt after the first checkpoint" had time to cut in, so there was no resumed run to check. The pretraining itself ran correctly: the checkpoint loaded with a full match, two epochs, checkpoint saved, weights written, peak memory 7,14 GB at batch 32.

The only change, purely operational: the pretraining smoke test uses 640 clips instead of 64 and checks for the checkpoint every 2 seconds instead of 15 seconds; the test label is changed to `v5smoke2_*` so as not to mix with the first time. Nothing is changed in the model, the data, the gate or the readings. There is no result number before this line.
