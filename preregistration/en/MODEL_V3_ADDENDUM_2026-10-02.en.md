> English translation of `preregistration/MODEL_V3_ADDENDUM_2026-10-02.md` (SHA-256 `8e9d73b762714373247adeea5948bd0a1deb58d8c42f4812aca3bfa56fb6efd0`), made on 15 October 2026 for publication with the article. The Vietnamese original is the dated record and is unchanged; where the two differ, the original holds.

# Addendum to pre-registration v3, written on 02/10/2026 before the first run

Approved pre-registration: `MODEL_V3_DESIGN_2026-10-02.md`, sha256 6843bb9613b99ebb7310657e95cd46faf290c6ce69939241651aa12e2705ec70. That file is not edited. The four points below are operational changes, declared before any number exists. None of them touches the branches, gates, thresholds or readings.

1. Interpreter. Section 2 of the pre-registration states `venv/bin/python`. That statement is factually wrong: checked on the server, campaigns v1 and v2 ran with `~/venvs/engagenet/bin/python` (Python 3.13.13, torch 2.6.0+cu124, timm 1.0.27; `state/v2a_grid.sh`, `state/g7.sh`, `state/g8.sh`). The purpose of the sentence in the pre-registration is that all branches run in the same environment and that branch C is comparable with the v1 numbers. Therefore v3 uses `~/venvs/engagenet/bin/python` for all three branches. The environment check in row three of Section 5 remains unchanged.

2. Saving state to survive restarts. The author requested that state be saved during the run. `ft_video.py` gets an additional flag `--resume`: after each epoch it saves a checkpoint (learned weights, EMA, optimizer, scaler, step, log, random number generator state) to `ckpt/`, written atomically, and resumes from there if the process is interrupted. All v3 runs use this flag. A run that was resumed after an interruption is not bit-identical to an uninterrupted run, because the random state of the data-loading processes cannot be restored. The result file states explicitly the resumed epochs in the field `resumed_at_epochs`, and the paper must report those runs.

3. Waiting for GPU memory. The server is shared. The queue waits for enough free memory before each run so as not to break other jobs running on the same card. This affects only time.

4. Smoke test. Run branch S on 64 clips of fold 0 for two epochs, interrupt after the first checkpoint and then resume, to check memory, the segment code path and the resume path at the same time. The batch size of branch S is chosen here (4 × 8, falling back to 2 × 16 if memory is insufficient), keeping the effective batch at 32 clips. The scores of the smoke test are not read.
