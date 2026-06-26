# Current Task — last updated 2026-06-20

> The in-progress / foreground work. Read `PROJECT_NOTES.md` first for background, then this for "where exactly are we." When a task finishes, distill its conclusion into the PROJECT_NOTES timeline and clear this file for the next task.

## Goal

Build a **cleaner version of Probe 2** whose per-language "frequent word" lists contain only words the model **actually saw by a given S3 checkpoint step** (1500 / 3250 / 5500 / 6500), instead of full-corpus frequencies. This makes the cross-lingual anchor-overlap measurement faithful to what the model had actually been trained on at each checkpoint. Serves the from-scratch Probe 2 story in PROJECT_NOTES.

## Status — where we are RIGHT NOW

**Blocked by machine loss (again).** The training machine was lost, taking with it the S3 checkpoints, the dumped token IDs, and the decoded text. **All runtime state is gone; the analysis code is intact in git and verified.** So we are rebuilding from the start of the pipeline: re-download data → re-train S3 (with token-id dump) → re-decode → re-count → Probe 2.

## Pipeline + commands (run in order on the rebuilt training machine)

Env names below (`embeddings_hub`, `fasttext_env`) must match the conda envs on the machine. Paths assume the default smoke output base `/opt/dlami/nvme/smoke_tests`; set once:
```bash
TID=/opt/dlami/nvme/smoke_tests/S3/token_ids   # ids dumps + decoded text + per-step word counts
CKPT=/opt/dlami/nvme/smoke_tests/S3            # checkpoint-1500, -3250, -5500, -6500
```

**0. Data** — env `embeddings_hub`, one-time, ~175 GB (run in tmux/nohup):
```bash
python prepare_data.py download sample --tokenizer-name Qwen/Qwen3-0.6B --num-workers 4
```

**1. Train S3 + dump token IDs** — env `embeddings_hub`, 8 GPUs:
```bash
python run_smoke_tests.py --arms S3 --save-token-ids --stop-at-step 6500
```
Produces `$CKPT/checkpoint-{1500,3250,5500,6500}` (save_steps=250) and `$TID/ids_rank{0..7}.jsonl` + `meta.json` = the exact training sequences in order (`rank = local_process_index`; 64 seq/step/rank → line *i* in a rank file belongs to step `i // 64`).

**2. Decode IDs → text** — env `embeddings_hub`:
```bash
python diagnostics/decode_token_ids.py --input-dir $TID
```
Writes `$TID/text_rank{N}.jsonl`, same line order. ⚠️ Committed version is SINGLE-PROCESS (slow); the ~3× faster multiprocessing.Pool version is NOT in git (see Decisions).

**3. Language-ID + per-step word counts** — env `fasttext_env`:
```bash
python diagnostics/count_words_from_text.py --input-dir $TID --steps 1500 3250 5500 6500 --min-count 5 --workers 64
```
Writes `$TID/frequent_words_step{N}.json` (cumulative). Sanity check: at step 1500, `en` should be ~95% of sequences — if not, the decode/order is wrong, stop and debug.

**4–5. Per step: build MUSE pairs → run Probe 2** — env `embeddings_hub` (step 5 needs a GPU). Each step uses ITS OWN word list and the matching checkpoint:
```bash
for S in 1500 3250 5500 6500; do
  python diagnostics/build_frequent_translations.py \
    --freq-file $TID/frequent_words_step${S}.json \
    --output temp/frequent_translations_step${S}.json
  python diagnostics/anchor_probe2_muse_no_loan_word.py \
    --checkpoints $CKPT/checkpoint-${S} \
    --translations temp/frequent_translations_step${S}.json \
    --output temp/probe2_step${S}_RESULTS.md
done
```
Then compare the JS-gap to the prior full-corpus Probe 2 v3 numbers recorded in PROJECT_NOTES (+0.0041 / +0.0042 / +0.0038 at 1500/3250/5500). The old `temp/probe2_muse_*` files were lost with the machine — compare against those documented numbers, not files.

## Decisions already made (don't re-litigate)

- **Decode is parallelized with `multiprocessing.Pool`, NOT `datasets.map`.** Benchmarked: `datasets.map` is ~3× *slower* here (Arrow cache I/O + fork-storm on 2048-int rows; it even drops below single-process past ~8 procs). ⚠️ **But the Pool version is NOT in git** — the committed `decode_token_ids.py` is the slower single-process one. Re-implement the Pool version (or accept single-process) before the next decode.
- **Decode and count are two scripts in two envs by design** — fasttext runs under multiprocessing, and importing transformers in the same process tree risks a fork-after-threads deadlock. Keep them separate.
- Inside fasttext workers: plain per-text `predict` (well-tested API), each worker loads its own model. Not batch-predict.

## Where we are in the pipeline

At **step 0** (re-downloading data after the machine loss). Run steps 0 → 5 above in order; nothing downstream exists yet. Update this line as steps complete.

## Dead ends (don't repeat)

- `datasets.map` for the decode step — ~3× slower for this workload.
- fasttext batch-predict — API safety under multiprocessing was uncertain; use per-text predict.
