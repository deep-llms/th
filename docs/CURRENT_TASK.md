# Current Task — last updated 2026-06-26

> The in-progress / foreground work. Read `PROJECT_NOTES.md` first for background, then this for "where exactly are we." When a task finishes, distill its conclusion into the PROJECT_NOTES timeline and clear this file for the next task.

## Goal

Train S3 alpha variants (alpha=0.2, 0.3, 0.5, 1.0) to test whether stronger hub coupling improves cross-lingual alignment. Previous experiments at alpha=0.05 showed the hub's effect was marginal compared to baseline.

## Status — where we are RIGHT NOW

**Setting up new 8x H200 machine.** Data download in progress on the new EC2 instance. Next: train alpha variants + baseline, run Probe 2 Test B, compare.

## What was completed (previous task, 2026-06-20 → 2026-06-26)

The per-step Probe 2 pipeline was rebuilt from scratch and extended significantly. Full results in `docs/EXPERIMENT_RESULTS.md`.

### Completed steps:

1. **Data download + sampling** on 4x A100 machine. Sharded output to avoid OOM (`prepare_data.py` updated with `--flush-every`). ✓
2. **Trained S3** (alpha=0.05) to step 6500 with token-id dump. ✓
3. **Decoded token IDs → text → per-step word counts** (steps 1500/3250/5500/6500). ✓
4. **Probe 2 with MUSE translations** — ran Test A (anchor weight overlap) + added **Test B** (post-hub embedding cosine similarity at alpha=0.0/0.05/0.1/0.2/0.3). ✓
5. **Built LLM translation pipeline** (`build_translations_llm.py`) using GPT-4o — 4,804 tuples, much higher quality than MUSE (handles multi-word Vietnamese, no truncation). ✓
6. **Probe 2 with LLM translations** — all-words + single-token-only variants. ✓
7. **Trained baseline** (no EmbHub, identical hyperparameters) to step 6500. ✓
8. **Baseline comparison** — ran Probe 2 Test B on baseline checkpoints. ✓

### Code changes made:

- `prepare_data.py`: sharded output (`--flush-every`, default=1) to avoid OOM on limited-RAM machines. No early `os.makedirs` (crash recovery safe). `HF_TOKEN` from env var.
- `train.py` + `smoke_train.py`: shard-aware data loading (detects `shard_*` subdirs, loads each, concatenates). Backward-compatible with old single-dir layout.
- `run_smoke_tests.py`: added S3 alpha variants (`S3_a01` through `S3_a10`). Config updated for current machine.
- `anchor_probe2_muse_no_loan_word.py`: added Test B (post-hub embedding cosine at multiple alphas), `--baseline` flag (loads model without EmbHub, Test B at alpha=0.0 only), `--single-token-only` filter.
- `diagnostics/build_translations_llm.py`: NEW — replaces MUSE with LLM-generated translations. Supports Gemini/OpenAI, batched, resumable, deduplicates across multiple freq files.
- `scripts/setup_env.sh`: NEW — installs miniconda + both conda envs non-interactively.
- `scripts/train_qwen3_0.6b.sh` + `baseline.sh`: updated for 8x H200 (batch=16, grad_accum=4, 160 workers, nvme paths).
- `commands.sh`: EC2 remote runner entry point — the new machine pulls the repo and runs this file.
- `docs/EXPERIMENT_RESULTS.md`: NEW — full results with all numbers from MUSE/LLM/single-token/baseline runs.

### Key findings:

- **Test B gap grows with training** (step 1500→6500): +0.013 → +0.057 on single-token words at alpha=0.05. Highly significant.
- **BUT baseline shows nearly identical growth**: +0.011 → +0.055. The cross-lingual structure comes from the base model, not the hub.
- **Hub's marginal contribution**: ~0.002 above baseline on single-token, within noise on all-words.
- **Gap declines with alpha > 0.05**: expected since model trained at alpha=0.05 only.
- **Single-token measurement is 3-4x cleaner** than all-words (mean-pooling dilutes signal).
- **LLM translations strictly better than MUSE** for this measurement.

### Interpretation / open question:

Alpha=0.05 is very small — the hub barely perturbs the embedding. Higher alpha during TRAINING (not just at test time) could force the model to rely more on the hub for cross-lingual alignment. This is the motivation for the alpha-variant experiments.

**Note:** The earlier short smoke tests (S9/S10, ~1000 steps) found alpha=0.20 measurably hurt LM loss (~4.3 vs ~3.9). S9=S3_a01 (alpha=0.10) and S10=S3_a02 (alpha=0.20) are identical configs. The higher LM loss is actually a **positive signal** — it means the hub is significantly changing the embeddings, unlike alpha=0.05 where the hub contributed nothing (identical loss to baseline). The question is whether that stronger perturbation also improves cross-lingual alignment (Test B gap). A slightly worse LM loss with a larger cross-lingual gap would be the desired trade-off.

## Current pipeline — new machine (8x H200, CUDA 13.0)

### Machine setup:
- Envs: `embeddings_hub` (cu130), `fasttext_env` — installed via `scripts/setup_env.sh`
- Data: `/opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/{train,eval}`
- Checkpoints: `/opt/dlami/nvme/smoke_test_outputs/{S3,baseline,S3_a02,...}`

### Step 0 — Data (in progress):
```bash
# In commands.sh — downloading now
python prepare_data.py download sample \
    --tokenizer-name Qwen/Qwen3-0.6B --num-workers 4 \
    --raw-dir /opt/dlami/nvme/embhub_data/raw \
    --data-dir /opt/dlami/nvme/embhub_data
```

### Step 1 — Train alpha variants + baseline:
```bash
# Alpha variants (sequential, all 8 GPUs each)
python run_smoke_tests.py --arms S3 S3_a02 S3_a03 S3_a05 S3_a10 --save-token-ids --stop-at-step 6500

# Baseline (separate, no smoke CSV so can't use run_smoke_tests.py)
bash scripts/train_qwen3_0.6b_baseline.sh
# Kill after checkpoint-6500 appears
```

### Step 2 — Decode + count words (for each arm):
```bash
# For S3 and each alpha variant
python diagnostics/decode_token_ids.py --input-dir /opt/dlami/nvme/smoke_test_outputs/S3/token_ids
python diagnostics/count_words_from_text.py \
    --input-dir /opt/dlami/nvme/smoke_test_outputs/S3/token_ids \
    --steps 1500 3250 5500 6500 --min-count 5 --workers 160
```

### Step 3 — Build LLM translations:
```bash
python diagnostics/build_translations_llm.py \
    --freq-files /opt/dlami/nvme/smoke_test_outputs/S3/token_ids/frequent_words_step*.json \
    --output temp/frequent_translations_llm.json \
    --provider openai
```

### Step 4 — Run Probe 2 on all arms:
```bash
# For each arm (S3, S3_a02, S3_a03, S3_a05, S3_a10, baseline)
for S in 1500 3250 5500 6500; do
  python diagnostics/anchor_probe2_muse_no_loan_word.py \
    --checkpoints /opt/dlami/nvme/smoke_test_outputs/{arm}/checkpoint-${S} \
    --translations temp/frequent_translations_llm.json \
    --single-token-only \
    --output temp/probe2_{arm}_single_step${S}_RESULTS.md
done
```

### What to look for in results:

Compare Test B gap at alpha=0.0 across arms. If higher-alpha training produces a larger gap than baseline, the hub IS learning cross-lingual alignment — it just needed stronger coupling.

| Arm | Training alpha | Expected gap vs baseline |
|-----|---------------|------------------------|
| baseline | N/A | reference |
| S3 | 0.05 | ~same as baseline (confirmed on 4x A100) |
| S3_a01 | 0.10 | ? (same config as S9, available but not prioritized) |
| S3_a02 | 0.20 | ? |
| S3_a03 | 0.30 | ? |
| S3_a05 | 0.50 | ? |
| S3_a10 | 1.00 | ? |

## Decisions already made (don't re-litigate)

- **LLM translations over MUSE.** GPT-4o produces strictly better translations. MUSE is single-word only with ~60% untranslated Vietnamese.
- **Single-token-only is the primary metric.** All-words mean-pooling dilutes signal; multi-token effects show after transformer layers, not at embedding level.
- **Decode and count are two scripts in two envs by design** — fasttext multiprocessing + transformers = deadlock risk.
- **Test B is the key measurement.** Test A (anchor weight overlap) is alpha-independent and shows weak signal.

## Dead ends (don't repeat)

- `datasets.map` for the decode step — ~3× slower for this workload.
- fasttext batch-predict — API safety under multiprocessing was uncertain; use per-text predict.
- MUSE dictionaries — truncated Vietnamese, wrong translations, ~60% untranslated. Use LLM translations.
- Alpha=0.05 from-scratch with no baseline comparison — can't distinguish hub effect from base model learning.
