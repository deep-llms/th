#1
#probe-s3a02-baseline-long
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3

nvidia-smi
sleep 3


CKPT=/opt/dlami/nvme/smoke_test_outputs
TRANS=resources/frequent_translations_llm.json

# Probe S3_a02 — single-token, all available steps
for S in 1500 3250 5500 6500 10000 12500 15000 17500 20000 22500 25000 27500 30000; do
  python diagnostics/anchor_probe2_muse_no_loan_word.py \
    --checkpoints $CKPT/S3_a02/checkpoint-${S} \
    --translations $TRANS \
    --single-token-only \
    --output $CKPT/S3_a02/probe2_single_step${S}_RESULTS.md
  sleep 3
done

# Probe baseline — skip 1500-6500 (already done), only new steps
for S in 10000 12500 15000 17500 20000 22500 25000 27500 30000; do
  python diagnostics/anchor_probe2_muse_no_loan_word.py \
    --checkpoints $CKPT/baseline/checkpoint-${S} \
    --translations $TRANS \
    --baseline --single-token-only \
    --output $CKPT/baseline/probe2_single_step${S}_RESULTS.md
  sleep 3
done
