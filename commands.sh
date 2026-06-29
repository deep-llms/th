#1
#probe-baseline
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3

CKPT=/opt/dlami/nvme/smoke_test_outputs
TRANS=resources/frequent_translations_llm.json

# Baseline probe — single-token
for S in 1500 3250 5500 6500; do
  python diagnostics/anchor_probe2_muse_no_loan_word.py \
    --checkpoints $CKPT/baseline/checkpoint-${S} \
    --translations $TRANS \
    --baseline --single-token-only \
    --output $CKPT/baseline/probe2_single_step${S}_RESULTS.md
done
