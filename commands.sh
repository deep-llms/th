#1
#probe-alpha-variants
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3

CKPT=/opt/dlami/nvme/smoke_test_outputs
TRANS=resources/frequent_translations_llm.json

# Probe 2 Test B — single-token, for both alpha variants
for ARM in S3_a015 S3_a02; do
  for S in 1500 3250 5500 6500; do
    python diagnostics/anchor_probe2_muse_no_loan_word.py \
      --checkpoints $CKPT/${ARM}/checkpoint-${S} \
      --translations $TRANS \
      --single-token-only \
      --output $CKPT/${ARM}/probe2_single_step${S}_RESULTS.md
  done
done
