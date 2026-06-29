#1
#eval-rerun-failed
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

CKPT=/opt/dlami/nvme/smoke_test_outputs
EVAL=/opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/eval

python eval/eval_parallel.py \
    --checkpoints \
    $CKPT/baseline/checkpoint-6500 \
    $CKPT/S3_a015/checkpoint-3250 \
    --eval-dir $EVAL \
    --bf16 \
    --num-gpus 8
