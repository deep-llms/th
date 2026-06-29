#1
#eval-all
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate eval
sleep 3

nvidia-smi
sleep 3


CKPT=/opt/dlami/nvme/smoke_test_outputs
EVAL=/opt/dlami/nvme/embhub_data/Qwen_Qwen3-0.6B/eval

python eval/eval_parallel.py \
    --checkpoints \
    $CKPT/baseline/checkpoint-1500 \
    $CKPT/baseline/checkpoint-3250 \
    $CKPT/baseline/checkpoint-5500 \
    $CKPT/baseline/checkpoint-6500 \
    $CKPT/S3_a015/checkpoint-1500 \
    $CKPT/S3_a015/checkpoint-3250 \
    $CKPT/S3_a015/checkpoint-5500 \
    $CKPT/S3_a015/checkpoint-6500 \
    $CKPT/S3_a02/checkpoint-1500 \
    $CKPT/S3_a02/checkpoint-3250 \
    $CKPT/S3_a02/checkpoint-5500 \
    $CKPT/S3_a02/checkpoint-6500 \
    --eval-dir $EVAL \
    --bf16 \
    --num-gpus 8
