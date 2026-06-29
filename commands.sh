#1
#train-s3a02-long
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3
export WANDB_MODE=offline
export NCCL_NVLS_ENABLE=0


nvidia-smi
sleep 3

# Move old S3_a02 results to _old
mv /opt/dlami/nvme/smoke_test_outputs/S3_a02 /opt/dlami/nvme/smoke_test_outputs/S3_a02_old
mv /opt/dlami/nvme/smoke_test_outputs/S3_a02.log /opt/dlami/nvme/smoke_test_outputs/S3_a02_old.log
sleep 3

python run_smoke_tests.py --arms S3_a02 --save-token-ids --stop-at-step 30000
