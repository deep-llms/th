#1
#train-alpha-variants
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3
export WANDB_MODE=offline
python run_smoke_tests.py --arms S3_a015 S3_a02 --save-token-ids --stop-at-step 6500
