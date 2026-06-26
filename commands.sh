#1
#verify-env
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate embeddings_hub
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)"
