#1
#download-and-sample
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3
conda activate embeddings_hub
sleep 3
python prepare_data.py download sample \
    --tokenizer-name Qwen/Qwen3-0.6B \
    --num-workers 4 \
    --raw-dir /opt/dlami/nvme/embhub_data/raw \
    --data-dir /opt/dlami/nvme/embhub_data
