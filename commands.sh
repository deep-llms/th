#1 +120+a
#setup-eval-env
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
sleep 3

conda create -n eval python=3.11 -y
sleep 3

LM_EVAL_DIR="/tmp/lm_eval_build"
git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$LM_EVAL_DIR"
conda run -n eval pip install "$LM_EVAL_DIR[hf,vllm,api]"
rm -rf "$LM_EVAL_DIR"

sleep 5
conda activate eval
python -c "import lm_eval; print('lm-eval version:', lm_eval.__version__)"
