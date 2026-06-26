FROM pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    pyarrow

COPY hub_layer_v2.py model_wrapper_v2.py train.py prepare_data.py ./
COPY configs/ configs/
