# Reproducible PockNet image pinned to CUDA 12.4.1 runtime on Ubuntu 22.04
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    POCKNET_DATA_ROOT=/workspace/data \
    POCKNET_CHECKPOINT_ROOT=/workspace/checkpoints \
    POCKNET_LOG_ROOT=/workspace/logs \
    PROJECT_ROOT=/workspace/PockNet \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python-is-python3 \
        git \
        build-essential \
        libgl1 \
        curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${PROJECT_ROOT}

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetch the released checkpoint from Hugging Face so every container shares identical weights.
ARG HF_CHECKPOINT_URL=https://huggingface.co/lal3lu03/PockNet/resolve/main/selective_swa_epoch09_12.ckpt
RUN mkdir -p ${POCKNET_CHECKPOINT_ROOT} && \
    curl -L --retry 3 "${HF_CHECKPOINT_URL}" -o ${POCKNET_CHECKPOINT_ROOT}/selective_swa_epoch09_12.ckpt

ENV PYTHONPATH=${PROJECT_ROOT}

ENTRYPOINT ["python","src/scripts/end_to_end_pipeline.py"]
CMD ["--help"]

