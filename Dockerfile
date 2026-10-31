# CipherTool GPU agent image (Tier 3 upscale on rented GPUs).
#
# Targets:
#   agent    - agent + FFmpeg only, `scale` test engine (CPU). Small; used to test the protocol and the container.
#   seedvr2  - + SeedVR2 v2.5.24 (Python 3.12, torch 2.8 / CUDA 12.8: Ampere, Ada, Hopper, Blackwell).
#   full     - + FlashVSR v1.1 (Python 3.11, torch 2.6 / CUDA 12.4, Block-Sparse-Attention for sm_80 / sm_90).
# Models are not baked in: they download on first use into AGENT_MODEL_DIR (put it on a network volume to keep them).
#
#   docker build --target seedvr2 -t ghcr.io/al3xmuntean/ciphertool-gpu-agent:seedvr2 tools/gpu_agent

ARG CUDA_RUNTIME=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ARG CUDA_DEVEL=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# --- agent ------------------------------------------------------------------------------------------------------------
FROM ${CUDA_RUNTIME} AS agent

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    AGENT_DATA_DIR=/data \
    AGENT_MODEL_DIR=/workspace/models \
    AGENT_PORT=8000

# Ubuntu's FFmpeg is built with NVENC (the driver library is mounted at run time, hence the `video` capability).
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git ca-certificates curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /bin/

ARG RUNPODCTL_VERSION=2.14.0
RUN curl -fsSL -o /usr/local/bin/runpodctl \
      "https://github.com/runpod/runpodctl/releases/download/v${RUNPODCTL_VERSION}/runpodctl-linux-amd64" \
 && chmod +x /usr/local/bin/runpodctl

RUN uv python install 3.12 && uv venv /opt/venv-agent --python 3.12
COPY requirements.txt /opt/agent/requirements.txt
RUN uv pip install --python /opt/venv-agent/bin/python -r /opt/agent/requirements.txt

COPY gpu_agent /opt/agent/gpu_agent
COPY start.sh /opt/agent/start.sh
RUN chmod +x /opt/agent/start.sh

ENV AGENT_ENGINES=scale
EXPOSE 8000
WORKDIR /opt/agent
CMD ["/opt/agent/start.sh"]

# --- seedvr2 ----------------------------------------------------------------------------------------------------------
FROM agent AS seedvr2

ARG SEEDVR2_COMMIT=4490bd1f482e026674543386bb2a4d176da245b9
RUN git init -q /opt/seedvr2 \
 && git -C /opt/seedvr2 remote add origin https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git \
 && git -C /opt/seedvr2 fetch -q --depth 1 origin ${SEEDVR2_COMMIT} \
 && git -C /opt/seedvr2 checkout -q FETCH_HEAD \
 && rm -rf /opt/seedvr2/.git

RUN uv venv /opt/venv-seedvr2 --python 3.12 \
 && uv pip install --python /opt/venv-seedvr2/bin/python \
      torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 \
 && uv pip install --python /opt/venv-seedvr2/bin/python -r /opt/seedvr2/requirements.txt \
 && uv pip freeze --python /opt/venv-seedvr2/bin/python > /opt/versions-seedvr2.txt

ENV AGENT_ENGINES=seedvr2,scale

# --- Block-Sparse-Attention wheel (FlashVSR) ---------------------------------------------------------------------------
FROM ${CUDA_DEVEL} AS bsa-build

ENV DEBIAN_FRONTEND=noninteractive UV_PYTHON_INSTALL_DIR=/opt/python
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.0 /uv /uvx /bin/

ARG BSA_COMMIT=49d6c39e4dc0303442cda3bb758b3925d4399c49
RUN git clone -q https://github.com/mit-han-lab/Block-Sparse-Attention /src/bsa \
 && git -C /src/bsa checkout -q ${BSA_COMMIT} \
 && git -C /src/bsa submodule update -q --init --recursive

RUN uv python install 3.11 && uv venv /opt/venv-build --python 3.11 \
 && uv pip install --python /opt/venv-build/bin/python \
      torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 \
 && uv pip install --python /opt/venv-build/bin/python packaging ninja wheel "setuptools<81" psutil

# The build is memory hungry: few parallel jobs. sm_80 code also runs on sm_86/sm_89 (A6000, 3090, 4090, L40S).
ARG MAX_JOBS=2
RUN cd /src/bsa \
 && MAX_JOBS=${MAX_JOBS} BLOCK_SPARSE_ATTN_CUDA_ARCHS="80;90" \
    /opt/venv-build/bin/python setup.py bdist_wheel -d /wheels

# --- full -------------------------------------------------------------------------------------------------------------
FROM seedvr2 AS full

ARG FLASHVSR_COMMIT=cf910c61a60733e610e9c6e8b607f80c3a6c202b
RUN git init -q /opt/flashvsr \
 && git -C /opt/flashvsr remote add origin https://github.com/OpenImagingLab/FlashVSR.git \
 && git -C /opt/flashvsr fetch -q --depth 1 origin ${FLASHVSR_COMMIT} \
 && git -C /opt/flashvsr checkout -q FETCH_HEAD \
 && rm -rf /opt/flashvsr/.git

COPY --from=bsa-build /wheels /tmp/wheels
RUN uv python install 3.11 && uv venv /opt/venv-flashvsr --python 3.11 \
 && uv pip install --python /opt/venv-flashvsr/bin/python -r /opt/flashvsr/requirements.txt \
      --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match \
 && uv pip install --python /opt/venv-flashvsr/bin/python /tmp/wheels/*.whl \
 && uv pip freeze --python /opt/venv-flashvsr/bin/python > /opt/versions-flashvsr.txt \
 && rm -rf /tmp/wheels

ENV AGENT_ENGINES=seedvr2,flashvsr,scale
