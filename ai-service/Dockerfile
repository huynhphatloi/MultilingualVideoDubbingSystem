# =============================================================================
# AI / media service - two build targets share one source tree.
#
#   docker build --target cpu .   -> python:3.11-slim  + torch CPU wheels
#   docker build --target gpu .   -> CUDA 12.1 runtime + torch cu121 wheels
# =============================================================================

# --------------------------------------------------------------- CPU base ---
FROM python:3.11-slim AS base-cpu
ENV PY=python3
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl ca-certificates build-essential libsndfile1 \
        pkg-config libopus-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------- GPU base ---
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS base-gpu
ENV PY=python3 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        ffmpeg git curl ca-certificates build-essential libsndfile1 \
        pkg-config libopus-dev \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ============================================================ shared layer ===
FROM base-cpu AS cpu
ARG TORCH_VARIANT=cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache \
    COQUI_TOS_AGREED=1 \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    PYTHONPATH=/srv

WORKDIR /srv
COPY requirements.txt requirements-cpu.txt ./
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r requirements-cpu.txt \
 && python3 -m pip install -r requirements.txt

RUN mkdir -p /models /data/jobs && useradd -m -u 1000 appuser \
 && chown -R appuser:appuser /models /data /srv
COPY --chown=appuser:appuser app /srv/app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "300", "--workers", "1"]

# ================================================================== GPU =====
FROM base-gpu AS gpu
ARG TORCH_VARIANT=cu128
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache \
    COQUI_TOS_AGREED=1 \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    PYTHONPATH=/srv

WORKDIR /srv
COPY requirements.txt requirements-gpu.txt ./
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install -r requirements-gpu.txt \
 && python3 -m pip install -r requirements.txt

RUN mkdir -p /models /data/jobs && useradd -m -u 1000 appuser \
 && chown -R appuser:appuser /models /data /srv
COPY --chown=appuser:appuser app /srv/app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--timeout-keep-alive", "300", "--workers", "1"]
