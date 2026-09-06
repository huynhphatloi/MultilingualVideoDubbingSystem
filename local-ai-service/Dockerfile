FROM python:3.11-slim

ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG INSTALL_MULTI_VOICE=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Select a CPU or CUDA wheel through the build argument.
RUN python -m pip install --upgrade pip \
 && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"

COPY colab/requirements.txt /tmp/colab-requirements.txt
RUN python -m pip install -r /tmp/colab-requirements.txt

COPY local-ai-service/requirements.multi-voice.txt /tmp/multi-voice-requirements.txt
RUN case "${INSTALL_MULTI_VOICE}" in \
      1|true|yes|on) python -m pip install -r /tmp/multi-voice-requirements.txt ;; \
    esac

COPY colab /app/colab
COPY dubflow_core /app/dubflow_core

RUN mkdir -p /models/huggingface \
 && useradd -m -u 1000 app \
 && chown -R app:app /models /app

USER app
EXPOSE 8000

CMD ["uvicorn", "colab.server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
