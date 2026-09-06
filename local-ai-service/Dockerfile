FROM python:3.11-slim

ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    JOBS_ROOT=/data/jobs

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The Colab image receives torch from its runtime. A local image has to supply
# it explicitly. Keeping the wheel source as a build argument lets the GPU
# override select the matching CUDA wheel without maintaining another image.
RUN python -m pip install --upgrade pip \
 && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}"

COPY colab/requirements.txt /tmp/colab-requirements.txt
RUN python -m pip install -r /tmp/colab-requirements.txt

COPY colab /app/colab
COPY dubflow_core /app/dubflow_core

RUN mkdir -p /data/jobs /models/huggingface \
 && useradd -m -u 1000 app \
 && chown -R app:app /data /models /app

USER app
EXPOSE 8000

# Imports remain identical to the notebook service. Models are downloaded and
# loaded lazily on the first request, not while the image is built or started.
CMD ["uvicorn", "colab.server:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
