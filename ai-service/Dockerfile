FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ai-service/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install -r requirements.txt

COPY ai-service/app.py ./app.py
# The language table, segment schema and alignment maths are shared with the
# Colab service; both import the same package rather than keeping a copy.
COPY dubflow_core ./dubflow_core
COPY frontend ./frontend
RUN mkdir -p /data/jobs \
 && useradd -m -u 1000 app \
 && chown -R app:app /data /app

USER app
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
