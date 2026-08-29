# ---------------------------------------------------------------------------
# Multilingual Video Dubbing System - developer shortcuts
# ---------------------------------------------------------------------------
SHELL := /bin/bash
COMPOSE := docker compose
GPU_COMPOSE := docker compose -f docker-compose.yml -f docker-compose.gpu.yml
# Prefer the project venv so `make lint` behaves the same as the running service
PY := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

.PHONY: help init up up-docker up-gpu down restart logs logs-ai ps build rebuild \
        import-workflow activate-workflow test-video smoke clean clean-scratch nuke \
        shell-ai shell-n8n native-setup native-run native-dev \
        test lint fmt check verify device clip urls colab-zip bench tidy

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

init: ## Copy .env.example -> .env (once)
	@test -f .env || (cp .env.example .env && echo "created .env - edit HUGGINGFACE_TOKEN before diarization")
	@mkdir -p data/jobs

up: init ## Hybrid (recommended): infra in Docker, AI service native
	./scripts/start.sh

up-docker: init ## Everything in Docker (CPU only on macOS)
	./scripts/start.sh --docker-ai

up-gpu: init ## Everything in Docker on an NVIDIA GPU
	./scripts/start.sh --docker-ai --gpu

native-setup: ## Create .venv and install torch + the AI stack natively
	./scripts/setup-native.sh

native-run: ## Run the AI service natively (Apple GPU when available)
	./scripts/run-native.sh

native-dev: ## Same, with auto-reload
	./scripts/run-native.sh --reload

device: ## Show which device each pipeline component resolved to
	@curl -fsS http://localhost:47800/health/device | (command -v jq >/dev/null && jq . || cat)

clip: ## Re-cut the demo clips from data/test/The-Avengers.mp4
	./scripts/make-clip.sh

colab-zip: ## Package colab/ for upload to a Colab runtime -> dist/colab.zip
	@mkdir -p dist
	@rm -f dist/colab.zip
	@cd colab && zip -q -r ../dist/colab.zip . \
	  -x '*__pycache__*' -x 'results/*' -x 'checkpoints/*' -x 'refs/*' -x 'out/*'
	@echo "dist/colab.zip - drag this into cell 2 of colab/tts_lab.ipynb"

bench: ## Run the TTS benchmark locally (slow without CUDA; use the notebook instead)
	@cd colab && ../$(PY) benchmark.py --language vi $(BENCH_ARGS)

urls:
	@echo ""
	@echo "  Web UI    -> http://localhost:47300"
	@echo "  n8n       -> http://localhost:47678"
	@echo "  FastAPI   -> http://localhost:47800/docs"
	@echo "  MinIO     -> http://localhost:47901"
	@echo ""

down: ## Stop everything (keeps volumes)
	$(COMPOSE) down

restart: ## Restart the ai-service only
	$(COMPOSE) restart ai-service

build: ## Build images
	$(COMPOSE) build

rebuild: ## Rebuild images from scratch
	$(COMPOSE) build --no-cache

logs: ## Tail all logs
	$(COMPOSE) logs -f --tail=100

logs-ai: ## Tail ai-service logs
	$(COMPOSE) logs -f --tail=200 ai-service

ps: ## Show container status
	$(COMPOSE) ps

import-workflow: ## Import n8n workflow JSON into the running n8n container
	./scripts/import-workflow.sh

test-video: ## Generate a synthetic multi-speaker test video into data/samples
	./scripts/make-test-video.sh

smoke: ## Run an end-to-end pipeline smoke test against the AI service
	./scripts/smoke-test.sh

verify: ## End-to-end run on a real clip with the models faked (no downloads)
	@if [ -d .venv ]; then .venv/bin/python scripts/verify_pipeline.py; \
	 else python3 scripts/verify_pipeline.py; fi

test: ## Run the unit tests (no models needed)
	@if [ -d .venv ]; then \
	  .venv/bin/python -c "import pytest" 2>/dev/null || \
	    .venv/bin/pip install --quiet pytest==8.3.4; \
	  .venv/bin/python -m pytest ai-service/tests -q; \
	 else $(COMPOSE) exec ai-service python -m pytest tests -q; fi

check: lint test verify ## Everything a change should pass before you commit
	@printf '\n\033[1;32mlint + tests + end-to-end verification all passed\033[0m\n'

fmt: ## Auto-fix whatever ruff can fix on its own
	@$(PY) -m ruff check --fix --config ai-service/pyproject.toml \
	  ai-service/app ai-service/tests colab

lint: ## Ruff over the AI service
	@if [ -d .venv ]; then \
	  .venv/bin/python -c "import ruff" 2>/dev/null || .venv/bin/pip install --quiet ruff==0.9.2; \
	  .venv/bin/ruff check --config ai-service/pyproject.toml \
	    ai-service/app ai-service/tests colab; fi

shell-ai:
	$(COMPOSE) --profile docker-ai exec ai-service bash

shell-n8n:
	$(COMPOSE) exec n8n sh

clean: ## Remove job artifacts + scratch on the local filesystem
	rm -rf data/jobs/* data/output/* data/verify/*

tidy: ## Delete caches and OS cruft (safe: everything here regenerates)
	@find . -name __pycache__ -type d -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@find . -name .DS_Store -not -path "./.venv/*" -delete 2>/dev/null || true
	@rm -rf ai-service/.pytest_cache ai-service/.ruff_cache colab/results dist
	@echo "tidied"

clean-scratch: ## Drop only the intermediate wavs (safe: they are copies of stored objects)
	@du -sh data/jobs/_scratch 2>/dev/null || true
	rm -rf data/jobs/_scratch/*
	@echo "scratch cleared"

nuke: ## Stop and delete ALL volumes (postgres, minio, n8n, model cache)
	$(COMPOSE) down -v
