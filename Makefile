SHELL := /bin/bash
COMPOSE := docker compose

.PHONY: start import stop logs status check

start: ## Build and start n8n + the AI service, then import the workflow
	$(COMPOSE) up -d --build --wait
	$(COMPOSE) exec n8n n8n import:workflow --input=/workflows/simple-dubbing.json
	@echo "1. Open http://localhost:5678 and activate 'Simple Multilingual Dubbing'."
	@echo "2. Open the demo app at http://localhost:8000."

import: ## Re-import the workflow JSON after editing it
	$(COMPOSE) exec n8n n8n import:workflow --input=/workflows/simple-dubbing.json

stop: ## Stop containers but keep jobs, models, and n8n data
	$(COMPOSE) down

logs: ## Follow both service logs
	$(COMPOSE) logs -f --tail=100

status: ## Show service status
	$(COMPOSE) ps

check: ## Offline syntax/configuration checks; does not download models
	@python3 -c "import ast, pathlib; ast.parse(pathlib.Path('ai-service/app.py').read_text())"
	@python3 -c "import ast, pathlib; ast.parse(pathlib.Path('colab/server.py').read_text())"
	@python3 -m json.tool n8n/workflows/simple-dubbing.json >/dev/null
	@python3 -m json.tool colab/tts_service.ipynb >/dev/null
	@$(COMPOSE) config --quiet
	@echo "checks passed"
