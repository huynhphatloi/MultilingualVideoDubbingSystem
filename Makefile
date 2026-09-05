SHELL := /bin/bash
COMPOSE := docker compose
WORKFLOW_JSON := n8n/workflows/simple-dubbing.json
WORKFLOW_ID := $(shell python3 -c "import json;print(json.load(open('$(WORKFLOW_JSON)'))['id'])" 2>/dev/null)

.PHONY: start env import activate stop restart colab kaggle backends logs status check

start: env ## Build and start n8n + the AI service, then import and activate the workflow
	$(COMPOSE) up -d --build --wait
	$(COMPOSE) exec n8n n8n import:workflow --input=/workflows/simple-dubbing.json
	@$(MAKE) --no-print-directory activate
	@echo
	@echo "Demo app:  http://localhost:8000"
	@echo "Pipeline:  http://localhost:5678"

env: .env ## Create .env from .env.example on first run

.env:
	@cp .env.example .env
	@echo "Created .env from .env.example."
	@echo "Paste COLAB_API_URL and COLAB_API_TOKEN from the Colab notebook,"
	@echo "then run 'make restart'. Uploads fail until they are set."

import: ## Re-import the workflow JSON after editing it
	$(COMPOSE) exec n8n n8n import:workflow --input=/workflows/simple-dubbing.json
	@$(MAKE) --no-print-directory activate

activate: ## Activate the workflow and restart n8n so it registers the webhook
	@test -n "$(WORKFLOW_ID)" || { echo "Could not read the workflow id from $(WORKFLOW_JSON)."; exit 1; }
	@if $(COMPOSE) exec n8n n8n update:workflow --id=$(WORKFLOW_ID) --active=true; then \
	  echo "Restarting n8n so the activation takes effect..."; \
	  $(COMPOSE) restart n8n; \
	  $(COMPOSE) up -d --wait n8n; \
	else \
	  echo "Auto-activate unavailable. Open http://localhost:5678, open 'Simple Multilingual Dubbing', save it, and switch it to Active."; \
	fi

backends: ## Show which notebook backends are alive right now
	@curl -fsS http://localhost:8000/backends \
	  | python3 -m json.tool 2>/dev/null \
	  || echo "AI service is not answering. Run 'make start' first."

colab: ## Point the stack at a Colab session: make colab URL=https://... TOKEN=...
	@python3 scripts/set_backend.py colab "$(URL)" "$(TOKEN)"
	@$(MAKE) --no-print-directory restart
	@$(MAKE) --no-print-directory backends

kaggle: ## Point the stack at a Kaggle session: make kaggle URL=https://... TOKEN=...
	@python3 scripts/set_backend.py kaggle "$(URL)" "$(TOKEN)"
	@$(MAKE) --no-print-directory restart
	@$(MAKE) --no-print-directory backends

restart: ## Recreate the AI service to pick up .env or app.py changes
	@$(COMPOSE) up -d --force-recreate --wait ai-service

stop: ## Stop containers but keep jobs and n8n data
	$(COMPOSE) down

logs: ## Follow both service logs
	$(COMPOSE) logs -f --tail=100

status: ## Show service status
	$(COMPOSE) ps

check: ## Offline syntax/configuration checks; does not download models
	@python3 -c "import ast, pathlib; ast.parse(pathlib.Path('ai-service/app.py').read_text())"
	@python3 -c "import ast, pathlib; ast.parse(pathlib.Path('colab/server.py').read_text())"
	@python3 -m json.tool n8n/workflows/simple-dubbing.json >/dev/null
	@python3 -m json.tool colab/ai_service.ipynb >/dev/null
	@python3 scripts/check_contract.py
	@$(COMPOSE) config --quiet
	@echo "checks passed"
