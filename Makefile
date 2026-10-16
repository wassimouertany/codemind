.DEFAULT_GOAL := help
COMPOSE := docker compose -f docker/docker-compose.yml

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## create venv and install all extras
	uv sync --all-extras
	@mkdir -p data
	@test -f .env || cp .env.example .env

up: ## start qdrant
	$(COMPOSE) up -d

down: ## stop containers
	$(COMPOSE) down

dev: up ## run the api with reload
	uv run uvicorn codemind.main:app --reload --port 8000

ingest: ## make ingest REPO=https://github.com/owner/name
	# onnxruntime 1.30 needs CUDA 13 runtime libs that are not installed;
	# fastembed silently falls back to CPU, so do not pretend otherwise.
	uv run python scripts/ingest_repo.py --repo $(REPO)

test: ## run tests
	uv run pytest

lint: ## ruff + mypy + import contracts
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run mypy
	uv run lint-imports

fmt: ## autoformat
	uv run ruff check --fix src tests
	uv run ruff format src tests

validate-goldset: ## check every gold path is indexed before measuring
	uv run python -m evaluation.validate_goldset

eval-retrieval: validate-goldset ## Recall@k / MRR / nDCG + ablation table
	uv run python -m evaluation.retrieval_eval --ablation

eval-ragas: ## faithfulness / context precision / context recall
	uv run python evaluation/ragas_eval.py

bench-model: ## base vs QLoRA on identical contexts
	uv run python evaluation/model_benchmark.py

.PHONY: help install up down dev ingest test lint fmt validate-goldset eval-retrieval eval-ragas bench-model

status: ## show project progress and what to build next
	@uv run python scripts/status.py

next: ## show the files for the current day
	@uv run python scripts/status.py --next
