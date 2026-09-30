.PHONY: help install dev test test-integration lint format migrate api worker ask queue eval up down clean

help:
	@echo "install            install the package"
	@echo "dev                install with dev extras"
	@echo "test               run unit tests (no infrastructure needed)"
	@echo "test-integration   run integration tests (needs DR_TEST_DATABASE_URL)"
	@echo "lint / format      ruff"
	@echo "migrate            apply database migrations"
	@echo "api                run the API on :8100"
	@echo "worker             run the worker"
	@echo "ask Q='...'        run a research question end to end"
	@echo "queue Q='...'      enqueue a run for a worker"
	@echo "eval               run the evaluation harness"
	@echo "up / down          docker compose stack"

install:
	pip install -e .

dev:
	pip install -e ".[dev,observability]"

test:
	pytest -m "not integration"

test-integration:
	pytest -m integration

lint:
	ruff check src tests

format:
	ruff check --fix src tests

migrate:
	python -c "from deep_research.agent import ResearchAgent; ResearchAgent.from_settings()"

api:
	python -m deep_research

worker:
	research-worker

ask:
	research "$(Q)"

queue:
	research "$(Q)" --queue

eval:
	research-eval --dataset eval/dataset.jsonl --limit 2

up:
	docker compose up --build -d

down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache eval/reports build dist
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
