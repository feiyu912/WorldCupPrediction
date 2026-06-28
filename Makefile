.PHONY: help install install-dev test lint format run api ingest-matches build-features train predict backtest report clean docker-up docker-down

help:
	@echo "football-advance-predictor - targets:"
	@echo "  make install        - install production dependencies via uv"
	@echo "  make install-dev    - install dev dependencies via uv"
	@echo "  make test           - run pytest"
	@echo "  make lint           - run ruff"
	@echo "  make format         - run ruff format"
	@echo "  make run            - run the API locally"
	@echo "  make ingest-matches - ingest local fixture match CSV"
	@echo "  make build-features - build a feature snapshot"
	@echo "  make train          - run training pipeline"
	@echo "  make predict        - produce a single prediction"
	@echo "  make backtest       - run rolling backtest"
	@echo "  make report         - render a backtest report"
	@echo "  make docker-up      - start docker compose stack"
	@echo "  make docker-down    - stop docker compose stack"
	@echo "  make clean          - remove caches and build artifacts"

install:
	uv sync

install-dev:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

run:
	uv run uvicorn football_advance_predictor.app.api.main:app --reload --host 0.0.0.0 --port 8000

ingest-matches:
	uv run football ingest-matches --file data/fixtures/matches.csv

build-features:
	uv run football build-features --match-id MATCH_KO_001 --cutoff 2026-06-29T00:00:00Z

train:
	uv run football train --config configs/mvp.yaml

predict:
	uv run football predict --match-id MATCH_KO_001 --cutoff 2026-06-29T00:00:00Z

backtest:
	uv run football backtest --config configs/backtest.yaml

report:
	uv run football report --run-id RUN_001

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache dist build src/**/__pycache__ tests/**/__pycache__
	find . -name "*.pyc" -delete
