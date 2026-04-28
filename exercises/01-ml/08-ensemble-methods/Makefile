# EnsembleAI Makefile
# Purpose: Standardized commands for development and production

.PHONY: help setup train test lint clean serve docker-build docker-run

help:
	@echo "EnsembleAI — Available Commands:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make train        - Train ensemble models with config.yaml"
	@echo "  make test         - Run test suite with coverage"
	@echo "  make lint         - Check code style with black and flake8"
	@echo "  make clean        - Remove generated files and cache"
	@echo "  make serve        - Start Flask API server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt

train:
	python -m src.models --config config.yaml

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	black src/ tests/ --check
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf logs/*.log

serve:
	python -m src.api

docker-build:
	docker build -t ensembleai-api:latest .

docker-run:
	docker run -p 5000:5000 ensembleai-api:latest

mlflow-ui:
	mlflow ui --backend-store-uri file:./mlruns

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f ensembleai-api
