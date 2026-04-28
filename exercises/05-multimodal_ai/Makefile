.PHONY: help install test lint format clean docker-build docker-up docker-down run

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "PixelSmith - Multimodal AI System"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast: ## Run fast tests only
	pytest tests/ -v -m "not slow"

lint: ## Run linters
	black --check src/ tests/
	flake8 src/ tests/ || true

format: ## Format code
	black src/ tests/

clean: ## Clean generated files
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf uploads/* logs/*

docker-build: ## Build Docker image
	docker-compose build

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

run: ## Run API server locally
	python -m src.api

run-dev: ## Run API in development mode
	FLASK_ENV=development python -m src.api

download-models: ## Download models (placeholder)
	@echo "Downloading models..."
	@echo "Models will be downloaded on first use"

setup: install ## Setup environment
	mkdir -p models logs uploads data mlruns
	@echo "Setup complete!"

all: clean install test ## Run all checks
