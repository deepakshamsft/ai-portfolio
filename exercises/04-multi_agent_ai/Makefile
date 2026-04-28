# Makefile for multi-agent system

.PHONY: help install test lint format docker-build docker-up docker-down run clean

help:
	@echo "Multi-Agent System - Available Commands"
	@echo "========================================"
	@echo "install       - Install dependencies"
	@echo "test          - Run tests"
	@echo "lint          - Run linting"
	@echo "format        - Format code"
	@echo "docker-build  - Build Docker image"
	@echo "docker-up     - Start Docker containers"
	@echo "docker-down   - Stop Docker containers"
	@echo "run           - Run API server locally"
	@echo "clean         - Clean temporary files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v

lint:
	flake8 src/ tests/ --max-line-length=100
	black --check src/ tests/

format:
	black src/ tests/

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

run:
	python src/api.py

run-dev:
	FLASK_ENV=development python src/api.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

# Example task execution
example:
	curl -X POST http://localhost:5000/task \
		-H "Content-Type: application/json" \
		-d '{"description": "Research and analyze machine learning deployment patterns"}'

# Health check
health:
	curl http://localhost:5000/health

# Get metrics
metrics:
	curl http://localhost:5000/metrics
