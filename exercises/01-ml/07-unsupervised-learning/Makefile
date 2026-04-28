# SegmentAI Makefile
# Common commands for development and deployment

.PHONY: help install test lint format run docker-build docker-up clean

help:
	@echo "SegmentAI - Production Clustering System"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make test         Run tests"
	@echo "  make lint         Lint code"
	@echo "  make format       Format code"
	@echo "  make run          Run API server"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up    Start Docker containers"
	@echo "  make clean        Clean temporary files"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/ --line-length=100

run:
	python -m src.api

docker-build:
	docker build -t segmentai:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	rm -rf htmlcov/ .coverage
