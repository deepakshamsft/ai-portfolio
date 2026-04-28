# ProductionCV Makefile
# Purpose: Standardized commands for computer vision model development

.PHONY: help setup train test lint clean serve docker-build docker-run compress deploy-edge

help:
	@echo "ProductionCV — Available Commands:"
	@echo "  make setup        - Create virtual environment and install dependencies"
	@echo "  make train        - Train baseline object detection model"
	@echo "  make compress     - Apply compression pipeline (distill → prune → quantize)"
	@echo "  make test         - Run test suite with coverage"
	@echo "  make lint         - Check code style with black and flake8"
	@echo "  make clean        - Remove generated files and cache"
	@echo "  make serve        - Start Flask API server"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container with API"
	@echo "  make deploy-edge  - Export model for edge deployment (ONNX + TFLite)"

setup:
	python -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt

train:
	python -m src.models --config config.yaml --mode train

compress:
	python -m src.models --config config.yaml --mode compress

evaluate:
	python -m src.evaluate --config config.yaml

deploy-edge:
	python -m src.edge --config config.yaml --export

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
	rm -rf models/checkpoints/*

serve:
	python -m src.api

docker-build:
	docker build -t productioncv-api:latest .

docker-run:
	docker run -p 5000:5000 -p 9090:9090 productioncv-api:latest

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5001

benchmark:
	python -m src.evaluate --config config.yaml --benchmark
