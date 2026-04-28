.PHONY: help install test run docker-build docker-up docker-down clean

help:
	@echo "PizzaBot - RAG Chatbot Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run tests with coverage"
	@echo "  run           - Run Flask development server"
	@echo "  run-prod      - Run with Gunicorn (production)"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-up     - Start Docker Compose services"
	@echo "  docker-down   - Stop Docker Compose services"
	@echo "  clean         - Clean generated files"
	@echo "  format        - Format code with Black"
	@echo "  lint          - Run linting checks"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

run:
	python -m src.api

run-prod:
	gunicorn --bind 0.0.0.0:5000 --workers 4 --timeout 120 src.api:app

docker-build:
	docker build -t pizzabot:latest .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f pizzabot

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	black src/ tests/

lint:
	flake8 src/ tests/ --max-line-length=100

init-db:
	python -c "from src.data import DataLoader; from src.embeddings import EmbeddingManager; \
	           from src.utils import load_config; \
	           config = load_config(); \
	           loader = DataLoader(); \
	           em = EmbeddingManager(config['models']['embedding_model'], config['vector_db']['persist_directory'], config['vector_db']['collection_name']); \
	           em.clear_collection(); \
	           em.add_documents(loader.get_all_documents()); \
	           print('Database initialized')"
