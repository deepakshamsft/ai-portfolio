.PHONY: help install test lint docker-build docker-run k8s-deploy clean

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install Python dependencies
	pip install -r requirements.txt

test:  ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

lint:  ## Run code linters
	black --check src/ tests/
	flake8 src/ tests/ --max-line-length=120

format:  ## Format code with black
	black src/ tests/

docker-build:  ## Build Docker image
	docker build -t ml-model-serving:latest .

docker-run:  ## Run Docker container locally
	docker run -d -p 8000:8000 --name ml-serving ml-model-serving:latest

docker-stop:  ## Stop Docker container
	docker stop ml-serving || true
	docker rm ml-serving || true

docker-compose-up:  ## Start all services with docker-compose
	docker-compose up -d

docker-compose-down:  ## Stop all services
	docker-compose down

docker-compose-logs:  ## View docker-compose logs
	docker-compose logs -f

k8s-deploy:  ## Deploy to Kubernetes
	kubectl apply -f k8s/

k8s-delete:  ## Delete Kubernetes deployment
	kubectl delete -f k8s/

mlflow-server:  ## Start MLflow server locally
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

airflow-init:  ## Initialize Airflow database
	airflow db init
	airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

load-test:  ## Run load tests
	locust -f src/load_test.py --host=http://localhost:8000 --users=100 --spawn-rate=10 --run-time=2m --headless

terraform-init:  ## Initialize Terraform
	cd terraform && terraform init

terraform-plan:  ## Plan Terraform changes
	cd terraform && terraform plan

terraform-apply:  ## Apply Terraform changes
	cd terraform && terraform apply

terraform-destroy:  ## Destroy Terraform infrastructure
	cd terraform && terraform destroy

clean:  ## Clean up generated files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf htmlcov .coverage .pytest_cache
	rm -rf data/temp models/*.pkl

setup-local:  ## Setup local development environment
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	mkdir -p data models mlruns

health-check:  ## Check if services are healthy
	@echo "Checking model serving..."
	@curl -f http://localhost:8000/health || echo "Model serving not available"
	@echo "\nChecking MLflow..."
	@curl -f http://localhost:5000/health || echo "MLflow not available"
	@echo "\nChecking Airflow..."
	@curl -f http://localhost:8080/health || echo "Airflow not available"
