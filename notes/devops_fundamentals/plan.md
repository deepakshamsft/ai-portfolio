# DevOps Fundamentals — Implementation Plan

> **Status:** Planning phase. Extracted from AI Infrastructure track.  
> **Last updated:** April 26, 2026  
> **Philosophy:** Generic DevOps skills applicable to any software deployment (web apps, microservices, ML models, etc.)

---

## Overview

This track covers **general-purpose infrastructure and deployment practices** that every engineer should know, independent of AI/ML. All topics use **100% free local tools** with optional cloud integration.

**Target audience:** Engineers transitioning from development to production, backend engineers learning DevOps, data scientists deploying applications.

---

## Planned Chapters

| Chapter | Status | Description |
|---------|--------|-------------|
| Ch.1: Docker Fundamentals | ⏳ Planned | Containers, images, Dockerfiles, volumes, networks |
| Ch.2: Container Orchestration | ⏳ Planned | Docker Compose, multi-container apps, service dependencies |
| Ch.3: Kubernetes Basics | ⏳ Planned | Pods, services, deployments (using Kind locally) |
| Ch.4: CI/CD Pipelines | ⏳ Planned | GitHub Actions, automated testing, deployment workflows |
| Ch.5: Monitoring & Observability | ⏳ Planned | Prometheus, Grafana, logging, metrics, alerting |
| Ch.6: Infrastructure as Code | ⏳ Planned | Terraform basics, reproducible deployments |
| Ch.7: Networking & Load Balancing | ⏳ Planned | Reverse proxies, load balancers, service meshes |
| Ch.8: Security & Secrets Management | ⏳ Planned | Environment variables, secrets rotation, RBAC |

---

## Ch.1 — Docker Fundamentals

### Overview

**Running example:** Containerize a Python Flask web app  
**Constraint:** Must run identically on dev laptop and production server  
**Tech stack (FREE):**
- Docker Desktop (free for personal use)
- Docker Hub (free tier: unlimited public images)

### Content Structure

#### README.md Sections
1. **The Challenge:** "Works on my machine" deployment failures
2. **Core Idea:** Container = lightweight, reproducible runtime environment
3. **Running Example:** Flask app with Redis cache
   - Step 1: Write Dockerfile
   - Step 2: Build image
   - Step 3: Run container
   - Step 4: Debug common issues (port conflicts, volume mounts)
4. **Mental Model:** Image vs. container (blueprint vs. running instance)
5. **Code Skeleton:** Dockerfile best practices, .dockerignore
6. **What Can Go Wrong:** Layer caching, image size bloat, security vulnerabilities
7. **Progress Check:** 3 questions on image layers, port mapping, volume persistence
8. **Bridge to Ch.2:** Multi-container apps need orchestration

#### notebook.ipynb (Local)
- Cell 1: Install Docker, verify installation
- Cell 2: Pull an image (e.g., `python:3.11-slim`)
- Cell 3: Run a container interactively
- Cell 4: Write a simple Dockerfile
- Cell 5: Build custom image
- Cell 6: Run custom container with port mapping
- Cell 7: Mount volumes for persistent data
- Cell 8: Inspect logs, exec into running container
- Cell 9: Multi-stage builds for smaller images
- Cell 10: Push image to Docker Hub

#### notebook_supplement.ipynb (Azure)
- Cell 1: Azure credentials
  ```python
  AZURE_SUBSCRIPTION_ID = ""  # Stripped by pre-push hook
  AZURE_CONTAINER_REGISTRY_NAME = ""
  ```
- Cell 2: Azure Container Registry (ACR) setup
- Cell 3: Build and push image to ACR
- Cell 4: Deploy to Azure Container Instances (ACI)
- Cell 5: Monitor with Azure Container Insights

#### Diagrams
- `gen_ch01_docker_architecture.py` → Image layers diagram
- `gen_ch01_container_lifecycle.py` → Build → Run → Stop → Remove flow
- `gen_ch01_volume_mounts.py` → Host filesystem vs. container filesystem

---

## Ch.2 — Container Orchestration (Docker Compose)

### Overview

**Running example:** Web app + database + cache (3-tier architecture)  
**Constraint:** All services must start with one command  
**Tech stack (FREE):** Docker Compose (included with Docker Desktop)

### Content Structure

#### README.md Sections
1. **The Challenge:** Managing multiple interdependent containers
2. **Core Idea:** Declarative service definitions with dependency ordering
3. **Running Example:** Flask + PostgreSQL + Redis
   - Step 1: Write `docker-compose.yml`
   - Step 2: Define service dependencies
   - Step 3: Configure networks and volumes
   - Step 4: Start entire stack with `docker compose up`
4. **Mental Model:** Services, networks, volumes as YAML primitives
5. **Code Skeleton:** docker-compose.yml template, environment variables
6. **What Can Go Wrong:** Service startup ordering, network isolation, port conflicts
7. **Progress Check:** Debug a broken docker-compose.yml
8. **Bridge to Ch.3:** Compose works on one machine; Kubernetes scales across many

#### notebook.ipynb (Local)
- Cell 1: Install Docker Compose
- Cell 2: Simple 2-service example (web + Redis)
- Cell 3: Add database service (PostgreSQL)
- Cell 4: Configure environment variables
- Cell 5: Persistent volumes for database
- Cell 6: Health checks and restart policies
- Cell 7: Network isolation between services
- Cell 8: Scale services (multiple replicas)
- Cell 9: View logs from all services
- Cell 10: Production-ready compose file

#### notebook_supplement.ipynb (Azure)
- Cell 1: Azure credentials
- Cell 2: Deploy multi-container group to ACI
- Cell 3: Azure Container Apps (serverless alternative)
- Cell 4: Monitor multi-container deployments

#### Diagrams
- `gen_ch02_compose_architecture.py` → Service dependency graph
- `gen_ch02_network_isolation.py` → Internal vs. external networks
- `gen_ch02_volume_persistence.py` → Named volumes across restarts

---

## Ch.3 — Kubernetes Basics (Kind)

### Overview

**Running example:** Deploy Flask app to local Kubernetes cluster  
**Constraint:** Learn K8s without cloud spend  
**Tech stack (FREE):**
- Kind (Kubernetes in Docker)
- kubectl CLI
- K9s (optional TUI for cluster management)

### Content Structure

#### README.md Sections
1. **The Challenge:** Docker Compose doesn't scale across machines or handle failures
2. **Core Idea:** Kubernetes = declarative orchestration with self-healing
3. **Running Example:** Flask deployment with 3 replicas
   - Step 1: Create Kind cluster
   - Step 2: Write deployment YAML
   - Step 3: Create service (load balancer)
   - Step 4: Simulate pod failure (watch auto-recovery)
4. **Mental Model:** Pods, ReplicaSets, Deployments, Services
5. **Code Skeleton:** Deployment + Service YAML templates
6. **What Can Go Wrong:** ImagePullBackOff, CrashLoopBackOff, networking issues
7. **Progress Check:** Debug a pod stuck in Pending state
8. **Bridge to Ch.4:** CI/CD automates deployments to K8s

#### notebook.ipynb (Local)
- Cell 1: Install Kind and kubectl
- Cell 2: Create local Kubernetes cluster
- Cell 3: Deploy a pod (single container)
- Cell 4: Create a deployment (multiple replicas)
- Cell 5: Expose deployment as a service
- Cell 6: Access service from host machine
- Cell 7: Rolling updates (change image version)
- Cell 8: Rollback a deployment
- Cell 9: ConfigMaps and Secrets
- Cell 10: Debugging with `kubectl logs`, `kubectl describe`

#### notebook_supplement.ipynb (Azure)
- Cell 1: Azure credentials
- Cell 2: Create Azure Kubernetes Service (AKS) cluster
- Cell 3: Connect kubectl to AKS
- Cell 4: Deploy application to AKS
- Cell 5: Scale cluster nodes (horizontal scaling)
- Cell 6: Monitor with Azure Monitor for Containers

#### Diagrams
- `gen_ch03_k8s_architecture.py` → Control plane + worker nodes
- `gen_ch03_pod_lifecycle.py` → Pending → Running → Succeeded/Failed
- `gen_ch03_service_discovery.py` → ClusterIP vs. NodePort vs. LoadBalancer

---

## Ch.4 — CI/CD Pipelines (GitHub Actions)

### Overview

**Running example:** Auto-deploy Flask app on every push to main  
**Constraint:** Use only GitHub's free tier (2,000 minutes/month)  
**Tech stack (FREE):** GitHub Actions, Docker Hub

### Content Structure

#### README.md Sections
1. **The Challenge:** Manual deployments are slow and error-prone
2. **Core Idea:** Automate test → build → deploy pipeline
3. **Running Example:** Flask app CI/CD
   - Step 1: Write unit tests (pytest)
   - Step 2: Create GitHub Actions workflow
   - Step 3: Auto-build Docker image on push
   - Step 4: Push image to Docker Hub
   - Step 5: Deploy to production (optional: Kind cluster)
4. **Mental Model:** Trigger → Jobs → Steps → Actions
5. **Code Skeleton:** .github/workflows/ci-cd.yml template
6. **What Can Go Wrong:** Secrets management, workflow debugging, runner timeouts
7. **Progress Check:** Fix a broken workflow YAML
8. **Bridge to Ch.5:** Monitoring catches issues after deployment

#### notebook.ipynb (Local)
- Cell 1: Set up GitHub Actions in a repo
- Cell 2: Write a simple workflow (run tests)
- Cell 3: Add Docker build step
- Cell 4: Push image to Docker Hub (using secrets)
- Cell 5: Deploy to local Kind cluster from CI
- Cell 6: Matrix builds (test on multiple Python versions)
- Cell 7: Caching dependencies (speed up builds)
- Cell 8: Conditional workflows (deploy only on main branch)
- Cell 9: Manual workflow triggers (workflow_dispatch)
- Cell 10: View logs and artifacts

#### notebook_supplement.ipynb (Azure)
- Cell 1: Azure credentials
- Cell 2: Azure DevOps Pipelines (alternative to GitHub Actions)
- Cell 3: Build and push to ACR
- Cell 4: Deploy to ACI or AKS
- Cell 5: Azure Pipelines YAML syntax

#### Diagrams
- `gen_ch04_cicd_pipeline.py` → Full workflow: commit → test → build → deploy
- `gen_ch04_workflow_triggers.py` → Push, PR, schedule, manual
- `gen_ch04_secrets_management.py` → GitHub Secrets → Actions → Docker Hub

---

## Ch.5 — Monitoring & Observability (Prometheus + Grafana)

### Overview

**Running example:** Monitor Flask app metrics (requests/sec, latency, errors)  
**Constraint:** Run entire monitoring stack locally  
**Tech stack (FREE):**
- Prometheus (metrics collection)
- Grafana (visualization)
- Docker Compose (stack deployment)

### Content Structure

#### README.md Sections
1. **The Challenge:** Can't fix what you can't see (blind deployments)
2. **Core Idea:** Metrics + logs + traces = observability
3. **Running Example:** Flask app with custom metrics
   - Step 1: Instrument app with prometheus_client
   - Step 2: Deploy Prometheus (scrape metrics)
   - Step 3: Deploy Grafana (visualize metrics)
   - Step 4: Create dashboard (requests/sec, latency, error rate)
4. **Mental Model:** Metrics vs. logs vs. traces
5. **Code Skeleton:** Prometheus config, Grafana dashboard JSON
6. **What Can Go Wrong:** Cardinality explosion, query performance, retention limits
7. **Progress Check:** Build a dashboard from scratch
8. **Bridge to Ch.6:** IaC automates infrastructure setup

#### notebook.ipynb (Local)
- Cell 1: Install prometheus_client library
- Cell 2: Instrument Flask app (counter, histogram, gauge)
- Cell 3: Deploy Prometheus with docker-compose.yml
- Cell 4: Verify metrics scraping (Prometheus UI)
- Cell 5: Deploy Grafana
- Cell 6: Connect Grafana to Prometheus data source
- Cell 7: Create a dashboard (request rate graph)
- Cell 8: Add alerting rules (Prometheus Alertmanager)
- Cell 9: Simulate traffic and observe metrics
- Cell 10: Export dashboard as JSON

#### notebook_supplement.ipynb (Azure)
- Cell 1: Azure credentials
- Cell 2: Azure Application Insights (APM alternative)
- Cell 3: Instrument Flask app with OpenTelemetry
- Cell 4: Send metrics to Application Insights
- Cell 5: Create Azure Monitor workbook (dashboard)
- Cell 6: Set up alerts (email/SMS on high error rate)

#### Diagrams
- `gen_ch05_prometheus_architecture.py` → Scrape targets → TSDB → Query
- `gen_ch05_metrics_types.py` → Counter, Gauge, Histogram, Summary
- `gen_ch05_grafana_dashboard.py` → Time series visualization

---

## Ch.6 — Infrastructure as Code (Terraform Basics)

### Overview

**Running example:** Provision Docker containers with Terraform  
**Constraint:** Infrastructure must be version-controlled and reproducible  
**Tech stack (FREE):** Terraform, Docker provider

### Content Structure

#### README.md Sections
1. **The Challenge:** Manual infrastructure changes drift over time
2. **Core Idea:** Infrastructure = code (version control, review, rollback)
3. **Running Example:** Terraform + Docker
   - Step 1: Install Terraform
   - Step 2: Write .tf files (Docker containers)
   - Step 3: `terraform plan` → see changes
   - Step 4: `terraform apply` → provision infrastructure
   - Step 5: `terraform destroy` → clean up
4. **Mental Model:** Resources, providers, state, plan vs. apply
5. **Code Skeleton:** main.tf, variables.tf, outputs.tf template
6. **What Can Go Wrong:** State file corruption, provider version conflicts, drift detection
7. **Progress Check:** Modify a .tf file and predict the plan output
8. **Bridge to Ch.7:** Networking makes services discoverable

#### notebook.ipynb (Local)
- Cell 1: Install Terraform
- Cell 2: Write Terraform config for Docker container
- Cell 3: Initialize Terraform (`terraform init`)
- Cell 4: Plan changes (`terraform plan`)
- Cell 5: Apply changes (`terraform apply`)
- Cell 6: Inspect state file
- Cell 7: Update container image (rolling update)
- Cell 8: Destroy infrastructure (`terraform destroy`)
- Cell 9: Use variables and outputs
- Cell 10: Terraform modules (reusable components)

#### notebook_supplement.ipynb (Azure)
- Cell 1: Azure credentials
- Cell 2: Terraform Azure provider setup
- Cell 3: Provision Azure Container Instance with Terraform
- Cell 4: Provision Azure Storage Account
- Cell 5: Manage Azure resource groups
- Cell 6: Remote state backend (Azure Storage)

#### Diagrams
- `gen_ch06_terraform_workflow.py` → Write → Plan → Apply → State
- `gen_ch06_resource_graph.py` → Dependency graph visualization
- `gen_ch06_state_management.py` → Local vs. remote state

---

## Ch.7 — Networking & Load Balancing

### Overview

**Running example:** Nginx reverse proxy + 3 Flask replicas  
**Constraint:** Load balance requests evenly, handle failures gracefully  
**Tech stack (FREE):** Nginx, Docker Compose

### Content Structure

#### README.md Sections
1. **The Challenge:** Single server = single point of failure
2. **Core Idea:** Reverse proxy distributes load, provides failover
3. **Running Example:** Nginx + 3 Flask backends
   - Step 1: Deploy 3 Flask containers
   - Step 2: Configure Nginx as reverse proxy
   - Step 3: Round-robin load balancing
   - Step 4: Health checks (remove unhealthy backend)
4. **Mental Model:** Client → Proxy → Backend pool
5. **Code Skeleton:** nginx.conf template, upstream blocks
6. **What Can Go Wrong:** Session affinity, SSL termination, backend overload
7. **Progress Check:** Debug a backend that's not receiving traffic
8. **Bridge to Ch.8:** Secrets protect API keys and credentials

#### notebook.ipynb (Local)
- Cell 1: Deploy multiple Flask replicas
- Cell 2: Write nginx.conf (reverse proxy)
- Cell 3: Deploy Nginx container
- Cell 4: Send requests, observe load distribution
- Cell 5: Simulate backend failure (stop one container)
- Cell 6: Health checks (passive vs. active)
- Cell 7: Sticky sessions (session affinity)
- Cell 8: SSL termination at proxy
- Cell 9: Rate limiting (protect backends)
- Cell 10: Logging and access logs analysis

#### notebook_supplement.ipynb (Azure)
- Cell 1: Azure credentials
- Cell 2: Azure Application Gateway (L7 load balancer)
- Cell 3: Backend pool configuration
- Cell 4: Health probes
- Cell 5: SSL offloading
- Cell 6: Azure Load Balancer (L4 alternative)

#### Diagrams
- `gen_ch07_reverse_proxy.py` → Client → Nginx → Backend pool
- `gen_ch07_load_balancing_algorithms.py` → Round-robin, least-conn, IP hash
- `gen_ch07_health_checks.py` → Active probing vs. passive failure detection

---

## Ch.8 — Security & Secrets Management

### Overview

**Running example:** Secure API keys in containerized app  
**Constraint:** No secrets in Dockerfiles, images, or git  
**Tech stack (FREE):** Docker secrets, environment variables, .env files

### Content Structure

#### README.md Sections
1. **The Challenge:** Hardcoded secrets = security breach
2. **Core Idea:** Secrets are runtime configuration, not build-time
3. **Running Example:** Flask app with database password
   - Step 1: Environment variables (.env file)
   - Step 2: Docker secrets (Compose integration)
   - Step 3: Kubernetes secrets (Base64 encoding)
   - Step 4: Pre-push hook (strip secrets from notebooks)
4. **Mental Model:** Build-time vs. runtime vs. secret stores
5. **Code Skeleton:** .env.example, docker-compose secrets, K8s secret YAML
6. **What Can Go Wrong:** Secrets in logs, base64 != encryption, secret rotation
7. **Progress Check:** Audit a Dockerfile for security issues
8. **Bridge to Future:** Apply DevOps to AI/ML deployments (AI Infrastructure track)

#### notebook.ipynb (Local)
- Cell 1: Bad practice: Hardcoded API key in Dockerfile
- Cell 2: Good practice: Environment variables
- Cell 3: .env file + Docker Compose
- Cell 4: Docker secrets (Swarm mode)
- Cell 5: Kubernetes secrets (Base64)
- Cell 6: Scanning images for vulnerabilities (Trivy)
- Cell 7: Pre-push hook for secret detection
- Cell 8: Secrets rotation strategy
- Cell 9: RBAC (least privilege principle)
- Cell 10: Audit logs (who accessed what)

#### notebook_supplement.ipynb (Azure)
- Cell 1: Azure credentials
- Cell 2: Azure Key Vault setup
- Cell 3: Store secrets in Key Vault
- Cell 4: Access Key Vault from container (Managed Identity)
- Cell 5: Secret rotation with Key Vault
- Cell 6: Azure Policy (enforce secret scanning)

#### Diagrams
- `gen_ch08_secrets_lifecycle.py` → Create → Store → Access → Rotate → Revoke
- `gen_ch08_attack_surface.py` → Git, Docker images, logs, environment variables
- `gen_ch08_defense_layers.py` → Pre-commit hooks, image scanning, runtime policies

---

## Prerequisites (Root Setup Scripts)

### Dependencies Installation

**File:** `scripts/setup.ps1` (Windows)

```powershell
# DevOps Fundamentals dependencies
Write-Host "📦 Installing DevOps Fundamentals dependencies..." -ForegroundColor Cyan

# Docker Desktop (manual)
Write-Host "⚠️  Manual setup required:" -ForegroundColor Yellow
Write-Host "   1. Install Docker Desktop: https://www.docker.com/products/docker-desktop"
Write-Host "   2. Verify: docker --version"

# Kind (Kubernetes in Docker)
Write-Host "Installing Kind..." -ForegroundColor Cyan
choco install kind -y

# Kubectl
Write-Host "Installing kubectl..." -ForegroundColor Cyan
choco install kubernetes-cli -y

# Terraform
Write-Host "Installing Terraform..." -ForegroundColor Cyan
choco install terraform -y

# Optional: K9s (Kubernetes TUI)
Write-Host "Installing K9s (optional)..." -ForegroundColor Cyan
choco install k9s -y

Write-Host "✅ DevOps Fundamentals setup complete" -ForegroundColor Green
Write-Host "   Verify: docker --version && kind --version && kubectl version --client" -ForegroundColor Cyan
```

**File:** `scripts/setup.sh` (Unix)

```bash
# DevOps Fundamentals dependencies
echo "📦 Installing DevOps Fundamentals dependencies..."

# Docker (Ubuntu/Debian)
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
fi

# Kind
if ! command -v kind &> /dev/null; then
    echo "Installing Kind..."
    curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
fi

# Kubectl
if ! command -v kubectl &> /dev/null; then
    echo "Installing kubectl..."
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/kubectl
fi

# Terraform
if ! command -v terraform &> /dev/null; then
    echo "Installing Terraform..."
    wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    sudo apt update && sudo apt install terraform -y
fi

# Optional: K9s
if ! command -v k9s &> /dev/null; then
    echo "Installing K9s (optional)..."
    curl -sS https://webinstall.dev/k9s | bash
fi

echo "✅ DevOps Fundamentals setup complete"
echo "   Verify: docker --version && kind --version && kubectl version --client"
```

---

## Tasks

### Root Setup Script Updates

- [ ] **Update `scripts/setup.ps1`**
  - Add DevOps Fundamentals dependencies section
  - Install Kind, kubectl, Terraform via Chocolatey
  - Add Docker Desktop manual install instructions

- [ ] **Update `scripts/setup.sh`**
  - Add DevOps Fundamentals dependencies section
  - Auto-install Docker, Kind, kubectl, Terraform on Linux/macOS
  - Test on Ubuntu 22.04 and macOS

---

## Timeline

### Phase 1: Containers (Weeks 1-2)
- [ ] Ch.1: Docker Fundamentals (README + notebook + supplement + diagrams)
- [ ] Ch.2: Docker Compose (README + notebook + supplement + diagrams)

### Phase 2: Orchestration (Weeks 3-4)
- [ ] Ch.3: Kubernetes Basics (README + notebook + supplement + diagrams)
- [ ] Ch.4: CI/CD Pipelines (README + notebook + supplement + diagrams)

### Phase 3: Observability & Security (Weeks 5-6)
- [ ] Ch.5: Monitoring (README + notebook + supplement + diagrams)
- [ ] Ch.6: Infrastructure as Code (README + notebook + supplement + diagrams)

### Phase 4: Advanced Topics (Weeks 7-8)
- [ ] Ch.7: Networking (README + notebook + supplement + diagrams)
- [ ] Ch.8: Security (README + notebook + supplement + diagrams)

---

## Success Criteria

- ✅ All chapters run 100% locally with free tools
- ✅ Cloud supplements optional (Azure, AWS, GCP examples)
- ✅ No paid subscriptions required for core learning
- ✅ Graduates can deploy any application (web, ML, microservices) to production
- ✅ DevOps skills transferable across all domains (not AI-specific)

---

## Bridge to Other Tracks

- **→ AI Infrastructure:** Apply DevOps to ML model deployments (model serving, A/B testing, feature stores)
- **→ AI Primer:** Deploy chatbot applications with Docker + K8s
- **→ Multi-Agent AI:** Orchestrate multi-agent systems with container networking
- **→ ML Track:** CI/CD for ML pipelines (model training → evaluation → deployment)

---

## Notes

- **Generic infrastructure focus:** Docker, K8s, monitoring work for ANY application (not just AI/ML)
- **Local-first learning:** Every concept demonstrated on localhost before showing cloud
- **Cross-platform:** All examples work on Windows, Linux, macOS
- **Free tier only:** Docker Desktop (personal use), GitHub Actions (2,000 min/month), no cloud spend required
