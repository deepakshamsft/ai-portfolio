# Ch.2 — Container Orchestration (Docker Compose)

> **Story:** Docker Compose emerged in 2014 (originally as Fig, acquired by Docker Inc.) to solve a universal frustration: managing multi-container applications with bash scripts. Before Compose, spinning up a web app + database + cache meant running 3+ `docker run` commands, remembering ports, networks, and environment variables. Sam Alba and the Docker team formalized the YAML-based declarative approach — define your entire stack once, start it with one command. Today, every engineer deploying microservices or multi-tier applications uses Compose daily — whether for local development or as a stepping stone to Kubernetes.
> 
> **Where you are:** Ch.1 showed you how to containerize a single application with Docker — but production systems rarely run in isolation. You need a web server that talks to a database that connects to a cache layer. Managing these dependencies manually breaks down fast — port conflicts, startup ordering, network isolation. This chapter teaches you to orchestrate multiple containers as a single coordinated system, unlocking the ability to deploy real 3-tier architectures locally before scaling to cloud orchestrators like Kubernetes.
> 
> **Notation:** `service` — a container definition in docker-compose.yml; `network` — isolated communication layer between services; `volume` — persistent storage shared across container restarts; `depends_on` — startup dependency declaration; `healthcheck` — automated readiness probe.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Deploy a production-ready 3-tier web application (frontend → backend → database) that survives restarts, scales horizontally, and starts with one command.

**What we know so far:**
- ✅ Ch.1: We can containerize a single Python Flask app with Docker
- ❌ **But we can't coordinate multiple services** — manually running `docker run` for web + Postgres + Redis = 3 terminal windows, hardcoded IPs, brittle startup scripts

**What's blocking us:**

You've containerized a Flask API. Great. Now you need PostgreSQL for persistent data and Redis for session caching. The manual workflow:

```bash
docker run -d --name postgres -e POSTGRES_PASSWORD=secret postgres:16
docker run -d --name redis redis:7-alpine
docker run -d --name web --link postgres --link redis -p 5000:5000 my-flask-app
```

This breaks in three ways:
1. **Startup ordering**: The web container crashes if it starts before Postgres is ready to accept connections (race condition).
2. **Configuration drift**: Environment variables, ports, and volume mounts are scattered across 3 commands — one typo kills the stack.
3. **No persistence**: Stopping the Postgres container wipes your database unless you remember to manually create a named volume.

**What this chapter unlocks:**

Docker Compose — declarative multi-container orchestration. Define all services, networks, and volumes in one `docker-compose.yml` file. Start the entire stack with `docker compose up`, tear it down with `docker compose down`. Guaranteed startup ordering, automatic network isolation, persistent volumes. Production-grade 3-tier apps run locally with the same reliability as Kubernetes — but with zero cluster overhead.

---

## 1 · Core Idea — Declarative Service Orchestration

**The insight**: Don't script container startup — declare the *desired state* of your system.

A Docker Compose file is a blueprint:
- **Services** — each service is one container (or a scaled group of identical containers).
- **Networks** — isolated communication layers. Services on the same network can talk by name (DNS built-in).
- **Volumes** — persistent storage that survives `docker compose down`.

The Compose engine reads the YAML, creates networks and volumes, starts containers in dependency order, and monitors health. You get:
- **Idempotency**: Running `docker compose up` multiple times produces the same result.
- **Atomic teardown**: `docker compose down` removes all containers and networks (but preserves volumes unless you pass `--volumes`).
- **Environment parity**: The same `docker-compose.yml` runs on your laptop, CI server, and staging environment.

**Mental model**: Think of Compose as a *state manager*. You declare "I want 1 web container, 1 Postgres container, and 1 Redis container connected via a shared network," and Compose materializes that state. Stopping services returns the system to a clean slate (minus persistent volumes).

---

## Animation

![Multi-container orchestration flow](img/ch02-compose-orchestration.gif)

> **What you're seeing:** Docker Compose startup sequence — Compose reads YAML → creates network → starts `cache` (no dependencies) → starts `db` (waits for health check) → starts `web` (waits for both). The animation demonstrates health-check-based dependency ordering preventing race conditions. This is the mental model for declarative orchestration: **declare desired state, Compose materializes it in correct order**.

---

## 2 · Running Example — Flask + PostgreSQL + Redis

**The system**: A Flask REST API (`/users` endpoint) that:
- Stores user records in **PostgreSQL** (persistent data)
- Caches frequently accessed queries in **Redis** (fast in-memory lookups)
- Exposes port 5000 externally

This is a canonical 3-tier architecture:
```
Client → Flask (web tier) → Redis (cache tier) + PostgreSQL (data tier)
```

### Step 1: Write `docker-compose.yml`

Create a new directory `flask-stack/` with:

```yaml
# docker-compose.yml
version: '3.9'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/appdb
      - REDIS_URL=redis://cache:6379
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    networks:
      - app-network

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: appdb
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      timeout: 3s
      retries: 5
    networks:
      - app-network

  cache:
    image: redis:7-alpine
    networks:
      - app-network

networks:
  app-network:
    driver: bridge

volumes:
  postgres-data:
```

**Key details**:
- `depends_on.condition: service_healthy` ensures `web` waits for Postgres readiness (not just "container started").
- `networks: app-network` isolates services — external clients can't directly talk to `db` or `cache`.
- `volumes: postgres-data` persists database files across `docker compose down` cycles.

### Step 2: Define Service Dependencies

The `depends_on` block controls startup order:

```yaml
web:
  depends_on:
    db:
      condition: service_healthy  # Wait for health check to pass
    cache:
      condition: service_started  # Just wait for container to start
```

Without this, Flask would crash with "connection refused" errors on startup.

### Step 3: Configure Networks and Volumes

**Networks**: Services reference each other by name (`db`, `cache`) — Compose's internal DNS resolves these to container IPs automatically. The `app-network` is a private bridge network; only containers on the same network can communicate.

**Volumes**: The `postgres-data` named volume stores `/var/lib/postgresql/data` (Postgres's data directory). When you run `docker compose down`, the database persists. To wipe it: `docker compose down --volumes`.

### Step 4: Start the Entire Stack

```bash
docker compose up -d
```

**What happens**:
1. Compose creates `app-network` and `postgres-data` volume (if they don't exist).
2. Starts `cache` (Redis) — no dependencies.
3. Starts `db` (Postgres) — waits for health check to pass.
4. Starts `web` (Flask) — only after `db` is healthy and `cache` is running.

**Check logs**:
```bash
docker compose logs -f web
```

**Test the API**:
```bash
curl http://localhost:5000/users
```

**Tear down**:
```bash
docker compose down        # Stops containers, removes networks (keeps volumes)
docker compose down --volumes  # Also deletes postgres-data volume
```

---

## 3 · Mental Model — Services, Networks, Volumes as YAML Primitives

Think of Docker Compose as a **configuration-driven runtime** with three core primitives:

### Primitive 1: Services

A service is a container blueprint. You define:
- `image` or `build` (where to get the container)
- `ports` (expose to host)
- `environment` (config variables)
- `depends_on` (startup dependencies)
- `restart` policy (`no`, `always`, `on-failure`, `unless-stopped`)

**Scaling**: Run multiple replicas of a service:
```bash
docker compose up -d --scale web=3
```
This creates 3 `web` containers. If you exposed port 5000, you'd need a load balancer in front (see Ch.7).

### Primitive 2: Networks

By default, Compose creates a single network for all services. You can define multiple networks to isolate tiers:

```yaml
networks:
  frontend:  # Web tier only
  backend:   # Database tier only

services:
  web:
    networks:
      - frontend
      - backend
  db:
    networks:
      - backend  # Not visible to external clients
```

Services communicate via DNS: `http://web:5000`, `postgresql://db:5432`.

### Primitive 3: Volumes

Three types:
1. **Named volumes**: Managed by Docker, persistent across restarts (`postgres-data:/var/lib/postgresql/data`).
2. **Bind mounts**: Link host directories into containers (`./app:/app` — common for development live-reload).
3. **Anonymous volumes**: Temporary storage, deleted when container stops.

**Rule**: Use named volumes for production data (databases, uploaded files). Use bind mounts for development (live code changes without rebuilding images).

---

## 4 · What Can Go Wrong — Service Startup Ordering, Network Isolation, Port Conflicts

### Failure 1: Race Condition on Database Readiness

**The scenario**: Your Flask app connects to Postgres in `__init__`:
```python
db = SQLAlchemy(app)  # Crashes if Postgres isn't ready
```

**Symptom**: `web` container starts, crashes with `connection refused`, restarts, crashes again (crash loop).

**Why**: `depends_on: db` only waits for the Postgres *container* to start — not for the database server *inside* the container to accept connections. Postgres takes 3-5 seconds to initialize.

**Fix 1: Health check condition**
```yaml
web:
  depends_on:
    db:
      condition: service_healthy  # Wait for health check to pass
```

**Fix 2: Retry logic in app code**
```python
import time
from sqlalchemy import create_engine

def connect_with_retry(db_url, retries=5):
    for i in range(retries):
        try:
            engine = create_engine(db_url)
            engine.connect()
            return engine
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"DB not ready, retrying in 2s... ({e})")
            time.sleep(2)
```

### Failure 2: Port Conflict

**The scenario**: You already have a PostgreSQL instance running on your laptop (port 5432). When you run `docker compose up`, you get:
```
Error: bind: address already in use
```

**Why**: Two processes can't bind to the same port. Your host Postgres and the containerized Postgres both want 5432.

**Fix 1: Change the host port mapping**
```yaml
db:
  ports:
    - "5433:5432"  # Expose on host port 5433 instead
```
Now connect via `postgresql://user:pass@localhost:5433/appdb`.

**Fix 2: Don't expose the port at all**
```yaml
db:
  # Remove ports section entirely
```
Services can still communicate via the internal `app-network`. Only expose ports that need external access (e.g., the `web` service).

### Failure 3: Network Isolation Breaks External Access

**The scenario**: You define a custom network for `web` and `db`, but forget to expose `web`'s port:
```yaml
services:
  web:
    networks:
      - backend  # No external access
```

**Symptom**: `curl http://localhost:5000` times out — the container is running but unreachable.

**Why**: Services on custom networks are isolated from the host by default. You must explicitly publish ports:
```yaml
web:
  ports:
    - "5000:5000"  # Bind container port 5000 to host port 5000
  networks:
    - backend
```

Alternatively, use `network_mode: host` (Linux only) to share the host's network stack — but this disables network isolation.

### Failure 4: Volumes Deleted by Accident

**The scenario**: You run `docker compose down` to restart your services, and all your database data is gone.

**Why**: Someone ran `docker compose down --volumes` (or `-v`), which deletes named volumes. The default `docker compose down` preserves volumes.

**Prevention**:
1. Always use named volumes for production data:
   ```yaml
   volumes:
     postgres-data:/var/lib/postgresql/data
   ```
2. Back up volumes before destructive operations:
   ```bash
   docker run --rm -v postgres-data:/data -v $(pwd):/backup \
     busybox tar czf /backup/postgres-backup.tar.gz /data
   ```

---

## 5 · Progress Check — Debug a Broken Compose File

**Scenario**: Your team gives you this `docker-compose.yml`:

```yaml
version: '3.9'

services:
  api:
    image: node:18
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://user:pass@db:5432/appdb
    command: npm start

  db:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: pass
```

You run `docker compose up` and see:
```
api_1 | Error: connect ECONNREFUSED db:5432
api_1 exited with code 1
```

**Question 1**: What's wrong with the service dependencies?

<details>
<summary>Click to reveal</summary>

**Answer**: The `api` service has no `depends_on` clause. It starts immediately and tries to connect to Postgres before the `db` container is ready. The fix:

```yaml
api:
  depends_on:
    db:
      condition: service_healthy
```

Add a health check to `db`:
```yaml
db:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U user"]
    interval: 5s
    timeout: 3s
    retries: 5
```
</details>

**Question 2**: The database data disappears every time you run `docker compose down`. Why?

<details>
<summary>Click to reveal</summary>

**Answer**: No named volume is defined for Postgres. By default, data is stored in an anonymous volume that gets deleted when the container is removed. Fix:

```yaml
db:
  volumes:
    - db-data:/var/lib/postgresql/data

volumes:
  db-data:
```
</details>

**Question 3**: You want to scale the `api` service to 3 replicas. What command do you run?

<details>
<summary>Click to reveal</summary>

**Answer**:
```bash
docker compose up -d --scale api=3
```

**Caveat**: If you exposed port 3000, all 3 containers will conflict trying to bind to the same port. Solutions:
- Remove the `ports` section and use a reverse proxy (e.g., Nginx, Traefik).
- Use a port range: `"3000-3002:3000"` (limited scalability).
</details>

---

## 6 · Bridge to Ch.3 — Compose Works on One Machine; Kubernetes Scales Across Many

**What you've learned**: Docker Compose orchestrates multi-container applications with declarative service definitions, health checks, and persistent volumes. You can run a production-grade 3-tier stack (web + cache + database) locally with one command.

**What Compose can't do**:
- **Horizontal scaling across machines**: `docker compose up --scale web=10` runs 10 containers *on your laptop*. Production systems need to distribute load across multiple nodes.
- **Self-healing**: If a container crashes, Compose restarts it (with `restart: always`) — but if the *host machine* fails, your entire stack goes down.
- **Rolling updates**: Deploying a new version requires `docker compose down && docker compose up` — causing downtime.
- **Load balancing**: Scaling a service creates multiple containers, but Compose doesn't automatically route traffic between them.

**Enter Kubernetes**: The next chapter introduces Kubernetes (K8s) — the industry-standard orchestrator for multi-node clusters. You'll learn:
- **Pods**: The Kubernetes equivalent of a Compose service.
- **Deployments**: Declarative updates with zero downtime.
- **Services**: Built-in load balancing for scaled applications.
- **Kind (Kubernetes in Docker)**: Run a full K8s cluster locally for free.

The mental models you learned here (services, networks, volumes) translate directly to Kubernetes — but K8s adds production-grade features like automatic failover, horizontal scaling across data centers, and declarative rollbacks.

**For now**: Use Docker Compose for local development, testing, and small production deployments (<10 containers on a single machine). When you need to scale beyond one node, graduate to Kubernetes.

---

## What's Next

- **Local**: Run `notebook.ipynb` to build a Flask + Postgres + Redis stack from scratch
- **Azure** (optional): Run `notebook_supplement.ipynb` to deploy multi-container groups to Azure Container Instances (ACI)
- **Ch.3**: Learn Kubernetes fundamentals with Kind (no cloud account required)
