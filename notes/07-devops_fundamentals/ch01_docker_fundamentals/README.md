# Ch.1 — Docker Fundamentals

> **The story.** In **2013**, Solomon Hykes and the team at dotCloud introduced Docker at PyCon — a tool that would transform software deployment from "works on my machine" uncertainty into reproducible container-based workflows. By 2015, Docker had become the de facto standard for packaging applications, solving the ancient problem of dependency conflicts and environment drift. Every Kubernetes pod, every CI/CD pipeline, every microservice deployment you'll see in production today runs inside a container — and Docker established the vocabulary and tooling that made it possible.
>
> **Where you are in the curriculum.** This is chapter one of the DevOps Fundamentals track. You're an engineer deploying a Python Flask web application, and your constraint is simple but absolute: it must run identically on your dev laptop and the production server. One Dockerfile, one build, infinite deployments. Every concept here — images, containers, volumes, networks — scales directly to Kubernetes orchestration in Ch.3 and multi-service architectures throughout this track.
>
> **Notation in this chapter.** `Dockerfile` — blueprint for building images; `image` — read-only template; `container` — running instance of an image; `volume` — persistent storage mounted into containers; `port mapping` — exposing container ports to the host; `layer` — cached filesystem change in an image.

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: Deploy a **production-ready Flask web app** satisfying 5 constraints:
> 1. **PORTABILITY**: Runs identically on dev, staging, production — 2. **REPRODUCIBILITY**: Same build every time — 3. **ISOLATION**: No dependency conflicts with host — 4. **EFFICIENCY**: Fast builds, small images — 5. **OBSERVABILITY**: Logs, debugging, health checks

**What we know so far:**
- ✅ We have a Flask app with Redis cache (standard 3-tier architecture)
- ✅ It works on the developer's laptop (Python 3.11, Redis installed locally)
- ❌ **But it fails on the production server!** Different Python version, missing Redis, wrong environment variables

**What's blocking us:**
We need **containerization** — packaging the app and all its dependencies into a single, portable unit. Without containers:
- Deployment means manual setup on every server (Python, pip, Redis, environment config)
- "It works on my machine" becomes a recurring nightmare
- Rollbacks require manual reversal of dozens of shell commands
- Scaling means repeating the entire setup process for each new server

**What this chapter unlocks:**
The **Docker workflow** — write once, run anywhere:
- **Dockerfile defines the build**: One file specifies Python version, dependencies, startup command
- **Image is the blueprint**: Built once, pushed to registry, pulled by any server
- **Container is the runtime**: Start, stop, inspect, scale without touching the host OS

✅ **This is the foundation** — every later chapter assumes your app is containerized.

---

## Animation

![Docker container lifecycle](img/ch01-container-lifecycle.gif)

> **What you're seeing:** The full lifecycle of a Docker container — write Dockerfile → build image → run container → inspect logs → exec into running container → stop → remove. Each frame shows the command and the resulting state. The animation demonstrates that containers are ephemeral (stop/remove destroys state) but images persist (rebuilt containers start fresh). This is the mental model you'll use for every deployment: **images are blueprints, containers are running instances**.

---

## 1 · Docker Solves "Works on My Machine" with Lightweight Containers

Docker packages your application, runtime, and dependencies into a **container** — a lightweight, isolated environment that runs identically on any machine with Docker installed. Unlike virtual machines (which virtualize hardware), containers share the host OS kernel, making them fast to start and efficient in resource usage.

**The core abstraction:**
- **Image** = read-only template (the blueprint)
- **Container** = running instance of an image (the house built from the blueprint)

Every container starts from an image. You build an image once (defining Python version, installing dependencies, copying code) and run it anywhere — your laptop, a staging server, a cloud VM, a Kubernetes cluster. The container always sees the same filesystem, the same Python version, the same installed packages. "Works on my machine" is no longer a problem — because the machine is now *inside the container*.

---

## 2 · Containerizing a Flask App with Redis Cache

You're a backend engineer at a startup. Your first production deployment is a Flask API that stores session data in Redis. The app works perfectly on your laptop — Python 3.11, `flask==3.0.0`, `redis==5.0.0`, and a locally running Redis server. The staging server has Python 3.9, no Redis, and conflicting system packages. Deployment fails.

Your manager's requirement: **the app must run identically on dev, staging, and production with zero manual setup**. One `docker run` command. No environment-specific instructions. No Slack messages asking "did you install Redis?"

**The running example: Flask + Redis in 5 steps**

| Step | What you do | Why it matters |
|------|-------------|----------------|
| **1. Write Dockerfile** | Define base image (`python:3.11-slim`), install deps, copy code | Creates reproducible build instructions |
| **2. Build image** | `docker build -t flask-app:v1 .` | Packages app into portable image |
| **3. Run container** | `docker run -p 5000:5000 flask-app:v1` | Starts isolated instance with port mapping |
| **4. Add Redis** | Multi-container with Docker network | Enables inter-service communication |
| **5. Persist data** | Mount volume for Redis data | Survives container restarts |

By step 5, you have a **production-ready deployment**: two containers (Flask + Redis), networked together, with persistent storage. One command starts both. One command stops both. The entire stack is defined in version-controlled files — no manual setup required.

---

## 3 · Mental Model — Image vs. Container (Blueprint vs. Instance)

> 💡 **The analogy that never fails:** An **image** is like a class definition in Python. A **container** is like an object instantiated from that class. You can create 10 containers from one image — they all start with the same code and dependencies, but they run independently with separate memory and state.

**Image:**
- Read-only filesystem layers
- Built once from a Dockerfile
- Stored in a registry (Docker Hub, AWS ECR, GitHub Container Registry)
- Versioned with tags (`flask-app:v1`, `flask-app:v2`, `flask-app:latest`)
- **Never changes after build** — if you modify code, you build a new image

**Container:**
- Running (or stopped) instance of an image
- Has its own filesystem, network, and process space
- Can write to its filesystem (but changes are lost when removed)
- Ephemeral by default — stop/remove destroys all state unless volumes are used
- **Can run multiple containers from the same image** — each gets isolated resources

**The lifecycle:**
```
Dockerfile → build → Image → run → Container
                        ↓
                    push to registry → pull on production → run → Container
```

**Key insight:** When you run `docker build`, you're creating a **read-only template**. When you run `docker run`, you're creating a **writable runtime instance**. This separation is why Docker enables immutable infrastructure — you never patch a running container. You build a new image and replace the old container.

---

## 4 · Step-by-Step — Build, Run, Debug

### 4.1 · Build the Image

```bash
# Build image tagged as flask-app:v1
docker build -t flask-app:v1 .
```

**What happens:**
1. Docker sends build context (all files not in `.dockerignore`) to daemon
2. Processes each Dockerfile instruction sequentially
3. Creates a layer for each instruction
4. Tags the final image as `flask-app:v1`

**Verify the image exists:**
```bash
docker images
# REPOSITORY    TAG    IMAGE ID       CREATED         SIZE
# flask-app     v1     a3c5d9f8b2e1   10 seconds ago  120MB
```

### 4.2 · Run the Container

```bash
# Run container, map host port 5000 to container port 5000
docker run -d -p 5000:5000 --name flask-api flask-app:v1
```

**Flags explained:**
- `-d` — detached mode (run in background)
- `-p 5000:5000` — map host port 5000 to container port 5000
- `--name flask-api` — assign human-readable name (instead of random hash)
- `flask-app:v1` — image to run

**Port mapping syntax:** `-p HOST_PORT:CONTAINER_PORT`

| Example | Meaning |
|---------|---------|
| `-p 5000:5000` | Host port 5000 → Container port 5000 |
| `-p 8080:5000` | Host port 8080 → Container port 5000 |
| `-p 5000:5000/tcp` | Explicit TCP (default) |

**Test the app:**
```bash
curl http://localhost:5000/
# {"status": "ok", "message": "Flask app running"}
```

### 4.3 · Inspect Logs

```bash
# View container logs (stdout/stderr)
docker logs flask-api

# Follow logs in real-time
docker logs -f flask-api
```

**Output:**
```
 * Serving Flask app 'app'
 * Running on http://0.0.0.0:5000
 * Debug mode: off
```

> 💡 **Best practice:** Flask apps should log to stdout, not files. Docker captures stdout/stderr automatically. In production, logs are shipped to centralized logging (e.g., Elasticsearch, Splunk) directly from container output.

### 4.4 · Execute Commands Inside Running Container

```bash
# Open interactive shell in running container
docker exec -it flask-api /bin/bash

# Run single command
docker exec flask-api ps aux
```

**Flags explained:**
- `-i` — interactive (keep stdin open)
- `-t` — allocate pseudo-TTY (enables shell features like tab completion)

**Use cases:**
- Debug dependency issues: `docker exec flask-api pip list`
- Check network connectivity: `docker exec flask-api ping redis`
- Inspect environment: `docker exec flask-api env`

> ⚠️ **Never exec into production containers to "fix" issues.** If a container is broken, stop it and deploy a new one from a fixed image. Exec is for **debugging**, not **patching**. Manual changes inside containers are lost on restart — the fix must be in the Dockerfile.

### 4.5 · Stop and Remove Container

```bash
# Stop gracefully (sends SIGTERM, waits 10s, then SIGKILL)
docker stop flask-api

# Remove stopped container
docker rm flask-api

# Stop and remove in one command
docker rm -f flask-api
```

**Why removal is necessary:**
Stopped containers still consume disk space and clutter `docker ps -a` output. In production, use `docker run --rm` to auto-remove on exit, or set up container orchestration (Docker Compose, Kubernetes) to manage lifecycle automatically.

---

## 5 · Volumes — Persistent Data Survives Container Restarts

### 5.1 · The Problem: Containers Are Ephemeral

**Scenario:** You add a Redis container for session storage.

```bash
docker run -d --name redis redis:7
```

Redis stores data inside the container's filesystem. When you stop and remove the container, **all data is lost**. This is by design — containers are ephemeral. For stateful services (databases, caches, file storage), you need **volumes**.

### 5.2 · Named Volumes (Recommended)

```bash
# Create named volume
docker volume create redis-data

# Run Redis with volume mounted
docker run -d \
  --name redis \
  -v redis-data:/data \
  redis:7

# Verify volume persists after container removal
docker rm -f redis
docker volume ls
# DRIVER    VOLUME NAME
# local     redis-data

# New container reuses same data
docker run -d --name redis -v redis-data:/data redis:7
```

**Volume mount syntax:** `-v VOLUME_NAME:CONTAINER_PATH`

**Key behavior:**
- Volume lives **outside the container** — managed by Docker
- Survives container stop/remove
- Can be shared between multiple containers (e.g., read-only mounts for shared config)
- Backed up independently (e.g., snapshot the volume, not the container)

### 5.3 · Bind Mounts (Development Only)

```bash
# Mount host directory into container
docker run -d \
  -p 5000:5000 \
  -v $(pwd)/app:/app \
  flask-app:v1
```

**Bind mount syntax:** `-v HOST_PATH:CONTAINER_PATH`

**Use case:** During development, mount source code directory so changes on host immediately reflect in container (no rebuild required). **Never use in production** — breaks portability and creates security risks.

**Development workflow:**
1. Edit `app.py` on host
2. Flask auto-reloads (debug mode)
3. Test changes instantly (no rebuild)

**Comparison:**

| Type | Managed by | Persistence | Use case |
|------|-----------|-------------|----------|
| **Named volume** | Docker | Survives container removal | Production databases, caches |
| **Bind mount** | Host filesystem | Tied to host directory | Development (live code editing) |
| **tmpfs mount** | Memory | Lost on container stop | Sensitive data (e.g., temporary tokens) |

---

## 6 · Multi-Container Setup — Flask + Redis Network

### 6.1 · Create Custom Network

```bash
# Create bridge network
docker network create flask-net
```

**Why networks matter:**
By default, containers are isolated. To enable Flask → Redis communication, both must be on the same Docker network. Containers on the same network can resolve each other **by container name** (automatic DNS).

### 6.2 · Run Redis on Custom Network

```bash
docker run -d \
  --name redis \
  --network flask-net \
  -v redis-data:/data \
  redis:7
```

### 6.3 · Run Flask with Redis Connection

```bash
docker run -d \
  --name flask-api \
  --network flask-net \
  -p 5000:5000 \
  -e REDIS_HOST=redis \
  flask-app:v1
```

**Environment variable injection:** `-e KEY=value`

**Flask app code:**
```python
import redis
import os

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
r = redis.Redis(host=REDIS_HOST, port=6379)
```

**How DNS resolution works:**
- Flask container resolves `redis` → Redis container's IP (e.g., `172.18.0.2`)
- No hardcoded IPs — portable across environments
- Docker's embedded DNS server handles name resolution automatically

### 6.4 · Verify Communication

```bash
# From Flask container, ping Redis
docker exec flask-api ping redis
# PING redis (172.18.0.2): 56 data bytes
# 64 bytes from 172.18.0.2: icmp_seq=0 ttl=64 time=0.123 ms
```

> 💡 **Bridge to Ch.2:** Managing two containers with manual `docker run` commands is fragile. Docker Compose (Ch.2) defines multi-container apps in a single YAML file — one command starts the entire stack with correct networks, volumes, and environment variables.

---

## 7 · Image Registries — Sharing Images Across Machines

### 7.1 · Tagging for Docker Hub

```bash
# Tag image with Docker Hub username
docker tag flask-app:v1 yourusername/flask-app:v1
```

**Tag format:** `REGISTRY/NAMESPACE/REPOSITORY:TAG`

| Example | Meaning |
|---------|---------|
| `flask-app:v1` | Local image (no registry) |
| `yourusername/flask-app:v1` | Docker Hub (default registry) |
| `ghcr.io/yourorg/flask-app:v1` | GitHub Container Registry |
| `123456.dkr.ecr.us-east-1.amazonaws.com/flask-app:v1` | AWS ECR |

### 7.2 · Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push image
docker push yourusername/flask-app:v1
```

**What happens:**
1. Docker compresses each layer
2. Uploads only changed layers (other layers already in registry)
3. Registry stores layers with content-addressable hashes (deduplication)

**Pull on production server:**
```bash
docker pull yourusername/flask-app:v1
docker run -d -p 5000:5000 yourusername/flask-app:v1
```

> ⚡ **Image size matters for deployment speed.** A 1 GB image takes ~2 minutes to pull over a 100 Mbps connection. A 120 MB image pulls in 10 seconds. Multi-stage builds and `.dockerignore` are not optional optimizations — they're production requirements.

---

## 8 · What Can Go Wrong

### 8.1 · Port Already in Use

**Symptom:**
```
Error starting userland proxy: listen tcp 0.0.0.0:5000: bind: address already in use
```

**Cause:** Another process (or stopped container) is using port 5000 on the host.

**Fix:**
```bash
# Find process using port 5000
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process or use different host port
docker run -d -p 8080:5000 flask-app:v1
```

### 8.2 · Image Bloat (500 MB+ Images)

**Symptom:** Image size is 800 MB for a simple Flask app.

**Causes:**
- Used `python:3.11` instead of `python:3.11-slim` (+600 MB)
- No `.dockerignore` — copied `venv/`, `.git/` (+300 MB)
- Installed build tools (`gcc`, `make`) and didn't remove them

**Fix checklist:**
1. ✅ Use `-slim` or `-alpine` base images
2. ✅ Add comprehensive `.dockerignore`
3. ✅ Use multi-stage builds to discard build tools
4. ✅ Chain `RUN` commands to reduce layers: `RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*`

**Target sizes:**

| App type | Reasonable size | Bloated size |
|----------|----------------|-------------|
| Python Flask | 100–150 MB | 500–800 MB |
| Node.js Express | 80–120 MB | 400–600 MB |
| Go binary | 10–20 MB | 100–200 MB |

### 8.3 · Layer Cache Not Working

**Symptom:** Every build reinstalls dependencies even when `requirements.txt` unchanged.

**Cause:** `COPY . .` appears before `COPY requirements.txt .` — any file change invalidates cache.

**Fix:** Reorder Dockerfile instructions by copying requirements.txt before application code to maximize cache hits.

**Cache invalidation order:**
```dockerfile
FROM python:3.11-slim        # Cached (base image unchanged)
WORKDIR /app                 # Cached (instruction unchanged)
COPY requirements.txt .      # Cached (file unchanged)
RUN pip install -r requirements.txt  # Cached (previous layer unchanged)
COPY . .                     # NOT CACHED (app.py changed)
CMD ["python", "app.py"]     # NOT CACHED (previous layer changed)
```

### 8.4 · Secrets Leaked in Image

**Symptom:** `docker history flask-app:v1` reveals API keys.

**Cause:**
```dockerfile
COPY .env .  # .env contains SECRET_KEY=abc123
RUN rm .env  # Too late — file exists in previous layer
```

**Fix:** **Never copy secrets into images.** Use:
- Environment variables at runtime: `docker run -e SECRET_KEY=abc123`
- Docker secrets (Swarm mode): `echo "abc123" | docker secret create secret_key -`
- Kubernetes secrets: `kubectl create secret generic api-keys --from-literal=key=abc123`
- Secret management services: AWS Secrets Manager, HashiCorp Vault

**Verify no secrets in layers:**
```bash
docker history flask-app:v1
# Check each layer's SIZE — large layers may contain secrets
```

### 8.5 · Container Exits Immediately

**Symptom:**
```bash
docker run flask-app:v1
# Container exits after 1 second
```

**Cause:** `CMD` instruction starts a process that exits immediately (e.g., `CMD ["echo", "hello"]`).

**Debug:**
```bash
# Check exit code and logs
docker ps -a
# STATUS: Exited (1) 5 seconds ago

docker logs <container-id>
# Error: Missing REDIS_HOST environment variable
```

**Fix:** Ensure `CMD` runs a **long-lived process** (web server, worker queue). For debugging, override command:
```bash
docker run -it flask-app:v1 /bin/bash
```

---

## 9 · Progress Check

> **Goal:** Verify you can build, run, debug, and optimize containerized applications. These questions cover the core concepts — if you can answer them without looking back, you're ready for Ch.2 (multi-container orchestration).

**1. Image layers and caching**

You change `app.py` in your Flask application. The Dockerfile is:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

**Question:** Why does Docker reinstall all dependencies (60s delay) even though `requirements.txt` didn't change?

<details>
<summary>Answer</summary>

`COPY . .` appears **before** `pip install`. When `app.py` changes, the `COPY . .` layer is invalidated. All subsequent layers are rebuilt, including `pip install`. 

**Fix:** Copy `requirements.txt` first, install dependencies, then copy application code:
```dockerfile
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

Now changing `app.py` only invalidates the final `COPY` layer — `pip install` stays cached.

</details>

**2. Port mapping troubleshooting**

You run:
```bash
docker run -d -p 8080:5000 flask-app:v1
curl http://localhost:8080/
# curl: (7) Failed to connect to localhost port 8080
```

The container is running (`docker ps` confirms it). What are three possible causes?

<details>
<summary>Answer</summary>

1. **Flask is bound to 127.0.0.1 inside container** — change to `app.run(host='0.0.0.0')` to accept external connections
2. **Container crashed after startup** — check logs: `docker logs <container-id>`
3. **Firewall blocking port 8080 on host** — test with `telnet localhost 8080`

**Key insight:** `-p 8080:5000` maps **host port 8080** to **container port 5000**. The Flask app must listen on `0.0.0.0:5000` (not `127.0.0.1:5000`) to accept connections from outside the container.

</details>

**3. Volume persistence**

You run Redis with:
```bash
docker run -d --name redis redis:7
```

You write data to Redis, then run:
```bash
docker stop redis
docker start redis
```

**Question A:** Is the data still there?  
**Question B:** What happens if you run `docker rm redis` instead of `docker start redis`?  
**Question C:** How do you ensure data survives container removal?

<details>
<summary>Answer</summary>

**A:** Yes — `docker stop` + `docker start` preserves the container's filesystem. Data survives.

**B:** `docker rm redis` deletes the container **and its filesystem**. All data is lost.

**C:** Use a **named volume**:
```bash
docker volume create redis-data
docker run -d --name redis -v redis-data:/data redis:7
```

Now data is stored in the volume (outside the container). Removing the container doesn't delete the volume. A new container can mount the same volume and access the data.

**Key lesson:** Containers are ephemeral. Volumes are persistent. For any stateful service (database, cache, file storage), mount a volume.

</details>

---

## 10 · Bridge to Chapter 2 — Multi-Container Apps Need Orchestration

You've successfully containerized a Flask app with Redis. But manual `docker run` commands become fragile at scale:

**What we have now:**
```bash
docker network create flask-net
docker run -d --name redis --network flask-net -v redis-data:/data redis:7
docker run -d --name flask-api --network flask-net -p 5000:5000 -e REDIS_HOST=redis flask-app:v1
```

**What breaks in production:**
- **Startup ordering:** Flask starts before Redis finishes initializing → connection refused
- **Dependency tracking:** Stopping Redis should stop Flask (dependent service)
- **Environment drift:** Different `.env` files on dev/staging/prod → manual synchronization
- **Scaling:** Running 5 Flask containers requires 5 manual commands with different ports

**What Ch.2 introduces:**
**Docker Compose** — define multi-container apps in a single YAML file:
```yaml
services:
  redis:
    image: redis:7
    volumes:
      - redis-data:/data
  flask-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      REDIS_HOST: redis
    depends_on:
      - redis

volumes:
  redis-data:
```

One command: `docker compose up`  
One command: `docker compose down`

**The progression:**
- **Ch.1 (Docker):** Single-container workflows — build, run, debug
- **Ch.2 (Compose):** Multi-container workflows — define services, networks, volumes declaratively
- **Ch.3 (Kubernetes):** Multi-host orchestration — scale across servers, auto-healing, load balancing

✅ You now understand containers at the primitive level. Ch.2 builds the orchestration layer on top.

---

## Further Reading

**Official Docker documentation:**
- [Dockerfile best practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker networking](https://docs.docker.com/network/)
- [Docker volumes](https://docs.docker.com/storage/volumes/)

**Security:**
- [Dockerfile security best practices (Snyk)](https://snyk.io/blog/10-docker-image-security-best-practices/)
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker)

**Advanced topics (covered in later chapters):**
- Multi-stage builds for microservices (Ch.2)
- Health checks and restart policies (Ch.2)
- Resource limits (CPU, memory) (Ch.3 — Kubernetes)
- Container scanning for vulnerabilities (Ch.8 — Security)

---

**Next:** [Ch.2 — Container Orchestration](../ch02_container_orchestration) — Docker Compose for multi-service applications.
