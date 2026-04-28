# Ch.8 — Security & Secrets Management

> **The story.** In **2014**, a single GitHub commit exposed the private keys for **50,000 AWS accounts** when developers pushed code containing hardcoded credentials. The incident cost organizations millions in compromised resources and led to the creation of automated secret scanning in CI/CD pipelines. By 2018, Docker Secrets, Kubernetes Secrets, and cloud-native secret managers (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault) had become standard practice. Every production deployment you'll build depends on secrets — database passwords, API keys, TLS certificates — and the difference between secure and catastrophic is knowing **secrets are runtime configuration, never build artifacts**.
>
> **Where you are in the curriculum.** This is chapter eight of the DevOps Fundamentals track. You've containerized apps (Ch.1), orchestrated services (Ch.2-3), automated deployments (Ch.4), and monitored systems (Ch.5). Now you're securing production deployments. Every container you run needs credentials — database passwords, API tokens, cloud provider keys. One hardcoded secret in a Dockerfile means every image pushed to a registry is a security breach waiting to happen. This chapter teaches you to **separate secrets from code** and **rotate credentials without redeploying**.
>
> **Notation in this chapter.** `secret` — sensitive data requiring access control (passwords, keys, tokens); `.env` — environment variable file (local dev only, never committed); `Docker Secrets` — encrypted secret distribution in Swarm mode; `Kubernetes Secret` — base64-encoded secret mounted as volume or env var; `secret rotation` — replacing credentials without downtime; `RBAC` — role-based access control (who can access what); `pre-commit hook` — git hook that blocks commits containing secrets.

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: Deploy a **production Flask app with database access** satisfying 5 constraints:
> 1. **NO SECRETS IN GIT**: Database password never appears in version control
> 2. **NO SECRETS IN IMAGES**: Docker image can be public — no credentials leaked
> 3. **RUNTIME INJECTION**: Secrets loaded at container startup from secure store
> 4. **ROTATION READY**: Can change password without rebuilding images
> 5. **AUDIT TRAIL**: Know who accessed secrets and when

**What we know so far:**
- ✅ We've containerized a Flask app (Ch.1)
- ✅ We've deployed it with Docker Compose (Ch.2)
- ✅ We've pushed images to registries (Ch.4)
- ❌ **But the database password is hardcoded in the Dockerfile!** Anyone with image access has credentials

**What's blocking us:**
We need **secrets management** — a secure way to provide credentials to running containers without embedding them in code or images. Without proper secrets handling:
- Hardcoded passwords in Dockerfiles leak when images are pushed to public registries
- Environment variables in docker-compose.yml are visible in process lists and logs
- Developers copy-paste credentials into Slack or email for debugging
- Rotating a compromised password requires rebuilding and redeploying every image
- No audit trail of who accessed what secret when

**What this chapter unlocks:**
The **secrets management workflow** — secure credentials from creation to revocation:
- **Build time**: No secrets in Dockerfile, docker-compose.yml, or git
- **Runtime**: Secrets injected via environment variables, mounted files, or secret stores
- **Rotation**: Update credentials in the secret store, restart containers — no rebuild
- **Audit**: Track secret access, enforce least privilege with RBAC

✅ **This is the security foundation** — every production deployment requires proper secrets handling.

---

## Animation

![Secrets lifecycle](img/ch08-secrets-lifecycle.gif)

> **What you're seeing:** The full lifecycle of a production secret — **Create** (generate in secret store) → **Store** (encrypted at rest) → **Access** (injected into container at runtime) → **Rotate** (update without downtime) → **Revoke** (remove access immediately). Each frame shows the security properties at that stage. The animation demonstrates that secrets never touch git, images, or build-time configuration — they're always runtime dependencies fetched from secure stores. This is the mental model for every production credential.

---

## 1 · Secrets Are Runtime Configuration, Not Build-Time Artifacts

Secrets are **data your application needs at runtime** but **must never be part of the build**. When you write a Dockerfile and run `docker build`, the resulting image should be **publishable to a public registry** without leaking credentials. This means:

- **No `ENV DB_PASSWORD=secret123` in Dockerfile** — environment variables at build time are baked into image layers
- **No `COPY .env /app/.env` in Dockerfile** — copying secret files into the image makes them visible in `docker history`
- **No hardcoded strings in source code** — `conn_string = "postgres://user:password@db:5432"` is a security breach

**The core principle:**
Secrets are **passed to containers**, not **embedded in containers**. At runtime, the container receives credentials through one of these mechanisms:
- **Environment variables** — `docker run -e DB_PASSWORD=secret123` (visible in `docker inspect`, use only for local dev)
- **Mounted secret files** — Docker Secrets, Kubernetes Secrets (files appear in `/run/secrets/`)
- **Secret stores** — Azure Key Vault, AWS Secrets Manager, HashiCorp Vault (fetched at startup via SDK)

The image itself contains **zero secrets**. Change the database password? Just update the secret store and restart containers — no rebuild, no redeploy.

---

## 2 · Securing a Flask App with Database Credentials

You're a backend engineer at a fintech startup. Your Flask API connects to a PostgreSQL database to store transaction records. The app works perfectly in development — you've hardcoded `DB_PASSWORD=dev123` in the Dockerfile for convenience. Your manager now requires **SOC 2 compliance**: no secrets in version control, all credential access must be audited, passwords must rotate every 90 days.

Your task: **refactor the deployment to use Docker Secrets in production and Azure Key Vault in the cloud**. The application code should never contain credentials. The CI/CD pipeline should block any commit that contains secret-like strings. The database password should be rotatable without downtime.

**The running example: Flask + PostgreSQL with secrets in 4 steps**

| Step | What you do | Why it matters |
|------|-------------|----------------|
| **1. .env file (local dev)** | `DB_PASSWORD=dev123` in `.env`, loaded by `python-dotenv` | Keeps secrets out of code, `.env` is gitignored |
| **2. Docker Compose secrets** | `docker-compose.yml` defines secrets, mounts as files | Secrets never in environment variables or images |
| **3. Kubernetes secrets** | `kubectl create secret` → mounted as volume in pod | Production-grade secret distribution |
| **4. Pre-commit hook** | Git hook scans for patterns like `password=`, `api_key=` | Prevents accidental secret commits |

By step 4, you have **SOC 2-compliant secrets handling**: no credentials in git, runtime-only injection, audit logs via secret store, rotation without redeployment.

---

## 3 · Mental Model — Build-Time vs. Runtime vs. Secret Stores

> 💡 **The analogy that never fails:** **Build time** is like constructing a house — you don't install the safe combination in the walls. **Runtime** is like moving in — the combination is handed to you separately. **Secret stores** are like a bank vault — the combination is kept offsite, only accessible to authorized residents.

**Build time (Dockerfile):**
- Packages application code, dependencies, runtime (Python, Flask)
- **Should contain zero secrets** — the image can be pushed to Docker Hub publicly
- Environment variables set with `ENV` are **baked into the image**, visible in `docker history`
- If you ever need to rotate a secret, you'd have to rebuild and redeploy — unacceptable in production

**Runtime (Container startup):**
- Secrets injected via `-e` flag, Docker Secrets, or Kubernetes Secrets
- Container reads secrets from environment variables or mounted files (`/run/secrets/db_password`)
- Secrets are **ephemeral** — stop the container, the secrets disappear from memory
- Rotate a secret? Update the source, restart the container — no image rebuild

**Secret stores (Azure Key Vault, AWS Secrets Manager, HashiCorp Vault):**
- Centralized, encrypted storage for secrets
- Access control via RBAC (only authorized services can read specific secrets)
- Audit logs track every secret access (who, what, when)
- Secrets can be rotated in the store — all containers fetch the new value on restart or refresh

**The lifecycle:**
```
Developer writes code (no secrets)
    ↓
Dockerfile builds image (no secrets)
    ↓
Image pushed to registry (public or private, no secrets leaked)
    ↓
Container started with secret injection:
    • Local dev: .env file
    • Docker Compose: Docker Secrets
    • Kubernetes: K8s Secrets
    • Cloud: Azure Key Vault / AWS Secrets Manager
    ↓
Application reads secret at runtime
    ↓
Secret rotated in store (container restarts or refreshes)
      ✅ WHAT THIS PREVENTS:
      • Compromised credentials remain valid indefinitely (rotation limits exposure window to 90 days max)
      • Single leaked password grants permanent access (old password stops working after rotation)
      • Insider threats (ex-employee's cached credentials expire on rotation schedule)
      • Compliance violations (SOC 2, PCI-DSS require 90-day password rotation)
```

---

## 4 · What Can Go Wrong — Common Pitfalls

| Problem | Why it happens | How to fix |
|---------|----------------|------------|
| **Secrets in logs** | Application logs `DB_PASSWORD` during connection error | Sanitize logs: `logger.info(f"Connecting to {host}:****")` |
| **Base64 ≠ encryption** | Kubernetes secrets are base64-encoded, not encrypted at rest | Enable encryption at rest (EncryptionConfiguration in K8s) |
| **Secrets in environment variables** | `docker inspect` reveals all `-e` vars | Use Docker Secrets or mounted files instead |
| **Secrets in `.bash_history`** | `docker run -e PASSWORD=secret` is saved in shell history | Use `--env-file` or secret mounts |
| **Stale secrets** | Password rotated in Key Vault, but container still uses cached old value | Implement secret refresh (restart container or poll vault) |
| **Over-privileged access** | All containers can read all secrets | Use RBAC: each service gets only its required secrets |
| **No audit trail** | Can't determine who accessed a secret | Enable audit logs in Key Vault / Secrets Manager |

### Real-world example: The Docker Hub Leak

In 2019, a developer pushed a Docker image to Docker Hub with `ENV AWS_ACCESS_KEY_ID=AKIA...` in the Dockerfile. The image was public. Within hours, attackers used the key to spin up 100 EC2 instances for cryptocurrency mining. Cost: $20,000 in 3 days.

**The fix:** Never use `ENV` for secrets. Always inject at runtime.

---

## 5 · Progress Check — Can You Audit This Dockerfile for Security Issues?

You're reviewing a colleague's Dockerfile for a Node.js API:

```dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
ENV DATABASE_URL=postgres://user:mypassword123@db:5432/prod
ENV API_KEY=sk_live_abcdef1234567890
CMD ["node", "server.js"]
```

**Task:** Identify **three** security issues and propose fixes.

<details>
<summary>Click to reveal answers</summary>

**Issues:**
1. **Hardcoded database password in `ENV DATABASE_URL`**  
   → **Fix:** Remove `ENV` line, pass `DATABASE_URL` at runtime via `-e` or Docker Secrets

2. **Hardcoded API key in `ENV API_KEY`**  
   → **Fix:** Store in secret manager (Key Vault, Secrets Manager), fetch at container startup

3. **Secrets are visible in image history**  
   → **Fix:** Run `docker history <image>` and you'll see both secrets in plaintext

**Secure version:**
```dockerfile
FROM node:18
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# No secrets in Dockerfile!
CMD ["node", "server.js"]
```

**Runtime injection:**
```bash
docker run -e DATABASE_URL=$DATABASE_URL -e API_KEY=$API_KEY my-api:latest
# Or use Docker Secrets / K8s Secrets
```

</details>

---

## 6 · Bridge to Future — Applying DevOps to AI/ML Deployments

Every concept in this chapter applies directly to **AI Infrastructure** deployments (Ch.9 onwards):

| DevOps Skill | AI/ML Application |
|--------------|-------------------|
| **Docker Secrets** | Secure API keys for OpenAI, Anthropic, Cohere in inference containers |
| **Kubernetes Secrets** | Store Azure OpenAI endpoint keys for distributed training jobs |
| **Pre-commit hooks** | Block commits containing `.env` with cloud provider credentials |
| **Secret rotation** | Rotate fine-tuning API keys without redeploying ML models |
| **RBAC** | Restrict which services can access production model endpoints |
| **Audit logs** | Track which team members accessed proprietary training data credentials |

**Next steps:**
- **Ch.9 (AI Infrastructure)**: Deploy LLM inference APIs with Azure Key Vault
- **Ch.10 (Model Serving)**: Secure ML model endpoints with API key rotation
- **Ch.11 (MLOps)**: Track experiment credentials in MLflow with secret backends

**The through-line:** Secrets management isn't just a DevOps checkbox — it's the foundation of **production AI security**. Every model deployment, every API call, every data pipeline requires credentials. This chapter taught you to handle them correctly.

---

## What's Next?

You've completed the DevOps Fundamentals track! You can:
1. **Containerize apps** (Docker)
2. **Orchestrate services** (Docker Compose, Kubernetes)
3. **Automate deployments** (CI/CD pipelines)
4. **Monitor systems** (Prometheus, Grafana)
5. **Secure credentials** (Secrets management)

**Continue to:** [AI Infrastructure Track](../../ai_infrastructure/README.md) — Apply these skills to deploying production AI systems.

**Practice:** Build a complete secure microservices stack — three services (Flask, FastAPI, Redis), secrets in Key Vault, CI/CD with GitHub Actions, monitoring with Prometheus.
