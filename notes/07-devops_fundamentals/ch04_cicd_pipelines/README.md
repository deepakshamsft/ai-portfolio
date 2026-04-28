# Ch.4 — CI/CD Pipelines (GitHub Actions)

> **The story.** In **2000**, the term *Continuous Integration* crystallized when **Martin Fowler** and **Kent Beck** formalized what XP teams already practiced: integrate code multiple times a day, run automated tests, catch breakages immediately. By 2010, "deployment pipeline" had become standard vocabulary (Jez Humble's *Continuous Delivery*), but the tooling — Jenkins, Travis CI — still required dedicated servers. GitHub Actions (2019) made CI/CD configuration as simple as adding a YAML file to your repo. Today every push can trigger tests, build containers, and deploy to production — all on GitHub's infrastructure at no cost for public repos.
>
> **Where you are in the curriculum.** You've learned Docker (Ch.1), orchestrated multi-container apps (Ch.2), and deployed to Kubernetes (Ch.3). But every deployment so far has been manual: you run commands, you push images, you wait. This chapter automates the entire flow — from `git push` to production — using GitHub Actions. The pattern you build here (test → build → deploy) is identical in every CI/CD platform you'll encounter.
>
> **Notation in this chapter.** **Workflow** — a YAML file defining automation rules; **Job** — a unit of work (e.g., "run tests"); **Step** — a single command or action; **Trigger** — event that starts a workflow (push, PR, schedule); **Runner** — GitHub's VM executing your workflow; **Secret** — encrypted environment variable (e.g., Docker Hub token).

---

## 0 · The Challenge — Manual Deployments Are Slow and Error-Prone

> 💡 **The mission**: Deploy a Flask web app automatically on every push to `main` — satisfying 3 constraints:
> 1. **AUTOMATED**: Zero manual steps from commit to production
> 2. **VALIDATED**: All tests must pass before deployment
> 3. **FREE**: Use only GitHub's free tier (2,000 CI/CD minutes/month)

**What we know so far:**
- ✅ We can containerize apps (Ch.1: Docker)
- ✅ We can orchestrate multi-container stacks (Ch.2: Docker Compose)
- ✅ We can deploy to Kubernetes (Ch.3: Kind)
- ❌ **But every deployment is still manual!**

**What's blocking us:**
Manual deployments have three problems:
1. **They're slow**: Developer waits 10 minutes for tests → builds Docker image → pushes to registry → updates K8s deployment
2. **They're error-prone**: Forgot to run tests? Pushed the wrong image tag? Deployed to wrong cluster?
3. **They don't scale**: 10 engineers × 5 deployments/day = 50 chances for human error

Without automation, deployments are the bottleneck. With CI/CD, deployments become invisible.

**What this chapter unlocks:**
A **GitHub Actions pipeline** that runs on every `git push`:
```
Trigger (push to main)
  ↓
Job: Test
  → Run pytest
  → Lint with flake8
  ✅ TEST STAGE: Caught 8 bugs before production (3 logic errors, 2 API contract violations, 3 style issues)
  ↓
Job: Build
  → Build Docker image
  → Push to Docker Hub
  ✅ BUILD STAGE: Created production-ready artifact, validated image builds in 2min 15s
  ↓
Job: Deploy
  → Update Kubernetes deployment
  → Verify rollout success
  ✅ DEPLOY STAGE: Zero-downtime rollout completed, health checks passed, traffic cutover in 45s
```

✅ **This is modern software delivery** — ship confidently, ship often, ship automatically. **Each stage acts as a quality gate**: test failures block builds, build failures block deployments, deployment failures trigger automated rollback.

---

## 1 · CI/CD Automates Test → Build → Deploy

**Continuous Integration (CI):** Merge code to main frequently (daily or hourly), run automated tests on every commit, catch bugs before they reach production.

**Continuous Deployment (CD):** Automatically deploy every commit that passes tests to production. No manual approval gates (except in regulated industries).

**The pipeline pattern:**
1. **Test** — Run unit tests, integration tests, linters. If any fail, stop.
2. **Build** — Compile code, build Docker image, tag with commit SHA.
3. **Deploy** — Push image to registry, update production environment.

This pattern is universal — GitHub Actions, GitLab CI, Azure Pipelines, Jenkins all implement the same flow. Learn it once, apply it everywhere.

---

## 2 · Running Example — Flask App with Automated Deployment

You're a backend engineer at a startup. Your Flask API serves product recommendations. The team ships 20 commits/day. Manual deployments are killing velocity.

**Starting state:**
- Flask app with `/health` and `/predict` endpoints
- Unit tests in `tests/test_app.py`
- Dockerfile that builds production image
- Local Kind cluster for testing

**Target state:**
- Every push to `main` triggers CI/CD
- Tests run first (fail fast)
- Docker image builds and pushes to Docker Hub
- Deployment updates production automatically
- Entire flow completes in <5 minutes

**Tech stack (100% free):**
- **GitHub Actions** — CI/CD platform (2,000 minutes/month free)
- **Docker Hub** — Container registry (unlimited public images)
- **Kind** — Local Kubernetes for testing

---

## 3 · Mental Model — Trigger → Jobs → Steps → Actions

A **workflow** is a YAML file (`.github/workflows/ci-cd.yml`) that defines automation.

### 3.1 · Triggers — What Starts the Workflow

```yaml
on:
  push:
    branches: [main]  # Run on every push to main
  pull_request:       # Run on PRs (before merge)
  schedule:
    - cron: '0 2 * * *'  # Run daily at 2 AM UTC
  workflow_dispatch:  # Allow manual trigger
```

**Common triggers:**
- `push` — Every commit (use branch filters to avoid running on feature branches)
- `pull_request` — Before merging (catch bugs before they reach main)
- `schedule` — Nightly builds, security scans
- `workflow_dispatch` — Manual button in GitHub UI (useful for deployments)

### 3.2 · Jobs — Units of Work That Run in Parallel

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    steps: [...]
  
  build:
    needs: test  # Wait for test job to succeed
    runs-on: ubuntu-latest
    steps: [...]
```

Jobs run on separate VMs. By default they run in parallel. Use `needs` to enforce ordering (e.g., don't build until tests pass).

### 3.3 · Steps — Commands and Actions

```yaml
steps:
  - uses: actions/checkout@v4       # Action: clone repo
  - name: Install dependencies
    run: pip install -r requirements.txt  # Shell command
  - uses: docker/build-push-action@v5    # Action: build Docker image
```

**Step types:**
1. **Shell commands** (`run:`) — Any bash command
2. **Actions** (`uses:`) — Reusable components from GitHub Marketplace
   - `actions/checkout` — Clone your repo
   - `actions/setup-python` — Install Python
   - `docker/build-push-action` — Build and push Docker images

### 3.4 · Secrets — Encrypted Environment Variables

Never commit API keys or passwords to Git. Store them as **repository secrets** (Settings → Secrets and variables → Actions).

```yaml
- name: Push to Docker Hub
  env:
    DOCKER_USERNAME: ${{ secrets.DOCKER_USERNAME }}
    DOCKER_TOKEN: ${{ secrets.DOCKER_TOKEN }}
  run: |
    echo "$DOCKER_TOKEN" | docker login -u "$DOCKER_USERNAME" --password-stdin
    docker push myapp:latest
```

**Mental model:** Secrets live in GitHub's encrypted vault. Workflows access them via `${{ secrets.NAME }}`. They never appear in logs.

---

## 4 · What Can Go Wrong — Three Common Pitfalls

### 5.1 · Secrets Not Set

**Symptom:** `Error: Username and password required`

**Cause:** Forgot to add `DOCKER_USERNAME` and `DOCKER_TOKEN` to repository secrets.

**Fix:**
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add `DOCKER_USERNAME` (your Docker Hub username)
4. Add `DOCKER_TOKEN` (create token at hub.docker.com → Account Settings → Security)

### 5.2 · Workflow Syntax Errors

**Symptom:** Workflow doesn't trigger or fails immediately with YAML parse error

**Cause:** Invalid YAML syntax (wrong indentation, missing colons, tabs instead of spaces)

**Fix:**
- Use a YAML linter (e.g., https://www.yamllint.com/)
- Check GitHub's workflow editor (shows syntax errors in real-time)
- Common mistake: `steps` must be indented under `jobs.<job_id>`

```yaml
# ❌ WRONG
jobs:
  test:
  runs-on: ubuntu-latest

# ✅ CORRECT
jobs:
  test:
    runs-on: ubuntu-latest
```

### 5.3 · Runner Timeout (6 Hours Max)

**Symptom:** Job cancelled after running for hours

**Cause:** GitHub Actions runners have a 6-hour timeout. Long-running tests or builds hit this limit.

**Fix:**
- Add `timeout-minutes: 30` to jobs (fail fast if job hangs)
- Cache dependencies to speed up builds (see notebook cell 7)
- Use matrix builds to parallelize tests across multiple runners

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10  # Kill job if it takes >10 minutes
```

---

## 5 · Progress Check — Debug This Broken Workflow

You pushed this workflow to your repo, but it's failing. Find and fix **three errors**:

```yaml
name: Deploy App

on:
  push:
  branches: [main]

jobs:
  build:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Build image
      run: docker build -t myapp .
    - name: Push to Docker Hub
      run: docker push myapp:latest
```

<details>
<summary><strong>Solution (click to expand)</strong></summary>

**Error 1:** Incorrect indentation for `branches`
```yaml
# ❌ WRONG
on:
  push:
  branches: [main]

# ✅ CORRECT
on:
  push:
    branches: [main]
```

**Error 2:** `runs-on` not indented under `build`
```yaml
# ❌ WRONG
jobs:
  build:
  runs-on: ubuntu-latest

# ✅ CORRECT
jobs:
  build:
    runs-on: ubuntu-latest
```

**Error 3:** Pushing to Docker Hub without logging in
```yaml
# ❌ WRONG (missing login step)
- name: Push to Docker Hub
  run: docker push myapp:latest

# ✅ CORRECT
- name: Log in to Docker Hub
  uses: docker/login-action@v3
  with:
    username: ${{ secrets.DOCKER_USERNAME }}
    password: ${{ secrets.DOCKER_TOKEN }}
- name: Push to Docker Hub
  run: docker push myapp:latest
```

**Fixed workflow:**
```yaml
name: Deploy App

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t myapp .
      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_TOKEN }}
      - name: Push to Docker Hub
        run: docker push ${{ secrets.DOCKER_USERNAME }}/myapp:latest
```

</details>

---

## 6 · Bridge to Ch.5 — Monitoring Catches Issues After Deployment

CI/CD ensures your code *deploys* successfully. But what happens after deployment?
- Is the app responding to requests?
- Are error rates spiking?
- Is latency within SLA?

**Ch.5 (Monitoring & Observability)** adds the missing layer: Prometheus for metrics, Grafana for dashboards, Alertmanager for notifications. You'll instrument your Flask app to emit custom metrics (request rate, latency, errors) and visualize them in real-time.

**The full production loop:**
```
Code → CI/CD → Deployment → Monitoring → Alerting → (back to Code)
```

CI/CD is the delivery mechanism. Monitoring is the feedback loop that tells you if what you delivered actually works.

---

## Further Reading

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Push Action](https://github.com/docker/build-push-action)
- *Continuous Delivery* by Jez Humble (2010) — The foundational book on deployment pipelines
- [GitHub Actions free tier limits](https://docs.github.com/en/billing/managing-billing-for-github-actions/about-billing-for-github-actions)

---

**Next:** [Ch.5 — Monitoring & Observability →](../ch05_monitoring_observability)
