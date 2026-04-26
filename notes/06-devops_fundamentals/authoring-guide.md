# DevOps Fundamentals Track — Authoring Guide

> **This document tracks the chapter-by-chapter build of the DevOps Fundamentals notes library.**  
> Each chapter lives under `notes/06-devops_fundamentals/` in its own folder, containing a README and a Jupyter notebook.  
> Read this before starting any chapter to keep tone, structure, and the running example consistent.
>
> **📚 Updated:** Adapted from ML authoring guide with infrastructure-focused conventions.

<!-- LLM-STYLE-FINGERPRINT-V1
canonical_chapters: ["notes/06-devops_fundamentals/ch01_docker_fundamentals/README.md", "notes/06-devops_fundamentals/ch02_container_orchestration/README.md"]
voice: second_person_practitioner
register: technical_but_conversational
concept_motivation: required_before_each_tool
hands_on_walkthroughs: production_stack_flask_postgresql_redis_explicit_commands
running_example: production_stack_only_no_synthetic_except_minimal_demos
failure_first_pedagogy: true
callout_system: {insight:"💡", warning:"⚠️", constraint:"⚡", optional_depth:"📖", forward_pointer:"➡️"}
mermaid_color_palette: {primary:"#1e3a8a", success:"#15803d", caution:"#b45309", danger:"#b91c1c", info:"#1d4ed8"}
image_background: dark_facecolor_1a1a2e_for_generated_diagrams
section_template: [story_header, challenge_0, animation, core_idea_1, running_example_2, mental_model_3, step_by_step_4, key_diagrams_5, configuration_dial_6, code_skeleton_7, what_can_go_wrong_8, progress_check_N, bridge_N1]
infrastructure_style: concrete_deployment_then_abstraction
ascii_architecture_diagrams: required_for_multi_component_systems
forward_backward_links: every_concept_links_to_where_introduced_and_where_reappears
conformance_check: compare_new_chapter_against_ch01_and_ch02_before_publishing
red_lines: [no_tool_without_failure_case, no_concept_without_production_stack_grounding, no_section_without_forward_backward_context, no_deployment_pattern_without_hands_on_example, no_callout_box_without_actionable_content]
free_local_first: 100_percent_free_local_tools_required_cloud_optional
-->

---

## The Plan

The DevOps Fundamentals track is 8 chapters covering production deployment practices. Each chapter builds toward deploying a production-ready Flask API with 5 measurable constraints. We're converting each into a standalone, runnable learning module:

```
notes/06-devops_fundamentals/
├── ch01_docker_fundamentals/
│   ├── README.md          ← Technical deep-dive + diagrams
│   ├── notebook.ipynb     ← Runnable local deployment
│   └── gen_scripts/       ← Scripts to generate animations
├── ch02_container_orchestration/
│   ├── README.md
│   ├── notebook.ipynb
│   └── gen_scripts/
... (8 chapters total)
```

Each module is self-contained. Read the README to understand the concept, run the notebook to see it in action. The README and notebook teach exactly the same things in the same order.

---

## The Running Example — ProductionStack

Every chapter uses a **single consistent application**: **ProductionStack** — a 3-tier Flask web application deployment.

The scenario: *you're a DevOps engineer deploying a production-ready Flask API for a fast-growing startup.*

This one application threads naturally through all 8 chapters:

| Chapter | What we do with ProductionStack |
|---|---|
| Ch.1 — Docker Fundamentals | Containerize Flask app with Redis cache |
| Ch.2 — Container Orchestration | Add PostgreSQL database, orchestrate 3 services with Docker Compose |
| Ch.3 — Kubernetes Basics | Deploy to local K8s cluster with self-healing and scaling |
| Ch.4 — CI/CD Pipelines | Automate testing, build, and deployment with GitHub Actions |
| Ch.5 — Monitoring & Observability | Add Prometheus metrics and Grafana dashboards |
| Ch.6 — Infrastructure as Code | Define entire stack in Terraform |
| Ch.7 — Networking & Load Balancing | Add Nginx reverse proxy and load balancing |
| Ch.8 — Security & Secrets Management | Secure database passwords and API keys |

> **Why this works:** Flask is widely used, the 3-tier architecture (web + database + cache) is ubiquitous, and the entire stack runs locally with free tools. Cloud deployment is optional.

---

## The Grand Challenge — ProductionStack Production System

Every chapter threads through a unified production-system challenge: **Deploy a production-ready Flask API** satisfying 5 measurable constraints.

### The Scenario

You're the **Lead DevOps Engineer** at a fast-growing startup. The CTO wants to launch **ProductionStack** — a production-grade API deployment that can scale from 100 to 100,000 users without manual intervention.

This isn't a tutorial project. It's a **production system** that engineers, customers, and investors will rely on. It must satisfy strict operational and business requirements.

### The 5 Core Constraints

Every chapter explicitly tracks which constraints it helps solve:

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **PORTABILITY** | Same deployment works locally and cloud without changes | "Works on my machine" = unacceptable. Dev/staging/prod must be identical |
| **#2** | **AUTOMATION** | Zero-touch deployment — one command from commit to production | Manual deployment = slow, error-prone, unscalable. CI/CD is mandatory |
| **#3** | **RELIABILITY** | 99% uptime with self-healing (automatic restart on failure) | Downtime = lost revenue. System must recover without human intervention |
| **#4** | **OBSERVABILITY** | <5min mean time to detect issues via metrics and alerts | Can't fix what you can't see. Must detect problems before users complain |
| **#5** | **SECURITY** | Zero secrets in git/images, automatic secrets rotation | Leaked credentials = data breach. Security must be built-in, not bolted-on |

### Progressive Capability Unlock (8 Chapters)

| Ch | What Unlocks | Constraints Addressed | Status |
|----|--------------|----------------------|--------|
| 1 | Containerization ($Docker) | **#1 ✅ Portability** | Foundation |
| 2 | Multi-container orchestration | #1 Extended | Composition unlocked |
| 3 | Kubernetes self-healing | **#3 ✅ Reliability (basic)** | Auto-restart |
| 4 | CI/CD pipelines | **#2 ✅ Automation** | Zero-touch deploy |
| 5 | Metrics & alerting | **#4 ✅ Observability** | <5min detection |
| 6 | Infrastructure as Code | #2 Extended | Reproducible infra |
| 7 | Load balancing | #3 Extended | High availability |
| 8 | Secrets management | **#5 ✅ Security** | 🎉 **COMPLETE!** |

---

## Chapter README Template

Every chapter README follows this **extended structure**:

```
# Ch.N — [Topic Name]

> **The story.** (Historical context — who invented this, when, why)
>
> **Where you are in the curriculum.** (Links to previous chapters, what this adds)
>
> **Notation in this chapter.** (Declare all terms upfront)

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Deploy **ProductionStack** — [one-sentence mission] satisfying 5 constraints:
> 1. PORTABILITY: [target and threshold]
> 2. AUTOMATION: [zero-touch target]
> 3. RELIABILITY: [uptime target]
> 4. OBSERVABILITY: [detection time target]
> 5. SECURITY: [security requirement]

**What we know so far:**
- ✅ [Summary of previous chapters' achievements]
- ❌ **But we still can't [X]!**

**What's blocking us:**
[Concrete description of the gap this chapter addresses]

**What this chapter unlocks:**
[Specific capability that advances one or more constraints]

---

## 1 · The Core Idea (2–3 sentences, plain English)

## 2 · Running Example: What We're Deploying
(one paragraph: plug ProductionStack into this chapter's tool — Flask + PostgreSQL + Redis)

## 3 · Mental Model
(the key conceptual framework — e.g., "Image vs Container = Blueprint vs Instance")

## 4 · How It Works — Step by Step
(numbered list or flow diagram in Mermaid/ASCII)

## 5 · The Key Diagrams
(Mermaid architecture diagrams or ASCII art — minimum 1)

## 6 · The Configuration Dial
(the main tunable parameter, its effect, typical starting value)

## 7 · Code Skeleton
(minimal shell commands or YAML — illustrative, copy-paste runnable)

## 8 · What Can Go Wrong
(3–5 bullet traps, each one sentence + Fix)

## N-1 · Where This Reappears
(Forward links to later chapters that build on this tool)

## N · Progress Check — What We Can Deploy Now

![Progress visualization](img/chNN-progress-check.png) ← **Optional**: Visual dashboard showing constraint progress

✅ **Unlocked capabilities:**
- [Specific things you can now deploy]
- [Constraint achievements: "Constraint #1 ✅ Achieved! Same Dockerfile dev→prod"]

❌ **Still can't solve:**
- ❌ [What's blocked — explicitly preview next chapter's unlock]
- ❌ [Other remaining challenges]

**Real-world status**: [One-sentence summary: "We can now X, but we can't yet Y"]

**Next up:** Ch.X gives us **[tool/pattern]** — [what it unlocks]

---

## N+1 · Bridge to the Next Chapter
(one clause what this established + one clause what next chapter adds)
```

---

## Jupyter Notebook Template

Each notebook mirrors the README exactly — same sections, same order. The notebook adds:
- **Runnable cells**: every command in the README is a cell in the notebook
- **Visual outputs**: terminal outputs, service logs, container status checks
- **Exercises**: 2–3 cells at the end where the reader changes a configuration and re-runs

Cell structure per notebook:

```
[markdown] Chapter title + one-liner
[markdown] ## The Core Idea
[markdown] ## Running Example
[code]     Setup: Pull Docker images, create project structure
[markdown] ## Mental Model
[code]     Demonstrate core concept (e.g., docker run, docker-compose up)
[markdown] ## Step by Step
[code]     The step-by-step walkthrough as runnable commands
[code]     Check service health, inspect logs
[markdown] ## The Configuration Dial
[code]     Change a dial (e.g., replicas, resource limits), observe effect
[markdown] ## What Can Go Wrong
[code]     Demonstrate one of the traps + fix
[markdown] ## Exercises
[code]     Exercise scaffolds (partially filled)
```

---

## Build Tracker

| # | Chapter | Folder | Focus | Status |
|---|---------|--------|-------|--------|
| 1 | Docker Fundamentals | `ch01_docker_fundamentals/` | Containerization basics | ✅ Complete |
| 2 | Container Orchestration | `ch02_container_orchestration/` | Docker Compose multi-service | ✅ Complete |
| 3 | Kubernetes Basics | `ch03_kubernetes_basics/` | K8s local deployment | ✅ Complete |
| 4 | CI/CD Pipelines | `ch04_cicd_pipelines/` | GitHub Actions automation | ✅ Complete |
| 5 | Monitoring & Observability | `ch05_monitoring_observability/` | Prometheus + Grafana | ✅ Complete |
| 6 | Infrastructure as Code | `ch06_infrastructure_as_code/` | Terraform basics | ✅ Complete |
| 7 | Networking & Load Balancing | `ch07_networking_load_balancing/` | Nginx reverse proxy | ✅ Complete |
| 8 | Security & Secrets Management | `ch08_security_secrets_management/` | Secrets handling | ✅ Complete |

---

## Chapter Summaries (Quick Reference)

Brief bullets on what each chapter covers:

### Ch.1 — Docker Fundamentals
- Concept: Containerization — package app + dependencies into portable unit
- Tools: Dockerfile, docker build, docker run, docker-compose (intro)
- Dial: Base image choice, layer caching strategy
- Trap: Not using .dockerignore → slow builds, bloated images

### Ch.2 — Container Orchestration
- Concept: Multi-container applications with service dependencies
- Tools: Docker Compose, docker-compose.yml, networks, volumes
- Dial: Service restart policies, resource limits
- Trap: Services start before dependencies ready → add health checks

### Ch.3 — Kubernetes Basics
- Concept: Declarative container orchestration with self-healing
- Tools: kubectl, Kind (local K8s), pods, deployments, services
- Dial: Replica count, resource requests/limits
- Trap: Using pods directly instead of deployments → no self-healing

### Ch.4 — CI/CD Pipelines
- Concept: Automated testing, building, and deployment on every commit
- Tools: GitHub Actions, workflows, jobs, steps, secrets
- Dial: Workflow triggers (push, PR, schedule), job parallelization
- Trap: Building on every commit without caching → slow pipelines

### Ch.5 — Monitoring & Observability
- Concept: Metrics collection, visualization, and alerting
- Tools: Prometheus, Grafana, node_exporter, PromQL
- Dial: Scrape interval, alert thresholds, dashboard refresh rate
- Trap: Alerting on symptoms not causes → alert fatigue

### Ch.6 — Infrastructure as Code
- Concept: Define infrastructure in version-controlled files
- Tools: Terraform, HCL, state files, providers
- Dial: Resource dependencies, state backends
- Trap: Manual changes after Terraform apply → state drift

### Ch.7 — Networking & Load Balancing
- Concept: Distribute traffic across multiple service instances
- Tools: Nginx, reverse proxy, upstream blocks, health checks
- Dial: Load balancing algorithm (round-robin, least_conn, ip_hash)
- Trap: No health checks → traffic to dead instances

### Ch.8 — Security & Secrets Management
- Concept: Secure credential storage and rotation
- Tools: Environment variables, Docker secrets, Kubernetes secrets, Azure Key Vault
- Dial: Secret rotation frequency, access policies
- Trap: Hardcoded secrets in Dockerfiles → git commits leak credentials

---

## Conventions

**Diagrams:** Use Mermaid architecture diagrams (`graph TD` or `graph LR`) for multi-component systems. Use ASCII art for container/image relationships and network topology where Mermaid is overkill.

**Code style:** Shell commands (bash/PowerShell), YAML configuration files (Docker Compose, Kubernetes manifests, GitHub Actions). Keep cells short — one operation per cell. Show full command output.

**Tone:** Direct and time-efficient. Assume the reader is an engineer who wants to deploy working systems. No "Let's explore together!" — every command earns its place.

**Commands:** Use `docker`, `kubectl`, `terraform`, `docker-compose` with full flags visible. Always show output (truncated if long). Annotate every command with `# what this does`.

---

## How to Use This Document

1. Open this file to check chapter structure and conventions.
2. Pick a chapter to author or review.
3. Use the README template and notebook template above — don't invent new structures.
4. Keep the ProductionStack scenario in focus: every example should tie back to deploying the Flask API.
5. After completing a chapter, verify it matches the template.

---

## Style Ground Truth — Derived from Ch.01 Docker Fundamentals

> **LLM instruction:** Before authoring or reviewing any chapter in this track, treat Ch.01 (`notes/06-devops_fundamentals/ch01_docker_fundamentals/README.md`) as the canonical style reference. Every dimension below was extracted from that chapter. When a new chapter deviates from any dimension, flag it.

---

### Voice and Register

**The register is: technical-practitioner, second person, conversational within precision.**

The reader is treated as a capable engineer who doesn't need flattery, gets impatient with abstract theory, and wants to know what to *do* and *why it matters*. The tone is direct — every sentence earns its place.

**Second person is the default.** The reader is placed inside the scenario at all times:

> *"You're an engineer deploying a Python Flask web application, and your constraint is simple but absolute: it must run identically on your dev laptop and the production server."*  
> *"Your manager's requirement: the app must run identically on dev, staging, and production with zero manual setup."*  
> *"By step 5, you have a production-ready deployment: two containers (Flask + Redis), networked together, with persistent storage."*

---

### Story Header Pattern

Every chapter opens with three specific items, in order, in a blockquote:

1. **The story** — historical context. Who invented/popularized this tool, in what year, on what problem. Always a real person/company and a real date.

2. **Where you are in the curriculum** — one paragraph precisely describing what the previous chapter(s) gave you and what gap this chapter fills. Must name specific capabilities from preceding chapters.

3. **Notation in this chapter** — a one-line inline declaration of every term introduced in the chapter, before the first section begins.

---

### The Challenge Section (§0)

**Required pattern:**

```
> 🎯 The mission: Deploy **ProductionStack** — [one-sentence description] satisfying 5 constraints:
> 1. PORTABILITY: [target]
> 2. AUTOMATION: [target]
> 3. RELIABILITY: [target]
> 4. OBSERVABILITY: [target]
> 5. SECURITY: [target]

What we know so far:
  ✅ [Summary of previous chapters' achievements]
  ❌ **But we still can't [X]!**

What's blocking us:
  [2–4 sentences: the concrete, named gap]

What this chapter unlocks:
  [Specific capability bullet points]
```

**Numbers are always named.** The gap is never "our deployment is not portable enough" — it is "Flask works on dev laptop (Python 3.11) but fails on production (Python 3.9, missing Redis)."

---

### The Failure-First Pedagogical Pattern

**Concepts are discovered by exposing what breaks.**

Example from Docker chapter:
- Act 1: Flask works on dev laptop → show exactly where it breaks (different Python version, missing Redis)
- Act 2: Containerize app → show what *that* breaks (no data persistence)
- Act 3: Add volumes → show what *that* breaks (single container, no Redis)
- Act 4: Multi-container with networking → resolves all issues

Each step: **approach → specific failure → minimal fix → that fix's failure → next approach**

---

### Hands-On Walkthrough Pattern

**Every tool must be demonstrated with actual commands before being abstracted.**

**The canonical walkthrough structure:**
1. State the goal (e.g., "Containerize Flask app")
2. Show the Dockerfile/command with inline annotations
3. Run the command: `$ docker build -t flask-app:v1 .`
4. Show the output (truncated if long)
5. Verify it works: `$ docker run -p 5000:5000 flask-app:v1`
6. Show the verification (curl output, browser screenshot, logs)
7. State what changed: "Flask now runs in isolation with Python 3.11 guaranteed"

**Every walkthrough ends with a verification sentence** — "The container starts successfully." or "curl localhost:5000/health returns 200 OK."

---

### Forward and Backward Linking

**Every new tool is linked to where it was first used and where it will matter again.**

**Backward link pattern:** *"This is the same Dockerfile pattern from Ch.1 — the only difference is the multi-stage build."*

**Forward link pattern:** *"This Docker network concept is the foundation for Kubernetes Services in Ch.3."*

---

### Callout Box System

Used consistently. Must be used exactly this way:

| Symbol | Meaning | When to use |
|---|---|---|
| `💡` | Key insight / conceptual payoff | After a result that reframes understanding |
| `⚠️` | Warning / common trap | Before or after a pattern often done wrong |
| `⚡` | ProductionStack constraint connection | When content advances one of the 5 constraints |
| `> 📖 **Optional:**` | Deeper technical detail | Advanced configs that break narrative flow |
| `> ➡️` | Forward pointer | When a tool needs to be planted before full treatment |

---

### Image and Animation Conventions

**Every image has a purpose — none are decorative.**

**Image naming convention:**
- `ch0N-[topic]-[type].gif/.png` for chapter-specific generated images
- `[concept]_generated.gif/.png` for algorithmically generated animations

**Generated diagrams use dark background `facecolor="#1a1a2e"`** — matching the rendered dark theme.

**Image types:**

| Type | Purpose | Examples |
|---|---|---|
| GIF animation | Show a process evolving: container lifecycle, deployment flow | `ch01-container-lifecycle.gif` |
| PNG architecture | Multi-component system diagrams | `3-tier-architecture.png` |
| PNG flow | Step-by-step process visualization | `docker-build-flow.png` |
| GIF needle | Chapter-level progress (needle moving toward constraint target) | `ch01-portability-needle.gif` |

**Every chapter has a needle GIF** showing which constraint needle moved.

**Mermaid diagram colour palette:**
- Container/infrastructure: `fill:#1e3a8a` (dark blue)
- Running/healthy: `fill:#15803d` (dark green)
- Warning/pending: `fill:#b45309` (amber)
- Error/down: `fill:#b91c1c` (dark red)

---

### Code Style

**Commands are minimal but complete.**

**Variable/service naming is consistent across all chapters:**

| Name | Meaning |
|---|---|
| `flask-app` | Main Flask application container |
| `postgres-db` | PostgreSQL database container |
| `redis-cache` | Redis cache container |
| `production-network` | Docker network name |
| `data-volume` | Persistent volume name |

**Comments explain *why*, not *what*.** The command `docker run -d flask-app` doesn't need "run container in detached mode" — it needs `# -d runs in background so terminal stays free`.

---

### Progress Check Section

The Progress Check is the last substantive section before the Bridge. Fixed format:

```
✅ Unlocked capabilities:
  [bulleted list — specific deployment capabilities]
  [e.g., "Portability: Same Docker image runs on dev and prod"]

❌ Still can't solve:
  [bulleted list — named gaps]
  [e.g., "❌ Manual deployment — no automation yet"]

Progress toward constraints:
  [table: Constraint | Status | Current State]
```

---

### What Can Go Wrong Section

**Format:** 3–5 traps, each following:
- **Bold name of the trap** — one clause description
- Explanation in 2–3 sentences with concrete example
- **Fix:** one actionable command/config starting with "`Fix:`"

---

## Pedagogical Patterns & Teaching DNA

### 1. Narrative Architecture Patterns

#### Pattern A: **Failure-First Discovery Arc**

New tools emerge from concrete deployment failures:

```
Act 1: Simple approach → Show where it breaks (with exact error messages)
Act 2: First tool → Show what IT breaks (new failure mode)
Act 3: Refined solution → Resolves tension
Act 4: Decision framework (when to use which)
```

#### Pattern B: **Historical Hook → Production Stakes**

Every chapter opens with:
```markdown
> **The story:** [Name] ([Year]) created [tool] to solve [specific problem]. 
> [One sentence on impact]. [One sentence connecting to production work].
>
> **Where you are:** Ch.[N-1] achieved [capability]. This chapter fixes [named blocker].
>
> **Notation in this chapter:** [Inline term declarations]
```

---

### 2. Concept Introduction Mechanics

#### Mechanism A: **Problem→Cost→Solution Pattern**

Every new tool appears AFTER showing:
1. The problem (specific deployment failure with error message)
2. The cost (production impact: downtime, manual work, security risk)
3. The solution (tool/pattern that resolves it)

#### Mechanism B: **"It Works" Verification Loop**

After introducing any tool, immediately prove it works:
1. Command with inline annotations
2. Expected output
3. Verification command (health check, curl, logs)
4. Confirmation: "The service responds with 200 OK"

---

### 3. Scaffolding Techniques

#### Technique A: **Concrete Deployment Anchors**

Every abstract concept needs a permanent deployment reference:
- **3-tier architecture** (Flask + PostgreSQL + Redis) — mentioned throughout
- **Port mappings** (5000, 5432, 6379) — consistent across all examples
- **Same Dockerfile pattern** — evolves but maintains structure

#### Technique B: **ASCII Architecture Diagrams**

Before showing configuration files, draw the architecture:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Flask     │────▶│ PostgreSQL  │     │   Redis     │
│  :5000      │     │   :5432     │◀────│   :6379     │
└─────────────┘     └─────────────┘     └─────────────┘
      │                                        │
      └────────────────┬───────────────────────┘
                  Docker Network
```

---

### 4. Voice & Tone Engineering

#### Voice Rule A: **Practitioner Focus**

Signals "this is for engineers deploying production systems."

- Commands for deployment
- Error messages for debugging
- Configs for reproducibility

#### Voice Rule B: **No Speculation**

Never "you might want to" or "you could consider" — always "Fix: Use X because Y."

---

### 5. Engagement Hooks

#### Hook A: **Production Crises**

Frame every tool as response to deployment failure:
- Manager: "Why doesn't staging match production?"
- You: "Different Python versions..."
- Manager: "Fix it."
- **Solution:** Docker ensures identical environments

#### Hook B: **Constraint Gamification**

The 5 ProductionStack constraints act as a quest dashboard.

**Format:** Revisit this table every chapter:

| Constraint | Status | Evidence |
|------------|--------|----------|
| #1 PORTABILITY | ✅ **ACHIEVED** | Same Dockerfile dev→prod |
| #2 AUTOMATION | ❌ **BLOCKED** | Manual docker run commands |
| #3 RELIABILITY | ⚠️ **PARTIAL** | Containers restart, but no orchestration |
| #4 OBSERVABILITY | ❌ **BLOCKED** | No metrics yet |
| #5 SECURITY | ❌ **BLOCKED** | Secrets in environment variables |

---

## Red Lines — Never Violate These

1. **No tool without demonstrating what breaks first** — motivation before solution
2. **No concept without ProductionStack grounding** — every example ties to Flask API deployment
3. **No deployment pattern without hands-on command** — show exactly what to run
4. **No callout box without actionable Fix or Rule** — no "interesting facts"
5. **Free local tools first, cloud second** — 100% of core content must run locally without cost
6. **Every command must be copy-paste runnable** — no pseudo-commands
7. **No abstraction before concrete example** — Docker concepts before Kubernetes abstractions

---

## Conformance Check

Before publishing any chapter:

1. **Compare against Ch.01** — does voice match? Same callout system? Failure-first pedagogy?
2. **Verify ProductionStack thread** — does every example use Flask/PostgreSQL/Redis?
3. **Check constraint tracking** — is progress table updated?
4. **Test commands** — can reader copy-paste and run successfully?
5. **Count traps** — are there 3-5 "What Can Go Wrong" items with Fixes?
6. **Verify free local** — does everything run without cloud account or paid tool?

---

## The Free-Local-First Mandate

**Non-negotiable:** Every chapter's core content must be completable with 100% free, locally-running tools.

**Required free tools:**
- Docker Desktop (free for individuals)
- Docker Compose (included with Docker Desktop)
- Kind (Kubernetes in Docker — free, local)
- Terraform (free, local state files)
- Prometheus & Grafana (free, Docker images)
- GitHub Actions (2,000 free minutes/month)

**Optional cloud sections:**
- Labeled clearly: "Optional: Azure Deployment"
- In supplemental notebooks: `notebook_supplement.ipynb`
- Never required for core learning outcomes

---

## What These Chapters Are Not

- **Not cloud vendor tutorials** — cloud deployment is optional, not required
- **Not enterprise DevOps** — focus on fundamentals every engineer needs, not enterprise-scale concerns
- **Not theoretical** — every concept must have hands-on deployment
- **Not abstraction-first** — concrete commands before architectural patterns
