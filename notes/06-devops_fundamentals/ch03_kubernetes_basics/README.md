# Ch.3 — Kubernetes Basics

> **The story.** In **2014** Google open-sourced **Kubernetes** (Greek for "helmsman"), the culmination of 15 years of production experience running containers at scale via internal systems like **Borg** and **Omega**. The project launched with seven co-founders including Joe Beda, Brendan Burns, and Craig McLuckie, and within a year became the fastest-growing open-source project in history. In 2015 it joined the Cloud Native Computing Foundation; by 2018 it had become the de facto standard for container orchestration after defeating rivals like Docker Swarm and Mesos. Today every major cloud provider offers managed Kubernetes (GKE, EKS, AKS), and it powers millions of production deployments worldwide — from microservices to machine learning platforms.
>
> **Where you are in the curriculum.** You've containerized applications with Docker (Ch.1) and orchestrated them locally with Docker Compose (Ch.2). But Docker Compose doesn't scale across machines or handle failures — it's a single-host tool. This chapter introduces **Kubernetes**, the distributed orchestrator that manages containers across *clusters* of machines with self-healing, declarative configuration, and automatic scaling. You'll run everything locally using **Kind** (Kubernetes in Docker), learning K8s without any cloud spend.
>
> **Notation in this chapter.** Pod — smallest deployable unit (1+ containers); ReplicaSet — maintains N identical pods; Deployment — manages ReplicaSets with rolling updates; Service — stable network endpoint for pods; ConfigMap — configuration data; Secret — sensitive data (passwords, tokens); kubectl — CLI for Kubernetes API.

---

## 0 · The Challenge — Where We Are

> 💡 **The mission**: Deploy **SmartVal API** — a production Flask service satisfying 5 constraints:
> 1. **HIGH AVAILABILITY**: Auto-restart on crashes
> 2. **HORIZONTAL SCALING**: 3+ replicas for load distribution
> 3. **ZERO-DOWNTIME UPDATES**: Rolling deployments
> 4. **SERVICE DISCOVERY**: Stable DNS name for clients
> 5. **LOCAL DEVELOPMENT**: Run full K8s cluster on laptop (no cloud spend)

**What we know so far:**
- ✅ We can containerize apps with Docker (Ch.1)
- ✅ We can orchestrate multi-container apps with Docker Compose (Ch.2)
- ❌ **But Docker Compose is single-host — no multi-machine scaling or self-healing!**

**What's blocking us:**
Production environments need:
- **Multiple machines** — can't put all replicas on one server
- **Self-healing** — crashed containers must restart automatically
- **Load balancing** — traffic distributed across healthy replicas
- **Rolling updates** — deploy new versions without downtime

Docker Compose can't do this — it's designed for single-host development, not distributed production.

**What this chapter unlocks:**
**Kubernetes orchestration** — declarative cluster management that:
- Runs on 1+ machines (from laptop to 1,000-node clusters)
- Automatically restarts failed pods
- Distributes traffic across healthy replicas
- Performs rolling updates with rollback capability

✅ **This is production-ready orchestration** — the foundation for microservices, ML platforms, and cloud-native apps.

---

## 1 · Kubernetes Is Declarative Orchestration with Self-Healing

Kubernetes (often abbreviated **K8s**) is a container orchestration platform that manages applications across clusters of machines. Instead of imperatively running containers (`docker run ...`), you declare the *desired state* in YAML files (e.g., "I want 3 replicas of this Flask app") and Kubernetes continuously reconciles reality to match. If a pod crashes, K8s immediately starts a replacement. If a node fails, K8s reschedules its pods elsewhere. This **declarative self-healing** approach is what makes K8s production-ready.

Key differences from Docker Compose:

| Feature | Docker Compose | Kubernetes |
|---------|----------------|------------|
| **Scope** | Single host | Multi-node cluster |
| **Self-healing** | No (manual restart) | Yes (automatic) |
| **Scaling** | Manual | Declarative + autoscaling |
| **Load balancing** | Basic | Built-in (Services) |
| **Rolling updates** | No | Yes (zero-downtime) |
| **Production use** | Development only | Designed for production |

**Why learn K8s?** It's the industry standard. If you deploy to AWS, Azure, GCP, or any modern cloud, you're likely using Kubernetes (EKS, AKS, GKE). Even on-premises data centers run K8s. Understanding it unlocks:
- **Microservices** — independent services communicating over a network
- **ML platforms** — training pipelines, model serving, experiment tracking
- **CI/CD** — automated deployments to production clusters

---

## 2 · Running Example: Deploy Flask API with 3 Replicas

You're deploying **SmartVal API** — a Flask app that predicts house values. Instead of running it on a single container (fragile), you'll deploy 3 replicas behind a load-balanced service. If one pod crashes, K8s auto-restarts it. If traffic spikes, you can scale to 10 replicas declaratively.

**The 4-step workflow:**
1. **Create a local Kubernetes cluster** (Kind = Kubernetes in Docker)
2. **Write deployment YAML** — defines desired state (3 replicas of Flask container)
3. **Create a Service** — stable endpoint that load-balances across pods
4. **Simulate failures** — kill a pod, watch K8s resurrect it instantly

By the end, you'll have a self-healing, load-balanced API running on your laptop — the same patterns used in production clusters with 1,000+ nodes.

---

## 3 · The Mental Model: Pods → ReplicaSets → Deployments → Services

Kubernetes has several layers of abstraction. Here's the hierarchy from bottom to top:

### Pod (Atomic Unit)
- **Smallest deployable unit** — one or more containers sharing network/storage
- Most pods are single-container (1 pod = 1 container)
- Ephemeral — can be killed/replaced at any time
- Has a unique IP address (within the cluster)

Example: A single Flask container running inside a pod.

### ReplicaSet (Maintains N Copies)
- **Ensures N identical pods are always running**
- If a pod dies, ReplicaSet immediately creates a replacement
- Rarely created directly — you use Deployments instead

Example: A ReplicaSet maintaining 3 Flask pods.

### Deployment (Declarative Updates)
- **Manages ReplicaSets** — handles rolling updates, rollbacks
- When you update the image version, Deployment creates a new ReplicaSet and gradually shifts traffic
- This is what you'll use 90% of the time

Example: A Deployment that runs 3 Flask pods (v1.0), then performs a rolling update to v1.1.

### Service (Stable Network Endpoint)
- **Load balancer** — distributes traffic across healthy pods
- Provides a stable DNS name (pods can restart with new IPs, but the Service name stays constant)
- Types: **ClusterIP** (internal), **NodePort** (external via node IP), **LoadBalancer** (cloud provider integration)

Example: A Service named `smartval-api` that routes `http://smartval-api:5000` to any of the 3 Flask pods.

**The flow:**
```
Client → Service (smartval-api:5000)
           ↓
   [Load balances across 3 pods]
           ↓
   Pod 1 (Flask)   Pod 2 (Flask)   Pod 3 (Flask)
```

If Pod 2 crashes, the Deployment's ReplicaSet immediately spawns a new Pod 2, and the Service automatically routes traffic to the replacement.

---

## 4 · Code Skeleton — Deployment and Service YAML

Here are the minimal YAML files you'll use to deploy the Flask API.

### Deployment YAML (`deployment.yaml`)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: smartval-api
spec:
  replicas: 3  # Number of pods
  selector:
    matchLabels:
      app: smartval-api
  template:
    metadata:
      labels:
        app: smartval-api
    spec:
      containers:
      - name: flask
        image: your-dockerhub-username/smartval-api:v1
        ports:
        - containerPort: 5000
        env:
        - name: MODEL_PATH
          value: "/models/model.pkl"
```

**Key fields:**
- `replicas: 3` — K8s ensures 3 pods are always running
- `image: ...` — the Docker image to run (must be pushed to a registry)
- `containerPort: 5000` — the port the Flask app listens on inside the container

### Service YAML (`service.yaml`)
```yaml
apiVersion: v1
kind: Service
metadata:
  name: smartval-api
spec:
  type: NodePort  # Expose on node IP:port (for local access)
  selector:
    app: smartval-api  # Route to pods with this label
  ports:
  - port: 5000        # Service port
    targetPort: 5000  # Pod port
    nodePort: 30000   # External port (30000-32767)
```

**Key fields:**
- `type: NodePort` — makes the service accessible from outside the cluster
- `selector: app=smartval-api` — routes traffic to pods with this label
- `nodePort: 30000` — you can access the service at `localhost:30000` (Kind maps this to your host)

**Apply to cluster:**
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

Kubernetes reads these files and makes it happen — creates pods, starts containers, sets up networking. You don't run commands like `docker run` manually.

> 📖 **Optional: Advanced kubectl debugging**
> 
> For production troubleshooting, see [Kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
> covering resource limits, node management, label selectors, and log aggregation. This chapter focuses on
> the local development essentials — the 5 commands you need to deploy and debug on your laptop.

---

## 5 · What Can Go Wrong

Kubernetes debugging follows a simple pattern: **check status → inspect details → read logs**. Here are the 3 most common traps:

### 1. ImagePullBackOff — Can't Pull Docker Image
**Symptom:** Pods stuck in `ImagePullBackOff` status.

**Cause:** K8s can't pull the Docker image (wrong name, private registry without credentials, or image doesn't exist).

**Fix:**
```bash
kubectl describe pod <pod-name>  # Check "Events" section for error message
# Common issues:
# - Image name typo (e.g., `smartval-api:v1` instead of `your-username/smartval-api:v1`)
# - Image not pushed to Docker Hub
# - Private registry without imagePullSecrets
```

### 2. CrashLoopBackOff — Container Keeps Restarting
**Symptom:** Pods restart repeatedly.

**Cause:** Container starts but immediately exits (e.g., Python import error, missing environment variable, port already in use).

**Fix:**
```bash
kubectl logs <pod-name>  # Check application logs for crash reason
kubectl describe pod <pod-name>  # Check restart count and exit code

# Common issues:
# - Missing dependencies in Docker image
# - Incorrect command/entrypoint
# - Application crashes on startup (e.g., `FileNotFoundError`)
```

### 3. Service Not Accessible — Can't Reach Pods
**Symptom:** Service DNS name resolves but requests time out.

**Cause:** Pods aren't labeled correctly, or the Service selector doesn't match pod labels.

**Fix:**
```bash
kubectl describe service <service-name>  # Check "Endpoints" (should list pod IPs)
kubectl get pods  # Verify pods are in "Running" state
kubectl describe pod <pod-name>  # Check pod labels match Service selector

# If Endpoints is empty:
# - Service selector doesn't match pod labels
# - Pods aren't in "Running" state
```

**The 3-step debugging workflow:**
1. `kubectl get pods` — check pod status (Running, Pending, CrashLoopBackOff, ImagePullBackOff)
2. `kubectl describe pod <name>` — detailed events, state, and labels
3. `kubectl logs <pod-name>` — application logs (stdout/stderr)

**When you need to dig deeper:**
```bash
kubectl exec -it <pod-name> -- sh  # SSH into running pod to inspect filesystem/network
```

---

## 6 · Progress Check — Debug a CrashLoopBackOff Pod

**Scenario:** You deploy a Flask app to your Kind cluster. Pods start but immediately crash and restart repeatedly.

**Your mission:** Use the 3-step debugging workflow to diagnose the issue.

**Step 1:** Check pod status
```bash
kubectl get pods
# Output shows: smartval-api-abc123  0/1  CrashLoopBackOff  5  2m
```

**Step 2:** Inspect pod details
```bash
kubectl describe pod smartval-api-abc123
# Look for: Exit Code (e.g., 137 = killed, 1 = error), Restart Count, Recent Events
```

**Step 3:** Read application logs
```bash
kubectl logs smartval-api-abc123
# Common crash causes:
# - ImportError: No module named 'flask'
# - FileNotFoundError: [Errno 2] No such file or directory: '/models/model.pkl'
# - Port 5000 already in use
```

**Challenge:** What if the pod crashes so fast you can't read the logs before it restarts?

Answer:
```bash
kubectl logs smartval-api-abc123 --previous  # Read logs from the last crashed container
```

---

## 7 · Bridge to Ch.4 — CI/CD Automates Deployments to K8s

You've manually deployed a Flask app to Kubernetes using `kubectl apply`. But in production, you don't run commands manually — every push to the main branch triggers a **CI/CD pipeline** that:
1. Runs tests
2. Builds a Docker image
3. Pushes it to a registry
4. Updates the Kubernetes deployment (new image tag)
5. Monitors the rollout

**Next chapter (Ch.4: CI/CD Pipelines)** covers GitHub Actions — you'll automate the entire deployment workflow so that a commit to `main` automatically deploys to your K8s cluster.

**Teaser question:** If a CI/CD pipeline updates a Deployment's image tag from `v1.0` to `v1.1`, how does Kubernetes perform the rollout without downtime? (Answer: Deployments create a new ReplicaSet for v1.1, gradually scale it up while scaling down the v1.0 ReplicaSet, ensuring some pods are always running.)

---

## What You've Learned

✅ **Kubernetes orchestrates containers across clusters** — not just single-host like Docker Compose  
✅ **Declarative configuration** — you specify desired state, K8s makes it happen  
✅ **Self-healing** — crashed pods restart automatically  
✅ **Deployments manage ReplicaSets** — rolling updates with zero downtime  
✅ **Services provide stable endpoints** — load-balance across pod replicas  
✅ **Kind runs K8s locally** — learn without cloud costs  
✅ **Debugging workflow** — `kubectl get/describe/logs` for troubleshooting

---

## Further Reading

- [Kubernetes Official Docs](https://kubernetes.io/docs/) — comprehensive reference
- [Kubernetes the Hard Way](https://github.com/kelseyhightower/kubernetes-the-hard-way) — manual setup from scratch
- [Kind Documentation](https://kind.sigs.k8s.io/) — local K8s clusters for testing
- [Kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Production-Grade Kubernetes (Book)](https://www.oreilly.com/library/view/production-kubernetes/9781492092292/)
