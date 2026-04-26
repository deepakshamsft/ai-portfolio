# Ch.5 — Monitoring & Observability

> **The story.** In **1998**, **Matt Welsh** at UC Berkeley published a paper on *self-tuning server systems* that tracked request latency and thread-pool saturation in real time — the seeds of modern observability. But the tool that democratised production metrics was **SoundCloud's Prometheus**, released in **2012** by **Matt T. Proud** and **Julius Volz** as an open-source replacement for proprietary APM vendors. Prometheus introduced a pull-based scraping model and a time-series database optimised for high-cardinality metrics — request rate *per route, per status code, per instance*. In 2016, **Grafana** became the de facto UI for Prometheus data, and by 2024 the Prometheus + Grafana stack is the standard for Kubernetes monitoring worldwide. Every dashboard you'll build uses the same PromQL queries and TSDB architecture they invented a decade ago.
>
> **Where you are in the curriculum.** You've deployed Flask apps with Docker (Ch.1), orchestrated multi-container stacks with Docker Compose (Ch.2), deployed to Kubernetes with auto-healing replicas (Ch.3), and automated deployments with CI/CD pipelines (Ch.4). You can ship code to production — **but you're flying blind**. If the app crashes at 2am, you find out from customer complaints. If latency spikes to 5 seconds, you only see it when users abandon checkouts. This chapter gives you **metrics + dashboards + alerts** — the observability foundation that shows you what's happening *before* it breaks.
>
> **Notation in this chapter.** `prometheus_client` — Python library for instrumenting code; **Counter** — monotonic metric (requests served); **Histogram** — bucketed distribution (latency); **Gauge** — instant snapshot (CPU %); **Prometheus** — time-series database that scrapes metrics endpoints; **Grafana** — visualization layer for queries; **PromQL** — query language (e.g., `rate(http_requests_total[5m])`); **Alertmanager** — alert routing and deduplication.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Deploy a production Flask app with full observability — monitor requests/sec, latency, error rate, and alert on failures.

**What we know so far:**
- ✅ We can containerize Flask apps (Ch.1 — Docker)
- ✅ We can orchestrate multi-container stacks (Ch.2 — Docker Compose)
- ✅ We can auto-deploy with GitHub Actions (Ch.4 — CI/CD)
- ❌ **But we have NO visibility into runtime behavior!**

**What's blocking us:**
We're deploying code into a black box:
- **No metrics**: Can't see requests/sec, latency distribution, or error rate
- **No dashboards**: No visual timeline of system health over the last hour/day/week
- **No alerts**: Discover failures from customer complaints, not proactive monitoring
- **No debugging context**: When latency spikes, we don't know *which route* or *which instance*

Without observability, you can't debug production issues, can't prove SLA compliance, and can't detect performance regressions before users notice.

**What this chapter unlocks:**
The **Prometheus + Grafana observability stack** — instrument Flask app, scrape metrics every 15 seconds, visualize time-series data, and alert when error rate crosses thresholds.
- **Establishes the three pillars**: Metrics (counters, histograms, gauges) + Logs + Traces
- **Provides concrete dashboards**: Request rate graph, latency histogram, error count
- **Teaches production debugging**: How to identify slow routes, instance failures, and traffic spikes

✅ **This is the foundation** — every later chapter assumes you can see what your system is doing.

---

## Animation

![Chapter animation](img/ch05-monitoring-observability-needle.gif)

## 1 · Observability Means You Can Ask Questions About Production Without Deploying New Code

Monitoring tells you *that* something is broken. Observability tells you *why* — and lets you explore production behavior without redeploying instrumented code. The three pillars:

1. **Metrics** — numeric time-series data (requests/sec, latency p95, error %)
2. **Logs** — structured event records (timestamp, level, message, trace ID)
3. **Traces** — distributed request paths across microservices (span durations, parent-child relationships)

This chapter focuses on **metrics** — the foundation. Logs and traces are covered in later chapters when we introduce microservices and distributed debugging.

---

## 2 · Monitoring a Flask App from Zero

You're a DevOps engineer at a fintech startup. Your Flask API just went live — it handles payment webhooks, and the CTO wants **real-time visibility** into request rate, latency, and error rate. No cloud vendor lock-in — everything must run locally first.

**The running example:**
- Flask app with 3 routes: `/health`, `/api/payment`, `/api/refund`
- Prometheus scrapes metrics every 15 seconds
- Grafana visualizes request rate and latency percentiles
- Alertmanager sends Slack notification when error rate > 5%

**Constraint:** Must run the full stack (Flask + Prometheus + Grafana + Alertmanager) with `docker compose up` — zero cloud dependencies.

---

## 3 · The Metrics Collection Stack at a Glance

Before diving into instrumentation, here's the full architecture you'll build. Each numbered component has a corresponding section below.

```
1. Flask app exposes /metrics endpoint
 └─ Instrumented with prometheus_client (counters, histograms, gauges)

2. Prometheus scrapes /metrics every 15 seconds
 └─ Stores time-series data in local TSDB
 └─ Configured via prometheus.yml (scrape targets, retention policy)

3. Grafana queries Prometheus
 └─ Visualizes metrics with time-series graphs, heatmaps, stat panels
 └─ Dashboards defined as JSON (version-controlled)

4. Alertmanager evaluates alerting rules
 └─ Fires alerts when thresholds crossed (e.g., error_rate > 5%)
 └─ Routes to Slack, PagerDuty, email

5. Generate traffic and observe dashboard
 └─ Simulate 100 requests/sec with `wrk` or Python script
 └─ Watch request rate spike, latency histogram shift
```

**Notation:**
- **Counter** — monotonic metric (only goes up). Example: `http_requests_total`
- **Histogram** — bucketed distribution. Example: `http_request_duration_seconds` with buckets `[0.1, 0.5, 1.0, 5.0]`
- **Gauge** — instant snapshot (can go up or down). Example: `active_connections`
- **PromQL** — query language. Example: `rate(http_requests_total[5m])` = requests/sec over 5min window

Sections 4–8 explain each component. Come back to this map when the detail feels overwhelming.

---

## 4 · The Math Defines Time-Series Aggregation and Rate Calculation

### 4.1 · Metrics Are Timestamped Numbers Stored in a Time-Series Database

Prometheus stores metrics as:

$$\text{metric\_name}\{\text{label1}=\text{value1}, \text{label2}=\text{value2}\} \quad t_1: v_1, \, t_2: v_2, \, t_3: v_3, \ldots$$

Example:
```
http_requests_total{method="GET", route="/api/payment", status="200"}  1609459200: 150
http_requests_total{method="GET", route="/api/payment", status="200"}  1609459215: 162
http_requests_total{method="GET", route="/api/payment", status="200"}  1609459230: 178
```

Each tuple `(metric_name, labels)` defines a unique **time series**. Every scrape appends a new `(timestamp, value)` pair.

**What does a label actually mean?** Labels are the query dimensions — they let you slice metrics by route, status code, instance, or any custom tag. The metric name says *what* you're measuring; the labels say *which instance of the thing*.

**Why counters only go up.** A counter like `http_requests_total` is monotonically increasing — it resets to 0 only when the process restarts. To compute *rate* (requests/sec), you take the difference over a time window:

$$\text{rate}(C[t_1 : t_2]) = \frac{C(t_2) - C(t_1)}{t_2 - t_1}$$

In PromQL:
```promql
rate(http_requests_total[5m])
```

This computes the per-second rate over the last 5 minutes. If the counter went from 1000 to 1300 in 300 seconds, the rate is $(1300 - 1000) / 300 = 1.0$ requests/sec.

### 4.2 · Histograms Use Bucketing to Estimate Percentiles

A histogram tracks how many observations fall into predefined buckets. For latency:

```
http_request_duration_seconds_bucket{route="/api/payment", le="0.1"}   150
http_request_duration_seconds_bucket{route="/api/payment", le="0.5"}   280
http_request_duration_seconds_bucket{route="/api/payment", le="1.0"}   295
http_request_duration_seconds_bucket{route="/api/payment", le="+Inf"}  300
http_request_duration_seconds_sum{route="/api/payment"}                85.2
http_request_duration_seconds_count{route="/api/payment"}              300
```

The `le` label means "less than or equal". This histogram says:
- 150 requests took ≤0.1s
- 280 requests took ≤0.5s
- 295 requests took ≤1.0s
- All 300 requests took <∞ (the `+Inf` bucket always equals the count)

**To compute the 95th percentile** (p95), find the bucket where 95% of samples land:

$$p_{95} \approx \text{bucket boundary where cumulative fraction} \geq 0.95$$

In PromQL:
```promql
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

This interpolates linearly between bucket boundaries. If 280 out of 300 samples (93%) are in the ≤0.5s bucket, and 295 (98%) are in the ≤1.0s bucket, then p95 falls between 0.5s and 1.0s.

> ⚠️ **Histogram precision depends on bucket boundaries.** If you define buckets `[0.1, 1.0, 10.0]`, you can't distinguish between a 0.2s request and a 0.9s request — both land in the same bucket. Choose buckets that match your SLA thresholds (e.g., if your SLA is <500ms, include a 0.5s bucket).

#### Numeric Verification — Histogram Percentile

Toy data: 10 requests with latencies `[0.05, 0.08, 0.12, 0.18, 0.22, 0.35, 0.48, 0.52, 0.78, 1.20]` seconds.

Buckets: `[0.1, 0.5, 1.0, +Inf]`

| Bucket | Count | Cumulative | Fraction |
|--------|-------|------------|----------|
| ≤0.1s  | 2     | 2          | 0.20     |
| ≤0.5s  | 6     | 8          | 0.80     |
| ≤1.0s  | 1     | 9          | 0.90     |
| ≤+Inf  | 1     | 10         | 1.00     |

**p50** (median): 50% of samples = 5 requests. Falls in the ≤0.5s bucket. Since 2 samples are in ≤0.1s and 8 are in ≤0.5s, p50 is interpolated as:

$$p_{50} = 0.1 + \frac{0.5 - 0.2}{0.8 - 0.2} \times (0.5 - 0.1) = 0.1 + 0.5 \times 0.4 = 0.3 \text{s}$$

**p95**: 95% of samples = 9.5 requests. Falls in the ≤1.0s bucket. Interpolate:

$$p_{95} = 0.5 + \frac{0.95 - 0.8}{0.9 - 0.8} \times (1.0 - 0.5) = 0.5 + 1.5 \times 0.5 = 1.0 \text{s}$$

Verify: The true sorted latencies show p95 at the 9.5th value (between 0.78s and 1.20s) ≈ 0.99s. The histogram approximation (1.0s) is within bucket precision.

---

## 5 · Mental Model — Metrics vs Logs vs Traces

| Aspect | Metrics | Logs | Traces |
|--------|---------|------|--------|
| **Format** | Numeric time series | Text events | Nested spans |
| **Storage** | TSDB (Prometheus) | Elasticsearch, Loki | Jaeger, Tempo |
| **Query** | PromQL aggregations | Grep, regex | Trace ID lookup |
| **Use case** | Dashboard, alerts | Debugging, auditing | Distributed request flow |
| **Example** | `http_requests_total{status="500"}` | `ERROR: Database timeout after 5s` | Trace shows API → DB → Cache (2s total) |
| **Cardinality** | Bounded (labels × values) | Unbounded (one log line per event) | Medium (one trace per request) |

**When to use each:**
- **Metrics** — continuous monitoring (CPU, memory, request rate, latency percentiles)
- **Logs** — one-off debugging ("what was the error message for request ID 12345?")
- **Traces** — distributed debugging ("which microservice is slow in the checkout flow?")

This chapter focuses on **metrics** because they're the foundation for dashboards and alerts. Logs and traces require structured logging libraries and distributed tracing instrumentation — covered in later chapters when we introduce microservices.

---

## 6 · Code Skeleton — Flask App Instrumented with Prometheus Client

```python
from flask import Flask
from prometheus_client import Counter, Histogram, Gauge, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import time

app = Flask(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'route', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['route'],
    buckets=[0.1, 0.5, 1.0, 5.0]  # SLA thresholds
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Current active connections'
)

# Middleware to track latency
@app.before_request
def before_request():
    request.start_time = time.time()
    ACTIVE_CONNECTIONS.inc()

@app.after_request
def after_request(response):
    latency = time.time() - request.start_time
    REQUEST_LATENCY.labels(route=request.path).observe(latency)
    REQUEST_COUNT.labels(
        method=request.method,
        route=request.path,
        status=response.status_code
    ).inc()
    ACTIVE_CONNECTIONS.dec()
    return response

# Routes
@app.route('/health')
def health():
    return {'status': 'ok'}

@app.route('/api/payment', methods=['POST'])
def payment():
    time.sleep(0.2)  # Simulate processing
    return {'status': 'success'}

# Expose metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**What's happening here?**
1. `Counter.inc()` increments `http_requests_total` by 1 for every request
2. `Histogram.observe(latency)` adds the request duration to the latency distribution
3. `Gauge.inc()` / `Gauge.dec()` tracks the current number of active connections
4. `/metrics` endpoint exposes all metrics in Prometheus text format

**Prometheus scrape config:**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s  # Scrape every 15 seconds

scrape_configs:
  - job_name: 'flask-app'
    static_configs:
      - targets: ['flask-app:5000']
```

**Docker Compose stack:**

```yaml
# docker-compose.yml
services:
  flask-app:
    build: .
    ports:
      - "5000:5000"
    networks:
      - monitoring

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge
```

Run the stack:
```bash
docker compose up -d
```

Access:
- Flask app: `http://localhost:5000`
- Prometheus UI: `http://localhost:9090`
- Grafana: `http://localhost:3000` (user: `admin`, password: `admin`)

---

## 7 · What Can Go Wrong — Three Production Observability Failures

### 7.1 · Cardinality Explosion Kills Prometheus Performance

**What breaks:** You instrument a metric with a high-cardinality label — e.g., `user_id` or `request_id` — and Prometheus runs out of memory.

**Why it happens:** Prometheus stores one time series per unique `(metric_name, labels)` tuple. If you have 1 million users, `http_requests_total{user_id="..."}` creates 1 million time series. Each time series has overhead — timestamps, samples, indexes.

**Rule of thumb:** Keep cardinality below **10,000 unique label values per metric**. If you need higher cardinality, use logs (unbounded) or traces (sampled).

**Bad:**
```python
REQUEST_COUNT.labels(user_id=user_id).inc()  # 1M users = 1M time series
```

**Good:**
```python
REQUEST_COUNT.labels(route=request.path, status=status_code).inc()  # ~100 routes × 10 status codes = 1K time series
```

**How to detect:** Prometheus UI → Status → TSDB Status. If you see "time series count" growing unbounded, you have a cardinality problem.

**How to fix:** Remove high-cardinality labels. If you must track per-user metrics, use **aggregation** — e.g., count requests per user in a separate system (logs or a database), not Prometheus.

---

### 7.2 · PromQL Queries Time Out When Aggregating Large Time Ranges

**What breaks:** You run a query like `sum(rate(http_requests_total[30d]))` and Grafana hangs for 30 seconds before timing out.

**Why it happens:** Prometheus must scan 30 days of raw samples (2 samples/min × 43,200 minutes = 86,400 samples per time series). If you have 1,000 time series, that's 86 million samples to load and aggregate.

**Rule of thumb:** Keep query time ranges below **6 hours** for dashboards, **1 day** for ad-hoc queries. Use **recording rules** to precompute expensive aggregations.

**Slow:**
```promql
sum(rate(http_requests_total[30d]))  # Scans 30 days of raw data
```

**Fast:**
```promql
sum(rate(http_requests_total[5m]))  # Scans 5 minutes of raw data
```

**Recording rule** (precompute hourly):
```yaml
# prometheus.yml
groups:
  - name: aggregations
    interval: 60s
    rules:
      - record: http_requests:rate5m
        expr: sum(rate(http_requests_total[5m]))
```

Now query `http_requests:rate5m` instead — it's precomputed every 60 seconds.

---

### 7.3 · Retention Limits Erase Historical Data Before You Notice Trends

**What breaks:** You deploy a change on Monday, and by Friday you notice latency slowly creeping up. But Prometheus only retains 7 days of data — you can't compare to last month's baseline.

**Why it happens:** Prometheus defaults to **15 days retention**. After that, old samples are deleted to free disk space.

**Rule of thumb:** Set retention to **at least 30 days** for production systems. If you need longer, use **remote storage** (e.g., Thanos, Cortex, or cloud-managed Prometheus).

**Configure retention:**
```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'  # Keep 30 days
      - '--storage.tsdb.retention.size=50GB'  # Or 50GB disk limit
```

**Remote storage:** For multi-year retention, send metrics to a remote backend:
- **Thanos** (open-source, S3-backed)
- **Cortex** (open-source, horizontally scalable)
- **AWS Managed Prometheus** (cloud, $$$)

---

## 8 · Progress Check — Three Scenarios to Test Your Understanding

### Scenario 1 — Request Rate Spike

You see this graph in Grafana:

```
Request rate (requests/sec)
   ▲
100│          ╱╲
 50│    ╱╲  ╱  ╲
  0│___/  \/    \___
   └────────────────▶ Time
      10am   11am   12pm
```

**Questions:**
1. What PromQL query generates this graph?
2. What might cause the spike at 11am?
3. How would you set an alert to fire when rate > 80 req/sec?

**Answers:**
1. `rate(http_requests_total[5m])` or `sum(rate(http_requests_total[5m])) by (route)`
2. Possible causes: traffic spike (marketing campaign, viral post), DDoS attack, retry storm from a failing client
3. Alerting rule:
   ```yaml
   groups:
     - name: alerts
       rules:
         - alert: HighRequestRate
           expr: sum(rate(http_requests_total[5m])) > 80
           for: 2m  # Fire only if sustained for 2 minutes
           annotations:
             summary: "Request rate exceeded 80 req/sec"
   ```

---

### Scenario 2 — Latency Histogram Shift

Your `/api/payment` route usually has p95 latency of 200ms. After deploying a new version, you see:

```
Latency p95 (seconds)
   ▲
1.0│          ████████
0.5│    ████  ████████
0.2│████████  ████████
   └────────────────────▶ Time
      v1.0      v1.1
```

**Questions:**
1. What PromQL query computes p95 latency?
2. What changed between v1.0 and v1.1?
3. How would you drill down to find the slow query?

**Answers:**
1. `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{route="/api/payment"}[5m]))`
2. p95 latency increased from 200ms to 1.0s — a 5x regression. Likely causes: slow database query, external API timeout, missing cache hit
3. Add more granular labels:
   ```python
   REQUEST_LATENCY.labels(route=request.path, handler='db_query').observe(latency)
   ```
   Then query `histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{handler="db_query"}[5m]))` to isolate the slow component.

---

### Scenario 3 — Error Rate Spike

Your dashboard shows:

```
Error rate (errors/sec)
   ▲
10 │          ██
 5 │          ██
 0 │__________██__________
   └────────────────────▶ Time
            3pm
```

**Questions:**
1. What PromQL query computes error rate?
2. What labels help identify the root cause?
3. How would you correlate this with logs?

**Answers:**
1. `sum(rate(http_requests_total{status=~"5.."}[5m]))` (matches status codes 500–599)
2. Break down by route and status:
   ```promql
   sum(rate(http_requests_total{status=~"5.."}[5m])) by (route, status)
   ```
   If `/api/payment` shows 100% 503 errors, the payment service is down.
3. **Correlate with logs:** Use a trace ID or request ID label:
   ```python
   REQUEST_COUNT.labels(route=request.path, status=status_code, trace_id=trace_id).inc()
   ```
   Then grep logs for `trace_id=abc123` to see the full error stack trace.

---

## 9 · Bridge to Ch.6 — Infrastructure as Code Automates This Entire Stack

You just deployed Prometheus + Grafana manually with `docker-compose.yml`. **But production systems need reproducibility** — if you tear down the stack and rebuild it, the Grafana dashboards, Prometheus scrape configs, and alerting rules must come back exactly the same.

**Next chapter preview:** **Infrastructure as Code (Terraform)** lets you define the entire monitoring stack as version-controlled `.tf` files:
- Provision Prometheus container with `terraform apply`
- Import Grafana dashboards as JSON (no manual clicking)
- Deploy alerting rules as code (reviewable, rollback-able)

**The bridge:** This chapter taught you *what* to monitor and *how* to visualize it. The next chapter teaches you *how to deploy it reproducibly* — so your observability stack is as reliable as the application it monitors.

**Forward pointer:** When we introduce microservices in later chapters, you'll need **distributed tracing** (Jaeger, OpenTelemetry) to follow a request across 10 services. But every trace system sends *aggregated metrics* to Prometheus — so the foundation you built here scales directly.

---

## Key Diagrams

![Prometheus Architecture](img/prometheus_architecture.png)  
*Figure 1: Prometheus scrapes `/metrics` endpoints from targets, stores time-series data in a local TSDB, and exposes PromQL for queries.*

![Metrics Types](img/metrics_types.png)  
*Figure 2: Counter (monotonic), Gauge (instant snapshot), Histogram (bucketed distribution), Summary (client-side percentiles).*

![Grafana Dashboard](img/grafana_dashboard.png)  
*Figure 3: Grafana visualizes Prometheus queries with time-series graphs, stat panels, and heatmaps.*

---

## Further Reading

- [Prometheus documentation](https://prometheus.io/docs/) — official PromQL reference, best practices, storage tuning
- [Grafana documentation](https://grafana.com/docs/) — dashboard design, templating, alerting
- [SRE Book — Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) — Google's four golden signals (latency, traffic, errors, saturation)
- [Brendan Gregg — USE Method](https://www.brendangregg.com/usemethod.html) — Utilization, Saturation, Errors framework for system monitoring
- [Cindy Sridharan — Monitoring in the time of Cloud Native](https://medium.com/@copyconstruct/monitoring-in-the-time-of-cloud-native-c87c7a5bfa3e) — observability vs monitoring, high-cardinality dimensions, distributed tracing

---

## Exercises

1. **Instrument a new route** — Add a `/api/refund` endpoint to the Flask app and verify that `http_requests_total{route="/api/refund"}` appears in Prometheus after hitting the route.

2. **Create a custom Gauge** — Add a `database_connection_pool_size` gauge that tracks the current number of active database connections. Simulate pool exhaustion by setting the gauge to 0 and verify the alert fires.

3. **Export a Grafana dashboard** — Create a dashboard with 3 panels (request rate, latency p95, error rate), export as JSON, and commit it to version control. Tear down the stack, restart, and import the JSON — the dashboard should reappear identically.

4. **Simulate cardinality explosion** — Modify the Flask app to label `http_requests_total` with `user_id={random_uuid()}`. Generate 10,000 requests and watch Prometheus memory usage spike. Then remove the label and observe the memory stabilize.

5. **Write a recording rule** — Precompute the hourly request rate with a recording rule `http_requests:rate1h`. Query it in Grafana and verify it updates every 60 seconds.
