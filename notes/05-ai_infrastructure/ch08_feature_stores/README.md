# Ch.8 — Feature Stores & Data Infrastructure

> **The story.** In **2019**, **Gojek** (Southeast Asia's ride-hailing giant) and **Google** open-sourced **Feast** (Feature Store), solving a problem that had been silently killing ML models in production for years: **training-serving skew**. The issue was simple but deadly — data scientists engineered features in Python/Pandas during training (aggregations, joins, window functions), but production inference required millisecond-latency feature lookups from a live database. Engineering teams rewrote the feature logic in SQL or Java for production, inevitably introducing bugs. A `user_avg_purchase_last_30d` feature computed as `df.groupby('user_id').rolling('30D').mean()` in training would be rewritten as a PostgreSQL query in production — but with different time zone handling, null behavior, or aggregation windows. The model would silently degrade because the training features ≠ production features. Feast's insight: **compute features once, serve them twice** — one offline store for training (Parquet, BigQuery), one online store for low-latency serving (Redis, DynamoDB). The same feature definition generates both. By **2020**, **Tecton** (founded by Uber's Michelangelo team) commercialized the concept with automatic monitoring, and by **2021**, AWS SageMaker, Azure ML, and Databricks had shipped their own feature stores. The discipline was now standardized: **if you're serving ML in production, you need a feature store**.
>
> **Where you are in the curriculum.** You've just finished [Ch.7: AI-Specific Networking](../networking) where you optimized GPU-to-GPU communication for distributed inference. Now you have **fast model serving** but **slow feature lookups** — your recommendation model needs `user_last_10_clicks`, `item_avg_rating`, and `user_item_affinity_score` in <20ms to meet your p95 latency SLA, but your current PostgreSQL queries take 200ms. This chapter teaches the infrastructure that makes real-time ML inference practical: **feature stores** that precompute, version, and serve features with single-digit millisecond latency. You'll build a recommendation system using Feast (free, local) and deploy it to Azure ML Feature Store + Redis (production-grade, cloud).
>
> **Notation in this chapter.** `feature` — a column derived from raw data (e.g., `user_avg_purchase_last_30d`); `entity` — the primary key for feature lookups (e.g., `user_id`, `item_id`); `online store` — low-latency key-value database for real-time serving (Redis, DynamoDB); `offline store` — high-throughput analytical storage for training data (Parquet, BigQuery, Snowflake); `materialization` — the process of computing features from raw data and writing to the online store; `point-in-time join` — historical feature lookup that respects temporal validity (prevents data leakage); `feature view` — a logical grouping of related features with a shared entity key and update schedule.

---

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Self-host Llama-3-8B for <$15k/month, replacing $80k OpenAI API costs
> 
> **6 Constraints**: #1 Cost (<$15k/mo) • #2 Latency (≤2s) • #3 Throughput (≥10k req/day) • #4 Memory (fit in VRAM) • #5 Quality (≥95% accuracy) • #6 Reliability (>99% uptime)

**What we know so far**:
- ✅ Ch.1: RTX 4090 (24GB VRAM, $1.50/hr) → hardware locked in
- ✅ Ch.2: INT4 Llama-3-8B = 8GB params + 4GB KV cache → fits in 24GB
- ✅ Ch.3: Quantization → 60% cost reduction, <1% accuracy loss
- ✅ Ch.4: Data parallelism → training throughput 4x
- ✅ Ch.5: PagedAttention + batching → 12k req/day on 1 GPU, 1.2s p95
- ✅ Ch.6: vLLM serving framework → production-ready inference stack
- ✅ Ch.7: NVLink → multi-GPU scaling to 40k req/day

**What's blocking us**:

🚨 **The product team just added a recommendation feature to the document extraction API** — now inference needs:
1. User context features (`user_last_10_document_types`, `user_avg_confidence_score`)
2. Document features (`doc_language`, `doc_page_count`, `doc_has_tables`)
3. Derived features (`user_doc_type_affinity`, `expected_processing_time`)

**Current situation:** Features are computed on-the-fly via PostgreSQL joins:
```sql
SELECT 
    u.user_id,
    ARRAY_AGG(d.doc_type ORDER BY d.created_at DESC LIMIT 10) as last_10_doc_types,
    AVG(d.confidence_score) as avg_confidence,
    -- ... 15 more columns ...
FROM users u
JOIN documents d ON u.user_id = d.user_id
WHERE d.created_at > NOW() - INTERVAL '30 days'
GROUP BY u.user_id;
```

**Problems**:
1. ❌ **Latency explosion** — Feature query takes **380ms** (vs 50ms target) → p95 latency now **2.8s** (40% over budget)
2. ❌ **Training-serving skew** — Data scientists compute features in Pandas during training, engineering rewrites in SQL for production → silent bugs
3. ❌ **No feature versioning** — Model trained on `user_avg_confidence` definition from Jan 1, deployed to production that computes it differently on Feb 1 → accuracy drops from 96% → 89%
4. ❌ **Repeated computation** — Same `user_last_10_clicks` computed 10k times/day (once per request) instead of precomputed once and cached

**Business impact**:
- **Latency SLA violated**: 2.8s p95 (vs 2s target) → 40% of users see slow responses → churn risk
- **Cost creeping up**: PostgreSQL read replicas scaled to 4 instances ($800/month) to handle feature queries
- **Silent accuracy degradation**: Training features ≠ production features → model quality unpredictable
- **Engineering bottleneck**: Every new feature requires dual implementation (Pandas for training, SQL for serving) → 3-day turnaround per feature

**What this chapter unlocks**:

🚀 **Feature store infrastructure**:
1. **Offline store** — Parquet/BigQuery for training data (point-in-time correct historical features)
2. **Online store** — Redis/DynamoDB for serving (sub-10ms feature lookups)
3. **Feature definitions** — Single Python definition generates both training and serving features (eliminates skew)
4. **Materialization** — Precompute features on a schedule (hourly, daily) → serve from cache
5. **Versioning** — Track which feature definitions were used to train each model → reproduce training data exactly

⚡ **Expected improvements**:
- **Latency**: 380ms feature lookup → **8ms** (97% reduction) → p95 back to 1.4s ✅
- **Cost**: 4 PostgreSQL replicas ($800/mo) → 1 Redis instance ($120/mo) → **85% cost reduction** ✅
- **Feature velocity**: 3 days per feature → **30 minutes** (define once, deploy to offline + online)
- **Accuracy stability**: Eliminate training-serving skew → maintain 96% accuracy ✅

**Constraint status after this chapter**:
- #1 (Cost): ✅ **MET** — $1,095/mo GPU + $120/mo Redis = $1,215/mo (vs $15k budget)
- #2 (Latency): ✅ **MET** — 1.4s p95 (vs 2s target, 30% margin)
- #3 (Throughput): ✅ **MET** — 12k req/day (vs 10k target)
- #4 (Memory): ✅ **MET** — 12GB VRAM used (vs 24GB capacity)
- #5 (Quality): ✅ **MET** — 96% accuracy maintained (vs 95% target)
- #6 (Reliability): ⚡ **ON TRACK** — Redis adds single point of failure risk (mitigated in Ch.9-10)

---

## Animation

![Chapter animation](img/ch08-feature-store-latency.gif)

*Feature lookup latency: 380ms (direct DB) → 8ms (feature store) — 97% reduction*

---

## 1 · The Core Idea — Compute Once, Serve Twice

Feature stores solve one fundamental problem: **eliminate the training-serving gap**. The solution has three parts:

### Part 1: Single Source of Truth for Feature Definitions

**Old world (without feature store):**
```python
# Training code (data scientist's Jupyter notebook)
def compute_user_features(df):
    return df.groupby('user_id').agg({
        'purchase_amount': 'mean',
        'last_purchase': 'max'
    }).rename(columns={'purchase_amount': 'user_avg_purchase'})
```

```sql
-- Production code (engineering's SQL query)
SELECT 
    user_id,
    AVG(purchase_amount) as user_avg_purchase,  -- Bug: includes refunds
    MAX(last_purchase) as last_purchase
FROM transactions
WHERE status = 'completed'  -- Different filter than training!
GROUP BY user_id;
```

**Result:** Silent training-serving skew → model accuracy degrades in production.

---

**New world (with feature store):**
```python
# Single feature definition (used by both training and serving)
from feast import FeatureView, Field
from feast.types import Float32, Int64

user_features = FeatureView(
    name="user_purchase_features",
    entities=["user"],
    schema=[
        Field(name="avg_purchase_last_30d", dtype=Float32),
        Field(name="total_purchases", dtype=Int64),
        Field(name="days_since_last_purchase", dtype=Int64)
    ],
    source=user_transactions_source,  # Parquet file or database table
    ttl=timedelta(days=30)
)
```

**Result:** Training and serving use **identical feature values** → zero skew.

### Part 2: Two Storage Layers for Different Access Patterns

| Store | Purpose | Backing DB | Latency | Query Pattern | Data Volume |
|---|---|---|---|---|---|
| **Offline Store** | Training data | Parquet, BigQuery, Snowflake | 10s–10min | Batch retrieval of historical features (millions of rows) | TBs of historical data |
| **Online Store** | Real-time serving | Redis, DynamoDB, Cassandra | <10ms | Point lookup by entity ID (1 row) | Only latest feature values (GBs) |

**Offline store usage (training):**
```python
# Training job: fetch 1 million historical feature vectors
training_data = fs.get_historical_features(
    entity_df=entity_df,  # user_id + timestamp for each training example
    features=["user_purchase_features:avg_purchase_last_30d", 
              "item_features:avg_rating"]
).to_df()

# Returns point-in-time correct features (no data leakage!)
# For a training example at timestamp T, features are computed using only data before T
```

**Online store usage (serving):**
```python
# Inference: fetch features for one user in <10ms
features = fs.get_online_features(
    entity_rows=[{"user_id": 12345}],
    features=["user_purchase_features:avg_purchase_last_30d",
              "item_features:avg_rating"]
).to_dict()

# Returns: {'avg_purchase_last_30d': 127.50, 'avg_rating': 4.3}
# Latency: ~5ms (Redis lookup)
```

### Part 3: Materialization — Precompute Features on a Schedule

**Materialization** is the process of computing features from raw data and writing to the online store:

```bash
# Materialize features (run this hourly via cron or Airflow)
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

**What happens during materialization:**
1. Read raw data from source (e.g., `transactions.parquet`)
2. Compute aggregations (`AVG(purchase_amount)`, `COUNT(*)`, etc.)
3. Write results to online store (Redis key-value: `user:12345:avg_purchase → 127.50`)
4. Update feature metadata (last materialized timestamp, row count)

**Why this matters:**
- **Serving latency** — Features are precomputed → lookup is a Redis `GET` (5ms) instead of a SQL aggregation (380ms)
- **Cost** — Compute once per hour (cheap batch job) vs 10k times per day (expensive real-time queries)
- **Freshness control** — Materialize hourly for low-latency features, daily for slow-changing features

---

## 2 · Running Example — Recommendation System for Document Extraction

You're building a **document type recommender** for the InferenceBase API. When a user uploads a PDF, the system predicts:
- Which document type to classify it as (invoice, contract, receipt, tax form)
- Estimated processing time
- Confidence score

The model needs **3 types of features**:

### Feature Type 1: User Context Features

| Feature Name | Description | Update Frequency | Source |
|---|---|---|---|
| `user_last_10_doc_types` | Array of last 10 document types uploaded | Hourly | `user_uploads` table |
| `user_avg_confidence_score` | Average confidence score of user's past documents | Hourly | `processed_documents` table |
| `user_total_pages_processed` | Total pages processed for this user (lifetime) | Daily | `user_uploads` table |
| `user_days_since_signup` | Days since user account creation | Daily | `users` table |

### Feature Type 2: Document Features

| Feature Name | Description | Update Frequency | Source |
|---|---|---|---|
| `doc_page_count` | Number of pages in document | Real-time | Request payload |
| `doc_file_size_mb` | File size in megabytes | Real-time | Request payload |
| `doc_language` | Detected language (eng, spa, fra) | Real-time | Request payload |
| `doc_has_tables` | Boolean: contains table elements | Real-time | Request payload |

### Feature Type 3: Derived Features (Cross-Entity)

| Feature Name | Description | Update Frequency | Source |
|---|---|---|---|
| `user_doc_type_affinity` | User's historical preference for this document type (%) | Hourly | Join of user + document features |
| `expected_processing_time` | Predicted time to process (seconds) | Hourly | Regression model on historical data |

### The Feature Engineering Pipeline

```
Raw Data Sources:
  ├── user_uploads (PostgreSQL) → Extract user behavior features
  ├── processed_documents (PostgreSQL) → Extract quality metrics
  └── users (PostgreSQL) → Extract account metadata

             ↓ (ETL job runs hourly)

Feature Definitions (Feast):
  ├── user_features.py → user_last_10_doc_types, user_avg_confidence_score, ...
  ├── document_features.py → doc_page_count, doc_file_size_mb, ...
  └── derived_features.py → user_doc_type_affinity, expected_processing_time

             ↓ (Materialization runs hourly)

Storage:
  ├── Offline Store (Parquet) → Training data (historical features for 1M users)
  ├── Online Store (Redis) → Serving data (latest features for active 10k users)
  └── Feature Registry (SQLite) → Metadata (feature schemas, update timestamps)

             ↓ (Inference request)

Serving:
  GET /api/predict?user_id=12345 
    → Fetch features from Redis (5ms)
    → Pass to model (50ms)
    → Return prediction (confidence=0.94, processing_time=12s)
```

### Latency Breakdown: Before vs After Feature Store

**Before (direct PostgreSQL queries during inference):**
```
Total latency: 2,800ms p95
  ├── Feature queries (5 separate SELECT statements): 1,800ms
  │   ├── user_last_10_doc_types: 420ms
  │   ├── user_avg_confidence_score: 380ms
  │   ├── user_total_pages_processed: 320ms
  │   ├── user_doc_type_affinity: 450ms
  │   └── expected_processing_time: 230ms
  ├── Model inference (Llama-3-8B): 950ms
  └── Response serialization: 50ms
```

**After (precomputed features in Redis):**
```
Total latency: 1,400ms p95 (50% reduction!)
  ├── Feature lookup (single Redis MGET): 8ms  ← 99.6% faster
  ├── Model inference (Llama-3-8B): 950ms
  └── Response serialization: 50ms
  └── Margin for p95 variability: 392ms
```

**Cost impact:**
- **Before**: 4 PostgreSQL read replicas ($200/mo each) = $800/mo
- **After**: 1 Redis instance (16GB) = $120/mo + $50/mo Parquet storage = **$170/mo** (79% cost reduction)

---

## 3 · Mental Model — Feature Store Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE STORE ARCHITECTURE                           │
│                                                                               │
│  SERVING LAYER (Real-time inference)                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  ML Application (API Server)                                            │ │
│  │  • GET /predict?user_id=12345                                           │ │
│  │  • Fetch features from Online Store (5-10ms)                            │ │
│  │  • Pass to model (50-1000ms depending on model size)                    │ │
│  │  • Return prediction                                                     │ │
│  └───────────────────────────────┬──────────────────────────────────────┘ │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  ONLINE STORE (Low-latency serving) — <10ms lookups                    │ │
│  │                                                                          │ │
│  │  Redis / DynamoDB / Cassandra                                           │ │
│  │  • Key: entity_id (user_id, item_id)                                    │ │
│  │  • Value: feature vector (JSON/binary)                                  │ │
│  │  • TTL: 30 days (configurable per feature view)                         │ │
│  │  • Storage: Only latest values (~10k active entities × 20 features)     │ │
│  └───────────────────────────────┬──────────────────────────────────────┘ │
│                                  │                                          │
│                                  ▲ (Materialization writes here)           │
│                                  │                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  FEATURE REGISTRY (Metadata catalog)                                    │ │
│  │                                                                          │ │
│  │  SQLite / PostgreSQL / Cloud registry                                   │ │
│  │  • Feature schemas (name, dtype, entity, source)                        │ │
│  │  • Materialization history (last run timestamp, row count)              │ │
│  │  • Feature lineage (which datasets produced which features)             │ │
│  └───────────────────────────────┬──────────────────────────────────────┘ │
│                                  │                                          │
│                                  ▲ (Feature definitions registered here)   │
│                                  │                                          │
│  TRAINING LAYER (Batch historical retrieval)                                │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  ML Training Job (Jupyter / Airflow / Azure ML)                         │ │
│  │  • Fetch historical features for 1M training examples                   │ │
│  │  • Point-in-time correct joins (no data leakage)                        │ │
│  │  • Returns: Pandas DataFrame (1M rows × 50 features)                    │ │
│  │  • Latency: 10s–10min (batch query, not latency-critical)               │ │
│  └───────────────────────────────┬──────────────────────────────────────┘ │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  OFFLINE STORE (High-throughput analytical storage)                     │ │
│  │                                                                          │ │
│  │  Parquet / BigQuery / Snowflake / Redshift                              │ │
│  │  • Partitioned by date (event_timestamp column)                         │ │
│  │  • Columnar format (efficient aggregation queries)                      │ │
│  │  • Retention: 1-2 years of historical features                          │ │
│  │  • Storage: TBs of data (full history for all entities)                 │ │
│  └───────────────────────────────┬──────────────────────────────────────┘ │
│                                  │                                          │
│                                  ▲ (Materialization writes here too)       │
│                                  │                                          │
│  DATA SOURCES (Raw event streams)                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  • PostgreSQL (transactions, user_uploads, processed_documents)         │ │
│  │  • Kafka (real-time event streams)                                      │ │
│  │  • S3/Azure Blob (CSV/Parquet dumps)                                    │ │
│  │  • REST APIs (third-party data sources)                                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ORCHESTRATION (Scheduled feature computation)                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Airflow / cron / Azure Data Factory                                    │ │
│  │  • Hourly: Materialize high-freshness features (user behavior)          │ │
│  │  • Daily: Materialize low-freshness features (user demographics)        │ │
│  │  • On-demand: Materialize features for new models                       │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Key Components Explained

#### 1. Online Store (Real-Time Serving)
- **Purpose:** Sub-10ms feature lookups during inference
- **Technology:** Redis (in-memory), DynamoDB (managed), Cassandra (distributed)
- **Data structure:** Key-value pairs
  - Key: `{entity_type}:{entity_id}` (e.g., `user:12345`)
  - Value: Feature vector (e.g., `{"avg_purchase": 127.5, "total_purchases": 42}`)
- **Storage:** Only latest feature values (GBs, not TBs)
- **Update frequency:** Hourly/daily via materialization jobs

#### 2. Offline Store (Training Data)
- **Purpose:** Batch retrieval of historical features for training
- **Technology:** Parquet (local), BigQuery/Snowflake/Redshift (cloud)
- **Data structure:** Columnar tables with `(entity_id, event_timestamp, feature_values)`
- **Storage:** Full historical data (TBs)
- **Query pattern:** Point-in-time joins (fetch features as of specific timestamps)

#### 3. Feature Registry (Metadata Catalog)
- **Purpose:** Track feature schemas, lineage, and materialization status
- **Technology:** SQLite (local), PostgreSQL (cloud), cloud-native registries
- **Stores:**
  - Feature definitions (name, dtype, entity, source table)
  - Feature lineage (which raw tables → which features)
  - Materialization history (last run timestamp, row count, errors)
  - Feature versions (track schema changes over time)

#### 4. Materialization Process
- **What it does:** Compute features from raw data, write to both online and offline stores
- **Triggers:** Scheduled (cron, Airflow), on-demand (manual), event-driven (Kafka)
- **Parallelization:** Process in chunks (e.g., 10k users per worker)
- **Idempotency:** Rerunning materialization for same time window produces same results

---

## 4 · Code Skeleton — Feast Feature Store (Local)

### 4.1 · Install Feast

```bash
pip install feast[redis]  # Include Redis online store support
```

### 4.2 · Initialize Feature Repository

```bash
feast init my_feature_repo
cd my_feature_repo
```

This creates:
```
my_feature_repo/
├── feature_store.yaml  ← Config file (online/offline store settings)
├── features.py         ← Feature definitions
└── data/              ← Sample data sources
```

### 4.3 · Define Feature Views

**File: `features.py`**
```python
from feast import FeatureView, Field, Entity, FileSource
from feast.types import Float32, Int64, String, Array
from datetime import timedelta

# Define entity (primary key for feature lookups)
user = Entity(
    name="user",
    join_keys=["user_id"],
    description="User entity for document processing"
)

# Define data source (where raw data lives)
user_stats_source = FileSource(
    path="data/user_stats.parquet",  # In production: BigQuery table or S3 path
    timestamp_field="event_timestamp",
    created_timestamp_column="created_at"
)

# Define feature view (logical grouping of features)
user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="avg_confidence_score", dtype=Float32),
        Field(name="total_pages_processed", dtype=Int64),
        Field(name="days_since_signup", dtype=Int64),
        Field(name="last_10_doc_types", dtype=Array(String)),
    ],
    source=user_stats_source,
    ttl=timedelta(days=30),  # Features expire after 30 days
    online=True,  # Enable online serving
    tags={"team": "ml-infrastructure", "priority": "high"}
)
```

### 4.4 · Configure Stores

**File: `feature_store.yaml`**
```yaml
project: inferencebase_features
provider: local  # Use 'aws', 'gcp', or 'azure' for cloud

registry: data/registry.db  # SQLite registry (use PostgreSQL in production)

online_store:
  type: redis
  connection_string: "localhost:6379"  # Redis server

offline_store:
  type: file  # Local Parquet files (use BigQuery/Snowflake in production)
```

### 4.5 · Materialize Features to Online Store

```bash
# Apply feature definitions to registry
feast apply

# Materialize features for the last 7 days
feast materialize-incremental $(date -u -d '7 days ago' +"%Y-%m-%dT%H:%M:%S")
```

**What happens:**
1. Reads `data/user_stats.parquet`
2. Filters rows with `event_timestamp` in last 7 days
3. Computes aggregations defined in feature view
4. Writes results to Redis (`user:12345:avg_confidence_score → 0.94`)
5. Updates registry with materialization timestamp

### 4.6 · Fetch Online Features (Serving)

```python
from feast import FeatureStore

# Initialize feature store
fs = FeatureStore(repo_path=".")

# Fetch features for inference (single user)
entity_rows = [{"user_id": 12345}]

features = fs.get_online_features(
    features=[
        "user_features:avg_confidence_score",
        "user_features:total_pages_processed",
        "user_features:days_since_signup"
    ],
    entity_rows=entity_rows
).to_dict()

print(features)
# Output: {
#   'user_id': [12345],
#   'avg_confidence_score': [0.94],
#   'total_pages_processed': [1250],
#   'days_since_signup': [180]
# }
```

**Latency:** ~5ms (Redis lookup)

### 4.7 · Fetch Historical Features (Training)

```python
import pandas as pd
from feast import FeatureStore

fs = FeatureStore(repo_path=".")

# Training dataset: user_id + timestamp for each example
entity_df = pd.DataFrame({
    "user_id": [12345, 67890, 11111],
    "event_timestamp": [
        pd.Timestamp("2024-01-15 10:00:00", tz="UTC"),
        pd.Timestamp("2024-01-15 11:00:00", tz="UTC"),
        pd.Timestamp("2024-01-15 12:00:00", tz="UTC")
    ]
})

# Fetch point-in-time correct features
training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_features:avg_confidence_score",
        "user_features:total_pages_processed"
    ]
).to_df()

print(training_df)
#    user_id event_timestamp         avg_confidence_score  total_pages_processed
# 0    12345  2024-01-15 10:00:00              0.89                      1100
# 1    67890  2024-01-15 11:00:00              0.92                       850
# 2    11111  2024-01-15 12:00:00              0.87                      2000

# Note: Features are computed using only data BEFORE each event_timestamp (no leakage!)
```

---

## 5 · Feature Store Comparison — Feast vs Tecton vs AWS SageMaker

| Feature | **Feast** (Open Source) | **Tecton** (Commercial) | **AWS SageMaker Feature Store** |
|---|---|---|---|
| **Cost** | Free (self-hosted) | $500+/month (SaaS) | Pay-per-use (compute + storage) |
| **Deployment** | Self-managed (Docker, K8s) | Managed SaaS | Managed AWS service |
| **Online stores** | Redis, DynamoDB, Datastore | Managed (Redis-compatible) | DynamoDB (automatic) |
| **Offline stores** | Parquet, BigQuery, Snowflake, Redshift | Snowflake, Databricks, S3 | S3 (Parquet), Athena (queries) |
| **Real-time features** | Kafka + Python UDFs | Native streaming (Spark, Flink) | Kinesis integration |
| **Feature monitoring** | Manual (external tools) | Built-in (drift, quality, freshness) | CloudWatch integration |
| **Point-in-time joins** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Feature versioning** | Git-based (feature_store.yaml) | Automatic (UI + API) | Manual (feature group versions) |
| **Team collaboration** | Git + shared registry | Built-in UI + RBAC | IAM-based access control |
| **Latency** | <10ms (self-tuned Redis) | <5ms (SLA guaranteed) | ~10ms (DynamoDB, region-dependent) |
| **Best for** | Startups, cost-conscious teams | Enterprise, mission-critical ML | AWS-native stacks, low ops overhead |

### When to Choose Each:

**Feast:**
- ✅ Cost-sensitive (seed/Series A startups)
- ✅ Full control over infrastructure
- ✅ Hybrid cloud or multi-cloud
- ✅ Open-source ecosystem (Kubernetes, Airflow)
- ❌ No built-in monitoring (need to integrate Prometheus/Grafana)

**Tecton:**
- ✅ Enterprise compliance requirements (SOC2, HIPAA)
- ✅ Need SLA guarantees (<5ms p99 latency)
- ✅ Built-in feature monitoring (drift, quality, freshness)
- ✅ Large teams (10+ ML engineers)
- ❌ Expensive ($500–$5000+/month depending on usage)

**AWS SageMaker Feature Store:**
- ✅ Already on AWS (native integration with SageMaker, Lambda, Kinesis)
- ✅ Low ops overhead (fully managed)
- ✅ Pay-per-use pricing (good for spiky traffic)
- ❌ Vendor lock-in (hard to migrate off AWS)
- ❌ Less flexible than Feast (limited customization)

---

## 6 · What Can Go Wrong — Feature Store Footguns

### Footgun #1: Training-Serving Skew (Silent Accuracy Degradation)

**Scenario:** You train a model using Pandas aggregations in a notebook, then reimplement feature logic in SQL for production. The logic looks identical but has subtle bugs:

**Training (Pandas):**
```python
user_features = df.groupby('user_id').agg({
    'purchase_amount': 'mean'  # Averages ALL purchases (including $0 items)
}).rename(columns={'purchase_amount': 'avg_purchase'})
```

**Production (SQL — rewritten by engineering):**
```sql
SELECT 
    user_id,
    AVG(purchase_amount) as avg_purchase
FROM transactions
WHERE purchase_amount > 0  ← BUG: Filters out $0 items!
GROUP BY user_id;
```

**Result:** Training saw `avg_purchase = $45` (including free trials), production sees `avg_purchase = $68` (excluding free trials) → model predicts higher propensity to buy → overestimates conversion rate.

**Fix:** Use feature store's single definition:
```python
# One definition, used by both training and serving
user_features = FeatureView(
    name="user_features",
    source=transactions_source,
    schema=[Field(name="avg_purchase", dtype=Float32)],
    # Feature logic defined once in Python, compiled to SQL/Spark automatically
)
```

### Footgun #2: Feature Staleness (Cached Data Too Old)

**Scenario:** You materialize features daily at midnight, but user behavior changes rapidly (e.g., flash sale at 2pm). Model uses stale features → poor predictions.

**Example:**
- User buys 10 items during flash sale (2pm)
- Model fetches `user_total_purchases` feature at 3pm
- Feature still shows old value from midnight (before flash sale)
- Model underpredicts user's purchase intent

**Fix:** Increase materialization frequency for high-velocity features:
```python
user_features = FeatureView(
    name="user_features",
    ttl=timedelta(hours=1),  # Materialize hourly instead of daily
    online=True
)
```

**Cost-latency tradeoff:**
- Hourly materialization: Higher compute cost, fresher features
- Daily materialization: Lower cost, acceptable for slow-changing features (demographics, lifetime stats)

### Footgun #3: Online Store Latency Spikes (Redis Overload)

**Scenario:** Your Redis instance hits memory limit → evicts keys → cache miss → falls back to PostgreSQL → 200ms latency spike → p95 SLA violated.

**Symptoms:**
- p50 latency: 8ms (cache hit)
- p95 latency: 180ms (10% cache miss → DB fallback)
- p99 latency: 450ms (DB query during peak traffic)

**Root cause:** Too many features stored in Redis, or TTL too long (30 days) → memory exhausted.

**Fix 1 — Reduce TTL:**
```python
user_features = FeatureView(
    name="user_features",
    ttl=timedelta(days=7),  # Was 30 days → reduced to 7
    online=True
)
```

**Fix 2 — Scale up Redis:**
```bash
# Increase Redis memory limit
redis-server --maxmemory 16gb --maxmemory-policy allkeys-lru
```

**Fix 3 — Feature pruning:**
- Audit which features are actually used in production models
- Remove unused features from online store (keep only in offline store)

### Footgun #4: Point-in-Time Join Bugs (Data Leakage)

**Scenario:** You fetch historical features without point-in-time correctness → model sees future data during training → inflated accuracy in training, poor performance in production.

**Example:**
```python
# WRONG: Naive join (data leakage!)
entity_df = pd.DataFrame({
    "user_id": [12345],
    "label": [1],  # User purchased on 2024-01-15
    "event_timestamp": pd.Timestamp("2024-01-15 10:00:00")
})

feature_df = pd.DataFrame({
    "user_id": [12345],
    "avg_purchase": [68.0],  # Computed using data up to 2024-01-20 (includes future!)
    "event_timestamp": pd.Timestamp("2024-01-20 00:00:00")
})

# Naive merge allows model to see avg_purchase from 5 days in the future!
training_df = entity_df.merge(feature_df, on="user_id")
```

**Fix:** Use Feast's point-in-time join:
```python
# CORRECT: Point-in-time join (no leakage)
training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=["user_features:avg_purchase"]
).to_df()

# Feast ensures avg_purchase is computed using only data BEFORE 2024-01-15 10:00
```

**How Feast prevents leakage:**
- For each `(entity_id, event_timestamp)` in training data
- Fetch feature value with `event_timestamp <= training_timestamp`
- If no feature value exists before training timestamp → return NULL (forces model to handle missing data)

### Footgun #5: Feature Version Drift (Can't Reproduce Training Data)

**Scenario:** You train a model on Jan 1 using `user_avg_purchase` definition v1, then update the feature definition on Feb 1 (v2 includes refunds). On March 1, you want to retrain the model but can't reproduce the original training data.

**Problem:** Feature definitions are mutable → historical feature values can't be reconstructed.

**Fix:** Version feature definitions using Git:
```bash
# Tag feature definition when training model
git tag model-v1-features
git push origin model-v1-features

# 2 months later: checkout exact feature definitions from training
git checkout model-v1-features
feast apply  # Re-register old feature definitions
feast materialize ...  # Recompute features using old definitions
```

**Production workflow:**
1. Train model → tag Git commit with feature definitions → log commit SHA in MLflow
2. Deploy model → pin to specific feature version in production config
3. Retrain model → checkout old Git tag → reproduce exact training data

---

## 7 · Progress Check — What We've Accomplished

🎉 **Feature store infrastructure deployed** — training and serving use identical feature definitions

**Unlocked capabilities**:
- ✅ **Sub-10ms feature lookups** — 380ms PostgreSQL queries → 8ms Redis lookups (97% reduction)
- ✅ **Zero training-serving skew** — single Python definition generates both training and serving features
- ✅ **Point-in-time correct training data** — historical features fetched without data leakage
- ✅ **Feature versioning** — track which features were used to train each model
- ✅ **Cost reduction** — 4 PostgreSQL replicas ($800/mo) → 1 Redis instance ($120/mo) = **85% savings**

**Progress toward constraints**:

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 COST | ✅ **MET** | $1,215/mo total ($1,095 GPU + $120 Redis) — **92% under budget** ($15k target) |
| #2 LATENCY | ✅ **MET** | 1.4s p95 (was 2.8s) — **30% better than 2s target** |
| #3 THROUGHPUT | ✅ **MET** | 12k req/day — **120% of 10k target** |
| #4 MEMORY | ✅ **MET** | 12GB VRAM used — **50% of 24GB capacity** |
| #5 QUALITY | ✅ **MET** | 96% accuracy maintained (eliminated skew) — **1% above 95% target** |
| #6 RELIABILITY | ⚡ **ON TRACK** | 99.5% uptime (Redis adds single point of failure risk, mitigated in Ch.9-10) |

**What we can solve now**:
- ✅ Deploy new features in **30 minutes** (was 3 days) — define once, deploy to both stores
- ✅ Debug model accuracy issues by comparing training vs serving feature distributions
- ✅ A/B test feature definitions (roll out new aggregation logic to 10% of traffic, measure impact)
- ✅ Scale to 100+ features without latency degradation (Redis handles 100k ops/sec)

**What's still missing (Ch.9-10):**
- ⚠️ **Feature monitoring** — no alerts for feature drift, staleness, or distribution shifts
- ⚠️ **Disaster recovery** — Redis single point of failure (need replication + backups)
- ⚠️ **Production deployment** — no CI/CD for feature updates (manual `feast apply` + `materialize`)

---

## Bridge to Next Chapter

You now have a feature store that eliminates training-serving skew and delivers sub-10ms feature lookups. The InferenceBase platform is **hitting all 6 constraints** with room to spare:
- Cost: $1,215/mo (92% under budget)
- Latency: 1.4s p95 (30% better than target)
- Throughput: 12k req/day (120% of target)

But the system is still **operationally fragile**:
- Redis is a single point of failure (no replication)
- No monitoring for feature staleness or drift
- Feature updates require manual `feast apply` + `materialize` (no automation)

**Next up: [Ch.9 — ML Experiment Tracking & Model Registry](../ch09_ml_experiment_tracking)**. You'll add the operational discipline to track every experiment, version every model, and automate the deployment pipeline. The question: **"Can we deploy new features with zero downtime and automatic rollback?"**
