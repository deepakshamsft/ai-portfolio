# Ch.4 — Event-Driven Agent Messaging

> **The story.** Async, durable messaging is older than ML — IBM's MQSeries shipped in 1993, **Apache Kafka** came out of LinkedIn in 2011, **Redpanda** and **NATS JetStream** modernised the Kafka recipe a decade later. The patterns themselves — dead-letter queues, idempotency keys, correlation IDs, fan-out/fan-in — came from Gregor Hohpe & Bobby Woolf's *Enterprise Integration Patterns* (2003), the same book every microservices architect kept on their desk in the 2010s. The multi-agent twist arrived in 2023–25: when an orchestrator agent has to coordinate dozens of long-running sub-agents (each potentially making LLM calls that take seconds), synchronous request-response collapses. The fix is exactly what the EIP book wrote down 20 years earlier — just with LLM tasks on the bus instead of stock trades. AWS's **Step Functions**, Azure's **Durable Functions**, **Temporal**, and **Inngest** are all cloud-native expressions of this pattern, increasingly billed as agent orchestrators.
>
> **Where you are in the curriculum.** [Ch.1](../ch01_message_formats)–[Ch.3](../ch03_a2a) used synchronous protocols. This chapter answers: **when does synchronous request-response break down in a multi-agent system, and how do you rebuild the coordination layer on top of async pub/sub messaging to handle thousands of concurrent agent tasks without blocking?** The patterns here are the foundation for the [SharedMemory](../ch05_shared_memory) blackboard architecture and for any production multi-agent system at scale.
**Notation.** `pub/sub` = publish-subscribe messaging pattern (producers emit events; consumers subscribe by topic). `DLQ` = dead-letter queue (messages that exhaust max retries). `idempotency key` = unique token attached to each event to prevent duplicate processing on retry. `correlation ID` = token linking all events belonging to one logical workflow instance. `fan-out` = one event triggers multiple independent consumers. `fan-in` = results from multiple parallel consumers merged into one output.

---

## § 0 · The Challenge — Where We Are

> 🎯 **The mission**: Build **OrderFlow** — AI-native B2B purchase order automation satisfying 8 constraints:
> 1. **THROUGHPUT**: 1,000 POs/day — 2. **LATENCY**: <4hr SLA — 3. **ACCURACY**: <2% error — 4. **SCALABILITY**: 10 agents/PO — 5. **RELIABILITY**: >99.9% uptime — 6. **AUDITABILITY**: Full traceability — 7. **OBSERVABILITY**: Real-time monitoring — 8. **DEPLOYABILITY**: Zero-downtime updates

**What we know so far**:
- ✅ **Ch.1 (Message Formats)**: Structured agent message schemas prevent context overflow — single-agent 8k token limit → multi-agent decomposition foundation
- ✅ **Ch.2 (MCP)**: Tool integration protocol collapses N×M to N+M — 10 agents × 15 systems = 150 hardcoded clients → 25 MCP connections
- ✅ **Ch.3 (A2A)**: Agents distributed across 3 Kubernetes pods, cross-service delegation via A2A protocol
- ⚡ **Current metrics**: Throughput 24 POs/day (2.4% of target), Latency 36hr median, Error rate 3.2%, 8 agents across 3 pods
- ❌ **But you still can't process 1,000 POs/day!** Synchronous orchestrator threads block waiting for supplier responses.

**What's blocking us**:

🚨 **Your synchronous orchestrator is the bottleneck — and it just failed in production.**

**The incident**: PO #2024-1847 (Sarah Chen's 10 standing desks) arrived at 09:15. Your Intake agent parsed it, delegated to Pricing agent via A2A at 09:17. TechFurnish's API took 47 minutes to respond with a quote. Your orchestrator thread waited. So did PO #2024-1848, #2024-1849, and 237 others in the queue behind it. By 14:30, you had 240 POs queued, orchestrator memory at 92%, and your on-call phone ringing.

**Problems**:
1. ❌ **Synchronous blocking kills throughput**: Intake agent waits 1-2 hours for Negotiation agent → orchestrator thread blocks → max throughput = 3 threads × 8hr = **24 POs/day** (2.4% of 1,000 target). Blocks **#1 THROUGHPUT**.
2. ❌ **Queue buildup destroys latency**: 240 POs in queue × 35 min/PO = **140 hours queuing delay**. Sarah's desk order from Tuesday morning won't start processing until Friday. Blocks **#2 LATENCY**.
3. ❌ **Cascading failures**: One slow supplier API (TechFurnish timeout) stalls the entire pipeline — no PO can proceed while the orchestrator waits. Blocks **#5 RELIABILITY**.

**Business impact**: At 24 POs/day capacity, you're processing **600 POs/month** — but the business needs **22,000 POs/month**. The gap: **21,400 POs/month unprocessed** = $64M/month in delayed procurement = equipment downtime, missed production schedules, manual fallback at $420k/year labor cost.

**What this chapter unlocks**:

🚀 **Async event-driven messaging decouples producers from consumers**:
1. **Message bus replaces synchronous orchestrator**: Agents publish events to Azure Service Bus / Kafka — no blocking waits
2. **Independent scaling per agent type**: 3× Inventory agents, 8× Negotiation agents (the bottleneck), 2× Approval agents — each scales to load
3. **Dead-letter queues for graceful degradation**: Failed messages route to human review (0.2% failure rate) — don't block the pipeline

⚡ **Expected improvements**:
- **Throughput**: 24 POs/day (synchronous) → **1,200 POs/day** (50 concurrent POs × 20 POs/hr) — **50× improvement, exceeds 1,000 target** ✅
- **Latency**: 36hr median (queue buildup) → **8hr median** (async eliminates queueing) — **4.5× faster** (still not at <4hr target)
- **Reliability**: Single failure stalls pipeline → **DLQ captures 0.2% failures, all recoverable** — **production-grade resilience**
- **Constraint #1 THROUGHPUT**: ✅ **ACHIEVED** (1,200 POs/day measured in load test)

**Constraint status after Ch.4**: 
- #1 (Throughput): ⚡ **ACHIEVED** — 1,200 POs/day (120% of 1,000 target)
- #2 (Latency): ⚡ **IMPROVED** — 8hr median (still not at <4hr target, need Ch.5 shared memory to eliminate redundant work)
- #3 (Accuracy): ⚡ **STABLE** — 3.2% error (maintained from Ch.3)
- #4 (Scalability): ✅ **DISTRIBUTED** — 50 concurrent POs × 8 agents each = 400 agent instances in-flight
- #5 (Reliability): ⚡ **IMPROVED** — DLQ captures failures, no cascade
- #6 (Auditability): ⚡ **STABLE** — Correlation IDs link events to POs
- #7 (Observability): ⚡ **IMPROVED** — Message bus metrics (throughput, lag), but no distributed tracing yet
- #8 (Deployability): ❌ **BLOCKED** — No deployment automation (Ch.7)

---

## 1 · The Core Idea

**When agent tasks take minutes or hours, synchronous request-response becomes a queue-buildup disaster.** Event-driven messaging decouples producers from consumers: agents publish events to a message bus, subscribers process asynchronously. The orchestrator disappears — replaced by the topology of subscriptions. Throughput scales with message bus capacity (thousands/sec), not orchestrator thread count (3-10).

---

## 2 · Running Example: PO #2024-1847 Lifecycle

Sarah Chen's standing desk order arrives as an email at 09:15. You're the Lead Architect at OrderFlow. Your synchronous A2A system from Ch.3 just failed on this PO — here's what happened, and how you rebuild it with event-driven messaging.

**Before (Ch.3 synchronous — failed)**:
```
09:15  Sarah's email arrives → Intake agent parses → publishes to orchestrator queue
09:17  Orchestrator thread #1 picks up PO #2024-1847
09:17  Orchestrator → A2A call to Pricing agent: "Get quotes for 10 standing desks"
09:18  Pricing agent → TechFurnish API: "Quote request"
       [TechFurnish API takes 47 minutes to respond — their database is slow today]
10:04  TechFurnish → Pricing agent: "$789/desk"
10:04  Pricing agent → Orchestrator: "Best quote: TechFurnish $789/desk"
10:04  Orchestrator → A2A call to Negotiation agent: "Negotiate with TechFurnish"
       [Orchestrator thread #1 has been blocked for 47 minutes waiting]
       [POs #2024-1848 through #2024-2087 are queued, waiting for thread #1]
```
Orchestrator thread count: **3 threads**. PO processing time: **35 min/PO** (when suppliers are fast). Max throughput: **3 threads × 60 min/hr ÷ 35 min/PO = 5 POs/hr = 24 POs/day** (assumes 8-hour workday, ignores queue buildup). Actual throughput when TechFurnish is slow: **1 PO every 47 minutes = 10 POs/day**.

**After (Ch.4 event-driven — success)**:
```
09:15  Sarah's email → Intake agent → publishes event: {"topic": "order.received", "po_id": "2024-1847", "items": [...]}
09:15  [Message bus] → Pricing agent (consumer) picks up "order.received" event
09:16  Pricing agent → TechFurnish API: "Quote request"
       [Pricing agent returns immediately — does not block any orchestrator]
       [Other Pricing agent replicas (8 total) process POs #2024-1848, #2024-1849, ...]
10:03  TechFurnish → Pricing agent: "$789/desk"
10:03  Pricing agent → publishes event: {"topic": "quote.completed", "po_id": "2024-1847", "quote": ...}
10:03  [Message bus] → Negotiation agent picks up "quote.completed" event
10:03  Negotiation agent → negotiates with TechFurnish → publishes "negotiation.completed"
10:18  Approval agent → auto-approves ($7,490 < $10k threshold) → publishes "po.approved"
10:20  Drafting agent → sends PO to TechFurnish → publishes "po.sent"
```
No orchestrator thread blocked. **50 concurrent POs in-flight** at any moment (limited only by message bus throughput, not thread count). **20 POs/hr throughput = 1,000 POs/day** (assumes 50-hour work week for the system, no downtime). TechFurnish's slow response affects only PO #2024-1847's latency — does not stall the pipeline.

---

## 3 · The Architecture

### Progress on the 8 Constraints

| Constraint | Status | Evidence |
|------------|--------|----------|
| #1 THROUGHPUT | ⚡ **ACHIEVED!** | **1,000 POs/day** measured in load test (120% of target) |
| #2 LATENCY | ⚡ **IMPROVED** | 36hr → **8hr median** (async eliminates queueing delays, but not at <4hr target yet) |
| #3 ACCURACY | ⚡ **STABLE** | 3.2% error (maintained from Ch.3) |
| #4 SCALABILITY | ✅ **DISTRIBUTED** | 50 concurrent POs × 8 agents each |
| #5 RELIABILITY | ⚡ **IMPROVED** | DLQ captures failed messages (0.2% failure rate, all recoverable) |
| #6 AUDITABILITY | ⚡ **STABLE** | Correlation IDs link events to POs |
| #7 OBSERVABILITY | ⚡ **IMPROVED** | Message bus metrics (throughput, lag), but no distributed tracing |
| #8 DEPLOYABILITY | ❌ **BLOCKED** | No deployment automation |

**What's still blocking**: Pricing agent doesn't see negotiation context → quotes wrong delivery terms. Approval agent doesn't know negotiation history → asks redundant questions. *(Ch.5 — SharedMemory solves this.)*

### Where Synchronous Stops Working

> 💡 **Async prevents blocking:** In synchronous orchestration, when Agent A calls Agent B and waits for a response, that orchestrator thread is **stuck** — it can't process other tasks while waiting. If Agent B takes 2 hours (e.g., waiting for supplier quotes), that thread is blocked for 2 hours. With only 3 threads, max throughput = 3 tasks/2 hours = 36 tasks/day. Async pub/sub breaks this: Agent A publishes an event ("I need quotes") and **immediately returns** — no waiting. When Agent B finishes (2 hours later), it publishes a "quotes ready" event. Agent A's thread processed 100 other tasks in those 2 hours. Throughput jumps from 36/day to 1,000+/day using the same 3 threads. The magic: **decoupling producer availability from consumer speed**.

A synchronous orchestrator is effectively a state machine that blocks. The orchestrator calls Agent A, waits, gets a result, calls Agent B, waits. While waiting, the orchestrator thread (or async coroutine) holds state in memory and cannot serve another task.

This is acceptable when:
- Tasks are short (< 5 seconds end-to-end)
- Concurrency is low (< 100 simultaneous tasks)
- Failure in one task should halt the whole pipeline

It breaks when:
- Tasks take minutes or hours (waiting for external systems or humans)
- You need to fan out one task to 10 sub-agents simultaneously
- Individual task failures should be retried, not cascade
- You have 1,000 tasks arriving per day and need to process them in parallel

The solution is to decouple producers from consumers using a **message bus**.

### The Async Pub/Sub Model for Agents

```
                    ┌────────────────────────────────────┐
                    │           MESSAGE BUS               │
                    │                                     │
Producer:           │  Topic: "order.received"            │  Consumer:
Intake Service ────▶│  Topic: "negotiation.completed"     │──▶ Negotiation Agent
                    │  Topic: "po.approved"               │──▶ PO Drafting Agent
                    │  Topic: "po.sent"                   │──▶ Audit Logger
                    │                                     │
                    │  DLQ: "negotiation.failed"          │──▶ Human Review Agent
                    └────────────────────────────────────┘
```

Key shift in mental model: **agents do not call each other**. Each agent publishes an event to the bus when it completes work. Any agent that cares about that event subscribes to it. The orchestrator is replaced by the topology of subscriptions.

### Message Structure and Correlation

Every message in an event-driven agent system needs at minimum:

```python
{
    "message_id": "msg-f28a4c91",          # unique envelope ID for deduplication
    "correlation_id": "po-4812",           # the business entity this message relates to
    "causation_id": "msg-e7b19a03",        # the message that caused this one (tracing)
    "topic": "negotiation.completed",
    "timestamp": "2025-07-14T09:23:11Z",
    "payload": {
        "supplier_id": "SUP-88412",
        "agreed_price_usd": 14.20,
        "quantity": 500,
        "delivery_days": 7
    },
    "schema_version": "1.0"
}
```

- **`correlation_id`**: This is how the orchestrator or downstream consumers know which business entity (which PO) this result belongs to. Without it you have messages floating in the bus with no way to associate results with their originating tasks.
- **`causation_id`**: Enables distributed tracing across the agent chain — if a message triggers another message, causation_id points back. This is how you reconstruct the full execution graph in a trace viewer.
- **`schema_version`**: Agents evolve independently. A consumer must be able to ignore fields it does not understand, and must be resilient to minor schema changes without breaking. Versioning the schema makes that explicit.

### Dead-Letter Queues

A dead-letter queue (DLQ) is where messages land when processing fails after all retry attempts. Every agent subscription should have a DLQ.

```python
# Azure Service Bus example: configure a subscription with DLQ on 3 failures
subscription_config = {
    "max_delivery_count": 3,         # retry up to 3 times
    "dead_lettering_on_message_expiration": True,
    "dead_lettering_on_filter_evaluation_exceptions": True
}
```

**Why DLQs matter for agents:** A model that hallucinates an invalid tool call will cause a message to fail. Without a DLQ, failed messages either block the queue (stalling all subsequent tasks) or are silently discarded. With a DLQ, failed messages accumulate in a separate queue where a human-review agent (or a monitoring alert) can inspect them.

### Delivery Guarantees and Idempotency

| Guarantee | What it means | Agent implication |
|-----------|---------------|-------------------|
| **At-most-once** | Message delivered zero or one time, may be lost | Suitable only if losing a task is acceptable — almost never for agents |
| **At-least-once** | Message delivered one or more times, may be duplicated | The default for production systems; **agents must be idempotent** |
| **Exactly-once** | Message delivered exactly once | Expensive, requires distributed transactions; rarely worth the cost |

**Idempotency for agents:** When using at-least-once delivery, the same message may arrive and be processed twice. An idempotent tool call produces the same result on repetition. For non-idempotent tools (e.g. "send email", "charge card"), implement deduplication using the `message_id`:

```python
async def handle_send_po_email(message: Message):
    # Check if this message was already processed
    if await dedup_store.exists(message.message_id):
        logger.info(f"Duplicate message {message.message_id} — skipping")
        return

    await email_client.send(...)
    await dedup_store.set(message.message_id, ttl_seconds=86400)
```

### Fan-Out and Fan-In

**Fan-out:** One event triggers multiple agents simultaneously.

```
"order.received" topic
    ├──▶ InventoryCheckAgent  (parallel)
    ├──▶ CreditCheckAgent     (parallel)
    └──▶ SupplierLookupAgent  (parallel)
```

All three agents consume the same message concurrently. The bus handles the distribution.

**Fan-in (aggregation):** A downstream agent needs results from all three before it can proceed.

```python
# Aggregator pattern: accumulate results until all parallel tasks complete
async def aggregate_pre_checks(correlation_id: str, result: dict):
    await shared_store.append(f"prechecks:{correlation_id}", result)
    all_results = await shared_store.get_all(f"prechecks:{correlation_id}")

    if len(all_results) == EXPECTED_PARALLEL_AGENTS:  # all 3 arrived
        await bus.publish("prechecks.completed", {
            "correlation_id": correlation_id,
            "results": all_results
        })
```

The aggregator listens to a shared topic, accumulates partial results in a store (see Ch.5 — Shared Memory), and publishes a single downstream event when all parallel results have landed.

### Message Bus Options

| Bus | When to choose it |
|-----|-------------------|
| **Azure Service Bus** | Enterprise workloads requiring FIFO ordering, sessions for per-entity sequencing, dead-lettering, message scheduling. Native managed identity support. |
| **Apache Kafka** | High-throughput event streaming (>100k msg/s), long retention, replay from any offset. Operationally heavier. |
| **Redis Streams** | Low-latency, moderate throughput, when Redis is already in the stack. Consumer groups provide at-least-once delivery. |
| **RabbitMQ** | Complex routing topologies (exchange types: direct, topic, fanout, headers). Mature ecosystem. |

**For agent workloads at OrderFlow's scale (1,000 POs/day):** Azure Service Bus is the pragmatic choice. The built-in session support gives per-PO FIFO ordering; native DLQ requires no custom code; managed identity authentication fits the security model from Ch.6.

---

## 4 · How It Works — Step by Step

You're rebuilding OrderFlow's pipeline after the synchronous failure. Here's how you convert each agent to an event-driven consumer:

**Step 1: Deploy the message bus**

You provision Azure Service Bus with three topics:
- `order.received` — Intake agent publishes when email parsed
- `quote.completed` — Pricing agent publishes when quotes gathered
- `po.approved` — Approval agent publishes when PO approved

Each topic has a dead-letter queue (`order.received/$DeadLetterQueue`, etc.) for failed messages.

**Step 2: Convert agents to consumers**

Each agent becomes a consumer group listening to one topic:
- **Pricing agent** (8 replicas) subscribes to `order.received` → processes `po_id` → publishes to `quote.completed`
- **Negotiation agent** (8 replicas) subscribes to `quote.completed` → negotiates → publishes to `negotiation.completed`
- **Approval agent** (2 replicas) subscribes to `negotiation.completed` → checks thresholds → publishes to `po.approved`

No agent calls another agent. The message bus routes events.

**Step 3: Implement idempotency**

Each agent checks a Redis deduplication store before processing:
```python
if await redis.exists(f"processed:{message.message_id}"):
    logger.info(f"Duplicate message {message.message_id} — skipping")
    await bus.acknowledge(message)
    return
```
This prevents duplicate email sends, duplicate PO submissions, duplicate approval notifications.

**Step 4: Configure dead-letter routing**

If a message fails 3 times (e.g., Pricing agent can't parse item description), Azure Service Bus automatically moves it to the DLQ. You deploy a **Human Review Agent** that subscribes to all DLQs and publishes failed POs to a Slack channel for manual intervention.

**Step 5: Scale to load**

You run a load test: 1,200 POs submitted over 24 hours. Negotiation agent CPU hits 80% → you scale to 12 replicas via Kubernetes Horizontal Pod Autoscaler. Throughput stabilizes at **1,200 POs/day** with **8hr median latency**.

---

## 5 · The Key Formulas

### Little's Law for Agent Queues

For a stable message queue (arrival rate equals throughput), **Little's Law** relates the number of in-flight messages $L$, the arrival rate $\lambda$, and the mean service time $W$:

$$L = \lambda W$$

For OrderFlow's negotiation queue: $\lambda = 14$ negotiations/hr (1000 POs/day ÷ 72-hr mean PO lifetime), $W = 0.5$ hr/negotiation $\Rightarrow L = 7$ concurrent negotiations. Set `max_concurrent_agents = 8` (20% headroom).

### Fan-Out Merge Cost

When one orchestrator fans out to $k$ parallel agents and waits for all (fan-in), the merge latency is:

$$T_{\text{fanout}} = \max_{i=1}^{k} T_i + T_{\text{merge}}$$

For supplier quote collection (fan-out $k=4$ suppliers): $T_{\text{fanout}} = \max(T_1, T_2, T_3, T_4) + T_{\text{merge}}$. Async pub/sub eliminates the sequential blocking that would give $\sum_i T_i$ in a synchronous design.

| Symbol | Meaning |
|--------|---------|
| $L$ | Mean number of in-flight messages |
| $\lambda$ | Message arrival rate (messages/hr) |
| $W$ | Mean processing time per message (hr) |
| $k$ | Fan-out width (number of parallel agents) |
| $T_{\text{merge}}$ | Time to aggregate $k$ responses |

---

## 6 · What Can Go Wrong

⚠️ **Trap 1: No idempotency → duplicate actions**
- **Problem**: At-least-once delivery means messages can be processed twice. Without deduplication, you send duplicate emails, charge cards twice, submit duplicate POs.
- **Fix**: Store `message_id` in Redis with 24-hour TTL. Check before executing non-idempotent actions.

⚠️ **Trap 2: Message ordering lost across agents**
- **Problem**: Pricing agent publishes "quote.completed" for PO #2024-1847. Negotiation agent publishes "negotiation.completed". But what if negotiation completes *before* the pricing agent's message is consumed by downstream agents? The approval agent sees negotiation results before it sees the quote.
- **Fix**: Use **Azure Service Bus sessions** (per-PO FIFO ordering) or **Kafka partition keys** (route all messages for one PO to same partition).

⚠️ **Trap 3: Dead-letter queue grows unbounded**
- **Problem**: A model change causes 2% of POs to fail parsing. 20 POs/day × 30 days = 600 POs in DLQ. No one notices until the business complains about missing POs.
- **Fix**: Set up **DLQ depth alerts** (PagerDuty alert when DLQ depth > 50). Deploy a Human Review Agent that processes DLQ messages daily.

⚠️ **Trap 4: Message payload too large**
- **Problem**: Pricing agent publishes full supplier catalog (5 MB JSON) in `quote.completed` event. Azure Service Bus max message size: **256 KB**. Message rejected.
- **Fix**: Store large payloads in **blob storage** (Azure Blob, S3). Publish only a **reference** in the message: `{"quote_blob_url": "https://..."}`. Consumer fetches blob on demand.

⚠️ **Trap 5: Fan-in without timeout → stuck workflows**
- **Problem**: You fan out to 4 suppliers. 3 respond within 10 seconds. 1 supplier times out after 60 seconds. Your aggregator waits forever for the 4th response.
- **Fix**: Set a **fan-in timeout** (e.g., 30 seconds). After timeout, aggregator publishes results from the 3 responders + marks the 4th as "timeout". The downstream agent proceeds with partial data.

---

## 7 · Where This Reappears

| Chapter | How event-driven patterns appear |
|---------|----------------------------------|
| **Ch.1 — Message Formats** | Async event messages carry the same structured payload format from Ch.1; correlation IDs link events to parent tasks |
| **Ch.3 — A2A** | A2A's SSE streaming is an event-driven delivery mechanism; long-running A2A tasks publish state changes as events |
| **Ch.5 — Shared Memory** | The Redis-backed blackboard in Ch.5 is the shared state that event-driven agents write to and read from asynchronously |
| **Ch.7 — Agent Frameworks** | LangGraph's graph state can receive external events via message channels; event-driven patterns enable human-in-the-loop pauses |
| **Multi-Agent AI — Trust** | Dead-Letter Queues capture failed messages for forensic analysis; trust violations can be detected by monitoring DLQ patterns |

---

## 8 · Progress Check — What We Can Solve Now

✅ **Unlocked capabilities**:
- ✅ **Async event-driven messaging**: Agents publish to Azure Service Bus — no blocking orchestrator threads
- ✅ **Independent agent scaling**: 3× Inventory, 8× Negotiation, 2× Approval agents — each scales to load
- ✅ **Dead-letter queues**: Failed messages (0.2% rate) route to human review — don't block pipeline
- ✅ **Fan-out parallelism**: One `order.received` event triggers 4 parallel supplier quote requests — results merged via aggregator
- ✅ **Idempotency**: Redis deduplication prevents duplicate email sends, duplicate PO submissions
- ⚡ **Constraint #1 THROUGHPUT**: ✅ **ACHIEVED!** 1,200 POs/day measured in load test (120% of 1,000 target)

**Progress toward constraints**:

```mermaid
graph LR
    Ch1["Ch.1\nMessage Formats"]:::done
    Ch2["Ch.2\nMCP"]:::done
    Ch3["Ch.3\nA2A"]:::done
    Ch4["Ch.4\nEvent-Driven"]:::current
    Ch5["Ch.5\nShared Memory"]:::upcoming
    Ch6["Ch.6\nTrust & Sandboxing"]:::upcoming
    Ch7["Ch.7\nAgent Frameworks"]:::upcoming
    Ch1 --> Ch2 --> Ch3 --> Ch4 --> Ch5 --> Ch6 --> Ch7
    classDef done fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef current fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef upcoming fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

| Constraint | Before | After Ch.4 | Change |
|------------|--------|------------|--------|
| #1 THROUGHPUT | 24 POs/day (synchronous blocking) | **1,200 POs/day** | ✅ **TARGET EXCEEDED** (50× improvement) |
| #2 LATENCY | 36 hours median | **8 hours median** | ⚡ **4.5× faster** (still not <4hr target) |
| #3 ACCURACY | 3.2% error | 3.2% error | ⚡ Stable |
| #4 SCALABILITY | 8 agents, 3 pods | 8 agents, 50 concurrent POs | ✅ Validated at scale |
| #5 RELIABILITY | Task retry via A2A | **DLQ + auto-retry + 0.2% failure rate** | ⚡ **Production-grade** |
| #6 AUDITABILITY | Task IDs | Correlation IDs link events | ⚡ **Improved** |
| #7 OBSERVABILITY | Task status queryable | **Message bus metrics** (throughput, lag) | ⚡ **Improved** |
| #8 DEPLOYABILITY | No automation | No automation | ❌ No change |

**What you can solve now**:

✅ **High-volume PO processing**:
```
Before Ch.4:
Synchronous orchestrator: 3 threads × 8hr = 24 POs/day (2.4% of target)
Queue buildup: 240 POs waiting, 36hr median latency
Single supplier timeout stalls entire pipeline

After Ch.4:
Async message bus: 50 concurrent POs × 20 POs/hr = 1,200 POs/day ✅
8hr median latency (4.5× faster)
DLQ captures 0.2% failures → all recoverable

Result: ✅ Throughput constraint achieved (120% of 1,000 POs/day target)
```

✅ **Graceful degradation on supplier timeouts**:
```
Before: TechFurnish API timeout → entire pipeline stalls → no POs processed
After: TechFurnish timeout → message moved to DLQ → other POs continue → human review agent notified
Result: ✅ Reliability improved (0.2% failure rate, no cascading failures)
```

**What you still can't solve**:

- ❌ **Cross-agent context blindness**: Pricing agent doesn't see negotiation results → quotes wrong delivery terms. Approval agent doesn't know negotiation history → asks redundant questions. Each agent operates in isolation. → **Need Ch.5 (Shared Memory) for blackboard architecture — shared Redis store gives all agents visibility into full PO context**
- ❌ **<4hr latency target**: Current 8hr median (4.5× faster than 36hr, but still double target). Agents repeat work (Pricing agent re-fetches quotes already cached by Negotiation agent). → **Need Ch.5 for shared cache to eliminate redundant API calls**
- ❌ **Zero-downtime deployment**: Manual kubectl apply → brief downtime → rollback requires re-deploying previous image. → **Need Ch.7 (Agent Frameworks) for blue-green deployment + health checks**

**Real-world status**: You can now process 1,200 POs/day (target achieved ✅), but latency is 8hr median (need 4hr) and agents can't share context (need shared memory).

**Next up**: Ch.5 (Shared Memory & Blackboard Architectures) gives you **Redis-backed shared state** — Pricing agent writes quotes, Negotiation agent reads them → eliminates redundant API calls → **target: <4hr latency** ✅

---

## 9 · Bridge to Ch.5

Ch.4 unlocked **1,000 POs/day throughput** (120% of target ✅) via async messaging, but agents operate in isolation — Pricing agent can't see what Negotiation agent learned, Approval agent can't see what Pricing agent quoted → redundant API calls, repeated work, 8hr median latency (2× the <4hr target). Ch.5 (**Shared Memory & Blackboard Architectures**) introduces a **Redis-backed blackboard** where all agents read/write shared PO state → Pricing agent writes `order:2024-1847:quotes`, Negotiation agent reads it → **eliminates redundant work** → **<4hr latency target** ✅.

---

## Interview Questions

**Q: When would you choose event-driven messaging over synchronous A2A delegation?**
When task duration is unpredictable, concurrency is high, or individual failures must be isolated rather than cascading. Synchronous A2A is simpler and appropriate for short, reliable sub-tasks. Event-driven is appropriate when tasks may take minutes or hours, when you need fan-out to parallel agents, or when you need the resilience properties of a message bus (DLQ, retry, replay).

**Q: What is a dead-letter queue and why is it essential in an agent pipeline?**
A DLQ receives messages that have failed processing after the maximum retry count. Without a DLQ, unprocessable messages either block the queue or are silently discarded. In an agent pipeline, a DLQ gives you a recoverable failure mode: failed tasks accumulate in the DLQ where they can be inspected, corrected, and replayed rather than disappearing. It is where you find out that a model change caused a class of messages to fail.

**Q: Why must agents be idempotent in an at-least-once message bus, and how do you implement it for a non-idempotent tool like "send email"?**
At-least-once delivery means a message may be delivered and processed more than once (network partition, consumer crash after processing but before ack). An agent that sends an email twice on duplicate delivery causes a real-world problem. The fix: store the `message_id` in a deduplication store (Redis with a TTL matching the message retention period) and check it before executing the non-idempotent action. If the ID is already in the store, skip the action and acknowledge the message.

**Q: How do you implement fan-in — collecting results from parallel agents — in an event-driven system?**
Use an aggregator pattern: each parallel agent publishes its result to a shared topic with the same `correlation_id`. An aggregator agent subscribes to that topic and accumulates results in a shared store keyed by `correlation_id`. When the expected number of results has arrived, the aggregator publishes a single downstream event. The key requirement is knowing how many results to expect — this is typically encoded in the initial event that triggered the fan-out.

---

## Notebook

`notebook.ipynb` implements:
1. A minimal producer-consumer pair using Redis Streams
2. Idempotency via `message_id` deduplication in Redis
3. The OrderFlow fan-out: one `order.received` event triggers three parallel agents; an aggregator collects and publishes `prechecks.completed`
4. DLQ simulation: a deliberately-failing agent, showing message re-delivery and DLQ accumulation

---

## Prerequisites

- [Ch.3 — Agent-to-Agent Protocol (A2A)](../a2a) — synchronous delegation and where it breaks down
- [Ch.5 — Shared Memory & Blackboard Architectures](../ch05_shared_memory) — fan-in aggregation requires a shared store

## Next

→ [Ch.5 — Shared Memory & Blackboard Architectures](../ch05_shared_memory) — how parallel agents read and update shared state without overwriting each other

## Illustrations

![Event-driven agents - sync vs async, pub/sub with DLQ, delivery guarantees, fan-out/in](img/Event-Driven%20Agents.png)
