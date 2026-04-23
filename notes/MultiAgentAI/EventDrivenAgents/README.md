# Ch.4 — Event-Driven Agent Messaging

> **The story.** Async, durable messaging is older than ML — IBM's MQSeries shipped in 1993, **Apache Kafka** came out of LinkedIn in 2011, **Redpanda** and **NATS JetStream** modernised the Kafka recipe a decade later. The patterns themselves — dead-letter queues, idempotency keys, correlation IDs, fan-out/fan-in — came from Gregor Hohpe & Bobby Woolf's *Enterprise Integration Patterns* (2003), the same book every microservices architect kept on their desk in the 2010s. The multi-agent twist arrived in 2023–25: when an orchestrator agent has to coordinate dozens of long-running sub-agents (each potentially making LLM calls that take seconds), synchronous request-response collapses. The fix is exactly what the EIP book wrote down 20 years earlier — just with LLM tasks on the bus instead of stock trades. AWS's **Step Functions**, Azure's **Durable Functions**, **Temporal**, and **Inngest** are all cloud-native expressions of this pattern, increasingly billed as agent orchestrators.
>
> **Where you are in the curriculum.** [Ch.1](../MessageFormats/)–[Ch.3](../A2A/) used synchronous protocols. This chapter answers: **when does synchronous request-response break down in a multi-agent system, and how do you rebuild the coordination layer on top of async pub/sub messaging to handle thousands of concurrent agent tasks without blocking?** The patterns here are the foundation for the [SharedMemory](../SharedMemory/) blackboard architecture and for any production multi-agent system at scale.

---

## § 0 · The Challenge — Where We Are

> 🎯 **The mission**: Build **OrderFlow** — AI-native B2B purchase order automation satisfying 8 constraints:
> 1. **THROUGHPUT**: 1,000 POs/day — 2. **LATENCY**: <4hr SLA — 3. **ACCURACY**: <2% error — 4. **SCALABILITY**: 10 agents/PO — 5. **RELIABILITY**: >99.9% uptime — 6. **AUDITABILITY**: Full traceability — 7. **OBSERVABILITY**: Real-time monitoring — 8. **DEPLOYABILITY**: Zero-downtime updates

**After Ch.3**: Agents distributed across 3 Kubernetes pods, cross-service delegation via A2A. Throughput: 24 POs/day (2.4% of target). Error rate: 3.2%.

### The Blocking Question This Chapter Solves

**"How do we handle 1,000 POs/day without blocking orchestrator threads?"**

Synchronous A2A: Intake agent waits 1-2 hours for Negotiation agent → orchestrator thread blocks → max throughput = 3 threads × 8hr = **24 POs/day** (2.4% of 1,000 target). Need async pub/sub to decouple producer from consumer.

### What We Unlock in This Chapter

- ✅ Async pub/sub messaging: Agents publish events to message bus (Kafka, Azure Service Bus, NATS)
- ✅ Decoupled orchestration: Intake agent publishes `order.received` → Pricing agent subscribes, processes async
- ✅ Throughput scaling: **50 concurrent POs in-flight × 20 POs/hr = 1,000 POs/day capacity**
- ✅ Dead-letter queues: Failed messages route to human review (don't block pipeline)

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

---

## Where Synchronous Stops Working

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

---

## The Async Pub/Sub Model for Agents

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

---

## Message Structure and Correlation

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

---

## Dead-Letter Queues

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

---

## Delivery Guarantees and Idempotency

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

---

## Fan-Out and Fan-In

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

---

## Message Bus Options

| Bus | When to choose it |
|-----|-------------------|
| **Azure Service Bus** | Enterprise workloads requiring FIFO ordering, sessions for per-entity sequencing, dead-lettering, message scheduling. Native managed identity support. |
| **Apache Kafka** | High-throughput event streaming (>100k msg/s), long retention, replay from any offset. Operationally heavier. |
| **Redis Streams** | Low-latency, moderate throughput, when Redis is already in the stack. Consumer groups provide at-least-once delivery. |
| **RabbitMQ** | Complex routing topologies (exchange types: direct, topic, fanout, headers). Mature ecosystem. |

**For agent workloads at OrderFlow's scale (1,000 POs/day):** Azure Service Bus is the pragmatic choice. The built-in session support gives per-PO FIFO ordering; native DLQ requires no custom code; managed identity authentication fits the security model from Ch.6.

---

## OrderFlow — Ch.4 Scenario

OrderFlow's synchronous pipeline hit its ceiling on day 3 of a high-demand period: 240 POs in the queue, the orchestrator running 240 blocking coroutines, memory at 92%, and supplier response times averaging 37 minutes.

The rebuild: each agent was converted to a queue consumer. The orchestrator became a thin coordinator that published `order.received` and watched for `po.complete`. Each specialist agent ran as an independently scaled container pool — inventory check agents at ×3 replicas, negotiation agents at ×8 replicas (the bottleneck).

When a negotiation agent crashed mid-task, the message was automatically re-delivered to another negotiation agent replica after the lock timeout. No task was lost. The crashed agent's partial work was visible in the shared PO record in Redis (see Ch.5).

---

## § 11.5 · Progress Check — What We Achieved

### Constraint Status After Ch.4

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

### The Win

✅ **1,000 POs/day achieved!** Async pub/sub decoupled orchestrator from agent execution time. 50 concurrent POs in-flight × 20 POs/hr = **1,200 POs/day capacity** (120% of target).

**Measured impact**:
- Throughput: 24 → 1,200 POs/day (50× improvement)
- Latency: 36hr → 8hr median (4.5× faster)
- Reliability: DLQ captures 0.2% failures → all recoverable

### Architecture Shift

```
Before (Ch.3 synchronous):
Intake ──blocks 1 hr──▶ Negotiation ──blocks 30 min──▶ Drafting
Max throughput: 3 orchestrator threads × 8 hr = 24 POs/day

After (Ch.4 async):
Intake ─publish─▶ Bus ─subscribe─▶ Negotiation ─publish─▶ Bus ─subscribe─▶ Drafting
Max throughput: 50 concurrent POs × 20 POs/hr = 1,000 POs/day ✅
```

### What's Still Blocking

**Cross-agent context blindness**: Pricing agent doesn't see negotiation results → quotes wrong delivery terms. Approval agent doesn't know negotiation history → asks redundant questions. Each agent operates in isolation.

**Next unlock** *(Ch.5 — SharedMemory)*: Blackboard architecture (shared Redis store) gives all agents visibility into full PO context. Pricing agent reads `order:PO-4812:negotiation` → instant access to agreed terms.

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

- [Ch.3 — Agent-to-Agent Protocol (A2A)](../A2A/) — synchronous delegation and where it breaks down
- [Ch.5 — Shared Memory & Blackboard Architectures](../SharedMemory/) — fan-in aggregation requires a shared store

## Next

→ [Ch.5 — Shared Memory & Blackboard Architectures](../SharedMemory/) — how parallel agents read and update shared state without overwriting each other

## Illustrations

![Event-driven agents - sync vs async, pub/sub with DLQ, delivery guarantees, fan-out/in](img/Event-Driven%20Agents.png)
