# Ch.5 — Shared Memory & Blackboard Architectures

> **Central question:** How do multiple agents read and update a single source of truth — and what are the tradeoffs between a shared blackboard, direct history passthrough, and per-entity key-value memory?

---

## The Memory Problem in Multi-Agent Systems

Multi-agent systems have a memory problem that does not exist in single-agent systems. In a single-agent ReAct loop, all context lives in one place: the context window. As soon as you split work across agents, that unified memory shatters.

Consider a five-agent pipeline where each agent needs to know what the previous four decided. The naive approach — passing full conversation history through every handoff — hits two walls simultaneously: the accumulated history grows linearly with chain length (potentially exceeding each agent's context window), and every agent receives the full reasoning trace of every other agent including irrelevant sections.

**Shared memory** solves this by giving agents a single external store they can all read from and write to, keyed by the entity they are all working on. Each agent appends only its own results; downstream agents read exactly what they need.

---

## The Blackboard Pattern

The blackboard is an architectural pattern where all agents communicate exclusively through a shared data structure — never directly with each other.

```
┌─────────────────────────────────────────────────────────────┐
│                    BLACKBOARD (Redis / DB)                    │
│                                                               │
│  po:PO-4812                                                   │
│    ├── intake:       { supplier, items, quantity }            │
│    ├── inventory:    { available: true, lead_time_days: 3 }   │
│    ├── negotiation:  { agreed_price: 14.20, supplier_id: ... }│
│    ├── approval:     { approved: true, approver: "auto" }     │
│    └── drafting:     { po_document_url: "..." }               │
└─────────────────────────────────────────────────────────────┘
         ▲                 ▲                 ▲
         │                 │                 │
  Intake Agent    Negotiation Agent    Drafting Agent
  (writes intake)  (reads intake,     (reads everything,
                   writes negotiation) writes drafting)
```

No agent calls another agent. Each agent subscribes to an event (or is scheduled) that signals its turn to work, reads what it needs from the blackboard, does its reasoning, and writes its section back.

---

## Memory Scope: Per-Task vs Per-Entity vs Per-User

| Scope | Key structure | Lifecycle | Use case |
|-------|--------------|-----------|----------|
| **Per-task** | `task:{task_id}` | Deleted on task completion | Ephemeral working memory within a pipeline run |
| **Per-entity** | `po:{po_id}`, `order:{order_id}` | Retained for the life of the entity | State that spans multiple pipeline runs on the same entity |
| **Per-user** | `user:{user_id}:preferences` | Long-lived, survives sessions | Preferences, interaction history, learned behaviours |

Mixing scopes in the same key namespace causes subtle bugs — a per-task key that is cleaned up too early corrupts the per-entity record. Design your key schema before writing a single agent.

---

## Implementation in Redis

### Writing Agent-Scoped Sections

The critical rule: each agent writes **only its own section** of the blackboard record. No agent overwrites another agent's keys. Use namespaced keys or a hash field per agent:

```python
import redis.asyncio as redis
import json

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

async def write_negotiation_result(po_id: str, result: dict):
    """Write negotiation results to the blackboard. Never touches other agents' fields."""
    await r.hset(
        f"po:{po_id}",
        mapping={"negotiation": json.dumps(result)}
    )
    # Publish event so downstream agents know this section is ready
    await r.publish(f"po:{po_id}:events", json.dumps({"section": "negotiation", "status": "complete"}))

async def read_intake_for_negotiation(po_id: str) -> dict:
    """Read only the intake section — no agent reads more than it needs."""
    raw = await r.hget(f"po:{po_id}", "intake")
    if raw is None:
        raise MissingBlackboardSection(f"intake section not found for {po_id}")
    return json.loads(raw)
```

### Optimistic Locking for Concurrent Writers

When two agents write to the same entity at the same time (e.g. two inventory agents racing), you need write protection. Redis `WATCH` implements optimistic locking:

```python
async def safe_write_section(po_id: str, section: str, data: dict):
    key = f"po:{po_id}"
    async with r.pipeline() as pipe:
        while True:
            try:
                await pipe.watch(key)
                current = await pipe.hget(key, section)
                if current is not None:
                    raise SectionAlreadyWritten(f"{section} already written for {po_id}")
                pipe.multi()
                pipe.hset(key, section, json.dumps(data))
                await pipe.execute()
                break
            except redis.WatchError:
                continue  # another client modified the key — retry
```

---

## Blackboard vs Direct Message Passing

| Dimension | Blackboard | Direct message passing |
|-----------|-----------|----------------------|
| **Coupling** | Agents are decoupled — neither knows the other exists | Agents are directly coupled — caller knows callee's interface |
| **Debugging** | Full state visible in one place at any point in time | State distributed across multiple agents' context windows |
| **Consistency** | Requires locking / versioning to prevent conflicts | Each agent's state is isolated — no contention |
| **Latency** | Store read adds round-trip latency (~1ms for Redis) | Synchronous chains have no store overhead |
| **Failure recovery** | Failed agent can be retried from where it left off | Failed agent loses all in-flight state unless explicitly checkpointed |
| **Scalability** | Store becomes a bottleneck under very high write throughput | Scales naturally — no shared resource |

**Decision rule:** Use a blackboard when there are more than 3 agents in a pipeline, when agents are async, or when failure recovery is critical. Use direct message passing for simple synchronous chains of 2–3 agents where context is small and control flow is linear.

---

## In-Memory vs External Store

A common shortcut: agents share a Python dict or a singleton class instance. This works only when all agents run in the same process. As soon as agents are distributed across containers or services, in-process shared state is gone.

```python
# WRONG in a distributed system — works in a local notebook, breaks in production
shared_blackboard = {}  # dies when the process restarts or when using multiple replicas

# RIGHT — external store that all agents can reach
r = redis.Redis(host=REDIS_HOST, port=6379)
```

During development, use an in-process dict. Before staging, replace it with Redis. Design the interface (read/write functions) such that the swap is a one-line configuration change, not a rewrite.

---

## Long-Term Agent Memory

Beyond per-task blackboards, production systems frequently need **long-term memory**: facts about users, entities, or the world that persist across sessions and tasks.

Patterns:
1. **Key-value store** (Redis, DynamoDB): fast lookups for structured facts. `user:U-1234:preferences → {currency: "GBP", notify_by: "email"}`.
2. **Vector database** (see AI / VectorDBs): semantic retrieval of past interactions. The agent embeds the current context and retrieves the most relevant past records. Useful when "what has the user told us before that is relevant to this question?" is the retrieval goal.
3. **Relational schema**: full CRUD on structured history. Slower but richer query capability — e.g. "all POs from this supplier in the past 6 months".

Long-term memory introduces data retention obligations (GDPR, etc.) that task-scoped blackboards do not. Design the memory model before storing anything.

---

## OrderFlow — Ch.5 Scenario

OrderFlow's negotiation agent was crashing mid-session on long supplier negotiations. Each crash lost all the accumulated negotiation context — which items the agent had offered, which the supplier had rejected, what the current floor price was.

The fix: the negotiation agent wrote its state to the blackboard after every exchange with the supplier (not just on completion). When a crash occurred and the message was re-delivered (see Ch.4 — at-least-once delivery), the new agent instance read the existing `negotiation_state` section from the blackboard and continued from where the previous instance had stopped — same conversation, no lost context, supplier unaware of the restart.

---

## Interview Questions

**Q: What is the blackboard pattern and when would you use it over direct agent-to-agent message passing?**
The blackboard pattern places all inter-agent communication through a single shared store. Agents read what they need, write what they produce, and never call each other directly. Use it when there are more than 3 agents in a pipeline (direct coupling becomes a combinatorial problem), when agents are async and not sequentially ordered, or when you need failure recovery — a crashed agent can restart and continue from its last write. Use direct message passing for simple synchronous 2–3 agent chains where context is small.

**Q: Why is namespace isolation critical when multiple agents write to the same blackboard?**
Without namespace isolation, agents can overwrite each other's data. For example, if both the inventory agent and the approval agent write to `po:{id}:status`, the last one wins and the first one's result is silently discarded. Use agent-scoped sections (hash fields or namespaced keys) and enforce the rule that each agent writes only to its own section. Treat another agent's section as read-only.

**Q: How does a blackboard help with failure recovery in an event-driven pipeline?**
When an agent fails mid-task and the message is re-delivered (at-least-once), the new agent instance can read the blackboard to find any partial progress. Instead of starting from scratch, it continues from the last successfully written state. This is particularly valuable for long-running tasks (e.g. multi-turn supplier negotiations) where restarting from zero is prohibitively expensive.

**Q: What is the difference between per-task, per-entity, and per-user memory scopes?**
**Per-task** memory (keyed by task_id) is ephemeral — it exists only for the duration of one pipeline execution and is deleted on completion. **Per-entity** memory (keyed by business entity like po_id) persists for the lifetime of that entity and spans multiple pipeline runs on the same entity. **Per-user** memory (keyed by user_id) is long-lived, survives sessions, and stores preferences and interaction history. Mixing scopes in the same key namespace is a common source of subtle bugs — design the key schema explicitly before writing agent code.

---

## Notebook

`notebook.ipynb` implements:
1. An in-process blackboard (Python dict) for a 4-agent pipeline — baseline reference
2. A Redis-backed blackboard with agent-scoped hash fields and publish/subscribe events
3. Optimistic locking with `WATCH` for concurrent writers
4. The OrderFlow failure recovery scenario: crash mid-negotiation, re-delivery, resume from last written state

---

## Prerequisites

- [Ch.1 — Message Formats & Shared Context](../MessageFormats/) — the three handoff strategies; this chapter expands Strategy 3
- [Ch.4 — Event-Driven Agent Messaging](../EventDrivenAgents/) — agents consume events and write results to the blackboard

## Next

→ [Ch.6 — Trust, Sandboxing & Authentication](../TrustAndSandboxing/) — now that agents share a blackboard and communicate at scale, what are the attack surfaces and how do you close them?
