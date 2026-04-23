# Ch.3 — Agent-to-Agent Protocol (A2A)

> **The story.** **Google** announced the **Agent-to-Agent (A2A)** protocol in **April 2025** at Google Cloud Next, with launch partners including Anthropic, MongoDB, Salesforce, and SAP. The motivating distinction: an agent is *not* a tool. A tool is stateless, returns in milliseconds, and trusts whoever called it. An agent is stateful, may take minutes to complete, can spawn sub-agents, and lives in a different trust domain. A2A standardises **Agent Cards** (machine-readable capability manifests at `/.well-known/agent.json`), the task lifecycle (submit → working → completed/failed/canceled), and **SSE streaming** for incremental updates. Where [MCP](../MCP/) is the protocol for an agent calling *tools*, A2A is the protocol for an agent delegating *to other agents* — and the two were explicitly designed to compose.
>
> **Where you are in the curriculum.** [Ch.2](../MCP/) gave you tool integration. This chapter explains how delegating a task to another agent is fundamentally different from calling a tool, and what the A2A protocol provides to make that difference manageable in production: capability discovery, async task lifecycle, streaming progress, and trust boundaries. After this you have the protocol vocabulary for the rest of the track.

---

## § 0 · The Challenge — Where We Are

> 🎯 **The mission**: Build **OrderFlow** — AI-native B2B purchase order automation satisfying 8 constraints:
> 1. **THROUGHPUT**: 1,000 POs/day — 2. **LATENCY**: <4hr SLA — 3. **ACCURACY**: <2% error — 4. **SCALABILITY**: 10 agents/PO — 5. **RELIABILITY**: >99.9% uptime — 6. **AUDITABILITY**: Full traceability — 7. **OBSERVABILITY**: Real-time monitoring — 8. **DEPLOYABILITY**: Zero-downtime updates

**After Ch.2**: 8 specialized agents connected to 20 data sources via MCP (28 components vs. 160 integrations). Error rate 3.2%.

### The Blocking Question This Chapter Solves

**"How do agents on different servers delegate tasks to each other?"**

Intake agent (Pod 1) needs to delegate to Negotiation agent (Pod 3) across Kubernetes cluster. No standard protocol. Current workaround: HTTP POST with custom JSON schema. But: no capability discovery, no lifecycle tracking, no streaming progress, no retry semantics.

### What We Unlock in This Chapter

- ✅ Understand A2A protocol: Agent Cards (capability manifest), task lifecycle (submitted → working → completed/failed)
- ✅ Cross-service delegation: Agents discover each other via `/.well-known/agent.json`, delegate via HTTP + SSE
- ✅ Distributed agent topology: Agents run on separate machines/clusters without tight coupling

### Progress on the 8 Constraints

| Constraint | Status | Evidence |
|------------|--------|----------|
| #1 THROUGHPUT | ❌ **BLOCKED** | Still 10 POs/day (synchronous blocking) |
| #2 LATENCY | ❌ **BLOCKED** | 36 hours median (synchronous polling) |
| #3 ACCURACY | ⚡ **STABLE** | 3.2% error (maintained from Ch.2) |
| #4 SCALABILITY | ✅ **DISTRIBUTED!** | Agents run on separate machines/clusters |
| #5 RELIABILITY | ⚡ **IMPROVED** | Task IDs enable retry after crashes |
| #6 AUDITABILITY | ⚡ **STABLE** | Task lifecycle persisted |
| #7 OBSERVABILITY | ⚡ **STABLE** | Task status queryable |
| #8 DEPLOYABILITY | ⚡ **FOUNDATION LAID** | Agent Cards enable versioning (but no CI/CD automation) |

**What's still blocking**: Synchronous A2A polling blocks Intake agent for 1-2 hours while Negotiation agent works → can only handle 3 × 8hr = **24 POs/day**. Need async pub/sub to hit 1,000 POs/day. *(Ch.4 — Event-driven solves this.)*

---

## Why Tools and Agents Are Not the Same Thing

A tool is a stateless function: you give it input, it executes synchronously, and it returns output. The tool does not reason, it does not call other tools, and it does not take minutes to complete. Calling a tool in a ReAct loop is a round trip measured in milliseconds.

An agent has its own reasoning loop. It may invoke multiple tools, make branching decisions, wait for external systems, and produce a result after seconds, minutes, or hours. Treating another agent like a tool — firing a request and blocking — means the calling agent's context window and memory footprint grow while it waits, and a failure in the sub-agent has no lifecycle management at all: it just never returns.

**A2A** (Agent-to-Agent Protocol), published by Google in 2025, is the open HTTP protocol that formalises this difference. It gives agents a standard vocabulary for discovering each other, delegating tasks, streaming progress, and handling failure — without one agent needing to know how the other is implemented.

---

## The Agent Card

Every A2A-compliant agent publishes an **Agent Card** at a well-known URL:

```
GET https://supplier-negotiation.orderflow.internal/.well-known/agent.json
```

```json
{
  "name": "SupplierNegotiationAgent",
  "description": "Negotiates purchase order terms with registered suppliers.",
  "version": "1.2.0",
  "url": "https://supplier-negotiation.orderflow.internal/a2a",
  "capabilities": {
    "streaming": true,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "negotiate_po",
      "name": "Negotiate Purchase Order",
      "description": "Given supplier options and a target price, negotiates final terms.",
      "inputModes": ["text/plain", "application/json"],
      "outputModes": ["application/json"]
    }
  ],
  "authentication": {
    "schemes": ["Bearer"]
  }
}
```

The Agent Card answers: what can this agent do, what formats does it accept, what authentication does it require, and does it support streaming? A calling agent can make an informed delegation decision from this card alone — no human configuration required.

---

## Task Lifecycle

A2A tasks follow a strict state machine. This is the core semantic difference from a tool call, which has no lifecycle — it either returns or throws.

```
                          ┌──────────┐
                          │submitted │  ← Client sends the task
                          └────┬─────┘
                               │
                          ┌────▼─────┐
                          │ working  │  ← Agent is actively processing
                          └────┬─────┘
                               │
              ┌────────────────┼─────────────────┐
              │                │                 │
        ┌─────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
        │ completed  │  │   failed    │  │  cancelled  │
        └────────────┘  └─────────────┘  └─────────────┘
```

Each state transition is observable by the calling agent through polling or SSE streaming.

### Sending a Task

```python
import httpx

async def delegate_to_negotiation_agent(order_details: dict, auth_token: str) -> str:
    """Returns a task_id that can be polled or streamed."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://supplier-negotiation.orderflow.internal/a2a/tasks",
            json={
                "skill_id": "negotiate_po",
                "input": {
                    "content": order_details,
                    "content_type": "application/json"
                },
                "metadata": {
                    "correlation_id": order_details["po_id"]
                }
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        return response.json()["task_id"]
```

### Streaming Progress (SSE)

```python
async def stream_task_progress(task_id: str, auth_token: str):
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "GET",
            f"https://supplier-negotiation.orderflow.internal/a2a/tasks/{task_id}/stream",
            headers={"Authorization": f"Bearer {auth_token}"}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    event = json.loads(line[5:])
                    if event["status"] == "completed":
                        return event["result"]
                    elif event["status"] == "failed":
                        raise AgentTaskFailed(event["error"])
```

SSE streaming means the calling agent does not poll in a loop and does not block a thread. It connects a streaming response and receives state transitions as they happen.

---

## MCP and A2A — Complementary, Not Competing

This is one of the most commonly misunderstood architectural questions in multi-agent design:

| Layer | Protocol | What it governs |
|-------|----------|----------------|
| Tool / Resource access | **MCP** | How an agent accesses data sources and executable functions |
| Agent delegation | **A2A** | How one agent delegates a task to another agent |

They are designed to be stacked:

```
Orchestrator
    │ delegates via A2A
    ▼
SupplierNegotiationAgent
    │ accesses tools via MCP
    ├──▶ MCP ERP Server (Resource: supplier records)
    ├──▶ MCP Pricing Server (Tool: get_real_time_quote)
    └──▶ MCP Email Server (Tool: send_offer_email)
```

A calling agent should not care whether the sub-agent uses MCP, direct API calls, or some other internal mechanism to do its work. A2A abstracts the *task*; MCP abstracts the *tools*. The sub-agent uses MCP internally; the calling agent uses A2A to reach the sub-agent.

---

## OrderFlow — Ch.3 Scenario

OrderFlow's procurement orchestrator needed to call the supplier negotiation service — a team-owned Python service running in a separate container — without the orchestrator team coupling to the negotiation team's internal API.

The negotiation service published an Agent Card. The orchestrator read the card, confirmed it supported the `negotiate_po` skill, and delegated tasks via A2A. When the negotiation took 45 minutes (waiting for a human at the supplier side to respond), the orchestrator did not block: it submitted the task, stored the `task_id` alongside the PO record, and picked up the result via SSE when the negotiation completed.

The compliance team added a new requirement mid-project: all delegated tasks must include a `correlation_id` linking back to the PO. Because all task submissions went through A2A's `metadata` field, the change was a one-line addition to the orchestrator — no negotiation service code changed.

---

## § 11.5 · Progress Check — What We Achieved

### Constraint Status After Ch.3

| Constraint | Before | After Ch.3 | Change |
|------------|--------|------------|--------|
| #1 THROUGHPUT | 10 POs/day | **24 POs/day** | ⚡ **2.4× faster** (but still far from 1,000 target) |
| #2 LATENCY | 36 hours median | 36 hours median | ❌ No change |
| #3 ACCURACY | 3.2% error | 3.2% error | ⚡ Stable |
| #4 SCALABILITY | 8 agents, single cluster | **Distributed across 3 Kubernetes pods** | ✅ **Cluster-scale achieved** |
| #5 RELIABILITY | No retry logic | Task IDs enable retry after crash | ⚡ **Improved** |
| #6 AUDITABILITY | MCP tool call logging | Task lifecycle persisted | ⚡ **Improved** |
| #7 OBSERVABILITY | MCP logs | Task status queryable via A2A API | ⚡ **Improved** |
| #8 DEPLOYABILITY | No versioning | Agent Cards declare versions | ⚡ **Foundation laid** |

### The Win

✅ **Cross-service agent delegation**: Agents can now run on separate Kubernetes pods and delegate tasks via A2A protocol. Intake agent (Pod 1) delegates to Negotiation agent (Pod 3) via HTTP + SSE streaming.

**Measured impact**: Throughput increased 10 → 24 POs/day (3 orchestrator threads × 8hr). Task failures now retryable via task IDs.

### Agent Topology Deployed

```
Intake Agent (Pod 1) ──A2A──▶ Pricing Agent (Pod 2)
                      ──A2A──▶ Negotiation Agent (Pod 3)
                      ──A2A──▶ Legal Agent (Pod 4)
```

### What's Still Blocking

**Synchronous blocking**: Intake agent polls Negotiation agent for 1-2 hours (waits for "completed" status). During this time, orchestrator thread holds state in memory and cannot process another PO. Max throughput: **3 threads × 8hr = 24 POs/day** (2.4% of 1,000 target).

**Next unlock** *(Ch.4 — Event-driven)*: Async pub/sub messaging decouples orchestrator from agent execution time. 50 concurrent POs in-flight × 20 POs/hr = **1,000 POs/day capacity**.

---

## Interview Questions

**Q: How is calling an agent different from calling a tool, and why does that difference matter architecturally?**
A tool is a stateless, synchronous function — input in, output out, no state, typically milliseconds. An agent has its own reasoning loop, can invoke multiple tools, may take minutes or hours, and can fail at any intermediate step. Treating an agent call like a tool call means the calling agent must either block (consuming memory and context) or implement its own ad hoc polling, failure handling, and lifecycle tracking. A2A formalises the lifecycle (submitted → working → completed/failed/cancelled) and provides SSE streaming, so the calling agent can submit and move on.

**Q: What is an Agent Card and what information does it contain?**
An Agent Card is a JSON document served at `/.well-known/agent.json` that describes what an agent can do. It includes: name and version, the base URL for A2A requests, the list of skills with their input/output content types, capability flags (does it support streaming, push notifications?), and the authentication schemes it accepts. A calling agent can use the card to make a delegation decision without any human-configured knowledge about the sub-agent.

**Q: Can you use MCP and A2A together in the same system?**
Yes, they are designed to be complementary layers. MCP governs how an agent accesses tools and data sources. A2A governs how one agent delegates tasks to another agent. A typical architecture: the orchestrator uses A2A to delegate to specialist agents; each specialist agent uses MCP to access the tools it needs. The orchestrator does not need to know what tools the specialist uses internally.

**Q: What are the A2A task lifecycle states?**
`submitted` (the client has sent the task), `working` (the agent is processing), `completed` (successful result available), `failed` (the agent encountered an unrecoverable error), `cancelled` (the client or server cancelled the task). Each transition is observable via SSE streaming or polling.

**Q: A2A requires Bearer token authentication. Where do the tokens come from in a cloud deployment?**
In a cloud deployment, managed identity is the correct pattern: each agent service is assigned a managed identity (e.g. Azure Managed Identity, AWS IAM role) and exchanges it for short-lived bearer tokens via the platform's OAuth 2.0 token endpoint. No static secrets are stored; tokens rotate automatically; access can be scoped to specific agents. This integrates cleanly with A2A's `"authentication": {"schemes": ["Bearer"]}` declaration in the Agent Card.

---

## Notebook

`notebook.ipynb` implements:
1. A minimal A2A-compliant server (FastAPI) exposing one skill with the full task lifecycle
2. An A2A client that reads the Agent Card, delegates a task, and streams progress via SSE
3. The OrderFlow scenario: orchestrator delegates a 10-second mock negotiation to the A2A server and handles the `completed` and `failed` states
4. Side-by-side: synchronous blocking call vs A2A async delegation — token usage and wall time comparison

---

## Prerequisites

- [Ch.1 — Message Formats & Shared Context](../MessageFormats/) — understanding what is in the handoff payload
- [Ch.2 — Model Context Protocol (MCP)](../MCP/) — the tool layer that sub-agents use internally

## Next

→ [Ch.4 — Event-Driven Agent Messaging](../EventDrivenAgents/) — what happens when you have 1,000 tasks in flight simultaneously and synchronous delegation is no longer viable

## Illustrations

![A2A - tool vs agent, agent card, task lifecycle, MCP+A2A layering](img/A2A.png)
