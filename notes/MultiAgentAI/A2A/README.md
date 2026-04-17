# Ch.3 — Agent-to-Agent Protocol (A2A)

> **Central question:** How is delegating a task to another agent different from calling a tool — and what does the A2A protocol provide that makes that difference manageable in production?

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
