# Ch.1 — Message Formats & Shared Context

> **The story.** When **OpenAI's ChatCompletions API** launched in **March 2023**, it shipped a deceptively boring data structure — a JSON list of `{role, content}` messages with three roles (system, user, assistant). Within months, that envelope became the *de facto* lingua franca for the entire industry: Anthropic's Messages API, Google's Gemini API, every open-source serving framework (vLLM, llama.cpp, Ollama) all settled on the same shape. **Function calling** (OpenAI, June 2023) added a fourth role and turned messages into a structured action language. **JSON mode** and **structured outputs** (OpenAI, August 2024) made the envelope rigorously typed. Every multi-agent protocol in this track — [MCP](../MCP/), [A2A](../A2A/), [Event-driven agents](../EventDrivenAgents/) — either reuses this envelope verbatim or wraps it in transport metadata. Get this chapter right and every later chapter is just a different choreography over the same data structure.
>
> **Where you are in the curriculum.** This is the first chapter of the multi-agent track and it intentionally starts at the wire format, not at the orchestration layer. **Central question:** how do agents actually exchange information — what is physically in the message envelope, and how is shared context managed when the accumulated conversation history exceeds a single context window? The running scenario is **OrderFlow**, a B2B purchase-order automation platform.

---

## § 0 · The Challenge — Where We Are

> 🎯 **The mission**: Build **OrderFlow** — AI-native B2B purchase order automation satisfying 8 constraints:
> 1. **THROUGHPUT**: 1,000 POs/day — 2. **LATENCY**: <4hr SLA — 3. **ACCURACY**: <2% error — 4. **SCALABILITY**: 10 agents/PO — 5. **RELIABILITY**: >99.9% uptime — 6. **AUDITABILITY**: Full traceability — 7. **OBSERVABILITY**: Real-time monitoring — 8. **DEPLOYABILITY**: Zero-downtime updates

**Manual baseline**: 3 procurement specialists processing 50 POs/day @ $420k/year labor cost. 36-hour median latency, 5% error rate.

### The Blocking Question This Chapter Solves

**"How do multiple agents exchange information without exceeding context limits?"**

A single-agent system hits the 8k token context window after 3 supplier negotiations. The agent starts hallucinating supplier names from earlier orders. Need to decompose work across specialized agents without losing information.

### What We Unlock in This Chapter

- ✅ Understand the OpenAI message envelope (`role`, `content`, `tool_calls`) as the lingua franca of multi-agent systems
- ✅ Three handoff strategies: full history passthrough (audit-focused), structured payloads (production), blackboard (async)
- ✅ Context budget management: Split 24k token budget across 8 specialized agents (3k each) without overflow

### Progress on the 8 Constraints

| Constraint | Status | Evidence |
|------------|--------|----------|
| #1 THROUGHPUT | ❌ **BLOCKED** | Single-agent monolith handles 10 POs/day before context overflow |
| #2 LATENCY | ❌ **BLOCKED** | 36 hours median (manual baseline) |
| #3 ACCURACY | ❌ **BLOCKED** | 5% error rate (agent hallucinates suppliers after context fills) |
| #4 SCALABILITY | ⚡ **FOUNDATION LAID** | Can decompose single 16k-token agent → 8 agents (3k each) without context overflow |
| #5 RELIABILITY | ❌ **BLOCKED** | No retry logic, no graceful degradation |
| #6 AUDITABILITY | ⚡ **PARTIAL** | Full history passthrough enables audit, but no tracing infrastructure |
| #7 OBSERVABILITY | ⚡ **FOUNDATION LAID** | Message structure enables future tracing (but no tooling yet) |
| #8 DEPLOYABILITY | ❌ **BLOCKED** | Monolithic agent, no versioning or rollback |

**What's still blocking**: Decomposed agents into 8 specialists, but they can't access ERP, pricing APIs, or email. Need 8 × 20 = 160 custom integrations. *(Ch.2 — MCP solves this.)*

---

## Core Concepts

### The OpenAI Message Envelope

Every major agentic framework — LangChain, Semantic Kernel, AutoGen, LangGraph — serialises inter-agent communication as a list of message objects that conform to (or translate to) the OpenAI Chat Completions format. Understanding this schema makes every framework legible.

```python
# A message object — the atomic unit of agent communication
{
    "role": "user" | "assistant" | "tool" | "system",
    "content": "...",          # string or list of content parts
    "tool_calls": [...],       # present when role is "assistant" and agent invoked a tool
    "tool_call_id": "..."      # present when role is "tool" (the response to a tool_call)
}
```

**Role semantics in a multi-agent context:**

| Role | Who produces it | What it carries |
|------|----------------|-----------------|
| `system` | Orchestrator | The receiving agent's persona, constraints, and task definition |
| `user` | Calling agent / orchestrator | The task input — what this agent is asked to do |
| `assistant` | The agent itself | The agent's reasoning and/or tool invocation decisions |
| `tool` | Tool execution environment | The result of a tool call, keyed by `tool_call_id` |

When an orchestrator calls a sub-agent, the sub-agent's *entire prior conversation* (its own `system`, `user`, `assistant`, `tool` messages) is what it needs in its own context. The question is: what does the *calling* agent receive back?

---

### The Three Handoff Strategies

When Agent A calls Agent B and B finishes its work, what does A receive, and in what form?

#### Strategy 1 — Full History Passthrough

Agent B returns its entire message list to Agent A. Agent A appends it to its own context.

```python
# Agent B returns everything
handoff = {
    "status": "completed",
    "messages": [
        {"role": "system", "content": "You are the pricing specialist..."},
        {"role": "user", "content": "Negotiate supplier price for order #4812..."},
        {"role": "assistant", "tool_calls": [{"id": "tc_01", "function": {"name": "get_quote"}}]},
        {"role": "tool", "tool_call_id": "tc_01", "content": "{\"unit_price\": 14.20}"},
        {"role": "assistant", "content": "Agreed price: $14.20 per unit, 500 units."}
    ]
}
```

**When to use:** Auditing-critical systems (financial, medical) where every reasoning step must be traceable.
**Cost:** Token count accumulates multiplicatively. 3 agent hops × 2,000 tokens each = 6,000 tokens of overhead before the next agent has processed anything.

#### Strategy 2 — Structured Handoff Payload

Agent B returns only a structured summary of its result.

```python
# Agent B returns just what the next agent needs
handoff = {
    "status": "completed",
    "result": {
        "agreed_price_usd": 14.20,
        "quantity": 500,
        "delivery_days": 7,
        "supplier_id": "SUP-88412"
    }
}
```

**When to use:** Production pipelines where context budget matters more than full auditability. The result is deterministic and machine-readable.
**Cost:** Minimal. The orchestrator pays only for the output, not the reasoning trace.

#### Strategy 3 — Shared Key-Value Store (Blackboard)

Neither Agent A nor Agent B passes data directly. Both read from and write to a shared store keyed by the task ID.

```python
# Agent B writes its result to the shared store
store.set(f"order:{task_id}:pricing", {
    "agreed_price_usd": 14.20,
    "quantity": 500,
    "delivery_days": 7,
    "supplier_id": "SUP-88412"
})

# Agent A reads directly from the store without waiting for B to "return" anything
pricing = store.get(f"order:{task_id}:pricing")
```

**When to use:** Pipelines with more than 3 agents where conversation threading becomes unmanageable, or async pipelines where agents are not sequentially ordered.
**Tradeoff:** Decouples agents (neither needs to know the other's interface) at the cost of introducing a central store that becomes a consistency and latency concern.

*(See Ch.5 — Shared Memory for the full treatment.)*

---

### Context Budget Management

A context window is a finite resource. In a multi-agent chain, it is depleted by:
- The receiving agent's system prompt (typically 500–2,000 tokens)
- The task payload passed from the orchestrator
- All in-flight reasoning (assistant messages)
- All tool call/response pairs
- Any history passed in from prior agents

**The rule of thumb:** Reserve at least 20% of the context window for the model's output generation. If your accumulation is forecast to exceed 80% before the agent finishes, truncate aggressively from the *oldest* messages — not the most recent — keeping at minimum the system prompt and the current task.

```python
def trim_to_budget(messages, max_tokens, reserve_for_output=0.2):
    budget = int(max_tokens * (1 - reserve_for_output))
    # Always keep system message (messages[0]) and trim from oldest user/assistant pairs
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    while count_tokens(system + rest) > budget and len(rest) > 1:
        rest.pop(0)  # drop oldest non-system message
    return system + rest
```

---

## OrderFlow — Ch.1 Scenario

OrderFlow's first version used a single agent to handle a PO end-to-end. By order #12, the context filled up before the supplier negotiation was complete, and the model started hallucinating supplier names from earlier orders.

The fix: decompose into a pipeline of 8 specialized agents, each with a bounded context. The orchestrator calls them in sequence, passing a structured handoff payload (Strategy 2) between them. The full negotiation trace is stored in a log DB (queryable for audit) but not passed as a context message.

```
Orchestrator
  │
  ├─▶ Intake Agent       → validates request, extracts requirements
  ├─▶ Pricing Agent      → researches supplier options, gets quotes
  ├─▶ Negotiation Agent  → negotiates terms with selected supplier
  ├─▶ Legal Agent        → validates contract terms against policies
  ├─▶ Finance Agent      → confirms budget availability, approver
  ├─▶ Drafting Agent     → generates final PO document
  ├─▶ Sending Agent      → delivers PO to supplier via email
  └─▶ Reconciliation Agent → tracks delivery, closes order
```

Each agent sees only what it needs (structured payload from previous agent). No agent exceeds 40% of its context window (3k tokens used of 8k available) on the largest real orders.

---

## § 11.5 · Progress Check — What We Achieved

### Constraint Status After Ch.1

| Constraint | Before | After Ch.1 | Change |
|------------|--------|------------|--------|
| #1 THROUGHPUT | 10 POs/day (context overflow) | Still 10 POs/day | ❌ No change |
| #2 LATENCY | 36 hours median | 36 hours median | ❌ No change |
| #3 ACCURACY | 5% error (hallucinated suppliers) | **3.8% error** | ⚡ **24% better** (context overflow eliminated) |
| #4 SCALABILITY | Single agent, 16k tokens → overflow | **8 agents, 3k tokens each** | ✅ **FOUNDATION COMPLETE** |
| #5 RELIABILITY | No retry logic | No retry logic | ❌ No change |
| #6 AUDITABILITY | None | Structured payloads logged | ⚡ **Basic logging** |
| #7 OBSERVABILITY | None | Message structure enables tracing | ⚡ **Foundation laid** |
| #8 DEPLOYABILITY | Monolith | Still monolith | ❌ No change |

### The Win

✅ **Eliminated context overflow** by decomposing single agent into 8 specialized agents (Intake, Pricing, Negotiation, Legal, Finance, Drafting, Sending, Reconciliation), each under 4k tokens (50% of budget).

**Measured impact**: Error rate dropped from 5% → 3.8% (hallucination errors eliminated). Can now process arbitrarily complex POs without context overflow.

### What's Still Blocking

**The integration explosion**: 8 agents need access to 20 data sources (ERP inventory, pricing APIs, supplier emails, legal templates, approval workflows). Building custom adapters = **8 × 20 = 160 integrations**. Unmaintainable.

**Next unlock** *(Ch.2 — MCP)*: Model Context Protocol collapses 160 integrations to **8 clients + 20 servers = 28 components**. Any agent connects to any data source through a shared protocol.

---

## Interview Questions

**Q: What fields are in an OpenAI tool call message and why do they matter for multi-agent tracing?**
`tool_calls` is present on `role: assistant` messages and contains `id`, `type`, `function.name`, and `function.arguments`. The `id` (e.g. `"tc_01"`) is echoed in the subsequent `role: tool` message as `tool_call_id`. In a multi-agent trace, this pairing is what lets you reconstruct which tool response corresponds to which invocation — essential for debugging non-deterministic agent behaviour.

**Q: When would you prefer a structured handoff payload over passing full conversation history?**
When the downstream agent does not need to understand *how* the upstream agent arrived at the answer — only *what* the answer is. Full history is for auditability; structured payload is for efficiency. The cost of full history grows linearly with chain length and can exceed the context limit of the receiving agent.

**Q: A user says their agentic pipeline "gets confused" on long-running tasks. What is the first thing you check?**
Context accumulation. Run the pipeline with token counting on each agent invocation and graph the context length over time. "Gets confused" in an otherwise correct agent almost always means the oldest context has been truncated (by a framework's default limit) and the model is missing a critical earlier instruction.

**Q: What is the risk of passing the full message history from Agent B back to Agent A?**
Token cost compounds multiplicatively: a 4-agent chain where each agent accumulates 2,000 tokens of internal reasoning delivers 8,000 tokens of overhead to its caller — before the caller has started its own reasoning. In a deep chain, the accumulated handoff history can fill the entire context window before the final agent has read its own task. Additionally, if any agent injects sensitive data (API keys, PII from tool results) the full history propagates that data to every downstream agent whether it needs it or not.

---

## Notebook

`notebook.ipynb` implements:
1. A minimal two-agent pipeline using raw OpenAI message lists (no framework)
2. Token counting and trimming for each strategy
3. A side-by-side comparison of the three handoff strategies on an OrderFlow scenario: total tokens sent, reconstruction fidelity, time to implement

---

## Prerequisites

- [AI / ReActAndSemanticKernel](../../AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md) — single-agent ReAct loop before decomposing it
- [AI / PromptEngineering](../../AI/PromptEngineering/PromptEngineering.md) — system prompt construction for agent roles

## Next

→ [Ch.2 — Model Context Protocol (MCP)](../MCP/) — how to standardise tool access so any agent can call any tool without bespoke integration

## Illustrations

![Message formats - anatomy, payload styles, context growth, serialization](img/Message%20Formats.png)
