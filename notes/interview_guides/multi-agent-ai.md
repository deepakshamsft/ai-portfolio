# Multi-Agent AI Systems · Interview Guide

This guide consolidates interview preparation material from all chapters in the Multi-Agent AI track, focusing on production-grade multi-agent architectures and protocols.

---

## 1 · Concept Map — The 10 Questions That Matter

Every multi-agent interview revolves around 10 core question clusters. Senior answers demonstrate end-to-end systems thinking — not just protocol names, but tradeoffs, failure modes, and production integration.

| # | Cluster | What the interviewer is testing |
|---|---------|----------------------------------|
| 1 | **Message Formats & Handoff** | Do you know the three handoff strategies? Can you explain why blackboard scales and full-history doesn't? |
| 2 | **MCP N×M Reduction** | Can you explain why MCP solves the integration problem? Know Resources vs. Tools vs. Prompts? |
| 3 | **A2A Task Lifecycle** | Do you know the 5 task states and SSE streaming? Can you contrast agent calls vs. tool calls architecturally? |
| 4 | **Async Pub/Sub Messaging** | Can you apply Little's Law to size a queue? Know fan-out/fan-in patterns and DLQ? |
| 5 | **Blackboard Architecture** | Do you know namespace isolation, scope hierarchy, and failure recovery via the blackboard? |
| 6 | **Trust & Prompt Injection** | Can you explain prompt injection propagation? Know HMAC timing attack defence and sandbox requirements? |
| 7 | **Framework Tradeoffs** | Can you distinguish AutoGen from LangGraph from Semantic Kernel and recommend the right one? |
| 8 | **Idempotency & Reliability** | Do you know why at-least-once delivery requires idempotent agents and how to implement deduplication? |
| 9 | **Auth & Credential Management** | Can you describe the managed identity pattern for agent-to-agent auth in cloud deployments? |
| 10 | **Protocol Composition** | Do you know how MCP + A2A + event bus compose into a complete production architecture? |

---

## 2 · Section-by-Section Deep Dives

**Q: What is the difference between "reduce to shared context" and "pass full history" message handoff strategies?**

**Reduce to shared context**: the orchestrator compresses the conversation into a summary or key-value context object before sending to the sub-agent. Sub-agent receives only the essentials, not the full message history. Pro: smaller payload, lower cost. Con: lossy compression, context may miss details.

**Pass full history**: the orchestrator sends the entire conversation (all system/user/assistant/tool messages) to the sub-agent. Sub-agent has complete context. Pro: no information loss. Con: token count scales linearly with conversation length, expensive at scale.

**Q: When would you use Strategy 2 (system prompt specialisation) over Strategy 3 (blackboard)?**

Strategy 2 when the sub-agent's role is self-contained and ephemeral — a pricing calculation, a single email draft. You inject the orchestrator's summary as the sub-agent's system prompt and throw away the sub-agent's context after it responds.

Strategy 3 when multiple agents need to coordinate and share state — inventory check, pricing approval, supplier negotiation all write to the same shared memory. Use a blackboard when the task requires more than 3 agents or when agents are async.

**Q: What are the three ways to pass context to an agent, and which one scales to 10+ agents?**

1. **Full history** — pass entire message array; works for 1–2 agents, breaks at scale  
2. **Summary in system prompt** — orchestrator injects context into sub-agent's system prompt; works for 3–5 independent agents  
3. **Blackboard** — all agents read/write to a shared store; only this scales to 10+ agents because it decouples communication from orchestration

---

## Ch.2 — Model Context Protocol (MCP)

**Q: What are the three MCP primitive types and how do you decide which to use?**

**Resource** when the agent needs to read data without modifying it (a product catalogue, a database record). **Tool** when the agent needs to take an action or mutate state (send an email, write a record). **Prompt** when you have reusable, parameterised instruction templates that should live server-side rather than in agent code.

**Q: What problem does MCP solve that plain function calling does not?**

Discovery and reuse. With plain function calling, the agent must already know the tool's schema (hardcoded or injected via system prompt). With MCP, the server self-describes its capabilities at connection time. Any MCP-compliant agent can discover and use any MCP-compliant server without prior configuration — the `N × M` integration problem becomes `N + M`.

**Q: What is the difference between stdio and HTTP+SSE transports, and when would you choose each?**

`stdio` starts the MCP server as a subprocess and communicates via stdin/stdout — lowest latency, suitable for local tools (code execution, local file access), but limited to one client at a time. `HTTP + SSE` exposes the server over a network — supports multiple concurrent clients, can be scaled independently, required for remote services. Use stdio in development and for trusted local tools; use HTTP+SSE for production services.

**Q: Does MCP replace RAG?**

No. MCP exposes data as Resources that an agent can address by URI — it handles the access and delivery layer. The retrieval strategy (what to retrieve, how to chunk, how to rank), the vector database, and the embedding model are still entirely your responsibility. MCP and RAG are complementary: the RAG retrieval logic can be packaged as an MCP Tool.

---

## Ch.3 — Agent-to-Agent Protocol (A2A)

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

## Ch.4 — Event-Driven Agent Messaging

**Q: When would you choose event-driven messaging over synchronous A2A delegation?**

When task duration is unpredictable, concurrency is high, or individual failures must be isolated rather than cascading. Synchronous A2A is simpler and appropriate for short, reliable sub-tasks. Event-driven is appropriate when tasks may take minutes or hours, when you need fan-out to parallel agents, or when you need the resilience properties of a message bus (DLQ, retry, replay).

**Q: What is a dead-letter queue and why is it essential in an agent pipeline?**

A DLQ receives messages that have failed processing after the maximum retry count. Without a DLQ, unprocessable messages either block the queue or are silently discarded. In an agent pipeline, a DLQ gives you a recoverable failure mode: failed tasks accumulate in the DLQ where they can be inspected, corrected, and replayed rather than disappearing. It is where you find out that a model change caused a class of messages to fail.

**Q: Why must agents be idempotent in an at-least-once message bus, and how do you implement it for a non-idempotent tool like "send email"?**

At-least-once delivery means a message may be delivered and processed more than once (network partition, consumer crash after processing but before ack). An agent that sends an email twice on duplicate delivery causes a real-world problem. The fix: store the `message_id` in a deduplication store (Redis with a TTL matching the message retention period) and check it before executing the non-idempotent action. If the ID is already in the store, skip the action and acknowledge the message.

**Q: How do you implement fan-in — collecting results from parallel agents — in an event-driven system?**

Use an aggregator pattern: each parallel agent publishes its result to a shared topic with the same `correlation_id`. An aggregator agent subscribes to that topic and accumulates results in a shared store keyed by `correlation_id`. When the expected number of results has arrived, the aggregator publishes a single downstream event. The key requirement is knowing how many results to expect — this is typically encoded in the initial event that triggered the fan-out.

---

## Ch.5 — Shared Memory & Blackboard Architectures

**Q: What is the blackboard pattern and when would you use it over direct agent-to-agent message passing?**

The blackboard pattern places all inter-agent communication through a single shared store. Agents read what they need, write what they produce, and never call each other directly. Use it when there are more than 3 agents in a pipeline (direct coupling becomes a combinatorial problem), when agents are async and not sequentially ordered, or when you need failure recovery — a crashed agent can restart and continue from its last write. Use direct message passing for simple synchronous 2–3 agent chains where context is small.

**Q: Why is namespace isolation critical when multiple agents write to the same blackboard?**

Without namespace isolation, agents can overwrite each other's data. For example, if both the inventory agent and the approval agent write to `po:{id}:status`, the last one wins and the first one's result is silently discarded. Use agent-scoped sections (hash fields or namespaced keys) and enforce the rule that each agent writes only to its own section. Treat another agent's section as read-only.

**Q: How does a blackboard help with failure recovery in an event-driven pipeline?**

When an agent fails mid-task and the message is re-delivered (at-least-once), the new agent instance can read the blackboard to find any partial progress. Instead of starting from scratch, it continues from the last successfully written state. This is particularly valuable for long-running tasks (e.g. multi-turn supplier negotiations) where restarting from zero is prohibitively expensive.

**Q: What is the difference between per-task, per-entity, and per-user memory scopes?**

**Per-task** memory (keyed by task_id) is ephemeral — it exists only for the duration of one pipeline execution and is deleted on completion. **Per-entity** memory (keyed by business entity like po_id) persists for the lifetime of that entity and spans multiple pipeline runs on the same entity. **Per-user** memory (keyed by user_id) is long-lived, survives sessions, and stores preferences and interaction history. Mixing scopes in the same key namespace is a common source of subtle bugs — design the key schema explicitly before writing agent code.

---

## Ch.6 — Trust, Sandboxing & Authentication

**Q: What is the biggest security risk in a multi-agent system?**

Prompt injection propagating through the agent chain. External content (web pages, documents, emails, API responses) that passes through an agent's reasoning can contain embedded instructions. If that content is then passed to the next agent as a trusted message (especially as system-role content), the injected instructions may be executed by the downstream agent without the user's knowledge. The defence is to treat any content that originated outside your trust boundary as `user`-role input, not `system`, regardless of which agent retrieved it.

**Q: Why should `hmac.compare_digest` be used instead of `==` when verifying signatures?**

String comparison with `==` is vulnerable to timing attacks: the comparison short-circuits on the first mismatching character and returns faster for strings that match the expected value in the first few characters. An attacker who can measure response time can use this to incrementally guess the correct signature. `hmac.compare_digest` always takes the same time regardless of where the mismatch occurs, making timing attacks infeasible.

**Q: A model generates and executes code as part of an agent tool. What sandboxing would you apply?**

At minimum: subprocess isolation (the code runs in a separate process, not the agent's process). In production: Docker-per-execution with network disabled, memory limit, CPU quota, and `remove=True` so the container is destroyed after execution. The goal is zero persistence and zero outbound network access, so even a fully successful code injection has no path to exfiltration or persistence.

**Q: What is the recommended authentication pattern for agent-to-agent calls in a cloud deployment?**

Managed identity. Each agent service is assigned a managed identity and exchanges it for short-lived bearer tokens at runtime. No static credentials exist in code, config files, or environment variables that could be leaked. Access can be scoped to the exact resources and agents each service needs, and tokens rotate automatically.

**Q: Where in the message schema should external content (supplier emails, web page content) be injected?**

Always in the `user` role, never the `system` role. The `system` prompt defines the agent's identity, constraints, and decision rules — it is the high-authority instruction. The `user` role is where input data lives. If external content is interpolated into the `system` prompt, injected instructions in that content inherit system-level authority. If it is in the `user` role, the agent's `system` instructions still govern its behaviour.

---

## Ch.7 — Agent Frameworks

**Q: When would you use AutoGen over LangGraph?**

When the control flow is genuinely open-ended or emergent — when you do not know in advance which agent should speak next or how many rounds the task will take. AutoGen's conversation model is well-suited to debate patterns (proposer-critic), research (search-summarise-critique-refine), and exploratory tasks. LangGraph is better when the workflow is known and fixed — when you need deterministic control flow, conditional branching by explicit criteria, or regulatory compliance that requires a guaranteed execution order.

**Q: Can you use AutoGen and LangGraph together in the same system?**

Yes. An AutoGen conversation can be encapsulated as a function/node inside a LangGraph graph — LangGraph controls the overall pipeline; AutoGen handles open-ended sub-tasks within it. They are not mutually exclusive and often complement each other: LangGraph for the outer deterministic orchestration, AutoGen for inner emergent reasoning steps.

**Q: What does Semantic Kernel add beyond what AutoGen or LangGraph provide?**

Production hooks: a filter pipeline for pre/post-processing every function invocation (audit logs, PII scrubbing, cost tracking), OpenTelemetry-compatible telemetry pluggable into Azure Monitor, explicit `TerminationStrategy` and `SelectionStrategy` as testable code objects rather than heuristics, and native MCP plugin integration. It is designed for enterprise deployments where the conversation itself is not the hard part — the compliance, auditability, and operational observability are.

**Q: How does MCP interact with AutoGen, LangGraph, and SK?**

In all three, MCP tools appear as callables that the agent framework can invoke. AutoGen: register the MCP tool on a `ConversableAgent`'s tool list. LangGraph: wrap the MCP client call in a node function or use a LangChain-MCP adapter as a tool. Semantic Kernel: use the SK MCP plugin connector to register an MCP server as an SK plugin — SK's function-calling infrastructure then handles invocation, result parsing, and sending back to the model.

---

## Related Topics

- [Agentic AI Interview Guide](agentic-ai.md) — CoT, ReAct, RAG, embeddings, semantic caching
- [AI / ReAct & Semantic Kernel](../ai/react_and_semantic_kernel) — SK plugin basics and ReAct pattern fundamentals
- [AI / Safety & Hallucination](../ai/safety_and_hallucination) — hallucination mitigation that complements injection defence

---

## 3 · The Rapid-Fire Round

> 20 Q&A pairs. Each answer: ≤ 3 sentences.

**1. What are the three message handoff strategies?**
Full history (pass all messages), system-prompt specialisation (inject summary into sub-agent's system prompt), and blackboard (shared store). Full history is simplest but costs grow quadratically. Blackboard is the only one that scales to 10+ agents.

**2. What problem does MCP solve?**
The N×M integration problem: without MCP, N agents each need custom code to call M tools (N×M connections). MCP gives servers a self-describing protocol, so any agent discovers any server at connection time (N+M connections).

**3. What are the three MCP primitive types?**
Resource (read-only data, like a catalogue), Tool (action/mutation, like sending an email), and Prompt (server-side reusable instruction templates). Use the type that matches the intent — mixing them undermines intent-clarity.

**4. How is an agent call different from a tool call?**
Tools are stateless and synchronous — milliseconds, no state. Agent calls initiate a reasoning loop that can take minutes, involve multiple tools, and fail at intermediate steps. A2A formalises the lifecycle (submitted → working → completed/failed).

**5. What is an Agent Card?**
A JSON document served at `/.well-known/agent.json` describing the agent's skills, input/output types, transport capabilities, and authentication schemes. Enables discovery without prior configuration.

**6. What is Little's Law and how does it size an agent queue?**
L = λW: the mean number of in-flight messages equals arrival rate times mean processing time. For 14 negotiations/hr at 0.5 hr each → 7 concurrent; set max_concurrent_agents = 8 (20% headroom).

**7. What is a dead-letter queue?**
A queue that receives messages after the maximum retry count. Without a DLQ, failed messages are silently discarded. The DLQ is where you detect that a model change caused a class of tasks to fail permanently.

**8. Why must agents be idempotent in at-least-once delivery?**
A message may be processed twice (consumer crash after processing but before ack). Non-idempotent actions (send email, charge payment) must be guarded by a deduplication store checked before execution.

**9. What is the blackboard pattern?**
All inter-agent communication goes through a shared store. Agents read and write; they never call each other directly. Required for 10+ agents or async pipelines where direct coupling becomes combinatorial.

**10. Why is namespace isolation critical in a blackboard?**
Without it, agents overwrite each other's keys. Each agent owns one namespace section; other agents' sections are read-only. Violating this causes silent data corruption.

**11. What is the biggest security risk in multi-agent systems?**
Prompt injection propagating through the chain. External content retrieved by agent A passes to agent B as a trusted message, where injected instructions execute with agent B's authority.

**12. Why use `hmac.compare_digest` instead of `==` for signature verification?**
String `==` short-circuits on the first mismatch — timing varies with match position, enabling a timing attack to guess the signature. `compare_digest` always takes constant time.

**13. Where should external content be injected in a message?**
Always in the `user` role, never `system`. The `system` role conveys high-authority instructions; injecting external content there gives any embedded attacker instructions system-level authority.

**14. AutoGen vs. LangGraph — when to use each?**
AutoGen for open-ended/emergent workflows where the number of agent turns is not known (debate, research). LangGraph for deterministic control flow with explicit conditional branching and compliance requirements.

**15. Can AutoGen and LangGraph be combined?**
Yes. An AutoGen conversation can be a node inside a LangGraph graph. LangGraph controls the outer deterministic pipeline; AutoGen handles inner emergent sub-tasks.

**16. What does Semantic Kernel add over LangGraph?**
Production hooks: filter pipeline for audit/PII, OpenTelemetry telemetry, `TerminationStrategy` and `SelectionStrategy` as testable code objects, and native MCP plugin integration. Designed for enterprise auditability.

**17. How does MCP compose with A2A?**
MCP governs agent-to-tool access; A2A governs agent-to-agent delegation. A typical architecture: orchestrator uses A2A to delegate to specialist agents; each specialist uses MCP to access its tools internally.

**18. What is the recommended auth pattern for agent-to-agent calls in cloud?**
Managed identity. Each agent service exchanges its managed identity for short-lived bearer tokens. No static credentials; tokens rotate automatically; access is scoped to exact needed resources.

**19. How do you implement fan-in from parallel agents?**
Each parallel agent publishes its result with the same `correlation_id`. An aggregator accumulates results in a shared store; when the expected count arrives, it publishes a single downstream event.

**20. MCP stdio vs. HTTP+SSE — when to use each?**
Stdio is fastest and suitable for local trusted tools but supports only one client. HTTP+SSE supports multiple concurrent clients and can be scaled independently — required for production remote services.

---

## 4 · Signal Words That Distinguish Answers

**✅ Say this:**
- \"N×M becomes N+M\" (MCP's architectural value)
- \"task lifecycle\" (not \"agent call\")
- \"namespace isolation\" (blackboard safety)
- \"prompt injection propagation\" (not just \"prompt injection\")
- \"managed identity\" (not \"API key\")
- \"dead-letter queue\" (not \"error handling\")
- \"idempotency guard\" (not \"retry logic\")
- \"correlation ID\" (linking fan-out results)
- \"at-least-once delivery\" (naming the delivery guarantee)
- \"trust boundary\" (where external content enters)

**❌ Don't say this:**
- \"agents call each other\" (ignores the protocol layer)
- \"just add more agents\" (shows no understanding of coordination cost)
- \"it retries automatically\" (ignores idempotency requirement)
- \"put it in the system prompt\" (insecure for external content)
- \"store the API key in config\" (should be managed identity)

