# Ch.2 — Model Context Protocol (MCP)

> **The story.** **Anthropic** announced the **Model Context Protocol** on **25 November 2024** as an open standard built on JSON-RPC 2.0. The motivating problem was the **N×M integration explosion** — every agent had to ship custom adapter code for every data source. MCP defined three reusable primitives — *Resources*, *Tools*, *Prompts* — plus standard transports (stdio for local, SSE/HTTP for remote). Adoption was unusually fast for a protocol: by mid-2025 OpenAI, Microsoft, and Google had all shipped MCP support; Claude Desktop, Cursor, Zed, and VS Code Copilot all spoke MCP natively; and the public registry had passed several thousand servers. **MCP is now the protocol for tool/data integration in multi-agent systems**, the same way HTTP is the protocol for hypertext.
>
> **Where you are in the curriculum.** [Ch.1](../ch01_message_formats) gave you the message envelope. This chapter answers: **what problem does MCP solve that plain function calling does not, and how does the JSON-RPC 2.0 protocol turn any data source or executable function into something any compliant agent can discover and use without bespoke adapter code?** Master this and the [A2A](../ch03_a2a) chapter — agent-to-agent delegation — will compose cleanly with it.
>
> **Notation.** `MCP` = Model Context Protocol (Anthropic, November 2024). `JSON-RPC 2.0` = the wire transport format (`method`, `params`, `id`, `result` / `error`). `Resource` = read-only data exposed by an MCP server. `Tool` = executable function exposed by an MCP server. `Prompt` = reusable prompt template registered with an MCP server. `SSE` = Server-Sent Events (HTTP streaming transport for remote MCP servers). `stdio` = standard-input/output transport for local in-process MCP servers.

---

## § 0 · The Challenge — Where We Are

> 🎯 **The mission**: Build **OrderFlow** — AI-native B2B purchase order automation satisfying 8 constraints:
> 1. **THROUGHPUT**: 1,000 POs/day — 2. **LATENCY**: <4hr SLA — 3. **ACCURACY**: <2% error — 4. **SCALABILITY**: 10 agents/PO — 5. **RELIABILITY**: >99.9% uptime — 6. **AUDITABILITY**: Full traceability — 7. **OBSERVABILITY**: Real-time monitoring — 8. **DEPLOYABILITY**: Zero-downtime updates

**What we know so far**:
- ✅ **Ch.1 Message Formats**: Decomposed single agent into 8 specialized agents (Intake, Pricing, Negotiation, Legal, Finance, Drafting, Sending, Reconciliation)
- ✅ **Context overflow eliminated**: Each agent stays under 4k token budget (50% of 8k limit)
- ✅ **Error rate improved**: 5% → 3.8% (structured message schemas prevent parsing failures)
- ⚡ **Current metrics**: 10 POs/day throughput, 36 hours median latency, 3.8% error rate
- ❌ **But we still can't ground agents in real-time data!** Each agent needs access to ~20 data sources (ERP, pricing APIs, supplier APIs, email, legal templates). Without a standard protocol, that's **8 agents × 20 integrations = 160 bespoke implementations**.

**What's blocking us**:

🚨 **The N×M Integration Explosion**

You're the Lead Architect at OrderFlow. Your 8 agents are working, but they're blind. The Pricing agent needs live supplier quotes — someone hardcoded an HTTP client for TechFurnish's API. Then OfficeDepot. Then 18 other suppliers. The Negotiation agent needs the same data — a different engineer wrote different wrappers. The Finance agent needs ERP access — a third team wrote a third set of adapters.

**Current situation**: Three teams, nine custom wrappers (3 agents × 3 systems), zero reusability. Each wrapper has subtly different error handling. One silently swallows 404s. Another retries infinitely on timeout. The third crashes the agent.

```
Problems:
1. ❌ **Integration explosion**: N agents × M tools = N×M bespoke adapters (8 × 20 = 160 for OrderFlow) → **Blocks #4 SCALABILITY**
2. ❌ **No schema discovery**: Agent code hardcodes API schemas; when supplier API changes, agents break silently → **Blocks #3 ACCURACY**
3. ❌ **Zero observability**: Custom wrappers don't log tool calls consistently; cannot debug which agent called which supplier → **Blocks #7 OBSERVABILITY**
```

**Business impact**: You hired 2 engineers for 6 months just to write integration adapters ($180k labor cost). When TechFurnish changed their pricing API, 3 agents broke in production. OrderFlow processed zero POs for 4 hours. The CTO is demanding: **"Why can't we add a new supplier without rewriting half the codebase?"**

**What this chapter unlocks**:

🚀 **Model Context Protocol (MCP) — collapse N×M to N+M**:
1. **Standard protocol for tool access**: JSON-RPC 2.0 transport → any agent connects to any data source without custom code
2. **Self-describing servers**: Tools declare their JSON Schema at runtime → agents discover capabilities dynamically, no hardcoded schemas
3. **Integration collapse**: 160 bespoke integrations → **8 MCP clients + 20 MCP servers = 28 components** (94% reduction)

⚡ **Expected improvements**:
- **Throughput**: 10 → 10 POs/day (no change yet — still sequential architecture)
- **Latency**: 36 hours → 36 hours (no change yet — still synchronous)
- **Error rate**: 3.8% → **3.2%** (agents grounded in real-time ERP data, no hallucinated pricing)
- **Scalability**: 160 integrations → **28 components** (94% reduction) → **#4 SCALABILITY foundation ✅**
- **Auditability**: Basic logging → **MCP tool call logging** (partial observability) → **#6 AUDITABILITY improved ⚡**
- **Observability**: Message structure only → **MCP request/response logged** (standardized format) → **#7 OBSERVABILITY improved ⚡**

### Progress on the 8 Constraints

| Constraint | Status | Evidence |
|------------|--------|----------|
| #1 THROUGHPUT | ❌ **BLOCKED** | Still 10 POs/day (integration bottleneck) |
| #2 LATENCY | ❌ **BLOCKED** | 36 hours median (manual baseline) |
| #3 ACCURACY | ⚡ **IMPROVED** | 3.8% → **3.2% error** (agents grounded in real ERP data, no hallucinated pricing) |
| #4 SCALABILITY | ✅ **VALIDATED** | 8 agents share 20 MCP servers (no integration duplication) |
| #5 RELIABILITY | ❌ **BLOCKED** | No graceful degradation |
| #6 AUDITABILITY | ⚡ **IMPROVED** | MCP servers log all tool calls (partial observability) |
| #7 OBSERVABILITY | ⚡ **IMPROVED** | MCP tool calls logged (but no distributed tracing) |
| #8 DEPLOYABILITY | ❌ **BLOCKED** | No deployment automation |

**What's still blocking**: Agents on different servers can't delegate tasks to each other (e.g., Intake agent can't call Negotiation agent across Kubernetes pods). *(Ch.3 — A2A solves this.)*

---

## § 1 · The Core Idea

**Model Context Protocol (MCP)** collapses the N×M integration problem to N+M by defining a standard JSON-RPC 2.0 protocol for agent-tool communication. You write each tool integration once as an MCP server; any compliant agent becomes an MCP client and discovers available tools at runtime without hardcoded schemas. **Integration count scales linearly with agents plus tools, not multiplicatively.**

### Agent vs MCP Server — Role Clarity

One common point of confusion: **what is an agent, and what is an MCP server?**

| Aspect | Agent (MCP Client) | MCP Server |
|--------|-------------------|------------|
| **Role** | **Consumes** tools and resources | **Exposes** tools and resources |
| **Example** | Pricing Agent, Negotiation Agent | ERP Server, Supplier Quote Server |
| **Protocol Side** | Sends `tools/call` requests | Responds with `result` or `error` |
| **Implementation** | LangChain, AutoGen, Semantic Kernel, Claude Desktop | Python MCP SDK, Node.js MCP SDK, custom JSON-RPC server |
| **Cardinality** | N agents (8 in OrderFlow) | M servers (20 in OrderFlow) |
| **Reusability** | Agent-specific logic (orchestration, decision-making) | Cross-agent reusable (any client can call any server) |
| **LLM Required?** | Yes (agent makes decisions) | No (server is deterministic code) |

**Key insight**: One agent is an MCP client. It can call multiple MCP servers. One MCP server can be called by multiple agents. This is the **N+M collapse** — instead of writing N×M custom integrations, you write N clients + M servers.

---

## § 2 · Running Example: PO #2024-1847 Pricing Lookup

Your Pricing agent needs to quote 10 standing desks from TechFurnish and OfficeDepot. Before MCP: you wrote `techfurnish_client.py` with hardcoded HTTP endpoints and response parsing. When TechFurnish changed their API schema, your agent crashed. When you added OfficeDepot, you wrote `officedepot_client.py` — same pattern, different bugs.

With MCP: you write `pricing-mcp-server` once, exposing `get_supplier_quote(supplier_name, item_id, quantity)` as a Tool. Both TechFurnish and OfficeDepot are wrapped behind this single server. Your Pricing agent calls `tools/call` with `{"name": "get_supplier_quote", "arguments": {"supplier_name": "TechFurnish", "item_id": "DESK-001", "quantity": 10}}`. The server returns `{"price": 789, "delivery_days": 14}`. When TechFurnish's API changes, you update the server implementation — zero agent code changes.

**Result**: PO #2024-1847 pricing lookup succeeded in 847ms (real-time supplier API call). Before MCP, this same lookup would have failed silently when TechFurnish changed their schema 3 weeks ago.

---

## § 3 · The Protocol Specification

MCP is built on **JSON-RPC 2.0** — a lightweight remote procedure call protocol using JSON serialization. Every MCP interaction is a request-response pair over stdio (subprocess pipes) or HTTP+SSE (Server-Sent Events) transports.

### The N×M Integration Problem

Without MCP, every integration between an agent and a tool is a custom adapter: the agent team writes a Python wrapper for the ERP, another team writes a different wrapper for the pricing API, and none of them are reusable across agents or reusable by agents built in different frameworks.

With `N` agents and `M` tools, that is `N × M` bespoke integrations. MCP collapses it to `N + M`: each tool becomes an MCP server (written once), and each agent becomes an MCP client (written once per framework). Any client connects to any server through a shared protocol.

```
Without MCP:                        With MCP:
Agent A → ERP adapter               Agent A ─────┐
Agent A → Pricing adapter           Agent B ─────┼──▶ MCP Protocol ──▶ ERP Server
Agent B → ERP adapter               Agent C ─────┘                ──▶ Pricing Server
Agent B → Pricing adapter                                          ──▶ Email Server
   = N × M adapters                    = N clients + M servers
```

### Protocol Mechanics

MCP is an open standard published by Anthropic (November 2024). It is built on **JSON-RPC 2.0** — a lightweight remote procedure call protocol that uses JSON as its serialisation format and requires no HTTP (though HTTP is supported).

### Handshake

```json
// Client → Server: initialise the connection and negotiate capabilities
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": { "tools": {} }
  }
}

// Server → Client: confirm capabilities
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": { "tools": { "listChanged": true } }
  }
}
```

The server is now self-described. The client does not need prior knowledge of what the server can do — it discovers it through the protocol.

---

## The Three Primitives

MCP defines exactly three types of thing a server can expose. Understanding the semantic difference between them is the most common interview test.

### Resources — Read-only data the agent can inspect

Resources are URI-addressable content. The agent requests a resource, the server returns its content. The agent does not modify it (that would be a Tool).

```python
# Server exposes order records as resources
@mcp_server.resource("order://{order_id}")
def get_order(order_id: str) -> str:
    record = db.fetch_order(order_id)
    return json.dumps(record)

# Client reads the resource — same pattern as fetching a URL
content = await mcp_client.read_resource(f"order://PO-4812")
```

**Examples:** database records, file contents, API schema documentation, product catalogues.

### Tools — Callable functions with side effects

Tools are functions the agent can invoke. Unlike Resources, Tools can mutate state.

```python
# Server exposes a tool
@mcp_server.tool()
def send_purchase_order(po_document: str, supplier_email: str) -> dict:
    """Send a purchase order to a supplier via email."""
    result = email_client.send(to=supplier_email, body=po_document)
    return {"message_id": result.id, "status": "sent"}
```

The critical detail: the server's `tools/list` response includes the full JSON Schema for each tool's input parameters. The agent never has to guess or hardcode the schema — the server declares it.

```json
{
  "name": "send_purchase_order",
  "description": "Send a purchase order to a supplier via email.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "po_document": {"type": "string"},
      "supplier_email": {"type": "string", "format": "email"}
    },
    "required": ["po_document", "supplier_email"]
  }
}
```

### Prompts — Reusable, parameterised instruction templates

Prompts are pre-built instruction templates stored server-side. The client can request a prompt by name, pass parameters, and receive a fully-formed message list. This is where domain-specific instruction engineering lives, separated from agent code.

```python
# Server exposes a prompt template
@mcp_server.prompt()
def negotiate_price_prompt(supplier_name: str, target_price: float) -> list[dict]:
    return [
        {"role": "system", "content": f"You are a procurement specialist negotiating with {supplier_name}."},
        {"role": "user", "content": f"Your target price is ${target_price:.2f} per unit. Begin negotiation."}
    ]

# Client retrieves it
messages = await mcp_client.get_prompt("negotiate_price_prompt",
                                        arguments={"supplier_name": "Acme Corp", "target_price": 13.50})
```

**Why Prompts belong in MCP:** The prompt to negotiate with a specific supplier changes when the business changes. Storing it server-side means the agent binary does not need to be redeployed when the instruction changes.

---

## Transport Options

| Transport | Mechanism | Typical use case |
|-----------|-----------|-----------------|
| `stdio` | stdin/stdout pipes to a subprocess | Local tools: the agent spawns the MCP server as a child process on the same machine |
| `HTTP + SSE` | HTTP POST for requests, Server-Sent Events for streaming responses | Remote tools: the MCP server is a service reachable over a network |

**Choosing transport:** Use `stdio` when the tool is a local process (e.g. a code executor, a local file system scanner). Use `HTTP + SSE` when the tool is deployed as a service, needs to scale independently, or must be accessed from multiple agents concurrently.

---

## § 4 · How It Works — Step by Step

**OrderFlow MCP interaction flow** (Pricing agent queries TechFurnish supplier):

```
PricingAgent (MCP Client)          pricing-mcp-server          TechFurnish API
    |                                     |                          |
    |──1. initialize──────────────────▶  |                          |
    |◀─────capabilities: {tools}─────────|                          |
    |                                     |                          |
    |──2. tools/list──────────────────▶  |                          |
    |◀─────[get_supplier_quote]──────────|                          |
    |                                     |                          |
    |──3. tools/call────────────────────▶|                          |
    |   {name: "get_supplier_quote",     |──HTTP POST──────────────▶|
    |    arguments: {supplier: "TechF",  |                          |
    |                item: "DESK-001",   |◀─────{price: 789}────────|
    |                qty: 10}}            |                          |
    |◀─────result: {price: 789}──────────|                          |
```

**Step-by-step**:
1. **Handshake**: Agent sends `initialize` with protocol version, server confirms capabilities
2. **Discovery**: Agent calls `tools/list`, server returns JSON Schema for each tool
3. **Invocation**: Agent calls `tools/call` with tool name + arguments, server validates against schema and executes
4. **Response**: Server returns result or error in standard JSON-RPC format

💡 **Key insight**: The agent never hardcoded TechFurnish's API. The server wrapped it. When you add OfficeDepot tomorrow, the Pricing agent's code doesn't change — you just add OfficeDepot logic inside the MCP server.

---

## § 5 · Key Diagrams

*(See illustration at end of chapter)*

---

## § 6 · Production Considerations

| Concern | MCP Solution |
|---------|-------------|
| **Latency** | stdio transport: <1ms overhead for local tools; HTTP+SSE: ~10ms for remote servers |
| **Error handling** | JSON-RPC 2.0 error codes: -32700 (parse error), -32600 (invalid request), -32601 (method not found) |
| **Authentication** | OAuth 2.0 bearer tokens for HTTP transport; process-level trust for stdio |
| **Rate limiting** | Server-side: implement token bucket or leaky bucket inside MCP server |
| **Monitoring** | Every MCP call is a JSON-RPC request → standard middleware can intercept and log |
| **Deployment** | stdio servers: ship as standalone binaries; HTTP servers: containerize and deploy behind load balancer |

⚠️ **Common trap**: Exposing raw database access as MCP Resources. Instead, expose semantic operations as Tools (e.g., `get_inventory_level(item_id)` not `SELECT * FROM inventory`). This prevents agents from issuing arbitrary SQL.

---

## MCP vs Plain Function Calling

| Dimension | Plain function calling | MCP |
|-----------|----------------------|-----|
| Schema discovery | Schema is hardcoded in the client or passed in the system prompt | Schema is declared by the server at runtime via `tools/list` |
| Cross-agent reuse | The wrapper is coupled to the agent framework | Any MCP client can call any MCP server regardless of framework |
| Versioning | Client breaks silently if the tool's signature changes | Server negotiates protocol version during handshake; schema is always current |
| Observability | Tool calls are opaque unless the agent framework provides hooks | Every MCP call is a JSON-RPC request; standard middleware can intercept and log |
| Authentication | Ad hoc per integration | Standardised — OAuth 2.0 bearer tokens for HTTP transport; process-level trust for stdio |

---

## § 7 · What Can Go Wrong

**1. Tool schema drift**: Server updates tool signature without versioning → clients send old argument shape → validation fails.
- **Fix**: Version your tools (e.g., `get_supplier_quote_v2`) or use semantic versioning in MCP server manifest. Deprecate old versions gracefully.

**2. stdio deadlock**: Agent writes to server stdin, blocks waiting for response, but server is blocked writing to stdout (buffer full) → both processes hang.
- **Fix**: Use non-blocking I/O or larger OS pipe buffers. Production: prefer HTTP+SSE transport for concurrent clients.

**3. Exposing raw database access as Resources**: Agent requests `resource://db/inventory/all` and gets 500 MB of JSON → context window explosion.
- **Fix**: Resources should be semantic, not raw. Expose `resource://inventory/item/{id}` not `resource://db/*`. Rate-limit resource size to <10 KB.

**4. No authentication**: HTTP MCP server has no auth → any network client can call `tools/delete_all_orders`.
- **Fix**: Require OAuth 2.0 bearer tokens for HTTP transport. Validate tokens before processing `tools/call`.

**5. Logging proxy breaks observability**: Custom MCP proxy strips `correlation_id` from requests → distributed tracing fails.
- **Fix**: Proxies must preserve all JSON-RPC fields. Add new fields (e.g., `audit_timestamp`) but never remove existing ones.

---

## § 8 · Progress Check — What We Can Solve Now

```mermaid
graph LR
    Ch1["Ch.1\nMessage Formats"]:::done
    Ch2["Ch.2\nMCP"]:::current
    Ch3["Ch.3\nA2A"]:::upcoming
    Ch4["Ch.4\nEvent-Driven"]:::upcoming
    Ch5["Ch.5\nShared Memory"]:::upcoming
    Ch6["Ch.6\nTrust & Sandboxing"]:::upcoming
    Ch7["Ch.7\nAgent Frameworks"]:::upcoming
    Ch1 --> Ch2 --> Ch3 --> Ch4 --> Ch5 --> Ch6 --> Ch7
    classDef done fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef current fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef upcoming fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

✅ **Unlocked capabilities**:
- ✅ **Standard tool protocol**: Any agent can call any tool via JSON-RPC 2.0 without custom adapter code
- ✅ **Runtime schema discovery**: Agents discover tool schemas dynamically at connection time (no hardcoded schemas)
- ✅ **Integration collapse**: 160 bespoke integrations reduced to 28 components (8 MCP clients + 20 MCP servers)
- ✅ **Real-time data grounding**: Agents now query live ERP inventory, supplier pricing APIs, legal templates (no hallucinated data)

### Constraint Status After Ch.2

| Constraint | Before | After Ch.2 | Change |
|------------|--------|------------|--------|
| #1 THROUGHPUT | 10 POs/day | Still 10 POs/day | ❌ No change |
| #2 LATENCY | 36 hours median | 36 hours median | ❌ No change |
| #3 ACCURACY | 3.8% error | **3.2% error** | ⚡ **16% better** (grounded in real ERP data) |
| #4 SCALABILITY | 8 agents, but 160 integrations | **8 clients + 20 servers = 28 components** | ✅ **94% reduction** |
| #5 RELIABILITY | No retry logic | No retry logic | ❌ No change |
| #6 AUDITABILITY | Basic logging | MCP tool call logging | ⚡ **Improved** |
| #7 OBSERVABILITY | Message structure only | MCP tool calls logged | ⚡ **Improved** |
| #8 DEPLOYABILITY | No automation | No automation | ❌ No change |

### What We Can Solve Now

✅ **Scenario 1: Add new supplier without rewriting agents**
```
Before Ch.2:
New supplier API → write custom wrapper → update 3 agents → deploy → test → 2 weeks

After Ch.2:
New supplier API → add to existing pricing-mcp-server → test server → deploy server → 2 days
Agents unchanged, automatically discover new supplier via tools/list

Result: ✅ 7× faster supplier onboarding ($14k engineering cost → $2k)
```

✅ **Scenario 2: Recover from supplier API schema change**
```
Before Ch.2:
TechFurnish changes API → 3 agents crash → emergency hotfix → 4-hour production outage

After Ch.2:
TechFurnish changes API → update pricing-mcp-server (single file) → agents reconnect → discover new schema → 15-minute fix

Result: ✅ 16× faster recovery (4 hours → 15 minutes), zero agent code changes
```

✅ **Scenario 3: Compliance audit of tool usage**
```
Before Ch.2:
CFO asks "Which agent approved PO #2024-1847?" → manually grep 8 agent logs → inconsistent formats → 2-hour forensics

After Ch.2:
Every MCP tools/call logged in standard JSON-RPC format → query audit DB:
  SELECT * FROM mcp_calls WHERE correlation_id = 'PO-2024-1847' AND tool_name = 'approve_purchase_order'

Result: ✅ Full decision chain reconstructed in 30 seconds
```

### MCP Servers Deployed

- `erp-server`: Inventory levels, PO history, approval workflows (Resources)
- `pricing-server`: `get_supplier_quote(item_id, quantity)` (Tool)
- `email-server`: `send_email()`, `fetch_inbox()` (Tools)
- `legal-server`: Contract templates, approval policies (Resources)
- `audit-proxy-server`: Logging proxy that records all tool calls to compliance database

### What's Still Blocking

❌ **Cross-service agent delegation**: Your Intake agent (Kubernetes Pod 1) receives PO #2024-1847. It needs to delegate pricing lookup to the Pricing agent (Pod 3) and negotiation to the Negotiation agent (Pod 5). MCP solves agent-to-tool communication, but you have no protocol for agent-to-agent task handoff. You're still hardcoding HTTP endpoints: `POST http://pricing-agent:8080/lookup`. When you add a 9th agent, you update 8 agent configs. **Constraint #4 SCALABILITY blocked: can't add agents without N² coordination.**

❌ **Synchronous bottleneck**: Intake agent calls Pricing agent (847ms supplier API call), waits for response, then calls Negotiation agent (12-second LLM call), waits, then calls Approval agent. Total: 2 min + 5 min + 8 min + 15 min + 2 min + 3 min = **35 minutes best-case, 36 hours with queue time**. **Constraint #1 THROUGHPUT blocked at 10 POs/day** (need 1,000 POs/day).

❌ **No shared state management**: Pricing agent and Negotiation agent both need to update the PO line items. Race condition: both agents read `quantity: 10`, Pricing agent updates to `quantity: 10, price: 789`, Negotiation agent updates to `quantity: 10, price: 749`, last write wins → Pricing agent's work silently lost. **Constraint #3 ACCURACY risk: data corruption in concurrent workflows**.

**Real-world status**: You can now add tools (suppliers, APIs, services) without rewriting agents. Error rate improved 3.8% → 3.2% (agents grounded in real data). But you're still at 10 POs/day throughput with 36-hour latency — agents can't delegate to each other across service boundaries.

**Next up:** [Ch.3 — Agent-to-Agent Protocol (A2A)](../ch03_a2a) gives us **agent delegation** — Intake agent discovers and calls Pricing agent via Agent Cards, task lifecycle (submitted → working → completed → failed), and SSE streaming for long-running tasks. Unlocks distributed multi-agent orchestration across Kubernetes cluster.

---

## § 9 · Bridge to Chapter 3

MCP collapsed agent-to-tool integration from N×M to N+M, but agent-to-agent coordination is still point-to-point. Ch.3 (A2A) defines the delegation protocol — Agent Cards advertise capabilities, task lifecycle tracks request→response, SSE streams long-running results → **enables hierarchical orchestration without hardcoded agent endpoints**.

---

## § 10 · The Math

### The N×M Integration Problem

Without a standard protocol, $N$ agents connecting to $M$ data sources require $N \times M$ bespoke integrations. MCP changes this to $N + M$ (each agent and server implements the standard once):

$$\text{Without MCP:} \quad \text{integrations} = N \times M$$
$$\text{With MCP:} \quad \text{integrations} = N + M$$

For OrderFlow: 8 agents × 20 tools = **160 custom integrations** without MCP → **28 implementations** with MCP (8 client adapters + 20 server adapters).

### Tool Schema as a Type Contract

Each MCP tool has a JSON Schema input specification. The agent sends:

$$\text{call} = \bigl\{\text{name}: t, \ \text{arguments}: \mathbf{a}\bigr\}$$

where $t$ is the tool name and $\mathbf{a}$ is a JSON object validated against the tool's schema $\mathcal{S}_t$. Validation passes iff $\mathbf{a} \models \mathcal{S}_t$.

The server returns a result object $r_t(\mathbf{a})$. The MCP protocol guarantees that the shape of $r_t$ is declared in the server's capability manifest, making the contract machine-verifiable.

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of MCP clients (agents) |
| $M$ | Number of MCP servers (tool providers) |
| $t$ | Tool name |
| $\mathbf{a}$ | Tool arguments (JSON object) |
| $\mathcal{S}_t$ | Tool's JSON Schema contract |
| $r_t(\mathbf{a})$ | Tool result for arguments $\mathbf{a}$ |



---

## § 11 · Where This Reappears

| Chapter | How MCP concepts appear |
|---------|------------------------|
| **Ch.1 — Message Formats** | MCP tool calls use the same `tool_calls`/`tool` role message format from Ch.1; MCP is the standardised envelope around function calling |
| **Ch.3 — A2A** | A2A agents can advertise their tools as MCP Resources; the two protocols are complementary — MCP for tool access, A2A for agent-to-agent task delegation |
| **Ch.7 — Agent Frameworks** | LangChain's `load_mcp_tools`, LangGraph's node-level tool access, and Semantic Kernel's plugin model all support MCP as the underlying tool discovery mechanism |
| **AI track — ReAct** | The ReAct Thought→Action→Observation loop calls MCP tools in the Action step; MCP is the production protocol that replaces hardcoded function schemas |
| **AI Infrastructure — Inference Optimization** | MCP servers can be deployed behind vLLM inference endpoints; the MCP client routes tool calls to the serving layer |

---

## Interview Questions

**Q: What are the three MCP primitive types and how do you decide which to use?**
**Resource** when the agent needs to read data without modifying it (a product catalogue, a database record). **Tool** when the agent needs to take an action or mutate state (send an email, write a record). **Prompt** when you have reusable, parameterised instruction templates that should live server-side rather than in agent code.

**Q: What problem does MCP solve that plain function calling does not?**
Discovery and reuse. With plain function calling, the agent must already know the tool's schema (hardcoded or injected via system prompt). With MCP, the server self-describes its capabilities at connection time. Any MCP-compliant agent can discover and use any MCP-compliant server without prior configuration — the `N × M` integration problem becomes `N + M`.

**Q: What is the difference between stdio and HTTP+SSE transports, and when would you choose each?**
`stdio` starts the MCP server as a subprocess and communicates via stdin/stdout — lowest latency, suitable for local tools (code execution, local file access), but limited to one client at a time. `HTTP + SSE` exposes the server over a network — supports multiple concurrent clients, can be scaled independently, required for remote services. Use stdio in development and for trusted local tools; use HTTP+SSE for production services.

**Q: Does MCP replace RAG?**
No. MCP exposes data as Resources that an agent can address by URI — it handles the access and delivery layer. The retrieval strategy (what to retrieve, how to chunk, how to rank), the vector database, and the embedding model are still entirely your responsibility. MCP and RAG are complementary: the RAG retrieval logic can be packaged as an MCP Tool.

---

## Notebook

`notebook.ipynb_solution.ipynb` (reference) or `notebook.ipynb_exercise.ipynb` (practice) implements:
1. A minimal MCP server exposing one Resource, one Tool, and one Prompt (using the `mcp` Python SDK)
2. An MCP client that discovers and calls them
3. The OrderFlow scenario: three MCP servers (ERP mock, pricing mock, email mock) called by a single orchestrator agent
4. The logging proxy: an MCP server that wraps another MCP server and records all calls

---

## Prerequisites

- [Ch.1 — Message Formats & Shared Context](../ch01_message_formats) — tool calls in the OpenAI message schema
- [AI / ReActAndSemanticKernel](../.03-ai/ch06_react_and_semantic_kernel/react-and-semantic-kernel.md) — the ReAct tool-use loop that MCP standardises

## Next

→ [Ch.3 — Agent-to-Agent Protocol (A2A)](../a2a) — how agents delegate tasks to other agents across service boundaries

## Illustrations

![MCP - N x M to N + M, handshake, primitives, transports](img/MCP.png)
