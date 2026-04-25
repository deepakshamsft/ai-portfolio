# Ch.2 — Model Context Protocol (MCP)

> **The story.** **Anthropic** announced the **Model Context Protocol** on **25 November 2024** as an open standard built on JSON-RPC 2.0. The motivating problem was the **N×M integration explosion** — every agent had to ship custom adapter code for every data source. MCP defined three reusable primitives — *Resources*, *Tools*, *Prompts* — plus standard transports (stdio for local, SSE/HTTP for remote). Adoption was unusually fast for a protocol: by mid-2025 OpenAI, Microsoft, and Google had all shipped MCP support; Claude Desktop, Cursor, Zed, and VS Code Copilot all spoke MCP natively; and the public registry had passed several thousand servers. **MCP is now the protocol for tool/data integration in multi-agent systems**, the same way HTTP is the protocol for hypertext.
>
> **Where you are in the curriculum.** [Ch.1](../message_formats) gave you the message envelope. This chapter answers: **what problem does MCP solve that plain function calling does not, and how does the JSON-RPC 2.0 protocol turn any data source or executable function into something any compliant agent can discover and use without bespoke adapter code?** Master this and the [A2A](../a2a) chapter — agent-to-agent delegation — will compose cleanly with it.
**Notation.** `MCP` = Model Context Protocol (Anthropic, November 2024). `JSON-RPC 2.0` = the wire transport format (`method`, `params`, `id`, `result` / `error`). `Resource` = read-only data exposed by an MCP server. `Tool` = executable function exposed by an MCP server. `Prompt` = reusable prompt template registered with an MCP server. `SSE` = Server-Sent Events (HTTP streaming transport for remote MCP servers). `stdio` = standard-input/output transport for local in-process MCP servers.

---

## § 0 · The Challenge — Where We Are

> 🎯 **The mission**: Build **OrderFlow** — AI-native B2B purchase order automation satisfying 8 constraints:
> 1. **THROUGHPUT**: 1,000 POs/day — 2. **LATENCY**: <4hr SLA — 3. **ACCURACY**: <2% error — 4. **SCALABILITY**: 10 agents/PO — 5. **RELIABILITY**: >99.9% uptime — 6. **AUDITABILITY**: Full traceability — 7. **OBSERVABILITY**: Real-time monitoring — 8. **DEPLOYABILITY**: Zero-downtime updates

**After Ch.1**: Decomposed single agent into 8 specialized agents (Intake, Pricing, Negotiation, Legal, Finance, Drafting, Sending, Reconciliation). Error rate dropped 5% → 3.8%. Context overflow eliminated.

### The Blocking Question This Chapter Solves

**"How do 8 agents access ERP, pricing APIs, email without writing 8 × 20 = 160 custom integrations?"**

Each agent needs access to ~20 data sources. Building custom adapters = **8 agents × 20 integrations = 160 bespoke implementations**. Unmaintainable, untestable, unscalable.

### What We Unlock in This Chapter

- ✅ Understand MCP protocol: Resources (read-only data), Tools (callable functions), Prompts (templated instructions)
- ✅ JSON-RPC 2.0 transport: Any agent connects to any data source without custom code
- ✅ Integration collapse: **160 integrations → 8 MCP clients + 20 MCP servers = 28 components**

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

## The N×M Integration Problem

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

---

## Protocol Mechanics

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

## MCP vs Plain Function Calling

| Dimension | Plain function calling | MCP |
|-----------|----------------------|-----|
| Schema discovery | Schema is hardcoded in the client or passed in the system prompt | Schema is declared by the server at runtime via `tools/list` |
| Cross-agent reuse | The wrapper is coupled to the agent framework | Any MCP client can call any MCP server regardless of framework |
| Versioning | Client breaks silently if the tool's signature changes | Server negotiates protocol version during handshake; schema is always current |
| Observability | Tool calls are opaque unless the agent framework provides hooks | Every MCP call is a JSON-RPC request; standard middleware can intercept and log |
| Authentication | Ad hoc per integration | Standardised — OAuth 2.0 bearer tokens for HTTP transport; process-level trust for stdio |

---

## 2 · Running Example

OrderFlow's tool sprawl had become a maintenance bottleneck: three teams, each owning one agent, each maintaining their own wrappers for the ERP, pricing API, and email service — nine wrappers total, each with subtly different error handling.

The fix: three MCP servers, one per system (ERP, Pricing, Email). All agents become MCP clients. Each team now owns only one server and zero agent-side adapters.

The additional gain: the compliance team added a logging proxy MCP server that sits in front of every tool call and records to the audit database. No agent code was modified — the proxy is just another MCP server that passes calls through.

---


## 3 · How It Works

> Step-by-step walkthrough of the mechanism.


## 4 · Key Diagrams

> Add 2–3 diagrams showing the key data flows here.


## 5 · Hyperparameter Dial

> List the key knobs and their effect on behaviour.


## 6 · What Can Go Wrong

> 3–5 common failure modes and mitigations.

## 7 · Progress Check — What We Achieved

```mermaid
graph LR
    Ch1["Ch.1\nMessage Formats"]:::done
    Ch2["Ch.2\nMCP"]:::done
    Ch3["Ch.3\nA2A"]:::done
    Ch4["Ch.4\nEvent-Driven"]:::done
    Ch5["Ch.5\nShared Memory"]:::done
    Ch6["Ch.6\nTrust & Sandboxing"]:::done
    Ch7["Ch.7\nAgent Frameworks"]:::done
    Ch1 --> Ch2 --> Ch3 --> Ch4 --> Ch5 --> Ch6 --> Ch7
    classDef done fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef current fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef upcoming fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```

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

### The Win

✅ **Integration collapse**: Reduced 160 bespoke integrations to **28 components** (8 MCP clients + 20 MCP servers). Any agent can now access any data source through JSON-RPC 2.0 protocol.

**Measured impact**: Error rate dropped 3.8% → 3.2% (agents now grounded in real-time ERP inventory, live pricing APIs, actual supplier emails — no hallucinated data).

### MCP Servers Deployed

- `erp-server`: Inventory levels, PO history, approval workflows (Resources)
- `pricing-server`: `get_supplier_quote(item_id, quantity)` (Tool)
- `email-server`: `send_email()`, `fetch_inbox()` (Tools)
- `legal-server`: Contract templates, approval policies (Resources)

### What's Still Blocking

**Cross-service agent delegation**: Intake agent (Pod 1) can't delegate to Negotiation agent (Pod 3) across Kubernetes cluster. No standard protocol for agent-to-agent task handoff.

**Next unlock** *(Ch.3 — A2A)*: Agent-to-Agent protocol enables cross-service delegation via Agent Cards, task lifecycle (submitted → working → completed), and SSE streaming.

---

## 8 · The Math

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

## Code Skeleton

```python
# Educational: minimal MCP client from scratch (stdio transport)
import json, subprocess
from typing import Any

class MCPClient:
    """
    Minimal MCP stdio client — illustrates the JSON-RPC 2.0 message exchange.
    Production: use the official `mcp` Python library instead.
    """
    def __init__(self, command: list[str]):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True)
        self._request_id = 0

    def _call(self, method: str, params: dict) -> Any:
        self._request_id += 1
        msg = {"jsonrpc": "2.0", "id": self._request_id, "method": method, "params": params}
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        response = json.loads(self.proc.stdout.readline())
        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")
        return response["result"]

    def list_tools(self) -> list:
        return self._call("tools/list", {})["tools"]

    def call_tool(self, name: str, arguments: dict) -> Any:
        return self._call("tools/call", {"name": name, "arguments": arguments})
```

```python
# Production: MCP client with the official SDK + LangChain integration
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def build_orderflow_mcp_agent():
    """
    Connect OrderFlow agents to all 20 tool servers via MCP.
    Each agent gets its own subset of tools (least-privilege).
    """
    server_params = StdioServerParameters(
        command="uvx", args=["mcp-server-orderflow"]  # OrderFlow MCP server
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            # Filter to only tools this agent needs (least-privilege)
            pricing_tools = [t for t in tools if t.name in ("get_quote", "compare_suppliers")]
            agent = create_react_agent(ChatOpenAI(model="gpt-4o"), pricing_tools)
            return agent
```

---

## Where This Reappears

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

`notebook.ipynb` implements:
1. A minimal MCP server exposing one Resource, one Tool, and one Prompt (using the `mcp` Python SDK)
2. An MCP client that discovers and calls them
3. The OrderFlow scenario: three MCP servers (ERP mock, pricing mock, email mock) called by a single orchestrator agent
4. The logging proxy: an MCP server that wraps another MCP server and records all calls

---

## Prerequisites

- [Ch.1 — Message Formats & Shared Context](../message_formats) — tool calls in the OpenAI message schema
- [AI / ReActAndSemanticKernel](../../ai/react_and_semantic_kernel/react-and-semantic-kernel.md) — the ReAct tool-use loop that MCP standardises

## Next

→ [Ch.3 — Agent-to-Agent Protocol (A2A)](../a2a) — how agents delegate tasks to other agents across service boundaries

## Illustrations

![MCP - N x M to N + M, handshake, primitives, transports](img/MCP.png)
