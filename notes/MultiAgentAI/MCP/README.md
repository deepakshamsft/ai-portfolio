# Ch.2 — Model Context Protocol (MCP)

> **Central question:** What problem does MCP solve that plain function calling does not, and how does the JSON-RPC 2.0 protocol turn any data source or executable function into something any compliant agent can discover and use without bespoke adapter code?

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

## OrderFlow — Ch.2 Scenario

OrderFlow's tool sprawl had become a maintenance bottleneck: three teams, each owning one agent, each maintaining their own wrappers for the ERP, pricing API, and email service — nine wrappers total, each with subtly different error handling.

The fix: three MCP servers, one per system (ERP, Pricing, Email). All agents become MCP clients. Each team now owns only one server and zero agent-side adapters.

The additional gain: the compliance team added a logging proxy MCP server that sits in front of every tool call and records to the audit database. No agent code was modified — the proxy is just another MCP server that passes calls through.

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

- [Ch.1 — Message Formats & Shared Context](../MessageFormats/) — tool calls in the OpenAI message schema
- [AI / ReActAndSemanticKernel](../AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md) — the ReAct tool-use loop that MCP standardises

## Next

→ [Ch.3 — Agent-to-Agent Protocol (A2A)](../A2A/) — how agents delegate tasks to other agents across service boundaries

## Illustrations

![MCP - N x M to N + M, handshake, primitives, transports](img/MCP.png)
