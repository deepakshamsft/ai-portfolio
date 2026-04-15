# Ch.7 — Agent Frameworks

> **Central question:** AutoGen, LangGraph, and Semantic Kernel AgentGroupChat all express multi-agent coordination — so how are they architecturally different, when does each pattern apply, and how do the primitives from previous chapters (message formats, MCP, trust) compose with each?

---

## Why Framework Choice is Not Trivial

A common mistake: treat framework selection as a dependency choice (pick one, stick with it). The frameworks have fundamentally different execution models that impose different constraints on your system. Picking the wrong one for your control flow requirements costs more to undo than it would have cost to understand the tradeoffs upfront.

The three frameworks covered here sit at different points on two axes:

```
                      CONTROL FLOW
                  Explicit ◄──────► Emergent
                       │               │
                       │               │
COUPLING:       LangGraph           AutoGen
Graph-first              SK Group      (conversation-first)
(deterministic)          Chat          (agent decides)
                       │               │
                       │               │
               Tight (graph          Loose (agents
               enforces order)        negotiate)
```

Understanding where your use case sits on these axes is the decision:
- Known, fixed workflow → LangGraph
- Open-ended, emergent dialogue → AutoGen
- Enterprise pipeline with hooks and compliance → Semantic Kernel

---

## AutoGen — Conversation-First

AutoGen (Microsoft Research) models multi-agent interaction as a conversation between `ConversableAgent` objects. Agents speak to each other in rounds; the flow emerges from the conversation rather than being defined in advance.

### Two-Agent Pattern (Proposer + Critic)

```python
from autogen import ConversableAgent

llm_config = {"config_list": [{"model": "gpt-4o", "api_key": os.environ["OPENAI_API_KEY"]}]}

pricing_proposer = ConversableAgent(
    name="PricingProposer",
    system_message="""You are a procurement specialist. Propose a purchase price for
    the given item based on market data. Wait for the critic's feedback before finalising.""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

pricing_critic = ConversableAgent(
    name="PricingCritic",
    system_message="""You are a financial risk officer. Critique the proposed price.
    If the price is within 5% of the 90-day average, output APPROVED. Otherwise, push back.""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Initiate conversation — flow is emergent
result = pricing_critic.initiate_chat(
    recipient=pricing_proposer,
    message="We need a price for 500 units of widget SKU-8812.",
    max_turns=6
)
```

The conversation continues until `APPROVED` appears or `max_turns` is reached. Neither agent calls the other directly — they exchange messages and the AutoGen runtime arbitrates turns.

### Group Chat (Multiple Agents)

```python
from autogen import GroupChat, GroupChatManager

negotiation_agent = ConversableAgent(name="Negotiation", ...)
legal_agent = ConversableAgent(name="Legal", ...)
finance_agent = ConversableAgent(name="Finance", ...)

group_chat = GroupChat(
    agents=[negotiation_agent, legal_agent, finance_agent],
    messages=[],
    max_round=12,
    speaker_selection_method="auto"  # LLM selects next speaker dynamically
)

manager = GroupChatManager(groupchat=group_chat, llm_config=llm_config)

# The manager routes messages — the order is NOT predetermined
negotiation_agent.initiate_chat(manager, message="We have a PO to negotiate.")
```

`speaker_selection_method="auto"` means the GroupChatManager uses an LLM to decide who speaks next. This is powerful but non-deterministic — the same input can produce different agent orderings.

---

## LangGraph — Graph-First

LangGraph (LangChain Labs) models agent coordination as an explicit directed graph. Nodes are functions (agents, tools); edges define allowed transitions. The execution order is deterministic and inspectable before the graph runs.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class POWorkflowState(TypedDict):
    po_id: str
    inventory_ok: bool
    negotiation_result: dict
    approved: bool
    po_document_url: str

def run_inventory_check(state: POWorkflowState) -> POWorkflowState:
    # Call inventory agent / MCP server
    result = inventory_mcp_client.check(state["po_id"])
    return {**state, "inventory_ok": result["available"]}

def run_negotiation(state: POWorkflowState) -> POWorkflowState:
    result = negotiation_agent.run(state["po_id"])
    return {**state, "negotiation_result": result}

def route_after_inventory(state: POWorkflowState) -> Literal["negotiate", "reject"]:
    return "negotiate" if state["inventory_ok"] else "reject"

# Build the graph explicitly
workflow = StateGraph(POWorkflowState)
workflow.add_node("inventory", run_inventory_check)
workflow.add_node("negotiate", run_negotiation)
workflow.add_node("approve", run_approval)
workflow.add_node("draft_po", run_po_drafting)
workflow.add_node("reject", run_rejection)

workflow.set_entry_point("inventory")
workflow.add_conditional_edges("inventory", route_after_inventory)
workflow.add_edge("negotiate", "approve")
workflow.add_edge("approve", "draft_po")
workflow.add_edge("draft_po", END)

app = workflow.compile()
```

The graph is an explicit declaration of what can follow what. You can visualise it, test it, and reason about it statically. Non-determinism is only in the *logic* inside each node, not in the control flow between nodes.

---

## Semantic Kernel AgentGroupChat — Enterprise-First

Semantic Kernel (Microsoft) is designed for enterprise workloads that need production hooks: filter pipelines, observability middleware, compliance constraints, and integration with Azure AI and Azure OpenAI Service.

```python
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import TerminationStrategy, SelectionStrategy
from semantic_kernel import Kernel

kernel = Kernel()

negotiation_agent = ChatCompletionAgent(
    kernel=kernel,
    name="NegotiationAgent",
    instructions="You negotiate purchase order terms with suppliers..."
)

approval_agent = ChatCompletionAgent(
    kernel=kernel,
    name="ApprovalAgent",
    instructions="You review negotiated terms and approve or reject..."
)

class ApprovalReachedTermination(TerminationStrategy):
    async def should_agent_terminate(self, agent, history):
        return any("APPROVED" in m.content for m in history[-2:])

group_chat = AgentGroupChat(
    agents=[negotiation_agent, approval_agent],
    termination_strategy=ApprovalReachedTermination(maximum_iterations=8)
)

await group_chat.add_chat_message(
    ChatMessageContent(role=AuthorRole.USER, content="Process PO-4812.")
)

async for message in group_chat.invoke():
    print(f"{message.name}: {message.content}")
```

SK's key enterprise-specific features:
- **Filter pipeline:** pre/post hooks on every function invocation for logging, PII scrubbing, cost tracking
- **Telemetry:** OpenTelemetry-compatible tracing with Azure Monitor integration out of the box  
- **Plugin system:** SK plugins map directly to MCP tools — an MCP server can be registered as an SK plugin
- **`TerminationStrategy`:** explicit, testable exit conditions rather than `max_turns` or LLM-decided termination

---

## Framework Comparison

| Dimension | AutoGen | LangGraph | Semantic Kernel |
|-----------|---------|-----------|-----------------|
| **Execution model** | Message-passing between agent objects; turn-taking via `GroupChatManager` | Directed graph; conditional edges define control flow deterministically | Conversation with pluggable strategies, filter pipeline, telemetry hooks |
| **Control flow** | Emergent — agents negotiate who speaks next | Explicit — graph topology defines allowed transitions | Semi-explicit — termination and selection strategies are code, not graph |
| **Determinism** | Low — same input can produce different agent orderings | High — graph structure is fixed; only node internals vary | Medium — strategies are deterministic; agent content is not |
| **Debugging** | Trace through conversation history; no visual graph | Visualise the graph; breakpoint at any node | Filter hooks capture every invocation; Azure Monitor for production |
| **MCP integration** | Via tool registration on `ConversableAgent` | Via LangChain-MCP adapter or direct tool node | Via SK MCP plugin connector (native) |
| **Best for** | Open-ended research, debate patterns, rapid prototyping | Production pipelines with known control flow, compliance-required determinism | Enterprise Azure deployments, teams that need audit hooks and telemetry |
| **Avoid if** | Your workflow has strict ordering requirements (use LangGraph) | Your workflow is genuinely open-ended (the graph becomes a spaghetti of edges) | Framework overhead is unjustified for simple pipelines (use AutoGen or raw) |

---

## When Each Pattern Wins

| Use case | Best choice | Reason |
|----------|-------------|--------|
| Pricing debate (proposer + critic) | AutoGen two-agent | Emergent conversation is natural; termination is criteria-based |
| Regulatory document review (fixed sequence of specialist reviewers) | LangGraph | Explicit ordering is a compliance requirement, not a preference |
| Document approval with audit log for SOC 2 | Semantic Kernel | Filter hooks provide the audit trail; Azure Monitor integration |
| Research agent (web search → summarise → critique → refine) | AutoGen GroupChat or LangGraph | Depends: if the research path varies, AutoGen; if it is always the same steps, LangGraph |
| Multi-modal pipeline (image → OCR → classify → route) | LangGraph | Branching on image type is deterministic conditional-edge logic |
| High-compliance financial workflow (PO creation) | Semantic Kernel + LangGraph | SK for hooks and telemetry; LangGraph for control flow within the SK kernel |

---

## OrderFlow — Ch.7 Scenario

OrderFlow used LangGraph for its production PO lifecycle (deterministic compliance requirement: always inventory before negotiation before approval before drafting, no exceptions). But the pricing team wanted to experiment with multi-agent debate for pricing decisions without rebuilding the production graph.

The solution: a standalone AutoGen two-agent debate (`PricingProposer` + `PricingCritic`) was deployed as a microservice. The LangGraph negotiation node calls this microservice and gets back a consensus price. The AutoGen conversation is internal to the microservice; LangGraph sees it as just another tool call.

This is the practical lesson: AutoGen and LangGraph are not mutually exclusive. An AutoGen debate loop can be a *node* in a LangGraph graph, or a *tool* accessible via MCP.

---

## Interview Questions

**Q: When would you use AutoGen over LangGraph?**
When the control flow is genuinely open-ended or emergent — when you do not know in advance which agent should speak next or how many rounds the task will take. AutoGen's conversation model is well-suited to debate patterns (proposer-critic), research (search-summarise-critique-refine), and exploratory tasks. LangGraph is better when the workflow is known and fixed — when you need deterministic control flow, conditional branching by explicit criteria, or regulatory compliance that requires a guaranteed execution order.

**Q: Can you use AutoGen and LangGraph together in the same system?**
Yes. An AutoGen conversation can be encapsulated as a function/node inside a LangGraph graph — LangGraph controls the overall pipeline; AutoGen handles open-ended sub-tasks within it. They are not mutually exclusive and often complement each other: LangGraph for the outer deterministic orchestration, AutoGen for inner emergent reasoning steps.

**Q: What does Semantic Kernel add beyond what AutoGen or LangGraph provide?**
Production hooks: a filter pipeline for pre/post-processing every function invocation (audit logs, PII scrubbing, cost tracking), OpenTelemetry-compatible telemetry pluggable into Azure Monitor, explicit `TerminationStrategy` and `SelectionStrategy` as testable code objects rather than heuristics, and native MCP plugin integration. It is designed for enterprise deployments where the conversation itself is not the hard part — the compliance, auditability, and operational observability are.

**Q: How does MCP interact with AutoGen, LangGraph, and SK?**
In all three, MCP tools appear as callables that the agent framework can invoke. AutoGen: register the MCP tool on a `ConversableAgent`'s tool list. LangGraph: wrap the MCP client call in a node function or use a LangChain-MCP adapter as a tool. Semantic Kernel: use the SK MCP plugin connector to register an MCP server as an SK plugin — SK's function-calling infrastructure then handles invocation, result parsing, and sending back to the model.

---

## Notebook

`notebook.ipynb` implements:
1. AutoGen two-agent debate: `PricingProposer` + `PricingCritic` for OrderFlow pricing approval
2. LangGraph PO pipeline: 5-node graph (inventory → negotiate → approve → draft → end) with conditional edge on inventory failure
3. Semantic Kernel `AgentGroupChat` with `ApprovalReachedTermination` and a mock filter hook that logs every agent invocation to stdout
4. Composition: AutoGen debate encapsulated as a LangGraph node

---

## Prerequisites

- All prior chapters — this chapter assumes all the primitives are understood: message formats (Ch.1), MCP (Ch.2), A2A (Ch.3), event-driven (Ch.4), shared memory (Ch.5), trust (Ch.6)
- [AI / ReActAndSemanticKernel](../AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md) — SK plugin basics

## This is the Final Chapter in the Track

← Return to [README](../README.md) for the full reading path and cross-track connections.
