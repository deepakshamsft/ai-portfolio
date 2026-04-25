# Plan — Multi-Agent AI Track

**Last updated:** 2026-04-24
**Audit scope:** All 7 chapters under `notes/multi_agent_ai/`
**Running example:** OrderFlow (B2B purchase-order automation, 1,000 POs/day, <4hr SLA)

## Legend
- 🐍 = Python script needed — listed in Scripts table, generated in one batch pass
- ✏️ = Manual content edit — implemented directly by Copilot
- ✅ = Done

---

## Track-Wide Context

All 7 chapters have strong technical content but share the same structural gaps — the universal numbered section spine is absent. Every chapter is missing:
- Numbered section headers (`## 1 · Core Idea`, `## 2 · Running Example`, etc.)
- `## Animation` section with needle GIF
- `## 3 · The Math` section (formal scalar-first formula with verbal gloss)
- `## 5 · Key Diagrams` as a dedicated section
- `## 6 · Hyperparameter Dial`
- `## 7 · Code Skeleton` with Educational/Production labels
- `## 8 · What Can Go Wrong`
- `## Where This Reappears` / forward links
- Mermaid arc in Progress Check
- `img/` folder
- Notation sentence in blockquote header

---

## 🐍 Script Todos

<!-- SCRIPTS_HOOK_START
track: multi_agent_ai
-->

| Script | File(s) | What it does |
|--------|---------|--------------|
| `scripts/create_multi_agent_img_dirs.py` | All 7 chapter directories | Create `img/` directory in each chapter folder |
| `scripts/fix_multi_agent_section_stubs.py` | All 7 `README.md` files | (1) Insert `## Animation\n\n> 🎬 *Animation placeholder — needle GIF to be generated.*\n` after `## 0 · The Challenge` closing line; (2) Rename `## Core Concepts` → `## 1 · Core Idea`; (3) Rename `## OrderFlow — Ch.N Scenario` → `## 2 · Running Example`; (4) Append stubs for `## 3 · The Math`, `## 4 · How It Works`, `## 5 · Key Diagrams`, `## 6 · Hyperparameter Dial`, `## 7 · Code Skeleton`, `## 8 · What Can Go Wrong`, `## Where This Reappears` before the `## Progress Check` section |
| `scripts/fix_multi_agent_progress_check_heading.py` | All 7 `README.md` files | Normalize `## § 11.5 · Progress Check` → `## 11 · Progress Check` for consistency with universal standard |
| `scripts/add_multi_agent_mermaid_arc_stubs.py` | All 7 `README.md` files | Append a 7-chapter Mermaid `graph LR` stub to each `## 11 · Progress Check` section |
| `scripts/add_multi_agent_notation_placeholders.py` | All 7 `README.md` files | Insert `<!-- TODO: notation sentence — define symbols used in chapter -->` at end of opening blockquote |

<!-- SCRIPTS_HOOK_END -->

---

## ✏️ Manual Content Todos

### Track-wide (all 7 chapters)

- [ ] **Generate Animation GIFs** — one per chapter showing OrderFlow constraint needle movement
- [x] ✅ **Write `## 3 · The Math`** for each chapter — done for all 7 (token budget, N×M reduction, FSM lifecycle, Little's Law, CAS semantics, HMAC formula, LangGraph state machine)
- [ ] **Write `## 6 · Hyperparameter Dial`** for each chapter
- [x] ✅ **Write `## 7 · Code Skeleton`** for each chapter — Educational + Production pairs added to all 7
- [ ] **Write `## 8 · What Can Go Wrong`** for each chapter
- [x] ✅ **Write `## Where This Reappears`** forward links for each chapter — done for all 7
- [ ] **Write notation sentences** for all blockquote headers
- [ ] **Write Mermaid chapter arcs** for all Progress Check sections

### message_formats
- [x] ✅ Math: token budget formula — $C_{\text{total}} = \sum c_i \leq C_{\max}$; context budget allocation
- [x] ✅ Code Skeleton: Educational (message handoff strategies) + Production (typed AgentContext with Pydantic)
- [x] ✅ Where This Reappears: Ch.2, Ch.5, Ch.7, AI track
- [ ] Hyperparameter Dial: context budget split (system prompt vs tool-call reservation vs response budget)
- [ ] What Can Go Wrong: context overflow, schema mismatch, budget exhaustion

### mcp
- [x] ✅ Math: N×M integration problem → N+M with MCP
- [x] ✅ Code Skeleton: Educational (minimal MCPClient stdio) + Production (official SDK + LangChain)
- [x] ✅ Where This Reappears: Ch.3, Ch.6, Ch.7, AI track
- [ ] Hyperparameter Dial: MCP server pool size, per-server timeout, retry policy
- [ ] What Can Go Wrong: server discovery failure, transport protocol mismatch, schema version drift

### a2a
- [x] ✅ Math: task lifecycle FSM over states $\{submitted, working, completed, failed, cancelled\}$; SLA compliance probability
- [x] ✅ Code Skeleton: Educational (A2AClientSimple) + Production (official SDK with SSE streaming)
- [x] ✅ Where This Reappears: Ch.1, Ch.2, Ch.4, Ch.7, Trust & Sandboxing
- [ ] Hyperparameter Dial: SSE reconnect timeout, max retry count, `acceptedOutputModes` strategy
- [ ] What Can Go Wrong: SSE drop mid-task, partial AgentCard claims, task expiry
- [ ] Normalize `## § 11.5 · Progress Check` heading *(also covered by script)*

### event_driven_agents
- [x] ✅ Math: Little's Law $L = \lambda W$; fan-out merge cost $T_{\text{fanout}} = \max_i T_i + T_{\text{merge}}$
- [x] ✅ Code Skeleton: Educational (SimpleEventBus) + Production (Redis Streams with consumer groups)
- [x] ✅ Where This Reappears: Ch.1, Ch.3, Ch.5, Ch.7, Trust
- [ ] Move existing ASCII pub/sub diagram into a dedicated `## 5 · Key Diagrams` section
- [ ] Hyperparameter Dial: partition count, consumer group lag threshold, DLQ max retries
- [ ] What Can Go Wrong: consumer lag accumulation, DLQ overflow, poison message loops

### shared_memory
- [x] ✅ Math: optimistic CAS semantics; conflict rate formula; memory scope hierarchy
- [x] ✅ Code Skeleton: Educational (in-memory dict blackboard) + Production (Redis-backed with TTL and distributed locking)
- [x] ✅ Where This Reappears: Ch.1, Ch.4, Ch.6, Ch.7, AI track Evaluating
- [ ] Move existing ASCII blackboard diagram into `## 5 · Key Diagrams`
- [ ] Hyperparameter Dial: TTL per key, write-lock timeout, cache eviction policy
- [ ] What Can Go Wrong: stale read after TTL miss, namespace collision, lock starvation

### trust_and_sandboxing
- [x] ✅ Math: HMAC formula $H = \text{HMAC-SHA256}(K, m)$; structured output validation reject rate $r$
- [x] ✅ Code Skeleton: Educational (HMAC signing + JSON Schema validation) + Production (FastAPI trust middleware)
- [x] ✅ Where This Reappears: Ch.1, Ch.2, Ch.3, AI Safety, OWASP LLM Top 10
- [ ] Hyperparameter Dial: token TTL, approval threshold ($ value), sandbox memory/CPU cap
- [ ] What Can Go Wrong: expired token reuse, prompt injection propagation, sandbox escape via shared filesystem

### agent_frameworks
- [x] ✅ Math: LangGraph state machine $G = (V, E)$; token budget per graph run $T_{\text{run}} = \sum_{v_i \in \tau} c_i$
- [x] ✅ Code Skeleton: Educational (minimal state machine) + Production (LangGraph with checkpointing + LangSmith)
- [x] ✅ Where This Reappears: Ch.1, Ch.2, Ch.3, Ch.5, Ch.6
- [ ] Move `## Framework Comparison` table to `## 5 · Key Diagrams`
- [ ] Rename `## When Each Pattern Wins` → `## 8 · What Can Go Wrong` or add proper section
- [ ] Hyperparameter Dial: graph max-steps, checkpoint interval, blue-green rollout %, memory backend

---

## Notes
- Technical content depth is high across all 7 chapters — the gaps are structural, not content quality.
- The scripts in `notes/multi_agent_ai/scripts/` may already have animation infrastructure — check before writing new generators.
