# Agentic AI Track — Implementation Plan

> **Status:** Ch.1-10 complete. Ch.11 (Advanced Agentic Patterns) planned.  
> **Last updated:** April 26, 2026  
> **Focus:** Add theoretical agentic AI design patterns (reflection, debate, orchestration) with rich animations

---

## Current Status

| Chapter | Status | README | Notebook | Animations |
|---------|--------|--------|----------|------------|
| Ch.1: LLM Fundamentals | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.2: Prompt Engineering | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.3: Chain-of-Thought Reasoning | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.4: RAG & Embeddings | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.5: Vector Databases | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.6: ReAct & Semantic Kernel | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.7: Safety & Hallucination | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.8: Evaluating AI Systems | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.9: Cost & Latency Optimization | ✅ Complete | ✅ | ✅ | ✅ |
| Ch.10: Fine-Tuning | ✅ Complete | ✅ | ✅ | ✅ |
| **Ch.11: Advanced Agentic Patterns** | ⏳ **Planned** | ❌ | ❌ | ❌ |

---

## Ch.11 — Advanced Agentic Patterns: Reflection, Debate, and Orchestration

### Overview

**Running example:** PizzaBot v2.0 — Advanced order handling with self-critique, multi-agent debate, and hierarchical orchestration  
**Constraint:** Handle edge cases (ambiguous orders, menu conflicts, pricing disputes) with <1% error rate  
**Patterns covered:**
1. **Reflection** (self-critique and iterative refinement)
2. **Debate & Consensus** (multi-agent reasoning)
3. **Hierarchical Orchestration** (planner → worker → verifier)
4. **Tool Selection Strategies** (when to use which tool, error recovery)

**What's missing in current track:**

Current PizzaBot (Ch.1-10) can handle typical orders:
- ✅ RAG grounding eliminates menu hallucinations (Ch.4)
- ✅ ReAct orchestration enables tool-calling (Ch.6)
- ✅ Safety guards prevent prompt injection (Ch.7)
- ✅ Cost/latency optimization hits production targets (Ch.9)

**But edge cases still fail:**

```
Customer: "I want a large pepperoni pizza, but make it gluten-free,
           dairy-free, and add extra cheese."

PizzaBot v1.0 (Ch.10):
❌ Contradiction detected but no recovery path
❌ Outputs: "Error: gluten-free + dairy-free + extra cheese conflict"
❌ Customer abandons order

PizzaBot v2.0 (Ch.11 - with Reflection):
✅ Step 1: Draft response: "We can't add dairy cheese to a dairy-free pizza"
✅ Step 2: Self-critique: "User wants dairy-free but said 'extra cheese' — 
           check if vegan cheese is an option"
✅ Step 3: Revised response: "For dairy-free, we use vegan mozzarella. 
           Would you like extra vegan cheese (+$2.50)?"
✅ Customer: "Yes, perfect!"
✅ Order completed
```

**Why this matters:**

| Metric | Ch.10 (no patterns) | Ch.11 (with patterns) | Impact |
|--------|---------------------|----------------------|--------|
| **Edge case error rate** | 8% | <1% | ✅ 8× improvement |
| **Customer satisfaction** | 4.2/5 | 4.7/5 | ✅ +0.5 stars |
| **Escalation to human** | 12% | 3% | ✅ 4× reduction |
| **Complex order conversion** | 15% | 32% | ✅ 2× improvement |

---

## Implementation Tasks

### ✅ Prerequisites

- [ ] **Review existing mentions**
  - Audit `ch02_prompt_engineering/prompt-engineering.md` §8 (self-critique)
  - Audit `ch03_cot_reasoning/cot-reasoning-supplement.md` (Reflexion pattern)
  - Audit `ch06_react_and_semantic_kernel/react-and-semantic-kernel-supplement.md` (debate pattern)
  - Extract core concepts, expand into full theoretical treatment

- [ ] **Set up animation infrastructure**
  - Create `notes/02-ai/ch11_advanced_agentic_patterns/` folder
  - Create `img/` subfolder for generated animations
  - Create `gen_scripts/` subfolder for animation generators

### 📝 Chapter Content

- [ ] **Ch.11 README.md**
  - Structure: Follow Ch.1-10 pattern (Challenge → Core Idea → Running Example → etc.)
  - Sections:
    1. **The Challenge:** Edge cases where single-pass reasoning fails
    2. **Core Idea:** Iterative refinement > one-shot prediction
    3. **Pattern 1: Reflection (Self-Critique)**
       - Generate → Critique → Revise loop
       - When to use: Ambiguous inputs, complex reasoning, high-stakes outputs
       - Running example: PizzaBot handling contradictory order (gluten-free + extra cheese)
       - Cost model: 3× tokens vs. single-pass
       - Pitfall: Model can hallucinate that its hallucination is correct (need external grounding)
    4. **Pattern 2: Debate & Consensus (Multi-Agent Reasoning)**
       - Propose → Challenge → Defend → Vote loop
       - When to use: High-stakes decisions (medical diagnosis, legal reasoning, fraud detection)
       - Running example: PizzaBot pricing dispute (coupon + loyalty + promo conflicts)
       - Agents: Pricer1 (strict), Pricer2 (generous), Judge (arbiter)
       - Cost model: N agents × tokens per round
       - Pitfall: Agents agree on wrong answer (groupthink)
    5. **Pattern 3: Hierarchical Orchestration (Planner → Workers → Verifier)**
       - Decompose → Execute → Verify loop
       - When to use: Complex multi-step tasks (research, code generation, data pipelines)
       - Running example: PizzaBot catering order (15 pizzas, 3 delivery times, budget constraint)
       - Agents: Planner (split into subtasks), Workers (execute), Verifier (check constraints)
       - Cost model: 1 planner call + N worker calls + 1 verifier call
       - Pitfall: Workers drift from plan (need coordination protocol)
    6. **Pattern 4: Tool Selection Strategies**
       - When to use: Multiple tools available, need to pick best one
       - Strategies: Rule-based (if-then), Cost-based (cheapest first), LLM-based (meta-agent)
       - Running example: PizzaBot inventory check (DB vs. API vs. cached estimate)
       - Error recovery: Fallback chain (try fast tool → if fails, try expensive tool)
    7. **Mental Model:** Patterns = trading tokens for reliability
    8. **Code Skeleton:** LangGraph state machines, custom reflection loop
    9. **What Can Go Wrong:**
       - Over-iteration (10 refinement loops for simple query)
       - Debate deadlock (agents never converge)
       - Hierarchical drift (workers ignore planner)
       - Tool thrashing (retry loop with no convergence)
    10. **Progress Check:** Given a use case, pick the right pattern
    11. **Bridge to Multi-Agent Track:** These patterns scale to multi-agent systems (see `notes/multi_agent_ai/`)

- [ ] **Ch.11 notebook.ipynb**
  - Cell 1: Setup (LangChain, LangGraph, OpenAI API)
  - Cell 2: **Reflection pattern** — Generate → Critique → Revise loop
    - Draft response to ambiguous order
    - Self-critique: Identify contradictions
    - Revise: Resolve contradictions with clarifying question
  - Cell 3: **Debate pattern** — 2 agents propose solutions, 1 judge decides
    - Scenario: Pricing conflict (coupon + loyalty discount stack?)
    - Agent 1: "Apply both discounts" (generous)
    - Agent 2: "Apply only one discount" (strict)
    - Judge: Picks based on company policy
  - Cell 4: **Hierarchical orchestration** — Planner → Workers → Verifier
    - Scenario: Catering order (15 pizzas, 3 delivery slots)
    - Planner: Split into 3 batches (5 pizzas each)
    - Workers: Process each batch
    - Verifier: Check total cost < budget
  - Cell 5: **Tool selection strategies**
    - Rule-based: "If inventory check, use DB"
    - Cost-based: "Try cached estimate first (free), fallback to API ($0.001)"
    - LLM-based: "Ask meta-agent which tool to use"
  - Cell 6: **Error recovery** — Retry with fallback chain
    - Try fast tool → fails → try expensive tool → fails → escalate to human
  - Cell 7: **Cost comparison** — Single-pass vs. reflection vs. debate
    - Show token counts and latency for each pattern
  - Cell 8: **When to use each pattern** — Decision tree
  - Cell 9: **Production deployment** — LangGraph state machine
  - Cell 10: **Monitoring** — Log reflection loops, debate rounds, tool retries

- [ ] **Ch.11 notebook_supplement.ipynb (Production Patterns)**
  - Cell 1: Azure OpenAI credentials
  - Cell 2: Deploy reflection agent to Azure Container Instances
  - Cell 3: Monitor with Azure Application Insights (track loop counts, convergence rate)
  - Cell 4: Cost tracking (reflection adds 3× token cost → need budget alerts)
  - Cell 5: A/B test: Single-pass vs. Reflection (measure error rate improvement)
  - Cell 6: Debate pattern in production (latency: 3× single-pass → need async)
  - Cell 7: Hierarchical orchestration with Azure Durable Functions
  - Cell 8: Tool selection with Azure ML endpoint (meta-agent as deployed model)

### 🖼️ Diagrams & Animations (Rich Animations!)

#### Pattern 1: Reflection

- [ ] **`gen_ch11_reflection_loop.py`** → Animated flow: Generate → Critique → Revise
  - Frame 1: Draft response (with contradiction highlighted in red)
  - Frame 2: Self-critique bubble: "Detected contradiction: dairy-free + extra cheese"
  - Frame 3: Revised response (contradiction resolved, highlighted in green)
  - Frame 4: Comparison: Single-pass (failed) vs. Reflection (succeeded)
  - **Animation timing:** 0.5s per frame, 2s total
  - **Color scheme:** Red = error, Yellow = critique, Green = success

- [ ] **`gen_ch11_reflection_convergence.py`** → Graph: Error rate vs. number of refinement loops
  - X-axis: Number of loops (1, 2, 3, 5, 10)
  - Y-axis: Error rate (%)
  - Curve: Exponential decay (8% → 4% → 1% → 0.5% → 0.5%)
  - Insight: Diminishing returns after 3 loops

#### Pattern 2: Debate & Consensus

- [ ] **`gen_ch11_debate_flow.py`** → Animated multi-agent debate
  - Frame 1: Scenario (pricing conflict with 3 overlapping discounts)
  - Frame 2: Agent 1 proposes solution (apply all discounts = $12.50)
  - Frame 3: Agent 2 challenges (only 1 discount per company policy = $18.99)
  - Frame 4: Agent 1 defends (loyalty > coupon priority)
  - Frame 5: Judge decides (checks policy doc via RAG → Agent 2 wins)
  - **Animation timing:** 0.8s per frame, 4s total
  - **Visual:** Speech bubbles, arrows between agents, policy doc popup

- [ ] **`gen_ch11_debate_consensus.py`** → Venn diagram: Agreement zones
  - Circle 1: Agent 1 proposed solutions
  - Circle 2: Agent 2 proposed solutions
  - Circle 3: Agent 3 proposed solutions
  - Intersection: Consensus zone (all agents agree)
  - **Insight:** Consensus ≠ correctness (groupthink risk)

#### Pattern 3: Hierarchical Orchestration

- [ ] **`gen_ch11_hierarchical_flow.py`** → Animated planner → workers → verifier
  - Frame 1: Catering order (15 pizzas, 3 time slots, $200 budget)
  - Frame 2: Planner decomposes (5 pizzas at 11am, 5 at 12pm, 5 at 1pm)
  - Frame 3: Workers execute in parallel (3 worker agents, 1 per slot)
  - Frame 4: Verifier checks constraints (total cost $185 < $200 ✅)
  - Frame 5: Success confirmation (order placed)
  - **Animation timing:** 1s per frame, 5s total
  - **Visual:** Tree structure (planner → 3 workers), checkmarks for verification

- [ ] **`gen_ch11_hierarchical_coordination.py`** → Gantt chart: Worker task timeline
  - X-axis: Time (seconds)
  - Y-axis: Workers (W1, W2, W3)
  - Bars: Task execution (colored by status: running, completed, failed)
  - **Insight:** Parallel execution reduces latency (3 workers finish in 2s vs. 1 worker in 6s)

#### Pattern 4: Tool Selection

- [ ] **`gen_ch11_tool_selection_decision_tree.py`** → Animated decision tree
  - Root: "Need inventory count"
  - Branch 1: Cached estimate available? → Yes (use cache, free)
  - Branch 2: Cached estimate stale? → Query DB ($0.0001, 50ms)
  - Branch 3: DB unavailable? → Call API ($0.001, 200ms)
  - Branch 4: API unavailable? → Escalate to human
  - **Animation timing:** Follow decision path with highlighting

- [ ] **`gen_ch11_tool_fallback_chain.py`** → Waterfall diagram: Try tools in sequence
  - Stage 1: Try cache (free, 10ms) → ❌ stale
  - Stage 2: Try DB ($0.0001, 50ms) → ❌ timeout
  - Stage 3: Try API ($0.001, 200ms) → ✅ success
  - **Color:** Green = success, Red = failure, Yellow = retry

#### Cross-Pattern Comparison

- [ ] **`gen_ch11_pattern_comparison.py`** → Table: When to use each pattern
  - Columns: Pattern, Use case, Cost (tokens), Latency, Error reduction
  - Rows: Single-pass, Reflection, Debate, Hierarchical, Tool selection
  - **Insight:** Visual heatmap (green = best, red = worst) per metric

- [ ] **`gen_ch11_pattern_needle_movement.py`** → Animated needle: Error rate reduction
  - Before patterns: 8% error rate
  - After reflection: 4% error rate (needle moves right)
  - After debate: 2% error rate (needle moves further)
  - After hierarchical: 1% error rate (needle moves to green zone)
  - **Animation timing:** Smooth needle movement over 3s

### 🔧 Supporting Scripts

- [ ] **Pattern implementations**
  - `scripts/agentic_patterns/reflection_loop.py` → Reusable reflection pattern
  - `scripts/agentic_patterns/debate_framework.py` → N-agent debate with judge
  - `scripts/agentic_patterns/hierarchical_orchestrator.py` → Planner → Workers → Verifier
  - `scripts/agentic_patterns/tool_selector.py` → Meta-agent for tool selection

- [ ] **Demo workflows**
  - `scripts/agentic_patterns/pizzabot_v2_demo.py` → End-to-end demo with all patterns
  - `scripts/agentic_patterns/benchmark_patterns.py` → Cost/latency/error comparison

---

## Timeline

### Week 1: Research & Content Writing
- [ ] Audit existing mentions of reflection, debate, orchestration
- [ ] Write Ch.11 README.md (all 11 sections)
- [ ] Define animation specifications (timing, colors, captions)

### Week 2: Notebooks & Code
- [ ] Create Ch.11 notebook.ipynb (10 cells)
- [ ] Create Ch.11 notebook_supplement.ipynb (8 cells)
- [ ] Implement supporting scripts (reflection_loop.py, debate_framework.py, etc.)

### Week 3: Animations (Reflection & Debate)
- [ ] Generate reflection loop animation
- [ ] Generate reflection convergence graph
- [ ] Generate debate flow animation
- [ ] Generate debate consensus Venn diagram

### Week 4: Animations (Hierarchical & Tool Selection)
- [ ] Generate hierarchical flow animation
- [ ] Generate hierarchical coordination Gantt chart
- [ ] Generate tool selection decision tree
- [ ] Generate tool fallback chain diagram

### Week 5: Cross-Pattern Animations & Testing
- [ ] Generate pattern comparison table
- [ ] Generate pattern needle movement animation
- [ ] Test all notebooks end-to-end
- [ ] Update main README.md with Ch.11

---

## Success Criteria

- ✅ Ch.11 covers 4 core agentic patterns (reflection, debate, hierarchical, tool selection)
- ✅ **Rich animations** for every pattern (8+ animations total)
- ✅ Running example (PizzaBot v2.0) shows edge case handling
- ✅ Cost/latency/error comparison across patterns
- ✅ Decision tree: "Which pattern should I use?"
- ✅ Production deployment examples (Azure Container Instances, Durable Functions)
- ✅ Bridges to Multi-Agent AI track (`notes/multi_agent_ai/`)

---

## Bridge to Multi-Agent AI Track

**Single-agent patterns (Ch.11)** → **Multi-agent systems** (`notes/multi_agent_ai/`):

| Ch.11 Pattern | Multi-Agent Extension | Multi-Agent Chapter |
|---------------|----------------------|---------------------|
| Reflection | Agent A critiques Agent B's output | [Message Formats](../03-multi_agent_ai/ch01_message_formats/) |
| Debate | 3+ agents negotiate via protocol | [A2A](../03-multi_agent_ai/ch03_a2a/) |
| Hierarchical | Orchestrator delegates to specialist agents | [Agent Frameworks](../03-multi_agent_ai/ch07_agent_frameworks/) |
| Tool selection | MCP servers standardize tool access | [MCP](../03-multi_agent_ai/ch02_mcp/) |

**After Ch.11, you're ready for:**
- Multi-agent coordination (MCP, A2A)
- Event-driven agent systems (pub/sub, queues)
- Shared memory patterns (blackboard, Redis)
- Trust boundaries (agent sandboxing, auth)

---

## Notes

- **Animation-first approach:** Every pattern must have 2+ rich animations (flow + comparison)
- **PizzaBot v2.0 continuity:** Use same running example from Ch.1-10, add edge case handling
- **Cost transparency:** Show exact token counts and latency for each pattern
- **Production-ready:** All patterns deployed to Azure with monitoring
- **Multi-agent bridge:** Explicitly connect to `notes/multi_agent_ai/` for scaling patterns

---

## Questions / Blockers

- [ ] Should reflection use a separate critic model (small/fast) or same model (consistent but slow)?
- [ ] Should debate support >3 agents, or keep it simple with 2 proposers + 1 judge?
- [ ] Should hierarchical orchestration use LangGraph state machines or custom loop?
- [ ] Should tool selection meta-agent be fine-tuned, or use few-shot prompting?
