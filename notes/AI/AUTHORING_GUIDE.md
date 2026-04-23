# AI Track — Authoring Guide

> **This document tracks the chapter-by-chapter structure of the AI notes library.**  
> Each chapter lives under `notes/AI/` in its own folder, containing a .md file and a Jupyter notebook.  
> Read this before editing any chapter to keep tone, structure, and the running example consistent.

---

## The Plan

The AI track is currently 10 core chapters covering the full LLM agent stack from fundamentals to production. We're maintaining them as standalone, interconnected learning modules with a unified Grand Challenge arc:

```
notes/AI/
├── LLMFundamentals/
│   ├── LLMFundamentals.md          ← Technical deep-dive + diagrams
│   └── notebook.ipynb              ← Runnable code examples
├── PromptEngineering/
│   ├── PromptEngineering.md
│   └── notebook.ipynb
├── CoTReasoning/
│   ├── CoTReasoning.md
│   └── notebook.ipynb
... (10 chapters total)
```

Each module is self-contained but builds on previous chapters. The running example (Mamma Rosa's PizzaBot) threads through all 10 chapters, showing progressive capability unlocks toward a production-ready conversational AI system.

---

## The Running Example — Mamma Rosa's PizzaBot

Every chapter uses **one consistent system**: **Mamma Rosa's Pizza** — a regional pizza chain replacing phone-based ordering with an AI chatbot.

**The scenario**: *You're the Lead AI Engineer at Mamma Rosa's Pizza, and the CEO demands proof that AI chatbots deliver better business outcomes than traditional phone orders.*

The system is defined in [AIPrimer.md](./AIPrimer.md) and includes:
- **User interface**: Web widget + SMS
- **RAG corpus**: Menu, recipes, allergens, locations, delivery zones, FAQ, pricing (all private company data)
- **External tools**: `find_nearest_location()`, `check_item_availability()`, `calculate_order_total()`
- **Example queries**: "cheapest gluten-free pizza under 600 calories, available now"

This one system threads naturally through all 10 chapters:

| Chapter | What We Build / Learn |
|---|---|
| Ch.1 — LLM Fundamentals | Understand tokenization, sampling, context windows — but raw GPT gives unreliable answers |
| Ch.2 — Prompt Engineering | System prompts + few-shot → structured outputs, but still 15% error rate |
| Ch.3 — CoT Reasoning | Step-by-step reasoning → can handle multi-constraint queries ("cheapest gluten-free <600 cal") |
| Ch.4 — RAG & Embeddings | Semantic search over menu corpus → grounded answers, <5% error rate ✅ |
| Ch.5 — Vector DBs | HNSW/IVF indexes → faster retrieval (5s → 4s response time) |
| Ch.6 — ReAct & Semantic Kernel | Tool orchestration → can call APIs + proactive upselling ("add garlic bread?") |
| Ch.7 — Evaluating AI Systems | RAGAS metrics, conversion tracking → measure accuracy, business impact, hallucination rate |
| Ch.8 — Fine-Tuning | LoRA adapter for Mamma Rosa's brand voice → cost reduction + better upsells |
| Ch.9 — Safety & Hallucination | Prompt injection defense, guardrails → zero successful attacks ✅ |
| Ch.10 — Cost & Latency | KV caching, model tiers, streaming → optimized for <3s, <$0.08/conv |

> **Why this works:** The system demonstrates RAG (private menu data), tool use (external APIs), reasoning (multi-step queries), safety (adversarial users), and cost optimization (business constraints) — all the production challenges real AI engineers face.

---

## The Grand Challenge — Production-Ready PizzaBot

> Every chapter explicitly tracks progress toward a production system that satisfies strict **business, performance, and safety** requirements.

### The Scenario

You're the **Lead AI Engineer** at Mamma Rosa's Pizza. The CEO wants to launch an AI ordering chatbot, but they're skeptical. Traditional phone orders work fine — why invest $300k in AI?

**Your job**: Prove that AI delivers measurably better business outcomes:
- Higher order conversion rates
- Increased average order value (via intelligent upselling)
- Lower labor costs
- All while maintaining accuracy, speed, and safety standards

This isn't a demo or hackathon project. It's a **production system** handling real $30-60 customer transactions with zero tolerance for hallucinated menu items, slow responses, or security breaches.

### The 6 Core Constraints

Every chapter explicitly tracks which constraints it helps solve:

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **BUSINESS VALUE** | >25% order conversion + +$2.50 AOV vs. phone + 70% labor cost reduction | CEO's question: "Why spend $300k building this vs. hiring more phone staff?" Need clear ROI. Traditional phone orders: 22% conversion, $38.50 AOV, $157k/year labor |
| **#2** | **ACCURACY** | <5% error rate on menu queries + order placement | Hallucinated menu items → lost customer trust. Wrong orders → refunds, complaints. Must ground in truth |
| **#3** | **LATENCY** | <3s p95 response time | Customers abandon slow chatbots. Industry data: every second of delay = 10% conversation drop-off |
| **#4** | **COST** | <$0.08 per conversation average | 10,000 daily conversations × $0.20/conv = $60k/month (unsustainable). Target: <$25k/month to beat labor costs |
| **#5** | **SAFETY** | Zero successful prompt injections + appropriate refusals | Adversarial users can extract training data, manipulate orders, or bypass content policies. One viral incident = project shutdown |
| **#6** | **RELIABILITY** | >99% uptime + graceful degradation when tools fail | System outages during Friday dinner rush = direct revenue loss. Must handle tool failures gracefully |

### Business Baseline (Traditional Phone Orders)

For comparison, traditional phone order system metrics:

| Metric | Phone Baseline |
|--------|----------------|
| **Conversion rate** | 22% (of callers who engage with staff) |
| **Average order value** | $38.50 |
| **Labor cost** | 3 phone staff × $18/hr × 8hr × 365 days = **$157,680/year** |
| **Capacity** | ~45 simultaneous calls max → orders queued, customers hang up during peak hours |

### Target AI System Metrics

| Metric | AI Target | How AI Achieves It |
|--------|-----------|-------------------|
| **Conversion rate** | >25% | 24/7 availability, no wait times, proactive upselling, handles complex multi-constraint queries |
| **Average order value** | >$41 | AI suggests add-ons (drinks, sides, desserts) based on order + context |
| **Labor cost** | <$50k/year | 0.5 phone staff for edge cases + $25k API costs = **$43,920/year** (72% reduction) |
| **Capacity** | Unlimited | Handles unlimited simultaneous conversations |

### ROI Calculation

- **Development cost**: $300k (6 months × $50k/month for 1 senior AI engineer)
- **Monthly savings**: ($157,680 - $43,920) / 12 = **$9,480/month** (labor cost reduction)
- **Additional revenue**: 10k conversations/day × 25% conversion × $2.50 AOV increase = **$18,750/month**
- **Total monthly benefit**: $28,230
- **Payback period**: $300k ÷ $28,230 = **10.6 months**

### Progressive Capability Unlock

| Ch | Title | What Unlocks | Business Metrics | Constraint Progress |
|----|-------|--------------|------------------|---------------------|
| **1** | LLM Fundamentals | Understand tokenization, context windows, sampling | 8% conversion (raw GPT-3.5) | Foundation |
| **2** | Prompt Engineering | Structured prompts, few-shot, system prompts | 12% conversion, 15% error | #2 Partial |
| **3** | CoT Reasoning | Step-by-step planning, multi-constraint queries | 15% conversion, 10% error | #2 Partial |
| **4** | RAG & Embeddings | Grounded retrieval from menu corpus | 18% conversion, <5% error | #2 ✅ **ACHIEVED** |
| **5** | Vector DBs | Fast ANN search (infrastructure change) | 18% conversion (unchanged) | #3 Partial |
| **6** | ReAct & SK | Tool orchestration + proactive upselling | **28% conversion** (beats phone!) | #1 Partial, #6 Partial |
| **7** | Evaluating AI | RAGAS metrics, conversion tracking, A/B testing | 28% conversion (maintained) | Measurement infra |
| **8** | Fine-Tuning | LoRA for brand voice + cost reduction | 30% conversion, +$2.50 AOV, $0.008/conv | #1 + #4 Partial |
| **9** | Safety & Hallucination | Prompt injection defense, guardrails | Attack success rate: 0% | #5 ✅ **ACHIEVED** |
| **10** | Cost & Latency | KV caching, model tiers, optimized upsells | **32% conversion**, +$2.80 AOV, $0.07/conv, <2.8s p95 | #1 + #3 + #4 ✅ **ACHIEVED** |

**Final System Status**: All 6 constraints achieved. PizzaBot delivers:
- **32% conversion** (10 points above phone baseline)
- **+$2.80 AOV** (AI upselling works)
- **$0.07/conv cost** (sustainable economics)
- **<2.8s p95 latency** (no abandonment)
- **<3% error rate** (customer trust maintained)
- **0 successful attacks** (production-grade safety)
- **ROI: 10.6 month payback**

---

## Chapter Template Structure

Every chapter follows this structure to maintain consistency:

### Required Sections

#### § 0 · The Challenge — Where We Are

```markdown
## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Launch **Mamma Rosa's PizzaBot** — satisfying 6 constraints:
> 1. **BUSINESS VALUE**: >25% conversion + +$2.50 AOV + 70% labor savings — 2. **ACCURACY**: <5% error — 3. **LATENCY**: <3s p95 — 4. **COST**: <$0.08/conv — 5. **SAFETY**: Zero attacks — 6. **RELIABILITY**: >99% uptime

**What we know so far:**
- ✅ Ch.X: [Previous capabilities unlocked]
- ✅ Constraints #A, #B achieved
- ❌ **But [specific blocker]**
- 📊 **Current business metrics**: X% conversion (phone baseline: 22%), $Y.ZZ AOV (baseline: $38.50), $A/conv cost

**What's blocking us:**
🚨 **[Specific problem this chapter solves]**

[Concrete business scenario showing the problem — e.g., "Customer asks 'cheapest gluten-free pizza under 600 calories' but bot hallucinates a menu item that doesn't exist → order fails, customer lost"]

**Business impact**: [Why this matters for ROI — conversion drop, trust erosion, labor cost implications]

**What this chapter unlocks:**
🚀 **[Key capability]:**
1. [Specific technique/tool — e.g., "RAG: Semantic search over menu corpus"]
2. [How it addresses the blocker — e.g., "Grounds all menu answers in retrieved documents"]
3. [Expected business metric improvement — e.g., "Should reduce error rate from 15% → <5%"]

⚡ **Constraint #N [ACHIEVED/PARTIAL]**: [Evidence with business metrics — e.g., "Error rate now 4.2% (target: <5%) → Constraint #2 ACHIEVED! ✅ Conversion improves to 18%"]
```

**Key principles for § 0:**
- Start with the **business problem**, not the technical solution
- Show concrete failure scenarios from Mamma Rosa's perspective
- Quantify the business impact (conversion %, AOV, cost)
- Make it clear why the CEO would care about this chapter's content

#### § N · Progress Check — What We Can Solve Now

This section appears at the end of the chapter (section number varies based on chapter length).

```markdown
## N · Progress Check — What We Can Solve Now

🎉 **MAJOR MILESTONE**: ✅ **Constraint #N [DESCRIPTION] ACHIEVED!** (if applicable)

**Unlocked capabilities:**
- ✅ **[Technique 1]**: [What it enables — e.g., "Semantic search over menu corpus"]
- ✅ **[Technique 2]**: [What it enables]
- ✅ **[Real use case]**: [Concrete example — e.g., "Can now answer 'show me all gluten-free options under $15' with 99.2% accuracy"]

**Progress toward constraints:**

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 BUSINESS VALUE | ✅/❌/⚡ | X% conversion (target >25%), $Y AOV vs. $38.50 baseline, Z% labor savings |
| #2 ACCURACY | ✅/❌/⚡ | X% error rate (target <5%) — evidence from test set |
| #3 LATENCY | ✅/❌/⚡ | Xs p95 latency (target <3s) — measured across 1000 test conversations |
| #4 COST | ✅/❌/⚡ | $X/conv (target <$0.08) — includes LLM API + embedding + vector DB costs |
| #5 SAFETY | ✅/❌/⚡ | X successful attacks / Y attempted (target 0/Y) — from red team testing |
| #6 RELIABILITY | ✅/❌/⚡ | X% uptime (target >99%) + graceful degradation tested |

**What we can solve:**

✅ **[Use case 1]**:
- User: "[Example query]"
- System: [How it handles it now]
- Result: [Business outcome — e.g., "Order placed successfully, +$4.50 upsell"]

✅ **[Use case 2]**:
[Another concrete example]

❌ **What we can't solve yet:**
- **[Remaining blocker 1]**: [Why — e.g., "Responses take 6 seconds → 40% abandonment rate"]
- **[Remaining blocker 2]**: [What's needed]

**Business metrics update:**
- **Order conversion**: X% (baseline: 22% phone orders) — [interpretation]
- **Average order value**: $X.XX (baseline: $38.50 phone) — [interpretation]
- **Cost per conversation**: $X.XX (target: <$0.08) — [interpretation]
- **Error rate**: X% (target: <5%) — [interpretation]

**Next chapter**: [What capability unlocks next — e.g., "Vector DBs will reduce retrieval latency from 5s → <1s, improving conversion by eliminating abandonment"]
```

**Key principles for Progress Check:**
- Always show the **constraint status table** with current measurements
- Give **concrete use cases** that now work (with example queries)
- Be honest about **what still doesn't work** and why
- Update **business metrics** (conversion, AOV, cost) with actual numbers
- Connect to the next chapter's motivation

### Optional Sections (Use When Appropriate)

#### Bridge to Next Chapter

For chapters that naturally connect to the next one, add:

```markdown
## Bridge to Chapter X

Ch.Y unlocked [capability]. But [what's still broken/slow/expensive]. Chapter X tackles this by [preview of next technique], which will [expected improvement].

[Optional: Show a concrete failure case that motivates the next chapter]
```

---

## Content Guidelines

### Tone & Style

- **Direct and pragmatic**: This is production engineering, not research. Focus on "what works" and "what breaks."
- **Business-first**: Always connect technical choices to business outcomes (conversion, AOV, cost, trust)
- **Concrete examples**: Use real Mamma Rosa's queries throughout (e.g., "cheapest gluten-free pizza under 600 calories")
- **Honest about trade-offs**: If a technique solves X but makes Y worse, say so explicitly

### Math & Code

- **Math**: Only include formulas when they're essential to understanding (e.g., cosine similarity for embeddings, softmax for attention)
- **Code**: Prefer minimal, runnable examples. Use Python + OpenAI/Anthropic APIs where possible
- **Avoid**: Don't include "toy" examples (like "hello world" bots). Every example should relate to PizzaBot

### Figures & Diagrams

Each chapter should include:

1. **Architecture diagrams**: Show system components (LLM + RAG + tools) and data flow
2. **Concept diagrams**: Visualize key ideas (e.g., attention mechanism, vector search, ReAct loop)
3. **Performance charts**: Show business metrics improving over chapters (conversion, latency, cost)
4. **Milestone cards**: Celebrate constraint achievements with visual callouts

Store images in `notes/AI/{ChapterName}/img/` and reference with relative paths.

---

## Testing & Validation

Before marking a chapter complete:

1. ✅ **§ 0 Challenge section exists** and includes:
   - Current business metrics (conversion %, AOV, cost/conv)
   - Concrete failure scenario from PizzaBot
   - Clear statement of what this chapter unlocks

2. ✅ **Progress Check section exists** and includes:
   - Constraint status table with measurements
   - Concrete use cases that now work
   - Business metrics update
   - Honest assessment of remaining blockers

3. ✅ **Business narrative is coherent**:
   - Does the conversion rate progression make sense? (8% → 12% → 15% → 18% → ...)
   - Are constraint achievements justified with evidence?
   - Would a CEO reading this understand the ROI story?

4. ✅ **Technical content is accurate**:
   - Code examples run without errors
   - Diagrams match the text
   - References to other chapters are correct

---

## Constraint Achievement Evidence Standards

When marking a constraint as ✅ **ACHIEVED**, provide concrete evidence:

### #1 BUSINESS VALUE
- **Conversion rate**: A/B test results showing >25% conversion (include sample size, confidence interval)
- **AOV increase**: Average order value data showing ≥$2.50 increase from AI suggestions
- **Labor savings**: Cost breakdown comparing phone staff vs. AI system

### #2 ACCURACY
- **Error rate**: Measured on held-out test set of 1000+ queries with ground truth labels
- **Hallucination rate**: Manual review of 500 responses for factual correctness
- **Menu grounding**: 100% of menu item claims verified against RAG corpus

### #3 LATENCY
- **p95 latency**: <3s measured across 1000 production-like conversations
- **Abandonment rate**: <5% of users abandon before response (tracked via analytics)

### #4 COST
- **Per-conversation cost**: Detailed breakdown (LLM tokens, embeddings, vector DB, tools)
- **Monthly cost**: <$25k for 10,000 daily conversations (include calculation)

### #5 SAFETY
- **Attack success rate**: 0/100 prompt injection attempts succeed (from red team testing)
- **Refusal accuracy**: >99% appropriate refusals for out-of-scope / harmful requests

### #6 RELIABILITY
- **Uptime**: >99% over 30-day test period
- **Graceful degradation**: System handles tool failures without crashing (tested with 10 failure scenarios)

---

## FAQ

**Q: Should every chapter have a § 0 Challenge section?**  
A: Yes. Even foundation chapters (Ch.1) should set up the business context and show why we're building this.

**Q: What if a chapter doesn't improve any constraint?**  
A: That's fine (e.g., Ch.7 Evaluating AI just builds measurement infrastructure). Still show the constraint table with no changes, and explain that this chapter enables us to measure the others.

**Q: Can a chapter achieve multiple constraints at once?**  
A: Yes (e.g., Ch.10 Cost & Latency achieves #1, #3, #4 simultaneously via multiple optimizations).

**Q: Should supplement docs (e.g., RAGAndEmbeddings_Supplement.md) get Challenge sections?**  
A: No. Supplements are deep-dives for advanced readers. Keep them focused on technical depth without the business narrative.

**Q: How strict are the business metric targets?**  
A: They're realistic but aspirational. If your evidence shows 24% conversion instead of 25%, that's acceptable as long as it's above the 22% baseline and you explain the gap.

---

## Visualization Scripts

Generate constraint dashboards and business metric charts using:

```bash
python notes/AI/gen_scripts/generate_progress_visualizations.py
```

This creates:
- `ai-track-constraint-dashboard.png` (6×10 heatmap)
- `ai-track-conversion-progress.png` (8% → 32% conversion over chapters)
- `ai-track-cost-progress.png` (Cost/conv reduction)
- `ai-track-latency-progress.png` (Response time improvements)
- Milestone cards for Constraint achievements

---

## Final Checklist for Each Chapter

- [ ] § 0 Challenge section with business context
- [ ] Constraint status explicitly stated at the start
- [ ] Concrete PizzaBot failure scenario shown
- [ ] Main content (existing technical material) preserved
- [ ] Progress Check section with constraint table
- [ ] Business metrics updated (conversion, AOV, cost)
- [ ] Evidence provided for any constraint marked ✅
- [ ] Next chapter motivation clear
- [ ] Figures/diagrams added where appropriate
- [ ] Code examples tested and runnable
- [ ] No "TODO" or placeholder content
- [ ] Cross-references to other chapters verified

---

**Last updated**: April 2026  
**Status**: Active — 10 core chapters in AI track
