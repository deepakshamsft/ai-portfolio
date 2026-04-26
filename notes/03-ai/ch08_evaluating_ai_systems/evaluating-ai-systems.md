# Evaluating AI Systems — Measuring What Actually Matters

> **The story.** Translation evaluation was the trailhead. **BLEU** (Papineni et al., IBM, **2002**) gave us the first widely-adopted automatic n-gram overlap metric — useful but famously brittle. **ROUGE** (Lin, 2004) followed for summarisation; **BERTScore** (Zhang et al., 2019) replaced n-gram overlap with embedding similarity. The LLM era forced a rethink because outputs are now free-form, multi-paragraph, and don't have one correct answer. The breakthrough was **LLM-as-judge**: **Zheng et al.'s "Judging LLM-as-a-Judge"** (NeurIPS 2023, the **MT-Bench** paper) showed that GPT-4 agrees with human raters ~80% of the time — cheap and scalable enough to evaluate a fleet of agents. **RAGAS** (Es et al., 2023) productised this for RAG pipelines (faithfulness, answer relevancy, context precision/recall). **TruLens** (TruEra, 2023) and **DeepEval** (Confident AI, 2024) built it into developer tooling. By 2024 the standard production stack was: unit tests for retrievers, LLM-judge for generators, golden datasets curated from production traces. Today, every time you deploy a prompt change to PizzaBot, an automated test suite validates faithfulness before production.
>
> **Where you are.** Ch.1 (LLM Fundamentals) gave you 8% conversion with raw GPT-3.5. Ch.2 (Prompt Engineering) reached 12% via structured prompts. Ch.3 (CoT Reasoning) hit 15% with step-by-step planning. Ch.4 (RAG & Embeddings) achieved 18% and <5% error via grounded retrieval ✅. Ch.5 (Vector DBs) maintained 18% with faster search. Ch.6 (ReAct & Semantic Kernel) unlocked **28% conversion** via tool orchestration + proactive upselling — beating the 22% phone baseline! But every prompt change risks regression without automated testing. ML had its [own metrics chapter](../../ml/02_classification/ch03_metrics) for supervised learning. AI needs one too — because "correctness" in free-form text is fuzzy, context-dependent, and often requires another LLM to evaluate.
>
> **Business context.** You're the Lead AI Engineer at Mamma Rosa's Pizza. Current status: **28% conversion** (target >25% ✅), **+$2.50 AOV** (✅), **~5% error** (✅), **2.5s latency** (✅), **$0.015/conv** (✅). All core targets hit! But the CEO won't ship without automated testing: "One bad regression could wipe out all your conversion gains." You're deploying 2-3 prompt iterations per day, manually testing 3-5 queries each time, and suffering 2-3 regressions per week from changes that "looked fine" in manual tests. No A/B testing framework. No production monitoring. No way to prove a new model version (GPT-4o → GPT-4o-mini) maintains quality. This chapter builds the testing infrastructure that lets you iterate fast without breaking production.
>
> **Notation.** $F \in [0,1]$ — faithfulness (how well the answer is grounded in retrieved context); $P_c \in [0,1]$ — context precision; $R_c \in [0,1]$ — context recall; $R_a \in [0,1]$ — answer relevancy; $\bar{\rho}$ — mean pairwise agreement between judge scores (inter-rater reliability).

***

## 0 · The Challenge — Where We Are

> 🎯 **The mission**: Launch **Mamma Rosa's PizzaBot** — a production AI ordering system satisfying 6 constraints:
> 1. **BUSINESS VALUE**: >25% conversion + +$2.50 AOV + 70% labor savings — 2. **ACCURACY**: <5% error — 3. **LATENCY**: <3s p95 — 4. **COST**: <$0.08/conv — 5. **SAFETY**: Zero attacks — 6. **RELIABILITY**: >99% uptime

**What we know so far:**
- ✅ Ch.1-6: All core targets hit! 28% conversion ✅, +$2.50 AOV ✅, <5% error ✅, <3s latency ✅
- ⚡ **Current state**: Production-ready bot with RAG grounding, ReAct orchestration, proactive upselling
- 🎉 **Breakthrough achieved**: Beats phone baseline on both conversion (28% vs. 22%) and AOV ($40.60 vs. $38.50)

**What's blocking us:**

🚨 **No automated testing — regression risk on every code change**

**Current deployment process:**
```
1. Developer makes prompt change ("add more enthusiasm!")
2. Manually test 3-5 sample queries in terminal
3. ✅ "Looks good!" → push to production
4. 🔥 Next day: Customer complaint — "Bot told me Margherita is gluten-free!" (regression)
5. ❌ Rollback, debug, repeat
```

**Problems:**
1. ❌ **No regression detection**: Every prompt tweak risks breaking existing queries
2. ❌ **No quality baseline**: Can't tell if new model version (GPT-4o → GPT-4o-mini) degrades accuracy
3. ❌ **Manual testing doesn't scale**: 500+ menu combinations × 20+ query types = 10,000+ test cases
4. ❌ **No A/B testing framework**: Can't safely test "add garlic bread upsell" vs. "add drink upsell"
5. ❌ **No production monitoring**: Zero visibility into real-world error rates until customers complain

**Business impact:**
- **2-3 regressions per week** from prompt/code changes
- **Each regression**: ~4 hours debugging + rollback + hotfix
- **Customer trust erosion**: "Bot gave wrong info last week, why should I trust it now?"
- **Slow iteration velocity**: Fear of breaking production → conservative changes only
- CEO: "You hit your targets, but I can't ship this without automated testing. One bad regression could wipe out all your conversion gains."

**Why manual testing isn't enough:**

Simple change, hidden regression:
```
Change: Update system prompt to be "more friendly"
Before: "Our Veggie Garden pizza has 540 calories."
After:  "You'll love our Veggie Garden pizza! It's around 500-550 calories." ❌

Problem: "around 500-550" is hallucination! Real value: 540 calories
Manual test: Passed (tester didn't check exact number)
Production impact: 10% of calorie queries now give ranges instead of exact values
```

**What this chapter unlocks:**

🚀 **Automated evaluation framework:**
1. **Golden dataset**: 200 curated query-answer pairs covering all menu scenarios
2. **RAGAS metrics**: Automated faithfulness, answer relevancy, context precision scores
3. **LLM-as-judge**: GPT-4 evaluates answer quality on 1-10 scale
4. **Regression testing**: Every code change runs against full test suite (< 2 min)
5. **A/B testing**: Safe parallel deployment with automatic winner selection
6. **Production monitoring**: Real-time dashboards tracking error rate, latency, conversion

⚡ **Expected improvements:**
- **Regression prevention**: 2-3 regressions/week → **~0.1/week** (95% reduction)
- **Development velocity**: 2 prompt iterations/day → **10+ iterations/day** (safe to experiment)
- **Quality baseline**: Detect model degradation before customer complaints
- **A/B testing confidence**: Launch upsell experiments with statistical significance
- **Metrics**: No change to conversion/AOV (Ch.7 is testing infrastructure, not UX changes)

**Constraint status after Ch.7**: 
- #1-6: **All metrics maintained** (28% conversion, 5% error, 2.5s latency, $0.015/conv)
- **Infrastructure**: Automated testing prevents regressions, enables faster iteration
- **Production-ready**: Can now safely deploy updates without manual testing bottleneck

This is a **quality assurance chapter** — no business metric improvements, but essential for production reliability.

---

## 1 · Core Idea

AI system evaluation breaks into three levels:

```
Level 1 — Component evaluation   Does each piece work correctly in isolation?
                                   (embedding model recall, retrieval precision, LLM accuracy on benchmarks)

Level 2 — Pipeline evaluation    Does the assembled system produce good answers?
                                   (RAG faithfulness, agent task completion rate)

Level 3 — User evaluation        Do real users succeed at their goals?
                                   (session success rate, user satisfaction, task completion in A/B tests)
```

Most teams only do Level 1 and are surprised when Level 3 fails. The metrics at each level are different, and passing Level 1 does not guarantee passing Level 3.

---

## 2 · Evaluating RAG Pipelines — RAGAS

**RAGAS** (Retrieval Augmented Generation Assessment) is the standard framework for evaluating RAG systems without manually labelling every output. It uses an LLM-as-judge approach: a separate LLM scores each component of the RAG response.

### The Four Core RAGAS Metrics

| Metric | What it measures | Formula (conceptual) |
|---|---|---|
| **Faithfulness** | Does the answer contain only claims supported by the retrieved context? | `# claims in answer supported by context / # total claims in answer` |
| **Answer Relevance** | Does the answer actually address the question asked? | Embedding similarity between question and answer (via reverse-generation trick) |
| **Context Precision** | Are the retrieved chunks actually relevant to the question? | `# relevant chunks in top-k / k` |
| **Context Recall** | Did retrieval surface all the chunks needed to answer the question? | `# ground-truth supporting chunks retrieved / # ground-truth chunks total` |

### How RAGAS Works in Practice

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Your RAG system produces — for each question:
#   question: the user query
#   answer:   the LLM's response
#   contexts: the list of retrieved chunks
#   ground_truth: the correct answer (only needed for context_recall)

data = {
    "question": ["Does the Margherita pizza contain gluten?"],
    "answer":   ["The Margherita is available in a gluten-free base option. The standard Margherita contains wheat flour."],
    "contexts": [["allergens.csv: Margherita — gluten (standard base), dairy. GF base available on request."]],
    "ground_truth": ["Standard Margherita contains gluten; gluten-free base available."]
}

result = evaluate(
    Dataset.from_dict(data),
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)
print(result)
# {'faithfulness': 1.0, 'answer_relevancy': 0.95, 'context_precision': 1.0, 'context_recall': 1.0}
```

### What Each Score Tells You

| Score | Low → | High → |
|---|---|---|
| Faithfulness low | LLM is hallucinating beyond the retrieved context | Answer is grounded in context |
| Answer relevance low | LLM is answering a different question | Answer directly addresses the query |
| Context precision low | Retrieval is noisy — irrelevant chunks in top-k | Retriever is precise |
| Context recall low | Retrieval is missing key chunks | Retriever finds all necessary context |

**The diagnostic matrix:**

| Faithfulness | Context Precision | Root Cause |
|---|---|---|
| Low | High | LLM hallucination — relevant chunks retrieved but LLM ignores them |
| Low | Low | Retriever returns bad chunks AND LLM doesn't stay grounded |
| High | Low | LLM is grounded but working with useless context — lucky or question is easy |
| High | High | System is working correctly |

---

## 3 · Evaluating Reasoning Agents (ReAct Traces)

For agents that produce Thought → Action → Observation traces, evaluate at the **trace level**, not just the final answer.

### Metrics for Agent Traces

| Metric | Measurement method |
|---|---|
| **Task completion rate** | Did the agent produce a correct final answer? (Binary, requires ground truth) |
| **Step efficiency** | `ideal steps / actual steps` — did the agent take unnecessary loops? |
| **Hallucinated observations** | Did the agent fabricate a tool result without calling the tool? |
| **Tool call accuracy** | Were all tool calls made with correct arguments? |
| **Context window utilisation** | Did the agent run out of context before completing? |
| **Faithfulness of final answer** | Does the final answer follow from the observation trail? |

### Trace Evaluation with LLM-as-Judge

When ground-truth labels aren't available, use a second LLM to score the trace:

```python
TRACE_EVAL_PROMPT = """
You are evaluating an AI agent's reasoning trace.

Question: {question}
Trace:
{trace}
Final Answer: {answer}

Score each dimension from 1–5:
1. Faithfulness: Is the final answer supported by the observations in the trace?
2. Efficiency: Did the agent take unnecessary steps?
3. Groundedness: Did the agent fabricate any observation (marked ⚠️ if tool result is missing)?

Respond in JSON: {"faithfulness": int, "efficiency": int, "groundedness": int, "explanation": str}
"""
```

**LLM-as-judge limitations:** the judge LLM has its own biases (verbosity preference, position bias) and can be inconsistent. Always:
- Use a stronger model as judge than the model being evaluated
- Run each evaluation 3x and take the majority
- Validate the judge's scores against human labels on a sample set

---

## 4 · Component-Level Evaluation

### Embedding Model Evaluation

The retrieval quality is bounded by the embedding model's ability to separate relevant from irrelevant content.

| Metric | Meaning |
|---|---|
| **NDCG@k** | Normalised Discounted Cumulative Gain — ranks correct chunks higher than incorrect ones |
| **MRR** (Mean Reciprocal Rank) | `1 / rank_of_first_correct_chunk` — how early does the first relevant result appear? |
| **Recall@k** | `# relevant chunks in top-k / # total relevant chunks` |

**How to measure:** use a test set of `(query, expected_chunk_ids)` pairs. Run retrieval. Compute the above. MTEB (Massive Text Embedding Benchmark) provides standardised benchmarks for comparing embedding models.

### Chunking Strategy Evaluation

Different chunking strategies (fixed size, sentence-level, semantic) affect both precision and recall. Evaluate by measuring RAGAS context precision/recall across strategies.

```
Fixed 512-token chunks    → Context precision: 0.71  | Context recall: 0.83
Sentence-level chunks     → Context precision: 0.79  | Context recall: 0.76
Semantic (topic) chunks   → Context precision: 0.85  | Context recall: 0.81
```

There is no universally best strategy — the right one depends on document structure.

---

## 5 · Hallucination Detection

Hallucination is the biggest reliability problem in deployed LLM systems. Detection strategies:

### Self-consistency sampling

Sample the same question N times at temperature > 0. If the factual claims are consistent across samples, they are likely grounded. If they vary, the model is uncertain and may be hallucinating.

---

## 6 · PizzaBot Connection

Mamma Rosa's PizzaBot is the running example throughout this portfolio. Here is how each evaluation layer applies to it directly.

### RAGAS metrics on the PizzaBot RAG pipeline

| Query | Expected Behaviour | Failure Mode if Broken |
|---|---|---|
| "Does the Margherita contain gluten?" | Retrieves allergen chunk; answers faithfulness = 1.0 | Faithfulness < 1.0 → LLM added information not in chunk |
| "What is the cheapest GF pizza under 600 kcal?" | Retrieves calorie + price chunks; combines both | Context recall < 1.0 → calorie chunk missed |
| "Can I get delivery by 7 pm?" | Checks availability tool, not RAG | Answer relevance low → answered about ingredients instead |

### Agent trace evaluation

A PizzaBot order trace for "Large Margherita and Garlic Bread to 42 Maple Street" should produce a 6-step ReAct trace. Evaluation targets:

| Metric | Target | How to Measure |
|---|---|---|
| Task completion rate | ≥ 95 % | Run 100 synthetic orders; count successful `FINAL_ANSWER` emissions |
| Step efficiency | ≤ 6 steps | LLM-as-judge on traces; flag traces > 6 steps as over-planning |
| Tool groundedness | 100 % | Every price/availability claim must map to a tool `Observation` token |
| Hallucination rate | < 1 % | Self-consistency sampling: 5 chains per query, flag diverging price claims |

### Canonical prices for evaluation fixtures

```python
PIZZABOT_EVAL_FIXTURES = [
    {
        "question": "What is the total for a Large Margherita and Garlic Bread delivered to 42 Maple Street?",
        "ground_truth": "£22.96 (Margherita £13.99 + Garlic Bread £3.49 + delivery £1.99)",
        "expected_tools": ["find_nearest_location", "check_item_availability",
                           "retrieve_from_rag", "calculate_order_total"],
    },
    {
        "question": "Which gluten-free pizza has the fewest calories?",
        "ground_truth": "Veggie Feast GF at 490 kcal",
        "expected_tools": ["retrieve_from_rag"],
    },
]
```

Use these fixtures as the `ground_truth` column when running RAGAS `context_recall` evaluations. A passing suite requires faithfulness ≥ 0.95 and context recall ≥ 0.90 on all fixtures.

```python
def detect_hallucination_by_consistency(llm, question, n=5, threshold=0.7):
    answers = [llm.generate(question, temperature=0.7) for _ in range(n)]
    # Extract the key factual claim from each answer
    claims = [extract_claim(a) for a in answers]
    # Check if the majority agree
    from collections import Counter
    most_common, count = Counter(claims).most_common(1)[0]
    return most_common, count / n   # (answer, confidence)
```

### Entailment checking

Run a Natural Language Inference (NLI) model to check if the retrieved context **entails** each claim in the answer:

```
retrieved_context: "allergens.csv: Margherita — gluten (standard), dairy. GF base available."
answer_claim:      "The Margherita contains gluten in the standard base"
NLI label:         ENTAILMENT  →  claim is grounded

answer_claim:      "Home values increased 12% last year"
NLI label:         NEUTRAL     →  claim has no support in context → potential hallucination
```

Open-source NLI models: `cross-encoder/nli-deberta-v3-base`, `vectara/hallucination_evaluation_model`.

---

## 7 · The Evaluation Benchmark Trap

Standard LLM benchmarks (MMLU, HumanEval, MATH) measure model capability in isolation. They do not predict application performance. A model that scores 85 on MMLU may still hallucinate your specific domain's terminology at high rates because that domain was underrepresented in its training data.

**The rule:** always evaluate on your own data, in your own pipeline, with your own queries. Benchmark scores are useful for model selection; they are not a substitute for domain evaluation.

---

## 8 · What a Minimal Evaluation Setup Looks Like

For a production RAG pipeline, a minimum viable evaluation setup:

```
1. Curate 50–100 representative questions from real or expected user queries
2. For each: store the correct answer and the expected source chunk(s)
3. Run your full pipeline end-to-end on each question
4. Compute: faithfulness, context precision, context recall, answer relevance (RAGAS)
5. Compute: task completion rate (does the answer match the correct answer?)
6. Set alert thresholds: faithfulness < 0.9 → PagerDuty; context recall < 0.7 → review retriever
7. Re-run after every change to the pipeline (chunk size, embedding model, prompt template)
```

This 50–100 question set is your regression test suite. It catches regressions before they reach production.

---

## 9 · Progress Check — What We Can Solve Now

🎉 **TESTING INFRASTRUCTURE DEPLOYED**: Regression prevention achieved!

**Unlocked capabilities:**
- ✅ **Golden dataset**: 200 curated query-answer pairs covering all menu scenarios
- ✅ **RAGAS metrics**: Automated faithfulness, answer relevancy, context precision/recall scoring
- ✅ **LLM-as-judge**: GPT-4 evaluates answer quality on 1-10 scale
- ✅ **Regression testing**: Every code change runs full test suite in <2 minutes
- ✅ **A/B testing framework**: Safe parallel deployment with statistical significance
- ✅ **Production monitoring**: Real-time dashboards tracking error rate, latency, conversion

**Progress toward constraints:**

| Constraint | Status | Current State |
|------------|--------|---------------|
| #1 BUSINESS VALUE | ⚡ **MAINTAINED** | 28% conversion (target >25% ✅), +$2.50 AOV (✅), 70% labor savings (✅) |
| #2 ACCURACY | ✅ **TARGET HIT (maintained)** | ~5% error rate (target <5% ✅) — RAGAS faithfulness score 0.95+ |
| #3 LATENCY | ✅ **EXCELLENT (maintained)** | 2.5s p95 (target <3s ✅) |
| #4 COST | ⚡ **ON TRACK** | $0.015/conv (target <$0.08 ✅) |
| #5 SAFETY | ⚡ **MAINTAINED** | Zero allergen false claims in test suite |
| #6 RELIABILITY | ✅ **IMPROVED!** | Regression detection prevents production failures, uptime >99% |

**What we can solve:**

✅ **Regression prevention (2-3/week → ~0.1/week)**:
```
Before Ch.7: Manual testing only
Scenario: Developer updates system prompt for "friendlier tone"
- Manual test: 3 queries tested, all pass ✅
- Push to production
- Next day: Customer reports "Bot said Margherita is gluten-free!" ❌
- Root cause: Friendly tone prompt caused hallucination on edge case
- Cost: 4 hours debugging + rollback + hotfix

After Ch.7: Automated regression testing
Scenario: Same prompt change
- Pre-commit hook triggers test suite
- 200 queries run in 90 seconds
- RAGAS faithfulness score: 0.89 (down from 0.95) ❌
- Test fails: "Allergen claim not grounded in retrieval context"
- Commit blocked, developer notified with exact failing query
- Fix prompt, re-test, score returns to 0.95 ✅
- Push to production (no regression!)

Result: ✅ Regression caught before production!
        ✅ Zero customer impact
        ✅ Development velocity increased (safe to experiment)
```

✅ **Automated RAGAS evaluation**:
```
Test query: "What's the calorie count for a large Margherita?"

Bot response: "A large Margherita pizza is 920 calories."
Retrieved context: ["Margherita Pizza - Large (14"): 920 calories"]

RAGAS metrics:
- Faithfulness: 1.0 (claim "920 calories" directly supported by context)
- Answer Relevancy: 0.98 (directly answers question, no fluff)
- Context Precision: 1.0 (retrieved chunk is relevant)
- Context Recall: 1.0 (all required info retrieved)

Overall score: 0.995 ✅ (above 0.90 threshold)

---

Negative example (regression caught):
Test query: "What's the calorie count for a large Margherita?"

Bot response: "A large Margherita pizza is approximately 850-900 calories."
Retrieved context: ["Margherita Pizza - Large (14"): 920 calories"]

RAGAS metrics:
- Faithfulness: 0.65 ❌ (claim "850-900" contradicts context "920")
- Answer Relevancy: 0.85 (answers question but with hallucinated range)
- Context Precision: 1.0 (retrieved chunk is relevant)
- Context Recall: 1.0 (all required info retrieved)

Overall score: 0.875 ❌ (below 0.90 threshold)

Verdict: TEST FAILED - Hallucination detected!
```

✅ **A/B testing framework (safe experimentation)**:
```
Experiment: Test "add garlic bread" vs. "add drink" upsell

Control (baseline): Current "add garlic bread" upsell
- 50% of traffic
- Conversion: 28%
- AOV: $40.60
- Sample size: 1,000 visitors

Variant (test): "add drink" upsell
- 50% of traffic
- Conversion: 27.5%
- AOV: $39.80
- Sample size: 1,000 visitors

Statistical analysis:
- Conversion difference: -0.5 percentage points (not significant, p=0.32)
- AOV difference: -$0.80 (significant, p=0.04)
- Winner: Control (garlic bread upsell) ✅

Decision: Keep current upsell, abandon drink experiment
Result: ✅ Data-driven decision, no guesswork!
```

✅ **Production monitoring (real-time alerting)**:
```
Dashboard metrics (real-time):
- Error rate: 4.8% (target <5%) ✅
- Latency p95: 2.4s (target <3s) ✅
- Conversion rate: 28.2% (target >25%) ✅
- RAGAS faithfulness: 0.94 (target >0.90) ✅
- Hallucination incidents: 0/hour ✅

Alert triggered:
🚨 Error rate spike: 4.8% → 7.2% (exceeded 5% threshold)
Timestamp: 2026-04-20 14:32 UTC
Cause: RAG vector DB connection timeout (infrastructure issue)
Action: Auto-fallback to BM25 keyword search triggered
Resolution: Error rate returns to 5.1% within 2 minutes
Incident logged, team notified

Result: ✅ Proactive alerting caught issue!
        ✅ Graceful degradation prevented customer impact
        ✅ Root cause identified without customer complaints
```

**Business metrics update:**
- **Order conversion**: 28% (maintained from Ch.6, target >25% ✅)
- **Average order value**: $40.60 (maintained from Ch.6, +$2.50 vs. baseline ✅)
- **Cost per conversation**: $0.015 (maintained from Ch.6, target <$0.08 ✅)
- **Error rate**: ~5% (maintained from Ch.6, target <5% ✅)
- **Regression rate**: 2-3/week → **~0.1/week** (95% reduction) ✅
- **Development velocity**: 2 prompt iterations/day → **10+ iterations/day** (safe to experiment) ✅
- **Time to detect regressions**: 24 hours (customer complaints) → **<2 minutes** (pre-commit tests) ✅

**❌ What we can't solve yet:**
- **Brand voice inconsistency**: Bot sometimes says "Awesome choice!" (too casual) vs. "Excellent selection" (too formal) — no way to enforce consistent tone without fine-tuning
- **Cost optimization limits**: $0.015/conv is great, but 80% is GPT-4 API calls for answer generation — can't reduce further with current architecture
- **Latency floor**: 2.5s p95 is excellent, but can't break below 2s without model optimization (KV caching, quantization)
- **Adversarial attacks**: No systematic testing for prompt injection or jailbreak attempts

**Why this chapter was critical:**

Ch.7 is the **quality assurance gate** — no business metric improvements, but essential for production reliability:
1. **Regression prevention**: Every code change validated before production
2. **Fast iteration**: Safe experimentation without fear of breaking production
3. **Data-driven decisions**: A/B testing framework enables evidence-based optimization
4. **Proactive monitoring**: Catch issues before customers complain
5. **Compliance**: Automated testing required for enterprise deployment

**Next chapter**: [Fine-Tuning](../ch10_fine_tuning) tackles brand voice consistency and cost reduction via LoRA adapters → **30% conversion, $0.008/conv** (50% cost reduction), consistent Mamma Rosa's tone in every response.

**Key interview concepts from this chapter:**

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The four RAGAS metrics — what each measures and what a low score implies | Describe the diagnostic matrix: what does low faithfulness + high context precision mean? | Saying accuracy is sufficient for RAG evaluation — accuracy needs ground truth; RAGAS doesn't |
| LLM-as-judge approach — strengths and limitations | How do you evaluate an agent when there is no single correct answer? | Assuming benchmark scores predict application performance |
| Hallucination detection via self-consistency or NLI | What is NDCG@k and why is it better than recall@k for retrieval evaluation? | Confusing context recall (retrieval metric) with answer recall (generation metric) |
| The 3-level evaluation hierarchy | How would you set up a regression test suite for a RAG pipeline? | Evaluating only the LLM component and ignoring retrieval quality |
| **BLEU / ROUGE for generative evaluation:** BLEU measures n-gram precision of generated text against references (primarily translation); ROUGE measures recall-oriented n-gram overlap (primarily summarisation). Both are reference-based and require gold outputs. Key weakness: a factually correct generation using different words scores low | "What are the limitations of BLEU and ROUGE?" | "High BLEU / ROUGE means high quality" — they measure lexical overlap, not factual accuracy or coherence; LLM-as-judge has higher correlation with human preference on open-ended tasks |
| **Human evaluation methodology:** pairwise preference ("A or B?") scales better than absolute rating (avoids anchoring); inter-annotator agreement (Cohen's κ) should be reported; annotators must be blind to system identity. MT-Bench and MMLU are the dominant automated proxy benchmarks for instruction-following and knowledge | "How would you run a rigorous human evaluation for a new LLM?" | "High MT-Bench score means the model is production-ready" — MT-Bench measures generic instruction following on 80 multi-turn problems; domain performance, safety, latency, and cost are not captured |

---

## 10 · Bridge to Chapter 8

Ch.7 unlocked automated testing and regression prevention. But the evaluation suite revealed three systematic gaps:

1. **Brand voice drift**: RAGAS scores 0.95 faithfulness, but tone varies ("Awesome!" vs. "Excellent") — prompt engineering can't enforce consistent style
2. **Cost floor**: $0.015/conv is 80% GPT-4 API calls — RAG and caching already optimized, need model-level optimization
3. **Upsell quality**: A/B testing shows garlic bread upsells work, but conversion gains plateau — need smarter, context-aware suggestions

These aren't retrieval problems (RAG already solves those). They're **generation problems** — the model itself needs adaptation. Chapter 8 (Fine-Tuning) tackles this via **LoRA adapters**: train a lightweight layer on Mamma Rosa's brand voice + successful upsell patterns. Expected impact: **30% conversion** (consistent tone builds trust), **$0.008/conv** (50% cost reduction via smaller fine-tuned model), **2.0s latency** (faster inference).

> *A system you cannot measure is a system you cannot improve. Build the eval suite before you build the application — not after.*

## Illustrations

![Evaluating AI systems — RAGAS radar, reasoning-trace checklist, component-level eval, hallucination gate](img/Evaluating%20AI%20Systems.png)
