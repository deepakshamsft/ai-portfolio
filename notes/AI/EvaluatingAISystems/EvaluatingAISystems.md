# Evaluating AI Systems — Measuring What Actually Matters

> **The story.** Translation evaluation was the trailhead. **BLEU** (Papineni et al., IBM, **2002**) gave us the first widely-adopted automatic n-gram overlap metric — useful but famously brittle. **ROUGE** (Lin, 2004) followed for summarisation; **BERTScore** (Zhang et al., 2019) replaced n-gram overlap with embedding similarity. The LLM era forced a rethink because outputs are now free-form, multi-paragraph, and don't have one correct answer. The breakthrough was **LLM-as-judge**: **Zheng et al.'s "Judging LLM-as-a-Judge"** (NeurIPS 2023, the **MT-Bench** paper) showed that GPT-4 agrees with human raters ~80% of the time — cheap and scalable enough to evaluate a fleet of agents. **RAGAS** (Es et al., 2023) productised this for RAG pipelines (faithfulness, answer relevancy, context precision/recall). **TruLens** (TruEra, 2023) and **DeepEval** (Confident AI, 2024) built it into developer tooling. By 2024 the standard production stack was: unit tests for retrievers, LLM-judge for generators, golden datasets curated from production traces.
>
> **Where you are in the curriculum.** ML had its [own metrics chapter](../../ML/ch09-metrics/). AI needs one too. Accuracy, RMSE, and AUC work for supervised ML because you have ground-truth labels. AI systems — RAG pipelines, reasoning agents, chatbots — produce free-form text where "correctness" is fuzzy, context-dependent, and often requires another LLM to evaluate. This document covers the evaluation frameworks, metrics, and practices that distinguish production AI systems from demos.

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

## 9 · Interview Checklist

| Must know | Likely asked | Trap to avoid |
|---|---|---|
| The four RAGAS metrics — what each measures and what a low score implies | Describe the diagnostic matrix: what does low faithfulness + high context precision mean? | Saying accuracy is sufficient for RAG evaluation — accuracy needs ground truth; RAGAS doesn't |
| LLM-as-judge approach — strengths and limitations | How do you evaluate an agent when there is no single correct answer? | Assuming benchmark scores predict application performance |
| Hallucination detection via self-consistency or NLI | What is NDCG@k and why is it better than recall@k for retrieval evaluation? | Confusing context recall (retrieval metric) with answer recall (generation metric) |
| The 3-level evaluation hierarchy | How would you set up a regression test suite for a RAG pipeline? | Evaluating only the LLM component and ignoring retrieval quality |
| **BLEU / ROUGE for generative evaluation:** BLEU measures n-gram precision of generated text against references (primarily translation); ROUGE measures recall-oriented n-gram overlap (primarily summarisation). Both are reference-based and require gold outputs. Key weakness: a factually correct generation using different words scores low | "What are the limitations of BLEU and ROUGE?" | "High BLEU / ROUGE means high quality" — they measure lexical overlap, not factual accuracy or coherence; LLM-as-judge has higher correlation with human preference on open-ended tasks |
| **Human evaluation methodology:** pairwise preference ("A or B?") scales better than absolute rating (avoids anchoring); inter-annotator agreement (Cohen's κ) should be reported; annotators must be blind to system identity. MT-Bench and MMLU are the dominant automated proxy benchmarks for instruction-following and knowledge | "How would you run a rigorous human evaluation for a new LLM?" | "High MT-Bench score means the model is production-ready" — MT-Bench measures generic instruction following on 80 multi-turn problems; domain performance, safety, latency, and cost are not captured |

---

## 10 · Bridge

Evaluating AI Systems provided the measurement layer that closes the loop on every other AI note — RAG pipelines, reasoning agents, and retrieval systems all need evaluation to move from demo to production. `FineTuning.md` shows the next natural step once evaluation reveals a systematic capability gap that prompt engineering and RAG cannot close.

> *A system you cannot measure is a system you cannot improve. Build the eval suite before you build the application — not after.*

## Illustrations

![Evaluating AI systems — RAGAS radar, reasoning-trace checklist, component-level eval, hallucination gate](img/Evaluating%20AI%20Systems.png)

## Illustrations

![Evaluating AI systems — RAGAS radar, reasoning-trace checklist, component-level eval, hallucination gate](img/Evaluating%20AI%20Systems.png)
