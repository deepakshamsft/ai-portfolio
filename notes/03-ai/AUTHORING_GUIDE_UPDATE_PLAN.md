# AI Track Authoring Guide Update — Implementation Plan

**Target:** `notes/03-ai/authoring-guide.md`  
**Effort:** 2-3 hours  
**LLM Calls:** 1

---

## Quick Context

Add workflow-based chapter pattern for procedural chapters (ch02 prompt engineering, ch06 react, ch08/ch12 evaluation). AI track has good industry tools coverage but lacks workflow organization patterns.

---

## Single Addition Required

**Location:** After "Chapter Template" section

**Content:** Same workflow pattern section from ML track, adapted for AI tools:

**Industry Tools to Cover:**
- Prompt engineering: `LangChain PromptTemplate`, `OpenAI ChatCompletion`, `Anthropic Claude`
- ReAct: `LangChain Agent`, `Semantic Kernel Planner`
- Evaluation: `RAGAS`, `TruLens`, `DeepEval`

**Procedural Chapters:**
- ch02: Prompt refinement loop (iterate → evaluate → refine)
- ch06: Tool orchestration workflow (define → chain → execute)
- ch08/ch12: Testing workflow (setup → run → analyze)

---

## Implementation

**Call 1:** Insert workflow pattern section after line ~150

**Content template:**
```markdown
## Workflow-Based Chapter Pattern

### When to Use
- Iterative refinement workflows (prompt engineering)
- Tool orchestration sequences (ReAct, function calling)
- Testing/evaluation procedures

### Decision Checkpoint Format
**What you observed:** [Prompt output quality]
**What it means:** [Interpretation of failure mode]
**What to do next:** [Specific prompt modification]

### Industry Tools Integration
Show manual implementation → then `LangChain` equivalent
```

---

## TODO: Notebook Audit

- [ ] ch02: Show manual prompt iteration → then `PromptTemplate`
- [ ] ch06: Show manual tool chaining → then `LangChain Agent`
- [ ] ch08: Show manual evaluation → then `RAGAS.evaluate()`

**Pattern:**
```python
# Manual (learning)
prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
response = llm(prompt)

# Industry standard (production)
from langchain.prompts import ChatPromptTemplate
template = ChatPromptTemplate.from_messages([...])
chain = template | llm
response = chain.invoke({"context": context, "question": question})
```

---

## Success Criteria

1. ✅ Workflow pattern section added
2. ✅ AI-specific tool examples (LangChain, RAGAS)
3. ✅ Decision checkpoints for iterative workflows
4. ✅ Clear guidance on when to use workflow vs concept structure

**Files Modified:** 1 (`authoring-guide.md`)
