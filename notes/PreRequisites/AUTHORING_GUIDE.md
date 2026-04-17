# PreRequisites — Authoring Guide

Conventions for authoring chapters under `notes/PreRequisites/`. Read this before starting or editing any chapter so tone, structure, and interactive style stay consistent with Ch.1–7.

---

## Folder layout

Each chapter folder contains:

```
chNN-<slug>/
  README.md        # narrative, math, diagrams, exercises
  notebook.ipynb   # hands-on Python with interactive widgets
  img/             # static PNGs referenced from README
```

---

## Running thread — "The Perfect Free Throw"

Every chapter uses the same real-world problem: a basketball player shoots a free throw from 4.57 m at a 3.05 m hoop. Each chapter makes the problem one step more realistic, and in doing so adds exactly one new piece of mathematics to the reader's toolkit.

| Chapter | What we add | Free-throw question it answers |
|---|---|---|
| Ch.1 Linear Algebra | lines, weights, biases | What's the ball's height during the first 0.2 s? |
| Ch.2 Non-linear Algebra | polynomials, feature expansion | What's the *full* parabolic trajectory? |
| Ch.3 Calculus Intro | derivatives + integrals | What's the ball's instantaneous velocity at the apex? |
| Ch.4 Small Steps on a Curve | iterative optimisation | What release angle maximises range? |
| Ch.5 Matrices | linear algebra at scale | How do we handle wind + spin + 5 other variables at once? |
| Ch.6 Derivatives × Matrices | gradient, Jacobian, chain rule | Which variable should we adjust, and by how much? |
| Ch.7 Probability & Statistics | distributions, likelihood | How do we reason about a *noisy* release? |

When adding a new chapter, extend this thread — do not introduce an unrelated example.

---

## README style

- Short paragraphs.
- One-sentence-per-line math.
- No wall of symbols without an inline definition.
- Each chapter ends with 3–5 short exercises and a pointer to the ML chapter that reuses the material.
- Open with an epigraph that is **used** by the chapter (not decorative).
- Derive every rule before stating it — Taylor expansion, step-size bounds, convergence conditions, etc.

### Section order

1. Core Idea (2–3 sentences, plain English)
2. Running Example (the free-throw twist for this chapter)
3. Math (derived, not dumped)
4. Step by Step (numbered recipe)
5. Key Diagram (single hero PNG in `img/`)
6. Code Skeleton (6–20 lines, copy-pasteable)
7. What Can Go Wrong (common failure modes)
8. Exercises (3–5, increasing difficulty)
9. Where This Reappears (explicit pointers to ML chapters)
10. References (Krohn, 3Blue1Brown, Strang, Goodfellow as primary sources)

---

## Notebook style

- `numpy` + `matplotlib` + `ipywidgets` + `scipy` as needed.
- Every slider is paired with an editable numeric text box that syncs **bidirectionally**, so the reader can type exact values or drag for intuition.
- Notebooks mirror the README's section order — a reader should be able to follow both in parallel without context-switching.
- Must run on a stock developer laptop — no GPU required.

---

## External references (standard set for this track)

- **Jon Krohn — *Linear Algebra for Machine Learning*** (YouTube). Primary video companion for Ch.1 and Ch.5.
- **3Blue1Brown — *Essence of Linear Algebra* and *Essence of Calculus***. Visual companion for Ch.1, Ch.3, Ch.5.
- **Gilbert Strang — *Introduction to Linear Algebra* (MIT OCW 18.06)**. Definitive long-form reference for Ch.5.
- **Goodfellow, Bengio, Courville — *Deep Learning*, Chs. 2–4**. The gap-filler between Pre-Reqs and the ML book.
