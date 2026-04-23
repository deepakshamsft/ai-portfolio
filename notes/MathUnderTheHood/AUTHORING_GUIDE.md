# Math Under the Hood — Authoring Guide

Conventions for authoring chapters under `notes/MathUnderTheHood/`. Read this before starting or editing any chapter so tone, structure, and interactive style stay consistent with Ch.1–7.

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

## Running thread — "The Perfect Knuckleball Free Kick"

Every chapter uses the same real-world problem: a football (soccer) striker lines up a direct **knuckleball free kick** 20 m from goal, aiming to clear a defensive wall and dip the ball under the 2.44 m crossbar. Because the strike has near-zero spin, the path is governed by gravity alone and reduces to a clean 2D parabola — the same problem that forced Newton and Leibniz to invent calculus.

### **The Grand Challenge**

**Can we score this goal?** To succeed, the ball must satisfy **THREE constraints simultaneously**:

1. **🧱 Wall Clearance**: At the wall position (9.15 m horizontal distance, ~0.6s flight time), the ball must be **above 1.8 m** (wall height)
2. **🎯 Crossbar Clearance**: At the goal line (20 m horizontal distance, ~1.2s flight time), the ball must be **below 2.44 m** (crossbar height)
3. **⚡ Keeper-Beating Speed**: The ball must arrive fast enough (or with short enough flight time) that the goalkeeper cannot react

These are **competing constraints** — a high launch angle clears the wall easily but might sail over the crossbar. A low angle beats the keeper but hits the wall. Finding parameters (launch speed v₀, angle θ) that satisfy all three is a **constrained multi-objective optimization problem** — exactly what real ML does.

### **Progressive Capability Unlock**

Each chapter gives us ONE new mathematical tool to solve ONE piece of the challenge. Early chapters can only predict; later chapters can optimize. The progression is:

| Chapter | Math Tool | What It Unlocks | Challenge Progress |
|---|---|---|---|
| Ch.1 Linear Algebra | Lines, slopes, intercepts | Predict height during first 0.1s (linear approx) | ❌ Can't model full curve yet |
| Ch.2 Non-linear Algebra | Polynomials, parabolas | Model full trajectory h(t) = v₀ᵧt - ½gt² | ❌ Can't find peak/crossings yet |
| Ch.3 Calculus Intro | Derivatives, rate of change | Find apex (h'=0), compute wall/goal heights | ✅ **Can check constraints 1 & 2!** |
| Ch.4 Small Steps | Gradient descent, iteration | Optimize ONE parameter (angle for max range) | ✅ Can find best single-variable solution |
| Ch.5 Matrices | Multi-variable systems | Handle v₀ AND θ AND wind simultaneously | ❌ Can't optimize multi-dim yet |
| Ch.6 Chain Rule | Gradients, Jacobians | Optimize MULTIPLE parameters at once | ✅ **Can solve full challenge!** |
| Ch.7 Probability | Distributions, noise | Handle striker fatigue, ball variance | ✅ Can reason about success rate |

**Key narrative arc**: We move from "can we predict?" (Ch.1-3) → "can we optimize one thing?" (Ch.4) → "can we optimize everything?" (Ch.5-6) → "can we handle uncertainty?" (Ch.7).

When adding a new chapter, extend this thread — do not introduce an unrelated example. Every chapter should explicitly state:
- **What constraint(s) we're working toward**
- **What we can now solve**  
- **What's still blocked (and which future chapter unlocks it)**

---

## Required Chapter Sections

### Opening: "The Challenge"

Every chapter README must open (after the title/epigraph) with this structure:

```markdown
## 0 · The Challenge — Where We Are

> 🎯 **The goal**: Score a free kick that clears a 1.8m wall and dips under a 2.44m crossbar while beating the keeper's reaction time.

**What we know so far:**
- ✅ [Capabilities from previous chapters]
- ❌ [What we still can't do]

**What's blocking us:**
[Specific problem this chapter solves]

**What this chapter unlocks:**
[The new capability we gain]
```

### Closing: "Progress Check"

Every chapter must end (before References) with:

```markdown
## N · Progress Check — What We Can Solve Now

✅ **Unlocked capabilities:**
- [List what reader can now do]

❌ **Still can't solve:**
- [What's blocked until future chapters]

**Next up:** [Preview of Ch.N+1's unlock]

![Progressive free-kick diagram showing green zones (solvable) and red zones (blocked)](img/chNN-progress-check.png)
```

The PNG should show the trajectory with color-coded regions indicating what we can vs. can't verify/optimize yet.

---

## README style

- Short paragraphs.
- One-sentence-per-line math.
- No wall of symbols without an inline definition.
- Each chapter ends with 3–5 short exercises and a pointer to the ML chapter that reuses the material.
- Open with an epigraph that is **used** by the chapter (not decorative).
- Derive every rule before stating it — Taylor expansion, step-size bounds, convergence conditions, etc.

### Section order

Two templates are allowed. Pick one and stay inside it.

**Template A — Standard chapter (Ch.1, 2, 4, 5).** Use this whenever the chapter has a clear single mathematical object to derive and exercise.

1. Core Idea (2–3 sentences, plain English)
2. Running Example (the free-kick twist for this chapter)
3. Math (derived, not dumped)
4. Step by Step (numbered recipe)
5. Key Diagram (single hero PNG in `img/`)
6. Code Skeleton (6–20 lines, copy-pasteable)
7. What Can Go Wrong (common failure modes)
8. Exercises (3–5, increasing difficulty)
9. Where This Reappears (explicit pointers to ML chapters)
10. References (Krohn, 3Blue1Brown, Strang, Goodfellow as primary sources)

**Template B — Math-heavy chapter (Ch.3, 6, 7).** Use when the chapter introduces several distinct mathematical objects (e.g. derivative *and* integral; gradient *and* chain rule *and* Hessian; three distributions *and* MLE) that each need their own derivation block. The hero PNG may sit immediately under the title (above the epigraph) when the chapter has no natural §5 home for it. Required sections in any order that flows: Core Idea, every named math object as its own section, a Worked Example or Step-by-Step, Pitfalls / What Can Go Wrong, Where This Reappears, References. Exercises and Code Skeleton are optional in Template B if the notebook already covers them — link to the notebook explicitly when omitting.

Whichever template you pick, every chapter must end with **Where This Reappears** and **References**.

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
