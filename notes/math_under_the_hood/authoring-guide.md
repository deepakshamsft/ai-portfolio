# Math Under the Hood — Authoring Guide

> **This document tracks the chapter-by-chapter build of the Math Under the Hood track.**  
> Each chapter lives under `notes/math_under_the_hood/` in its own folder, containing a README and a Jupyter notebook with interactive widgets.  
> Read this before starting any chapter to keep tone, structure, and the knuckleball free-kick running example consistent.
>
> **📚 Updated:** Now includes comprehensive pedagogical patterns, voice guidelines, and conformance standards aligned with ML track (see §"Style Ground Truth" and §"Pedagogical Patterns" below).

<!-- LLM-STYLE-FINGERPRINT-V1
canonical_chapters: ["notes/math_under_the_hood/ch01_linear_algebra/README.md", "notes/math_under_the_hood/ch03_calculus_intro/README.md"]
voice: second_person_practitioner
register: technical_but_conversational
formula_motivation: required_before_each_formula
numerical_walkthroughs: knuckleball_trajectory_with_explicit_arithmetic
dataset: knuckleball_free_kick_physics_2d_parabola
failure_first_pedagogy: true
callout_system: {insight:"💡", warning:"⚠️", constraint:"⚡", optional_depth:"📖", forward_pointer:"➡️"}
mermaid_color_palette: {primary:"#1e3a8a", success:"#15803d", caution:"#b45309", danger:"#b91c1c", info:"#1d4ed8"}
image_background: dark_facecolor_1a1a2e_for_generated_plots
section_template: [story_header, challenge_0, core_idea_1, running_example_2, math_3, step_by_step_4, key_diagrams_5, code_skeleton_6, what_can_go_wrong_7, progress_check_N, bridge_N1]
math_style: scalar_first_then_vector_generalization
ascii_matrix_diagrams: required_for_matrix_operations
forward_backward_links: every_concept_links_to_where_it_was_introduced_and_where_it_reappears
conformance_check: compare_new_chapter_against_ch01_and_ch03_before_publishing
red_lines: [no_formula_without_verbal_explanation, no_concept_without_knuckleball_grounding, no_section_without_forward_backward_context, no_math_derivation_without_toy_numerical_example, no_callout_box_without_actionable_content]
interactive_widgets: bidirectional_sliders_with_numeric_inputs
-->

---

## Conventions for authoring chapters under `notes/math_under_the_hood/`

Read this before starting or editing any chapter so tone, structure, and interactive style stay consistent with Ch.1–7.

---

## Folder Layout

Each chapter folder contains:

```
chNN-<slug>/
  README.md        # narrative, math derivations, diagrams, exercises
  notebook.ipynb   # hands-on Python with interactive widgets
  img/             # static PNGs and GIFs referenced from README
  gen_scripts/     # (optional) Python scripts that generate animations
```

**Naming conventions:**
- Images: `chNN-[topic]-[type].png/.gif` (chapter-specific) or `[concept]_generated.gif/.png` (algorithmically generated)
- Animation needles: `chNN-[slug]-needle.gif` (progress visualization)
- All generated plots use dark background `facecolor="#1a1a2e"` to match rendered theme

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

## Chapter README Template

Every chapter README follows this **extended structure**:

```markdown
# Ch.N — [Topic Name]

> **The story.** (Historical context — who invented this, when, why, what problem drove the invention)
>
> **Where you are in the curriculum.** (Links to previous chapters, what this adds, what capability gap it fills)
>
> **Notation in this chapter.** (Declare all symbols inline before §0: $v_0$ — initial velocity; $\theta$ — launch angle; $g=9.8\text{ m/s}^2$ — gravity; etc.)

---

## 0 · The Challenge — Where We Are

> 🎯 **The goal**: Score a free kick that clears a 1.8m wall and dips under a 2.44m crossbar while beating the keeper's reaction time.
> 
> **THREE constraints:**
> 1. 🧱 WALL CLEARANCE: Ball height > 1.8m at 9.15m distance
> 2. 🎯 CROSSBAR CLEARANCE: Ball height < 2.44m at 20m distance  
> 3. ⚡ KEEPER-BEATING SPEED: Flight time or arrival velocity sufficient

**What we know so far:**
- ✅ [Summary of previous chapters' achievements with specific numbers]
- ❌ **But we still can't [X]!** [Specific capability blocked]

**What's blocking us:**
[Concrete description of the mathematical gap this chapter addresses — never abstract "we need to learn X" but specific "we can't find the apex of the trajectory" or "we can't optimize two variables simultaneously"]

**What this chapter unlocks:**
[Specific capability bullet points that advance one or more constraints]

---

## 1 · The Core Idea (2–3 sentences, plain English)

[The concept in practitioner language — no jargon, no symbols]

## 2 · Running Example — The Free Kick Twist

[How this chapter's math applies to the knuckleball problem specifically]

## 3 · The Math

[Derived formulas, not dumped — show why each step follows from first principles. Every symbol verbally glossed within 3 lines of introduction.]

### Scalar Form First

[Show concept for single variable or 1D case]

### Vector/Matrix Generalization

[Extend to multiple dimensions — show structural identity with scalar form]

## 4 · How It Works — Step by Step

[Numbered recipe with specific computational steps]

## 5 · The Key Diagrams

[Mermaid flowcharts or static PNG — minimum 1 hero diagram. Caption must interpret the diagram, not just describe it: "Without derivatives: guess-and-check (slow). With derivatives: follow the slope (fast)."]

## 6 · The Hyperparameter Dial (if applicable)

[The main tunable parameter, its effect on trajectory, typical starting value]

## 7 · Code Skeleton

[Minimal Python (6–20 lines) — illustrative, copy-pasteable, uses numpy + matplotlib]

## 8 · What Can Go Wrong

[3–5 bullet traps following pattern: **Bold trap name** — description (2-3 sentences with numbers) → **Fix:** one actionable sentence]

## 9 · Where This Reappears

[Forward links to ML chapters that use this math + later Math chapters that build on it]

## N · Progress Check — What We Can Solve Now

![Progress visualization](img/chNN-[slug]-progress-check.png) ← **Required**: Visual showing green zones (solvable) vs red zones (blocked)

✅ **Unlocked capabilities:**
- [Specific things you can now do with exact constraint numbers]
- [e.g., "Can now verify wall clearance: h(0.6s) = 1.92m > 1.8m ✅"]

❌ **Still can't solve:**
- [What's blocked — explicitly preview next chapter's unlock]
- [e.g., "❌ Can't find optimal launch angle yet — no way to maximize range"]

**Trajectory status**: [One-sentence summary showing trajectory plot with verified vs unverified regions]

**Next up:** Ch.X gives us **[concept]** — [what it unlocks for the free kick]

---

## N+1 · Bridge to the Next Chapter

[One clause what this established + one clause what next chapter adds — always tie to the knuckleball challenge]

---

## Exercises

[3–5 exercises, increasing difficulty. Each must use knuckleball parameters or variations]

---

## References

[Standard references + chapter-specific additions]
```

**Template variants:**
- **Template A** (standard): Use for chapters with single clear mathematical object (Ch.1, 2, 4, 5)
- **Template B** (math-heavy): Use when chapter introduces multiple distinct objects (Ch.3 derivative+integral; Ch.6 gradient+chain rule+Hessian). Hero diagram can sit under title (above story). Required sections in any flow order: Core Idea, each math object as own section, Worked Example or Step-by-Step, What Can Go Wrong, Where This Reappears, References.

---

## Required Chapter Sections

### Opening: "The Challenge"

Every chapter README must open (after the title/story/notation blockquote) with this structure:

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

## Style Ground Truth — Voice, Register, and Mathematical Conventions

> **LLM instruction:** Before authoring or reviewing any chapter in this track, treat Ch.01 (Linear Algebra) and Ch.03 (Calculus Intro) as the canonical style reference. Every dimension below was extracted from cross-track analysis aligned with ML standards. When a new or existing chapter deviates from any dimension, flag it. When generating new content, verify against each dimension before outputting.

---

### Voice and Register

**The register is: technical-practitioner, second person, conversational within precision.**

The reader is treated as a capable engineer who doesn't need flattery, gets impatient with abstract theory, and wants to know what to *do* and *why it matters*. The tone is direct — every sentence earns its place. There is no "Let's explore together!", no "In this section we will discuss", no hedging language that softens a concrete fact into a vague observation.

**Second person is the default.** The reader is placed inside the scenario at all times:

> *"You're lining up the free kick. 20 meters out. One shot."*  
> *"You just did calculus. By hand. And it worked."*  
> *"The striker asks: 'What angle gives maximum range?' You can't answer yet."*

**Dry, brief humour appears exactly once per major concept.** It is never laboured. Examples:
> *"The ball doesn't care about our coordinate system."*  
> *"Newton and Leibniz spent 20 years fighting about who invented this. You'll master it in 20 minutes."*

The register: wry, businesslike, never cute.

**Contractions and em-dashes are used freely** when they tighten a sentence:
> *"That's the derivative."*  
> *"Parabolas curve — lines don't."*  
> *"Full stop."*

**Academic register is forbidden.** Phrases like "In this section we demonstrate", "It can be shown that", "The reader may note", "we present", "we propose" do not appear in these chapters and must not appear in any new chapter.

---

### Story Header Pattern

Every chapter opens with three specific items, in order, in a blockquote:

1. **The story** — historical context. Who invented this concept, in what year, on what problem. Always a real person and a real date. Examples:
   - Descartes (1637) — *La Géométrie* — algebra meets geometry
   - Newton (1687) — *Principia* — calculus to solve planetary motion
   - Gauss (1809) — least squares on Ceres asteroid orbit

   The history is brief (one paragraph), specific (named people, named papers, named years), and closes with a sentence connecting the historical moment to the practitioner's daily work.

2. **Where you are in the curriculum** — one paragraph precisely describing what the previous chapter(s) gave you and what gap this chapter fills. Must name specific capabilities or constraint statuses from preceding chapters.

3. **Notation in this chapter** — a one-line inline declaration of every symbol introduced in the chapter, before the first section begins. Not a table — a single sentence with $inline$ math. Example: *"$v_0$ — initial velocity (m/s); $\theta$ — launch angle (degrees); $g=9.8\text{ m/s}^2$ — gravitational acceleration; $h(t)$ — height at time $t$; $x(t)$ — horizontal distance at time $t$."*

---

### The Failure-First Pedagogical Pattern

**This is the most important stylistic rule.** Concepts are never listed and explained — they are *discovered by exposing what breaks*.

The trajectory analysis is a canonical example:
- Act 1: Try linear approximation (works for first 0.1s) → show exactly where it breaks (can't model apex or full flight)
- Act 2: Introduce parabola → show it models full arc → but can't find critical points yet
- Act 3: Introduce derivatives → find apex precisely
- Act 4: Show derivatives can't optimize multi-variable problems alone → motivates gradients (Ch.6)

Each step in the arc: **tool → specific failure → minimal fix → that fix's failure → next tool**. The reader is never asked to memorise a taxonomy. They experience the need for each concept before seeing it.

**This pattern must appear in every subsection that covers multiple options or variants.** If a section presents three methods, the section must show *what breaks* with the simpler method before introducing the more complex one. Listing methods without demonstrating failure is the wrong pattern.

---

### Mathematical Style

**Rule 1: scalar form before vector form.** Every formula is first shown for one dimension or one variable, then generalised. The generalisation is presented as a direct extension, not a separate derivation.

Example: Show `h(t) = v₀t - ½gt²` (scalar trajectory) first, then `r⃗(t) = r⃗₀ + v⃗₀t + ½a⃗t²` (vector generalization).

**Rule 2: every formula is verbally glossed immediately after it appears.** Not in a table of notation (though those also exist) — in the prose directly below the LaTeX block:

> *"The first term is initial velocity times time — how far you'd go with no gravity. The second term is the gravitational pull — it grows with t² because acceleration accumulates."*

If a formula has no verbal gloss within three lines, it is incomplete.

**Rule 3: the notation table lives in the header.** All symbols are declared in the "Notation in this chapter" header blockquote before any section. Subsections add no new notation without glossing it immediately.

**Rule 4: optional depth gets a callout box.** Derivations that would break the flow of a practitioner reading for intuition go inside an indented `> 📖 **Optional:**` block. These are clearly labelled and can be skipped without losing the main thread. Full Taylor series expansions, Lagrange multiplier derivations, rigorous epsilon-delta proofs — all go in these blocks. The optional block ends with a note on where to find the complete treatment if needed.

**Rule 5: ASCII matrix diagrams for matrix operations.** When showing a matrix multiply or a matrix structure, draw it in ASCII with aligned brackets, showing the dimensions of each operand and the result. Example:

```
A · x                                              (2×2) · (2×1) → (2×1)

  A                            x
  ┌  2   3  ┐                 ┌  1  ┐
  └  4   1  ┘  ×              └  2  ┘
  
= ┌  2×1 + 3×2  ┐  =  ┌  8  ┐
  └  4×1 + 1×2  ┘     └  6  ┘
```

---

### Numerical Walkthrough Pattern

**Every mathematical concept must be demonstrated on actual numbers before being generalised.** The walkthrough always uses knuckleball parameters: v₀=20 m/s, typical angles 20°-45°, distances 9.15m (wall) and 20m (goal), heights 1.8m (wall) and 2.44m (crossbar).

**The canonical walkthrough structure:**
1. State the scenario with exact numbers
2. State the formula to be applied
3. Show step-by-step arithmetic with intermediate values
4. Show the numerical result bolded
5. Verify: "At 0.6s, height is **1.92m** — clears the 1.8m wall ✅"

**Example:**
```markdown
**Scenario:** v₀ = 20 m/s, θ = 30°, t = 0.6s  
**Formula:** h(t) = v₀sin(θ)·t - ½g·t²

**Step-by-step:**
1. v₀sin(30°) = 20 × 0.5 = 10 m/s (vertical velocity component)
2. ½g·t² = 0.5 × 9.8 × 0.36 = 1.76 m (gravity drop)
3. h(0.6) = 10 × 0.6 - 1.76 = 6.0 - 1.76 = **4.24m**

**Verification:** At wall position (0.6s), height is 4.24m >> 1.8m → wall cleared ✅
```

**Every walkthrough ends with a verification sentence** confirming the arithmetic and tying it back to the constraint.

---

### Forward and Backward Linking

**Every new concept is linked to where it was first introduced and where it will matter again.** This is not optional.

**Backward link pattern:** *"This is the same linear interpolation from Ch.1 — the only difference is we now know when it breaks (curves)."*

**Forward link pattern:** *"This derivative machinery is the entire foundation of gradient descent in ML. Every time you see `loss.backward()` in PyTorch, this chain rule is running — one step per layer."*

**The forward pointer callout box** (`> ➡️`) is used for concepts that will be formally introduced later but need to be planted early. Example: plant gradients in Ch.3 with `> ➡️ Ch.6 extends this to multi-variable optimization (gradients + chain rule)`.

**Cross-track links** to ML chapters are standard. Always reference the specific chapter: `[ML 01-Regression/ch01-linear-regression](../../ml/01_regression/ch01_linear_regression)`.

---

### Callout Box System

Used consistently across all chapters. Must be used exactly this way — no improvised emoji or callout patterns:

| Symbol | Meaning | When to use |
|---|---|---|
| `💡` | Key insight / conceptual payoff | After a result that surprises or reframes understanding |
| `⚠️` | Warning / common trap | Before or immediately after a pattern often done wrong |
| `⚡` | Constraint connection | When content advances or validates one of the 3 free-kick constraints |
| `> 📖 **Optional:**` | Deeper derivation | Full proofs that break narrative flow |
| `> ➡️` | Forward pointer | When a concept needs to be planted before its full treatment |

The callout box content is always **actionable**: it ends with a Fix, a Rule, a What-to-do. No callout box that just says "this is interesting" without consequence.

---

## README Style — Detailed Conventions

- **Short paragraphs.** Target 3-5 sentences maximum per paragraph.
- **One-sentence-per-line math.** LaTeX formulas get their own line, followed immediately by verbal gloss.
- **No wall of symbols without inline definition.** Every term explained on first use.
- **Each chapter ends with 3–5 short exercises** that extend the knuckleball scenario (different angles, wind, slopes, etc.)
- **Pointer to ML chapter that reuses the material** (e.g., "This gradient appears in ML 01-Regression/ch01 as the core of linear regression training")
- **Open with an epigraph that is used by the chapter** (not decorative). If Newton quote about parabolas, the chapter must reference Newton's work specifically.
- **Derive every rule before stating it** — Taylor expansion justifies approximations, step-size bounds justify learning rates, convergence conditions justify stopping criteria.

---

### Section order

Two templates are allowed. Pick one and stay inside it.

**Template A — Standard chapter (Ch.1, 2, 4, 5).** Use this whenever the chapter has a clear single mathematical object to derive and exercise.

1. Core Idea (2–3 sentences, plain English)
2. Running Example (the free-kick twist for this chapter)
3. Math (derived, not dumped)
4. Step by Step (numbered recipe)
5. Key Diagram (single hero PNG in `img/` OR Mermaid flowchart)
6. Code Skeleton (6–20 lines, copy-pasteable)
7. What Can Go Wrong (3–5 common failure modes with Fixes)
8. Where This Reappears (explicit pointers to ML chapters + future Math chapters)
9. Progress Check (with visual)
10. Bridge to Next Chapter
11. Exercises (3–5, increasing difficulty)
12. References (Krohn, 3Blue1Brown, Strang, Goodfellow as primary sources)

**Template B — Math-heavy chapter (Ch.3, 6, 7).** Use when the chapter introduces several distinct mathematical objects (e.g. derivative *and* integral; gradient *and* chain rule *and* Hessian; three distributions *and* MLE) that each need their own derivation block. The hero PNG may sit immediately under the title (above the epigraph) when the chapter has no natural §5 home for it. Required sections in any order that flows: Core Idea, every named math object as its own section, a Worked Example or Step-by-Step, Pitfalls / What Can Go Wrong, Where This Reappears, References. Exercises and Code Skeleton are optional in Template B if the notebook already covers them — link to the notebook explicitly when omitting.

Whichever template you pick, every chapter must end with **Progress Check**, **Bridge to Next Chapter**, and **References**.

---

## Image and Animation Conventions

**Every image has a purpose — none are decorative.** Images demonstrate something the prose cannot fully convey: how a derivative changes slope, how parabola parameters affect trajectory, how gradient descent navigates a surface.

**Image naming convention:**
- `chNN-[topic]-[type].png/.gif` for chapter-specific generated images
- `[concept]_generated.gif/.png` for algorithmically generated animations  
- Descriptive alt-text is mandatory: `![Parabolic trajectory showing three launch angles (20°, 35°, 45°) with wall and crossbar constraints highlighted](img/ch02-trajectory-comparison.png)`

**Generated plots use dark background `facecolor="#1a1a2e"`** — matching the chapter's rendered dark theme. Light-background plots are not used.

**Image types:**

| Type | Purpose | Examples |
|---|---|---|
| GIF animation | Show process evolving: optimization steps, convergence | `gradient_descent_trajectory.gif`, `taylor_approximation_zoom.gif` |
| PNG comparison | Side-by-side before/after | `linear_vs_parabola_fit.png`, `derivative_vs_finite_difference.png` |
| PNG breakdown | Annotated diagram explaining one concept | `chain_rule_decomposition.png`, `matrix_multiply_structure.png` |
| PNG plot | Mathematical function visualization | `parabola_family.png`, `loss_surface_3d.png` |
| GIF needle | Chapter-level progress animation (needle moving toward constraint) | `ch03-calculus-needle.gif` |

**Every chapter has a needle GIF** — the chapter-level animation showing which constraint capability advanced. This appears immediately after §0 under the heading `## Animation`.

**Mermaid diagram colour palette** — used consistently for all flowcharts:
- Primary/concept: `fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff` (dark blue)
- Success/achieved: `fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff` (dark green)
- Caution/in-progress: `fill:#b45309,stroke:#e2e8f0,stroke-width:2px,color:#ffffff` (amber)
- Danger/blocked: `fill:#b91c1c,stroke:#e2e8f0,stroke-width:2px,color:#ffffff` (dark red)
- Info: `fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff` (medium blue)

All Mermaid nodes use white text (`color:#ffffff`) for legibility against dark backgrounds.

---

## Code Style

**Code blocks are minimal but complete.** The standard is "enough to run end-to-end with real output, nothing extra." No scaffolding classes, no type annotations on internal code, no error handling beyond what a practitioner would actually need.

**Variable naming is consistent across all chapters:**

| Variable | Meaning |
|---|---|
| `v0` | Initial velocity (m/s) |
| `theta` | Launch angle (degrees or radians — state units in comment) |
| `g` | Gravitational acceleration (9.8 m/s²) |
| `t` | Time (seconds) |
| `h`, `x` | Height and horizontal distance functions |
| `wall_dist` | Wall position (9.15m) |
| `goal_dist` | Goal position (20m) |
| `wall_height` | Wall height (1.8m) |
| `crossbar_height` | Crossbar height (2.44m) |

**Comments explain *why*, not *what*.** The code line `h_apex = v0_y**2 / (2*g)` does not need a comment saying "calculate apex height". It needs: `# max height when vertical velocity reaches zero (kinematic formula)`.

**The manual derivation loop always appears alongside the numpy vectorized version** in the Code Skeleton section. The manual version is labelled "Educational: step-by-step from first principles" and the numpy version is the practical reference.

**Example code structure:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Constants
g = 9.8  # m/s² — gravity
v0 = 20  # m/s — initial velocity
theta_deg = 35  # degrees — launch angle
theta = np.radians(theta_deg)

# Components
v0_x = v0 * np.cos(theta)  # horizontal velocity (constant)
v0_y = v0 * np.sin(theta)  # initial vertical velocity

# Trajectory function
def h(t):
    return v0_y * t - 0.5 * g * t**2

# Check constraints
t_wall = 9.15 / v0_x  # time to reach wall
h_wall = h(t_wall)
print(f"Wall clearance: {h_wall:.2f}m {'✅' if h_wall > 1.8 else '❌'}")
```

---

## Progress Check Section Format

The Progress Check is the penultimate section (before Bridge). It has a fixed format:

```markdown
## N · Progress Check — What We Can Solve Now

![Progress visualization](img/chNN-[slug]-progress-check.png)

✅ **Unlocked capabilities:**
- [Specific capability with exact numbers]
- [e.g., "Can now find trajectory apex: h_max = 5.10m at t = 1.02s"]
- [Constraint advancement: "Constraint #1 ✅ Wall clearance verified!"]

❌ **Still can't solve:**
- [Named gaps with specifics]
- [e.g., "❌ Can't find optimal angle yet — no way to maximize range"]

**Trajectory status**: [One-sentence summary: "We can now verify any single trajectory satisfies constraints, but can't find the best trajectory"]

**Next up:** Ch.X gives us **[concept]** — [what it unlocks for the free kick]
```

The progress visualization always shows the trajectory with color-coded regions: green for verified/optimized sections, red for blocked capabilities.

---

## What Can Go Wrong Section Format

**Required format:** 3–5 traps, each following the pattern:
- **Bold name of the trap** — one clause description in the heading
- Explanation in 2–3 sentences with concrete numbers from knuckleball parameters
- **Fix:** one actionable sentence starting with "`Fix:`"

**Example:**
```markdown
### **Degrees vs Radians Confusion**

You calculate sin(30) = -0.988 instead of 0.5. The trajectory predicts the ball goes *down* initially. Python's `math.sin()` expects radians, not degrees. At 30 radians (~1718°), you're in the wrong quadrant.

**Fix:** Always convert: `theta = np.radians(theta_deg)` or use `np.sin(np.deg2rad(30))`.

### **Ignoring Air Resistance**

Model predicts 40m range but real kicks barely reach 25m. Our ballistic model assumes vacuum. Real soccer balls experience drag proportional to v². For knuckleball (minimal spin), drag coefficient ≈ 0.2.

**Fix:** Treat as upper bound. Add drag term (-0.5 × C_d × v²) for realistic simulation. See [Advanced Ballistics] in references.
```

The section ends with a Mermaid diagnostic flowchart showing decision branches for common problems.

---

## Pedagogical Patterns & Teaching DNA

> **Source:** Adapted from ML track cross-chapter analysis and specialized for mathematical foundations. These are the implicit techniques that make chapters effective, beyond the explicit style rules.

### 1. Narrative Architecture Patterns

#### Pattern A: **Failure-First Discovery Arc**

**Rule:** New mathematical tools emerge from concrete breakdowns, never as a priori lists.

**Implementation:**
```
Act 1: Simple approach → Show where it breaks (with exact numbers)
Act 2: First fix → Show what IT breaks (new failure mode)
Act 3: Refined solution → Resolves tension
Act 4: Decision framework (when to use which)
```

**Example from trajectory analysis:**
- Linear model (Ch.1) → Works for 0.1s → Breaks at apex (can't model curvature)
- Parabola (Ch.2) → Models full arc → But can't find critical points
- Derivative (Ch.3) → Finds apex precisely → But can't optimize parameters
- Gradient (Ch.6) → Optimizes multi-variable problems → Resolves progression

**Anti-pattern:** Listing "Here are 5 ways to analyze trajectories" without demonstrating need.

#### Pattern B: **Historical Hook → Physics Stakes**

**Rule:** Every chapter opens with real person + real year + real problem, then immediately connects to knuckleball challenge.

**Template:**
```markdown
> **The story:** [Name] ([Year]) solved [specific problem] using [this technique]. 
> [One sentence on lasting impact]. [One sentence connecting to free kick].
>
> **Where you are:** Ch.[N-1] achieved [specific capability]. This chapter fixes [named blocker].
>
> **Notation in this chapter:** [Inline symbol declarations]
```

**Example:**
> Newton (1687) *Principia* — used calculus to prove planets follow ellipses → "Every trajectory calculation in ML — from gradient descent to robot arm motion — uses this same machinery"

#### Pattern C: **Three-Act Dramatic Structure**

**For:** Chapters introducing competing methods (Euler vs Runge-Kutta, symbolic vs numerical derivatives)

**Structure:**
- **Act 1:** Problem discovered (need derivatives)
- **Act 2:** Solution tested (finite differences work)
- **Act 3:** Solution refined (symbolic differentiation exact)

**Why effective:** Converts technical comparison into narrative with rising tension.

---

### 2. Concept Introduction Mechanics

#### Mechanism A: **Problem→Cost→Solution Pattern**

**Rule:** Every new technique appears AFTER showing:
1. The problem (specific failure with numbers)
2. The cost of ignoring it (constraint violation)
3. The solution (formula/algorithm that resolves it)

**Example:**
1. **Problem:** Can't find apex of h(t) = 10t - 4.9t² by guessing
2. **Cost:** Might clear wall but miss optimal trajectory — wastes shot
3. **Solution:** h'(t) = 0 gives t_apex = 1.02s exactly

**Anti-pattern:** "Here's the derivative formula..." (solution before problem).

#### Mechanism B: **"The Match Is Exact" Validation Loop**

**Rule:** After introducing any formula, immediately prove it works with hand-computable numbers.

**Template:**
```markdown
1. Formula in LaTeX
2. Specific scenario (v₀=20 m/s, θ=30°, t=0.6s)
3. Hand calculation step-by-step
4. Numerical result: **4.24m**
5. Confirmation: "At 0.6s, height is **4.24m** — clears 1.8m wall ✅"
```

**Why effective:** Builds trust before moving to abstraction. Readers verify the math themselves.

#### Mechanism C: **Geometric Visualizations with Narrative**

**Rule:** Every visualization needs a caption that interprets it, not just describes it.

**Example:**
> ![Three trajectories](img/ch02-angle-sweep.png)
> "Low angle (20°): hits wall. High angle (50°): clears but sails over crossbar. Middle angle (35°): sweet spot — thread the needle."

**Pattern:** Image + one-sentence insight that tells reader WHAT TO SEE.

---

### 3. Voice & Tone Engineering

#### Voice Rule A: **Practitioner Confession + Mathematical Rigor**

**Mix these modes fluidly:**
- **Confession:** "You could try 100 angles by hand. Or let calculus find it in one shot." (Ch.3)
- **Rigor:** Full derivations in `> 📖 Optional` boxes with proofs
- **Tutorial:** "Fix: Always use `np.radians()` for trig functions"

**Why effective:** Signals "this is for practitioners who also need to justify decisions to advisors."

#### Voice Rule B: **Tone Shifts by Section Function**

Map tone to pedagogical purpose:

| Section Type | Tone | Example |
|--------------|------|---------|
| Historical intro | Authoritative narrator | "Newton (1687)..." |
| Challenge setup | Direct practitioner | "You're 20 meters out. One shot." |
| Math derivation | Patient teacher | "Every parabola has three parameters: a, b, c..." |
| Failure moments | Conspiratorial peer | "The ball goes *down*? You're in radians hell." |
| Resolution | Confident guide | "Rule: derivatives find extrema exactly" |

#### Voice Rule C: **Dry Humor at Failure/Resolution Moments**

**When:** Humor appears at:
1. **Failure modes** — makes mistakes memorable
2. **Resolution moments** — celebrates insight

**When NOT:** During math derivations or complex proofs.

**Examples:**
- Failure: "sin(30) = -0.988? The ball tunnels underground." (degrees/radians trap)
- Resolution: "Newton and Leibniz fought for 20 years over who invented this. You just used it in 5 minutes." (derivative payoff)

**Pattern:** Irony, understatement, or mild exaggeration. Never jokes or puns.

---

### Anti-Patterns (What NOT to Do)

❌ **Listing formulas without demonstrating need**  
Example: "Here are five trajectory formulas: parabolic, ballistic, drag-adjusted..." (without showing when/why)

❌ **Formulas without verbal glossing**  
Example: Dropping LaTeX `h(t) = v₀t - ½gt²` with no "In English:" paragraph

❌ **Vague capability claims**  
Example: "The model got better" instead of "Can now verify wall clearance: 1.92m > 1.8m ✅"

❌ **Academic register**  
Example: "We demonstrate that...", "It can be shown that...", "In this section we will present..."

❌ **Synthetic toy scenarios**  
Example: Using `h(t) = 5t - 2t²` without physical grounding instead of actual knuckleball kinematics

❌ **Improvised emoji**  
Example: Using 🔍🎯✨🚀 as inline callouts (only 💡⚠️⚡📖➡️ allowed)

❌ **Topic-label section headings**  
Example: "## 3 · Math" instead of "## 3 · Math — How Parabolas Encode Initial Conditions"

❌ **Skipping numerical verification**  
Example: Showing formula, then immediately generalizing without computing a specific example

---

## Jupyter Notebook Template

Each notebook mirrors the README exactly — same sections, same order. The notebook adds:
- **Runnable cells**: every code block in the README is a cell in the notebook
- **Visual outputs**: `matplotlib` plots that generate the diagrams described in the README
- **Interactive widgets**: `ipywidgets` sliders + numeric inputs (bidirectionally synced) for exploring parameters
- **Exercises**: 2–3 cells at the end where the reader changes parameters and re-runs

Cell structure per notebook:

```
[markdown] Chapter title + one-liner
[markdown] ## 0 · The Challenge
[markdown] ## 1 · The Core Idea
[markdown] ## 2 · Running Example
[code]     Define knuckleball parameters (v0, theta, g, etc.)
[markdown] ## 3 · The Math
[code]     Implement the math (numpy + symbolic where needed)
[markdown] ## 4 · Step by Step
[code]     The step-by-step walkthrough as runnable code
[code]     Plotting the key diagram
[code]     Interactive widget (slider + text input synced)
[markdown] ## 5 · The Key Diagrams
[code]     Generate hero visualization
[markdown] ## 6 · Code Skeleton
[code]     Manual version (educational)
[code]     Vectorized version (practical)
[markdown] ## 7 · What Can Go Wrong
[code]     Demonstrate one trap (degrees/radians, etc.)
[markdown] ## 8 · Exercises
[code]     Exercise scaffolds (partially filled)
```

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

---

## Conformance Checklist for New or Revised Chapters

Before publishing any chapter, verify each item:

### Story & Structure
- [ ] Story header: real person, real year, real problem — and a bridge to ML practitioner's daily work
- [ ] §0 Challenge: THREE constraints stated (wall/crossbar/keeper), specific capabilities unlocked, named gaps
- [ ] Notation block in header: all symbols declared inline before §0 (not in separate table)
- [ ] Animation section: needle GIF showing constraint progress immediately after §0
- [ ] Bridge section: one clause what this chapter established + one clause what next chapter adds (knuckleball-grounded)

### Mathematical Style
- [ ] Every formula: verbally glossed within 3 lines (no LaTeX without "In English:" explanation)
- [ ] Every formula: scalar form shown first, vector/matrix form second with structural identity noted
- [ ] Every non-trivial formula: demonstrated on knuckleball parameters with explicit arithmetic (v₀=20m/s, θ=30°, etc.)
- [ ] Failure-first pedagogy: new concepts introduced because simpler one broke, not listed a priori
- [ ] Optional depth: full derivations behind `> 📖 Optional` callout boxes (Taylor series, Lagrange, epsilon-delta proofs)
- [ ] ASCII matrix diagrams: used for all matrix operations (show dimensions, explicit element-wise computation)

### Links & References
- [ ] Forward/backward links: every concept links to where it was introduced and where it reappears (ML chapters + future Math chapters)
- [ ] Callout boxes: only `💡 ⚠️ ⚡ 📖 ➡️` — no improvised emoji
- [ ] ML cross-references: specific chapter paths (e.g., `[ML 01-Regression/ch01](../../ml/01_regression/ch01_linear_regression)`)

### Visual Elements
- [ ] Mermaid diagrams: colour palette respected (dark blue primary, green success, amber caution, red blocked)
- [ ] Images: dark background (`facecolor="#1a1a2e"`), descriptive alt-text, purposeful (not decorative)
- [ ] Image naming: `chNN-[topic]-[type].png/.gif` or `[concept]_generated.gif`
- [ ] Needle GIF: chapter-level progress animation present and shows constraint advancement
- [ ] Hero diagram: at least one key visualization with interpretive caption (not just descriptive)

### Code
- [ ] Variable naming: `v0`, `theta`, `g`, `t`, `h`, `x`, `wall_dist`, `goal_dist` (consistent across chapters)
- [ ] Comments: explain *why*, not *what* (e.g., "# kinematic formula" not "# calculate height")
- [ ] Dual approach: manual step-by-step + vectorized equivalent in Code Skeleton
- [ ] Runnable: all code blocks copy-paste executable (or explicitly marked as pseudocode/conceptual)

### Progress & Traps
- [ ] Progress Check: ✅/❌ bullets with exact constraint numbers (e.g., "h(0.6s)=1.92m > 1.8m ✅")
- [ ] Progress Check: visual showing green (solvable) vs red (blocked) trajectory regions
- [ ] Progress Check: trajectory status one-liner + "Next up:" preview
- [ ] What Can Go Wrong: 3–5 traps with pattern **Bold trap** — explanation (2-3 sentences) → **Fix:** actionable sentence
- [ ] What Can Go Wrong: Mermaid diagnostic flowchart at end of section

### Voice & Register
- [ ] Second person: reader inside scenario at all times ("You're lining up the free kick...")
- [ ] No academic register: no "We demonstrate...", "It can be shown...", "In this section we will..."
- [ ] Dry humour: exactly once per major concept, at failure or resolution moments (never during derivations)
- [ ] Tone shifts: matches section function (authoritative history, direct challenge, patient math, conspiratorial traps)
- [ ] Contractions: used freely when they tighten sentences ("That's the derivative.", "Parabolas curve — lines don't.")

### Numerical Grounding
- [ ] Dataset: knuckleball parameters only (v₀=20m/s, θ typical 20-45°, wall at 9.15m/1.8m, goal at 20m/2.44m)
- [ ] No synthetic toy problems: always physically grounded (no `h(t)=5t-2t²` without physics context)
- [ ] Numerical verification: "The match is exact" pattern after every worked example
- [ ] Specific numbers: always exact ("1.92m > 1.8m ✅") never vague ("approximately cleared")

### Pedagogical Patterns
- [ ] Failure-first arc: tool → specific failure → minimal fix → that fix's failure → next tool (not taxonomy listing)
- [ ] Problem→Cost→Solution: every technique shown AFTER demonstrating need (specific failure + constraint violation)
- [ ] Historical hook → physics stakes: person + year → lasting impact → connection to free kick (3-sentence pattern)
- [ ] Forward pointers: `> ➡️` callouts plant concepts before full treatment (Ch.3 plants gradients for Ch.6)

### Notebook Alignment
- [ ] Widget system: every slider paired with bidirectional numeric text input (type or drag)
- [ ] Section mirroring: notebook follows README order exactly (parallel reading without context-switching)
- [ ] Interactive exploration: readers can change v₀, θ, g and see immediate trajectory updates
- [ ] Exercises: 2–3 cells for reader to modify parameters (different angles, wind effects, slope launch, etc.)

### Completeness
- [ ] All required sections present: Challenge (§0), Core Idea, Running Example, Math, Step-by-Step, Key Diagrams, Code Skeleton, What Can Go Wrong, Where This Reappears, Progress Check, Bridge, Exercises, References
- [ ] Template consistency: Template A (single object) or Template B (multiple objects) — not hybrid
- [ ] Section headings: descriptive (state conclusion) not labels (e.g., "## 3 · Math — How Parabolas Encode Initial Conditions" not "## 3 · Math")
- [ ] Ending trio: Progress Check → Bridge → Exercises → References (in that order)

---

## How to Use This Document

1. Open this file before authoring or revising any Math chapter
2. Use Ch.01 (Linear Algebra) and Ch.03 (Calculus Intro) as canonical style references for uncertain cases
3. Check the Conformance Checklist against your draft before committing
4. When in doubt, match the ML track's pattern (both tracks share pedagogical DNA) but always ground examples in knuckleball physics, not California Housing
5. Keep the knuckleball arc continuous — every chapter must state what constraints it advances

**Remember:** The goal is not documentation — it's building mathematical intuition through failure-first discovery, numerical grounding, and a coherent physical challenge that threads through all 7 chapters.
