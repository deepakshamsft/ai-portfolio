# Pre-Requisites — Authoring Plan

> **Purpose.** Before the ML book (`notes/ML/ch01…ch19`) teaches a model, the reader needs the *language* the model is written in: linear equations, non-linear composition, calculus, optimisation on curves, matrices, and matrix calculus. These chapters are the on-ramp. They assume high-school algebra and build to the point where Ch.5 of the ML book ("Backprop & Optimisers") reads like a natural next step instead of a cliff.
>
> **Audience.** Someone who has written Python and seen basic algebra but feels fuzzy every time a paper writes $\nabla_\theta \mathcal{L}$ or $\mathbf{X}^\top \mathbf{X}$.

---

## Progress

| Ch | Title | README | Notebook | Hero img | Status |
|---|---|---|---|---|---|
| 1 | Linear Algebra — Lines, Weights, Biases | ✅ | ✅ | ✅ | **Done** |
| 2 | Non-Linear Algebra — Polynomials & Feature Expansion | ✅ | ✅ | ✅ | **Done** |
| 3 | Calculus — Derivatives, Integrals, FTC | ✅ | ✅ | ✅ | **Done** |
| 4 | Small Steps on a Curve (iterative optimisation) | ✅ | ✅ | ✅ | **Done** |
| 5 | Matrices & Matrix Calculus | ✅ | ✅ | ✅ | **Done** |
| 6 | Gradient + Matrix Chain Rule | ✅ | ✅ | ✅ | **Done** |
| 7 | Probability & Statistics | ✅ | ✅ | ✅ | **Done** |

### Authoring notes collected from Ch.1–3

- **ipywidgets pattern that works well.** A `linked_pair(label, value, min, max)` helper returning a `(FloatSlider, FloatText)` pair linked with `jslink((sl,'value'),(tx,'value'))` → bidirectional binding with zero observers. Copy it into every chapter's notebook.
- **matplotlib mathtext gotchas.** No `\tfrac`, `\Big`, `\Bigg`, `\underbrace`, `\overbrace`. Use `\frac` and plain inline. `ax.text(...)` does **not** accept `letterspacing` — kwarg error at render time. Keep figure-text annotations inside the axes (y-coord ≥ 0 in `transAxes`) or they render at figure bottom.
- **Teaching-trick for convergence demos.** A midpoint Riemann rule is *exact* on linear integrands; use **left-endpoint** when you want a visible n → ∞ convergence on $v(t) = v_{0y} - g t$. (Caught and fixed in Ch.3.)
- **3-D feature-space plot.** In `mpl_toolkits.mplot3d`, label pads default to bad positions — set `labelpad=2` on `x/y/zlabel` and give the subplot adequate `figsize` (≥ 4.5 inches wide per 3-D axis) or labels clip.
- **Colour palette standardised.** `DARK=#2C3E50, BLUE=#2E86C1, GREEN=#27AE60, PURPLE=#8E44AD, ORANGE=#E67E22, RED=#E74C3C, GREY=#7F8C8D, GOLD=#F39C12`. Re-use across Ch.4–7.
- **Chapter effort.** Ch.1–3 each: one 3-panel hero PNG, ~10-section README (1500–2000 words), notebook with 4–6 interactive widgets. Budget ~2 dense agent turns per chapter.
- **Running-thread fit.** The free-throw physics cleanly yields: Ch.1 (straight-line approximation, first 0.2 s), Ch.2 (full parabola), Ch.3 (velocity = derivative; rise = integral). Ch.4 will use "range vs release-angle" as its 1-D optimisation landscape — continuity preserved.

---

## Running thread — "The Perfect Free Throw"

Every chapter uses the **same** real-world problem: a basketball player wants to sink a free throw from 4.57 m (15 ft) at a 3.05 m (10 ft) hoop. What release angle, velocity, and spin gets it in? What if there is wind, a defender, or a new court with a different floor?

This thread was chosen because it is the problem that *forced* Newton and Leibniz to invent calculus in the 1660s–1680s, so the history of the mathematics and the history of the example line up exactly.

| Chapter | Role the free-throw plays |
|---|---|
| Ch.1 Linear Algebra | Height as a *linear* function of time — horizontal-only toy model |
| Ch.2 Non-linear Algebra | Real trajectory is parabolic: $y = v_{0y}t - \tfrac{1}{2}gt^2$ — polynomial features |
| Ch.3 Calculus Intro | Instantaneous velocity = slope of position curve; area under velocity curve = distance |
| Ch.4 Small Steps on a Curve | Walk along "range vs. angle" curve to find the 45° optimum |
| Ch.5 Matrices & Matrix Calculus | Add wind, air-drag, spin, release height → 5-variable system → matrix form |
| Ch.6 Derivatives × Matrices | Gradient of range w.r.t. all 5 variables at once; teaser for gradient descent |
| Ch.7 Probability & Statistics *(added)* | Your release is noisy — model shot outcomes as a distribution |

---

## Chapter roadmap

Every chapter follows the existing repo convention:

```
notes/PreRequisites/chNN-<slug>/
  README.md        # narrative + math + diagrams
  notebook.ipynb   # hands-on Python: plots, widgets, exercises
  img/             # static PNGs referenced from README
```

### Ch.1 — Linear Algebra: Lines, Weights, Biases

**Real-world hook.** In the first 0.2 s after release, the ball rises almost in a straight line — a useful linearisation.

**Concepts.**
- The equation of a line: $y = wx + b$, both in "slope-intercept" and "weight-bias" framing.
- Vectors as arrows; vectors as columns of numbers.
- Dot product as weighted sum; `np.dot` as one line of Python.
- Why ML calls $w$ the *weight* and $b$ the *bias*: the weight scales the input, the bias shifts the whole line.

**Notebook.** Interactive matplotlib (ipywidgets sliders + editable text boxes) that moves a line as the user drags $w$ and $b$. Numeric text boxes sync bidirectionally with the sliders. A second cell overlays a scatter of real free-throw (time, height) samples and lets the reader *manually* fit a line — motivating the need for Ch.3/Ch.4 (there has to be a better way to pick $w, b$).

**Reference.** [Jon Krohn — Linear Algebra for Machine Learning (YouTube course)](https://www.jonkrohn.com/posts/2021/5/9/linear-algebra-for-machine-learning-complete-math-course-on-youtube) — the primary external video reference for this chapter.

---

### Ch.2 — Non-Linear Algebra: Polynomials and the Feature-Expansion Trick

**Real-world hook.** Gravity bends the trajectory into a parabola: $y(t) = v_{0y} t - \tfrac{1}{2} g t^2$. A straight line cannot fit this.

**Concepts.**
- Polynomials of degree 2, 3, …, $n$. What each term controls geometrically.
- **The key insight (your intuition, confirmed):** $ax^2 + bx + c$ is *non-linear in $x$* but *linear in the parameters* $(a, b, c)$. Substitute $x_1 = x^2$, $x_2 = x$ and you get $a x_1 + b x_2 + c$ — a plain multi-variable linear model. This is called **basis expansion** or **polynomial features**.
- Why that matters: *every linear regression library can fit any polynomial* by engineering features first. `sklearn.preprocessing.PolynomialFeatures` is literally this one trick.
- Brief mention of other basis expansions: piecewise polynomials (splines), radial basis functions, Fourier features.
- What truly non-linear-in-parameters models look like (neural networks) and why they need a different approach — teaser for the ML book.

**Notebook.** Same widget pattern as Ch.1, but now three sliders ($a, b, c$) shape a parabola. Overlay real projectile data. Then a second widget shows the same parabola as a flat *plane* in $(x_1, x_2, y)$ space once you substitute $x_1 = x^2$ — viscerally demonstrating "polynomial in 1-D = linear in 2-D feature space".

---

### Ch.3 — Calculus: Derivatives and Integrals from Scratch

**Real-world hook.** At the top of the arc the ball is momentarily *stationary vertically*. What is its velocity there? Average velocity over a second is useless — we need an *instantaneous* velocity. That question is why calculus exists.

**Concepts.**
- **History and motivation (with honest dates).** Newton (1665–1687, fluxions, *Principia*) and Leibniz (1675–1684, differentials, the $dy/dx$ notation we still use) independently invent the derivative. The *specific* problems that forced it: projectile motion (Galileo → Newton), planetary orbits (Kepler → Newton), finding tangents to arbitrary curves (Fermat, Descartes), and computing areas of curved regions (Archimedes → Cavalieri → Newton).
- **The derivative** as: (1) limit of a secant slope $\tfrac{f(x+h)-f(x)}{h}$ as $h \to 0$, (2) instantaneous rate of change, (3) best local linear approximation $f(x+h) \approx f(x) + f'(x)\,h$.
- **The integral** as: (1) limit of Riemann sums $\sum f(x_i)\,\Delta x$ as $\Delta x \to 0$, (2) total accumulation, (3) the inverse of differentiation (Fundamental Theorem of Calculus).
- **Why "break a curve into infinitesimals" works.** Concrete demo: approximate a circle's area by polygons ($n=4, 8, 16, 64$) and watch the error shrink; approximate the free-throw arc-length similarly.

**Notebook.** Animation: a secant line on $y(t) = v_{0y} t - \tfrac{1}{2} g t^2$ collapses into a tangent as $h \to 0$. Side-by-side: Riemann-sum rectangles under the velocity curve filling in to give the exact displacement. No ML yet.

---

### Ch.4 — Finding Minima by Taking Small Steps

**Real-world hook.** You want the release angle $\theta$ that *maximises* the horizontal range from a fixed release speed. You can solve it analytically (the answer is 45° in a vacuum), but what if air drag makes the equation nasty? Then you have to **walk down the curve**.

**Concepts.**
- Critical points: $f'(\theta) = 0$. Maxima, minima, saddle points.
- The iterative update $\theta \leftarrow \theta - \eta\, f'(\theta)$ — but framed as "take a small step opposite the slope", not as "gradient descent". The word *gradient* is deferred to Ch.6.
- Step-size intuition: too small → crawls, too big → overshoots and oscillates. (Same intuition you will see again in ML Ch.5, but discovered here first on a 1-D curve.)
- Convex vs. non-convex: one valley vs. many. Why a single start is risky in non-convex problems.

**Notebook.** Widget that drops a ball onto the range-vs-angle curve and animates the walk. Sliders for start position and step size. A toggle switches the curve from "no drag" (smooth parabola-like) to "with drag" (bumpy, multiple local maxima) so the reader *sees* convexity matter.

---

### Ch.5 — Matrices, Linear Systems, and Matrix Calculus

**Real-world hook.** Real coaching data does not have one variable — it has release angle, release velocity, release height, wind speed, wind direction, ball spin, court altitude… Eight variables, 500 recorded shots. You cannot handle this with scalars.

**Concepts.**
- **Why matrices.** A matrix is a compact way to write "apply the same linear rule to many inputs, or many rules to the same input, at once". Exchange rates, rotations, and regression designs are all linear maps; they are all matrices.
- **The three views of $Ax = b$:** (1) row picture — intersection of hyperplanes, (2) column picture — $b$ as a linear combination of $A$'s columns, (3) transformation picture — $A$ stretches/rotates/skews the space.
- **Core operations on paper and in NumPy:** addition, scalar multiply, matrix-vector product, matrix-matrix product, transpose, identity, inverse (when it exists). Shapes and why they matter more than values while you are learning.
- **Normal equations for regression:** $\hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$ — derived and applied to the free-throw data.
- **Matrix calculus, starter kit.** $\nabla_x (\mathbf{a}^\top x) = \mathbf{a}$, $\nabla_x (x^\top \mathbf{A} x) = (\mathbf{A} + \mathbf{A}^\top)x$, chain rule in matrix form. Just the rules needed to read the ML book.

**Notebook.** (a) Solve $\mathbf{X}\mathbf{w} = \mathbf{y}$ for the free-throw dataset with `np.linalg.lstsq`. (b) Visualise a $2\times 2$ matrix acting on the unit square — grid-distortion animation. (c) Derive and verify $(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ numerically against the widget-fit from Ch.1.

---

### Ch.6 — Derivatives × Matrices: the Gradient, and a Teaser for Gradient Descent

**Real-world hook.** You now have 8 coaching variables and a loss function $\mathcal{L}(\mathbf{w})$ that measures "how often this shot misses". Each variable has its own slope. Stack them into a single vector and you have the **gradient**.

**Concepts.**
- **Partial derivatives.** $\partial f / \partial w_i$ — slope holding every other variable fixed.
- **The gradient** $\nabla f = (\partial f/\partial w_1, \ldots, \partial f/\partial w_d)$ as a vector that points uphill, with magnitude equal to the steepest slope.
- **Why the gradient lives in the same space as $\mathbf{w}$.** Why the update $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla f$ is a direct generalisation of Ch.4's single-variable step.
- **Jacobian and Hessian** — one-page definitions; when you will meet them again.
- **Chain rule, matrix version.** $\nabla_x (g \circ f) = J_f(x)^\top \nabla g(f(x))$. This is the equation behind backpropagation. The reader will see it in full with neurons in ML Ch.5.
- **Teaser.** Now the reader is armed to read ML Ch.1 (Linear Regression) *and* the derivation of the normal equations *and* the gradient-descent derivation — in one pass.

**Notebook.** (a) 3-D surface plot of a 2-variable loss $\mathcal{L}(w_1, w_2)$ with gradient arrows as a quiver plot. (b) Walk down it with the Ch.4 update, now vector-valued — same mechanics, higher dimension. (c) Verify numerically that the analytic gradient equals the finite-difference gradient (this is how every deep-learning library tests its autodiff).

---

### Ch.7 — Probability & Statistics: Samples, Distributions, Likelihood *(added)*

**Real-world hook.** Real free throws are noisy: the same intended shot lands differently every time. You need to reason about *distributions* of outcomes, not single values.

**Why added.** Every loss function in the ML book is derived from a probability assumption. MSE assumes Gaussian noise (ML Ch.15 proves this). Cross-entropy comes from the Bernoulli / categorical likelihood. A reader who has not seen expectation, variance, and likelihood will hit a wall at ML Ch.15. One 6-page chapter at the end of Pre-Reqs closes that gap.

**Concepts.**
- Random variables, discrete vs continuous. PMF vs PDF.
- Expectation, variance, standard deviation — operationally and visually.
- The three distributions a beginner must know cold: Bernoulli (shot makes it or not), Binomial (how many of N shots go in), Gaussian (release-velocity noise).
- **Likelihood.** Given shots $y_1, \ldots, y_n$, the likelihood of parameters $\theta$ is $\mathcal{L}(\theta) = \prod p(y_i \mid \theta)$. Maximising this is **MLE** — the entire subject of ML Ch.15. This chapter is the prerequisite.
- Law of Large Numbers and the Central Limit Theorem in one paragraph each.

**Notebook.** Simulate 10 000 noisy free throws; show empirical histogram converging to the theoretical Gaussian. Fit the mean and variance by MLE; compare to `np.mean`, `np.var`.

---

## Historical and chronological evolution (README section)

The Pre-Requisites `README.md` will include a timeline tying the mathematics to the people and problems that produced it — so the reader sees the subject *in motion*, not as a finished textbook.

- **c. 300 BCE — Euclid's *Elements*.** First axiomatic treatment of geometry; lines and planes defined precisely. Still the root of "linear" (`Ch.1`).
- **c. 250 BCE — Archimedes.** *Method of Exhaustion* for areas and volumes of curved shapes — the intellectual ancestor of the integral (`Ch.3`).
- **c. 820 — al-Khwārizmī, *al-jabr*.** The word *algebra*, and the first systematic treatment of linear and quadratic equations (`Ch.1, Ch.2`).
- **1600s — Descartes & Fermat.** Coordinate geometry; equations become curves and vice versa. Polynomials gain a geometric meaning (`Ch.2`).
- **1665–1687 — Newton.** Fluxions; *Principia Mathematica*; projectile motion and orbits solved with what we now call calculus (`Ch.3, Ch.4`).
- **1675–1684 — Leibniz.** Differential and integral notation ($dy/dx$, $\int$) — the notation we still use (`Ch.3`).
- **1750s — Euler, Lagrange.** Calculus of variations; derivatives extended to functions of functions (`Ch.4`).
- **1800s — Cauchy, Weierstrass, Riemann.** Limits made rigorous; Riemann sums; modern definition of the integral (`Ch.3`).
- **1850s–1900s — Cayley, Sylvester.** Matrix algebra formalised as its own object (`Ch.5`).
- **1901 — Karl Pearson.** Least-squares regression popularised; the normal equations are in use (`Ch.5`).
- **1930s — Kolmogorov.** Probability given its measure-theoretic foundation (`Ch.7`).
- **1847 / 1951 — Cauchy / Kiefer–Wolfowitz.** Gradient descent as a method (`Ch.6`).
- **1960s — numerical linear algebra (Householder, Wilkinson).** $\mathbf{X}^\top\mathbf{X}$ solved stably at scale; modern BLAS is their direct descendant (`Ch.5`).
- **1986 — Rumelhart, Hinton, Williams.** Backpropagation — the chain rule of `Ch.6` applied recursively through a neural network. The ML book picks up here.

Reading these dates in order, the story is: **linear equations → curves → the calculus that tames curves → matrices that pack many curves together → gradients that navigate them → probability that accepts the noise**. Every one of those steps addresses a problem that had no satisfactory prior answer. That is the spirit to carry into each chapter.

---

## External references (to live in the folder README)

- **Jon Krohn — *Linear Algebra for Machine Learning* (YouTube course).** <https://www.jonkrohn.com/posts/2021/5/9/linear-algebra-for-machine-learning-complete-math-course-on-youtube> — primary video companion for Ch.1 and Ch.5.
- 3Blue1Brown — *Essence of Linear Algebra* and *Essence of Calculus* (YouTube playlists) — visual intuition for Ch.1, Ch.3, Ch.5.
- Gilbert Strang — *Introduction to Linear Algebra* (MIT OCW 18.06) — the definitive text for Ch.5.
- Goodfellow, Bengio, Courville — *Deep Learning*, Chs. 2–4 — gap-filler between Pre-Reqs and the ML book.

---

## Authoring conventions (apply to every chapter)

1. **Length target.** 1 200–1 800 words of README prose per chapter + notebook.
2. **Section skeleton** (README):
   1. Core idea — one paragraph, the mental model.
   2. Running example — the specific free-throw question this chapter answers.
   3. Math — full derivation with all steps.
   4. Step by step — algorithmic recipe in plain English.
   5. Key diagrams — PNGs in `img/` with captions.
   6. Code walkthrough — cross-reference the notebook cells.
   7. What can go wrong — traps and misconceptions.
   8. Exercises — 3–5 short, answerable-in-5-min problems.
   9. References — book/video sections for this chapter.
3. **Notebook conventions.**
   - First cell: markdown title + learning outcomes.
   - Imports in cell 2: `numpy`, `matplotlib`, `ipywidgets`, `scipy` as needed.
   - Every interactive widget has an editable text input bound *bidirectionally* to its slider, per user request.
   - Final cell: "Where this reappears in the ML book" cross-reference.
4. **Figures.** Matplotlib, dpi=150, Agg backend for any script-generated PNGs in `img/`. Keep mathtext within the subset proven safe in `notes/Reference/` (no `\Big`, `\underbrace`, `\overbrace`).
5. **Cross-links.** Every chapter closes with one-line pointers to the ML chapter(s) that pick the thread back up.

---

## Open questions for the reviewer before content writing begins

1. Folder naming — `PreRequisites` (CamelCase, matches `MultiAgentAI`) vs `prerequisites` (lowercase, matches `ML`). Current plan uses `PreRequisites`.
2. Should Ch.7 Probability be in-scope for this book, or deferred? Current plan: **include it** — without it, ML Ch.15 has a hole.
3. Any appetite for a Ch.0 "Setting up Python + NumPy + ipywidgets" zero-prereq chapter? Default: no; rely on the existing repo-level setup docs.
4. Notebook interactivity target — pure `ipywidgets` (works in classic Jupyter and VS Code) vs `plotly` (prettier, but heavier). Current plan: **ipywidgets**.
