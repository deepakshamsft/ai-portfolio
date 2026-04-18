# Pre-Requisites — Mathematical Foundations for ML

> The on-ramp to the rest of the repo. If `notes/ML/` opens with $\hat{y} = wx + b$ and the symbols already feel slippery, start here. Seven short chapters that walk from the equation of a line to the matrix chain rule — the mathematics that every later chapter silently assumes.

---

## Who this is for

- You have written Python and seen basic algebra.
- You have *heard* of derivatives, matrices, and gradients but they do not feel like tools you reach for.
- You want to read an ML paper and have $\nabla_\theta \mathcal{L}$ mean something concrete.

If that is you, read these seven chapters in order, run every notebook. If not, skip to [`notes/ML/`](../ML/README.md) and come back only when a symbol bites.

---

## The running thread — "The Perfect Knuckleball Free Kick"

Every chapter uses the *same* real-world problem: a football (soccer) striker lines up a direct free kick 20 m from goal, aiming to clear a defensive wall at 9.15 m and dip the ball under the 2.44 m crossbar. Because it's struck as a **knuckleball** — almost zero spin, Juninho/Pirlo/Ronaldo style — the ball's path is dominated by gravity alone, so we can model it as a clean 2D parabola and leave the Magnus curve out of the story entirely. What strike speed, angle, and foot contact get the ball in? What changes when the wall gets taller, the pitch gets wet, or the kicker fatigues in the 89th minute?

This thread was picked deliberately — projectile motion is the problem that *forced* Newton and Leibniz to invent calculus in the 1660s–80s, so the mathematics and the example grew up together. Each chapter makes the problem one step more realistic, and in doing so adds exactly one new piece of mathematics to the reader's toolkit.

| Chapter | What we add | Free-kick question it answers |
|---|---|---|
| Ch.1 Linear Algebra | lines, weights, biases | What's the ball's height during the first 0.1 s off the boot? |
| Ch.2 Non-linear Algebra | polynomials, feature expansion | What's the *full* parabolic trajectory from boot to goal? |
| Ch.3 Calculus Intro | derivatives + integrals | What's the ball's instantaneous velocity at the apex? |
| Ch.4 Small Steps on a Curve | iterative optimisation | What launch angle drives the ball the farthest (long goal kick)? |
| Ch.5 Matrices | linear algebra at scale | How do we handle wind + wall size + 6 other variables at once? |
| Ch.6 Derivatives × Matrices | gradient, Jacobian, chain rule | Which variable should we adjust, and by how much? |
| Ch.7 Probability & Statistics | distributions, likelihood | How do we reason about a *noisy* strike? |

---

## Roadmap — chapters and what each delivers

### [Ch.1 — Linear Algebra: Lines, Weights, Biases](./ch01-linear-algebra/README.md)
**Mental model:** a line is a two-parameter object. The parameters are a weight (slope) and a bias (offset). Vectors are lists of numbers; the dot product is a weighted sum.
**Artifact:** an interactive plot where you drag $w$ and $b$ and watch the line re-orient, plus a scatter of (time, height) samples the reader fits *by hand*.

### [Ch.2 — Non-linear Algebra: Polynomials and the Feature-Expansion Trick](./ch02-nonlinear-algebra/README.md)
**Mental model:** $ax^2 + bx + c$ is non-linear in $x$ but linear in $(a, b, c)$. Substitute $x_1 = x^2, x_2 = x$ and you've turned a curve into a plane in 2-D feature space. This single trick is how "linear" models fit non-linear data.
**Artifact:** slider-driven parabola fitting a real projectile, plus a 3-D view of the same parabola as a flat plane in $(x_1, x_2, y)$.

### [Ch.3 — Calculus: Derivatives and Integrals from Scratch](./ch03-calculus-intro/README.md)
**Mental model:** a derivative is the slope of a curve at one point; an integral is the area under it. Both are limits — a secant collapsing into a tangent, rectangles shrinking to the curve. The Fundamental Theorem says they're inverses of each other.
**Artifact:** a secant-to-tangent animation and a Riemann-sum accumulator on the free-kick trajectory.

### [Ch.4 — Small Steps on a Curve](./ch04-small-steps/README.md)
**Mental model:** when you can't solve $f'(x) = 0$ analytically, walk downhill. The update $x \leftarrow x - \eta f'(x)$ converges if the step size is right — and if the landscape has only one valley. This is gradient descent one dimension early, with all the warts (step-size tuning, non-convexity, local optima) visible.
**Artifact:** start-angle and step-size sliders on a long-goal-kick range curve, plus a basin-of-attraction map on a wind-affected non-convex version.

### [Ch.5 — Matrices, Linear Systems, and Matrix Calculus](./ch05-matrices/README.md)
**Mental model:** a matrix is a linear map. $A\mathbf{x} = \mathbf{b}$ has three views — row (intersecting hyperplanes), column (weighted sum of columns), transformation (warp of space). Normal equations $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top \mathbf{y}$ are just high-dimensional line-fitting.
**Artifact:** a $2 \times 2$ matrix-warping widget with live determinant, plus the full free-kick parabola fitted in one `lstsq` call with physics constants read straight off the weight vector.

### [Ch.6 — Gradient + Matrix Chain Rule](./ch06-gradient-chain-rule/README.md)
**Mental model:** the gradient $\nabla f$ packs every partial derivative into a vector that points uphill. The matrix chain rule $\nabla_\mathbf{x}(g \circ f) = J_f^\top\,\nabla g$ is the single equation behind every `.backward()` call in PyTorch — reverse-mode autodiff is just that product evaluated right-to-left.
**Artifact:** a 2-D descent widget with tunable $(\theta_1^{(0)}, \theta_2^{(0)}, \eta, \text{iters})$, a one-layer neural-net shape drill, finite-difference and PyTorch `autograd` cross-checks, plus a forward-vs-reverse-mode benchmark showing reverse mode's $d$-fold speed-up on deep stacks.

### [Ch.7 — Probability & Statistics](./ch07-probability-statistics/README.md)
**Mental model:** expectation, variance, likelihood. Mean-squared error isn't a design choice — it's what Gaussian noise mathematically demands. Change the noise model and you change the loss: Gaussian→MSE, Laplace→MAE, Bernoulli→cross-entropy. Every supervised loss in ML Ch.15 falls out of this one principle.
**Artifact:** an interactive CLT widget (switch source distribution and batch size, watch the sample-mean histogram morph to a Gaussian), Gaussian MLE by closed form and grid search agreeing exactly, a confirmation that OLS on the free-kick parabola equals its Gaussian MLE, and a mean-vs-median robustness demo showing why swapping noise models swaps loss functions.

---

## Historical and chronological evolution

Reading the mathematics in the order it was *discovered* makes it stick. Every concept below solved a problem that had no prior answer — understanding *which* problem is half the intuition.

| Year(s) | Mathematician | Contribution | Ties to |
|---|---|---|---|
| c. 300 BCE | **Euclid** | *Elements* — first axiomatic geometry; lines and planes defined precisely. The word *linear* descends from here. | Ch.1 |
| c. 250 BCE | **Archimedes** | Method of Exhaustion — areas and volumes of curved shapes by squeezing polygons. Intellectual ancestor of the integral. | Ch.3 |
| c. 820 | **al-Khwārizmī** | *al-Kitāb al-mukhtaṣar fī ḥisāb al-jabr wa-l-muqābala* — the word *algebra* and the first systematic treatment of linear and quadratic equations. | Ch.1, Ch.2 |
| 1630s | **Descartes, Fermat** | Coordinate geometry — equations become curves, curves become equations. Polynomials gain a geometric meaning. | Ch.2 |
| 1665–1687 | **Newton** | Fluxions; *Principia Mathematica* — projectile motion and planetary orbits solved with the first calculus. | Ch.3, Ch.4 |
| 1675–1684 | **Leibniz** | Independent invention of differential and integral calculus. The notation $dy/dx$ and $\int$ we still use is his. | Ch.3 |
| 1750s | **Euler, Lagrange** | Calculus of variations — derivatives of *functionals*. Multivariable calculus matures. | Ch.4, Ch.6 |
| 1800s | **Cauchy, Weierstrass, Riemann** | Limits made rigorous; Riemann integral; the modern $\varepsilon$–$\delta$ definitions. | Ch.3 |
| 1847 | **Cauchy** | Method of steepest descent — iterative minimisation by following the gradient. | Ch.4, Ch.6 |
| 1850s–1900s | **Cayley, Sylvester** | Matrix algebra established as an object of study with its own laws. | Ch.5 |
| 1901 | **Karl Pearson** | Least-squares regression popularised; normal equations in routine use. | Ch.5 |
| 1930s | **Kolmogorov** | Probability given its measure-theoretic foundation. | Ch.7 |
| 1951 | **Kiefer, Wolfowitz** | Stochastic approximation — gradient descent with noisy gradients. | Ch.4, Ch.6 |
| 1960s | **Householder, Wilkinson** | Numerical linear algebra — $X^\top X$ solved stably at scale. Modern BLAS is the direct descendant. | Ch.5 |
| 1986 | **Rumelhart, Hinton, Williams** | Backpropagation — the matrix chain rule applied recursively. This is where the ML book picks up. | Ch.6 → ML Ch.5 |

**Story arc.** Linear equations → curves → calculus that tames curves → matrices that pack many curves together → gradients that navigate them → probability that accepts the noise. Every step addresses a specific problem that had no satisfactory prior answer. Keep that arc in your head as you read.

---

## Authoring conventions

See [AUTHORING_GUIDE.md](./AUTHORING_GUIDE.md) for chapter folder layout, README/notebook style rules, section order, and the running-thread convention.

---

## External references

- **Jon Krohn — *Linear Algebra for Machine Learning* (YouTube course).** <https://www.jonkrohn.com/posts/2021/5/9/linear-algebra-for-machine-learning-complete-math-course-on-youtube> — primary video companion for Ch.1 and Ch.5. Free, paced for self-study, covers exactly the subset of linear algebra that appears in neural networks.
- **3Blue1Brown — *Essence of Linear Algebra* and *Essence of Calculus* (YouTube playlists).** The visual companion to Ch.1, Ch.3, and Ch.5. If a concept here still feels abstract, watch the matching 3B1B video.
- **Gilbert Strang — *Introduction to Linear Algebra* (MIT OCW 18.06).** The definitive long-form text for Ch.5.
- **Goodfellow, Bengio, Courville — *Deep Learning*, Chs. 2–4.** The gap-filler between Pre-Reqs and the ML book.

---

## Where to go next

Once Ch.1–7 feel comfortable, open [`notes/ML/ch01-linear-regression/`](../ML/ch01-linear-regression/README.md). Every chapter of the ML book has a one-line pointer back to the Pre-Req it sits on, so you can always check where a symbol was introduced.
