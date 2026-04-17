# Ch.3 — Calculus: Slopes, Areas, and the Meeting in the Middle

> **Running theme.** We want to know two things about the free throw:
> *how fast is the ball rising at any instant*, and
> *how high has it gone by time $T$*.
> Both questions live outside algebra. Calculus — derivative for the first, integral for the second — was invented to answer them.

---

## 1 · Core Idea

Calculus rests on a single idea repeated in two disguises: **replace a curved thing with infinitely many tiny straight things, and take a limit.**

- **Derivative** = slope of the tangent = limit of secant slopes as two points collapse into one.
- **Integral** = area under a curve = limit of a sum of rectangle areas as the rectangles get arbitrarily thin.

The stunning fact — the **Fundamental Theorem of Calculus** — is that these two operations are inverses of each other.

---

## 2 · Running Example

Free throw, vertical component, release at ground level for clarity:

$$h(t) \;=\; v_{0y}\,t - \tfrac{1}{2}\,g\,t^2 \qquad v_{0y} = 7.2\text{ m/s},\ g = 9.81\text{ m/s}^2$$

Two questions:

1. **Instantaneous vertical velocity $v(t)$ at any time $t$.** The ball slows as it rises, stops for an instant at the apex, then accelerates downward. What is $v$ *exactly* at $t = 0.5$ s?
2. **Total distance risen from release to apex.** Can we recover $h(t)$ from $v(t)$ by summing up the tiny rises?

Both questions look innocent until you realise $h(t)$ is a curve, not a line, so simple division and multiplication fail.

---

## 3 · Historical Detour — How We Got Here

Calculus was not invented in a day.

| Year | Thinker | Contribution |
|---|---|---|
| ~250 BC | Archimedes | **Method of exhaustion** — approximate a circle's area with inscribed/circumscribed polygons. Same idea as a Riemann sum, 1900 years early. |
| 1630s | Fermat | Tangent lines via "adequality" — set $\varepsilon$ small, compute, drop $\varepsilon$. The derivative in embryo. |
| 1660s | Wallis, Barrow | Area under a curve via systematic integration rules. |
| 1665–1687 | **Newton** | "Fluxions" — time-based derivative notation $\dot{x}$. Used calculus to derive Kepler's planetary laws from gravity. |
| 1675–1684 | **Leibniz** | "Differentials" — modern $dy/dx$ and $\int$ notation. Cleaner, taught today. |
| 1823 | Cauchy | Limits made rigorous. Finally answered "what is $dx$?" |
| 1872 | Weierstrass | $\varepsilon$–$\delta$ formalisation. Calculus becomes a real branch of maths, not just a tool. |

By the time machine learning arrived (1950s onward), calculus was a settled language. Gradient descent, backprop, every loss landscape — all of it rests on Newton and Leibniz.

---

## 4 · The Derivative — Slope of the Tangent

### 4.1 · Secant to tangent

Pick a point $t_0$ on the curve. Pick another point $t_0 + \Delta t$. The **secant slope** between them is

$$\text{secant slope} \;=\; \frac{h(t_0 + \Delta t) - h(t_0)}{\Delta t}$$

That's rise over run — the Ch.1 idea. Now shrink $\Delta t$ toward zero. The secant rotates into a **tangent** that touches the curve at just one point. Its slope — if the limit exists — is the **derivative** at $t_0$:

$$h'(t_0) \;=\; \lim_{\Delta t \to 0} \; \frac{h(t_0 + \Delta t) - h(t_0)}{\Delta t}$$

For our free throw $h(t) = v_{0y}\,t - \tfrac{1}{2}g t^2$, algebra (expand, subtract, simplify, let $\Delta t \to 0$) gives

$$h'(t) \;=\; v_{0y} - g\,t$$

This is the **instantaneous vertical velocity** $v(t)$. The apex is where $v(t) = 0$, i.e. $t_\star = v_{0y}/g$. For $v_{0y} = 7.2$, $t_\star \approx 0.734$ s.

### 4.2 · Derivative rules you will actually use

Memorise five and you cover 90% of cases in the ML book:

| Function | Derivative |
|---|---|
| $c$ | $0$ |
| $x^n$ | $n x^{n-1}$ |
| $e^x$ | $e^x$ |
| $\log x$ | $1/x$ |
| $\sin x$, $\cos x$ | $\cos x$, $-\sin x$ |

Plus three combination rules:

| Rule | Formula |
|---|---|
| Sum | $(f + g)' = f' + g'$ |
| Product | $(f \cdot g)' = f' g + f g'$ |
| Chain | $(f(g(x)))' = f'(g(x)) \cdot g'(x)$ |

The chain rule is the star of the show — we'll spend an entire Ch.6 on it because that's what backpropagation *is*.

### 4.3 · Higher derivatives — the second-order story

Differentiate again: $h''(t) = -g$. A constant. It says "the vertical velocity changes at a constant rate" — gravity. This is Newton's second law for our free throw: $F = ma$ with $a = -g$.

**In ML.** The second derivative of the loss tells you about curvature. Small second derivative: gentle bowl, easy to optimise. Large: steep walls, small learning rates required. Hessian = matrix of second derivatives, central to Newton's method (and its approximations).

---

## 5 · The Integral — Area Under a Curve

### 5.1 · Riemann sum to integral

Turn the question around: we know the velocity $v(t) = v_{0y} - g\,t$. How do we recover height $h(T) - h(0)$?

Slice the time interval $[0, T]$ into $n$ equal pieces, each of width $\Delta t = T/n$. On each slice, approximate the moving ball as if it had constant velocity — say, the velocity at the left edge of the slice. The ball's rise over that slice is approximately $v(t_i)\,\Delta t$. Sum all slices:

$$S_n \;=\; \sum_{i=0}^{n-1} v(t_i)\,\Delta t$$

This is a **Riemann sum**. Shrink $\Delta t$ toward zero (equivalently, $n \to \infty$). If $v$ is well-behaved, $S_n$ converges to the **definite integral**:

$$\int_0^T v(t)\,dt \;=\; \lim_{n \to \infty}\,S_n$$

That integral *is* the total rise $h(T) - h(0)$. For our free throw with $T = v_{0y}/g$ (release to apex):

$$\int_0^{v_{0y}/g} (v_{0y} - g\,t)\,dt \;=\; \left[v_{0y}\,t - \tfrac{1}{2}g\,t^2\right]_0^{v_{0y}/g} \;=\; \frac{v_{0y}^2}{2g} \;\approx\; 2.64\text{ m}$$

### 5.2 · Fundamental Theorem of Calculus (FTC)

Two statements, one theorem:

> **(Part 1)** If $F(x) = \int_a^x f(t)\,dt$, then $F'(x) = f(x)$. Integration and differentiation undo each other.
>
> **(Part 2)** $\int_a^b f(x)\,dx = F(b) - F(a)$, where $F$ is any antiderivative of $f$. So if you can *recognise* $f$ as the derivative of some known $F$, the integral is a one-line subtraction.

**In the free-throw example.** $v(t) = v_{0y} - gt$ is the derivative of $h(t) = v_{0y}t - \tfrac{1}{2}gt^2$. FTC Part 2: $\int_0^T v(t)\,dt = h(T) - h(0)$. Derivative and integral are the two faces of the same coin.

---

## 6 · Step by Step — the derivative of the free-throw height, from scratch

1. Start from the definition: $h'(t) = \lim_{\Delta t \to 0} \frac{h(t + \Delta t) - h(t)}{\Delta t}$.
2. Substitute $h$: numerator $= v_{0y}(t + \Delta t) - \tfrac{1}{2}g(t + \Delta t)^2 - [v_{0y}\,t - \tfrac{1}{2}g t^2]$.
3. Expand $(t + \Delta t)^2 = t^2 + 2 t \Delta t + (\Delta t)^2$. Cancel the $v_{0y}\,t$ and $\tfrac{1}{2}g t^2$ terms.
4. Numerator becomes $v_{0y}\,\Delta t - g\,t\,\Delta t - \tfrac{1}{2}g (\Delta t)^2$.
5. Divide by $\Delta t$: $v_{0y} - g\,t - \tfrac{1}{2}g\,\Delta t$.
6. Take $\Delta t \to 0$: the last term vanishes. Result: $h'(t) = v_{0y} - g\,t$. ✓

The modern rule "differentiate $t^2$ → $2t$" just packages this calculation so you never have to do it by hand again.

---

## 7 · Key Diagram

![Ch.3 hero: left panel shows the free-throw height h(t) with three coloured secants at Δt = 0.6, 0.25, 0.08 collapsing onto the dashed tangent; middle panel shows the velocity v(t) filled with orange Riemann rectangles and a legend of left-endpoint sums for n = 4, 16, 64 converging to the exact integral; right panel shows the unit circle inscribed with polygons at n = 4, 8, 16, 64 approaching π](img/ch03-secant-riemann-archimedes.png)

Left: as $\Delta t$ shrinks from 0.60 to 0.08, the secant slope slides from +0.82 up toward the tangent slope +3.77 — that limit is the derivative. Middle: the orange rectangles are gross at $n=8$, but the tabulated sums show convergence 3.30 → 2.81 → 2.68 → 2.64 toward the exact integral. Right: Archimedes had the integral idea 1900 years before Newton — and his polygon method nails $\pi$ to five decimal places with 64 sides.

---

## 8 · Code Skeleton

```python
import numpy as np

v0y, g = 7.2, 9.81
h = lambda t: v0y * t - 0.5 * g * t ** 2
v = lambda t: v0y - g * t   # analytic derivative

# --- numerical derivative at t = 0.5 using the secant definition ---
t0 = 0.5
for dt in [1e-1, 1e-3, 1e-6]:
    est = (h(t0 + dt) - h(t0)) / dt
    print(f"dt = {dt:.0e}   estimate = {est:+.6f}   true = {v(t0):+.6f}")

# --- numerical integral of v(t) from 0 to t_apex via left-rectangle sum ---
t_apex = v0y / g
for n in [4, 16, 64, 1024]:
    edges = np.linspace(0, t_apex, n + 1)
    left_pts = edges[:-1]
    est = np.sum(v(left_pts) * np.diff(edges))
    print(f"n = {n:5d}   sum = {est:.6f}   exact = {h(t_apex):.6f}")
```

---

## 9 · What Can Go Wrong

- **$\Delta t$ too small.** Numerical derivatives are a balance: large $\Delta t$ has truncation error, tiny $\Delta t$ has round-off error from finite-precision subtraction. "Catastrophic cancellation" is the official name. A rule of thumb: $\Delta t \approx \sqrt{\varepsilon_\text{mach}} \approx 10^{-8}$ for 64-bit floats.
- **Non-differentiable points.** The ReLU function (ML Ch.4) is not differentiable at 0. We use subgradients in practice — a small concession the framework hides.
- **Left vs right vs midpoint.** For a decreasing function, left-endpoint rectangles overestimate, right-endpoint underestimate, midpoint is usually the most accurate. `scipy.integrate.quad` uses adaptive rules; hand-rolled sums should use midpoint or trapezoidal.
- **Integrals with no closed form.** $\int e^{-x^2} dx$ has no elementary antiderivative (Liouville 1835). Many ML quantities — KL divergence, evidence lower bound — get numerically integrated or Monte-Carlo estimated.
- **Infinities.** $\int_0^1 1/x\,dx$ diverges. Always check the integrand for singularities inside your interval before trusting a numerical answer.

---

## 10 · Exercises

1. **Derivatives by rule.** Compute $f'$ for:
   (a) $f(x) = 3x^5 - 2x + 7$,  (b) $f(x) = (2x+1)^3$ (chain rule),  (c) $f(x) = x\,e^x$ (product rule).
2. **Free-throw apex.** Given $v_{0y} = 7.2$, use $h'(t) = 0$ to find the apex time analytically, then compare to a numerical derivative zero-finder.
3. **Numerical derivative.** For $f(x) = \sin(x)$ at $x = 1$, plot the error $|f'_\text{numeric}(x) - \cos(1)|$ as $\Delta t$ sweeps from $1$ down to $10^{-14}$. You should see a V-shape: error decreases, bottoms out near $10^{-8}$, then *grows* due to round-off. Confirm.
4. **Riemann sum convergence.** For $\int_0^\pi \sin(x)\,dx = 2$, tabulate the left-endpoint sum for $n = 4, 16, 64, 256$. How does the error scale with $n$? (Answer: roughly $1/n$ for left-endpoint.)
5. **FTC in action.** Verify by computer that $\int_0^T (v_{0y} - g t)\,dt = h(T)$ for a handful of $T$ values. The subtraction $h(T) - h(0)$ should agree to machine precision with `scipy.integrate.quad`.

---

## 11 · Where This Reappears

- **Pre-Req Ch.4.** A derivative tells you which direction is *down*. Gradient descent is walking downhill, one tiny step at a time.
- **Pre-Req Ch.6.** Chain rule → automatic differentiation → PyTorch's `backward()`.
- **ML Ch.1.** The least-squares loss is a parabola; its derivative is zero at the optimum. That's how linear regression "closed form" works.
- **ML Ch.4 Neural Networks.** Every layer is a differentiable function. Training *is* repeatedly computing $\partial L / \partial w$ and stepping against it.
- **Everywhere else in the ML book.** You cannot escape calculus. This chapter is the minimum-viable dose.

---

## 12 · References

- **Jon Krohn — *Calculus 1 for Machine Learning*.** Video companion to this chapter; the "limits" and "derivative definition" episodes map one-to-one to Sections 4 and 5.
- **3Blue1Brown — *Essence of Calculus*, eps. 1–4.** The visual intuition for secant→tangent and Archimedes is unmatched.
- **Spivak — *Calculus*.** If you want the rigorous $\varepsilon$–$\delta$ version one day.
- **Stewart — *Calculus: Early Transcendentals*.** The standard undergrad reference, enormous but friendly.
- **Strogatz — *Infinite Powers* (2019).** History of calculus as pop-science, brilliant companion reading for Section 3.
