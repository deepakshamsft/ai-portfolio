# Ch.2 — Non-Linear Algebra: Polynomials and the Feature-Expansion Trick

> **The story.** Once Descartes and Fermat had pinned curves to equations in the 1630s, mathematicians spent the next two centuries discovering an awkward truth: most real-world phenomena — falling apples, planetary orbits, vibrating strings — are not lines. They are curves. The breakthrough was that you didn't need new mathematics for them; you just needed a *trick*. Treat the curve's shape as a sum of simpler pieces ($1, x, x^2, x^3, \dots$) and the linear machinery of Ch.1 still works — you just have more knobs to turn. That trick is what kept linear regression alive into the 20th century, and it is the same trick that kernel methods (Ch.11) and neural-network feature learning quietly rely on.
>
> **Where you are in the curriculum.** Ch.1 fit a straight line to the *very first* part of the free-kick trajectory. Now we admit gravity. The full trajectory from boot to goal is a parabola, not a line, and a straight line is hopeless as a fit — yet the same linear-regression machinery can fit this parabola exactly. Understanding *why* unlocks every later "non-linear" model in this curriculum, all the way up to deep networks.
>
> **Notation in this chapter.** $x$ — scalar input; $a_n,\dots,a_0$ — polynomial coefficients (the parameters we fit); $n$ — polynomial degree; $\phi(x)=[1,x,x^2,\dots,x^n]$ — basis expansion / *polynomial features* (the trick that makes a curve linear in the parameters); $\mathbf{w}$ — weight vector on the expanded features; $\hat{y}$ — prediction; $t$ — time since the boot leaves the ball; $y(t)=v_{0y}t-\tfrac{1}{2}gt^2$ — height of the free-kick parabola; $g\approx 9.81\,\text{m/s}^2$ — gravity.

---

## 1 · Core Idea

$a x^2 + b x + c$ is **non-linear in the input $x$** — plot it and you get a curve. But it is **linear in the coefficients $a, b, c$** — adjust $a$, you get a scaled copy; adjust $c$, you shift. If we invent two new features

$$x_1 = x^2 \qquad x_2 = x$$

then the equation becomes

$$y = a x_1 + b x_2 + c$$

— which is exactly Ch.1's multi-feature linear model. One curve in one dimension has become one *flat plane* in two dimensions. That is **basis expansion** (or, in scikit-learn terminology, **polynomial features**), and it is how "linear" models fit curves.

---

## 2 · Running Example

A direct knuckleball free kick from 20 m, with the goal's crossbar at 2.44 m and a defensive wall at 9.15 m. The ball is struck at 25 m/s and released at 15° above horizontal. The full trajectory is

$$y(t) = v_{0y} t - \tfrac{1}{2} g t^2$$

Ch.1 gave us the straight-line approximation valid only for the first 0.1 s. Now we want the whole arc — rise, apex over the wall, and the late dip that drops the ball under the crossbar.

---

## 3 · Math

### 3.1 · Polynomials of degree $n$

A polynomial in one variable:

$$p(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_1 x + a_0$$

| Degree | Shape | Example in the running story |
|---|---|---|
| 0 | horizontal line | constant release height only |
| 1 | straight line | the 0.1 s approximation (Ch.1) |
| 2 | parabola | free-kick trajectory |
| 3 | cubic, one inflection | trajectory with a slight sideways swerve |
| $n$ | up to $n-1$ bends | anything smooth on a finite range (Weierstrass' theorem) |

**Why this matters.** Polynomials can approximate essentially any continuous curve on a bounded interval arbitrarily closely — a classical result due to Weierstrass (1885). That's why they're the first tool out of the box for non-linear fitting.

### 3.2 · Linear *in the parameters* — the key distinction

Compare two statements:

$$(\star)\quad y = 3 x^2 - 2 x + 1 \qquad (\text{non-linear in } x)$$
$$(\dagger)\quad y = 3 x_1 - 2 x_2 + 1 \quad \text{where } x_1 = x^2, x_2 = x \qquad (\text{linear in } x_1, x_2)$$

Both equations produce identical $y$ for every $x$. They are the same curve, written differently. And equation $(\dagger)$ is in Ch.1's form $\hat{y} = \mathbf{w}\cdot\mathbf{x} + b$, with $\mathbf{w} = [3, -2]$ and $b = 1$.

**The trick generalises.** For any polynomial of degree $n$:

$$\hat{y} = a_n x_n + a_{n-1} x_{n-1} + \cdots + a_1 x_1 + a_0 \quad \text{where } x_k = x^k$$

You just *engineered* new features. Linear regression fits $a_0, a_1, \ldots, a_n$ in closed form (Ch.5 will show how). No curve-fitting routine needed.

### 3.3 · Beyond polynomials — other basis expansions

Polynomials are one family; they are not the only one. The same trick works for:

| Basis | Features | Good for |
|---|---|---|
| Polynomial | $x, x^2, x^3, \ldots$ | Smooth curves on a bounded range |
| Fourier | $\sin(kx), \cos(kx)$ | Periodic signals (vibration, audio) |
| Radial | $\exp(-\|x-c_i\|^2)$ | Localised bumps (radial basis networks) |
| Spline / piecewise poly | a polynomial per segment | Flexible fits with controllable smoothness |
| Interaction | $x_i \cdot x_j$ | Capturing feature pairs (e.g. income × location) |

Every row is the same pattern: invent features $\phi_k(x)$ that are non-linear in $x$, then fit $\hat{y} = \sum_k w_k \phi_k(x) + b$ linearly in the weights.

### 3.4 · The limit of the trick — what it cannot do

Feature expansion turns *input* non-linearity into linear fitting. It does **not** help with *parameter* non-linearity.

**Genuinely non-linear in parameters:**

$$y = a \cdot e^{b x}$$

Here $b$ sits inside an exponent. No feature substitution turns this into a linear-in-$(a,b)$ model. You need iterative optimisation (Pre-Req Ch.4) — and when the parameter surface is large and non-convex, you need neural networks (ML Ch.4 onwards). That is the reason the ML book exists.

### 3.5 · Multi-input polynomials

With two inputs $x$ and $z$, a degree-2 polynomial includes all terms up to total degree 2:

$$y = a_0 + a_1 x + a_2 z + a_3 x^2 + a_4 x z + a_5 z^2$$

The $x z$ term is an **interaction** — it captures "the effect of $x$ *depends on* $z$". Every new interaction is a new column in the feature matrix; the linear machinery is unchanged.

**Scaling pain.** With $d$ inputs and degree $n$, the number of polynomial features is $\binom{d+n}{n}$. For $d=8$, $n=4$ that's 495 features — tolerable. For $d=100$, $n=4$ it's over 4.5 million. This **combinatorial explosion** is why polynomial features stop being the tool of choice for high-dimensional problems; neural networks learn a *compressed* non-linearity instead.

---

## 4 · Step by Step — fit a parabola with a linear regression

1. Record $(t_i, y_i)$ samples along the trajectory.
2. **Engineer features.** For each sample, produce $\mathbf{x}_i = [t_i, t_i^2]$.
3. **Fit linearly.** Solve $\hat{y} = w_1 t + w_2 t^2 + b$ using any least-squares routine (`np.polyfit`, `sklearn.LinearRegression`, or the normal equations in Ch.5).
4. **Read off physics.** If the model is $y(t) = v_{0y} t - \tfrac{1}{2}g t^2$, then the fitted $w_1$ recovers $v_{0y}$ and $w_2$ recovers $-\tfrac{1}{2}g \approx -4.905$.
5. **Predict.** Plug any new $t$ into the polynomial to get the ball's height.

The whole recipe is one feature-engineering line away from Ch.1.

---

## 5 · Key Diagram

![Ch.2 hero: left panel shows a parabolic knuckleball free-kick trajectory with a horizontal best line that obviously fails to fit; middle panel shows a parabola y = a x^2 + b x + c with vertex and y-intercept annotated; right panel is a 3-D view of the same parabola as a flat plane in feature space x1 = x^2, x2 = x](img/ch02-polynomials-and-plane.png)

Left: a straight line is the wrong tool for a parabolic path. Middle: $a$ controls curvature, $b$ shifts the vertex sideways, $c$ is the $y$-intercept. Right: the exact same parabola $y = 3x^2 - 2x + 1$ shown as the *intersection* of a flat plane with the curved constraint surface $x_1 = x_2^2$ in 3-D feature space. The plane *is* linear; the curve looks bent only because we're looking at a 1-D slice of it.

---

## 6 · Code Skeleton

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# --- simulate a knuckleball free kick ---
v0, theta, g = 25.0, np.radians(15), 9.81
v0y = v0 * np.sin(theta)
t = np.linspace(0, 1.0, 30)
y = v0y * t - 0.5 * g * t ** 2 # true curve
y_noisy = y + np.random.normal(0, 0.03, t.shape) # measurement noise

# --- engineer features: [t, t^2] ---
poly = PolynomialFeatures(degree=2, include_bias=False)
T = poly.fit_transform(t.reshape(-1, 1)) # shape (N, 2)

# --- fit linearly on those features ---
model = LinearRegression().fit(T, y_noisy)
w1, w2 = model.coef_
b = model.intercept_
print(f"fitted w1 (~v0y) : {w1:+.3f} (true: {v0y:+.3f})")
print(f"fitted w2 (~-g/2 = -4.905): {w2:+.3f}")
print(f"fitted b (~h0 = 0) : {b:+.3f}")
```

---

## 7 · What Can Go Wrong

- **Over-fitting with high degree.** Polynomial fits of degree 10 on 15 samples will pass through every point *and* oscillate wildly between them — Runge's phenomenon. Keep the degree low, or use a spline / regularisation.
- **Numerical ill-conditioning.** Columns $x, x^2, x^3, \ldots$ become highly correlated. For degrees beyond 5, centre and scale $x$ first, or switch to orthogonal polynomials (e.g. Chebyshev).
- **Extrapolation disasters.** A polynomial fit is only trustworthy inside the range of the training data. A degree-3 fit on $x \in [0, 1]$ tells you nothing about $x = 5$.
- **Forgetting the bias.** Dropping $c$ pins the parabola through the origin — usually wrong, exactly as in Ch.1.
- **Interactions forgotten.** In 2-D polynomial features, `sklearn.PolynomialFeatures(degree=2, interaction_only=True)` gives only $x_1 x_2$ and drops $x_1^2, x_2^2$. Use the default unless you know you want only interactions.

---

## 8 · Exercises

1. Write $y = 2(x+1)^2 - 3$ in the standard form $y = a x^2 + b x + c$. What are $a, b, c$?
2. Apply the feature-expansion trick to $y = 5 x^3 - 2 x$. What are $x_1, x_2, x_3$ and what are the weights $w_1, w_2, w_3$?
3. Can you turn $y = a \sin(x) + b \cos(x)$ into a linear-in-parameters model via feature expansion? What are the features? (Answer: yes, $\phi_1 = \sin(x), \phi_2 = \cos(x)$ — it's called a *Fourier basis*.)
4. What about $y = \sin(a x)$? Can you do it? Why or why not? (Answer: no — $a$ is inside the $\sin$; the model is genuinely non-linear in $a$.)
5. For the free-kick simulation in the notebook, fit a degree-5 polynomial to the 30-sample trajectory and plot the residuals. Where does the fit behave badly?

---

## 9 · Where This Reappears

- **Pre-Req Ch.4** — when the loss surface over $(a, b, c)$ isn't solvable in closed form, we walk downhill on it.
- **Pre-Req Ch.5** — the least-squares fit of Section 6 is literally $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ with $\mathbf{X}$ being the polynomial-feature matrix.
- **ML Ch.2 Logistic Regression** — logistic regression + polynomial features = a linear classifier that draws curved decision boundaries.
- **ML Ch.6 Regularisation** — high-degree polynomial fits demand regularisation (Ridge, Lasso) to stay sane.
- **ML Ch.4 Neural Networks** — the *alternative* to hand-engineered features: let the model learn its own basis expansion.

---

## 10 · References

- **Jon Krohn — Linear Algebra for Machine Learning.** Segment on "feature engineering" links directly to this chapter.
- **Hastie, Tibshirani, Friedman — *Elements of Statistical Learning*, Ch.5 "Basis Expansions and Regularization".** The canonical, deeper treatment.
- **3Blue1Brown — *Essence of Linear Algebra*, ep. 3 "Linear transformations".** The 3-D plane-in-feature-space view in Section 3 is the same mental model.
