# Ch.1 — Linear Algebra: Lines, Weights, and Biases

> **The story.** Around 300 BCE, Euclid wrote *Elements* and gave geometry its first axioms — the word *linear* descends from the Latin *linea*, "a line," and Euclid's straight line is still the object we are about to fit. The algebra came later: in 820 CE the Persian mathematician al-Khwārizmī wrote *al-Kitāb al-mukhtaṣar fī ḥisāb al-jabr* — the book that gave us the word "algebra" and the first systematic recipe for solving equations like $y = wx + b$. Eight centuries on, Descartes and Fermat in the 1630s glued geometry and algebra together with coordinates: every line was now an equation, every equation a line. That coordinate-and-equation pair is the entire mathematical machinery of this chapter — and, scaled up, the entire machinery of linear regression.
>
> **Where you are in the curriculum.** This is chapter one. You bring high-school algebra; you leave with the geometric intuition for $y = wx + b$ that every later chapter (gradients, matrices, neural networks) is going to lean on. The running example is a football striker lining up a direct **knuckleball free kick** — struck with almost no spin, so the ball's path is governed by gravity alone. The goal is 20 m away, the crossbar 2.44 m off the ground. In the first **0.1 seconds** after the boot leaves the ball, the ball rises *almost* in a straight line — fast enough that gravity's curve is still invisible. That is the regime where a single equation $y = wx + b$ tells us everything, and where every idea in machine learning is simplest to see.
>
> **Notation in this chapter.** $w$ — slope (the *weight* in ML); $b$ — intercept (the *bias*); $\hat{y}$ — the model's predicted output; $\mathbf{x}=[x_1,\dots,x_d]$ — input feature vector; $\mathbf{w}$ — weight vector; $\mathbf{a}\cdot\mathbf{b}=\sum_i a_i b_i$ — dot product; $\|\mathbf{a}\|$ — Euclidean norm; $\theta$ — angle between two vectors; $h(t)$ — ball height at time $t$; $v_{0y}$ — initial vertical velocity; $g\approx 9.81\,\text{m/s}^2$ — gravity; $h_0$ — release height of the ball.

---

## 1 · Core Idea

A **line** is a two-parameter object. Pick any two numbers $w$ and $b$ and you have a line:

$$y = w x + b$$

The first number, $w$, tilts the line. The second, $b$, shifts it up or down. Every linear model in machine learning — from single-variable regression to the 175-billion-parameter matrix in GPT's first linear layer — is a direct generalisation of this one equation.

---

## 2 · Running Example

A striker wants to score a direct knuckleball free kick from 20 m, clearing a defensive wall at 9.15 m (10 yd) and dipping the ball under the 2.44 m crossbar. The full trajectory is a parabola (that's Ch.2). But in the *first 0.1 seconds* after the boot strikes the ball, gravity has not yet bent the path noticeably, and the ball's height $h$ as a function of time $t$ is well-approximated by:

$$h(t) \approx v_{0y} t + h_0$$

where $v_{0y}$ is the vertical component of the release velocity and $h_0$ is the release height (for a ball on the turf, $h_0 = 0$). Written in ML notation: $w = v_{0y}$ and $b = h_0$. **That is why machine learning calls them weights and biases** — the weight scales the input, the bias is the starting offset. Same equation, different name tags.

---

## 3 · Math

### 3.1 · The equation of a line — three equivalent framings

| Framing | Equation | Where you see it |
|---|---|---|
| Slope–intercept | $y = m x + c$ | High-school algebra |
| Physics | $h(t) = v t + h_0$ | Projectile motion |
| Machine learning | $\hat{y} = w x + b$ | Every neural network |

Same object, same two parameters, different traditions. Once you see that, most ML papers become one translation away from a physics textbook.

### 3.2 · Vectors — a list of numbers with a job

A **vector** is an ordered list of numbers. For us there are two jobs:

1. **A point or direction in space.** The vector $[3, 1]$ points 3 right and 1 up.
2. **A bag of features.** The vector $[v_0, \theta, h_0, m, \ldots]$ describes one free-kick attempt — release speed, launch angle, foot height, ball mass, and so on.

Both jobs are just lists of numbers — what you do with them depends on the problem.

### 3.3 · The dot product — one operation, everywhere

Given two vectors $\mathbf{a} = [a_1, a_2, \ldots, a_d]$ and $\mathbf{b} = [b_1, b_2, \ldots, b_d]$:

$$\mathbf{a}\cdot\mathbf{b} = \sum_{i=1}^{d} a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$$

Two readings of the *same* number:

- **Algebraic.** A weighted sum: multiply matching entries, add them up.
- **Geometric.** Scalar measuring how much $\mathbf{a}$ and $\mathbf{b}$ agree. Positive = same direction. Zero = perpendicular. Negative = opposite.

The dot product is the single most-used operation in ML. A neuron is a dot product plus a bias plus a non-linearity. Attention (ML Ch.17–18) is dot products. Cosine similarity in embedding search is a dot product. `np.dot(a, b)` in Python.

### 3.4 · From one feature to many — the general linear equation

When you have $d$ features $\mathbf{x} = [x_1, x_2, \ldots, x_d]$ and a weight per feature $\mathbf{w} = [w_1, w_2, \ldots, w_d]$:

$$\hat{y} = \mathbf{w}\cdot\mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b$$

This is just Section 3.1's equation with more terms. Every "linear" model — linear regression, logistic regression, the first layer of any MLP — is this equation. Chapter 5 will rewrite it in matrix form so we can apply it to thousands of samples at once; right now, it's a scaled and shifted sum.

---

## 4 · Step by Step — fitting a line by eye

1. **Plot the data.** Scatter the $(x_i, y_i)$ points.
2. **Pick a $w$.** Guess the slope that follows the cloud's tilt.
3. **Pick a $b$.** Guess the $y$-intercept: where would the line cross $x=0$?
4. **Draw it. Look.** Which points sit above the line, which below?
5. **Adjust.** If most points are above the line, raise $b$ (or increase $w$). If the line is too steep, lower $w$. Iterate.
6. **Stop when the line "bisects" the cloud.** You have an approximate least-squares fit — without having written a single formula.

Steps 2–5 are literally what gradient descent (Pre-Req Ch.4 and ML Ch.5) automates. Doing it by eye first makes the automated version feel inevitable.

---

## 5 · Key Diagram

![Ch.1 hero: top-left varying weight rotates the line around origin, top-right varying bias shifts the line vertically, bottom-left two vectors with their dot product highlighted, bottom-right linear fit to first 0.1 s of a knuckleball free kick](img/ch01-lines-and-biases.png)

Top row: the weight $w$ and the bias $b$ do geometrically *different* jobs — one tilts, one shifts — and every line in two dimensions can be expressed by picking them. Bottom-left: two vectors and their dot product, the operation underneath every weighted sum in ML. Bottom-right: the free-kick running example with two candidate fits — the *criterion* for picking the green one over the red one is the subject of Ch.3 and Ch.4.

---

## 6 · Code Skeleton

Minimal Python for every piece of Section 3. The notebook expands each line into an interactive widget.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 3.1 · line equation ---
w, b = 6.5, 0.0 # vertical release velocity (m/s) and release height (m)
t = np.linspace(0, 0.10, 50)
h = w * t + b # height in the straight-line regime

# --- 3.3 · dot product, three equivalent forms ---
a, v = np.array([3.0, 1.0]), np.array([2.0, 2.5])
print("explicit sum: ", a[0] * v[0] + a[1] * v[1])
print("numpy dot: ", np.dot(a, v))
print("matrix mul: ", a @ v) # PEP 465 operator

# --- 3.4 · multi-feature prediction (free-kick goal probability, toy weights) ---
x = np.array([25.0, 15.0, 0.10]) # release speed (m/s), launch angle (deg), foot height (m)
W = np.array([0.03, 0.04, -1.50]) # one weight per feature
bias = -0.40
y_hat = W @ x + bias # weighted sum of features + baseline
print("score: ", y_hat)
```

---

## 7 · What Can Go Wrong

- **Conflating units.** If release speed is in m/s and launch angle is in degrees, the fitted $w$ carries a unit (e.g. "score per m/s" vs "score per degree"). Plot with labels or you will misread the scale.
- **No bias term.** Dropping $b$ forces the line through the origin — usually wrong. A free kick struck at zero speed does not have a zero outcome score; $b$ carries the baseline.
- **Treating "linear" as "works in a line."** A linear model is *linear in the parameters*, not necessarily in the inputs. That subtlety is the entire subject of Ch.2.
- **Not normalising features.** If feature 1 ranges 0–1 and feature 2 ranges 0–10 000, their weights are on wildly different scales and the model is hard to interpret. Standardise ($x_i \leftarrow (x_i - \mu_i)/\sigma_i$) before eyeballing.

---

## 8 · Exercises

1. Write $2x - 3y = 6$ in the form $y = wx + b$. What are $w$ and $b$?
2. You strike the ball off the turf ($h_0 = 0$) with a vertical velocity of 6.5 m/s. What is the height at $t = 0.08$ s, in the straight-line regime? Sanity-check against $h(0) = 0$.
3. Compute $\mathbf{a}\cdot\mathbf{b}$ for $\mathbf{a} = [1, 2, 3]$ and $\mathbf{b} = [4, -1, 2]$. Then find a non-zero $\mathbf{c}$ such that $\mathbf{a}\cdot\mathbf{c} = 0$. (There are infinitely many — any one works.)
4. Using the widget in the notebook, fit a line to the recorded (time, height) samples. Record the $(w, b)$ you landed on. How close is $w$ to the true $v_{0y}$?
5. If every input is doubled, $\mathbf{x}' = 2\mathbf{x}$, what happens to $\hat{y} = \mathbf{w}\cdot\mathbf{x} + b$? What about if *one* coordinate $x_1$ is doubled and the rest stay the same?

---

## 9 · Where This Reappears

- **Pre-Req Ch.2** — you keep the same equation but engineer new features $x_1 = x, x_2 = x^2, x_3 = x^3, \ldots$ to fit curves with a linear formula.
- **Pre-Req Ch.5** — matrix form $\mathbf{\hat{y}} = \mathbf{X}\mathbf{w} + b$ handles thousands of samples at once.
- **ML Ch.1 Linear Regression** — adds a *loss function* and an *optimiser* so a computer can pick $w, b$ for you.
- **ML Ch.4 Neural Networks** — a neuron is one line of this chapter plus a non-linearity.
- **ML Ch.17 Attention** — every attention weight is a dot product.

---

## 10 · References

- **[Jon Krohn — Linear Algebra for Machine Learning (YouTube course).](https://www.jonkrohn.com/posts/2021/5/9/linear-algebra-for-machine-learning-complete-math-course-on-youtube)** Segments 1–3 cover everything in this chapter at video pace. Pair with the chapter if a concept still feels slippery.
- **3Blue1Brown — *Essence of Linear Algebra*, episodes 1–2.** Vectors and linear combinations with the best visual grammar available.
- **Strang — *Introduction to Linear Algebra*, §1.1–§1.2.** The formal version of Sections 3.2–3.4 above.
