# Ch.5 — Matrices, Linear Systems, and Matrix Calculus

> **Running theme.** Real set-piece data doesn't have one variable — it has strike speed, launch angle, strike zone on the boot, wall distance, wall height, wind speed, wind direction, pitch wetness, kicker fatigue. Eight-plus variables, 500 recorded free kicks. Scalars collapse. We need **matrices**: compact containers for "apply the same linear rule to every sample in one shot."

---

## 1 · Core Idea

A matrix is a **linear map**. It takes a vector in and produces another vector out, obeying two rules:

$$A(\mathbf{x} + \mathbf{y}) = A\mathbf{x} + A\mathbf{y} \qquad A(c\mathbf{x}) = c(A\mathbf{x})$$

That's it. Every rotation, stretch, projection, regression fit, and linear layer of a neural network is a matrix. The whole chapter is learning to *think* in three equivalent ways about $A\mathbf{x} = \mathbf{b}$ so that when you meet it in the ML book you recognise it instantly.

---

## 2 · Running Example

Same knuckleball free kick, full parabolic trajectory. We want to recover the physics constants $(h_0, v_{0y}, -g/2)$ from a set of noisy $(t_i, y_i)$ measurements. In Ch.2 we did this with `sklearn.LinearRegression`; here we do it from scratch with one matrix solve. Then we extend to multi-variable regression — eight free-kick features predicting scored vs missed.

---

## 3 · Math

### 3.1 · Vectors and matrices — shapes

A vector is a column of numbers:

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_d \end{bmatrix} \in \mathbb{R}^d$$

A matrix is a rectangular block:

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

**Shapes first, values second.** While you're learning, stop after every operation and confirm the shape. $A\,B$ is legal only if $A$'s column count equals $B$'s row count; the result has $A$'s rows and $B$'s columns. Most student bugs are shape bugs in disguise.

### 3.2 · Matrix-vector product — three views of $A\mathbf{x}$

**Row view (dot-products into a column).** Each entry of $A\mathbf{x}$ is the dot product of a row of $A$ with $\mathbf{x}$:

$$(A\mathbf{x})_i \;=\; \sum_j a_{ij}\,x_j \;=\; \text{row}_i(A) \cdot \mathbf{x}$$

**Column view (weighted sum of columns).** $A\mathbf{x}$ is a linear combination of $A$'s columns, weighted by $\mathbf{x}$:

$$A\mathbf{x} \;=\; x_1\,\mathbf{a}_1 + x_2\,\mathbf{a}_2 + \cdots + x_n\,\mathbf{a}_n$$

This is the view that explains why the set of achievable $A\mathbf{x}$ (the **column space**) is the span of $A$'s columns.

**Transformation view (warp the space).** $A$ *does something* to every point of $\mathbb{R}^n$: stretches, rotates, shears, projects. See the left panel of the hero image — the unit square becomes a parallelogram whose sides are exactly the columns of $A$.

All three views describe the same computation. Different problems become obvious under different views. Good practitioners fluently switch.

### 3.3 · Matrix-matrix product

$(A B)_{ij} = \sum_k a_{ik}\,b_{kj}$ — dot product of row $i$ of $A$ with column $j$ of $B$. Shape check: $(m \times n) \cdot (n \times p) = (m \times p)$.

Compositionally: $A B$ is the matrix that performs "first apply $B$, then apply $A$". Matrix multiplication is associative $(AB)C = A(BC)$, distributive $A(B+C) = AB+AC$, but **not commutative** in general: $AB \neq BA$. A rotation followed by a stretch is different from a stretch followed by a rotation.

### 3.4 · Transpose

$A^\top$ swaps rows and columns: $(A^\top)_{ij} = a_{ji}$. Two identities you will use constantly:

- $(AB)^\top = B^\top A^\top$ (notice the reversal).
- $(A^\top)^\top = A$.

A matrix is **symmetric** if $A = A^\top$. Covariance matrices, $X^\top X$ gram matrices, and the Hessian of a scalar function are all symmetric.

### 3.5 · Identity, inverse, determinant

- **Identity.** $I_n$ has 1s on the diagonal and 0s elsewhere; $A I = I A = A$.
- **Inverse.** $A^{-1}$ exists iff $A$ is square and has full rank. Then $A A^{-1} = I$. Practical note: **never** compute $A^{-1}$ explicitly in code — call `np.linalg.solve(A, b)` or the equivalent. Explicit inverses are numerically worse than factorisations.
- **Determinant.** $\det(A)$ is the signed volume scale factor of the linear map. $\det(A) = 0$ iff $A$ squashes space onto a lower-dimensional subspace (columns are linearly dependent). In the hero image, $\det(A) = 1.65$ — areas are scaled by 1.65.

### 3.6 · The three views of $A\mathbf{x} = \mathbf{b}$

Solving a linear system is the same question asked three ways:

| View | "Solve $A\mathbf{x} = \mathbf{b}$" means… |
|---|---|
| Row | Find $\mathbf{x}$ that lies on the intersection of $m$ hyperplanes $\text{row}_i(A) \cdot \mathbf{x} = b_i$. |
| Column | Find weights $\mathbf{x}$ so that $\mathbf{b}$ is a linear combination of $A$'s columns. |
| Transformation | Find the pre-image of $\mathbf{b}$ under the map $A$ — "what input, when transformed, gives $\mathbf{b}$?" |

Depending on $A$'s shape there are three cases:

- **Square, full-rank $A$ (unique $\mathbf{x}$).** Invertible. Solve with $\mathbf{x} = A^{-1}\mathbf{b}$ conceptually, `np.linalg.solve(A, b)` in code.
- **Tall $A$ ($m > n$, over-determined).** More equations than unknowns — usually no exact solution. **Least-squares** minimises $\|A\mathbf{x} - \mathbf{b}\|_2^2$; this is the setting of every regression in the ML book.
- **Wide $A$ ($m < n$, under-determined).** More unknowns than equations — infinitely many solutions. Pick one by an extra rule (minimum norm, sparsity, prior).

### 3.7 · Least squares and the normal equations

This is the algebraic heart of linear regression. Stack $N$ samples into the **design matrix** $X \in \mathbb{R}^{N \times d}$, target vector $\mathbf{y} \in \mathbb{R}^N$. We want weights $\mathbf{w} \in \mathbb{R}^d$ minimising the sum of squared errors:

$$\mathcal{L}(\mathbf{w}) \;=\; \|X\mathbf{w} - \mathbf{y}\|_2^2 \;=\; (X\mathbf{w} - \mathbf{y})^\top (X\mathbf{w} - \mathbf{y})$$

Expand:

$$\mathcal{L}(\mathbf{w}) \;=\; \mathbf{w}^\top X^\top X \mathbf{w} \,-\, 2\,\mathbf{w}^\top X^\top \mathbf{y} \,+\, \mathbf{y}^\top \mathbf{y}$$

Take the gradient (Section 4 below has the rules), set it to zero:

$$\nabla_\mathbf{w} \mathcal{L} \;=\; 2\,X^\top X \mathbf{w} \,-\, 2\,X^\top \mathbf{y} \;=\; 0 \;\;\Longrightarrow\;\; \boxed{\hat{\mathbf{w}} \;=\; (X^\top X)^{-1} X^\top \mathbf{y}}$$

These are the **normal equations**. The closed form for linear regression. No iteration, no step size — just one matrix solve. The hero image's right panel shows it fitting the free-kick trajectory in one line.

**In code.** Never use `np.linalg.inv(X.T @ X) @ X.T @ y`. Use `np.linalg.lstsq(X, y, rcond=None)` or `np.linalg.solve(X.T @ X, X.T @ y)`. These are faster, numerically stabler, and handle rank-deficient cases gracefully.

### 3.8 · Matrix calculus — the five rules you'll actually use

Scalar-valued functions of a vector $\mathbf{w} \in \mathbb{R}^d$. The **gradient** $\nabla_\mathbf{w} f$ is a vector of partial derivatives, same shape as $\mathbf{w}$.

| Function $f(\mathbf{w})$ | Gradient $\nabla_\mathbf{w} f$ | Use case |
|---|---|---|
| $\mathbf{a}^\top \mathbf{w}$ | $\mathbf{a}$ | linear part of any model |
| $\mathbf{w}^\top \mathbf{w}$ | $2\mathbf{w}$ | L2 regularisation |
| $\mathbf{w}^\top A \mathbf{w}$ ($A$ symmetric) | $2 A \mathbf{w}$ | quadratic forms, Hessians |
| $\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ | $2 X^\top(X\mathbf{w} - \mathbf{y})$ | least-squares gradient |
| $\log\det(W)$ ($W$ invertible) | $(W^{-1})^\top$ | determinant likelihoods |

Memorise the first four; the fifth comes up in probability (Ch.7). Every derivation in the ML book reduces to these rules plus the chain rule (Ch.6).

---

## 4 · Step by Step — fit the free-kick parabola with one matrix solve

1. Collect $(t_i, y_i)$ for $i = 1, \ldots, N$.
2. Build the design matrix $X \in \mathbb{R}^{N \times 3}$ with columns $[\mathbf{1},\, \mathbf{t},\, \mathbf{t}^2]$.
3. Solve $\hat{\mathbf{w}} = \arg\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|^2$ using `np.linalg.lstsq`.
4. Read off physics: $\hat{w}_0 \approx h_0$, $\hat{w}_1 \approx v_{0y}$, $\hat{w}_2 \approx -g/2 \approx -4.905$.
5. Predict at any new $t$: $\hat{y} = \hat{w}_0 + \hat{w}_1 t + \hat{w}_2 t^2$.

Same procedure generalises to $d$ features. Build $X$, solve, read off weights.

---

## 5 · Key Diagram

![Ch.5 hero: three panels illustrating matrices. Left panel shows a 2×2 matrix A = [[1.5, 0.5], [0.3, 1.2]] transforming the unit square into a parallelogram whose sides are the columns of A; det(A) = 1.65 is the area scale. Middle panel is the column view of Ax showing the vector sum x1·col1 + x2·col2 + x3·col3 built tip-to-tail. Right panel shows the normal equations fitting a parabola to noisy free-kick samples, recovering physics constants h0≈0.02 m, v0y≈6.44 m/s, −g/2≈−4.89.](img/ch05-matrix-views.png)

Left: the columns of $A$ tell you *where the basis vectors land*. Middle: every matrix-vector product is a weighted sum of the matrix's columns — a picture worth a thousand index-chasing proofs. Right: the entire chapter pays for itself on the trajectory-fitting problem — `(XᵀX)⁻¹Xᵀy` recovers the laws of motion from noisy data.

---

## 6 · Code Skeleton

```python
import numpy as np

# --- three views of Ax ---
A = np.array([[1.5, 0.5],
              [0.3, 1.2]])
x = np.array([0.7, 0.4])
b1 = A @ x
b2 = np.array([A[0] @ x, A[1] @ x])      # row view
b3 = x[0] * A[:, 0] + x[1] * A[:, 1]     # column view
assert np.allclose(b1, b2) and np.allclose(b1, b3)

# --- free-kick fit via normal equations ---
rng = np.random.default_rng(0)
v0y, h0, g = 6.5, 0.0, 9.81
t = np.linspace(0.02, 1.3, 20)
y = h0 + v0y * t - 0.5 * g * t ** 2 + rng.normal(0, 0.05, t.shape)

X = np.column_stack([np.ones_like(t), t, t ** 2])
w_hat, *_ = np.linalg.lstsq(X, y, rcond=None)      # preferred
# equivalent, slower, less stable:
# w_hat = np.linalg.solve(X.T @ X, X.T @ y)

print(f"ŵ₀ (h0)   = {w_hat[0]:+.3f}   (true {h0})")
print(f"ŵ₁ (v0y)  = {w_hat[1]:+.3f}   (true {v0y})")
print(f"ŵ₂ (-g/2) = {w_hat[2]:+.3f}   (true {-g/2:.3f})")
```

---

## 7 · What Can Go Wrong

- **Shape mismatches.** The commonest bug. Print `.shape` after every operation while you're learning.
- **`np.dot` vs `np.matmul` vs `@`.** For 2-D × 2-D, all three agree. For 1-D, they differ (dot product vs broadcasted matrix product). Use `@` — it's unambiguous and matches the maths.
- **Explicit inverse.** `np.linalg.inv(A) @ b` is slower and less accurate than `np.linalg.solve(A, b)`. Avoid `inv()` except in derivations.
- **Rank deficiency.** If columns of $X$ are linearly dependent (e.g., one feature is a scalar multiple of another), $X^\top X$ is singular and the normal equations have infinitely many solutions. `lstsq` returns the minimum-norm one. Check with `np.linalg.matrix_rank(X)`.
- **Conditioning.** Even if $X^\top X$ is technically invertible, it can be **ill-conditioned** — tiny input changes cause huge output changes. Polynomial features of high degree are notorious. Rescale your inputs (zero mean, unit variance) before fitting.
- **Row major vs column major.** NumPy is row-major; many stats texts are column-major. When in doubt, assume $\mathbf{x}$ is a column vector and $X \in \mathbb{R}^{N \times d}$ has one sample per row — the convention in ML code.

---

## 8 · Exercises

1. **Shape drill.** $A$ is $3 \times 5$, $B$ is $5 \times 2$, $\mathbf{x}$ is a 5-vector. Which of these are legal, and what shape do they produce? $A\mathbf{x}$, $B\mathbf{x}$, $AB$, $BA$, $A^\top A$, $B B^\top$, $A^\top \mathbf{x}$.
2. **Columns are images of basis vectors.** Verify numerically that for any matrix $A$, `A @ np.eye(n)[:, k]` returns the $k$-th column of $A$. What does that tell you about the transformation view?
3. **Recover physics.** Generate 30 noisy free-kick samples as in Section 6. Vary the noise standard deviation from 0.01 to 0.5; plot the estimated $\hat{v}_{0y}$ against noise level. How does estimation error grow?
4. **Over-determined vs under-determined.** Build a $20 \times 3$ design matrix $X$ and compare `np.linalg.lstsq(X, y)` to `np.linalg.solve(X.T @ X, X.T @ y)`. Now try $X$ of shape $3 \times 20$ with a given $y \in \mathbb{R}^3$. What does `lstsq` return? What is $\|X \hat{\mathbf{w}} - y\|$ in each case?
5. **Matrix-calculus check.** For $f(\mathbf{w}) = \mathbf{w}^\top A \mathbf{w}$ with $A$ a random $5 \times 5$ matrix, verify numerically that $\nabla f(\mathbf{w}) = (A + A^\top)\mathbf{w}$ (not $2A\mathbf{w}$ unless $A$ is symmetric) using finite differences.

---

## 9 · Where This Reappears

- **Pre-Req Ch.6.** Chain rule in matrix form — the gradient-of-a-composition machinery that `torch.autograd` automates.
- **Pre-Req Ch.7.** Covariance matrices ($\Sigma = \mathbf{X}^\top \mathbf{X} / n$ after centring) and multivariate Gaussians.
- **ML Ch.1 Linear Regression.** The normal equations *are* linear regression's closed-form solution.
- **ML Ch.2 Logistic Regression.** Same $X \mathbf{w}$ machinery, squashed through a sigmoid.
- **ML Ch.4 Neural Networks.** Every layer is $\mathbf{h} = \sigma(W \mathbf{x} + \mathbf{b})$ — a matrix-vector product plus bias plus non-linearity.
- **ML Ch.13 Dimensionality Reduction.** SVD, PCA, and low-rank approximation are pure linear algebra.
- **ML Ch.18 Transformers.** The attention score matrix is exactly $Q K^\top$ (softmax-normalised). Same $A\mathbf{x}$ all the way up the stack.

---

## 10 · References

- **Gilbert Strang — *Introduction to Linear Algebra* (MIT OCW 18.06).** The canonical undergraduate text and free video lectures; the "three views of $A\mathbf{x} = \mathbf{b}$" framing of Section 3.6 is lifted directly from his lectures.
- **Jon Krohn — *Linear Algebra II: Tensors, Matrices & Dimensions*.** Video companion, builds on the Ch.1 segment.
- **3Blue1Brown — *Essence of Linear Algebra*, eps. 3–7.** The transformation view is much more intuitive after these videos than after any textbook.
- **Petersen & Pedersen — *The Matrix Cookbook*** (free PDF). Every matrix-calculus identity you'll ever need, no proofs, densely indexed. Bookmark it.
- **Trefethen & Bau — *Numerical Linear Algebra*.** If you ever need to know *why* `np.linalg.lstsq` is better than `inv`, this is the book.
