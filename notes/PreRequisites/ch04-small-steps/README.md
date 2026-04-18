# Ch.4 — Small Steps on a Curve

> *"That's one small step for [a] man, one giant leap for mankind."*
> — **Neil Armstrong**, Apollo 11, Sea of Tranquility, 20 July 1969
>
> Armstrong's boot-print was 30 cm of motion at the end of a 384 000 km journey, and that sentence is also the best one-line description of how every modern machine-learning model is trained. A neural network learns by taking billions of those 30-cm-class steps — each one a tiny, almost imperceptible parameter adjustment — and the *leap* (recognising faces, translating languages, playing Go) is just the cumulative trajectory. Ch.4 is the chapter where we work out what a single step should look like.

> **Running theme.** You're taking a long goal kick: your boot can deliver a fixed strike speed, but you *can* choose the launch angle. Which angle drives the ball the farthest from that fixed launch speed? You *could* solve it with calculus. But the moment you add air drag, a stiff crosswind, or a sloping pitch, the equation becomes ugly and the analytical answer vanishes. The fall-back is ancient and universal: **pick a starting guess and walk downhill.**

---

## 1 · Core Idea

Suppose you have a smooth curve $f(\theta)$ and you want to find the $\theta^\star$ that minimises it (or maximises — same problem, sign flipped). The recipe is:

1. Start somewhere: $\theta_0$.
2. Compute the slope there: $f'(\theta_0)$.
3. Step a *little* in the opposite direction: $\theta_1 = \theta_0 - \eta\,f'(\theta_0)$.
4. Repeat until you stop moving.

That's it. That's gradient descent in 1-D. The whole of deep-learning optimisation is this idea wearing more elaborate clothes.

The only question is **how big a step?** The answer is the subject of this chapter.

---

## 2 · Running Example

Long goal kick, vacuum physics, fixed strike speed $v_0 = 25$ m/s, ball struck from the turf. The horizontal range when you launch at angle $\theta$ is

$$R(\theta) \;=\; \frac{v_0^2}{g}\,\sin(2\theta)$$

Maximum at $\theta = 45^\circ$, giving $R \approx 63.7$ m. We *know* the answer; this makes it easy to test whether an optimisation algorithm actually finds it.

Two twists in this chapter:

1. **Step-size twist.** Even on this simple curve, a bad $\eta$ either crawls forever or overshoots wildly.
2. **Non-convexity twist.** Add a wind penalty and the curve grows a second hump. Now the starting angle decides whether you find the *global* optimum or get stuck at a *local* one.

---

## 3 · Math

### 3.1 · The update rule

To *minimise* $f(\theta)$:

$$\theta_{k+1} \;=\; \theta_k \,-\, \eta\,f'(\theta_k)$$

- $\eta$ (eta) is the **step size** or **learning rate** — always positive.
- If $f'(\theta_k) > 0$, the curve is going up to the right; subtracting moves us left — *down*. ✓
- If $f'(\theta_k) < 0$, the curve is going down to the right; subtracting moves us right — *down*. ✓
- If $f'(\theta_k) = 0$, we've reached a critical point. Stop.

To *maximise* $f(\theta)$ (our range case), flip the sign:

$$\theta_{k+1} \;=\; \theta_k \,+\, \eta\,f'(\theta_k)$$

Same algorithm, walking uphill instead. Most ML code is written in the minimise form; if your real target is "maximise likelihood" you minimise *negative* likelihood and keep the sign convention.

### 3.2 · Why small steps work — Taylor's theorem in one line

Near a point $\theta_k$, any smooth function is approximately

$$f(\theta_k + \Delta) \;\approx\; f(\theta_k) + f'(\theta_k)\,\Delta + \mathcal{O}(\Delta^2)$$

The first-order term $f'(\theta_k)\,\Delta$ is a *linear* function of $\Delta$ — and its sign tells us which direction $\Delta$ makes $f$ smaller. If $\Delta$ is tiny, the $\mathcal{O}(\Delta^2)$ curvature leftover is negligible, so the linear prediction is trustworthy.

If $\Delta$ is *large*, the quadratic curvature term dominates and the linear approximation lies. That is the entire reason step sizes must be small.

This is the mathematical content of Armstrong's epigraph: a *small* step is one short enough for the linear approximation to hold, so we can trust its sign. Ask for a giant leap in one update and the curvature lies to you — the iterate lands somewhere unrelated to the direction you thought you were walking in.

### 3.3 · When the step is too large

On a quadratic bowl $f(\theta) = \tfrac{1}{2}c(\theta - \theta^\star)^2$ the update becomes

$$\theta_{k+1} - \theta^\star \;=\; (1 - \eta\,c)\,(\theta_k - \theta^\star)$$

Let $\rho = 1 - \eta\,c$. Behaviour depends on $|\rho|$:

| Value of $\rho$ | Behaviour |
|---|---|
| $0 < \rho < 1$ | monotone convergence (step-by-step closer) |
| $-1 < \rho < 0$ | oscillating convergence (overshoots, but shrinking) |
| $\rho = \pm 1$ | perpetual orbit — never converges |
| $|\rho| > 1$ | **divergence** — iterates blow up |

So the safe range is $0 < \eta < 2/c$. For a goal-kick-style range curve, $c$ is set by the second derivative at the peak. Practitioners usually start small ($\eta \sim 10^{-3}$ to $10^{-1}$ relative to the scale of the problem) and tune by watching the loss.

### 3.4 · Convergence is not guaranteed to be *global*

The update rule only sees the *local* slope. If your landscape has multiple minima (or maxima), you converge to the nearest one reachable from your start. This is the headline problem of deep learning: neural-network loss surfaces are staggeringly non-convex, and we have no general method to guarantee a global optimum.

Practical defences:

- **Random restarts.** Try many starts and keep the best.
- **Momentum.** Add inertia so small bumps don't trap you. (ML Ch.5.)
- **Noise injection.** Stochastic gradient descent is noisy enough to jiggle out of shallow traps.
- **Smart initialisation.** Xavier, He, LeCun schemes — not random, but calibrated to the network depth. (ML Ch.4.)

None of these *solves* non-convexity; they make it survivable.

### 3.5 · Stopping criteria

The loop has to end somehow. Common tests:

1. $|f'(\theta_k)| < \varepsilon$ — gradient has essentially vanished.
2. $|\theta_{k+1} - \theta_k| < \varepsilon$ — iterates stop moving.
3. $|f(\theta_{k+1}) - f(\theta_k)| < \varepsilon$ — loss stops decreasing.
4. $k \geq K_\text{max}$ — give up after $K_\text{max}$ iterations.

In practice you use (1) or (4); (2) and (3) can fire falsely on slow plateaus.

---

## 4 · Step by Step — maximise goal-kick range by gradient ascent

1. Set $v_0 = 25$, $g = 9.81$. Define $R(\theta) = (v_0^2/g)\sin(2\theta)$ with $\theta$ in radians.
2. Compute the analytic derivative: $R'(\theta) = (2 v_0^2 / g)\cos(2\theta)$.
3. Pick a start $\theta_0$ (say, $20^\circ$ in radians) and a step size $\eta$.
4. Loop: $\theta \leftarrow \theta + \eta\,R'(\theta)$.
5. Stop when $|R'(\theta)| < 10^{-6}$ or after 500 iterations.
6. Print $\theta_\text{final}$ in degrees. It should land near $45^\circ$.

The whole algorithm is six lines of Python. It scales to billions of parameters (deep nets) with no change of principle.

---

## 5 · Key Diagram

![Ch.4 hero: three panels showing iterative optimisation on the goal-kick range curve. Left panel: three starting angles (20°, 65°, 80°) all converge via gradient ascent to the 45° optimum on the smooth R(θ) curve. Middle panel: three step sizes — η=2 crawls, η=35 converges cleanly, η=180 overshoots and oscillates. Right panel: a non-convex wind-affected curve where a start at 18° reaches the global optimum at 32° but a start at 72° gets stuck at a local plateau around 68°.](img/ch04-small-steps.png)

Left: on a convex curve, the starting point doesn't matter — everyone arrives. Middle: on the *same* curve, a poorly chosen $\eta$ ruins everything; orange overshoots from 20° all the way past the peak on the first step. Right: a windy landscape has a dominant global maximum at $32^\circ$ and a seductive local one near $68^\circ$; the starting angle decides your fate.

---

## 6 · Code Skeleton

```python
import numpy as np

v0, g = 25.0, 9.81

def R(theta):                       # range, theta in RADIANS
    return v0 ** 2 / g * np.sin(2 * theta)

def dR(theta):                      # derivative w.r.t. theta (rad)
    return 2 * v0 ** 2 / g * np.cos(2 * theta)

def maximise(start_deg, eta, tol=1e-6, max_iter=500):
    theta = np.radians(start_deg)
    history = [theta]
    for k in range(max_iter):
        g_k = dR(theta)
        if abs(g_k) < tol:
            break
        theta = theta + eta * g_k    # ASCENT (+); use - for descent
        history.append(theta)
    return np.degrees(theta), np.degrees(history), k + 1

best, traj, steps = maximise(start_deg=20, eta=0.6)
print(f"converged to θ = {best:.4f}°  in {steps} steps")
```

Note the step size here ($\eta = 0.6$) is in *radian* space; the hero image uses degree-space so $\eta$ looks much larger (35) to produce the same effect.

---

## 7 · What Can Go Wrong

- **Wrong sign.** Forgetting that maximisation is ascent ($+$) and minimisation is descent ($-$) sends you straight away from the answer at top speed. Symptom: loss *increases* every step.
- **Units matter for $\eta$.** If you switch from radians to degrees, your effective step size rescales by $\approx 57$. Always re-tune $\eta$ after changing coordinates.
- **Plateaus.** Where the gradient is near zero but we are not at an optimum (shallow terrain), the algorithm moves glacially. Momentum helps.
- **Saddle points** (Ch.6 will formalise). Zero-gradient points that are minima in one direction and maxima in another. 1-D has no saddles, but 2-D and beyond have them *everywhere*; deep-learning losses are said to have far more saddles than local minima.
- **Numerical gradient too noisy.** If $f'$ is estimated by finite differences, the step direction jitters. Always use the analytic derivative when you can, or autodiff (Pre-Req Ch.6).
- **Forgetting to stop.** Running 10 000 iterations when 50 suffice wastes compute; running 5 iterations when you needed 500 gives garbage. Always log the loss curve and inspect it.

---

## 8 · Exercises

1. **Analytic check.** Compute $R'(\theta) = 0$ by hand for the vacuum range formula. Confirm that $\theta = 45^\circ$ is the unique solution in $(0^\circ, 90^\circ)$.
2. **Bad sign.** Change `+ eta * g_k` to `- eta * g_k` in the code skeleton and predict what happens from $\theta_0 = 20^\circ$. Then run it.
3. **Step-size upper bound.** For the quadratic bowl $f(\theta) = \tfrac{1}{2}(\theta - 3)^2$, what is the largest $\eta$ that still converges? Verify your answer empirically.
4. **Divergence.** For the same bowl, pick $\eta$ just above the bound from Exercise 3. Log $|\theta_k - 3|$ and confirm exponential blow-up.
5. **Restart rescue.** In the non-convex panel of the hero image, which starting angles converge to the global optimum and which to the local? Sweep $\theta_0$ from $5^\circ$ to $85^\circ$ in $2^\circ$ increments and plot the final $R$ against $\theta_0$. The staircase you see is called a **basin-of-attraction plot**.

---

## 9 · Where This Reappears

- **Pre-Req Ch.6.** The update $\theta \leftarrow \theta - \eta\,\nabla f(\theta)$ is the vector version of this chapter — same logic, with many dimensions.
- **ML Ch.1 Linear Regression.** Stochastic gradient descent on the MSE loss. Convex, so Ch.4's easy case applies.
- **ML Ch.5 Backprop & Optimisers.** Momentum, Adam, RMSProp, learning-rate schedules — every one is a patch on the Ch.4 base algorithm.
- **ML Ch.6 Regularisation.** Adds a penalty term to the loss, still optimised with the same walk-downhill recipe.
- **Reinforcement learning** and **variational inference.** Both ultimately maximise an expectation using stochastic ascent.

Read back to Armstrong's line with fresh eyes: *"one small step… one giant leap…"* is literally the update rule of gradient descent *and* its long-run behaviour. Pre-Req Ch.6 makes the step a vector, and the entire ML book compounds those vector steps into the leaps we call *learning*.

---

## 10 · References

- **Jon Krohn — *Calculus 2 for Machine Learning*.** The gradient-descent episode uses the same geometric framing as this chapter.
- **3Blue1Brown — *Gradient descent, how neural networks learn*.** The 2-D visual intuition in ep. 2 maps directly onto the left panel of our hero image.
- **Boyd & Vandenberghe — *Convex Optimization*, Ch.9.** The rigorous treatment of step-size selection, including backtracking line search — the standard industrial fix for Section 3.3's tuning question.
- **Bottou, Curtis, Nocedal (2018), *Optimization Methods for Large-Scale Machine Learning*.** Modern survey; explains why SGD's noise is a *feature* in the non-convex ML setting.
