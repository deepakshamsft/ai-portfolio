# Ch.7 — Probability & Statistics

![Bernoulli/Binomial/Gaussian, CLT histogram, and MLE likelihood surface](./img/ch07-probability-mle.png)

> **The story.** Probability began in 1654 as a letter exchange between **Pascal** and **Fermat** about a gambler's dice problem; **Jacob Bernoulli** turned it into a science with *Ars Conjectandi* (1713), and **Laplace** (1812) gave it the language of distributions and expectations. **Gauss** (1809) introduced the normal distribution while reconciling astronomical observations and — crucially for us — proved that least-squares regression is the right answer when noise is Gaussian. **R. A. Fisher** (1922) then formalised **maximum likelihood estimation**, which is *the* reason we use mean-squared error for regression and cross-entropy for classification. Finally **Kolmogorov** (1933) put the whole field on a measure-theoretic foundation. Every loss function in this curriculum is, secretly, a likelihood from this chapter.
>
> **Where you are in the curriculum.** Up to Ch.6 we pretended a knuckleball free kick was deterministic: same angle, same speed, same outcome. Reality is noisier — muscle fatigue, boot-on-ball contact variance, a divot under the ball, micro gusts of wind. Every outcome is a *draw* from a distribution. This is the chapter that gives you the language to (a) describe those distributions, (b) summarise them with expectations and variances, (c) estimate their parameters from data, and (d) discover why **MSE is not a design choice — it is what Gaussian noise asks for**. ML Ch.15 will pull this idea apart in detail.
>
> **Notation in this chapter.** $P(A)$ — probability of event $A$; $X$ — a random variable; $p(x)$ — probability density (continuous) or mass (discrete); $\mathbb{E}[X]$ — expectation (mean); $\text{Var}(X)=\sigma^2$ — variance; $\sigma$ — standard deviation; $\mu$ — distribution mean; $\mathcal{N}(\mu,\sigma^2)$ — Gaussian (normal) distribution with mean $\mu$ and variance $\sigma^2$; $\boldsymbol{\theta}$ — parameter vector of a probabilistic model; $p(y\mid\boldsymbol{\theta})$ — conditional density of $y$ given parameters; $\mathcal{L}(\boldsymbol{\theta})=\prod_i p(y_i\mid\boldsymbol{\theta})$ — **likelihood**; $\hat{\boldsymbol{\theta}}=\arg\max_{\boldsymbol{\theta}}\log\mathcal{L}$ — **maximum-likelihood estimate (MLE)**; $\varepsilon_i$ — a noise term.

---

## 1 · Core Idea

A **random variable** $X$ is a quantity whose value is uncertain, characterised by a probability distribution. Three operations exhaust most of what we'll need:

1. **Describe** — give the distribution (PMF for discrete, PDF for continuous).
2. **Summarise** — compute expectations $\mathbb{E}[X]$ and variances $\mathrm{Var}(X)$.
3. **Infer** — given data, estimate the distribution's parameters (MLE, Bayes).

Almost every machine-learning loss function is the negative-log-likelihood of a carefully chosen distribution. Once you see that, loss design becomes distribution design.

---

## 2 · Probability, in One Page

- **Sample space** $\Omega$ — the set of possible outcomes of a random experiment. (All free-kick trajectories you could ever see.)
- **Event** — a subset of $\Omega$. (The event *"the ball goes in"*.)
- **Probability** $\mathbb{P}: \text{events} \to [0, 1]$ with $\mathbb{P}(\Omega) = 1$ and countable additivity.
- **Conditional probability**: $\mathbb{P}(A \mid B) = \mathbb{P}(A \cap B) / \mathbb{P}(B)$ for $\mathbb{P}(B) > 0$.
- **Independence**: $A$ and $B$ independent iff $\mathbb{P}(A \cap B) = \mathbb{P}(A) \mathbb{P}(B)$.
- **Bayes' theorem**: $\mathbb{P}(A \mid B) = \mathbb{P}(B \mid A) \mathbb{P}(A) / \mathbb{P}(B)$. The whole of inference is applications of this one line.

A **random variable** $X$ is a function $X: \Omega \to \mathbb{R}$. Its **PMF** (discrete) or **PDF** (continuous) specifies how probability mass is distributed over values.

---

## 3 · Three Workhorse Distributions

| Name | Type | Parameters | PMF / PDF | Mean | Variance |
|---|---|---|---|---|---|
| Bernoulli | discrete, $\{0,1\}$ | $p$ | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ |
| Binomial | discrete, $\{0,\dots,n\}$ | $n, p$ | $\binom{n}{k} p^k (1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Gaussian | continuous, $\mathbb{R}$ | $\mu, \sigma^2$ | $\frac{1}{\sqrt{2\pi \sigma^2}} e^{-(x-\mu)^2/(2\sigma^2)}$ | $\mu$ | $\sigma^2$ |

**Free-kick stories:**

- **Bernoulli.** A single free kick is scored (1) or missed (0) with success probability $p$.
- **Binomial.** How many goals in $n$ free kicks?
- **Gaussian.** Launch-angle tremor, noise added to the *continuous* trajectory, measurement error on ball-tracking cameras.

Many others matter (Poisson for counts, Exponential for waiting times, Dirichlet for proportions, Beta as Bernoulli's conjugate prior) — but these three carry 80% of ML applications.

---

## 4 · Expectation and Variance

The **expectation** is the probability-weighted average:

- Discrete: $\mathbb{E}[X] = \sum_x x p(x)$.
- Continuous: $\mathbb{E}[X] = \int x p(x) dx$.

**Linearity** is the workhorse property: $\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$, independence not required.

The **variance** measures spread: $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$.

Two results we'll lean on:

- $\mathrm{Var}(aX + b) = a^2 \mathrm{Var}(X)$.
- For *independent* $X_1, \dots, X_n$: $\mathrm{Var}(X_1 + \dots + X_n) = \sum \mathrm{Var}(X_i)$. This is why averaging $n$ i.i.d. samples reduces variance by $1/n$.

---

## 5 · The Central Limit Theorem

**Statement.** Let $X_1, \dots, X_n$ be i.i.d. with finite mean $\mu$ and finite variance $\sigma^2$. Let $\bar X_n = \tfrac{1}{n}\sum X_i$. Then

$$\sqrt{n} (\bar X_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2) \quad \text{as } n \to \infty.$$

Equivalently, for large $n$: $\bar X_n \approx \mathcal{N}(\mu, \sigma^2/n)$.

**Why you should care.** This is the reason Gaussians are everywhere:

- Noise in a measurement is the sum of many small independent contributions — it looks Gaussian.
- A batch-averaged stochastic gradient is Gaussian-ish around the true gradient, which is why SGD behaves predictably.
- Confidence intervals $\hat\mu \pm 1.96 \hat\sigma/\sqrt{n}$ use CLT as their authority.

The middle panel of the hero image shows it viscerally: the *source* distribution is a skewed exponential, but the distribution of 10 000 sample means of size-50 batches is almost perfectly Gaussian.

---

## 6 · Maximum Likelihood Estimation

Given observations $\mathbf{y} = (y_1, \dots, y_N)$ drawn i.i.d. from a parametric model $p(y \mid \boldsymbol{\theta})$, the **likelihood** of $\boldsymbol{\theta}$ is

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} p(y_i \mid \boldsymbol{\theta}).$$

The **MLE** is the $\hat{\boldsymbol{\theta}}$ maximising $\mathcal{L}$, equivalently minimising the **negative log-likelihood** (NLL):

$$\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \log \mathcal{L}(\boldsymbol{\theta}) = \arg\min_{\boldsymbol{\theta}} -\sum_{i=1}^{N} \log p(y_i \mid \boldsymbol{\theta}).$$

Logarithms turn products into sums (numerically stable) and stretch the likelihood axis (smoother gradient landscape). This is the **NLL** that you'll minimise with the Ch.4/Ch.6 tools.

---

## 7 · Headline Derivation — MLE with Gaussian Noise ⟹ MSE

Regression setup: $y_i = f(\mathbf{x}_i; \boldsymbol{\theta}) + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$ i.i.d. Then $y_i \mid \mathbf{x}_i, \boldsymbol{\theta} \sim \mathcal{N}(f(\mathbf{x}_i;\boldsymbol{\theta}), \sigma^2)$ and

$$\log p(y_i \mid \boldsymbol{\theta}) = -\tfrac{1}{2}\log(2\pi\sigma^2) - \frac{(y_i - f(\mathbf{x}_i;\boldsymbol{\theta}))^2}{2\sigma^2}.$$

Summing over $i$ and dropping terms that don't depend on $\boldsymbol{\theta}$:

$$-\log \mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2\sigma^2} \sum_{i=1}^{N} (y_i - f(\mathbf{x}_i;\boldsymbol{\theta}))^2 + \text{const}.$$

Minimising NLL is therefore identical to minimising **sum of squared errors**. Mean-squared error wasn't a "nice convex choice" — it's a mathematical consequence of assuming Gaussian noise. Change the noise distribution and you change the loss:

| Noise / likelihood | ⟹ Loss |
|---|---|
| Gaussian | squared error (MSE) |
| Laplace | absolute error (MAE, L1) |
| Bernoulli on labels | binary cross-entropy |
| Categorical | cross-entropy |
| Poisson | Poisson deviance |

That mapping is the single most useful mental model in supervised learning.

---

## 8 · Pitfalls

1. **PMF vs PDF confusion.** A PDF can exceed 1 (it's a *density*, not a probability). Only integrals over intervals are probabilities.
2. **i.i.d. assumptions.** The MLE derivation needs samples to be independent and identically distributed. Time-series data usually aren't.
3. **Log-sum-exp.** Computing $\log \sum_i \exp(x_i)$ naively overflows. Use $\max_i x_i + \log \sum_i \exp(x_i - \max)$.
4. **MLE can overfit small samples.** With $N$ tiny, the MLE sticks to every quirk in the data. This is why we add a prior (Bayesian / MAP) or a regulariser (Ch.6 / ML Ch.6).
5. **Variance of a sample mean is $\sigma^2/n$, not $\sigma/\sqrt{n}$.** The standard *error* is $\sigma/\sqrt{n}$. Two different quantities.
6. **Correlation is not causation** and **independence implies zero correlation but not vice versa.** Classic traps.

---

## 9 · Where This Reappears

- **ML Ch.1 Linear Regression.** MSE loss is MLE under Gaussian noise — the derivation in §7 *is* Ch.1's theoretical backbone.
- **ML Ch.2 Logistic Regression.** Cross-entropy loss is MLE under a Bernoulli likelihood on the labels.
- **ML Ch.4 Neural Networks.** Softmax + cross-entropy = MLE on a categorical likelihood. Swap in a Gaussian head and you get regression.
- **ML Ch.15 MLE & Loss Functions.** Full catalogue of likelihood-to-loss correspondences.
- **Ch.6 just now.** The gradient of the NLL is what backprop propagates.
- **AI Ch.6 Bayesian models and uncertainty quantification.** Replace MLE with full Bayesian inference and you get posteriors instead of point estimates.

---

## 10 · References

- Wasserman, *All of Statistics* — the single densest statistics reference for ML practitioners.
- Bishop, *Pattern Recognition and Machine Learning*, Ch. 1–2.
- MacKay, *Information Theory, Inference, and Learning Algorithms* — the Bayesian counterpoint.
- 3Blue1Brown, *Bayes' theorem* and *Central Limit Theorem* videos.
- Murphy, *Probabilistic Machine Learning: An Introduction* Ch. 2–4.
