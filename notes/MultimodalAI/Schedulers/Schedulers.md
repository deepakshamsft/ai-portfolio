# Schedulers — From 1000 Steps to 20 Without Retraining

> After reading this you will understand why swapping the sampler is the cheapest quality/speed lever in modern diffusion, and how DDIM, DPM-Solver, and LCM differ.

## 1 · Core Idea

Training a DDPM takes 1 000 noisy steps. But *inference doesn't have to*. A **scheduler** is the algorithm that converts a trained noise predictor into actual images. The model weights never change; only the sequence of steps and the update rule change.

| Scheduler | Steps needed | Deterministic? | Year |
|-----------|-------------|----------------|------|
| DDPM      | 1 000       | No (stochastic)| 2020 |
| DDIM      | 20–50       | Yes            | 2020 |
| DPM-Solver| 5–20        | Mostly yes     | 2022 |
| DPM-Solver++ | 5–15     | Yes            | 2022 |
| UniPC     | 5–10        | Yes            | 2023 |
| LCM       | 1–4         | Yes (separate fine-tune) | 2023 |

**Key insight:** DDPM's 1 000 steps are required during *training* because gradients must backpropagate through fine-grained time increments. At *inference* you can skip steps—as long as you can solve the underlying reverse ODE accurately.

## 2 · Running Example

PixelSmith v4 (coming in Ch.7) will use SD-Turbo or SDXL-Turbo with a **DPM-Solver++ 4-step** schedule. Right now we illustrate scheduling by replaying stored noise trajectories from our Ch.4 DDPM to show what happens when you take coarser strides.

## 3 · The Math

### DDPM — Why 1 000 Steps?

The DDPM posterior is:

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}\!\left(\tilde{\mu}_t(x_t, x_0),\; \tilde{\beta}_t\,\mathbf{I}\right)$$

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,x_0 + \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t, \qquad \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\,\beta_t$$

Each step is small (β≈0.02 at most) so the Gaussian approximation holds. Bigger jumps → approximation breaks down.

### DDIM — Deterministic ODE Solver

Song et al. (2020) reformulated diffusion as an **ODE** (no stochastic term). The DDIM update for going from $x_\tau$ at timestep $\tau$ to $x_{\tau'}$ at timestep $\tau' < \tau$ is:

$$x_{\tau'} = \sqrt{\bar{\alpha}_{\tau'}}\,\hat{x}_0 + \sqrt{1-\bar{\alpha}_{\tau'}-\sigma_\tau^2}\;\hat{\epsilon}_\theta(x_\tau, \tau) + \sigma_\tau\,\epsilon_\tau$$

where $\hat{x}_0 = (x_\tau - \sqrt{1-\bar{\alpha}_\tau}\,\hat{\epsilon})/\sqrt{\bar{\alpha}_\tau}$ and $\sigma_\tau=0$ gives the fully deterministic (ODE) case.

Because it's an ODE, you can jump from $\tau=999$ to $\tau=950$ to $\tau=900$… using only 20 timesteps and still get coherent images. The same model weights, a different index sequence.

### DPM-Solver — High-Order ODE Integration

DPM-Solver treats the reverse process as solving:

$$\frac{dx}{d\lambda} = -x + \hat{\epsilon}_\theta\!\left(x, t(\lambda)\right)$$

where $\lambda = \log\!\left(\sqrt{\bar{\alpha}_t}/\sqrt{1-\bar{\alpha}_t}\right)$ is the log-SNR. By applying **exponential integrators** (2nd/3rd order Taylor expansion), DPM-Solver achieves much lower numerical error per step than DDIM's first-order method.

Practical consequence: DDIM needs ~50 steps for clean output; DPM-Solver++ needs 10–15.

### LCM — Latent Consistency Model

LCM adds a **consistency distillation** fine-tune: the model learns to predict $x_0$ directly in 1–4 steps by supervising the consistency condition $f_\theta(x_t, t) = f_\theta(x_{t'}, t')$ for all pairs $(t, t')$ on the same trajectory. Requires retraining on top of a pretrained SD checkpoint.

## 4 · How It Works — Step by Step

### Choosing a Sub-Sequence of Timesteps

Standard DDPM trains on $t \in \{0, 1, \ldots, 999\}$. At inference you pick a **subsequence**:

```
DDPM-1000: [999, 998, 997, ..., 1, 0]          (1000 steps)
DDIM-50:   [999, 979, 959, ..., 19, 0]          (50 steps, uniform stride 20)
DDIM-20:   [999, 949, 899, ..., 49, 0]          (20 steps, stride 50)
DPM-10:    [999, 892, 756, 617, 492, ...] (10 steps, non-uniform, SNR-optimal)
```

The non-uniform spacing in DPM-Solver concentrates steps where the noise schedule changes rapidly (low-SNR region, i.e., early timesteps).

### Speed vs Quality Trade-Off Table

| Steps | Scheduler | FID (COCO 5k) | Wall time (A100) |
|-------|-----------|---------------|-----------------|
| 1000  | DDPM      | 3.2           | 42 s            |
| 50    | DDIM      | 4.0           | 2.1 s           |
| 20    | DDIM      | 5.5           | 0.85 s          |
| 15    | DPM-Solver++ | 4.1        | 0.64 s          |
| 4     | LCM       | 6.8           | 0.17 s          |

*(Illustrative figures. Real numbers depend on model and resolution.)*

### Stochastic vs Deterministic Sampling

- **Stochastic (DDPM):** Add noise at every step. Same seed → different image. Good for diversity.
- **Deterministic (DDIM, DPM-Solver):** No added noise. Same seed + same scheduler → same image every time. Enables interpolation in latent space by interpolating the seed $x_T$.

## 5 · The Key Diagrams

```
Noise schedule visualised — bar_alpha per timestep:

 1.0 |████████████████▓▓▓▓▓░░░░                         |
     |                          ░░░░                      |
 0.0 |                                ▄▄▄▄▄▄▄▄▄▄▄████████|
      t=0 (pure signal)                         t=999 (pure noise)

DDIM seeks "big strides" across this curve, treating it as a
continuous ODE instead of 1000 discrete Gaussian transitions.


Timestep sub-sequences:

    DDPM (1000)  ████████████████████████████████████████ (1000 dots)
    DDIM (50)    █   █   █   █   █   █   █   █   █   █   (50 dots)
    DPM (15)     ██    ██   █    █    █     █      █       (clustered near endpoints)
```

## 6 · What Changes at Scale

- **SD 1.x / SDXL** ship with PNDM scheduler by default but accept any compatible scheduler via the `diffusers` `Scheduler` API.
- **SDXL-Turbo / SD-Turbo** are specifically distilled for 1–4 step schedules (ADD distillation). Using a standard 50-step DDIM on them wastes compute.
- **Consistency Models (CM)** and **LCM** require fine-tuning but unlock real-time inference on consumer hardware.
- **Flux (2024)** uses a rectified flow scheduler — a straight-line path from noise to data instead of the cosine/linear variance schedule.

## 7 · Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "DDIM needs retraining" | No — any DDPM-trained model can use DDIM at inference |
| "More steps always = better quality" | Past ~30 steps with DPM-Solver, quality plateaus |
| "Deterministic means boring" | Use different seeds for diversity; determinism just means reproducibility |
| "DDPM noise schedule β_t is fixed" | You can redesign it: cosine schedule (improved DDPM) gives more uniform SNR across timesteps |
| "CFG doubles compute; so does DDIM" | DDIM halves step count; net effect is a large speedup even with CFG |

## 8 · Interview Checklist

### Must Know
- Why DDPM needs 1 000 steps at inference: each step is a Gaussian approximation that only holds for small β — larger strides violate the Markov assumption
- DDIM key insight: rewrite as ODE, enabling deterministic sub-sequence sampling
- The trade-off axes: steps ↓ speed ↑ quality ↓ diversity (generally true)

### Likely Asked
- *"What scheduler does Stable Diffusion use by default?"* — PNDM (pseudo-numerical) or DDIM; SDXL defaults to EulerDiscreteScheduler
- *"How would you halve inference time without quality loss?"* — Switch from 50-step DDIM to 15-step DPM-Solver++
- *"What is the relationship between DDIM and DDPM?"* — DDIM is a non-Markovian generalization that reduces to DDPM when σ=original noise level; at σ=0 it becomes fully deterministic

### Trap to Avoid
- Don't confuse the **training** noise schedule (always 1 000 steps, defines q(x_t|x_0)) with the **inference** step count (scheduler-specific). They are independent after training.- **LCM / distillation:** Latent Consistency Models learn to map any noisy latent directly to the clean latent in 1–4 steps by enforcing self-consistency along the ODE trajectory; LCM-LoRA distils this as a lightweight adapter. SD-Turbo and SDXL-Turbo use adversarial diffusion distillation. Trap: "LCM images are the same quality as 50-step DDIM" — 1–4 step models sacrifice fine detail and diversity; 8+ steps are usually needed to match 50-step DDIM quality
## 9 · What's Next

[LatentDiffusion.md](../LatentDiffusion/LatentDiffusion.md) — the 512×512 pixel image is too expensive to diffuse directly. SD compresses it 8× into a latent space first: that's what makes real-time generation possible.
