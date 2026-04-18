# Pre-Requisites — "Basketball → Knuckleball Free Kick" rebrand plan

> Scope: rewrite the running-theme layer across `notes/PreRequisites/` so the spine becomes a **direct knuckleball free kick** instead of a basketball free throw. Math is unchanged. Numerical constants update cleanly. Hero-image alt-text gets updated to match; actual image regeneration is tracked separately.

---

## 1 · Theme rebrand — at a glance

| | Before (basketball) | After (football / soccer) |
|---|---|---|
| Shot type | Free throw | **Direct knuckleball free kick** (no spin → pure 2D parabola, no Magnus) |
| Player | Basketball player / shooter | Striker / kicker |
| Distance | 4.57 m (15 ft) to hoop | **20 m** to goal |
| Target height | 3.05 m hoop | **2.44 m crossbar** (top of goal frame) |
| Release height $h_0$ | 2.10 m (player's hand) | **0 m** (ball on ground) |
| Obstacles | Defender's raised hand | **Defensive wall** at 9.15 m (10 yd), ~2 m effective height |
| "Linear regime" | First 0.2 s | **First ~0.1 s** (ball leaves boot faster) |
| Vertical release velocity | $v_{0y} = 7.2$ m/s | $v_{0y} = 6.5$ m/s |
| Full-shot speed & angle (Ch.2) | 7.5 m/s at 52° | **25 m/s at 15°** (typical direct-kick release) |
| Fixed-speed range problem (Ch.4) | $v_0 = 8$ m/s | $v_0 = 25$ m/s (long goal-kick framing) |
| 8 features (Ch.5–6) | release angle, speed, height, wind×2, spin, altitude, fatigue, defender-distance | strike speed, strike angle, strike zone (boot inside/outside), wall height, wall distance, wind speed, pitch wetness, fatigue |
| Ch.7 "make / miss" | hoop success | goal scored / saved / missed |

Why knuckleball: it's the purist's direct free kick (Juninho, Pirlo, Ronaldo) — struck with minimal spin so the ball's trajectory stays dominated by gravity. This sidesteps the Magnus curve entirely and keeps the 2D-parabola physics identical to the basketball version.

---

## 2 · Canonical numerical constants (shared across chapters)

```
g        = 9.81 m/s²
v0       = 25.0 m/s        # strike speed (Ch.2, Ch.4)
theta0   = 15°             # release angle (Ch.2)
v0y      = 6.5 m/s         # vertical component (Ch.1, Ch.3, Ch.5)
v0x      = 24.15 m/s       # horizontal component (Ch.2 derivations)
h0       = 0.0 m           # release height (Ch.1, Ch.5)
D_goal   = 20.0 m          # distance to goal line
H_bar    = 2.44 m          # crossbar height
D_wall   = 9.15 m          # wall distance (10 yd)
H_wall   = 2.0 m           # effective wall height (static + jump)
t_linear = 0.10 s          # linear-regime cutoff (Ch.1)
t_apex   = v0y/g ≈ 0.66 s  # apex of parabola (Ch.3)
R_max    = v0²/g ≈ 63.7 m  # max range at 45° (Ch.4)
```

---

## 3 · Checklist — files to update

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done.

### 3.1 · READMEs (narrative layer only — math untouched)

- [x] [README.md](README.md) — main index: running-thread paragraph, chapter table, artifact blurbs
- [x] [ch01-linear-algebra/README.md](ch01-linear-algebra/README.md) — running theme, §2, §6 code (`w=7.2→6.5`), §8 exercise 2, §5 image caption
- [x] [ch02-nonlinear-algebra/README.md](ch02-nonlinear-algebra/README.md) — running theme, §2, §3.1 polynomial table, §5 image caption, §6 code (`v0=7.5→25, theta=52→15`), §8 exercise 5
- [x] [ch03-calculus-intro/README.md](ch03-calculus-intro/README.md) — running theme, §2, §4.1 apex computation, §5 integral, §6 step-by-step, §7 image caption, §10 exercise 2
- [x] [ch04-small-steps/README.md](ch04-small-steps/README.md) — running theme, §2 (`v0=8→25`), §5 image caption, §6 code, §7 bullet, §8 exercise 5
- [x] [ch05-matrices/README.md](ch05-matrices/README.md) — running theme (8 features), §2, §4 step-by-step, §5 image caption, §6 code (`v0y=7.2→6.5, h0=2.10→0.0`), §8 exercise 3
- [x] [ch06-gradient-chain-rule/README.md](ch06-gradient-chain-rule/README.md) — running theme (8 features), image caption at top
- [x] [ch07-probability-statistics/README.md](ch07-probability-statistics/README.md) — running theme, §2 sample-space example, §3 "free-kick stories" bullets

### 3.2 · Out of scope for this pass (tracked as follow-ups)

- [ ] Notebooks (`ch0*/notebook.ipynb`) — numerical constants and captions need the same refresh; run + render plots
- [ ] Hero images (`ch0*/img/*.png`) — currently depict basketball scenes; regenerate using the updated story. Alt-text in READMEs has been updated to describe the **new** intended image; existing PNGs are stale until regenerated.
- [ ] `AUTHORING_GUIDE.md` — confirm running-thread convention example if it names basketball explicitly.
- [ ] Cross-references from `notes/ML/` that point back at Pre-Req chapters mentioning "free throw" — grep and fix if any remain.

---

## 4 · Authoring notes

- Keep all mathematics, derivations, rules-of-thumb and pitfalls intact. Only the narrative framing and numerical constants change.
- Prefer "striker" or "kicker" over "player" once context is established; first mention each chapter can use "football (soccer) player striking a direct knuckleball free kick" so readers worldwide know what's being described.
- On first mention in each chapter, call out **knuckleball** explicitly — it's the in-fiction justification for modelling the ball as a pure 2D parabola (no Magnus curve).
- For Ch.4's "maximise range" setup, frame it as a **long goal kick** at a fixed strike speed rather than a direct free kick — the 45° answer requires maximum range, which isn't what a 20 m free kick wants. Keep the math identical.
