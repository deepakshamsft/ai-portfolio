# Plan — Math Under the Hood Track

**Last updated:** 2026-04-24
**Audit scope:** All 7 chapters under `notes/math_under_the_hood/`
**Status:** Near-compliant — all chapters pass content checks; only minor structural gaps remain

## Legend
- 🐍 = Python script needed — listed in Scripts table, generated in one batch pass
- ✏️ = Manual content edit — implemented directly by Copilot
- ✅ = Done

---

## 🐍 Script Todos

<!-- SCRIPTS_HOOK_START
track: math_under_the_hood
-->

| Script | File(s) | What it does |
|--------|---------|--------------|
| `scripts/fix_math_animation_stubs.py` | All 7 `README.md` files | Insert `## Animation\n\n> 🎬 *Animation placeholder — see `img/chNN-[topic]-animation.gif`.*\n` immediately after `## 0 · The Challenge` heading (ch04 and ch06 have inline GIF references inside §3/§5 — the script should check and skip those or wrap them with a section header instead) |
| `scripts/fix_math_mermaid_arc_stubs.py` | All 7 `README.md` files | Append a 7-chapter Mermaid `graph LR` stub to each Progress Check section |
| ~~`scripts/fix_math_ch03_missing_section.py`~~ | ~~`ch03_calculus_intro/README.md`~~ | ✅ Not needed — `## 10 · Where This Reappears` already present with content |
| ~~`scripts/fix_math_ch06_duplicate_section0.py`~~ | ~~`ch06_gradient_chain_rule/README.md`~~ | ✅ Done manually — `## 0 · Bridge from Ch.3` renamed to `## Bridge from Ch.3` (non-numbered) |
| ~~`scripts/fix_math_ch06_code_skeleton_stub.py`~~ | ~~`ch06_gradient_chain_rule/README.md`, `ch07_probability_statistics/README.md`~~ | ✅ Done manually — full `## 8.5 · Code Skeleton` sections added to both chapters |

<!-- SCRIPTS_HOOK_END -->

---

## ✏️ Manual Content Todos

### Track-wide

- [ ] **Generate Animation GIFs** for ch01, ch02, ch03, ch05, ch07 — ch04 and ch06 already have inline GIFs (gradient descent and backprop animations). The remaining 5 chapters need visual animations matching their core concept. Suggestions:
  - ch01: dot-product and matrix-vector multiply animation
  - ch02: parabola vs linear boundary comparison
  - ch03: secant → tangent limit visualization
  - ch05: matrix-vector product (broadcast across columns)
  - ch07: Gaussian likelihood surface with MLE optimum

- [ ] **Write Mermaid chapter arcs** for all 7 Progress Check sections — a `graph LR` showing ch01→ch07 with green (completed), amber (current), grey (upcoming) color coding using approved palette (`#15803d`, `#b45309`, `#1e3a8a`).

### ch03_calculus_intro
- [x] ✅ `## 10 · Where This Reappears` already present with forward pointers to ML track chapters — no action needed

### ch04_small_steps
- [ ] **Wrap inline GIF reference** in a dedicated `## Animation` section header *(script handles the header; manual step: verify the GIF file `img/ch04-gradient-descent-animation.gif` exists and the reference inside `## 3 · Math` is either kept or moved to the new dedicated section)*

### ch06_gradient_chain_rule
- [x] ✅ **Fixed duplicate `## 0 ·` section** — renamed preamble to `## Bridge from Ch.3 — One Variable to Many` (non-numbered)
- [ ] **Wrap inline backprop GIF** in a dedicated `## Animation` section header
- [x] ✅ **Added `## 2 · Running Example`** — knuckleball free-kick 8-parameter gradient scenario
- [x] ✅ **Added `## 8.5 · Code Skeleton`** — backprop from scratch (Educational) + PyTorch autograd (Production)

### ch07_probability_statistics
- [x] ✅ **Renamed `## 2 · Probability Foundations`** → `## 2 · Running Example — Probability Foundations`
- [ ] **Add Step-by-Step wrapper** around the MLE derivation in `## 7 · Headline Derivation`
- [x] ✅ **Added `## 8.5 · Code Skeleton`** — Gaussian MLE from scratch (Educational) + scipy.stats (Production)

---

## Notes
- ch01, ch02, ch04, ch05 are near-complete (PASS with minor gaps — only Animation stub and Mermaid arc missing). Quick wins.
- ch03, ch06, ch07 are PARTIAL — need slightly more work (missing sections, structural fixes).
- All 7 chapters have `img/` folders and strong mathematical content ✅.
- All chapters use scalar-first derivations and verbal glosses correctly ✅.
