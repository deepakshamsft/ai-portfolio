# Plan — 01-Regression (remaining work)

**Audit date:** 2026-04-24  
**Completed:** Notation blocks, `0 · The Challenge` sections, 3–5 row numeric verification in all Math sections, Animation GIFs verified, RNG seeds set in all notebooks.  
**Re-verified:** 2026-04-24 — all items below confirmed still pending (Mermaid emojis present; notebook todos require manual verification).

---

## README TODOs

- [ ] **Mermaid emoji cleanup** — Remove non-approved emojis from Mermaid flowchart node labels. Remaining instances:
  - `ch02-multiple-regression/README.md` line 55: `TARGET["🎯 $40k target"]` → replace 🎯 with text only
  - `ch04-polynomial-features/README.md` line 58: `TARGET["🎯 $40k target…"]` → replace 🎯 with text only
  - `ch03-feature-importance/README.md` line 1015: `FIX3["📈 Increase n_repeats…"]` → replace 📈 with text only

---

## Notebook TODOs

- [ ] **Mirror README section structure** — Add `## Running Example`, `## Math`, and `## Step by Step` (or equivalent walkthrough heading) as markdown cells in the notebooks for **ch02, ch03, ch05, ch07**. ch01 and ch06 already have the structure.

- [ ] **Add deterministic small dataset cells** — For each chapter notebook, add a code cell that reproduces the README's numeric worked example using a 3–5 row toy dataset. RNG seeds (`SEED = 42`) are already set; what is missing is the explicit cell that mirrors the README table (e.g., the OLS 3-row table in ch01 Math §4.2, the gradient-step table in ch02 Math §3.3, etc.). Needed in: **ch02, ch03, ch04, ch05, ch06, ch07**.

- [ ] **TensorBoard logging (optional)** — Add an optional, commented-out TensorBoard `SummaryWriter` cell in chapters with explicit training loops: **ch01, ch02, ch04, ch05, ch07**. Mark clearly as `# Optional: TensorBoard` so CI skips it by default.
