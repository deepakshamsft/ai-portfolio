# Ch.3 Plan — Remaining Work

**Status:** 7 of 8 approved TODOs complete. 1 item outstanding.

---

## Remaining TODO

### TODO 8 (partial) — Add Spearman correlation to Filter Methods subsection

**Location:** `README.md` → `§3 Math` → `### Filter Methods — Scoring Features Before Training`

**What's already done:** Pearson correlation (formula + by-hand walkthrough), Mutual Information (formula + reference), Lasso pointer.

**What's missing:** Spearman correlation entry.

**Insert** a `**Spearman Correlation (monotonic, non-parametric):**` block between the Pearson section and the Mutual Information section. Content (≈ 6–8 lines):

- Uses rank transforms — handles non-linear but monotonic relationships (e.g., a curve that always goes up, just not at a constant rate).
- Formula: $\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$ where $d_i$ is the rank difference between $x_i$ and $y_i$.
- When to prefer over Pearson: when the scatter plot shows a curve rather than a straight line (concave/convex trend).
- No numerical walkthrough needed — just the formula + one-sentence interpretation.
- Add one sentence: `scipy.stats.spearmanr` or `pd.DataFrame.corr(method='spearman')`.

**No new visuals, no new gen scripts, no new code blocks required** — the existing code block in §7 already uses `pearsonr` and `mutual_info_regression`; adding Spearman is prose-only.

---

**After this item is merged, delete this plan.md.**
