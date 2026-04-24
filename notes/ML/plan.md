# ML Track — Open Issues & Remaining Work

**Created:** 2026-04-24  
**Source:** `content_audit.md` (full content quality review) + per-track plan.md audits  
**Status:** Active — fix these before tagging v2.0

---

## Priority 1 — High Impact (Grand Challenge / Story Arc)

### 1.1 Wrong mission name in two NN chapters
- [ ] `notes/ML/03-NeuralNetworks/ch01-xor-problem/README.md` — `## 0 · The Challenge` blockquote says "Launch **SmartVal AI**" → change to "Launch **UnifiedAI**"
- [ ] `notes/ML/03-NeuralNetworks/ch07-mle-loss-functions/README.md` — same fix: "SmartVal AI" → "UnifiedAI"

### 1.2 Stale chapter numbering in NN track "What we know so far" bullets (6 chapters)
All NN chapters ch02–ch09 reference old single-track numbers. Remap to current NN-track chapter numbers:
- XOR = Ch.1, Neural Networks = Ch.2, Backprop = Ch.3, Regularisation = Ch.4, CNNs = Ch.5, RNNs = Ch.6, MLE = Ch.7, TensorBoard = Ch.8, Sequences = Ch.9, Transformers = Ch.10
- [ ] `03-NeuralNetworks/ch02-neural-networks/README.md` — `What we know so far` bullets
- [ ] `03-NeuralNetworks/ch03-backprop-optimisers/README.md` — `What we know so far` bullets
- [ ] `03-NeuralNetworks/ch04-regularisation/README.md` — `What we know so far` bullets (also changes "Ch.5 Backprop" → "Ch.3 Backprop")
- [ ] `03-NeuralNetworks/ch05-cnns/README.md` — `What we know so far` bullets
- [ ] `03-NeuralNetworks/ch06-rnns-lstms/README.md` — `What we know so far` bullets
- [ ] `03-NeuralNetworks/ch07-mle-loss-functions/README.md` — `What we know so far` bullets (also incorrectly claims Constraints #3 and #4 already achieved at Ch.7 — remove those claims)
- [ ] `03-NeuralNetworks/ch08-tensorboard/README.md` — `What we know so far` bullets
- [ ] `03-NeuralNetworks/ch09-sequences-to-attention/README.md` — `What we know so far` bullets

### 1.3 Constraint #1 accuracy target inconsistency in NN track
- [ ] `03-NeuralNetworks/ch03-backprop-optimisers/README.md` — challenge blockquote says `<$40k MAE` (Regression target); change to `<$50k MAE` (NN track intermediate target). Also in §9 Progress Check: note it is test MAE not training MAE.

---

## Priority 2 — Medium Impact (Structural / Correctness)

### 2.1 Misplaced section headings
- [ ] `03-NeuralNetworks/ch06-rnns-lstms/README.md` — `## 9 · Where This Reappears` stub appears before `## 1 · Core Idea`; move it to after `## 8 · What Can Go Wrong`
- [ ] `03-NeuralNetworks/ch10-transformers/README.md` — same issue: move `## 9 · Where This Reappears` to after section 8
- [x] ~~`03-NeuralNetworks/ch03-backprop-optimisers/README.md` — duplicate `## 9 · Where This Reappears` stub~~ removed; only one instance remains at line 786 (canonical position)

### 2.2 Duplicate Notation block in ch03-feature-importance
- [ ] `01-Regression/ch03-feature-importance/README.md` — standalone `## Notation` section between the opening blockquote and `## 0 · The Challenge` duplicates the blockquote's notation; remove the standalone section, keeping notation only in the opening blockquote

### 2.3 Broken cross-references (already fixed in Phase 4 — verify not re-introduced)
- [x] ~~`ch01-linear-regression/README.md:134`~~ fixed → `../../../MathUnderTheHood/ch05-matrices/`
- [x] ~~`ch02-multiple-regression/README.md:274,447`~~ fixed → `../../../MathUnderTheHood/...`

---

## Priority 3 — Low-Medium Impact (Content Completeness / Voice)

### 3.1 Missing numeric walkthroughs (still outstanding from Check 3)
- [ ] `03-NeuralNetworks/ch02-neural-networks/README.md` — §3 Math has no concrete forward-pass numeric example (3 rows through a 2-layer network showing actual activations); needs `### Numeric Forward-Pass Example`
- [ ] `03-NeuralNetworks/ch08-tensorboard/README.md` — §3 has incomplete gradient health table (headers only, no filled rows); fill in the 3-row diagnostic table

### 3.2 Dataset policy exceptions not documented
- [ ] `03-NeuralNetworks/ch05-cnns/README.md` — uses synthetic 8×8 pixel grid; add a note in `## 2 · Running Example` explaining why (CNNs require image data; synthetic grid is the minimal example)
- [ ] `03-NeuralNetworks/ch06-rnns-lstms/README.md` — uses synthetic monthly price index; add a note explaining the sequence domain requires temporal data, not housing
- [ ] `07-UnsupervisedLearning/README.md` — all 3 chapters use UCI Wholesale Customers; add one sentence in track README acknowledging the intentional dataset change for unsupervised track (clustering needs multi-dimensional non-housing data)

---

## Per-Track Remaining Work (from individual plan.md files)

See individual `plan.md` files under each track folder for granular todos. Summary:

| Track | Plan.md | Remaining todos |
|-------|---------|-----------------|
| 01-Regression | `01-Regression/plan.md` | 4 (mermaid emoji, notebook mirrors, small numeric cells, TensorBoard) |
| 02-Classification | `02-Classification/plan.md` | 4 (emoji cleanup, notebook mirrors, ch01 BCE demo, audit scripts) |
| 03-NeuralNetworks | `03-NeuralNetworks/plan.md` | 14 (see plan.md for details) |
| 04-RecommenderSystems | `04-RecommenderSystems/plan.md` | 6 (emoji ✅/❌, notebook mirroring, cross-links, CI checks) |
| 05-AnomalyDetection | `05-AnomalyDetection/plan.md` | 4 (emoji norm, precision@k numeric, cross-links, CI checks) |
| 06-ReinforcementLearning | `06-ReinforcementLearning/plan.md` | 6 (emoji ✅/❌, 3-state examples ch03–ch06, notebooks, cross-links) |
| 07-UnsupervisedLearning | `07-UnsupervisedLearning/plan.md` | 5 (emoji callouts, PCA numeric hand-computation, notebook mirror, CPU examples, CI checks) |
| 08-EnsembleMethods | `08-EnsembleMethods/plan.md` | 3 (notebook mirror, cross-links, CI checks) |

---

## Cross-Cutting (All Tracks)

- [ ] Add automated CI scripts: emoji audit, section checklist, numeric walkthrough detector, notebook mirror check (see `validation_report.md` for spec)
- [ ] Run all notebooks end-to-end on CPU to verify <5 min execution and deterministic outputs
- [ ] Update `validation_report.md` verdict to ✅ PASS after all Priority 1 and 2 items above are complete
- [ ] Tag `ml-track-editorial-v2.0` after full pass
