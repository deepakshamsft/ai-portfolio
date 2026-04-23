# Recommender Systems Track

> **The Mission**: Build **FlixAI** — a production-grade movie recommendation engine that achieves >85% hit rate @ top-10 while handling cold start, scaling to millions of ratings, maintaining diversity, and providing explainable recommendations.

This is not a Kaggle leaderboard chase. Every chapter builds toward a single production challenge: you're the Lead ML Engineer at a streaming platform, and the VP of Product demands a system that keeps users engaged, surfaces hidden gems, and can explain *why* it recommends each title.

---

## The Grand Challenge: 5 Core Constraints

| # | Constraint | Target | Why It Matters |
|---|------------|--------|----------------|
| **#1** | **ACCURACY** | >85% hit rate @ top-10 | Users who don't click within 10 suggestions churn within 30 days. Every percentage point = $2M ARR |
| **#2** | **COLD START** | Handle new users/items gracefully | 15% of monthly traffic is new signups with zero watch history. New releases have no ratings on day one |
| **#3** | **SCALABILITY** | 1M+ ratings, <200ms latency | Production traffic is 10× the dev dataset. Batch retraining + real-time serving required |
| **#4** | **DIVERSITY** | Not just popular movies | Recommending "The Shawshank Redemption" to everyone is 60% accurate but useless. Long-tail discovery drives retention |
| **#5** | **EXPLAINABILITY** | "Because you liked X" | Users trust recommendations they understand. Black-box suggestions get ignored 40% more often |

---

## Progressive Capability Unlock

| Ch | Title | What Unlocks | Hit Rate@10 | Constraints | Status |
|----|-------|--------------|-------------|-------------|--------|
| **1** | [Fundamentals](ch01-fundamentals/) | Popularity baseline, evaluation metrics | 42% | #1 Partial | 🔧 |
| **2** | [Collaborative Filtering](ch02-collaborative-filtering/) | User-based & item-based CF | 68% | #1 Improved | 🔧 |
| **3** | [Matrix Factorization](ch03-matrix-factorization/) | SVD, ALS, latent factors | 78% | #1 Close! | 🔧 |
| **4** | [Neural Collaborative Filtering](ch04-neural-cf/) | Deep learning embeddings, NCF | 83% | #1 Almost | 🔧 |
| **5** | [Hybrid Systems](ch05-hybrid-systems/) | Content + collaborative fusion | **87%** | **#1 ✅ #4 ✅** | 🔧 |
| **6** | [Cold Start & Production](ch06-cold-start-production/) | Bandits, A/B testing, deployment | 87% | **All ✅** | 🔧 |

---

## Narrative Arc: From 42% Popularity Baseline to 87% Hybrid System

### 🎬 Act 1: Foundations (Ch.1–2)
**Build simple baselines, understand their limits**

- **Ch.1**: Can we just recommend popular movies? → Yes, but only 42% hit rate
  - *"Everyone gets the same 10 movies? That's not personalisation, that's a billboard." — VP Product*

- **Ch.2**: Use similar users' ratings? → Better (68%), but sparse data limits us
  - *"We're personalising now, but the system can't recommend anything it hasn't seen a similar user rate." — Head of Data*

**Status**: ❌ Accuracy target unmet. Need latent representations.

---

### ⚡ Act 2: Latent Factors & Deep Learning (Ch.3–4)
**Discover hidden taste dimensions, learn non-linear interactions**

- **Ch.3**: Matrix factorization discovers latent factors → 78% hit rate
  - *"Now we're capturing something deeper — users who like 'cerebral sci-fi' even if they never rated the same movies." — Lead Data Scientist*

- **Ch.4**: Neural collaborative filtering learns non-linear patterns → 83% hit rate
  - *"The neural model is catching taste interactions that linear factorization misses. We're 2 points away!" — VP Engineering*

**Status**: ❌ Still below 85%. Need content features to close the gap.

---

### 🏆 Act 3: Hybrid & Production (Ch.5–6)
**Fuse content + collaborative signals, deploy to production**

- **Ch.5**: Hybrid content + collaborative → **87% hit rate ✅ Target achieved!**
  - *"Combining genre/director metadata with collaborative signals did it. And we're surfacing niche films now!" — VP Product*

- **Ch.6**: Cold start handling + production deployment → Production-ready system
  - *"New users get decent recommendations from day one, and we can A/B test improvements safely." — CTO*

**Status**: ✅✅✅✅✅ **ALL CONSTRAINTS SATISFIED!**

---

## The Dataset: MovieLens 100k

Every chapter uses the same dataset: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/) from the GroupLens research lab.

```python
# 100,000 ratings from 943 users on 1,682 movies
# Ratings: 1–5 stars (explicit feedback)
# Metadata: movie genres (19 binary columns), release year
# Demographics: user age, gender, occupation

# Load via surprise library or direct download
from surprise import Dataset
data = Dataset.load_builtin('ml-100k')
```

**Why MovieLens?**
- ✅ **Industry benchmark**: The standard academic dataset for recommender systems research
- ✅ **Rich metadata**: 19 genre flags, timestamps, user demographics
- ✅ **Manageable size**: 100k ratings (fits in memory, trains in seconds)
- ✅ **Real sparsity**: 93.7% of the user-item matrix is empty — mirrors production
- ✅ **Cold start examples**: Can simulate new users/items by holding out data

**Key statistics:**

| Metric | Value |
|--------|-------|
| Users | 943 |
| Movies | 1,682 |
| Ratings | 100,000 |
| Rating scale | 1–5 stars |
| Sparsity | 93.7% |
| Avg ratings/user | 106 |
| Avg ratings/movie | 59 |

---

## How to Use This Track

1. **Read the README** for each chapter — understand the theory and math
2. **Work through the notebook** — run every cell, inspect every output
3. **Complete the exercises** — they target specific concepts from the chapter
4. **Check the Progress Table** — each chapter's §10 shows which constraints are met
5. **Follow the bridge** — each chapter's §11 explains what the next chapter solves

> 💡 **Tip**: The hit rate numbers (42% → 68% → 78% → 83% → 87%) are approximate targets, not exact values. Your results will vary slightly based on random seeds and train/test splits. The progression pattern matters more than the exact numbers.
