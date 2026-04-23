"""Fix all broken cross-track Markdown links in one pass."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Each entry: (relative_filepath, [(old_link_target, new_link_target), ...])
FIXES = [
    # ── ch02-neural-networks: cross-track refs + renamed sibling ──────────────
    ("notes/ML/03-NeuralNetworks/ch02-neural-networks/README.md", [
        ("../ch03-xor-problem/",         "../ch01-xor-problem/"),
        ("../ch01-linear-regression/",   "../../01-Regression/ch01-linear-regression/"),
        ("../ch02-logistic-regression/", "../../02-Classification/ch01-logistic-regression/"),
    ]),

    # ── ch06-rnns-lstms: old flat metrics chapter ──────────────────────────────
    ("notes/ML/03-NeuralNetworks/ch06-rnns-lstms/README.md", [
        ("../ch09-metrics/", "../../02-Classification/ch03-metrics/"),
    ]),

    # ── ch07-mle-loss-functions: cross-track refs (same replacements appear 2-3×)
    ("notes/ML/03-NeuralNetworks/ch07-mle-loss-functions/README.md", [
        ("../ch01-linear-regression/",   "../../01-Regression/ch01-linear-regression/"),
        ("../ch02-logistic-regression/", "../../02-Classification/ch01-logistic-regression/"),
    ]),

    # ── ch08-tensorboard: old flat dimensionality-reduction chapter ───────────
    ("notes/ML/03-NeuralNetworks/ch08-tensorboard/README.md", [
        ("../ch13-dimensionality-reduction/",
         "../../07-UnsupervisedLearning/ch02-dimensionality-reduction/"),
    ]),

    # ── ch10-transformers: wrong depth for AI/MultiAgentAI cross-track refs ──
    ("notes/ML/03-NeuralNetworks/ch10-transformers/README.md", [
        ("../../AI/RAGAndEmbeddings/",        "../../../AI/RAGAndEmbeddings/"),
        ("../../AI/LLMFundamentals/",         "../../../AI/LLMFundamentals/"),
        ("../../AI/ReActAndSemanticKernel/",  "../../../AI/ReActAndSemanticKernel/"),
        ("../../MultiAgentAI/",               "../../../MultiAgentAI/"),
    ]),

    # ── ch01-ensembles: old flat chapter refs ────────────────────────────────
    ("notes/ML/08-EnsembleMethods/ch01-ensembles/README.md", [
        ("../ch10-classical-classifiers/", "../../02-Classification/ch02-classical-classifiers/"),
        ("../ch12-clustering/",            "../../07-UnsupervisedLearning/ch01-clustering/"),
    ]),

    # ── MathUnderTheHood/README.md: old flat ML chapter paths ────────────────
    ("notes/MathUnderTheHood/README.md", [
        ("../ML/ch01-linear-regression/README.md",
         "../ML/01-Regression/ch01-linear-regression/README.md"),
        ("../ML/ch05-backprop-optimisers/",
         "../ML/03-NeuralNetworks/ch03-backprop-optimisers/"),
    ]),

    # ── MultimodalAI/README.md: old flat ML chapter paths ────────────────────
    ("notes/MultimodalAI/README.md", [
        ("../ML/ch04-neural-networks/README.md",
         "../ML/03-NeuralNetworks/ch02-neural-networks/README.md"),
        ("../ML/ch05-backprop-optimisers/README.md",
         "../ML/03-NeuralNetworks/ch03-backprop-optimisers/README.md"),
        ("../ML/ch07-cnns/README.md",
         "../ML/03-NeuralNetworks/ch05-cnns/README.md"),
        ("../ML/ch17-sequences-to-attention/README.md",
         "../ML/03-NeuralNetworks/ch09-sequences-to-attention/README.md"),
        ("../ML/ch18-transformers/README.md",
         "../ML/03-NeuralNetworks/ch10-transformers/README.md"),
    ]),

    # ── VisionTransformers: old flat ML path ─────────────────────────────────
    ("notes/MultimodalAI/VisionTransformers/VisionTransformers.md", [
        ("../../ML/ch18-transformers/", "../../ML/03-NeuralNetworks/ch10-transformers/"),
    ]),

    # ── LLMFundamentals: old flat ML path ────────────────────────────────────
    ("notes/AI/LLMFundamentals/LLMFundamentals.md", [
        ("../../ML/ch18-transformers/", "../../ML/03-NeuralNetworks/ch10-transformers/"),
    ]),

    # ── EvaluatingAISystems: old flat ML metrics chapter ─────────────────────
    ("notes/AI/EvaluatingAISystems/EvaluatingAISystems.md", [
        ("../../ML/ch09-metrics/", "../../ML/02-Classification/ch03-metrics/"),
    ]),

    # ── MultiAgentAI: all use ../AI/ but should be ../../AI/ ─────────────────
    ("notes/MultiAgentAI/AgentFrameworks/README.md", [
        ("../AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md",
         "../../AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md"),
    ]),
    ("notes/MultiAgentAI/MCP/README.md", [
        ("../AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md",
         "../../AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md"),
    ]),
    ("notes/MultiAgentAI/MessageFormats/README.md", [
        ("../AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md",
         "../../AI/ReActAndSemanticKernel/ReActAndSemanticKernel.md"),
        ("../AI/PromptEngineering/PromptEngineering.md",
         "../../AI/PromptEngineering/PromptEngineering.md"),
    ]),
    ("notes/MultiAgentAI/TrustAndSandboxing/README.md", [
        ("../AI/SafetyAndHallucination/SafetyAndHallucination.md",
         "../../AI/SafetyAndHallucination/SafetyAndHallucination.md"),
    ]),

    # ── projects/rag-pipeline: missing subfolder in AI note paths ────────────
    ("projects/ai/rag-pipeline/README.md", [
        ("../../notes/AI/RAGAndEmbeddings.md",
         "../../notes/AI/RAGAndEmbeddings/RAGAndEmbeddings.md"),
        ("../../notes/AI/VectorDBs.md",
         "../../notes/AI/VectorDBs/VectorDBs.md"),
        ("../../notes/AI/EvaluatingAISystems.md",
         "../../notes/AI/EvaluatingAISystems/EvaluatingAISystems.md"),
        ("../../notes/AI/CostAndLatency.md",
         "../../notes/AI/CostAndLatency/CostAndLatency.md"),
    ]),
]

total_changes = 0
for rel_path, replacements in FIXES:
    fp = REPO / rel_path
    try:
        text = fp.read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f"SKIP (not found): {rel_path}")
        continue

    original = text
    file_changes = 0
    for old, new in replacements:
        count = text.count(old)
        if count:
            text = text.replace(old, new)
            file_changes += count
            print(f"  [{count}x] {old!r:60s} → {new!r}")

    if text != original:
        fp.write_text(text, encoding='utf-8')
        total_changes += file_changes
        print(f"SAVED ({file_changes} change(s)): {rel_path}")
    else:
        print(f"NO_MATCH: {rel_path}")

print(f"\nTotal link targets changed: {total_changes}")
