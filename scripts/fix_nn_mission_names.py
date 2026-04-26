"""
fix_nn_mission_names.py
Replace "Launch **SmartVal AI**" with "Launch **UnifiedAI**" in the ## 0 · The Challenge
blockquotes of:
  - ch01_xor_problem/README.md
  - ch07_mle_loss_functions/README.md
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NN_ROOT = os.path.join(REPO_ROOT, "notes", "ml", "03_neural_networks")

FILES = [
    os.path.join(NN_ROOT, "ch01_xor_problem", "README.md"),
    os.path.join(NN_ROOT, "ch07_mle_loss_functions", "README.md"),
]

OLD = "Launch **SmartVal AI**"
NEW = "Launch **UnifiedAI**"

for filepath in FILES:
    if not os.path.exists(filepath):
        print(f"SKIP (not found): {filepath}")
        continue

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    count = content.count(OLD)
    if count == 0:
        print(f"No matches in {os.path.relpath(filepath, REPO_ROOT)}")
        continue

    new_content = content.replace(OLD, NEW)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Replaced {count} occurrence(s) in {os.path.relpath(filepath, REPO_ROOT)}")

print("Done.")
