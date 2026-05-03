"""
fix_math_mermaid_arc_stubs.py
Append a 7-chapter Mermaid graph LR stub to each Progress Check section
in Math Under the Hood chapter README.md files.
"""
import re
import os
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATH_ROOT = os.path.join(REPO_ROOT, "notes", "math_under_the_hood")

CHAPTERS_INFO = [
    ("ch01_linear_algebra",       "Linear Algebra"),
    ("ch02_nonlinear_algebra",    "Nonlinear Algebra"),
    ("ch03_calculus_intro",       "Calculus Intro"),
    ("ch04_small_steps",          "Small Steps (GD)"),
    ("ch05_matrices",             "Matrices"),
    ("ch06_gradient_chain_rule",  "Gradient + Chain Rule"),
    ("ch07_probability_statistics", "Probability & Stats"),
]

MERMAID_ARC = """\n```mermaid
graph LR
    C1["Ch.1\\nLinear Algebra"]:::done
    C2["Ch.2\\nNonlinear Algebra"]:::done
    C3["Ch.3\\nCalculus Intro"]:::done
    C4["Ch.4\\nSmall Steps"]:::done
    C5["Ch.5\\nMatrices"]:::done
    C6["Ch.6\\nGradient + Chain Rule"]:::done
    C7["Ch.7\\nProbability & Stats"]:::done
    C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7
    classDef done fill:#15803d,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef current fill:#1d4ed8,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
    classDef upcoming fill:#1e3a8a,stroke:#e2e8f0,stroke-width:2px,color:#ffffff
```\n"""

# Match Progress Check headings of various numbering
PC_PATTERN = re.compile(r"(^## \d+ · Progress Check[^\n]*\n)", re.MULTILINE)

for chapter_dir, chapter_label in CHAPTERS_INFO:
    filepath = os.path.join(MATH_ROOT, chapter_dir, "README.md")
    if not os.path.exists(filepath):
        print(f"SKIP (not found): {filepath}")
        continue

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    if "```mermaid" in content and "graph LR" in content:
        print(f"Mermaid arc already present in {chapter_dir}/README.md")
        continue

    match = PC_PATTERN.search(content)
    if not match:
        print(f"No Progress Check heading found in {chapter_dir}/README.md")
        continue

    insert_pos = match.end()
    new_content = content[:insert_pos] + MERMAID_ARC + content[insert_pos:]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Inserted Mermaid arc stub in {chapter_dir}/README.md")

print("Done.")
