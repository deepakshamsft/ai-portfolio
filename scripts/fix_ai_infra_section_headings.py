"""
fix_ai_infra_section_headings.py
For all 5 AI Infrastructure chapter .md files:
  1. Rename "## 2 · The InferenceBase Angle" -> "## 2 · Running Example"
  2. Renumber "## 11 · What Can Go Wrong" -> "## 8 · What Can Go Wrong" with downstream renumbering
"""
import re
import os
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_INFRA_ROOT = os.path.join(REPO_ROOT, "notes", "ai_infrastructure")

MD_FILES = glob.glob(os.path.join(AI_INFRA_ROOT, "**", "*.md"), recursive=True)
# Exclude README at track level
MD_FILES = [f for f in MD_FILES if os.path.basename(f) != "README.md" or
            os.path.dirname(f) != AI_INFRA_ROOT]

HEADING_RENAMES = [
    (r"^## 2 · The InferenceBase Angle", "## 2 · Running Example"),
    (r"^## 11 · What Can Go Wrong", "## 8 · What Can Go Wrong"),
]

for filepath in MD_FILES:
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    changed = False
    new_lines = []
    for line in lines:
        new_line = line
        for pattern, replacement in HEADING_RENAMES:
            if re.match(pattern, line.rstrip()):
                new_line = replacement + "\n"
                changed = True
                print(f"  Renamed in {os.path.relpath(filepath, REPO_ROOT)}: '{line.rstrip()}' -> '{replacement}'")
                break
        new_lines.append(new_line)

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

print("Done.")
