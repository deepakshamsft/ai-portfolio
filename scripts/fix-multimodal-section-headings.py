"""
fix_multimodal_section_headings.py
Rename §4–§8 headings across all 13 Multimodal AI chapter .md files to match
the track-authoritative names from the track authoring guide.

Renames:
  ## 4 · How It Works — Step by Step  ->  ## 4 · Visual Intuition
  ## 5 · The Key Diagrams             ->  ## 5 · Production Example — VisualForge in Action
  ## 6 · What Changes at Scale        ->  ## 6 · Common Failure Modes
  ## 7 · Common Misconceptions        ->  ## 7 · When to Use This vs Alternatives
  ## 8 · Interview Checklist          ->  ## 8 · Connection to Prior Chapters
"""
import re
import os
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MM_ROOT = os.path.join(REPO_ROOT, "notes", "multimodal_ai")

# Collect all chapter README.md files (exclude track-level README)
MD_FILES = [
    f for f in glob.glob(os.path.join(MM_ROOT, "**", "README.md"), recursive=True)
    if os.path.dirname(f) != MM_ROOT
]

RENAMES = [
    (re.compile(r"^(## 4 · How It Works)[^\n]*", re.MULTILINE),
     "## 4 · Visual Intuition"),
    (re.compile(r"^## 5 · The Key Diagrams\s*$", re.MULTILINE),
     "## 5 · Production Example — VisualForge in Action"),
    (re.compile(r"^## 6 · What Changes at Scale\s*$", re.MULTILINE),
     "## 6 · Common Failure Modes"),
    (re.compile(r"^## 7 · Common Misconceptions\s*$", re.MULTILINE),
     "## 7 · When to Use This vs Alternatives"),
    (re.compile(r"^## 8 · Interview Checklist\s*$", re.MULTILINE),
     "## 8 · Connection to Prior Chapters"),
]

for filepath in MD_FILES:
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    new_content = content
    changed = False
    for pattern, replacement in RENAMES:
        result, count = pattern.subn(replacement, new_content)
        if count:
            new_content = result
            changed = True
            rel = os.path.relpath(filepath, REPO_ROOT)
            print(f"  {rel}: '{pattern.pattern}' -> '{replacement}'")

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)

print("Done.")
