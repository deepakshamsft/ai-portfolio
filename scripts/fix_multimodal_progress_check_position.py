"""
fix_multimodal_progress_check_position.py
Move the "## 8.5 · Progress Check" block to after a "## 11 · Notebook" section
and renumber it to "## 11.5 · Progress Check" in all 13 Multimodal AI chapter READMEs.

Steps per file:
  1. Extract the entire ## 8.5 · Progress Check block (up to next ## heading or EOF).
  2. Remove it from its current position.
  3. Insert a "## 11 · Notebook" stub if not present.
  4. Append the renamed block after ## 11 · Notebook.
"""
import re
import os
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MM_ROOT = os.path.join(REPO_ROOT, "notes", "multimodal_ai")

MD_FILES = [
    f for f in glob.glob(os.path.join(MM_ROOT, "**", "README.md"), recursive=True)
    if os.path.dirname(f) != MM_ROOT
]

PC_PATTERN = re.compile(
    r"(^## 8\.5 · Progress Check.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL
)

NOTEBOOK_STUB = "\n## 11 · Notebook\n\n> See `notebook.ipynb` in this chapter's folder for interactive exercises.\n\n"

for filepath in MD_FILES:
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Already moved?
    if "## 11.5 · Progress Check" in content:
        print(f"Already updated: {os.path.relpath(filepath, REPO_ROOT)}")
        continue

    match = PC_PATTERN.search(content)
    if not match:
        print(f"No ## 8.5 · Progress Check found in {os.path.relpath(filepath, REPO_ROOT)}")
        continue

    block = match.group(0)
    renamed_block = block.replace("## 8.5 · Progress Check", "## 11.5 · Progress Check", 1)

    # Remove original block
    content_without = content[:match.start()] + content[match.end():]

    # Add ## 11 · Notebook stub + renamed progress check at end
    if "## 11 · Notebook" not in content_without:
        new_content = content_without.rstrip() + NOTEBOOK_STUB + renamed_block
    else:
        new_content = content_without.rstrip() + "\n\n" + renamed_block

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Moved Progress Check in {os.path.relpath(filepath, REPO_ROOT)}")

print("Done.")
