"""
fix_ai_infra_notation_placeholders.py
Append <!-- TODO: add notation sentence here --> inside the opening blockquote
of each AI Infrastructure chapter .md file.

The opening blockquote is the first consecutive block of lines starting with ">".
The placeholder is appended as a new "> " line at the end of that blockquote.
"""
import re
import os
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_INFRA_ROOT = os.path.join(REPO_ROOT, "notes", "ai_infrastructure")

# Collect chapter-level .md files (not subdirectory READMEs but each chapter's main file)
MD_FILES = glob.glob(os.path.join(AI_INFRA_ROOT, "**", "*.md"), recursive=True)
# Filter to direct chapter files (one level down, exclude track-level files)
MD_FILES = [
    f for f in MD_FILES
    if os.path.dirname(f) != AI_INFRA_ROOT
]

PLACEHOLDER = "> <!-- TODO: add notation sentence here -->"

for filepath in MD_FILES:
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    if "TODO: add notation sentence here" in content:
        print(f"Already has placeholder: {os.path.relpath(filepath, REPO_ROOT)}")
        continue

    lines = content.splitlines(keepends=True)
    # Find the first blockquote block (consecutive lines starting with ">")
    blockquote_end = None
    in_blockquote = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(">"):
            in_blockquote = True
            blockquote_end = i
        elif in_blockquote:
            break  # End of first blockquote

    if blockquote_end is None:
        print(f"No blockquote found in {os.path.relpath(filepath, REPO_ROOT)}")
        continue

    # Insert placeholder after the last line of the opening blockquote
    insert_pos = blockquote_end + 1
    lines.insert(insert_pos, PLACEHOLDER + "\n")
    new_content = "".join(lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Added notation placeholder in {os.path.relpath(filepath, REPO_ROOT)}")

print("Done.")
