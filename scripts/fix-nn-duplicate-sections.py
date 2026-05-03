"""
fix_nn_duplicate_sections.py
Fix duplicate and misplaced ## 9 · Where This Reappears sections in Neural Networks chapters:

  1. ch03_backprop_optimisers: Remove the SECOND ## 9 · Where This Reappears stub
     (the one that appears after ## 10 · Bridge at the end of the file).

  2. ch06_rnns_lstms: Move ## 9 · Where This Reappears from the very top of the file
     to after ## 8 · What Can Go Wrong.

  3. ch10_transformers: Same as ch06 — move misplaced ## 9 · Where This Reappears
     to after ## 8 · What Can Go Wrong.
"""
import re
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NN_ROOT = os.path.join(REPO_ROOT, "notes", "ml", "03_neural_networks")

# ── Helper ────────────────────────────────────────────────────────────────────

def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def extract_section(content, heading_pattern):
    """Extract a section starting at heading_pattern up to (not including) the next ## heading."""
    m = re.search(heading_pattern, content, re.MULTILINE)
    if not m:
        return None, None, None
    start = m.start()
    rest = content[m.end():]
    next_h = re.search(r"^## ", rest, re.MULTILINE)
    end = m.end() + next_h.start() if next_h else len(content)
    return content[start:end], start, end

# ── 1. ch03: Remove duplicate trailing ## 9 · Where This Reappears ────────────

ch03 = os.path.join(NN_ROOT, "ch03_backprop_optimisers", "README.md")
if os.path.exists(ch03):
    content = read(ch03)
    pattern = re.compile(r"^## 9 · Where This Reappears", re.MULTILINE)
    matches = list(pattern.finditer(content))
    if len(matches) >= 2:
        # Remove from second occurrence onward
        second_start = matches[1].start()
        rest_after = content[matches[1].end():]
        next_h = re.search(r"^## ", rest_after, re.MULTILINE)
        second_end = matches[1].end() + (next_h.start() if next_h else len(rest_after))
        new_content = content[:second_start] + content[second_end:]
        write(ch03, new_content)
        print("ch03: Removed duplicate ## 9 · Where This Reappears")
    else:
        print(f"ch03: Found {len(matches)} occurrence(s) — nothing to remove")

# ── 2. ch06: Move ## 9 · Where This Reappears from top to after ## 8 ──────────

for chapter_dir in ["ch06_rnns_lstms", "ch10_transformers"]:
    filepath = os.path.join(NN_ROOT, chapter_dir, "README.md")
    if not os.path.exists(filepath):
        print(f"SKIP: {filepath}")
        continue

    content = read(filepath)
    section, sec_start, sec_end = extract_section(
        content, r"^## 9 · Where This Reappears"
    )
    if section is None:
        print(f"{chapter_dir}: No ## 9 · Where This Reappears found")
        continue

    # Check if it's already after ## 8
    eight_match = re.search(r"^## 8 · What Can Go Wrong", content, re.MULTILINE)
    if eight_match and sec_start > eight_match.start():
        print(f"{chapter_dir}: ## 9 already after ## 8 — nothing to do")
        continue

    # Remove section from current position
    content_without = content[:sec_start] + content[sec_end:]

    # Find ## 8 in the modified content and insert after it
    eight_match2 = re.search(r"^## 8 · What Can Go Wrong", content_without, re.MULTILINE)
    if eight_match2 is None:
        print(f"{chapter_dir}: ## 8 · What Can Go Wrong not found — appending at end")
        new_content = content_without.rstrip() + "\n\n" + section
    else:
        rest_of_8 = content_without[eight_match2.end():]
        next_h = re.search(r"^## ", rest_of_8, re.MULTILINE)
        if next_h:
            insert_at = eight_match2.end() + next_h.start()
        else:
            insert_at = len(content_without)
        new_content = content_without[:insert_at] + "\n" + section + content_without[insert_at:]

    write(filepath, new_content)
    print(f"{chapter_dir}: Moved ## 9 · Where This Reappears to after ## 8")

print("Done.")
