"""
fix_ai_cot_duplicate_heading.py
Remove the duplicate "## Interview Checklist" heading in
notes/ai/cot_reasoning/cot-reasoning.md.
Keeps the first occurrence; removes the second (which is a bare heading with no body).
"""
import re
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILEPATH = os.path.join(
    REPO_ROOT, "notes", "ai", "cot_reasoning", "cot-reasoning.md"
)

if not os.path.exists(FILEPATH):
    print(f"File not found: {FILEPATH}")
    raise SystemExit(1)

with open(FILEPATH, encoding="utf-8") as f:
    content = f.read()

HEADING = "## Interview Checklist"
first = content.find(HEADING)
second = content.find(HEADING, first + 1)

if second == -1:
    print("Only one ## Interview Checklist found — nothing to do.")
    raise SystemExit(0)

# Determine extent of the second occurrence: up to next ## or EOF
rest_after = content[second + len(HEADING):]
next_h = re.search(r"\n## ", rest_after)
if next_h:
    remove_end = second + len(HEADING) + next_h.start()
else:
    remove_end = len(content)

# Remove the second occurrence (including any trailing newlines before it)
pre = content[:second].rstrip("\n") + "\n"
post = content[remove_end:]
new_content = pre + post

with open(FILEPATH, "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"Removed duplicate ## Interview Checklist from {os.path.relpath(FILEPATH, REPO_ROOT)}")
