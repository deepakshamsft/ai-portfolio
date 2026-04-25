"""
fix_multi_agent_progress_check_heading.py
Normalize "## § 11.5 · Progress Check" -> "## 11 · Progress Check" in all 7
Multi-Agent AI chapter README.md files, for consistency with the universal standard.
"""
import re
import os
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MA_ROOT = os.path.join(REPO_ROOT, "notes", "multi_agent_ai")

CHAPTERS = [
    "message_formats", "mcp", "a2a", "event_driven_agents",
    "shared_memory", "trust_and_sandboxing", "agent_frameworks",
]

OLD_PATTERN = re.compile(r"^## § 11\.5 · Progress Check", re.MULTILINE)
NEW_HEADING = "## 11 · Progress Check"

for chapter in CHAPTERS:
    filepath = os.path.join(MA_ROOT, chapter, "README.md")
    if not os.path.exists(filepath):
        print(f"SKIP (not found): {filepath}")
        continue

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    new_content, count = OLD_PATTERN.subn(NEW_HEADING, content)
    if count:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {count} heading(s) in {chapter}/README.md")
    else:
        print(f"No match in {chapter}/README.md")
