"""
fix_multi_agent_section_stubs.py
For all 7 Multi-Agent AI chapter README.md files, perform idempotent fixes:

  1. Insert ## Animation stub after ## 0 · The Challenge
  2. Rename ## Core Concepts         -> ## 1 · Core Idea
  3. Rename ## OrderFlow — Ch.N Scenario -> ## 2 · Running Example
     (handles any variant like "OrderFlow — Ch.1 Scenario" etc.)
  4. Insert stub ## 4 · How It Works (if missing) before Progress Check
  5. Insert stub ## 5 · Key Diagrams (if missing) before Progress Check
  6. Insert stub ## 6 · Hyperparameter Dial (if missing) before Progress Check
  7. Insert stub ## 8 · What Can Go Wrong (if missing) before Progress Check
"""
import re
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MA_ROOT = os.path.join(REPO_ROOT, "notes", "multi_agent_ai")

CHAPTERS = [
    "message_formats",
    "mcp",
    "a2a",
    "event_driven_agents",
    "shared_memory",
    "trust_and_sandboxing",
    "agent_frameworks",
]

ANIMATION_STUB = "\n## Animation\n\n> 🎬 *Animation placeholder — needle-builder agent will generate this.*\n\n"
HOW_IT_WORKS_STUB = "\n## 4 · How It Works\n\n> Step-by-step walkthrough of the mechanism.\n\n"
KEY_DIAGRAMS_STUB = "\n## 5 · Key Diagrams\n\n> Add 2–3 diagrams showing the key data flows here.\n\n"
HYPER_DIAL_STUB = "\n## 6 · Hyperparameter Dial\n\n> List the key knobs and their effect on behaviour.\n\n"
WHAT_WRONG_STUB = "\n## 8 · What Can Go Wrong\n\n> 3–5 common failure modes and mitigations.\n\n"

CHALLENGE_HEADING = re.compile(r"(^## 0 · The Challenge[^\n]*\n)", re.MULTILINE)
PC_PATTERN = re.compile(r"^## (?:§ )?11(?:\.5)? · Progress Check", re.MULTILINE)

CORE_CONCEPTS_RE = re.compile(r"^## Core Concepts\s*$", re.MULTILINE)
ORDERFLOW_RE = re.compile(r"^## OrderFlow — Ch\.\d+ Scenario[^\n]*", re.MULTILINE)


def insert_after(content, anchor_re, stub):
    m = anchor_re.search(content)
    if not m:
        return content, False
    pos = m.end()
    return content[:pos] + stub + content[pos:], True


def insert_before_pc(content, header_text, stub):
    if header_text in content:
        return content, False
    m = PC_PATTERN.search(content)
    if not m:
        return content.rstrip() + "\n\n" + stub, True
    return content[:m.start()] + stub + content[m.start():], True


for chapter_dir in CHAPTERS:
    filepath = os.path.join(MA_ROOT, chapter_dir, "README.md")
    if not os.path.exists(filepath):
        print(f"SKIP (not found): {filepath}")
        continue

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    changed = False

    # 1. Animation stub
    if "## Animation" not in content:
        content, c = insert_after(content, CHALLENGE_HEADING, ANIMATION_STUB)
        changed |= c

    # 2. Core Concepts -> 1 · Core Idea
    new_content, n = CORE_CONCEPTS_RE.subn("## 1 · Core Idea", content)
    if n:
        content = new_content
        changed = True

    # 3. OrderFlow scenario -> 2 · Running Example
    new_content, n = ORDERFLOW_RE.subn("## 2 · Running Example", content)
    if n:
        content = new_content
        changed = True

    # 4–7. Missing stubs
    content, c = insert_before_pc(content, "## 4 · How It Works", HOW_IT_WORKS_STUB)
    changed |= c
    content, c = insert_before_pc(content, "## 5 · Key Diagrams", KEY_DIAGRAMS_STUB)
    changed |= c
    content, c = insert_before_pc(content, "## 6 · Hyperparameter Dial", HYPER_DIAL_STUB)
    changed |= c
    content, c = insert_before_pc(content, "## 8 · What Can Go Wrong", WHAT_WRONG_STUB)
    changed |= c

    if changed:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Updated {chapter_dir}/README.md")
    else:
        print(f"No changes needed for {chapter_dir}/README.md")

print("Done.")
