"""
create_multi_agent_img_dirs.py
Create img/ directory in each of the 7 Multi-Agent AI chapter folders.
"""
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

for chapter in CHAPTERS:
    img_dir = os.path.join(MA_ROOT, chapter, "img")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created: {img_dir}")
    else:
        print(f"Already exists: {img_dir}")
