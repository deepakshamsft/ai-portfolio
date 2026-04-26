"""
fix_ai_img_dirs.py
Create missing img/ directories in AI track chapters that reference images but have no img/ folder.
Chapters: prompt_engineering, cot_reasoning, rag_and_embeddings, vector_dbs, react_and_semantic_kernel
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_ROOT = os.path.join(REPO_ROOT, "notes", "ai")

CHAPTERS = [
    "prompt_engineering",
    "cot_reasoning",
    "rag_and_embeddings",
    "vector_dbs",
    "react_and_semantic_kernel",
]

for chapter in CHAPTERS:
    img_dir = os.path.join(AI_ROOT, chapter, "img")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created: {img_dir}")
    else:
        print(f"Already exists: {img_dir}")
