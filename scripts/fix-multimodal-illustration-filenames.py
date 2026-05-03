"""
fix_multimodal_illustration_filenames.py
Rename dangling ## Illustrations references from "img/Title With Spaces.png"
to "img/title-with-spaces.png" (kebab-case) in all 13 Multimodal AI chapter READMEs.
Also renames the actual files on disk if they exist.
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

# Match image references with spaces in filename: ![...](img/Foo Bar Baz.png)
IMG_REF_RE = re.compile(r"(!\[.*?\]\(img/)([^)]+\.[a-zA-Z]+)(\))")


def to_kebab(name: str) -> str:
    """Convert "Foo Bar Baz.png" -> "foo-bar-baz.png"."""
    base, ext = os.path.splitext(name)
    return base.lower().replace(" ", "-") + ext


for filepath in MD_FILES:
    chapter_dir = os.path.dirname(filepath)
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    def replacer(m):
        prefix = m.group(1)
        filename = m.group(2)
        suffix = m.group(3)
        if " " not in filename:
            return m.group(0)
        new_name = to_kebab(filename)
        # Rename file on disk if it exists
        old_path = os.path.join(chapter_dir, "img", filename)
        new_path = os.path.join(chapter_dir, "img", new_name)
        if os.path.exists(old_path) and not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"  Renamed file: {filename} -> {new_name}")
        return prefix + new_name + suffix

    new_content = IMG_REF_RE.sub(replacer, content)
    if new_content != content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated image refs in {os.path.relpath(filepath, REPO_ROOT)}")

print("Done.")
