"""
fix_ai_vector_inline_binary.py
Detect and remove accidentally inlined base64 binary blob in vector_dbs/vector-dbs.md.
Replaces the blob with a placeholder image reference.
"""
import re
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(REPO_ROOT, "notes", "ai", "vector_dbs", "vector-dbs.md")
PLACEHOLDER = "![Vector DBs diagram placeholder](img/vector-dbs-placeholder.png)"

with open(TARGET, encoding="utf-8") as f:
    content = f.read()

# Match a base64 blob: long contiguous string of base64 chars (no spaces, >200 chars)
pattern = re.compile(r"!\[.*?\]\(data:image/[^)]{100,}\)", re.DOTALL)
matches = pattern.findall(content)

if matches:
    new_content = pattern.sub(PLACEHOLDER, content)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Replaced {len(matches)} inline base64 blob(s) with placeholder in {TARGET}")
else:
    print(f"No inline base64 image data found in {TARGET}")
