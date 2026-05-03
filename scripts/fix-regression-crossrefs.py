"""
fix_regression_crossrefs.py
Fix broken cross-references in ML / 01 Regression chapters.
Replaces all "../../../../MathUnderTheHood/" with "../../../math_under_the_hood/"
in ch01_linear_regression/README.md and ch02_multiple_regression/README.md.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGRESSION_ROOT = os.path.join(REPO_ROOT, "notes", "ml", "01_regression")

FILES = [
    os.path.join(REGRESSION_ROOT, "ch01_linear_regression", "README.md"),
    os.path.join(REGRESSION_ROOT, "ch02_multiple_regression", "README.md"),
]

OLD = "../../../../MathUnderTheHood/"
NEW = "../../../math_under_the_hood/"

for filepath in FILES:
    if not os.path.exists(filepath):
        print(f"SKIP (not found): {filepath}")
        continue

    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    count = content.count(OLD)
    if count == 0:
        print(f"No matches in {os.path.relpath(filepath, REPO_ROOT)}")
        continue

    new_content = content.replace(OLD, NEW)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"Replaced {count} occurrence(s) in {os.path.relpath(filepath, REPO_ROOT)}")

print("Done.")
