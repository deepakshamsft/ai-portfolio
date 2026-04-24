"""
fix_ml_links.py
===============
Fix broken relative links under notes/ML/.

What this script fixes
----------------------
1. Cross-track depth errors: ``../../../07-Unsupervised…`` → ``../../07-Unsupervised…``
   (from a sub-chapter dir two ``../`` reach notes/ML/, not three)
2. Image files with spaces in filenames → rename to hyphenated names and update all
   Markdown links that reference them (handles both raw-space and %20-encoded forms).
3. AUTHORING_GUIDE.md deep path: ``../../../../MathUnderTheHood/`` → ``../MathUnderTheHood/``

What this script SKIPS (not auto-fixable)
-----------------------------------------
- notes/ML/validation_report.md lines 74, 124: ``../path`` / ``../../path`` are
  intentional placeholder example strings inside the report body — leave as-is.
- notes/ML/01-Regression/ch03-feature-importance/img/ missing PNGs
  (ch03-log-transform.png, ch03-pearson-vs-mi.png, ch03-lasso-path.png):
  these will be created when the gen scripts are executed.
- notes/ML/AUTHORING_GUIDE.md img/ placeholders (chNN-progress-check.png,
  loss_parabola_generated.png): intentional illustrative examples in the guide.

Usage
-----
    cd c:\\repos\\ai-portfolio
    python scripts/fix_ml_links.py [--dry-run]

    --dry-run   Print what would change without modifying any files.
"""
import argparse
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

# ── Repository root (resolved from this script's location) ──────────────────
REPO = Path(__file__).resolve().parent.parent
ML = REPO / "notes" / "ML"


# ── 1. Cross-track depth fixes ───────────────────────────────────────────────
# Any link from a sub-chapter dir (depth notes/ML/<track>/<chapter>/) that points
# three levels up to a sibling ML track is wrong — only two levels needed.
# Pattern: ../../../<ML-sibling-track>/
ML_TRACKS = [
    "01-Regression",
    "02-Classification",
    "03-NeuralNetworks",
    "04-RecommenderSystems",
    "05-AnomalyDetection",
    "06-ReinforcementLearning",
    "07-UnsupervisedLearning",
    "08-EnsembleMethods",
]
_DEPTH_PATTERN = re.compile(
    r"\.\./\.\./\.\./(" + "|".join(re.escape(t) for t in ML_TRACKS) + r")/"
)


def fix_depth_errors(text: str) -> tuple[str, int]:
    """Replace ../../../<ML-track>/ with ../../<ML-track>/ — returns (new_text, count)."""
    count = 0

    def _replace(m):
        nonlocal count
        count += 1
        return f"../../{m.group(1)}/"

    new_text = _DEPTH_PATTERN.sub(_replace, text)
    return new_text, count


# ── 2. Image rename map ───────────────────────────────────────────────────────
# Files that contain spaces in their names. Key = current name (may have spaces),
# value = new hyphenated name.
IMAGE_RENAMES: dict[Path, str] = {}


def _collect_image_renames() -> None:
    """Populate IMAGE_RENAMES with all img/*.png files that contain spaces."""
    for img_dir in ML.rglob("img"):
        if not img_dir.is_dir():
            continue
        for f in img_dir.iterdir():
            if " " in f.name:
                new_name = f.name.replace(" ", "-").lower()
                IMAGE_RENAMES[f] = new_name


def _link_matches_image(link_path: str, old_name: str) -> bool:
    """True if the link's filename (URL-decoded) matches old_name."""
    decoded = unquote(link_path.split("/")[-1])
    return decoded == old_name


def fix_image_links(text: str) -> tuple[str, int]:
    """Update Markdown image/link references to renamed files."""
    count = 0
    for old_path, new_name in IMAGE_RENAMES.items():
        old_name = old_path.name
        old_encoded = old_name.replace(" ", "%20")
        # Match both raw-space and %20-encoded forms
        for variant in [old_name, old_encoded]:
            escaped = re.escape(variant)
            pattern = re.compile(r"(\]\()([^)]*)" + escaped + r"(\))")
            if pattern.search(text):
                text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}{new_name}{m.group(3)}", text)
                count += 1
    return text, count


# ── 3. AUTHORING_GUIDE.md depth fix ─────────────────────────────────────────
# From notes/ML/AUTHORING_GUIDE.md the path to MathUnderTheHood is ../
_AG_BAD = "../../../../MathUnderTheHood/"
_AG_GOOD = "../MathUnderTheHood/"


# ── Main ─────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False) -> None:
    _collect_image_renames()

    total_file_edits = 0
    total_renames = 0
    total_link_fixes = 0

    # ── Step 1: rename image files ───────────────────────────────────────────
    for old_path, new_name in IMAGE_RENAMES.items():
        new_path = old_path.parent / new_name
        if new_path.exists():
            print(f"  SKIP rename (target exists): {old_path.name} → {new_name}")
            continue
        print(f"  RENAME: {old_path.relative_to(REPO)}  →  {new_name}")
        if not dry_run:
            shutil.move(str(old_path), str(new_path))
        total_renames += 1

    # ── Step 2: fix all .md files under notes/ML ────────────────────────────
    for md in sorted(ML.rglob("*.md")):
        try:
            original = md.read_text(encoding="utf-8")
        except Exception as e:
            print(f"  SKIP (read error): {md.relative_to(REPO)}: {e}")
            continue

        text = original

        # 2a. depth errors (only applies to sub-chapter READMEs, but safe globally)
        text, n_depth = fix_depth_errors(text)

        # 2b. image link updates
        text, n_img = fix_image_links(text)

        # 2c. AUTHORING_GUIDE specific fix
        n_ag = 0
        if md.name == "AUTHORING_GUIDE.md" and _AG_BAD in text:
            text = text.replace(_AG_BAD, _AG_GOOD)
            n_ag = 1

        total_changes = n_depth + n_img + n_ag
        if total_changes == 0:
            continue

        total_file_edits += 1
        total_link_fixes += total_changes
        rel = md.relative_to(REPO)
        changes = []
        if n_depth:
            changes.append(f"{n_depth} depth fix(es)")
        if n_img:
            changes.append(f"{n_img} image link(s)")
        if n_ag:
            changes.append(f"{n_ag} AG path fix")
        print(f"  EDIT ({', '.join(changes)}): {rel}")

        if not dry_run:
            md.write_text(text, encoding="utf-8")

    print()
    print(f"Summary: {total_renames} file rename(s), "
          f"{total_file_edits} md file(s) edited, "
          f"{total_link_fixes} link(s) fixed")

    if dry_run:
        print("(dry-run — no files were modified)")

    # ── Step 3: report unfixable items ──────────────────────────────────────
    print()
    print("Not auto-fixed (require manual action or script execution):")
    unfixable = [
        "notes/ML/01-Regression/ch03-feature-importance/img/ch03-log-transform.png "
        "— run gen_scripts/gen_ch03_log_transform.py",
        "notes/ML/01-Regression/ch03-feature-importance/img/ch03-pearson-vs-mi.png "
        "— run gen_scripts/gen_ch03_pearson_vs_mi.py",
        "notes/ML/01-Regression/ch03-feature-importance/img/ch03-lasso-path.png "
        "— run gen_scripts/gen_ch03_lasso_path.py",
        "notes/ML/08-EnsembleMethods/ch01-ensembles/img/ch11-svm-and-ensembles.png "
        "— original chapter image; regenerate or remove the link",
        "notes/ML/AUTHORING_GUIDE.md ./img/chNN-progress-check.png "
        "— intentional placeholder example in guide",
        "notes/ML/AUTHORING_GUIDE.md ./img/loss_parabola_generated.png "
        "— intentional placeholder example in guide",
        "notes/ML/validation_report.md lines 74,124: ../path and ../../path "
        "— intentional example strings in report body",
    ]
    for item in unfixable:
        print(f"  • {item}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
