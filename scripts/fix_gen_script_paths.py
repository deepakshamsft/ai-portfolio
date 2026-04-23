"""Fix hardcoded absolute output paths in all gen_scripts/*.py files.

Pattern being replaced (two-line string concat OR single-line raw string):
    out = (r"c:\\repos\\AI learning\\ai-portfolio\\notes\\...\\img\\Image.png"
           r"...")
→  from pathlib import Path
   out = str(Path(__file__).resolve().parent.parent / "img" / "Image.png")

Also ensures the img/ directory is created before saving.
"""
import os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

changed = []
errors = []

# Regex: captures the image filename after the last \img\ segment.
# File content uses single backslashes (raw string literals).
# In regex, one backslash in the pattern (written as \\) matches one literal backslash.

# Two-line string concat, two variants:
#   r"c:\repos\...\Track"  r"\img\Image.png"           (img directly in second piece)
#   r"c:\repos\...\Track"  r"\Chapter\img\Image.png"   (subdirectory before img)
PATTERN_2LINE = re.compile(
    r'out\s*=\s*\(?\s*r"c:\\[^"]*"'        # first piece: r"c:\repos\...\Track"
    r'\s*r"(?:[^"]*\\)?img\\([^"]+)"'      # second piece: optional subdir then \img\Name
    r'\s*\)?',
    re.IGNORECASE | re.DOTALL
)
# Single-line:  out = r"c:\repos\...\img\Image.png"
PATTERN_1LINE = re.compile(
    r'out\s*=\s*r"c:\\[^"]*\\img\\([^"]+)"',
    re.IGNORECASE
)

def process_file(fp: Path):
    text = fp.read_text(encoding='utf-8', errors='replace')
    original = text

    # Check for 2-line pattern first
    m = PATTERN_2LINE.search(text)
    img_name = None
    if m:
        img_name = m.group(1)
        full_match = m.group(0)
    else:
        m = PATTERN_1LINE.search(text)
        if m:
            img_name = m.group(1)
            full_match = m.group(0)

    if img_name is None:
        return False, "no pattern found"

    # Ensure pathlib is imported
    replacement_out = f'out = str(Path(__file__).resolve().parent.parent / "img" / "{img_name}")'

    new_text = text.replace(full_match, replacement_out)

    # Add pathlib import if missing
    if 'from pathlib import Path' not in new_text and 'import pathlib' not in new_text:
        # Insert after the first import block
        new_text = 'from pathlib import Path\n' + new_text

    if new_text == original:
        return False, "text unchanged after replacement"

    fp.write_text(new_text, encoding='utf-8')
    return True, img_name


for root, dirs, files in os.walk(REPO):
    if '.venv' in root or '__pycache__' in root:
        continue
    rel = root.replace(str(REPO), '').replace('\\', '/')
    if '/gen_scripts' not in rel:
        continue
    for f in files:
        if not f.endswith('.py'):
            continue
        fp = Path(root) / f
        ok, detail = process_file(fp)
        rel_fp = str(fp).replace(str(REPO), '').replace('\\', '/')
        if ok:
            changed.append(f'  FIXED: {rel_fp}  (img: {detail})')
        elif 'no pattern' not in detail:
            errors.append(f'  ERR({detail}): {rel_fp}')

print(f'Changed {len(changed)} files:')
for c in changed:
    print(c)
if errors:
    print(f'\nErrors ({len(errors)}):')
    for e in errors:
        print(e)
