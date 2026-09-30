"""Copy a notebook with all outputs, execution counts and widget state removed.

Outputs of these notebooks can contain donor-level clinical values, so only
stripped notebooks may be committed.  Usage:
    python tools/strip_notebook.py SRC.ipynb DST.ipynb
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
nb = json.load(open(src, encoding="utf-8"))
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cell.get("metadata", {}).pop("execution", None)
    cell.get("attachments") and cell.pop("attachments")
nb.get("metadata", {}).pop("widgets", None)
with open(dst, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
