#!/usr/bin/env bash
# Rewrite the absolute paths hardcoded in the scripts/notebooks for a new machine.
#
# The code was written on the original workstation, where
#   /home/nlapique/Blood/            = data root (h5ad files, scVI/MRVI models)
#   /home/nlapique/Blood/script/     = this repository (analysis code + output folders)
#   /home/nlapique/Downloads/BroadInst_DunedinPheno_20240829.txt = phenotype table
#
# Usage:  tools/set_paths.sh DATA_ROOT PHENO_TXT
#   e.g.  tools/set_paths.sh /data/bloodseq /data/pheno/BroadInst_DunedinPheno_20240829.txt
# Run from anywhere; edits files in place (commit or stash first so you can diff/revert).
set -euo pipefail
[[ $# -eq 2 ]] || { sed -n 2,13p "$0"; exit 1; }
DATA_ROOT=$(realpath -m "$1"); PHENO=$(realpath -m "$2")
REPO=$(cd "$(dirname "$0")/.." && pwd)
mapfile -d '' FILES < <(find "$REPO" \( -name '*.py' -o -name '*.ipynb' \) -not -path '*/.git/*' -not -path '*/repos/*' -print0)
sed -i \
  -e "s|/home/nlapique/Blood/script/|$REPO/|g" \
  -e "s|/home/nlapique/Downloads/BroadInst_DunedinPheno_20240829.txt|$PHENO|g" \
  -e "s|/home/nlapique/Blood/|$DATA_ROOT/|g" \
  "${FILES[@]}"
echo "Rewrote paths in ${#FILES[@]} files: repo=$REPO data=$DATA_ROOT pheno=$PHENO"
grep -rln '/home/nlapique' "${FILES[@]}" && echo "^ these still mention /home/nlapique (check by hand)" || true
