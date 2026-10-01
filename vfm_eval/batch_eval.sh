#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 GT_DIR OUTPUT_DIR RECON_DIR [RECON_DIR ...]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GT_DIR="$1"; OUTPUT_DIR="$2"; shift 2
mkdir -p "$OUTPUT_DIR"
for RECON_DIR in "$@"; do
  NAME="$(basename "${RECON_DIR%/}")"
  python "$ROOT/compute_semantic_distance.py" \
    --gt-dir "$GT_DIR" --recon-dir "$RECON_DIR" \
    --output "$OUTPUT_DIR/$NAME.json"
done
