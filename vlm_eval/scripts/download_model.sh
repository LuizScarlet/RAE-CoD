#!/usr/bin/env bash
set -euo pipefail
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
REVISION="${MODEL_REVISION:-c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
MODEL_DIR="${1:?Usage: $0 OUTPUT_DIR}"
if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID" --revision "$REVISION" --local-dir "$MODEL_DIR" --max-workers 4
else
  huggingface-cli download "$MODEL_ID" --revision "$REVISION" --local-dir "$MODEL_DIR" --max-workers 4
fi
echo "Downloaded $MODEL_ID@$REVISION to $MODEL_DIR"
