#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <dataset-config-name>" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="$1"
CONFIG="$ROOT/configs/${DATASET}.yaml"

if [[ ! -f "$CONFIG" ]]; then
    echo "Unknown dataset configuration: $DATASET" >&2
    exit 2
fi

: "${AREA_DATA_ROOT:?Set AREA_DATA_ROOT to the directory containing datasets/.}"
: "${MODEL:?Set MODEL to a supported Qwen2.5-VL checkpoint or local checkpoint path.}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
PART="${PART:-0}"
MODEL_BASENAME="$(basename "$MODEL")"
PREDICTIONS="$OUTPUT_ROOT/$DATASET/$MODEL_BASENAME/split_${PART}.json"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python "$ROOT/run.py" \
    --config "$CONFIG" \
    --model_name "$MODEL" \
    --output_root "$OUTPUT_ROOT"

if [[ "${EVALUATE:-1}" == "0" ]]; then
    exit 0
fi

if [[ "$DATASET" == "amber" ]]; then
    : "${AMBER_ANNOTATION:?Set AMBER_ANNOTATION for AMBER-D evaluation.}"
    python "$ROOT/evaluate.py" \
        --predictions "$PREDICTIONS" \
        --dataset "$DATASET" \
        --amber_annotation "$AMBER_ANNOTATION"
else
    python "$ROOT/evaluate.py" \
        --predictions "$PREDICTIONS" \
        --dataset "$DATASET"
fi
