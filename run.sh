#!/usr/bin/env bash

# TRAIL training entrypoint: usefulness-aware rationale evaluation.
# Defaults and model arguments are retained from the original experiment script.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${TRAIL_DATA_DIR:-${SCRIPT_DIR}/data/FCTI_HAL}"
MODEL_PATH="${TRAIL_MODEL_PATH:-${SCRIPT_DIR}/model/roberta-base}"
DATA_NAME="cti-hal-conflict"

# Runtime defaults
GPU_ID=3
EPOCHS=30
BATCH_SIZE_TRAIN=16
LEARNING_RATE="1e-4"
EARLY_STOP=10
MAX_LEN=256
WEIGHT_DECAY="1e-4"

# TRAIL defaults
USEFULNESS_MARGIN="0.05"
LAMBDA_USE="0.20"
POOLING_METHOD="attention"
ENABLE_CONFLICT_FEATURES=true

usage() {
    cat <<'EOF'
Usage: bash run.sh [options]

Options:
  --gpu ID
  --epochs N
  --batch_size N
  --lr VALUE
  --early_stop N
  --max_len N
  --usefulness_margin VALUE
  --lambda_use VALUE
  --pooling_method mean|attention
  --enable_conflict_features
  --disable_conflict_features
  -h, --help

Environment:
  PYTHON_BIN       Python executable (default: python3)
  TRAIL_DATA_DIR   Processed dataset directory
  TRAIL_MODEL_PATH Local Hugging Face RoBERTa model directory
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Missing value for $1" >&2
        usage >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            require_value "$@"
            GPU_ID="$2"
            shift 2
            ;;
        --epochs)
            require_value "$@"
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            require_value "$@"
            BATCH_SIZE_TRAIN="$2"
            shift 2
            ;;
        --lr)
            require_value "$@"
            LEARNING_RATE="$2"
            shift 2
            ;;
        --early_stop)
            require_value "$@"
            EARLY_STOP="$2"
            shift 2
            ;;
        --max_len)
            require_value "$@"
            MAX_LEN="$2"
            shift 2
            ;;
        --usefulness_margin)
            require_value "$@"
            USEFULNESS_MARGIN="$2"
            shift 2
            ;;
        --lambda_use)
            require_value "$@"
            LAMBDA_USE="$2"
            shift 2
            ;;
        --pooling_method)
            require_value "$@"
            POOLING_METHOD="$2"
            shift 2
            ;;
        --enable_conflict_features)
            ENABLE_CONFLICT_FEATURES=true
            shift
            ;;
        --disable_conflict_features)
            ENABLE_CONFLICT_FEATURES=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$POOLING_METHOD" != "mean" && "$POOLING_METHOD" != "attention" ]]; then
    echo "--pooling_method must be 'mean' or 'attention'." >&2
    exit 2
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

TRAIN_FILE="${OUTPUT_DIR}/train.json"
VAL_FILE="${OUTPUT_DIR}/val.json"
TEST_FILE="${OUTPUT_DIR}/test.json"

missing=0
for file in "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE"; do
    if [[ ! -f "$file" ]]; then
        echo "Missing processed dataset file: $file" >&2
        missing=1
    fi
done
if [[ "$missing" -ne 0 ]]; then
    echo "See README.md for the required processed-data layout." >&2
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "RoBERTa model directory not found: $MODEL_PATH" >&2
    echo "See README.md for model preparation." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cat <<EOF
============================================================
TRAIL training
============================================================
Python:                    ${PYTHON_BIN}
Dataset directory:         ${OUTPUT_DIR}
Model directory:           ${MODEL_PATH}
GPU:                       ${GPU_ID}
Epochs:                    ${EPOCHS}
Batch size:                ${BATCH_SIZE_TRAIN}
Learning rate:             ${LEARNING_RATE}
Early stopping patience:   ${EARLY_STOP}
Maximum sequence length:   ${MAX_LEN}
Weight decay:              ${WEIGHT_DECAY}
Usefulness margin:         ${USEFULNESS_MARGIN}
Usefulness loss weight:    ${LAMBDA_USE}
Pooling:                   ${POOLING_METHOD}
Conflict features:         ${ENABLE_CONFLICT_FEATURES}
============================================================
EOF

"$PYTHON_BIN" main.py \
    --model_name TRAIL \
    --epoch "$EPOCHS" \
    --max_len "$MAX_LEN" \
    --early_stop "$EARLY_STOP" \
    --weight_decay "$WEIGHT_DECAY" \
    --language en \
    --root_path "$OUTPUT_DIR" \
    --bert_path "$MODEL_PATH" \
    --data_name "$DATA_NAME" \
    --data_type usefulness \
    --batchsize "$BATCH_SIZE_TRAIN" \
    --lr "$LEARNING_RATE" \
    --usefulness_margin "$USEFULNESS_MARGIN" \
    --lambda_use "$LAMBDA_USE" \
    --pooling_method "$POOLING_METHOD" \
    --enable_conflict_features "$ENABLE_CONFLICT_FEATURES"

echo
echo "Training completed."
echo "Best checkpoints: ${SCRIPT_DIR}/param_model/TRAIL_${DATA_NAME}/1/"
echo "Metrics:          ${SCRIPT_DIR}/logs/test/TRAIL_${DATA_NAME}/month_1.json"
