#!/usr/bin/env bash
set -euo pipefail

if [[ "$PWD" != "/home/renku/work/atmt_2026" ]]; then
    cd "/home/renku/work/atmt_2026/"
fi

PYTHON="${PYTHON:-python}"
if [[ -x output/.venv/bin/python ]]; then
    PYTHON="output/.venv/bin/python"
fi

SOURCE_LANG="${SOURCE_LANG:-cz}"
TARGET_LANG="${TARGET_LANG:-en}"
DATA_NAME="${DATA_NAME:-${SOURCE_LANG}-${TARGET_LANG}}"
WORK_DIR="${WORK_DIR:-output/${DATA_NAME}}"
RAW_DATA="${RAW_DATA:-data/${DATA_NAME}/raw}"
PREPARED_DATA="${PREPARED_DATA:-${WORK_DIR}/prepared}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${WORK_DIR}/tokenizers}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_DIR}/checkpoints}"
LOG_FILE="${LOG_FILE:-${WORK_DIR}/logs/train.log}"
OUTPUT_FILE="${OUTPUT_FILE:-${WORK_DIR}/translations.txt}"
SRC_VOCAB_SIZE="${SRC_VOCAB_SIZE:-8000}"
TGT_VOCAB_SIZE="${TGT_VOCAB_SIZE:-8000}"
RUN_OUTPUT="${RUN_OUTPUT:-assignment1.out}"

exec > >(tee "$RUN_OUTPUT") 2>&1
echo "Writing run output to $RUN_OUTPUT"

# PREPARE DATA
"$PYTHON" preprocess.py \
    --source-lang "$SOURCE_LANG" \
    --target-lang "$TARGET_LANG" \
    --raw-data "$RAW_DATA" \
    --dest-dir "$PREPARED_DATA" \
    --model-dir "$TOKENIZER_DIR" \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --src-vocab-size "$SRC_VOCAB_SIZE" \
    --tgt-vocab-size "$TGT_VOCAB_SIZE" \
    --src-model "${TOKENIZER_DIR}/${SOURCE_LANG}-bpe-${SRC_VOCAB_SIZE}.model" \
    --tgt-model "${TOKENIZER_DIR}/${TARGET_LANG}-bpe-${TGT_VOCAB_SIZE}.model"

# TRAIN
"$PYTHON" train.py \
    --cuda \
    --data "$PREPARED_DATA" \
    --src-tokenizer "${TOKENIZER_DIR}/${SOURCE_LANG}-bpe-${SRC_VOCAB_SIZE}.model" \
    --tgt-tokenizer "${TOKENIZER_DIR}/${TARGET_LANG}-bpe-${TGT_VOCAB_SIZE}.model" \
    --source-lang "$SOURCE_LANG" \
    --target-lang "$TARGET_LANG" \
    --batch-size 64 \
    --arch transformer \
    --max-epoch 7 \
    --log-file "$LOG_FILE" \
    --save-dir "$CHECKPOINT_DIR" \
    --ignore-checkpoints \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 300 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3 

# TRANSLATE
"$PYTHON" translate.py \
    --cuda \
    --data "$PREPARED_DATA" \
    --dicts "$TOKENIZER_DIR" \
    --checkpoint-path "${CHECKPOINT_DIR}/checkpoint_best.pt" \
    --output "$OUTPUT_FILE" \
    --max-len 300
