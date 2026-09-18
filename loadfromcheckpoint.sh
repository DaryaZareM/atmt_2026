#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
if [[ -x output/.venv/bin/python ]]; then
    PYTHON="output/.venv/bin/python"
fi

SOURCE_LANG="${SOURCE_LANG:-en}"
TARGET_LANG="${TARGET_LANG:-sv}"
OUTPUT_DIR="${OUTPUT_DIR:-output/toy_example}"
RAW_DATA="${RAW_DATA:-data/toy_example/en-sv/infopankki/raw}"
PREPARED_DIR="${PREPARED_DIR:-${OUTPUT_DIR}/prepared}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${OUTPUT_DIR}/tokenizers}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${OUTPUT_DIR}/logs}"
TEST_PREFIX="${TEST_PREFIX:-test}"
SRC_VOCAB_SIZE="${SRC_VOCAB_SIZE:-1000}"
TGT_VOCAB_SIZE="${TGT_VOCAB_SIZE:-1000}"

"$PYTHON" train.py \
    --data "$PREPARED_DIR" \
    --src-tokenizer "${TOKENIZER_DIR}/${SOURCE_LANG}-bpe-${SRC_VOCAB_SIZE}.model" \
    --tgt-tokenizer "${TOKENIZER_DIR}/${TARGET_LANG}-bpe-${TGT_VOCAB_SIZE}.model" \
    --source-lang "$SOURCE_LANG" \
    --target-lang "$TARGET_LANG" \
    --batch-size 32 \
    --arch transformer \
    --max-epoch 3 \
    --log-file "${LOG_DIR}/train.log" \
    --save-dir "$CHECKPOINT_DIR" \
    --restore-file checkpoint_last.pt \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 100 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3

# "$PYTHON" translate.py \
#     --input "${RAW_DATA}/${TEST_PREFIX}.${SOURCE_LANG}" \
#     --src-tokenizer "${TOKENIZER_DIR}/${SOURCE_LANG}-bpe-${SRC_VOCAB_SIZE}.model" \
#     --tgt-tokenizer "${TOKENIZER_DIR}/${TARGET_LANG}-bpe-${TGT_VOCAB_SIZE}.model" \
#     --checkpoint-path "${CHECKPOINT_DIR}/checkpoint_best.pt" \
#     --batch-size 1 \
#     --max-len 100 \
#     --output "${OUTPUT_DIR}/toy_example_output.${TARGET_LANG}" \
#     --bleu \
#     --reference "${RAW_DATA}/${TEST_PREFIX}.${TARGET_LANG}"
