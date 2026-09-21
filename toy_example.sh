#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1


PYTHON="${PYTHON:-python}"


if [[ "$PWD" != "/home/renku/work/atmt_2026" ]]; then
    cd "/home/renku/work/atmt_2026/"
fi



SOURCE_LANG="${SOURCE_LANG:-en}"
TARGET_LANG="${TARGET_LANG:-sv}"
OUTPUT_DIR="${OUTPUT_DIR:-output/toy_example}"
TOY_DATA_ROOT="${TOY_DATA_ROOT:-data/toy_example}"
TOY_DATASET="${TOY_DATASET:-en-sv/infopankki/raw}"
RAW_DATA="${RAW_DATA:-${TOY_DATA_ROOT}/${TOY_DATASET}}"
TRAIN_PREFIX="${TRAIN_PREFIX:-tiny_train}"
VALID_PREFIX="${VALID_PREFIX:-valid}"
TEST_PREFIX="${TEST_PREFIX:-test}"
SRC_VOCAB_SIZE="${SRC_VOCAB_SIZE:-1000}"
TGT_VOCAB_SIZE="${TGT_VOCAB_SIZE:-1000}"
RUN_OUTPUT="${RUN_OUTPUT:-${OUTPUT_DIR}/toy_example.out}"

mkdir -p "$OUTPUT_DIR"
exec > >(tee "$RUN_OUTPUT") 2>&1
echo "Writing run output to $RUN_OUTPUT"

required_files=(
    "${RAW_DATA}/${TRAIN_PREFIX}.${SOURCE_LANG}"
    "${RAW_DATA}/${TRAIN_PREFIX}.${TARGET_LANG}"
    "${RAW_DATA}/${VALID_PREFIX}.${SOURCE_LANG}"
    "${RAW_DATA}/${VALID_PREFIX}.${TARGET_LANG}"
    "${RAW_DATA}/${TEST_PREFIX}.${SOURCE_LANG}"
    "${RAW_DATA}/${TEST_PREFIX}.${TARGET_LANG}"
)
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Missing expected data file: $file" >&2
        echo "Set RAW_DATA=/path/to/raw if your data is somewhere else." >&2
        exit 1
    fi
done

PREPARED_DIR="${OUTPUT_DIR}/prepared"
TOKENIZER_DIR="${OUTPUT_DIR}/tokenizers"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
LOG_DIR="${OUTPUT_DIR}/logs"
TRANSLATION_OUTPUT="${OUTPUT_DIR}/toy_example_output.${TARGET_LANG}"

# clean up from previous runs
rm -rf "$PREPARED_DIR" "$TOKENIZER_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"
rm -f "$TRANSLATION_OUTPUT"

"$PYTHON" preprocess.py \
    --source-lang "$SOURCE_LANG" \
    --target-lang "$TARGET_LANG" \
    --raw-data "$RAW_DATA" \
    --dest-dir "$PREPARED_DIR" \
    --model-dir "$TOKENIZER_DIR" \
    --test-prefix "$TEST_PREFIX" \
    --train-prefix "$TRAIN_PREFIX" \
    --valid-prefix "$VALID_PREFIX" \
    --src-vocab-size "$SRC_VOCAB_SIZE" \
    --tgt-vocab-size "$TGT_VOCAB_SIZE" \
    --ignore-existing \
    --force-train

"$PYTHON" train.py \
    --data "$PREPARED_DIR" \
    --src-tokenizer "${TOKENIZER_DIR}/${SOURCE_LANG}-bpe-${SRC_VOCAB_SIZE}.model" \
    --tgt-tokenizer "${TOKENIZER_DIR}/${TARGET_LANG}-bpe-${TGT_VOCAB_SIZE}.model" \
    --source-lang "$SOURCE_LANG" \
    --target-lang "$TARGET_LANG" \
    --batch-size 32 \
    --arch transformer \
    --max-epoch 10 \
    --log-file "${LOG_DIR}/train.log" \
    --save-dir "$CHECKPOINT_DIR" \
    --ignore-checkpoints \
    --encoder-dropout 0.1 \
    --decoder-dropout 0.1 \
    --dim-embedding 256 \
    --attention-heads 4 \
    --dim-feedforward-encoder 1024 \
    --dim-feedforward-decoder 1024 \
    --max-seq-len 100 \
    --n-encoder-layers 3 \
    --n-decoder-layers 3

"$PYTHON" translate.py \
    --input "${RAW_DATA}/${TEST_PREFIX}.${SOURCE_LANG}" \
    --src-tokenizer "${TOKENIZER_DIR}/${SOURCE_LANG}-bpe-${SRC_VOCAB_SIZE}.model" \
    --tgt-tokenizer "${TOKENIZER_DIR}/${TARGET_LANG}-bpe-${TGT_VOCAB_SIZE}.model" \
    --checkpoint-path "${CHECKPOINT_DIR}/checkpoint_best.pt" \
    --batch-size 1 \
    --max-len 100 \
    --output "$TRANSLATION_OUTPUT" \
    --bleu \
    --reference "${RAW_DATA}/${TEST_PREFIX}.${TARGET_LANG}"
