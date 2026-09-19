# ATMT 2026 NMT Toolkit

Repository for the Advanced Techniques of Machine Translation assignments.

## Project Layout

```text
.
├── preprocess.py              # trains SentencePiece tokenizers and prepares data
├── train.py                   # trains/evaluates the Transformer NMT model
├── translate.py               # translates text with a trained checkpoint
├── toy_example.sh             # small end-to-end pipeline for setup checks
├── assignment1.sh             # Czech-English training pipeline
├── loadfromcheckpoint.sh      # continues training from an existing checkpoint
├── seq2seq/                   # model, decoding, and utility code
├── data/                      # mounted input data; not committed
└── output/                    # generated outputs/checkpoints/logs; not committed
```

`data/` and `output/` are ignored by Git. In Renku, `data/` should come from the course data connector or a team data connector, and `output/` should be backed by the team's writable SwitchDrive connector.

## Environment

The Renku launchers provide the expected environment. For local setup, use Python 3.11 and install the dependencies listed in `requirements.txt`, or create the Conda environment from `requirement.yml`.

## Data Paths

The full Czech-English run expects raw parallel files here:

```text
data/cz-en/raw/
├── train.cz
├── train.en
├── valid.cz
├── valid.en
├── test.cz
└── test.en
```

The toy example expects data under `data/toy_example/en-sv/infopankki/raw/`.

## Main Commands

Run the toy pipeline:

```bash
bash toy_example.sh
```

Run the Czech-English pipeline:

```bash
bash assignment1.sh
```

Both scripts can be configured with environment variables such as `RAW_DATA`, `WORK_DIR`, `OUTPUT_DIR`, `SOURCE_LANG`, `TARGET_LANG`, `SRC_VOCAB_SIZE`, and `TGT_VOCAB_SIZE`.

## Pipeline Reference

### Pre-processing

`preprocess.py` trains one BPE tokenizer per language, tokenizes each split, and stores the prepared files in the destination directory.

```bash
python preprocess.py \
    --source-lang cz \
    --target-lang en \
    --raw-data data/cz-en/raw \
    --dest-dir output/cz-en/prepared \
    --model-dir output/cz-en/tokenizers \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --src-vocab-size 8000 \
    --tgt-vocab-size 8000
```

Notes:
- Raw files are expected as `<split>.<lang>`, for example `train.cz` and `train.en`.
- Tokenizer models are named `<lang>-bpe-<vocab_size>.model`.
- The `.vocab` files are human-readable vocabulary files; tokenization uses the `.model` files.

### Training

`train.py` trains the Transformer model from prepared data and matching tokenizer models.

```bash
python train.py \
    --cuda \
    --data output/cz-en/prepared \
    --src-tokenizer output/cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer output/cz-en/tokenizers/en-bpe-8000.model \
    --source-lang cz \
    --target-lang en \
    --batch-size 64 \
    --arch transformer \
    --max-epoch 7 \
    --log-file output/cz-en/logs/train.log \
    --save-dir output/cz-en/checkpoints \
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
```

Notes:
- Add `--cuda` when training on a GPU.
- The tokenizer paths must match the models created during pre-processing.
- Model-specific arguments are defined in `seq2seq/models/transformer.py`.

### Translation

`translate.py` translates a raw source file or a prepared split using a trained checkpoint.

```bash
python translate.py \
    --cuda \
    --input data/cz-en/raw/test.cz \
    --src-tokenizer output/cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer output/cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path output/cz-en/checkpoints/checkpoint_best.pt \
    --batch-size 32 \
    --max-len 300 \
    --output output/cz-en/translations.txt \
    --bleu \
    --reference data/cz-en/raw/test.en
```

Notes:
- `--bleu` requires a reference file.
- Source and target language tags are loaded from the checkpoint.
- For prepared-data translation, use `--data output/cz-en/prepared` and `--dicts output/cz-en/tokenizers`.

## Outputs

Toy-example outputs are written to:

```text
output/toy_example/
```

Full Czech-English outputs are written to:

```text
output/cz-en/
├── prepared/
├── tokenizers/
├── checkpoints/
├── logs/train.log
└── translations.txt
```

Keep code changes in GitHub. Keep generated files, logs, checkpoints, and translations in `output/` so they persist through SwitchDrive.
