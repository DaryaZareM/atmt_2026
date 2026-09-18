# ATMT Codebase

Materials for "Advanced Techniques of Machine Translation" (UZH, HS26).

**Please refer to the assignment sheet for instructions for the actual assignments.**

The toolkit is based on last year's [version](https://github.com/davidguzmanp/atmt_2024/), but with significant upgrades, such as BPE-tokenization (using [Sentencepiece](https://github.com/google/sentencepiece)) and the change from an RNN to a transformer architecture (based on [this implementation](https://github.com/AliAbdien/Transformer-based-Machine-Translation-from-Scratch))


## Environment Setup

The coding assignments are designed to be run in a Renku session with GPU support. Open a terminal in the Renku session and run the Python scripts directly; do not submit the provided scripts with `sbatch`.

You can check that PyTorch sees the GPU with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Toy dry-run

To try the full pipeline on a very small dataset, run:

```bash
bash toy_example.sh
```

The script reads raw toy data from `data/toy_example/en-sv/infopankki/raw` and writes all generated files under `output/toy_example`: prepared data, tokenizers, logs, checkpoints, and translations. If your raw files are somewhere else, run with `RAW_DATA=/path/to/raw bash toy_example.sh`.

### Installing required packages

Use `uv` to create the Python environment. The helper script keeps the virtual environment, uv cache, and any uv-managed Python install under `output/`, which is useful on Renku when the project files are read-only:

```bash
bash setup_uv.sh
source output/.venv/bin/activate
```

After that, `bash toy_example.sh` will use `output/.venv/bin/python` automatically. The file `requirements-uv.txt` is the pip-style dependency list used by uv. The file `requirements.txt` is a conda export and should not be used with `pip`; `requirement.yml` is kept as a conda-style reference.
<!-- ### conda

```
# ensure that you have conda (or miniconda) installed (https://conda.io/projects/conda/en/latest/user-guide/install/index.html) and that it is activated

# create clean environment
conda create --name atmt311 python=3.11

# activate the environment
conda activate atmt311

# intall required packages
conda install pytorch=2.0.1 numpy tqdm sacrebleu
```

### virtualenv

```
# ensure that you have python > 3.6 downloaded and installed (https://www.python.org/downloads/)

# install virtualenv
pip install virtualenv  # for both powershell and WSL

# create a virtual environment named "atmt311"
virtualenv --python=python3.11 atmt311  # on WSL terminal
python -m venv atmt311    # on powershell

# launch the newly created environment
source atmt311/bin/activate
.\atmt311\Scripts\Activate.ps1   # on powershell


# intall required packages
pip install torch==2.0.1 numpy tqdm sacrebleu   # for both powershell and WSL
``` -->

<!-- # Data Preprocessing

```
# normalise, tokenize and truecase data
bash scripts/extract_splits.sh ../infopankki_raw data/en-sv/infopankki/raw

# binarize data for model training
bash scripts/run_preprocessing.sh data/en-sv/infopankki/raw/
``` -->


## Usage Example

Let's have a look at what how to use the repository.

The following steps are a working example on the data in the directory `toy_example`.

<details>

<summary>Example File Structure:</summary>

```
toy_example/
└── data/
    └── raw/
        ├── train.en
        ├── train.de
        ├── valid.en
        ├── valid.de
        ├── test.en
        └── test.de
```

</details>

### pre-processing

Here, the following things happen:
- A separate **BPE-tokenizer** is trained for each side (source and target), with the specified vocab-size
- Each split will be tokenized using the (corresponding language's) tokenizer, turning each sentence from a string into a tensor containing **token-IDs**
- Each split is _pickled_ (stored in binary format) in the specified `--dest-dir`-folder
- 

```bash
python preprocess.py \
    --source-lang cz \  # the tag of the source language (Czech)
    --target-lang en \  # the tag of the target language (English)
    --raw-data .\toy_example\data\raw \  # unprocessed .txt-files, usually named <split>.<tag>, e.g. train.cz
    --dest-dir output/toy_example/prepared \  # where the processed (tokenized, pickled) files will be stored to
    --model-dir output/toy_example/tokenizers \  # where to store the trained tokenization models
    --test-prefix test \  # expected prefix for files belonging to the test-split
    --train-prefix train \  # expected prefix for files belonging to the train-split
    --valid-prefix valid \  # expected prefix for files belonging to the validation-split
    --src-vocab-size 1000 \  # size of the source (BPE-)vocab
    --tgt-vocab-size 1000 \  # size of the target (BPE-)vocab
    --ignore-existing \  # overwrite existing processed files (only recommended for testing)
    --force-train  # force training, even if a tokenizer (same language and vocab-size) already exists
```

Notes:
- The data to train the tokenizer is expected to be stored as raw text under the name `<train-prefix>.<tag>` in the specified `raw-data` directory
<br>e.g. `.\toy_example\data\raw\train.en`
- Filename of the resulting tokenizer: `<tag>-bpe-<vocab_size>.model`
<br>e.g. `cz-bpe-1000.model`
- The `.vocab` files exist just to have a user-readable version of the vocabulary. All actual (de-)tokenization will use the `.model` files

-   <details>

    <summary> Resulting file structure </summary> 

    ```
    toy_example/
    ├── data/
    │   ├── raw/
    │   │   ├── train.en
    │   │   ├── train.de
    │   │   ├── valid.en
    │   │   ├── valid.de
    │   │   ├── test.en
    │   │   └── test.de
    │   └── prepared/
    │       ├── train.en
    │       ├── train.de
    │       ├── valid.en
    │       ├── valid.de
    │       ├── test.en
    │       └── test.de
    └── tokenizers/
        ├── en-bpe-1200.model
        ├── de-bpe-1200.model
        ├── en-bpe-1200.vocab
        └── de-bpe-1200.vocab
    ```

    </details>

### Training a model

Trains a transformer model on the prepared data.

<details>

<summary> More details </summary>

1. Build model with the specified parameters & load data with the provided tokenizers.
2. During each epoch, the model iterates over the **training** data in batches, computes the loss using teacher forcing, performs backpropagation, applies gradient , and updates the model parameters with the optimizer.
3. At the end of each epoch,
    - calculate the **validation** loss and perplexity _with teacher forcing_
    - generate translations _without teacher forcing_ and compute BLEU from it
4. Training stops early if the validation loss does not improve for a set number of epochs (patience).
5. The model's performance is evaluated on a final, unseen **test set**

</details>

<br>

```bash
python train.py \
    --data output/toy_example/prepared/ \  # output of preprocess.py
    --src-tokenizer output/toy_example/tokenizers/cz-bpe-1000.model \  # location of the source tokenizer model
    --tgt-tokenizer output/toy_example/tokenizers/en-bpe-1000.model \  # location of the target tokenizer model
    --source-lang cz \
    --target-lang en \
    --batch-size 32 \  # batch size for training and validation
    --arch transformer \  # architecture variant: by default, only transformer exists
    --max-epoch 10 \  # maximum number of epochs (early stopping possible)
    --log-file output/toy_example/logs/train.log \  # log-file name and location
    --save-dir output/toy_example/checkpoints/ \  # directory to save the model checkpoints to
    --ignore-checkpoints \  # ignore potential existing checkpoints & train from scratch
    --encoder-dropout 0.1 \ 
    --decoder-dropout 0.1 \
    --dim-embedding 256 \  # embedding (and general model) dimension
    --attention-heads 4 \  # number of attention heads
    --dim-feedforward-encoder 1024 \  # dimension of Encoder-FFNs
    --dim-feedforward-decoder 1024 \  # dimension of Decoder-FFNs
    --max-seq-len 100 \  # maximum sequence length (longer inputs/outputs will be trimmed to the specified number of tokens)
    --n-encoder-layers 3 \  # number of encoder layers
    --n-decoder-layers 3  # number of decoder layers
```

Notes:
- add the `--cuda` flag when training in a Renku GPU session
- the source- and target-tokenizer arguments have to match the model files created in the pre-processing step
- the progress-bar is hardcoded to update only in 2-second intervals. This reduces clutter during longer training runs
- the model specific arguments are added to `args` in the python file implementing said model, in this case `seq2seq/models/transformer.py`

# Translation

Translate a raw (source language) text file (one sentence per line) to the target language
```bash
python translate.py \
    --input toy_example/data/raw/test.cz \  #  Path to the raw source text file (one sentence per line, in Czech)
    --src-tokenizer output/toy_example/tokenizers/cz-bpe-1000.model \  # Path to the trained SentencePiece tokenizer model for the source language
    --tgt-tokenizer output/toy_example/tokenizers/en-bpe-1000.model \  # Path to the trained SentencePiece tokenizer model for the target language
    --checkpoint-path output/toy_example/checkpoints/checkpoint_best.pt \ # Path to the trained model checkpoint
    --batch-size 1 \  # Number of sentences to process in each batch 
    --max-len 100 \  # Maximum length of generated translations (in tokens)
    --output output/toy_example/toy_example_output.en \  # Path to write the generated translations (one per line)
    --bleu \  # If set, compute BLEU score after translation (score output vs. reference)
    --reference toy_example/data/raw/test.en  # Path to the reference translation file (one sentence per line, in English)
```

Notes:
- The source and target language tags do not have to be passed, as they are loaded and parsed as part of the model checkpoint
- If `--bleu` is set but no reference is provided, this will throw an error
- In Renku assignments you can also translate an already prepared source split with `--data`. When `--data` is a directory, `translate.py` reads `<prepared-split>.<source-lang>` from that directory; the default split is `test`.

Example prepared-data translation:

```bash
python translate.py \
    --cuda \
    --data data/en-sv/bible_uedin/prepared \
    --dicts data/en-sv/infopankki/tokenizers \
    --checkpoint-path assignments/01/baseline/checkpoints/checkpoint_last.pt \
    --output assignments/01/baseline/bible_translations.txt \
    --prepared-split test \
    --batch-size 32 \
    --max-len 300
```

`--dicts` should point to the directory containing the SentencePiece `.model` files. If you pass a neighboring directory such as `prepared`, the script also checks common sibling locations such as `../tokenizers`.


# Assignments

Assignments must be submitted on OLAT by 14:00 on their respective
due dates.
