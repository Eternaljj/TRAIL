# TRAIL

TRAIL is a usefulness-aware, dual-rationale classifier for fake cyber threat
intelligence detection. The training pipeline encodes a claim, a supporting
rationale, an opposing rationale, and path-derived evidence; evaluates rationale
usefulness; adaptively weights the two rationale experts; and performs binary
classification.

The authoritative training entrypoint is `run.sh`.

## Directory layout

```text
TRAIL/
├── README.md
├── requirements.txt
├── run.sh
├── main.py
├── grid_search.py
├── models/
│   ├── __init__.py
│   ├── layers.py
│   └── trail.py
├── utils/
│   ├── __init__.py
│   ├── dataloader.py
│   └── utils.py
├── data/
│   └── FCTI_HAL/          # processed JSON files go here
└── model/
    └── roberta-base/      # local RoBERTa files go here
```

The package intentionally excludes API keys, `.env` files, generated logs,
checkpoints, caches, raw datasets, and unrelated experiment implementations.

## 1. System requirements

- Linux
- Python 3.10 recommended
- NVIDIA GPU with a CUDA driver compatible with PyTorch 2.3.0
- Sufficient GPU memory for RoBERTa-base training with batch size 16

Check the environment:

```bash
nvidia-smi
python --version
```

## 2. Create the Python environment

From the TRAIL directory:

```bash
cd ~/workspace/experiments/TRAIL
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the server manages CUDA-specific PyTorch wheels separately, install the
CUDA-compatible PyTorch 2.3.0 build first, then install the remaining packages
from `requirements.txt`.

Verify the installation:

```bash
python -c "import torch, transformers; print(torch.__version__); print(torch.cuda.is_available())"
```

## 3. Prepare RoBERTa-base

Place a complete local Hugging Face RoBERTa-base model under:

```text
~/workspace/experiments/TRAIL/model/roberta-base/
```

The directory should contain the model configuration, tokenizer files, and
weights, for example:

```text
model/roberta-base/
├── config.json
├── merges.txt
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
└── model.safetensors        # or pytorch_model.bin
```

The location can be overridden without editing the script:

```bash
export TRAIL_MODEL_PATH=/absolute/path/to/roberta-base
```

## 4. Prepare the processed FCTI-HAL data

`data/FCTI_HAL/` is intentionally empty. The available FCTI-HAL CSV is a raw
dataset and cannot be consumed directly by this training entrypoint.

After preprocessing and rationale/path generation, place these three files in:

```text
~/workspace/experiments/TRAIL/data/FCTI_HAL/
├── train.json
├── val.json
└── test.json
```

Each file must be one UTF-8 JSON array. Each item requires:

- `content`: CTI claim text
- `label`: `real`/`fake` or the numeric equivalent accepted by the loader
- `support_rationale`: supporting rationale text
- `oppose_rationale`: opposing rationale text

Recommended fields:

- `id`: sample identifier
- `selected_paths`: list of evidence paths

When `selected_paths` is present, every path may contain:

```json
{
  "path_type": "support",
  "triples": [
    {
      "head": "entity A",
      "relation": "relation",
      "tail": "entity B"
    }
  ]
}
```

`path_type` must be `support` or `oppose`. If usable selected paths are absent,
the loader falls back to the corresponding rationale as evidence.

To use another processed-data directory:

```bash
export TRAIL_DATA_DIR=/absolute/path/to/processed/FCTI_HAL
```

The entrypoint uses `python3` by default. To use the `python` command from an
activated virtual or Conda environment:

```bash
export PYTHON_BIN=python
```

## 5. Run training

Default run:

```bash
cd ~/workspace/experiments/TRAIL
source .venv/bin/activate
bash run.sh
```

The default configuration is:

| Setting | Value |
|---|---:|
| GPU | 3 |
| Epochs | 30 |
| Batch size | 16 |
| Learning rate | 1e-4 |
| Early stopping patience | 10 |
| Maximum sequence length | 256 |
| Weight decay | 1e-4 |
| Usefulness margin | 0.05 |
| Usefulness loss weight | 0.20 |
| Pooling | attention |
| Explicit conflict features | enabled |

Example overrides:

```bash
bash run.sh \
  --gpu 0 \
  --batch_size 8 \
  --epochs 30 \
  --usefulness_margin 0.05 \
  --lambda_use 0.20 \
  --pooling_method attention
```

Show all options:

```bash
bash run.sh --help
```

## 6. Outputs

Training creates:

```text
param_model/TRAIL_cti-hal-conflict/1/
├── parameter_bert.pkl
└── parameter_bert_acc.pkl

logs/test/TRAIL_cti-hal-conflict/month_1.json
```

- `parameter_bert.pkl` is the checkpoint selected by validation macro F1.
- `parameter_bert_acc.pkl` is the checkpoint selected by validation accuracy.
- `month_1.json` records test metrics for both checkpoints.

Additional parameter, TensorBoard, and intermediate JSON logs are written below
`logs/`.

## 7. Common issues

### Processed dataset file is missing

Confirm that all three processed files are present:

```bash
ls -lh data/FCTI_HAL/train.json data/FCTI_HAL/val.json data/FCTI_HAL/test.json
```

### RoBERTa cannot be loaded

Confirm that `model/roberta-base/` is a complete Hugging Face model directory,
not only a tokenizer or configuration directory.

### CUDA is unavailable

Check:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Then verify the NVIDIA driver and installed PyTorch build.

### Out of GPU memory

Reduce the batch size:

```bash
bash run.sh --batch_size 8
```

### Disable explicit conflict features

```bash
bash run.sh --disable_conflict_features
```
