# TRAIL

This repository contains the experimental code for the paper *Grounded in
Knowledge, Guided by Reason: Automated Fake Cyber Threat Intelligence Detection
via Knowledge-Augmented LLM Rationales*.

## Repository Contents

```text
TRAIL/
├── run.sh                    # Training entrypoint
├── main.py                   # Parameter configuration and main program
├── grid_search.py            # Training scheduler
├── models/
│   ├── trail.py              # TRAIL model and trainer
│   └── layers.py             # Model layers
├── utils/
│   ├── dataloader.py         # Data loading
│   └── utils.py              # Training and evaluation utilities
├── requirements.txt          # Python dependencies
├── data/FCTI_HAL/            # Processed dataset
└── model/roberta-base/       # Local RoBERTa-base model
```

The repository does not include the raw dataset, processed data files, or
pretrained model weights.

## How to Run

### 1. Set Up the Environment

Python 3.10 is recommended:

```bash
git clone https://github.com/Eternaljj/TRAIL.git
cd TRAIL

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Prepare the Model

Place a complete Hugging Face RoBERTa-base model in:

```text
model/roberta-base/
```

Alternatively, specify another location with an environment variable:

```bash
export TRAIL_MODEL_PATH=/absolute/path/to/roberta-base
```

### 3. Prepare the Data

Place the processed FCTI-HAL dataset in:

```text
data/FCTI_HAL/
├── train.json
├── val.json
└── test.json
```

Each sample must contain at least `content`, `label`, `support_rationale`, and
`oppose_rationale`. The `id` and `selected_paths` fields are also recommended.

Alternatively, specify another data directory:

```bash
export TRAIL_DATA_DIR=/absolute/path/to/processed/FCTI_HAL
```

### 4. Start Training

```bash
bash run.sh
```

To select a GPU or change training parameters:

```bash
bash run.sh --gpu 0 --batch_size 8 --epochs 30
```

To view all available options:

```bash
bash run.sh --help
```
