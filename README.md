# TRAIL

该仓库为论文《Grounded in Knowledge, Guided by Reason: Automated Fake Cyber
Threat Intelligence Detection via Knowledge-Augmented LLM Rationales》的实验代码。

## 仓库内容

```text
TRAIL/
├── run.sh                    # 训练入口
├── main.py                   # 参数配置与主程序
├── grid_search.py            # 训练调度
├── models/
│   ├── trail.py              # TRAIL 模型与训练器
│   └── layers.py             # 模型基础层
├── utils/
│   ├── dataloader.py         # 数据加载
│   └── utils.py              # 训练与评估工具
├── requirements.txt          # Python 依赖
├── data/FCTI_HAL/            # 处理后的数据集
└── model/roberta-base/       # 本地 RoBERTa-base 模型
```

仓库不包含原始数据、处理后的数据文件及预训练模型权重。

## 运行方法

### 1. 安装环境

推荐使用 Python 3.10：

```bash
git clone https://github.com/Eternaljj/TRAIL.git
cd TRAIL

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. 准备模型

将完整的 Hugging Face RoBERTa-base 模型放到：

```text
model/roberta-base/
```

也可以通过环境变量指定其他位置：

```bash
export TRAIL_MODEL_PATH=/absolute/path/to/roberta-base
```

### 3. 准备数据

将处理后的 FCTI-HAL 数据放到：

```text
data/FCTI_HAL/
├── train.json
├── val.json
└── test.json
```

每条数据至少包含 `content`、`label`、`support_rationale` 和
`oppose_rationale` 字段，建议同时包含 `id` 与 `selected_paths`。

也可以通过环境变量指定其他数据目录：

```bash
export TRAIL_DATA_DIR=/absolute/path/to/processed/FCTI_HAL
```

### 4. 开始训练

```bash
bash run.sh
```

指定 GPU 或调整训练参数：

```bash
bash run.sh --gpu 0 --batch_size 8 --epochs 30
```

查看全部参数：

```bash
bash run.sh --help
```
