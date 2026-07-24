import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Suppress transformers truncation warnings
warnings.filterwarnings('ignore', message='.*overflowing tokens.*')

import torch
import random
import pandas as pd
import json
import numpy as np
import nltk
import jieba
from transformers import BertTokenizer, RobertaTokenizer, AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

label_dict = {
    "real": 0,
    "fake": 1,
    0: 0,
    1: 1
}

label_dict_ftr_pred = {
    "real": 0,
    "fake": 1,
    "other": 2,
    0: 0,
    1: 1,
    2: 2
}

def word2input(texts, max_len, tokenizer):
    token_ids = []
    # Suppress warnings for this function
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*overflowing tokens.*')
        for i, text in enumerate(texts):
            token_ids.append(
                tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                                 truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks

def get_dataloader(path, max_len, batch_size, shuffle, bert_path, data_type, language, top_k_paths=None, **kwargs):
    # 【修复】使用 AutoTokenizer 自动识别 tokenizer 类型
    # 这样可以兼容 ModernBERT (PreTrainedTokenizerFast) 和传统 BERT (BertTokenizer)
    try:
        # 优先尝试使用 AutoTokenizer（兼容所有类型）
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
    except Exception as e:
        # 如果 AutoTokenizer 失败，回退到手动选择
        print(f"⚠️  AutoTokenizer 加载失败，尝试手动选择: {e}")
        if 'roberta' in bert_path.lower():
            # 使用 RoBERTa tokenizer
            tokenizer = RobertaTokenizer.from_pretrained(bert_path)
        else:
            # 使用 BERT tokenizer（默认）
            tokenizer = BertTokenizer.from_pretrained(bert_path)

    if data_type == 'rationale':
        data_list = json.load(open(path, 'r',encoding='utf-8'))
        data_rows = []
        for item in data_list:
            tmp_data = {}

            # content info
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item['source_id']

            tmp_data['FTR_2'] = item['td_rationale']
            tmp_data['FTR_3'] = item['cs_rationale']

            tmp_data['FTR_2_pred'] = item['td_pred']
            tmp_data['FTR_3_pred'] = item['cs_pred']

            tmp_data['FTR_2_acc'] = item['td_acc']
            tmp_data['FTR_3_acc'] = item['cs_acc']

            data_rows.append(tmp_data)

        df_data = pd.DataFrame(data_rows)

        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        # Convert string IDs to numeric IDs
        id_to_num = {id_str: idx for idx, id_str in enumerate(df_data['id'].unique())}
        id = torch.tensor(df_data['id'].apply(lambda x: id_to_num[x]).astype(int).to_numpy())

        FTR_2_pred = torch.tensor(df_data['FTR_2_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())
        FTR_3_pred = torch.tensor(df_data['FTR_3_pred'].apply(lambda c: label_dict_ftr_pred[c]).astype(int).to_numpy())

        FTR_2_acc = torch.tensor(df_data['FTR_2_acc'].astype(int).to_numpy())
        FTR_3_acc = torch.tensor(df_data['FTR_3_acc'].astype(int).to_numpy())

        FTR_2 = df_data['FTR_2'].to_numpy()
        FTR_3 = df_data['FTR_3'].to_numpy()

        content_token_ids, content_masks = word2input(content, max_len, tokenizer)

        FTR_2_token_ids, FTR_2_masks = word2input(FTR_2, max_len, tokenizer)
        FTR_3_token_ids, FTR_3_masks = word2input(FTR_3, max_len, tokenizer)

        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                FTR_2_pred,
                                FTR_2_acc,
                                FTR_3_pred,
                                FTR_3_acc,
                                FTR_2_token_ids,
                                FTR_2_masks,
                                FTR_3_token_ids,
                                FTR_3_masks,
                                label,
                                id,
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )
        return dataloader
    elif data_type == 'content_only':
        # 只加载content，不使用rationale
        data_list = json.load(open(path, 'r', encoding='utf-8'))
        data_rows = []
        for item in data_list:
            tmp_data = {}
            # 只加载content和label
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item['source_id']
            data_rows.append(tmp_data)

        df_data = pd.DataFrame(data_rows)

        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        # Convert string IDs to numeric IDs
        id_to_num = {id_str: idx for idx, id_str in enumerate(df_data['id'].unique())}
        id = torch.tensor(df_data['id'].apply(lambda x: id_to_num[x]).astype(int).to_numpy())

        content_token_ids, content_masks = word2input(content, max_len, tokenizer)

        # 只包含content相关的数据
        dataset = TensorDataset(
            content_token_ids,
            content_masks,
            label,
            id,
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )
        return dataloader

    elif data_type == 'conflict_rationale':
        """
        Stage 4: conflict-aware KG-grounded rationale
        Load content and conflict_rationale

        Expert Design:
        - Expert 2 (FTR_2): content × conflict_rationale (external KG knowledge)
        - Expert 3 (FTR_3): content × content (self-interaction)

        This allows the model to learn from both:
        1. External knowledge (KG-grounded conflict analysis)
        2. Internal representation (self-understanding)

        Filtering/Weighting Options:
        - filter_insufficient_evidence: Filter out [Conflict=INSUFFICIENT_EVIDENCE] samples
        - insufficient_evidence_weight: Weight for INSUFFICIENT_EVIDENCE samples (if not filtered)
        """
        import re

        def extract_conflict_type(rationale):
            """Extract conflict type from rationale text."""
            if not rationale:
                return "UNKNOWN"
            match = re.search(r'\[Conflict=(\w+)\]', rationale)
            if match:
                return match.group(1).upper()
            return "UNKNOWN"

        data_list = json.load(open(path, 'r', encoding='utf-8'))
        data_rows = []

        # Get filtering/weighting config from kwargs or use defaults
        filter_insufficient = kwargs.get('filter_insufficient_evidence', False) if kwargs else False
        insufficient_weight = kwargs.get('insufficient_evidence_weight', 1.0) if kwargs else 1.0

        filtered_count = 0
        insufficient_count = 0

        for item in data_list:
            tmp_data = {}
            # content info
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item.get('id') or item.get('source_id', '')

            # conflict rationale (single rationale)
            conflict_rationale = item.get('conflict_rationale', '')
            tmp_data['conflict_rationale'] = conflict_rationale

            # Extract conflict type
            conflict_type = extract_conflict_type(conflict_rationale)
            tmp_data['conflict_type'] = conflict_type

            # Filter or weight INSUFFICIENT_EVIDENCE samples
            if conflict_type == 'INSUFFICIENT_EVIDENCE':
                insufficient_count += 1
                if filter_insufficient:
                    # Skip this sample
                    filtered_count += 1
                    continue
                else:
                    # Add weight for this sample
                    tmp_data['sample_weight'] = insufficient_weight
            else:
                tmp_data['sample_weight'] = 1.0

            data_rows.append(tmp_data)

        if filter_insufficient:
            print(f"  ⚠️  过滤了 {filtered_count} 个 INSUFFICIENT_EVIDENCE 样本")
            print(f"  ⚠️  剩余样本数: {len(data_rows)} (原始: {len(data_list)})")
        elif insufficient_weight != 1.0:
            print(f"  ⚠️  降低了 {insufficient_count} 个 INSUFFICIENT_EVIDENCE 样本的权重 (weight={insufficient_weight})")
            print(f"  ⚠️  总样本数: {len(data_rows)}")
            print(f"  ⚠️  权重分布: INSUFFICIENT_EVIDENCE={insufficient_weight}, 其他=1.0")

        df_data = pd.DataFrame(data_rows)

        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        # Convert string IDs to numeric IDs
        id_to_num = {id_str: idx for idx, id_str in enumerate(df_data['id'].unique())}
        id = torch.tensor(df_data['id'].apply(lambda x: id_to_num[x]).astype(int).to_numpy())

        # Tokenize content and conflict_rationale
        conflict_rationale = df_data['conflict_rationale'].to_numpy()

        content_token_ids, content_masks = word2input(content, max_len, tokenizer)
        conflict_token_ids, conflict_masks = word2input(conflict_rationale, max_len, tokenizer)

        # Create dummy labels for auxiliary losses (will be disabled in training)
        # These are required by ARGModel even when auxiliary loss weights = -1
        num_samples = len(label)
        FTR_2_pred = torch.zeros(num_samples, dtype=torch.long)  # dummy: all 0 (real)
        FTR_2_acc = torch.zeros(num_samples, dtype=torch.long)   # dummy: all 0 (incorrect)
        FTR_3_pred = torch.zeros(num_samples, dtype=torch.long)  # dummy: all 0 (real)
        FTR_3_acc = torch.zeros(num_samples, dtype=torch.long)   # dummy: all 0 (incorrect)

        # Sample weights for weighted sampling (if not filtering)
        sample_weights = torch.tensor(df_data['sample_weight'].astype(float).to_numpy())

        # Package as TensorDataset (compatible with ARGModel)
        # Expert 2 (FTR_2): conflict_rationale - external KG knowledge
        # Expert 3 (FTR_3): content itself - self-interaction
        dataset = TensorDataset(
            content_token_ids,    # [N, L]
            content_masks,        # [N, L]
            FTR_2_pred,           # [N] - dummy for LLM judgment predictor
            FTR_2_acc,            # [N] - dummy for rationale usefulness evaluator
            FTR_3_pred,           # [N] - dummy for LLM judgment predictor
            FTR_3_acc,            # [N] - dummy for rationale usefulness evaluator
            conflict_token_ids,   # [N, L] - as FTR_2 (conflict_rationale)
            conflict_masks,       # [N, L] - as FTR_2 mask
            content_token_ids,    # [N, L] - as FTR_3 (content itself, self-interaction)
            content_masks,        # [N, L] - as FTR_3 mask
            label,                # [N]
            id,                   # [N]
            sample_weights,       # [N] - sample weights for loss weighting
        )

        # Sampling strategy:
        # Option 1: Use WeightedRandomSampler (changes which samples are selected)
        # Option 2: Use normal sampling + loss weighting (only changes loss contribution)
        #
        # By default, we use both for maximum effect. But you can disable WeightedRandomSampler
        # by setting use_weighted_sampler=False in kwargs
        use_weighted_sampler = kwargs.get('use_weighted_sampler', True) if kwargs else True

        if not filter_insufficient and insufficient_weight != 1.0 and use_weighted_sampler:
            # Option 1: WeightedRandomSampler + Loss weighting (双重机制)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=1,
                pin_memory=False,
                shuffle=False  # sampler handles shuffling
            )
            print(f"  ⚠️  使用 WeightedRandomSampler + Loss 权重 (双重机制)")
        else:
            # Option 2: Normal sampling + Loss weighting only
            dataloader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=1,
                pin_memory=False,
                shuffle=shuffle
            )
            if not filter_insufficient and insufficient_weight != 1.0:
                print(f"  ⚠️  仅使用 Loss 权重 (不使用 WeightedRandomSampler)")
        return dataloader

    elif data_type == 'dual_rationale':
        """
        Stage 4 v3: Dual Rationale (Support + Oppose) with Confidence Scores

        Each sample contains:
        - content: claim text
        - support_rationale: rationale supporting the claim
        - support_confidence: confidence score for support (0-1)
        - oppose_rationale: rationale opposing the claim
        - oppose_confidence: confidence score for oppose (0-1)
        - label: 0 (real) or 1 (fake)
        - id: sample identifier

        Expert Design:
        - Expert 2 (FTR_2): content × support_rationale (weighted by support_confidence)
        - Expert 3 (FTR_3): content × oppose_rationale (weighted by oppose_confidence)

        This allows the model to learn from both supporting and opposing evidence,
        with confidence scores providing additional weighting signals.
        """
        data_list = json.load(open(path, 'r', encoding='utf-8'))
        data_rows = []

        for item in data_list:
            tmp_data = {}
            # content info
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item.get('id') or item.get('source_id', '')

            # dual rationales
            tmp_data['support_rationale'] = item.get('support_rationale', '')
            tmp_data['support_confidence'] = float(item.get('support_confidence', 0.5))
            tmp_data['oppose_rationale'] = item.get('oppose_rationale', '')
            tmp_data['oppose_confidence'] = float(item.get('oppose_confidence', 0.5))

            # Validate confidence scores (should be 0-1)
            tmp_data['support_confidence'] = max(0.0, min(1.0, tmp_data['support_confidence']))
            tmp_data['oppose_confidence'] = max(0.0, min(1.0, tmp_data['oppose_confidence']))

            data_rows.append(tmp_data)

        print(f"  ✅ 加载了 {len(data_rows)} 个样本 (dual rationale)")
        avg_support_conf = sum(d['support_confidence'] for d in data_rows) / len(data_rows)
        avg_oppose_conf = sum(d['oppose_confidence'] for d in data_rows) / len(data_rows)
        print(f"  📊 平均支持置信度: {avg_support_conf:.3f}")
        print(f"  📊 平均反对置信度: {avg_oppose_conf:.3f}")

        df_data = pd.DataFrame(data_rows)

        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        # Convert string IDs to numeric IDs
        id_to_num = {id_str: idx for idx, id_str in enumerate(df_data['id'].unique())}
        id = torch.tensor(df_data['id'].apply(lambda x: id_to_num[x]).astype(int).to_numpy())

        # Tokenize content, support_rationale, and oppose_rationale
        support_rationale = df_data['support_rationale'].to_numpy()
        oppose_rationale = df_data['oppose_rationale'].to_numpy()

        content_token_ids, content_masks = word2input(content, max_len, tokenizer)
        support_token_ids, support_masks = word2input(support_rationale, max_len, tokenizer)
        oppose_token_ids, oppose_masks = word2input(oppose_rationale, max_len, tokenizer)

        # Confidence scores as tensors (ensure float32 to match model weights)
        support_confidence = torch.tensor(df_data['support_confidence'].astype(np.float32).to_numpy(), dtype=torch.float32)
        oppose_confidence = torch.tensor(df_data['oppose_confidence'].astype(np.float32).to_numpy(), dtype=torch.float32)

        # Calculate conflict_score = support_confidence - oppose_confidence
        # Range: [-1, 1] (natural, since both confidences are in [0, 1])
        # Positive values indicate stronger support, negative values indicate stronger opposition
        conflict_score = support_confidence - oppose_confidence  # [N]

        # Create labels for auxiliary losses
        num_samples = len(label)

        # LLM Judgment Predictor labels (3-class: 0=Real, 1=Fake, 2=Uncertain)
        # For support_rationale: high confidence → predict actual label, low confidence → uncertain
        # For oppose_rationale: high confidence → predict opposite label, low confidence → uncertain
        judgment_threshold = 0.5

        # Support rationale judgment: if support_confidence is high, predict the actual label
        # If support_confidence is low, predict uncertain (2)
        FTR_2_pred = torch.where(
            support_confidence > oppose_confidence,
            label,  # High confidence: predict actual label (0=real, 1=fake)
            torch.full_like(label, 2)  # Low confidence: uncertain
        )

        # Oppose rationale judgment: if oppose_confidence is high, predict opposite of actual label
        # If oppose_confidence is low, predict uncertain (2)
        FTR_3_pred = torch.where(
            oppose_confidence > support_confidence,
            label,  # High confidence: predict opposite (0→1, 1→0)
            torch.full_like(label, 2)  # Low confidence: uncertain
        )

        # Rationale usefulness labels: use confidence scores as proxy
        # High confidence = rationale is useful (1), low confidence = less useful (0)
        # FTR_2_acc: 1 if support_confidence > oppose_confidence (support rationale is more useful)
        # FTR_3_acc: 1 if oppose_confidence > support_confidence (oppose rationale is more useful)
        FTR_2_acc = (support_confidence > oppose_confidence).long()  # support_rationale usefulness
        FTR_3_acc = (oppose_confidence > support_confidence).long()   # oppose_rationale usefulness

        # Package as TensorDataset (compatible with ARGModel)
        # Expert 2 (FTR_2): support_rationale - supporting evidence
        # Expert 3 (FTR_3): oppose_rationale - opposing evidence
        dataset = TensorDataset(
            content_token_ids,      # [N, L]
            content_masks,          # [N, L]
            FTR_2_pred,             # [N] - dummy for LLM judgment predictor
            FTR_2_acc,              # [N] - dummy for rationale usefulness evaluator
            FTR_3_pred,             # [N] - dummy for LLM judgment predictor
            FTR_3_acc,              # [N] - dummy for rationale usefulness evaluator
            support_token_ids,       # [N, L] - as FTR_2 (support_rationale)
            support_masks,           # [N, L] - as FTR_2 mask
            oppose_token_ids,        # [N, L] - as FTR_3 (oppose_rationale)
            oppose_masks,            # [N, L] - as FTR_3 mask
            label,                   # [N]
            id,                      # [N]
            support_confidence,      # [N] - confidence scores for support
            oppose_confidence,       # [N] - confidence scores for oppose
            conflict_score,          # [N] - conflict_score = oppose_conf - support_conf
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )

        return dataloader

    elif data_type == 'conflict_learning':
        """
        Conflict Learning mode (Stage 4 v5):
        - Uses support_rationale (evidence for the claim) and oppose_rationale (evidence against)
        - Computes conflict_margin = oppose_confidence - support_confidence
        - No auxiliary losses (LLM judgment, rationale usefulness)
        - Pure adversarial learning based on confidence conflict

        【新增部分】Evidence Text Extraction:
        - Extracts evidence_text from selected_paths.triples[].sentence_snippet
        - Top-K (3-5) snippets concatenated
        """
        print(f"Loading data for conflict_learning mode from {path}")

        data = json.load(open(path, 'r', encoding='utf-8'))
        data_rows = []

        # 【优化】Extract evidence_text from selected_paths
        # 使用三元组结构化信息 + 原始文本
        def extract_evidence_text(item, top_k=5):
            """
            Extract evidence text from selected_paths using structured triples + snippets
            Returns: concatenated string with structured triple info
            """
            selected_paths = item.get('selected_paths', [])
            if not selected_paths:
                return ""

            evidence_parts = []

            # 【新增】首先添加content_triple信息
            content_triple = item.get('content_triple', [])
            if content_triple and isinstance(content_triple, list):
                claim_triples = []
                for triple in content_triple[:3]:  # 最多3个
                    if isinstance(triple, dict):
                        h = triple.get('head', '')
                        r = triple.get('relation', '')
                        t = triple.get('tail', '')
                        if h and r and t:
                            claim_triples.append(f"{h} {r} {t}")
                if claim_triples:
                    evidence_parts.append("[Claim Facts] " + "; ".join(claim_triples))

            # 然后添加KG paths的结构化三元组
            for path in selected_paths[:top_k]:
                path_type = path.get('path_type', 'unknown')
                score = path.get('score', 0.0)
                triples = path.get('triples', [])

                path_triples = []
                for triple in triples:
                    # 使用结构化信息
                    h = triple.get('head', '')
                    r = triple.get('relation', '')
                    t = triple.get('tail', '')
                    if h and r and t:
                        path_triples.append(f"{h} {r} {t}")

                    # 也保留原始文本作为补充
                    snippet = triple.get('sentence_snippet', '')
                    if snippet and snippet.strip():
                        path_triples.append(snippet.strip()[:100])  # 限制长度

                if path_triples:
                    path_text = f"[{path_type.upper()} path, score={score:.2f}] " + " | ".join(path_triples)
                    evidence_parts.append(path_text)

            # 合并所有证据
            evidence_text = " ".join(evidence_parts[:top_k * 2])  # 增加容量
            return evidence_text

        for item in data:
            tmp_data = {}
            tmp_data['content'] = item.get('content', item.get('claim', ''))
            tmp_data['label'] = item.get('label', 0)
            tmp_data['id'] = item.get('id') or item.get('source_id', '')

            # Load dual rationales and confidence scores
            tmp_data['support_rationale'] = item.get('support_rationale', '')
            tmp_data['support_confidence'] = float(item.get('support_confidence', 0.5))
            tmp_data['oppose_rationale'] = item.get('oppose_rationale', '')
            tmp_data['oppose_confidence'] = float(item.get('oppose_confidence', 0.5))

            # Validate confidence scores (should be 0-1)
            tmp_data['support_confidence'] = max(0.0, min(1.0, tmp_data['support_confidence']))
            tmp_data['oppose_confidence'] = max(0.0, min(1.0, tmp_data['oppose_confidence']))

            # 【新增部分】Extract evidence_text from selected_paths
            tmp_data['evidence_text'] = extract_evidence_text(item, top_k=5)

            # 【新增部分】Load 3-way classification labels (for conflict_5.json)
            # tri_label_id: 0=SUPPORTED, 1=INSUFFICIENT, 2=CONTRADICTED
            tmp_data['tri_label_id'] = item.get('tri_label_id', 0)  # Default to 0 if not present
            tmp_data['sample_weight_3way'] = float(item.get('sample_weight_3way', 1.0))  # Default to 1.0

            data_rows.append(tmp_data)

        df_data = pd.DataFrame(data_rows)

        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        # Convert string IDs to numeric IDs
        id_to_num = {id_str: idx for idx, id_str in enumerate(df_data['id'].unique())}
        id = torch.tensor(df_data['id'].apply(lambda x: id_to_num[x]).astype(int).to_numpy())

        # Tokenize content, support_rationale, and oppose_rationale
        support_rationale = df_data['support_rationale'].to_numpy()
        oppose_rationale = df_data['oppose_rationale'].to_numpy()

        content_token_ids, content_masks = word2input(content, max_len, tokenizer)
        support_token_ids, support_masks = word2input(support_rationale, max_len, tokenizer)
        oppose_token_ids, oppose_masks = word2input(oppose_rationale, max_len, tokenizer)

        # 【新增部分】Tokenize evidence_text
        evidence_text = df_data['evidence_text'].to_numpy()
        evidence_token_ids, evidence_masks = word2input(evidence_text, max_len, tokenizer)

        # Confidence scores as tensors (ensure float32)
        support_confidence = torch.tensor(df_data['support_confidence'].astype(np.float32).to_numpy(), dtype=torch.float32)
        oppose_confidence = torch.tensor(df_data['oppose_confidence'].astype(np.float32).to_numpy(), dtype=torch.float32)

        # Calculate conflict_margin = oppose_confidence - support_confidence
        # Positive: oppose is stronger → should predict fake
        # Negative: support is stronger → should predict real
        conflict_margin = oppose_confidence - support_confidence  # [N]

        # 【新增部分】Load 3-way classification labels and sample weights
        tri_label_id = torch.tensor(df_data['tri_label_id'].astype(int).to_numpy(), dtype=torch.long)  # [N] - 0,1,2
        sample_weight_3way = torch.tensor(df_data['sample_weight_3way'].astype(np.float32).to_numpy(), dtype=torch.float32)  # [N]

        # Package as TensorDataset (simplified, no auxiliary loss labels)
        # 【新增部分】Add evidence_token_ids, evidence_masks, tri_label_id, sample_weight_3way
        dataset = TensorDataset(
            content_token_ids,      # [N, L]
            content_masks,         # [N, L]
            support_token_ids,      # [N, L]
            support_masks,          # [N, L]
            oppose_token_ids,       # [N, L]
            oppose_masks,          # [N, L]
            conflict_margin,       # [N] - float tensor
            label,                  # [N]
            id,                     # [N]
            evidence_token_ids,     # [N, L] - evidence text tokens
            evidence_masks,        # [N, L] - evidence text masks
            tri_label_id,          # [N] - 【新增部分】3-way label: 0=SUPPORTED, 1=INSUFFICIENT, 2=CONTRADICTED
            sample_weight_3way,    # [N] - 【新增部分】sample weights for 3-way loss
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )

        return dataloader

    elif data_type == 'conflict_ranking':
        """
        Conflict Ranking mode (Stage 4 v6):
        - Reuses conflict_learning data loading logic
        - Uses margin-based ranking loss instead of MSE loss
        - Same data structure as conflict_learning
        """
        # Reuse conflict_learning logic completely
        return get_dataloader(path, max_len, batch_size, shuffle, bert_path, 'conflict_learning', language, top_k_paths, **kwargs)

    elif data_type == 'conflict_rank_fake_only':
        """
        Conflict Ranking Fake-Only mode:
        - Reuses conflict_learning data loading logic
        - Conflict loss only applies to fake samples (label=1)
        - Uses pairwise ranking loss within batch
        - Same data structure as conflict_learning
        """
        # Reuse conflict_learning logic completely
        return get_dataloader(path, max_len, batch_size, shuffle, bert_path, 'conflict_learning', language, top_k_paths, **kwargs)

    elif data_type == 'kg_paths':
        """
        claim + KG paths setting.
        For each sample and each path_i in selected_paths:
            Input_i = [CLS] content [SEP] path_i [SEP]
        We encode K paths per sample and aggregate later in the model.

        selected_paths format: List[Dict] where each dict contains 'triples' field
        """
        if top_k_paths is None:
            top_k_paths = 3

        def path_dict_to_string(path_dict):
            """
            Convert path dictionary to readable string.

            Args:
                path_dict: Dict with 'triples' field containing list of triple dicts
            Returns:
                str: "head1 -> relation1 -> tail1 ; head2 -> relation2 -> tail2"
            """
            if not isinstance(path_dict, dict):
                return ""

            triples = path_dict.get('triples', [])
            if not triples:
                return ""

            # Convert each triple to "head -> relation -> tail"
            triple_strs = []
            for triple in triples:
                head = triple.get('head', '')
                relation = triple.get('relation', '')
                tail = triple.get('tail', '')
                triple_strs.append(f"{head} -> {relation} -> {tail}")

            # Join triples with " ; "
            return " ; ".join(triple_strs)

        data_list = json.load(open(path, 'r', encoding='utf-8'))
        data_rows = []
        for item in data_list:
            tmp_data = {}
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item['source_id']
            # selected_paths is List[Dict], convert to List[str]
            selected_paths_raw = item.get('selected_paths', []) or []
            tmp_data['selected_paths'] = [path_dict_to_string(p) for p in selected_paths_raw]
            data_rows.append(tmp_data)

        df_data = pd.DataFrame(data_rows)

        contents = df_data['content'].to_list()
        labels = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())
        # Convert string IDs to numeric IDs
        id_to_num = {id_str: idx for idx, id_str in enumerate(df_data['id'].unique())}
        ids = torch.tensor(df_data['id'].apply(lambda x: id_to_num[x]).astype(int).to_numpy())

        num_samples = len(contents)
        K = top_k_paths

        # Encode content alone (for compatibility, even if KG model may not use it)
        content_token_ids, content_masks = word2input(contents, max_len, tokenizer)

        # Tensors for paths: [N, K, L]
        paths_input_ids = torch.zeros((num_samples, K, max_len), dtype=torch.long)
        paths_attention_masks = torch.zeros((num_samples, K, max_len), dtype=torch.long)
        path_mask = torch.zeros((num_samples, K), dtype=torch.long)

        for idx, row in df_data.iterrows():
            content = row['content']
            selected_paths = row['selected_paths'][:K]  # Now these are strings

            # If no paths, keep all-zero mask; model will fallback to mean pooling
            if not selected_paths:
                selected_paths = []

            for j in range(min(len(selected_paths), K)):
                path_str = selected_paths[j] if (selected_paths[j] and selected_paths[j] != "") else ""

                if path_str:  # Only encode if path is not empty
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*overflowing tokens.*')
                        encoded = tokenizer.encode_plus(
                            content,
                            path_str,
                            max_length=max_len,
                            truncation=True,
                            padding='max_length',
                            add_special_tokens=True,
                            return_tensors='pt',
                            verbose=False  # Suppress truncation warnings
                        )
                    paths_input_ids[idx, j] = encoded['input_ids'][0]
                    paths_attention_masks[idx, j] = encoded['attention_mask'][0]
                    path_mask[idx, j] = 1  # real path
                else:
                    # Empty path: use content only
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*overflowing tokens.*')
                        encoded = tokenizer.encode_plus(
                            content,
                            max_length=max_len,
                            truncation=True,
                            padding='max_length',
                            add_special_tokens=True,
                            return_tensors='pt',
                            verbose=False  # Suppress truncation warnings
                        )
                    paths_input_ids[idx, j] = encoded['input_ids'][0]
                    paths_attention_masks[idx, j] = encoded['attention_mask'][0]
                    path_mask[idx, j] = 0  # padded path

        dataset = TensorDataset(
            content_token_ids,      # [N, L]
            content_masks,          # [N, L]
            paths_input_ids,        # [N, K, L]
            paths_attention_masks,  # [N, K, L]
            path_mask,              # [N, K]
            labels,                 # [N]
            ids,                    # [N]
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )
        return dataloader

    elif data_type == 'segment_aware':
        """
        Segment-Aware mode (Stage 4 v9):
        - 基于 CRAVE 的 MultiWindowClassification 架构
        - 根据 gpt_label 决定输入顺序
        - v1 = [Claim] [SEP] r(高置信度)
        - v2 = [Claim] [SEP] r(低置信度)

        数据格式要求：
        - content: claim 文本
        - support_rationale: 支持理由（rationale_true）
        - oppose_rationale: 反对理由（rationale_false）
        - gpt_label: "true" 或 "false"（LLM 的初步判断）
        - label: 0 (real) 或 1 (fake)
        """
        print(f"Loading data for segment_aware mode from {path}")

        data = json.load(open(path, 'r', encoding='utf-8'))
        data_rows = []

        for item in data:
            tmp_data = {}
            tmp_data['content'] = item.get('content', item.get('claim', ''))
            tmp_data['label'] = item.get('label', 0)
            tmp_data['id'] = item.get('id') or item.get('source_id', '')

            # Load rationales
            tmp_data['support_rationale'] = item.get('support_rationale', '')
            tmp_data['oppose_rationale'] = item.get('oppose_rationale', '')

            # Load gpt_label (required for segment ordering)
            tmp_data['gpt_label'] = item.get('gpt_label', 'false')  # 默认 false

            # Validate required fields
            if not tmp_data['content']:
                print(f"Warning: 跳过样本（缺少 content）: {tmp_data['id']}")
                continue
            if not tmp_data['support_rationale'] or not tmp_data['oppose_rationale']:
                print(f"Warning: 跳过样本（缺少 rationale）: {tmp_data['id']}")
                continue

            data_rows.append(tmp_data)

        print(f"  ✅ 加载了 {len(data_rows)} 个样本 (segment_aware)")

        df_data = pd.DataFrame(data_rows)

        content = df_data['content'].to_numpy()
        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c]).astype(int).to_numpy())

        # Convert string IDs to numeric IDs
        id_to_num = {id_str: idx for idx, id_str in enumerate(df_data['id'].unique())}
        id = torch.tensor(df_data['id'].apply(lambda x: id_to_num[x]).astype(int).to_numpy())

        support_rationale = df_data['support_rationale'].to_numpy()
        oppose_rationale = df_data['oppose_rationale'].to_numpy()
        gpt_label = df_data['gpt_label'].to_numpy()

        # 根据 gpt_label 决定输入顺序
        # 如果 gpt_label = "true" → v1=support, v2=oppose（LLM 更倾向 support）
        # 如果 gpt_label = "false" → v1=oppose, v2=support（LLM 更倾向 oppose）
        segment_1_texts = []
        segment_2_texts = []

        for idx, gpt_lbl in enumerate(gpt_label):
            claim = content[idx]
            support_r = support_rationale[idx]
            oppose_r = oppose_rationale[idx]

            # 构造 segment 文本：[Claim] [SEP] rationale
            if gpt_lbl == 'true':
                # LLM 更倾向 support → v1=support, v2=oppose
                segment_1_text = f"{claim} {tokenizer.sep_token} {support_r}"
                segment_2_text = f"{claim} {tokenizer.sep_token} {oppose_r}"
            else:
                # LLM 更倾向 oppose → v1=oppose, v2=support
                segment_1_text = f"{claim} {tokenizer.sep_token} {oppose_r}"
                segment_2_text = f"{claim} {tokenizer.sep_token} {support_r}"

            segment_1_texts.append(segment_1_text)
            segment_2_texts.append(segment_2_text)

        # Tokenize segments
        segment_1_token_ids, segment_1_masks = word2input(segment_1_texts, max_len, tokenizer)
        segment_2_token_ids, segment_2_masks = word2input(segment_2_texts, max_len, tokenizer)

        # Package as TensorDataset
        dataset = TensorDataset(
            segment_1_token_ids,      # [N, L]
            segment_1_masks,          # [N, L]
            segment_2_token_ids,      # [N, L]
            segment_2_masks,          # [N, L]
            label,                    # [N]
            id,                       # [N]
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )

        return dataloader

    elif data_type == 'usefulness':
        """
        TRAIL model data loader
        Load: claim, support_rationale, oppose_rationale, evidence_support, evidence_oppose

        Evidence extraction strategies:
        1. If 'selected_paths' exists: extract triples from paths (head-relation-tail format)
        2. If no 'selected_paths': use rationales directly as evidence (fallback mode)

        This allows the model to work with both CTI_HAL (with KG paths) and
        custom datasets (without KG paths).
        """
        data_list = json.load(open(path, 'r', encoding='utf-8'))
        data_rows = []

        def extract_evidence_from_paths(paths, path_type):
            """Extract and concatenate evidence from paths of a specific type."""
            evidence_texts = []
            for path in paths:
                if path.get('path_type') == path_type:
                    for triple in path.get('triples', []):
                        head = triple.get('head', '')
                        relation = triple.get('relation', '')
                        tail = triple.get('tail', '')
                        evidence_texts.append(f"{head} {relation} {tail}")

            if evidence_texts:
                return '. '.join(evidence_texts)
            else:
                return None  # Return None if no evidence found

        # Check if the first item has selected_paths to determine strategy
        has_selected_paths = 'selected_paths' in data_list[0] if data_list else False

        for item in data_list:
            tmp_data = {}

            # Basic info
            tmp_data['content'] = item['content']
            tmp_data['label'] = item['label']
            tmp_data['id'] = item.get('id', '')

            # Rationale
            support_rationale = item.get('support_rationale', '')
            oppose_rationale = item.get('oppose_rationale', '')
            tmp_data['support_rationale'] = support_rationale
            tmp_data['oppose_rationale'] = oppose_rationale

            # Strategy 1: Extract evidence from selected_paths (if available)
            if has_selected_paths and 'selected_paths' in item:
                selected_paths = item['selected_paths']
                evidence_support = extract_evidence_from_paths(selected_paths, 'support')
                evidence_oppose = extract_evidence_from_paths(selected_paths, 'oppose')

                # Fallback to rationale if no evidence found in paths
                tmp_data['evidence_support'] = evidence_support if evidence_support else support_rationale
                tmp_data['evidence_oppose'] = evidence_oppose if evidence_oppose else oppose_rationale
            else:
                # Strategy 2: Use rationales directly as evidence (no selected_paths)
                tmp_data['evidence_support'] = support_rationale
                tmp_data['evidence_oppose'] = oppose_rationale

            data_rows.append(tmp_data)

        df_data = pd.DataFrame(data_rows)

        # Extract arrays
        content = df_data['content'].to_numpy()
        support_rationale = df_data['support_rationale'].to_numpy()
        oppose_rationale = df_data['oppose_rationale'].to_numpy()
        evidence_support = df_data['evidence_support'].to_numpy()
        evidence_oppose = df_data['evidence_oppose'].to_numpy()

        label = torch.tensor(df_data['label'].apply(lambda c: label_dict[c] if c in label_dict else c).astype(int).to_numpy())

        # Convert string IDs to numeric IDs
        id_to_num = {id_str: idx for idx, id_str in enumerate(df_data['id'].unique())}
        id = torch.tensor(df_data['id'].apply(lambda x: id_to_num.get(x, 0)).astype(int).to_numpy())

        # Tokenize all texts
        content_token_ids, content_masks = word2input(content, max_len, tokenizer)
        support_token_ids, support_masks = word2input(support_rationale, max_len, tokenizer)
        oppose_token_ids, oppose_masks = word2input(oppose_rationale, max_len, tokenizer)
        evidence_support_token_ids, evidence_support_masks = word2input(evidence_support, max_len, tokenizer)
        evidence_oppose_token_ids, evidence_oppose_masks = word2input(evidence_oppose, max_len, tokenizer)

        # Create TensorDataset
        dataset = TensorDataset(
            content_token_ids,               # claim
            content_masks,
            support_token_ids,               # R+
            support_masks,
            oppose_token_ids,                # R-
            oppose_masks,
            evidence_support_token_ids,      # evidence for support
            evidence_support_masks,
            evidence_oppose_token_ids,       # evidence for oppose
            evidence_oppose_masks,
            label,
            id,
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=False,
            shuffle=shuffle
        )

        return dataloader

    else:
        print('No match data type!')
        exit()
