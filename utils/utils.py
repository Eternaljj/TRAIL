from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
from datetime import datetime as dt
from tensorboardX import SummaryWriter
import os

import json
import pandas as pd

class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.max_acc = {'acc': 0}  # Track best ACC separately
        self.cur = {'metric': 0}
        self.curindex = 0
        self.maxindex = 0
        self.maxindex_acc = 0  # Track epoch with best ACC
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("current", self.cur)
        return self.judge()

    def judge(self):
        save_f1 = False
        save_acc = False

        # Check F1 (macro) improvement
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur.copy()
            self.maxindex = self.curindex
            save_f1 = True

        # Check ACC improvement
        if self.cur.get('acc', 0) > self.max_acc.get('acc', 0):
            self.max_acc = self.cur.copy()
            self.maxindex_acc = self.curindex
            save_acc = True

        self.showfinal()

        # Determine return value
        if save_f1 and save_acc:
            return 'save_both'
        elif save_f1:
            return 'save_f1'
        elif save_acc:
            return 'save_acc'

        # Early stopping based on F1 (original behavior)
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max F1", self.max)
        print("Max ACC", self.max_acc)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def metrics(y_true, y_pred):
    all_metrics = {}

    try:
        all_metrics['auc'] = roc_auc_score(y_true, y_pred, average='macro')
    except ValueError:
        all_metrics['auc'] = -1
    try:
        all_metrics['spauc'] = roc_auc_score(y_true, y_pred, average='macro', max_fpr=0.1)
    except ValueError:
        all_metrics['spauc'] = -1
    y_pred = np.around(np.array(y_pred)).astype(int)
    all_metrics['metric'] = f1_score(y_true, y_pred, average='macro')
    try:
        all_metrics['f1_real'], all_metrics['f1_fake'] = f1_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['f1_real'], all_metrics['f1_fake'] = -1, -1
    all_metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    try:
        all_metrics['recall_real'], all_metrics['recall_fake'] = recall_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['recall_real'], all_metrics['recall_fake'] = -1, -1
    all_metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    try:
        all_metrics['precision_real'], all_metrics['precision_fake'] = precision_score(y_true, y_pred, average=None)
    except ValueError:
        all_metrics['precision_real'], all_metrics['precision_fake']= -1, -1
    all_metrics['acc'] = accuracy_score(y_true, y_pred)

    return all_metrics

def data2gpu(batch, use_cuda, data_type):
    if use_cuda:
        if data_type == 'rationale':
            batch_data = {
                'content': batch[0].cuda(),
                'content_masks': batch[1].cuda(),
                'FTR_2_pred': batch[2].cuda(),
                'FTR_2_acc': batch[3].cuda(),
                'FTR_3_pred': batch[4].cuda(),
                'FTR_3_acc': batch[5].cuda(),
                'FTR_2': batch[6].cuda(),
                'FTR_2_masks': batch[7].cuda(),
                'FTR_3': batch[8].cuda(),
                'FTR_3_masks': batch[9].cuda(),
                'label': batch[10].cuda(),
                'id': batch[11].cuda(),
            }
        elif data_type == 'content_only':
            # 只使用content，不使用rationale
            batch_data = {
                'content': batch[0].cuda(),
                'content_masks': batch[1].cuda(),
                'label': batch[2].cuda(),
                'id': batch[3].cuda(),
            }
        elif data_type == 'conflict_rationale':
            # Stage 4: conflict-aware KG-grounded rationale (reuse rationale format)
            batch_data = {
                'content': batch[0].cuda(),
                'content_masks': batch[1].cuda(),
                'FTR_2_pred': batch[2].cuda(),      # dummy for LLM judgment predictor
                'FTR_2_acc': batch[3].cuda(),       # dummy for rationale usefulness evaluator
                'FTR_3_pred': batch[4].cuda(),      # dummy for LLM judgment predictor
                'FTR_3_acc': batch[5].cuda(),       # dummy for rationale usefulness evaluator
                'FTR_2': batch[6].cuda(),           # conflict_rationale
                'FTR_2_masks': batch[7].cuda(),
                'FTR_3': batch[8].cuda(),           # content (self-interaction)
                'FTR_3_masks': batch[9].cuda(),
                'label': batch[10].cuda(),
                'id': batch[11].cuda(),
            }
            # Add sample weights if present (for loss weighting)
            if len(batch) > 12:
                batch_data['sample_weights'] = batch[12].cuda()
        elif data_type == 'dual_rationale':
            # Stage 4 v3: Dual Rationale (Support + Oppose) with Confidence Scores
            batch_data = {
                'content': batch[0].cuda(),
                'content_masks': batch[1].cuda(),
                'FTR_2_pred': batch[2].cuda(),      # dummy for LLM judgment predictor
                'FTR_2_acc': batch[3].cuda(),       # dummy for rationale usefulness evaluator
                'FTR_3_pred': batch[4].cuda(),      # dummy for LLM judgment predictor
                'FTR_3_acc': batch[5].cuda(),       # dummy for rationale usefulness evaluator
                'FTR_2': batch[6].cuda(),           # support_rationale
                'FTR_2_masks': batch[7].cuda(),
                'FTR_3': batch[8].cuda(),           # oppose_rationale
                'FTR_3_masks': batch[9].cuda(),
                'label': batch[10].cuda(),
                'id': batch[11].cuda(),
                'support_confidence': batch[12].cuda(),  # [B] - confidence scores for support
                'oppose_confidence': batch[13].cuda(),   # [B] - confidence scores for oppose
                'conflict_score': batch[14].cuda(),      # [B] - conflict_score = oppose_conf - support_conf
            }
        elif data_type == 'conflict_learning' or data_type == 'conflict_ranking' or data_type == 'conflict_rank_fake_only':
            # Stage 4 v5/v6/v7: Conflict Learning/Ranking/Fake-Only (Adversarial Paradigm)
            batch_data = {
                'content': batch[0].cuda(),
                'content_masks': batch[1].cuda(),
                'FTR_2': batch[2].cuda(),           # support_rationale
                'FTR_2_masks': batch[3].cuda(),
                'FTR_3': batch[4].cuda(),           # oppose_rationale
                'FTR_3_masks': batch[5].cuda(),
                'conflict_margin': batch[6].cuda(), # [B] - conflict_margin = oppose_conf - support_conf
                'label': batch[7].cuda(),
                'id': batch[8].cuda(),
            }
            # 【新增部分】Add evidence_text if present (for new data format)
            if len(batch) > 9:
                batch_data['evidence_text'] = batch[9].cuda()      # evidence_token_ids
                batch_data['evidence_masks'] = batch[10].cuda()   # evidence_masks
            # 【新增部分】Add 3-way classification labels and sample weights (for conflict_5.json)
            if len(batch) > 11:
                batch_data['tri_label_id'] = batch[11].cuda()      # [B] - 3-way label: 0=SUPPORTED, 1=INSUFFICIENT, 2=CONTRADICTED
                batch_data['sample_weight_3way'] = batch[12].cuda()  # [B] - sample weights for 3-way loss
        elif data_type == 'kg_paths':
            # claim + KG paths：content + K paths
            batch_data = {
                'content': batch[0].cuda(),                 # [B, L]
                'content_masks': batch[1].cuda(),           # [B, L]
                'paths_input_ids': batch[2].cuda(),         # [B, K, L]
                'paths_attention_masks': batch[3].cuda(),   # [B, K, L]
                'path_mask': batch[4].cuda(),               # [B, K]
                'label': batch[5].cuda(),                   # [B]
                'id': batch[6].cuda(),                      # [B]
            }
        elif data_type == 'segment_aware':
            # Stage 4 v9: Segment-Aware (CRAVE-based)
            batch_data = {
                'segment_1': batch[0].cuda(),               # [B, L] - segment 1 input_ids
                'segment_1_masks': batch[1].cuda(),         # [B, L] - segment 1 attention_mask
                'segment_2': batch[2].cuda(),               # [B, L] - segment 2 input_ids
                'segment_2_masks': batch[3].cuda(),         # [B, L] - segment 2 attention_mask
                'label': batch[4].cuda(),                   # [B] - label (0=real, 1=fake)
                'id': batch[5].cuda(),                      # [B] - id
            }
        elif data_type == 'usefulness':
            # TRAIL model: Usefulness-Aware Rationale Evaluation
            batch_data = {
                'content_ids': batch[0].cuda(),                    # [B, L] - claim input_ids
                'content_masks': batch[1].cuda(),                  # [B, L] - claim attention_mask
                'support_ids': batch[2].cuda(),                    # [B, L] - support_rationale input_ids
                'support_masks': batch[3].cuda(),                  # [B, L] - support_rationale attention_mask
                'oppose_ids': batch[4].cuda(),                     # [B, L] - oppose_rationale input_ids
                'oppose_masks': batch[5].cuda(),                   # [B, L] - oppose_rationale attention_mask
                'evidence_support_ids': batch[6].cuda(),           # [B, L] - evidence_support input_ids
                'evidence_support_masks': batch[7].cuda(),         # [B, L] - evidence_support attention_mask
                'evidence_oppose_ids': batch[8].cuda(),            # [B, L] - evidence_oppose input_ids
                'evidence_oppose_masks': batch[9].cuda(),          # [B, L] - evidence_oppose attention_mask
                'label': batch[10].cuda(),                         # [B] - label (0=real, 1=fake)
                'id': batch[11].cuda(),                            # [B] - id
            }
        else:
            print('error data type!')
            exit()
    else:
        if data_type == 'rationale':
            batch_data = {
                'content': batch[0],
                'content_masks': batch[1],
                'FTR_2_pred': batch[2],
                'FTR_2_acc': batch[3],
                'FTR_3_pred': batch[4],
                'FTR_3_acc': batch[5],
                'FTR_2': batch[6],
                'FTR_2_masks': batch[7],
                'FTR_3': batch[8],
                'FTR_3_masks': batch[9],
                'label': batch[10],
                'id': batch[11],
            }
        elif data_type == 'content_only':
            # 只使用content，不使用rationale
            batch_data = {
                'content': batch[0],
                'content_masks': batch[1],
                'label': batch[2],
                'id': batch[3],
            }
        elif data_type == 'conflict_rationale':
            # Stage 4: conflict-aware KG-grounded rationale (reuse rationale format)
            batch_data = {
                'content': batch[0],
                'content_masks': batch[1],
                'FTR_2_pred': batch[2],      # dummy for LLM judgment predictor
                'FTR_2_acc': batch[3],       # dummy for rationale usefulness evaluator
                'FTR_3_pred': batch[4],      # dummy for LLM judgment predictor
                'FTR_3_acc': batch[5],       # dummy for rationale usefulness evaluator
                'FTR_2': batch[6],           # conflict_rationale
                'FTR_2_masks': batch[7],
                'FTR_3': batch[8],           # content (self-interaction)
                'FTR_3_masks': batch[9],
                'label': batch[10],
                'id': batch[11],
            }
        elif data_type == 'dual_rationale':
            # Stage 4 v3: Dual Rationale (Support + Oppose) with Confidence Scores
            batch_data = {
                'content': batch[0],
                'content_masks': batch[1],
                'FTR_2_pred': batch[2],      # dummy for LLM judgment predictor
                'FTR_2_acc': batch[3],       # dummy for rationale usefulness evaluator
                'FTR_3_pred': batch[4],      # dummy for LLM judgment predictor
                'FTR_3_acc': batch[5],       # dummy for rationale usefulness evaluator
                'FTR_2': batch[6],           # support_rationale
                'FTR_2_masks': batch[7],
                'FTR_3': batch[8],           # oppose_rationale
                'FTR_3_masks': batch[9],
                'label': batch[10],
                'id': batch[11],
                'support_confidence': batch[12],  # [B] - confidence scores for support
                'oppose_confidence': batch[13],   # [B] - confidence scores for oppose
                'conflict_score': batch[14],      # [B] - conflict_score = oppose_conf - support_conf
            }
        elif data_type == 'conflict_learning' or data_type == 'conflict_ranking' or data_type == 'conflict_rank_fake_only':
            # Stage 4 v5/v6/v7: Conflict Learning/Ranking/Fake-Only (Adversarial Paradigm)
            batch_data = {
                'content': batch[0],
                'content_masks': batch[1],
                'FTR_2': batch[2],           # support_rationale
                'FTR_2_masks': batch[3],
                'FTR_3': batch[4],           # oppose_rationale
                'FTR_3_masks': batch[5],
                'conflict_margin': batch[6], # [B] - conflict_margin = oppose_conf - support_conf
                'label': batch[7],
                'id': batch[8],
            }
            # 【新增部分】Add evidence_text if present (for new data format)
            if len(batch) > 9:
                batch_data['evidence_text'] = batch[9]      # evidence_token_ids
                batch_data['evidence_masks'] = batch[10]   # evidence_masks
            # 【新增部分】Add 3-way classification labels and sample weights (for conflict_5.json)
            if len(batch) > 11:
                batch_data['tri_label_id'] = batch[11]      # [B] - 3-way label: 0=SUPPORTED, 1=INSUFFICIENT, 2=CONTRADICTED
                batch_data['sample_weight_3way'] = batch[12]  # [B] - sample weights for 3-way loss
        elif data_type == 'kg_paths':
            # claim + KG paths：content + K paths
            batch_data = {
                'content': batch[0],                 # [B, L]
                'content_masks': batch[1],           # [B, L]
                'paths_input_ids': batch[2],         # [B, K, L]
                'paths_attention_masks': batch[3],   # [B, K, L]
                'path_mask': batch[4],               # [B, K]
                'label': batch[5],                   # [B]
                'id': batch[6],                      # [B]
            }
        elif data_type == 'segment_aware':
            # Stage 4 v9: Segment-Aware (CRAVE-based)
            batch_data = {
                'segment_1': batch[0],               # [B, L] - segment 1 input_ids
                'segment_1_masks': batch[1],         # [B, L] - segment 1 attention_mask
                'segment_2': batch[2],               # [B, L] - segment 2 input_ids
                'segment_2_masks': batch[3],         # [B, L] - segment 2 attention_mask
                'label': batch[4],                   # [B] - label (0=real, 1=fake)
                'id': batch[5],                      # [B] - id
            }
        elif data_type == 'usefulness':
            # TRAIL model: Usefulness-Aware Rationale Evaluation
            batch_data = {
                'content_ids': batch[0],                    # [B, L] - claim input_ids
                'content_masks': batch[1],                  # [B, L] - claim attention_mask
                'support_ids': batch[2],                    # [B, L] - support_rationale input_ids
                'support_masks': batch[3],                  # [B, L] - support_rationale attention_mask
                'oppose_ids': batch[4],                     # [B, L] - oppose_rationale input_ids
                'oppose_masks': batch[5],                   # [B, L] - oppose_rationale attention_mask
                'evidence_support_ids': batch[6],           # [B, L] - evidence_support input_ids
                'evidence_support_masks': batch[7],         # [B, L] - evidence_support attention_mask
                'evidence_oppose_ids': batch[8],            # [B, L] - evidence_oppose input_ids
                'evidence_oppose_masks': batch[9],          # [B, L] - evidence_oppose attention_mask
                'label': batch[10],                         # [B] - label (0=real, 1=fake)
                'id': batch[11],                            # [B] - id
            }
        else:
            print('error data type!')
            exit()
    return batch_data

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def get_monthly_path(data_type, root_path, month, data_name, conflict_version=None, kg_paths_version=None):
    """
    Get the full path to data file based on data_type.

    Args:
        data_type: Type of data (e.g., 'conflict_ranking', 'kg_paths')
        root_path: Root directory for data files
        month: Month number (usually 1)
        data_name: Base name like 'train.json', 'val.json', 'test.json'
        conflict_version: Optional version number for conflict datasets (e.g., 6 for conflict_6.json)
                         If None, uses automatic fallback logic
        kg_paths_version: Optional version string for kg_paths datasets (e.g., 'v2' for *_with_paths_v2.json)
                         If None or empty, uses default *_with_paths.json

    For kg_paths: uses *_with_paths.json files (or *_with_paths_{version}.json if version specified)
    For conflict_rationale: uses *_conflict.json files
    For content_only/rationale: uses standard *.json files
    """
    if data_type == 'kg_paths':
        # For KG paths, use *_with_paths.json files (or versioned if specified)
        # data_name is like 'train.json', we need to convert to 'train_with_paths.json'
        base_name = data_name.replace('.json', '')
        if kg_paths_version and kg_paths_version.strip():
            file_name = f'{base_name}_with_paths_{kg_paths_version.strip()}.json'
        else:
            file_name = f'{base_name}_with_paths.json'
        file_path = os.path.join(root_path, file_name)
        return file_path
    elif data_type == 'conflict_rationale':
        # For conflict_rationale, prioritize *_conflict_2.json files (new format)
        # data_name is like 'train.json', we need to convert to 'train_conflict_2.json'
        # Or if data_name contains '_conflict_2', use it directly
        base_name = data_name.replace('.json', '')
        if '_conflict_2' in base_name:
            # Already has _conflict_2 suffix, use as is
            file_name = f'{base_name}.json'
        elif '_conflict' in base_name:
            # Already has _conflict suffix, use as is
            file_name = f'{base_name}.json'
        else:
            # Default: prioritize _conflict_2.json (new format), fallback to _conflict.json
            # First try _conflict_2.json
            file_name_2 = f'{base_name}_conflict_2.json'
            file_path_2 = os.path.join(root_path, file_name_2)
            if os.path.exists(file_path_2):
                return file_path_2
            # Fallback to _conflict.json (old format)
            file_name = f'{base_name}_conflict.json'
        file_path = os.path.join(root_path, file_name)
        return file_path
    elif data_type == 'dual_rationale':
        # For dual_rationale, use *_conflict_3.json files
        # data_name is like 'train.json', we need to convert to 'train_conflict_3.json'
        base_name = data_name.replace('.json', '')
        if '_conflict_3' in base_name:
            # Already has _conflict_3 suffix, use as is
            file_name = f'{base_name}.json'
        elif '_conflict' in base_name:
            # Already has _conflict suffix, use as is
            file_name = f'{base_name}.json'
        else:
            # Default: use _conflict_3.json
            file_name = f'{base_name}_conflict_3.json'
        file_path = os.path.join(root_path, file_name)
        return file_path
    elif data_type == 'conflict_learning' or data_type == 'conflict_ranking' or data_type == 'conflict_rank_fake_only':
        # For conflict_learning/conflict_ranking/conflict_rank_fake_only
        # data_name is like 'train.json', we need to convert to 'train_conflict_X.json'
        base_name = data_name.replace('.json', '')

        # If data_name already contains _conflict suffix, use as is
        if '_conflict' in base_name:
            file_name = f'{base_name}.json'
            file_path = os.path.join(root_path, file_name)
            return file_path

        # If conflict_version is specified, use it directly
        if conflict_version is not None:
            file_name = f'{base_name}_conflict_{conflict_version}.json'
            file_path = os.path.join(root_path, file_name)
            if os.path.exists(file_path):
                return file_path
            else:
                # If specified version doesn't exist, warn and fallback
                print(f"⚠️  警告: 指定的 conflict_version={conflict_version} 文件不存在: {file_path}")
                print(f"   将使用自动回退逻辑...")

        # Automatic fallback: try conflict_6 → conflict_5 → conflict_4
        for version in [6, 5, 4]:
            file_name = f'{base_name}_conflict_{version}.json'
            file_path = os.path.join(root_path, file_name)
            if os.path.exists(file_path):
                return file_path

        # Final fallback: use conflict_4.json even if it doesn't exist
        file_name = f'{base_name}_conflict_4.json'
        file_path = os.path.join(root_path, file_name)
        return file_path
    elif data_type == 'segment_aware':
        # For segment_aware, use *_conflict_7.json files (with gpt_label field)
        # data_name is like 'train.json', we need to convert to 'train_conflict_7.json'
        # Try conflict_7 first, then fallback to conflict_6 for backward compatibility
        base_name = data_name.replace('.json', '')
        if '_conflict_7' in base_name or '_conflict_6' in base_name:
            # Already has _conflict_7 or _conflict_6 suffix, use as is
            file_name = f'{base_name}.json'
        elif '_conflict' in base_name:
            # Already has _conflict suffix, use as is
            file_name = f'{base_name}.json'
        else:
            # Try conflict_7 first, then fallback to conflict_6
            file_name_7 = f'{base_name}_conflict_7.json'
            file_path_7 = os.path.join(root_path, file_name_7)
            if os.path.exists(file_path_7):
                return file_path_7
            # Fallback to conflict_6 for backward compatibility
            file_name = f'{base_name}_conflict_6.json'
        file_path = os.path.join(root_path, file_name)
        return file_path
    elif data_type == 'usefulness':
        return os.path.join(root_path, data_name)
    elif data_type in ['rationale', 'content_only']:
        # For content_only and rationale, use standard *.json files
        file_path = os.path.join(root_path, data_name)
        return file_path
    else:
        print('No match data type!')
        exit()

def get_tensorboard_writer(config):
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(dt.now())
    writer_dir = os.path.join(config['tensorboard_dir'], config['model_name'] + '_' + config['data_name'], TIMESTAMP)
    writer = SummaryWriter(logdir=writer_dir, flush_secs=5)
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    return writer

def process_test_results(test_file_path, test_res_path, label, pred, id, ae, acc):
    test_result = []
    test_df = pd.read_json(test_file_path)
    for index in range(len(label)):
        cur_res = {}
        cur_id = id[index]
        cur_data = test_df[test_df['id'] == int(cur_id)].iloc[0]
        for (key, val) in cur_data.iteritems():
            cur_res[key] = val
        cur_res['pred'] = pred[index]
        cur_res['ae'] = ae[index]
        cur_res['acc'] = acc[index]

        test_result.append(cur_res)

    json_str = json.dumps(test_result, indent=4, ensure_ascii=False, cls=NpEncoder)

    with open(test_res_path, 'w') as f:
        f.write(json_str)
    return
