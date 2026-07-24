import os
import argparse
import json
from utils.utils import get_tensorboard_writer

from grid_search import Run
import torch
import numpy as np
import random

def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {'true', '1', 'yes', 'y', 'on'}:
        return True
    if normalized in {'false', '0', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'invalid boolean value: {value}')

# 全局变量，用于存储配置（向后兼容）
args = None
config = None

def build_config_from_args(args_obj):
    """从 argparse.Namespace 对象构建 config 字典"""
    return {
        'use_cuda': True,
        'seed': args_obj.seed,
        'batchsize': args_obj.batchsize,
        'max_len': args_obj.max_len,
        'early_stop': args_obj.early_stop,
        'language': args_obj.language,
        'root_path': args_obj.root_path,
        'weight_decay': args_obj.weight_decay,
        'model': {
            'mlp': {'dims': [384], 'dropout': 0.2},
            'llm_judgment_predictor_weight': args_obj.llm_judgment_predictor_weight,
            'rationale_usefulness_evaluator_weight': args_obj.rationale_usefulness_evaluator_weight,
            'kd_loss_weight': args_obj.kd_loss_weight
        },
        'emb_dim': args_obj.emb_dim,
        'co_attention_dim': args_obj.co_attention_dim,
        'lr': args_obj.lr,
        'epoch': args_obj.epoch,
        'model_name': args_obj.model_name,
        'seed': args_obj.seed,
        'save_log_dir': args_obj.save_log_dir,
        'save_param_dir': args_obj.save_param_dir,
        'param_log_dir': args_obj.param_log_dir,
        'tensorboard_dir': args_obj.tensorboard_dir,
        'bert_path': args_obj.bert_path,
        'data_type': args_obj.data_type,
        'data_name': args_obj.data_name,
        'eval_mode': args_obj.eval_mode,
        'teacher_path': args_obj.teacher_path,
        'expert_weight_2': args_obj.expert_weight_2,
        'expert_weight_3': args_obj.expert_weight_3,
        'use_self_interaction': args_obj.use_self_interaction,
        'filter_insufficient_evidence': args_obj.filter_insufficient_evidence,
        'insufficient_evidence_weight': args_obj.insufficient_evidence_weight,
        'conflict_alpha': args_obj.conflict_alpha,
        'conflict_lambda': args_obj.conflict_lambda,
        'conflict_rank_margin': args_obj.conflict_rank_margin,
        'conflict_threshold': args_obj.conflict_threshold,
        'conflict_fake_tau': args_obj.conflict_fake_tau,
        'conflict_fake_delta': args_obj.conflict_fake_delta,
        'use_weighted_sampler': args_obj.use_weighted_sampler,
        # 【新增部分】Explicit conflict features and evidence loss parameters
        'use_explicit_conflict_features': args_obj.use_explicit_conflict_features,
        'use_evidence_loss': args_obj.use_evidence_loss,
        'evidence_rank_margin': args_obj.evidence_rank_margin,
        'evidence_lambda': args_obj.evidence_lambda,
        # 【新增部分】3-way classification parameters
        'use_3way_classification': args_obj.use_3way_classification,
        'lambda_3way': args_obj.lambda_3way,
        # 【新增部分】Graph Neural Network parameters (v8)
        'use_graph_enhancement': args_obj.use_graph_enhancement,
        'gnn_num_layers': args_obj.gnn_num_layers,
        'gnn_hidden_dim': args_obj.gnn_hidden_dim,
        'gnn_aggr': args_obj.gnn_aggr,
        'graph_residual': args_obj.graph_residual,
        # 【TRAIL 模型参数】
        'usefulness_margin': args_obj.usefulness_margin,
        'lambda_use': args_obj.lambda_use,
        'pooling_method': args_obj.pooling_method,
        'enable_conflict_features': args_obj.enable_conflict_features,
        'month': 1
    }

def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(output_dir: str = None, data_tag: str = None) -> dict:
    """
    供 dqnnew.py 调用的训练入口函数

    Args:
        output_dir: 训练数据目录（包含 train.json, val.json, test.json）
        data_tag: 数据标签（用于日志命名，可选）

    Returns:
        dict: 包含训练指标的字典（f1_real, f1_fake, acc, metric 等）
    """
    global args, config

    # 如果 args 和 config 已经设置（通过 dqnnew.py 的注入机制），直接使用
    if args is None or config is None:
        # 如果没有通过注入设置，尝试从环境变量或默认值构建
        # 这种情况不应该发生，因为 dqnnew.py 会先注入
        raise RuntimeError("main.main() called but args/config not set. Please use dqnnew.py's run_arg_training() instead.")

    # 如果提供了 output_dir，覆盖 config 中的 root_path
    if output_dir is not None:
        config['root_path'] = output_dir
        if config.get('data_name') is None:
            config['data_name'] = os.path.basename(output_dir.rstrip('/'))

    # 设置随机种子
    setup_seed(config['seed'])

    # 设置 GPU
    if 'gpu' in config:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu'])

    print('lr: {}; model name: {}; batchsize: {}; epoch: {}; gpu: {};'.format(
        config['lr'], config['model_name'], config['batchsize'], config['epoch'], config.get('gpu', '0')))
    print('data_type: {}; data_path: {}; data_name: {};'.format(
        config['data_type'], config['root_path'], config['data_name']))

    writer = get_tensorboard_writer(config)
    best_metric = Run(config=config, writer=writer).main()

    return best_metric

if __name__ == '__main__':
    # 只在直接运行 main.py 时解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='ARG')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--max_len', type=int, default=170)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3759)
    parser.add_argument('--gpu', type=str, default='3')
    parser.add_argument('--emb_dim', type=int, default=768)
    parser.add_argument('--co_attention_dim', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--save_log_dir', type=str, default='./logs')
    parser.add_argument('--save_param_dir', type=str, default='./param_model')
    parser.add_argument('--param_log_dir', type=str, default='./logs/param')

    # extra parameter
    parser.add_argument('--tensorboard_dir', type=str, default='./logs/tensorlog')
    parser.add_argument('--bert_path', type=str, default='/path/to/bert-base-uncased')
    parser.add_argument('--data_type', type=str, default='rationale')
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--eval_mode', type=bool, default=False)

    # model structure control
    parser.add_argument('--expert_interaction_method', type=str, default='cross_attention')
    parser.add_argument('--llm_judgment_predictor_weight', type=float, default=-1)
    parser.add_argument('--rationale_usefulness_evaluator_weight', type=float, default=-1)

    # distill config
    parser.add_argument('--kd_loss_weight', type=float, default=1)
    parser.add_argument('--teacher_path', type=str)

    # Expert weighting for conflict-aware training (Stage 4)
    parser.add_argument('--expert_weight_2', type=float, default=1.5,
                        help='Weight for expert_2 (rationale). Higher = trust rationale more. Default: 1.5')
    parser.add_argument('--expert_weight_3', type=float, default=0.5,
                        help='Weight for expert_3 (self-interaction). Lower = reduce content bias. Default: 0.5')
    parser.add_argument('--use_self_interaction', type=bool, default=True,
                        help='Whether to use self-interaction (claim×claim) branch. Set False to remove content bias. Default: True')
    parser.add_argument('--filter_insufficient_evidence', action='store_true',
                        help='Filter out [Conflict=INSUFFICIENT_EVIDENCE] samples during training')
    parser.add_argument('--insufficient_evidence_weight', type=float, default=1.0,
                        help='Weight for INSUFFICIENT_EVIDENCE samples (if not filtered). Lower = reduce influence. Default: 1.0')

    # Conflict Learning parameters (Stage 4 v5)
    parser.add_argument('--conflict_alpha', type=float, default=5.0,
                        help='Alpha parameter for conflict loss (controls transformation sharpness). Default: 5.0')
    parser.add_argument('--conflict_lambda', type=float, default=0.3,
                        help='Lambda parameter for conflict loss weight. Default: 0.3')

    # Conflict Ranking parameters (Stage 4 v6)
    parser.add_argument('--conflict_rank_margin', type=float, default=0.5,
                        help='Margin parameter for conflict ranking loss. Default: 0.5')
    parser.add_argument('--conflict_threshold', type=float, default=0.0,
                        help='Threshold for high-confidence conflict gating. Only samples with abs(conflict_margin) >= threshold use conflict loss. Default: 0.0 (no gating)')

    # Conflict Ranking Fake-Only parameters (Stage 4 v7)
    parser.add_argument('--conflict_fake_tau', type=float, default=0.4,
                        help='Threshold for high-confidence conflict in fake samples. Only fake samples with conflict_margin > tau participate in ranking loss. Default: 0.4')
    parser.add_argument('--conflict_fake_delta', type=float, default=0.2,
                        help='Margin for pairwise ranking loss in fake samples. If margin_i > margin_j, then logit_i > logit_j + delta. Default: 0.2')
    parser.add_argument('--use_weighted_sampler', type=bool, default=True,
                        help='Whether to use WeightedRandomSampler. If False, only use loss weighting. Default: True')

    # 【新增部分】Explicit conflict features and evidence loss parameters
    parser.add_argument('--use_explicit_conflict_features', action='store_true', default=False,
                        help='Enable explicit conflict features (h_diff, h_abs) concatenated to final feature. Default: False')
    parser.add_argument('--use_evidence_loss', action='store_true', default=False,
                        help='Enable evidence-based ranking loss. Requires evidence_text in data. Default: False')
    parser.add_argument('--evidence_rank_margin', type=float, default=0.2,
                        help='Margin for evidence-based ranking loss. Default: 0.2')
    parser.add_argument('--evidence_lambda', type=float, default=0.1,
                        help='Weight for evidence-based ranking loss. Default: 0.1')

    # 【新增部分】3-way classification parameters
    parser.add_argument('--use_3way_classification', action='store_true', default=False,
                        help='Enable 3-way classification head: SUPPORTED, INSUFFICIENT, CONTRADICTED. Default: False')
    parser.add_argument('--lambda_3way', type=float, default=0.3,
                        help='Weight for 3-way classification loss: L = L_binary + lambda_3way * L_3class. Default: 0.3')

    # 【新增部分】Graph Neural Network parameters (v8: Graph-ARG)
    parser.add_argument('--use_graph_enhancement', action='store_true', default=False,
                        help='Enable graph neural network enhancement module. Each sample builds its own local graph. Default: False')
    parser.add_argument('--gnn_num_layers', type=int, default=1,
                        help='Number of GNN layers (1 or 2). Default: 1')
    parser.add_argument('--gnn_hidden_dim', type=int, default=768,
                        help='GNN hidden dimension. Should match emb_dim. Default: 768')
    parser.add_argument('--gnn_aggr', type=str, default='mean',
                        choices=['mean', 'max', 'add'],
                        help='GNN aggregation method: mean, max, or add. Default: mean')
    parser.add_argument('--graph_residual', action='store_true', default=True,
                        help='Use residual connection for graph-enhanced claim: graph_enhanced_claim + attn_content. Default: True')
    parser.add_argument('--no_graph_residual', dest='graph_residual', action='store_false',
                        help='Disable residual connection for graph-enhanced claim')

    # 【TRAIL 模型参数】Usefulness-Aware Rationale Evaluation
    parser.add_argument('--usefulness_margin', type=float, default=0.1,
                        help='Margin for usefulness hard label generation: u_plus = I[d > m], u_minus = I[d < -m]. Default: 0.1')
    parser.add_argument('--lambda_use', type=float, default=1.0,
                        help='Weight for usefulness loss: L = L_label + lambda_use * L_use. Default: 1.0')
    parser.add_argument('--pooling_method', type=str, default='mean',
                        choices=['mean', 'attention'],
                        help='Pooling method for cross-attention output: mean or attention. Default: mean')
    parser.add_argument('--enable_conflict_features', type=str2bool, default=True,
                        help='Enable explicit conflict features (h_delta, h_abs) for TRAIL model. Default: True')

    args = parser.parse_args()
    config = build_config_from_args(args)

    print('before in config')
    print(config)
    best_metric = main()

    save_dir = './logs/log'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, config['data_name']+'.json')
    with open(save_path, 'w') as file:
        json.dump(best_metric, file, indent=4, ensure_ascii=False)
