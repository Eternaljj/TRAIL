"""
TRAIL: Tunable Rationale-Aware Interactive Learning
基于 usefulness evaluator 和 adaptive reweighting 的模型架构

模型架构：
1. 编码器 Enc (式4): RoBERTa-base 编码 claim, R+, R-
2. 双专家交互 (式5): CrossAttn(claim, R+) 和 CrossAttn(claim, R-)
3. Evidence 编码 (式6): 编码 evidence_support 和 evidence_oppose
4. 一致性打分 (式6): Score([e_s, p_s]) 和 Score([e_o, p_o])
5. Usefulness evaluator (式7-10): 基于一致性差异生成 hard label
6. Adaptive reweighting (式11-12): 动态调整 expert 权重
7. Explicit conflict features (式13): h_delta 和 h_abs
8. Final prediction (式14-16): 聚合所有特征进行分类
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time
from .layers import *
from sklearn.metrics import *
from transformers import AutoModel
from utils.utils import data2gpu, Averager, metrics, Recorder
from utils.dataloader import get_dataloader
from utils.utils import get_monthly_path, get_tensorboard_writer, process_test_results


class TRAILModel(nn.Module):
    """
    TRAIL: Tunable Rationale-Aware Interactive Learning
    """
    def __init__(self, config):
        super(TRAILModel, self).__init__()

        self.config = config
        self.emb_dim = config['emb_dim']

        # ============================================================
        # 1. 编码器 Enc (式4): 使用 AutoModel 自动选择 RoBERTa/BERT
        # ============================================================
        bert_path = config['bert_path']
        self.bert = AutoModel.from_pretrained(bert_path)

        # Fine-tune 最后一层
        for name, param in self.bert.named_parameters():
            if name.startswith("encoder.layer.11"):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # ============================================================
        # 2. 双专家交互表示 (式5): CrossAttention + Pooling
        # ============================================================
        # 使用 MultiHeadedAttention 实现 CrossAttention
        self.cross_attn_support = MultiHeadedAttention(
            h=8,  # 8 attention heads
            d_model=self.emb_dim,
            dropout=0.1
        )
        self.cross_attn_oppose = MultiHeadedAttention(
            h=8,
            d_model=self.emb_dim,
            dropout=0.1
        )

        # Pooling method: 'mean' or 'attention'
        self.pooling_method = config.get('pooling_method', 'mean')
        if self.pooling_method == 'attention':
            self.pool_support = MaskAttention(self.emb_dim)
            self.pool_oppose = MaskAttention(self.emb_dim)

        # ============================================================
        # 3. Evidence 编码: 复用 self.bert
        # ============================================================
        # Evidence 也使用同一个 BERT 编码器

        # ============================================================
        # 4. 一致性打分 Score (式6)
        # ============================================================
        # Score([a,b]) = MLP([a; b; |a-b|; a*b]) -> scalar
        # Input: concat(a, b, |a-b|, a*b) = 4 * emb_dim
        self.score_mlp = nn.Sequential(
            nn.Linear(4 * self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.emb_dim, 1),
            nn.Sigmoid()  # Output in (0,1)
        )

        # ============================================================
        # 5. Usefulness evaluator (式7-10)
        # ============================================================
        # Margin for hard label generation
        self.usefulness_margin = config.get('usefulness_margin', 0.1)

        # Usefulness predictor: MLP(e_s) -> u_hat_plus, MLP(e_o) -> u_hat_minus
        self.mlp_usefulness = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.emb_dim // 2, 1),
            nn.Sigmoid()  # Output in (0,1)
        )

        # Lambda for usefulness loss
        self.lambda_use = config.get('lambda_use', 1.0)

        # ============================================================
        # 6. Adaptive reweighting (式11-12)
        # ============================================================
        # MLP_w: [e_s; e_o] -> [alpha_plus, alpha_minus] (bidirectional softmax)
        self.mlp_weight = nn.Sequential(
            nn.Linear(2 * self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.emb_dim, 2)  # Output 2 logits for softmax
        )

        # ============================================================
        # 7. Explicit conflict features (式13)
        # ============================================================
        self.enable_conflict_features = config.get('enable_conflict_features', True)

        # ============================================================
        # 8. Final prediction (式14-16)
        # ============================================================
        # Aggregator: attention-based pooling over [h_claim, e_s', e_o']
        self.aggregator = MaskAttention(self.emb_dim)

        # MLP classifier input dimension
        # z = [z0; h_delta; h_abs] if enable_conflict_features
        # z = [z0] otherwise
        if self.enable_conflict_features:
            mlp_input_dim = self.emb_dim + 2 * self.emb_dim  # z0 + h_delta + h_abs
        else:
            mlp_input_dim = self.emb_dim  # only z0

        # Final classifier MLP
        self.classifier = MLP(
            input_dim=mlp_input_dim,
            embed_dims=config['model']['mlp']['dims'],
            dropout=config['model']['mlp']['dropout'],
            output_layer=True  # Output 1 dimension for binary classification
        )

    def encode_text(self, input_ids, attention_mask):
        """
        编码文本，返回 token-level 和 sentence-level 表示

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            token_states: (batch_size, seq_len, emb_dim)
            pooled_vec: (batch_size, emb_dim)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_states = outputs.last_hidden_state  # (batch, seq_len, emb_dim)

        # Pooling: CLS token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_vec = outputs.pooler_output  # (batch, emb_dim)
        else:
            # Mean pooling with attention mask
            pooled_vec = (token_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)

        return token_states, pooled_vec

    def pool_cross_attn_output(self, cross_attn_output, mask, pool_layer=None):
        """
        对 cross-attention 输出进行 pooling

        Args:
            cross_attn_output: (batch, seq_len, emb_dim)
            mask: (batch, seq_len)
            pool_layer: MaskAttention layer (如果使用 attention pooling)

        Returns:
            pooled: (batch, emb_dim)
        """
        if self.pooling_method == 'mean':
            # Mean pooling
            pooled = (cross_attn_output * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        elif self.pooling_method == 'attention':
            # Attention pooling
            pooled, _ = pool_layer(cross_attn_output, mask)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        return pooled

    def compute_score(self, a, b):
        """
        计算一致性打分 Score([a, b]) (式6)

        Args:
            a: (batch, emb_dim)
            b: (batch, emb_dim)

        Returns:
            score: (batch,)
        """
        # Concatenate features: [a; b; |a-b|; a*b]
        features = torch.cat([
            a,
            b,
            torch.abs(a - b),
            a * b
        ], dim=1)  # (batch, 4*emb_dim)

        score = self.score_mlp(features).squeeze(1)  # (batch,)
        return score

    def compute_usefulness_hard_labels(self, s_s, s_o):
        """
        计算 usefulness hard labels (式7)

        Args:
            s_s: (batch,) support consistency score
            s_o: (batch,) oppose consistency score

        Returns:
            u_plus: (batch,) hard label for support usefulness
            u_minus: (batch,) hard label for oppose usefulness
        """
        d = s_s - s_o  # (batch,)
        m = self.usefulness_margin

        # u_plus = I[d > m]
        u_plus = (d > m).float()

        # u_minus = I[d < -m]
        u_minus = (d < -m).float()

        # Neutral-margin case: set both labels to 0 when |d| <= m.

        return u_plus, u_minus

    def forward(self, data):
        """
        前向传播

        Args:
            data: dict containing:
                - content_ids, content_masks: claim
                - support_ids, support_masks: R+ (support rationale)
                - oppose_ids, oppose_masks: R- (oppose rationale)
                - evidence_support_ids, evidence_support_masks: evidence for support
                - evidence_oppose_ids, evidence_oppose_masks: evidence for oppose
                - label: (batch,) ground truth label (0/1)

        Returns:
            dict containing:
                - logits: (batch,) final prediction logits
                - loss: total loss
                - loss_label: BCE loss for label prediction
                - loss_use: BCE loss for usefulness prediction
                - (optional) other intermediate values for debugging
        """
        # ========== 1. 编码 claim, R+, R- (式4) ==========
        h_claim_tokens, h_claim = self.encode_text(
            data['content_ids'], data['content_masks']
        )  # (batch, seq_claim, emb), (batch, emb)

        h_s_tokens, h_s = self.encode_text(
            data['support_ids'], data['support_masks']
        )  # (batch, seq_s, emb), (batch, emb)

        h_o_tokens, h_o = self.encode_text(
            data['oppose_ids'], data['oppose_masks']
        )  # (batch, seq_o, emb), (batch, emb)

        # ========== 2. 双专家交互表示 (式5) ==========
        # e_s = Pool(CrossAttn(h_claim_tokens, h_s_tokens))
        cross_s, _ = self.cross_attn_support(
            query=h_claim_tokens,
            key=h_s_tokens,
            value=h_s_tokens,
            mask=None  # TODO: 添加 mask 支持（如果需要）
        )  # (batch, seq_claim, emb)

        e_s = self.pool_cross_attn_output(
            cross_s,
            data['content_masks'],
            self.pool_support if self.pooling_method == 'attention' else None
        )  # (batch, emb)

        # e_o = Pool(CrossAttn(h_claim_tokens, h_o_tokens))
        cross_o, _ = self.cross_attn_oppose(
            query=h_claim_tokens,
            key=h_o_tokens,
            value=h_o_tokens,
            mask=None
        )  # (batch, seq_claim, emb)

        e_o = self.pool_cross_attn_output(
            cross_o,
            data['content_masks'],
            self.pool_oppose if self.pooling_method == 'attention' else None
        )  # (batch, emb)

        # ========== 3. Evidence 编码 (用于式6) ==========
        _, p_s = self.encode_text(
            data['evidence_support_ids'], data['evidence_support_masks']
        )  # (batch, emb)

        _, p_o = self.encode_text(
            data['evidence_oppose_ids'], data['evidence_oppose_masks']
        )  # (batch, emb)

        # ========== 4. 一致性打分 (式6) ==========
        s_s = self.compute_score(e_s, p_s)  # (batch,)
        s_o = self.compute_score(e_o, p_o)  # (batch,)

        # ========== 5. Usefulness hard labels 和 predictor (式7-10) ==========
        u_plus, u_minus = self.compute_usefulness_hard_labels(s_s, s_o)

        # Usefulness prediction
        u_hat_plus = self.mlp_usefulness(e_s).squeeze(1)  # (batch,)
        u_hat_minus = self.mlp_usefulness(e_o).squeeze(1)  # (batch,)

        # Usefulness loss (式10)
        loss_use_plus = F.binary_cross_entropy(u_hat_plus, u_plus)
        loss_use_minus = F.binary_cross_entropy(u_hat_minus, u_minus)
        loss_use = loss_use_plus + loss_use_minus

        # ========== 6. Adaptive reweighting (式11-12) ==========
        # MLP_w([e_s; e_o]) -> [logit_plus, logit_minus]
        weight_logits = self.mlp_weight(torch.cat([e_s, e_o], dim=1))  # (batch, 2)

        # Bidirectional softmax
        alphas = F.softmax(weight_logits, dim=1)  # (batch, 2)
        alpha_plus = alphas[:, 0:1]  # (batch, 1)
        alpha_minus = alphas[:, 1:2]  # (batch, 1)

        # Reweight experts
        e_s_prime = alpha_plus * e_s  # (batch, emb)
        e_o_prime = alpha_minus * e_o  # (batch, emb)

        # ========== 7. Explicit conflict features (式13) ==========
        h_delta = e_s_prime - e_o_prime  # (batch, emb)
        h_abs = torch.abs(h_delta)  # (batch, emb)

        # ========== 8. Final prediction (式14-16) ==========
        # Agg(h_claim, e_s', e_o') -> z0
        # 使用 attention-based aggregation
        stacked = torch.stack([h_claim, e_s_prime, e_o_prime], dim=1)  # (batch, 3, emb)
        mask = torch.ones(stacked.size(0), stacked.size(1), device=stacked.device)  # (batch, 3)
        z0, _ = self.aggregator(stacked, mask)  # (batch, emb)

        # Concatenate features
        if self.enable_conflict_features:
            z = torch.cat([z0, h_delta, h_abs], dim=1)  # (batch, 3*emb)
        else:
            z = z0  # (batch, emb)

        # Final classifier
        logits = self.classifier(z).squeeze(1)  # (batch,)

        # ========== 9. 计算损失 (式16) ==========
        y_hat = torch.sigmoid(logits)
        y = data['label'].float()

        loss_label = F.binary_cross_entropy(y_hat, y)

        # Total loss
        loss = loss_label + self.lambda_use * loss_use

        return {
            'logits': logits,
            'y_hat': y_hat,
            'loss': loss,
            'loss_label': loss_label,
            'loss_use': loss_use,
            # 中间值（用于调试和分析）
            'e_s': e_s,               # claim×support 交互向量 (batch, emb)
            'e_o': e_o,               # claim×oppose  交互向量 (batch, emb)
            'h_claim': h_claim,       # claim CLS 表示 (batch, emb)
            's_s': s_s,
            's_o': s_o,
            'u_plus': u_plus,
            'u_minus': u_minus,
            'u_hat_plus': u_hat_plus,
            'u_hat_minus': u_hat_minus,
            'alpha_plus': alpha_plus.squeeze(1),
            'alpha_minus': alpha_minus.squeeze(1)
        }


class Trainer():
    """
    TRAIL Trainer
    """
    def __init__(self, config, writer=None):
        self.config = config
        self.writer = writer

        # 创建保存目录
        self.save_param_dir = os.path.join(
            config['save_param_dir'],
            f"TRAIL_{config['data_name']}"
        )
        os.makedirs(self.save_param_dir, exist_ok=True)

        # 创建日志目录
        self.save_log_dir = os.path.join(
            config['save_log_dir'],
            'log'
        )
        os.makedirs(self.save_log_dir, exist_ok=True)

        # 创建测试结果目录
        self.test_result_dir = os.path.join(
            config['save_log_dir'],
            'test',
            f"TRAIL_{config['data_name']}"
        )
        os.makedirs(self.test_result_dir, exist_ok=True)

    def train(self, logger=None):
        """
        训练模型
        """
        # 构建数据路径
        from utils.utils import get_monthly_path
        train_path = get_monthly_path(
            data_type='usefulness',
            root_path=self.config['root_path'],
            month=1,
            data_name='train.json'
        )
        val_path = get_monthly_path(
            data_type='usefulness',
            root_path=self.config['root_path'],
            month=1,
            data_name='val.json'
        )

        # 加载数据（需要特殊的数据加载器来提取 evidence）
        train_loader = get_dataloader(
            path=train_path,
            max_len=self.config['max_len'],
            batch_size=self.config['batchsize'],
            shuffle=True,
            bert_path=self.config['bert_path'],
            data_type='usefulness',
            language=self.config['language']
        )

        val_loader = get_dataloader(
            path=val_path,
            max_len=self.config['max_len'],
            batch_size=self.config['batchsize'],
            shuffle=False,
            bert_path=self.config['bert_path'],
            data_type='usefulness',
            language=self.config['language']
        )

        # 创建模型
        model = TRAILModel(self.config)

        if torch.cuda.is_available():
            model = model.cuda()

        # 优化器
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        # 训练循环
        best_f1 = 0
        best_acc = 0
        best_epoch = 0
        patience = 0
        early_stop = self.config['early_stop']

        for epoch in range(self.config['epoch']):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config['epoch']}")
            print(f"{'='*60}")

            # 训练
            model.train()
            train_loss_label = Averager()
            train_loss_use = Averager()
            train_loss_total = Averager()

            train_bar = tqdm.tqdm(train_loader, desc=f"Training")
            for batch_data in train_bar:
                batch_data = data2gpu(batch_data, self.config['use_cuda'], 'usefulness')

                optimizer.zero_grad()
                outputs = model(batch_data)

                loss = outputs['loss']
                loss.backward()
                optimizer.step()

                train_loss_label.add(outputs['loss_label'].item())
                train_loss_use.add(outputs['loss_use'].item())
                train_loss_total.add(loss.item())

                train_bar.set_postfix({
                    'loss': f"{train_loss_total.item():.4f}",
                    'loss_label': f"{train_loss_label.item():.4f}",
                    'loss_use': f"{train_loss_use.item():.4f}"
                })

            print(f"\n[Train] Loss: {train_loss_total.item():.4f} "
                  f"(Label: {train_loss_label.item():.4f}, Use: {train_loss_use.item():.4f})")

            # 验证
            results = self.test(model, val_loader)

            print(f"\n[Val] Acc: {results['acc']:.4f}, F1: {results['metric']:.4f}, "
                  f"AUC: {results['auc']:.4f}")
            print(f"      Real F1: {results['f1_real']:.4f}, Fake F1: {results['f1_fake']:.4f}")

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('Train/loss_total', train_loss_total.item(), epoch)
                self.writer.add_scalar('Train/loss_label', train_loss_label.item(), epoch)
                self.writer.add_scalar('Train/loss_use', train_loss_use.item(), epoch)
                self.writer.add_scalar('Val/acc', results['acc'], epoch)
                self.writer.add_scalar('Val/f1', results['metric'], epoch)
                self.writer.add_scalar('Val/auc', results['auc'], epoch)

            # 保存最佳模型
            if results['metric'] > best_f1:
                best_f1 = results['metric']
                best_epoch = epoch
                patience = 0

                # 保存 F1 最佳模型
                save_path_f1 = os.path.join(self.save_param_dir, '1', 'parameter_bert.pkl')
                os.makedirs(os.path.dirname(save_path_f1), exist_ok=True)
                torch.save(model.state_dict(), save_path_f1)
                print(f"✅ Saved best F1 model to {save_path_f1}")
            else:
                patience += 1

            if results['acc'] > best_acc:
                best_acc = results['acc']

                # 保存 ACC 最佳模型
                save_path_acc = os.path.join(self.save_param_dir, '1', 'parameter_bert_acc.pkl')
                os.makedirs(os.path.dirname(save_path_acc), exist_ok=True)
                torch.save(model.state_dict(), save_path_acc)
                print(f"✅ Saved best ACC model to {save_path_acc}")

            # Early stopping
            if patience >= early_stop:
                print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                break

        print(f"\n{'='*60}")
        print(f"🎉 Training finished!")
        print(f"   Best F1: {best_f1:.4f} at epoch {best_epoch+1}")
        print(f"   Best ACC: {best_acc:.4f}")
        print(f"{'='*60}\n")

        # 加载最佳模型并在测试集上评估
        test_path = get_monthly_path(
            data_type='usefulness',
            root_path=self.config['root_path'],
            month=1,
            data_name='test.json'
        )
        test_loader = get_dataloader(
            path=test_path,
            max_len=self.config['max_len'],
            batch_size=self.config['batchsize'],
            shuffle=False,
            bert_path=self.config['bert_path'],
            data_type='usefulness',
            language=self.config['language']
        )

        # 测试 F1 最佳模型
        model.load_state_dict(torch.load(os.path.join(self.save_param_dir, '1', 'parameter_bert.pkl')))
        test_results_f1 = self.test(model, test_loader)

        # 测试 ACC 最佳模型（如果存在）
        acc_model_path = os.path.join(self.save_param_dir, '1', 'parameter_bert_acc.pkl')
        test_results_acc = None
        if os.path.exists(acc_model_path):
            model.load_state_dict(torch.load(acc_model_path))
            test_results_acc = self.test(model, test_loader)

        # 保存测试结果
        self.save_test_results(test_results_f1, test_results_acc)

        # 返回测试结果字典（不是单个 metric 值）
        return test_results_f1, os.path.join(self.save_param_dir, '1'), best_epoch + 1

    def test(self, model, dataloader):
        """
        测试模型
        """
        model.eval()

        pred = []
        label = []

        with torch.no_grad():
            for batch_data in tqdm.tqdm(dataloader, desc="Testing"):
                batch_data = data2gpu(batch_data, self.config['use_cuda'], 'usefulness')
                outputs = model(batch_data)

                pred.extend(outputs['y_hat'].cpu().numpy().tolist())
                label.extend(batch_data['label'].cpu().numpy().tolist())

        # 计算指标
        return metrics(label, pred)

    def save_test_results(self, results_f1, results_acc=None):
        """
        保存测试结果
        """
        # 保存 F1 最佳模型结果到日志文件
        # 直接构建日志文件路径，而不使用 get_monthly_path（它用于数据文件）
        monthly_path = os.path.join(self.save_log_dir, f"TRAIL_{self.config['data_name']}", 'month_1.json')
        os.makedirs(os.path.dirname(monthly_path), exist_ok=True)

        with open(monthly_path, 'w', encoding='utf-8') as f:
            json.dump(results_f1, f, ensure_ascii=False, indent=2)

        print(f"\n📊 Test Results (F1 Best Model):")
        print(f"   ACC: {results_f1['acc']:.4f}")
        print(f"   F1:  {results_f1['metric']:.4f}")
        print(f"   AUC: {results_f1['auc']:.4f}")
        print(f"   Real F1: {results_f1['f1_real']:.4f}")
        print(f"   Fake F1: {results_f1['f1_fake']:.4f}")

        # 如果有 ACC 最佳模型结果，保存到测试结果目录
        if results_acc is not None:
            test_res_path = os.path.join(self.test_result_dir, 'month_1.json')
            with open(test_res_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'f1_best': results_f1,
                    'acc_best': results_acc
                }, f, ensure_ascii=False, indent=2)

            print(f"\n📊 Test Results (ACC Best Model):")
            print(f"   ACC: {results_acc['acc']:.4f}")
            print(f"   F1:  {results_acc['metric']:.4f}")
            print(f"   AUC: {results_acc['auc']:.4f}")
