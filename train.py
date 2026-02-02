#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
読唇術モデル訓練モジュール（ニュートラル版：CTCのみ / SEDなし）
- Pattern B: CNN → LSTM → Temporal Attention を想定
- vowel/consonant 共通: 途中評価も最終評価も CTCデコード + UnifiedEvaluationMetrics を使用
- 数値安定:
  * モデルが log確率 or ロジットを返すかを自動判定（returns_log_probs）
  * 空ラベル/無効長サンプルの除外
  * CTCの長さ整合（input_length のクランプ、target_length の下限1）
  * logp の finite 化
  * 勾配クリップ
"""

import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loss_soft_per import SoftPERLoss
from matrics_undefined import UnifiedEvaluationMetrics

# enhanced_metrics は任意
try:
    from enhanced_metrics import EnhancedEvaluationMetrics
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    print("Warning: enhanced_metrics not found. Using standard metrics only.")
    ENHANCED_METRICS_AVAILABLE = False

from utils_pattern_b import ctc_greedy_decode, ids_to_phonemes
try:
    from utils_pattern_b import ctc_beam_search_decode
except Exception:
    ctc_beam_search_decode = None


class LengthAwareCTCLoss(nn.Module):
    """Length-Aware CTC Loss（短い系列に偏らないように重み付け）"""

    def __init__(self, blank: int, zero_infinity: bool = True):
        super().__init__()
        self.blank = blank
        self.zero_infinity = zero_infinity

        try:
            # reduction='none' が使えるなら、サンプル毎lossに重みを掛ける
            self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction='none')
            self.manual_reduction = False
        except TypeError:
            # 古いPyTorch互換：平均しか取れない
            self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity, reduction='mean')
            self.manual_reduction = True
            print("⚠️  Using manual length weighting (PyTorch < 1.7)")

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        if not self.manual_reduction:
            losses = self.ctc(log_probs, targets, input_lengths, target_lengths)  # (B,)

            # 短いtargetほど損失が小さくなりがちなので補正（例：1/sqrt(L)）
            length_weights = 1.0 / torch.sqrt(target_lengths.float().clamp(min=1.0))
            length_weights = length_weights / (length_weights.mean() + 1e-8)

            weighted_loss = (losses * length_weights).mean()
            return weighted_loss
        else:
            base_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
            short_ratio = (target_lengths <= 3).float().mean()
            correction = 1.0 + short_ratio * 0.5
            return base_loss * correction


# ----------------------------
# PER (Levenshtein) helpers
# ----------------------------
def _levenshtein_sdi(ref, hyp):
    """ref/hyp: List[str] -> (S,D,I)"""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 0:diag, 1:up(del), 2:left(ins)
    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            a = dp[i - 1][j - 1] + cost
            b = dp[i - 1][j] + 1
            c = dp[i][j - 1] + 1
            if a <= b and a <= c:
                dp[i][j] = a
                bt[i][j] = 0
            elif b <= c:
                dp[i][j] = b
                bt[i][j] = 1
            else:
                dp[i][j] = c
                bt[i][j] = 2
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        code = bt[i][j]
        if i > 0 and j > 0 and code == 0:
            if ref[i - 1] != hyp[j - 1]:
                S += 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or code == 1):
            D += 1
            i -= 1
        else:
            I += 1
            j -= 1
    return S, D, I


def _lev_sdi_counts(ref, hyp):
    S, D, I = _levenshtein_sdi(ref, hyp)
    return S, D, I, len(ref), len(hyp)


def _sequence_per_percent(ref, hyp):
    """PER[%]（参照長基準）"""
    S, D, I, Nref, _ = _lev_sdi_counts(ref, hyp)
    return 100.0 * (S + D + I) / max(1, Nref)


# ----------------------------
# Trainer
# ----------------------------
class LipReadingTrainer:
    def __init__(
        self,
        model,
        phoneme_encoder,
        device='cuda',
        save_dir='checkpoints',
        early_stopping_metric='val_loss',
        min_delta=0.0,
        decode_beam_width: int = 1,
        use_length_aware_loss: bool = True,
        gradual_unfreezing: bool = True,
        result_dir: str = None,
        use_acoustic: bool = True, 
        save_confusion_matrix: bool = True,
        confusion_matrix_interval: int = 10,
        mode: str = "consonant",
        use_softper: bool = True,
        ctc_weight: float = 1.0,
        lambda_softper: float = 0.1,
        softper_tau: float = 0.1,
        separate_softper_loss: bool = True,
    ):
        self.mode = mode
        self.model = model.to(device)
        self.phoneme_encoder = phoneme_encoder
        self.device = device
        self.save_dir = save_dir
        self.result_dir = result_dir or save_dir.replace('checkpoints', 'results')
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        self.model_returns_log_probs = getattr(self.model, "returns_log_probs", False)

        # ===== CTC Loss =====
        if use_length_aware_loss:
            self.criterion = LengthAwareCTCLoss(
                blank=phoneme_encoder.blank_id,
                zero_infinity=True
            )
            print("✓ Using Length-Aware CTC Loss")
        else:
            self.criterion = nn.CTCLoss(
                blank=phoneme_encoder.blank_id,
                zero_infinity=True
            )
            print("✓ Using standard CTC Loss")

        # ===== SoftPER Loss =====
        self.use_softper = bool(use_softper)
        self.ctc_weight = float(ctc_weight)
        self.lambda_softper = float(lambda_softper)
        self.separate_softper_loss = bool(separate_softper_loss) 
        
        if self.use_softper:
            self.softper = SoftPERLoss(
                blank_id=self.phoneme_encoder.blank_id, 
                tau=float(softper_tau)
            ).to(self.device)
            print(f"✓ Using SoftPER Loss (lambda={self.lambda_softper}, tau={softper_tau})")
        else:
            self.softper = None
            print("✓ SoftPER Loss disabled")
                # 最適化・スケジューラ・AMP
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()

        # 統一評価器
        self.use_acoustic = use_acoustic
        self.save_confusion_matrix = save_confusion_matrix
        self.confusion_matrix_interval = confusion_matrix_interval

        # mode推定：configから渡すのがベストだが、最低限 encoder から推定も可能
        # ここでは trainerに mode を引数でもらう前提（後述の main 修正とセット）
        mode = getattr(self, "mode", None) or "consonant"

        if ENHANCED_METRICS_AVAILABLE and use_acoustic:
            self.evaluator = EnhancedEvaluationMetrics(
                use_acoustic=True,
                mode=mode,
                phoneme_encoder=self.phoneme_encoder
            )
            print(f"✓ Enhanced metrics with acoustic similarity enabled | mode={mode}")
        else:
            self.evaluator = UnifiedEvaluationMetrics()


        # 訓練履歴（CTCニュートラル）
        self.history = {
        'train_loss': [],        # CTC + lambda*SoftPER
        'val_loss': [],          # CTC + lambda*SoftPER
        'train_ctc': [],         # CTC のみ
        'train_softper': [],     # SoftPER のみ
        'val_ctc': [],           # CTC のみ
        'val_softper': [],       # SoftPER のみ
        'ctc_compressed_acc': [],
        'ctc_edit_distance': [],
        'ctc_normalized_distance': [],
        'per_percent': [],
        'lr': [],
        'epoch_time': [],
        'cumulative_time': []
    }

        # 監視設定（EarlyStopping）
        self.min_delta = float(min_delta)
        self.early_stopping_metric = early_stopping_metric  # 'val_loss' / 'per_per' / 'consonant_accuracy' / 'normalized_distance'
        self._monitor_mode = 'min' if early_stopping_metric in (
            'val_loss', 'per_per', 'normalized_distance'
        ) else 'max'
        self._best_monitor = float('inf') if self._monitor_mode == 'min' else -float('inf')
        self.best_val_loss = float('inf')
        self._patience_counter = 0

        # デコード
        self.decode_beam_width = int(decode_beam_width)

        # ===== Gradual Unfreezing設定 =====
        self.gradual_unfreezing = gradual_unfreezing

        if self.gradual_unfreezing:
            self.unfreeze_schedule = {
                0: 'init',
                15: 'lstm',
                25: 'attention',
                35: 'stage4',
                45: 'stage3',
                55: 'all',
            }
            print(f"✓ Gradual Unfreezing enabled (DEFAULT schedule)")
            print(f"  Default schedule:")
            for epoch, action in sorted(self.unfreeze_schedule.items()):
                print(f"    Epoch {epoch:3d} → {action}")
        else:
            self.unfreeze_schedule = None
            print(f"✓ Gradual Unfreezing disabled")

    def set_unfreeze_schedule(self, schedule: dict):
        if not self.gradual_unfreezing:
            print("⚠️  Gradual Unfreezing is disabled, schedule will be ignored")
            return
        self.unfreeze_schedule = schedule
        print(f"\n{'='*60}")
        print(f"✓ CUSTOM Unfreeze Schedule Set:")
        print(f"{'='*60}")
        for epoch, action in sorted(schedule.items()):
            print(f"  Epoch {epoch:3d} → {action}")
        print(f"{'='*60}\n")

    def apply_gradual_unfreezing(self, epoch: int):
        """エポックに応じて層を解凍（model側にfreeze/unfreeze APIがある前提）"""
        if not self.gradual_unfreezing or self.unfreeze_schedule is None:
            return
        if epoch not in self.unfreeze_schedule:
            return

        action = self.unfreeze_schedule[epoch]

        print(f"\n{'='*60}")
        print(f"🔄 Epoch {epoch}: Unfreezing schedule - '{action}'")
        print(f"{'='*60}")

        if action == 'init':
            self.model.freeze_all()
            self.model.unfreeze_classifier_and_lstm()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.optimizer.defaults['lr']

        elif action == 'lstm':
            self.model.unfreeze_lstm()

        elif action == 'attention':
            self.model.unfreeze_attention()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.8

        elif action.startswith('stage'):
            stage_num = int(action[-1])
            self.model.unfreeze_stage(stage_num)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.7

        elif action == 'all':
            self.model.unfreeze_all_layers()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5

        else:
            print(f"⚠️ Unknown unfreezing action: {action}")
            return

        self.model.print_trainable_status()
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"📊 Current learning rate: {current_lr:.2e}")
        print(f"{'='*60}\n")

    # ----------------------------
    # セットアップ
    # ----------------------------
    def setup_optimizer(self, optimizer_type='adamw', lr=3e-4, weight_decay=1e-2):
        ot = optimizer_type.lower()
        if ot == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif ot == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif ot == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        print(f"✓ Optimizer: {optimizer_type} (lr={lr})")

    def setup_scheduler(self, scheduler_type='cosine', **kwargs):
        st = scheduler_type.lower() if scheduler_type else ''
        if st == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
            )
        elif st == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('T_max', 50),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif st == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        else:
            self.scheduler = None
        print(f"✓ Scheduler: {scheduler_type if scheduler_type else 'None'}")

    # ----------------------------
    # NaN/長さ対策: バッチ整形
    # ----------------------------
    def _filter_valid_batch(self, batch):
        """
        tlen==0 や ilen==0 を除外し、CTC用に target を再構成。
        返り値: (x, y, ilen, tlen) or None
        """
        x = batch['video'].to(self.device)        # (B,T,1,64,64)
        y_cat = batch['target']                   # (sum_L,)
        ilen = batch['input_length']              # (B,)
        tlen = batch['target_length']             # (B,)

        valid = (tlen > 0) & (ilen > 0)
        if valid.sum() == 0:
            return None

        x = x[valid]
        ilen = ilen[valid]

        new_targets, new_tlens = [], []
        off = 0
        for i in range(len(tlen)):
            tl = int(tlen[i])
            seg = y_cat[off:off + tl]
            if valid[i] and tl > 0:
                new_targets.extend(seg.tolist())
                new_tlens.append(tl)
            off += tl

        y = torch.tensor(new_targets, dtype=torch.long, device=self.device)
        tlen = torch.tensor(new_tlens, dtype=torch.long, device=self.device)
        return x, y, ilen, tlen

    # ----------------------------
    # 学習・検証
    # ----------------------------
    # train.py の train_epoch() メソッド修正版

    # 2. train_epoch() の返り値を修正
    def train_epoch(self, train_loader):
        """1エポック学習（CTC+SoftPER / NaN対策込み）"""
        from tqdm import tqdm

        self.model.train()
        total_loss, total_ctc, total_softper, num_batches = 0.0, 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc="Training", leave=False)

        for batch_idx, batch in enumerate(pbar):
            flt = self._filter_valid_batch(batch)
            if flt is None:
                continue
            videos, targets, input_lengths, target_lengths = flt

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                outputs = self.model(videos)

                log_probs = (outputs if self.model_returns_log_probs
                            else torch.log_softmax(outputs, dim=-1)).permute(1, 0, 2)

                log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))

                Tcur = log_probs.size(0)
                input_lengths = torch.clamp(input_lengths, max=Tcur)
                target_lengths = torch.clamp(target_lengths, min=1)

                ctc_loss = self.criterion(log_probs, targets, input_lengths, target_lengths)

                if self.use_softper:
                    softper_loss = self.softper(log_probs, targets, input_lengths, target_lengths)
                    loss = self.ctc_weight * ctc_loss + self.lambda_softper * softper_loss  # ← 修正
                    
                    if batch_idx == 0 and num_batches == 0:
                        print(f"\n[DEBUG Train]")
                        print(f"  CTC: {ctc_loss.item():.4f}")
                        print(f"  SoftPER: {softper_loss.item():.4f}")
                        print(f"  lambda: {self.lambda_softper}")
                        print(f"  Combined: {loss.item():.4f}")
                        print(f"  self.softper type: {type(self.softper)}")
                        print(f"  self.use_softper: {self.use_softper}")
                else:
                    loss = ctc_loss
                    softper_loss = torch.tensor(0.0)

                if not torch.isfinite(loss):
                    continue

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += float(loss.item())
            total_ctc += float(ctc_loss.item())
            total_softper += float(softper_loss.item()) if self.use_softper else 0.0
            num_batches += 1

            avg_loss_so_far = total_loss / max(1, num_batches)
            pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})

        avg_loss = total_loss / max(1, num_batches)
        avg_ctc = total_ctc / max(1, num_batches)
        avg_softper = total_softper / max(1, num_batches)
        
        # ★ デバッグ追加
        print(f"\n[DEBUG train_epoch] num_batches={num_batches}")
        print(f"  avg_loss (Combined): {avg_loss:.4f}")
        print(f"  avg_ctc:             {avg_ctc:.4f}")
        print(f"  avg_softper:         {avg_softper:.4f}")
        
        if self.separate_softper_loss:
            return avg_loss, avg_ctc, avg_softper
        else:
            return avg_loss

    

        # train.py の validate() メソッド修正版

        # 3. validate() の返り値を修正
    def validate(self, val_loader):
        """検証（CTC+SoftPER）— 統一評価器を使用"""
        from tqdm import tqdm

        self.model.eval()
        total_loss = 0.0
        total_ctc = 0.0      # ← 追加
        total_softper = 0.0  # ← 追加
        all_predictions, all_targets = [], []

        pbar = tqdm(val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for batch in pbar:
                flt = self._filter_valid_batch(batch)
                if flt is None:
                    continue
                x, y, ilen, tlen = flt

                out = self.model(x)
                logp = (out if self.model_returns_log_probs
                        else torch.log_softmax(out, dim=-1)).permute(1, 0, 2)
                logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))

                Tcur = logp.size(0)
                ilen = torch.clamp(ilen, max=Tcur)
                tlen = torch.clamp(tlen, min=1)

                ctc_loss = self.criterion(logp, y, ilen, tlen)

                if self.use_softper:
                    softper_loss = self.softper(logp, y, ilen, tlen)
                    loss = self.ctc_weight * ctc_loss + self.lambda_softper * softper_loss  # ← 修正
                    print(f"[VAL Batch] ctc={ctc_loss.item():.4f}, softper={softper_loss.item():.4f}")  # デバッグ
                else:
                    loss = ctc_loss
                    softper_loss = torch.tensor(0.0, device=self.device)  # ← device追加

                total_loss += float(loss.item())
                total_ctc += float(ctc_loss.item())
                total_softper += float(softper_loss.item())  # if self.use_softper を削除してみる

                # デコード処理
                use_beam = (self.decode_beam_width is not None
                            and self.decode_beam_width >= 2
                            and ctc_beam_search_decode is not None)

                if not hasattr(self, '_decode_method_printed'):
                    print(f"✓ Using {'BEAM SEARCH' if use_beam else 'GREEDY'} decode"
                        f"{' (beam_width='+str(self.decode_beam_width)+')' if use_beam else ''}")
                    self._decode_method_printed = True

                if use_beam:
                    pred_ids = ctc_beam_search_decode(
                        logp, self.phoneme_encoder.blank_id, ilen, beam_width=self.decode_beam_width
                    )
                else:
                    pred_ids = ctc_greedy_decode(logp, self.phoneme_encoder.blank_id, ilen)


                    if len(all_predictions) == 0:
                        print(f"\n[DEBUG ctc_greedy_decode詳細]")
                        print(f"  max_ids[0,:5] = {logp.argmax(dim=-1)[0,:5]}")  # 最初5フレーム
                        print(f"  blank以外の出現: {(logp.argmax(dim=-1) != 0).sum().item()} / {logp.shape[0] * logp.shape[1]}")

                preds = ids_to_phonemes(pred_ids, self.phoneme_encoder)

                off = 0
                for tl in tlen.tolist():
                    ids = y[off:off + tl].detach().cpu().tolist()
                    all_targets.append(self.phoneme_encoder.decode_phonemes(ids))
                    off += tl

                if len(all_predictions) < 3:  # 最初の3バッチだけ
                    print(f"\n[DEBUG] Batch predictions:")
                    print(f"  pred_ids[:3]: {pred_ids[:3]}")
                    print(f"  preds[:3]: {preds[:3]}")
                    print(f"  targets[:3]: {all_targets[:min(3, len(all_targets))]}")

                all_predictions.extend(preds)

                pbar.set_postfix({'samples': len(all_predictions)})

        # メトリクス計算（既存コード省略）
        sample_results = []
        global_S = global_D = global_I = 0
        len_refs, len_hyps = [], []
        total_edit_dist = 0

        for p, t in zip(all_predictions, all_targets):
            S, D, I, Nref, Nhyp = _lev_sdi_counts(t, p)
            per_ref = 100.0 * (S + D + I) / max(1, Nref)
            if per_ref > 0.0:
                sample_results.append({
                    'per_ref': round(per_ref, 2),
                    'S': S, 'D': D, 'I': I,
                    'len_ref': Nref, 'len_hyp': Nhyp,
                    'predicted': p, 'target': t
                })
            global_S += S
            global_D += D
            global_I += I
            total_edit_dist += (S + D + I)
            len_refs.append(Nref)
            len_hyps.append(Nhyp)

        sample_results.sort(key=lambda x: -x['per_ref'])

        metrics = self.evaluator.calculate_all_metrics(all_predictions, all_targets)
        metrics['sample_results'] = sample_results[:10]
        metrics['global_S'] = global_S
        metrics['global_D'] = global_D
        metrics['global_I'] = global_I
        metrics['avg_len_ref'] = float(np.mean(len_refs)) if len_refs else 0.0
        metrics['avg_len_hyp'] = float(np.mean(len_hyps)) if len_hyps else 0.0
        avg_edit_dist = total_edit_dist / max(1, len(all_predictions))
        metrics['avg_edit_distance'] = avg_edit_dist

        bins = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 999)]
        bin_map = {}
        for lo, hi in bins:
            S = D = I = total_len = 0
            for p, t in zip(all_predictions, all_targets):
                if lo <= len(t) <= hi:
                    s, d, i = _levenshtein_sdi(t, p)
                    S += s
                    D += d
                    I += i
                    total_len += len(t)
            per_bin = 100.0 * (S + D + I) / max(1, total_len)
            bin_map[f"{lo}-{hi}"] = per_bin
        metrics['length_bucket_per'] = bin_map

        self.last_val_predictions = all_predictions
        self.last_val_targets = all_targets

        # SoftPER分離
        num_batches = max(1, len(val_loader))
        avg_val_loss = total_loss / num_batches
        avg_val_ctc = total_ctc / num_batches
        avg_val_softper = total_softper / num_batches

        
        # ★ デバッグ追加
        print(f"\n[DEBUG validate] num_batches={num_batches}")
        print(f"  avg_val_loss (Combined): {avg_val_loss:.4f}")
        print(f"  avg_val_ctc:             {avg_val_ctc:.4f}")
        print(f"  avg_val_softper:         {avg_val_softper:.4f}")
        
        if self.separate_softper_loss:
            return avg_val_loss, avg_val_ctc, avg_val_softper, metrics
        else:
            return avg_val_loss, metrics

    # ----------------------------
    # train
    # ----------------------------
    def train(self, train_loader, val_loader, epochs=100, early_stopping_patience=20):
        """学習ループ(Gradual Unfreezing対応・EarlyStopping可視化・Ctrl+C対応)"""
        print(f"\n{'=' * 70}\n訓練開始: {epochs}エポック\n{'=' * 70}")
        print(f"💡 Tips: Press Ctrl+C to stop training safely and save progress")
        print(f"{'=' * 70}\n")
        use_softper = bool(getattr(self, "use_softper", False))
        lambda_softper = float(getattr(self, "lambda_softper", 0.0)) if use_softper else 0.0

        print("\n" + "="*70)
        print("Loss / SoftPER Settings")
        print("="*70)
        print(f"CTC Loss:            ON")
        print(f"SoftPER:             {'ON' if use_softper else 'OFF'}")
        print(f"lambda_softper:      {lambda_softper:.6f}")
        print("="*70 + "\n")
        if self.gradual_unfreezing:
            self.apply_gradual_unfreezing(0)

        start_time = time.time()
        first_epoch = True
        current_epoch = 0

        try:
            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                self.apply_gradual_unfreezing(epoch)

                train_loss, train_ctc, train_softper = self.train_epoch(train_loader)  # ← 3つ受け取る
                val_loss, val_ctc, val_softper, val_metrics = self.validate(val_loader)  # ← 4つ受け取る

                print(f"\n[DEBUG train()] Epoch {epoch}")
                print(f"  train: loss={train_loss:.4f}, ctc={train_ctc:.4f}, softper={train_softper:.4f}")
                print(f"  val:   loss={val_loss:.4f}, ctc={val_ctc:.4f}, softper={val_softper:.4f}")

                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                self.history.setdefault('lr', []).append(float(self.optimizer.param_groups[0]['lr']))

                epoch_time = time.time() - epoch_start
                cumulative_time = time.time() - start_time

                # 履歴記録
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_ctc'].append(train_ctc)
                self.history['train_softper'].append(train_softper)
                self.history['val_ctc'].append(val_ctc)
                self.history['val_softper'].append(val_softper)
                self.history['ctc_compressed_acc'].append(val_metrics.get('consonant_accuracy', 0.0) / 100.0)

                avg_edit_dist = val_metrics.get('avg_edit_distance', 0.0)
                self.history['ctc_edit_distance'].append(avg_edit_dist)

                self.history['ctc_normalized_distance'].append(val_metrics.get('per_per', 0.0) / 100.0)
                self.history['per_percent'].append(val_metrics.get('per_per', 0.0))
                self.history['lr'].append(current_lr)
                self.history['epoch_time'].append(epoch_time)
                self.history['cumulative_time'].append(cumulative_time)

                print(f"  history['train_softper'][-1] = {self.history['train_softper'][-1]:.4f}")
                print(f"  history['val_softper'][-1] = {self.history['val_softper'][-1]:.4f}")

                # EarlyStopping monitor
                if self.early_stopping_metric == 'val_loss':
                    monitor = val_loss
                elif self.early_stopping_metric == 'val_softper':
                    if self.separate_softper_loss:
                        monitor = val_softper  # SoftPER単体
                    else:
                        monitor = val_loss     # CTC+SoftPER
                elif self.early_stopping_metric == 'per_per':
                    monitor = val_metrics.get('per_per', float('inf'))
                elif self.early_stopping_metric == 'consonant_accuracy':
                    monitor = val_metrics.get('consonant_accuracy', float('-inf'))
                else:
                    monitor = self.history['ctc_normalized_distance'][-1]

                improved = (monitor < (self._best_monitor - self.min_delta)) if \
                    (self._monitor_mode == 'min') else (monitor > (self._best_monitor + self.min_delta))

                if improved:
                    es_status = f"🌟 BEST"
                    self._best_monitor = monitor
                    self._patience_counter = 0
                else:
                    self._patience_counter += 1
                    es_status = f"⏳ Patience: {self._patience_counter}/{early_stopping_patience}"

                print(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                    f"Acc: {val_metrics.get('consonant_accuracy', 0.0):5.2f}% | "
                    f"PER: {val_metrics.get('per_per', 0.0):5.2f}% | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {epoch_time:4.1f}s | "
                    f"{es_status}"
                )

                if epoch % 5 == 0 or epoch == 1:
                    print(f"  詳細:")
                    print(f"    Train Loss: {train_loss:.4f}")
                    print(f"    Val   Loss: {val_loss:.4f}")
                    print(f"    Consonant Acc: {val_metrics.get('consonant_accuracy', 0.0):.2f}%")
                    print(f"    PER: {val_metrics.get('per_per', 0.0):.2f}%")
                    print(f"    Avg Edit Distance: {avg_edit_dist:.2f}")
                    print(f"    Monitor({self.early_stopping_metric}): {monitor:.4f} (Best: {self._best_monitor:.4f})")

                    if first_epoch:
                        print(f"    [DEBUG] metrics keys: {list(val_metrics.keys())}")
                        first_epoch = False

                # best monitor 保存
                if improved:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'monitor': self.early_stopping_metric,
                        'best_monitor_value': self._best_monitor,
                        'history': self.history,
                        'phoneme_encoder': self.phoneme_encoder,
                    }, os.path.join(self.save_dir, f'best_{self.early_stopping_metric}.pth'))

                # best val_loss 保存
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'best_val_loss': self.best_val_loss,
                        'history': self.history,
                        'phoneme_encoder': self.phoneme_encoder,
                    }, os.path.join(self.save_dir, 'best_val_loss.pth'))

                if epoch % 10 == 0:
                    self.save_checkpoint(epoch, is_best=False)

                if self._patience_counter >= early_stopping_patience:
                    print(f"\n{'='*70}")
                    print(f"🛑 Early Stopping at Epoch {epoch}")
                    print(f"   Metric: {self.early_stopping_metric}")
                    print(f"   Best value: {self._best_monitor:.4f}")
                    print(f"   No improvement for {early_stopping_patience} epochs")
                    print(f"{'='*70}\n")
                    break

        except KeyboardInterrupt:
            print(f"\n\n{'='*70}")
            print(f"⚠️  Training interrupted by user (Ctrl+C)")
            print(f"{'='*70}")
            print(f"🔍 Stopped at Epoch {current_epoch}/{epochs}")
            print(f"⏱️  Elapsed time: {(time.time() - start_time)/3600:.2f} hours")
            print(f"{'='*70}")

            interrupt_path = os.path.join(self.save_dir, f'interrupted_epoch_{current_epoch}.pth')
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_monitor_value': self._best_monitor,
                'best_val_loss': self.best_val_loss,
                'history': self.history,
                'phoneme_encoder': self.phoneme_encoder,
                'interrupted': True,
                'patience_counter': self._patience_counter
            }, interrupt_path)
            print(f"\n💾 Emergency checkpoint saved: {interrupt_path}")

        else:
            # 通常終了時：最終モデル保存
            print(f"\n{'='*70}")
            print("最終モデル保存中...")
            print(f"{'='*70}")

            final_model_path = os.path.join(self.result_dir, 'final_model.pth')

            # ======================================================
            # ① 最終評価を1回だけ実行して final を作る（ここが必須）
            # ======================================================
            final = None
            try:
                from train import evaluate_model as eval_fn  # 同一ファイルなら import不要だが安全側
                final = eval_fn(
                    self.model, val_loader, self.phoneme_encoder, self.device,
                    show_samples=False, num_samples=0, beam_width=1  # greedyなら1
                )

                mode = self.phoneme_encoder.mode if hasattr(self.phoneme_encoder, "mode") \
                    else ('vowel' if hasattr(self.phoneme_encoder, 'vowels') else 'consonant')

                phoneme_list = getattr(self.phoneme_encoder, 'vowels', None) if mode == 'vowel' \
                            else getattr(self.phoneme_encoder, 'consonants', None)

                out_png = os.path.join(self.result_dir, f"confusion_matrix_final_{mode}.png")
                self.evaluator.save_confusion_matrix(
                    final['raw']['predictions'],
                    final['raw']['targets'],
                    output_path=out_png,
                    phoneme_list=phoneme_list,
                    normalize=False
                )
            except Exception as e:
                print(f"⚠️ 最終評価(evaluate_model)でエラー: {e}")
                import traceback; traceback.print_exc()
                final = None

            # ======================================================
            # ② 最終混同行列（1回だけ）— utils outputdir (= self.result_dir) に保存
            # ======================================================
            if final is not None and hasattr(self.evaluator, 'save_confusion_matrix'):
                try:
                    print("\n" + "="*70)
                    print("最終混同行列を保存中...")
                    print("="*70)

                    output_dir = self.result_dir
                    os.makedirs(output_dir, exist_ok=True)

                    # mode に応じて phoneme_list を切替（vowel/consonant共通）
                    mode = self.phoneme_encoder.mode if hasattr(self.phoneme_encoder, "mode") \
                        else ('vowel' if hasattr(self.phoneme_encoder, 'vowels') else 'consonant')

                    if mode == 'vowel':
                        phoneme_list = getattr(self.phoneme_encoder, 'vowels', None)
                    else:
                        phoneme_list = getattr(self.phoneme_encoder, 'consonants', None)

                    output_path = os.path.join(output_dir, f'confusion_matrix_final_{mode}.png')

                    self.evaluator.save_confusion_matrix(
                        predictions=final['raw']['predictions'],
                        targets=final['raw']['targets'],
                        output_path=output_path,
                        phoneme_list=phoneme_list
                    )

                    print(f"✓ 最終混同行列保存完了: {output_path}")

                except Exception as e:
                    print(f"⚠️ 混同行列保存でエラー: {e}")
                    import traceback; traceback.print_exc()

            # ======================================================
            # ③ 最終モデル保存（finalを保存したいなら一緒に入れる）
            # ======================================================
            torch.save({
                'epoch': current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_monitor_value': self._best_monitor,
                'best_val_loss': self.best_val_loss,
                'history': self.history,
                'phoneme_encoder': self.phoneme_encoder,
                'final': final,  # ← あってもOK（不要なら消してOK）
            }, final_model_path)

            # ===== 最終混同行列（絶対保存版：finalに依存しない） =====
            try:
                print("\n" + "="*70)
                print("最終混同行列を保存中... (last_val_predictions/targets を使用)")
                print("="*70)

                # 1) evaluator が save_confusion_matrix を持ってるか確認
                print(f"[DEBUG] evaluator type: {type(self.evaluator).__name__}")
                if not hasattr(self.evaluator, "save_confusion_matrix"):
                    raise RuntimeError("evaluator に save_confusion_matrix がありません（EnhancedEvaluationMetricsになってない可能性）")

                # 2) validate() が最後に作った予測/正解を使う
                preds = getattr(self, "last_val_predictions", None)
                tgts  = getattr(self, "last_val_targets", None)
                if preds is None or tgts is None:
                    raise RuntimeError("last_val_predictions/targets が見つかりません。validate() 内で self.last_val_* を代入しているか確認して。")
                if len(preds) == 0 or len(tgts) == 0:
                    raise RuntimeError(f"pred/target が空です: len(preds)={len(preds)}, len(tgts)={len(tgts)}")

                # 3) 保存先
                output_dir = self.result_dir
                os.makedirs(output_dir, exist_ok=True)
                print(f"[DEBUG] output_dir: {output_dir}")

                # 4) mode と phoneme_list（vowel/consonant共通）
                mode = self.phoneme_encoder.mode if hasattr(self.phoneme_encoder, "mode") \
                    else ("vowel" if hasattr(self.phoneme_encoder, "vowels") else "consonant")

                if mode == "vowel":
                    phoneme_list = getattr(self.phoneme_encoder, "vowels", None)
                else:
                    phoneme_list = getattr(self.phoneme_encoder, "consonants", None)

                output_path = os.path.join(output_dir, f"confusion_matrix_final_{mode}.png")

                # 5) 保存実行（normalize=True にすると少数でも色が出る）
                self.evaluator.save_confusion_matrix(
                    predictions=preds,
                    targets=tgts,
                    output_path=output_path,
                    phoneme_list=phoneme_list,
                    normalize=True,  # ← enhanced_metrics.py側で引数対応してるならON推奨
                )

                # 6) 本当にファイルができたか確認
                exists = os.path.exists(output_path)
                size = os.path.getsize(output_path) if exists else -1
                print(f"[DEBUG] saved? {exists}  size={size}")
                if not exists or size <= 0:
                    raise RuntimeError("save_confusion_matrix を呼んだのに PNG が生成されていません（関数内部で早期returnしている可能性）")

                print(f"✓ 最終混同行列保存完了: {output_path}")

            except Exception as e:
                print(f"❌ 最終混同行列の保存に失敗: {e}")
                import traceback; traceback.print_exc()

            print(f"✓ 最終モデル保存完了: {final_model_path}")
            print(f"{'='*70}\n")

        finally:
            total_time = time.time() - start_time
            print(f"\n{'=' * 70}")
            print(f"訓練終了")
            print(f"  総時間: {total_time/3600:.2f}時間")
            print(f"  最終エポック: {current_epoch}")
            print(f"  最良Val Loss: {self.best_val_loss:.4f}")
            print(f"  最良{self.early_stopping_metric}: {self._best_monitor:.4f}")
            print(f"{'=' * 70}\n")

            if len(self.history['train_loss']) > 0:
                history_plot_path = os.path.join(self.result_dir, 'training_history.png')
                try:
                    self.plot_history(save_path=history_plot_path)
                    print(f"✓ Training history plot saved: {history_plot_path}")
                except Exception as e:
                    print(f"⚠️  Failed to save training plot: {e}")

                try:
                    os.makedirs(self.result_dir, exist_ok=True)
                    self.plot_lr_only(save_path=os.path.join(self.result_dir, "lr_history.png"))
                except Exception as e:
                    print(f"⚠️ lr plot failed: {e}")

        return self.history

    # ----------------------------
    # 保存・復帰・可視化
    # ----------------------------
    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'phoneme_encoder': self.phoneme_encoder,
        }
        fname = 'best_model.pth' if is_best else f'checkpoint_epoch_{epoch}.pth'
        torch.save(ckpt, os.path.join(self.save_dir, fname))

    def load_checkpoint(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.scheduler and ckpt.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.history = ckpt.get('history', self.history)
        print(f"✓ Checkpoint loaded: Epoch {ckpt.get('epoch', '?')}")
        return ckpt.get('epoch', 0)

    def plot_history(self, save_path=None):
        """
        Training history plot (1 figure, 2 panels side-by-side)
        - Left : Loss (Train / Val)
        - Right: PER [%]
        """
        if len(self.history.get('train_loss', [])) == 0:
            print("⚠ history が空なので plot をスキップします")
            return

        epochs = range(1, len(self.history['train_loss']) + 1)

        fig, (ax_loss, ax_per) = plt.subplots(1, 2, figsize=(14, 5))

        ax_loss.plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        ax_loss.plot(epochs, self.history['val_loss'], label='Val Loss', linestyle='--', linewidth=2)
        ax_loss.set_title('Loss', fontweight='bold', fontsize=13)
        ax_loss.set_xlabel('Epoch', fontsize=11)
        ax_loss.set_ylabel('Loss', fontsize=11)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc='best')

        per_hist = self.history.get('per_percent', [])
        if len(per_hist) > 0:
            ax_per.plot(epochs, per_hist, label='Val PER (%)', linewidth=2)
        ax_per.set_title('PER [%]', fontweight='bold', fontsize=13)
        ax_per.set_xlabel('Epoch', fontsize=11)
        ax_per.set_ylabel('PER [%]', fontsize=11)
        ax_per.grid(True, alpha=0.3)
        ax_per.legend(loc='best')

        fig.suptitle(f"Training History (CTC{'+SoftPER' if getattr(self,'use_softper',False) else ''})",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ Training history saved: {save_path}")
        plt.close()


    def plot_lr_only(self, save_path=None):
        """Learning rate only plot (optional utility)"""
        lr_hist = self.history.get('lr', [])
        if len(lr_hist) == 0:
            print("⚠ lr history が空なので lr plot をスキップします")
            return

        epochs = range(1, len(lr_hist) + 1)

        plt.figure(figsize=(7, 4))
        plt.plot(epochs, lr_hist, linewidth=2)
        plt.title("Learning Rate", fontweight='bold', fontsize=13)
        plt.xlabel("Epoch", fontsize=11)
        plt.ylabel("Learning Rate", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"✓ LR history saved: {save_path}")
        plt.close()



# ----------------------------
# 最終評価（validate と同じ“正しい経路”）
# ----------------------------
def evaluate_model(model, data_loader, phoneme_encoder, device,
                   show_samples=False, num_samples=5, beam_width=1,
                   confidence_threshold=0.0):
    """
    confidence_threshold: 予測確率がこの値以下なら出力しない (過剰予測抑制)
    """
    model.eval()
    evaluator = UnifiedEvaluationMetrics()

    all_predictions, all_targets = [], []
    anomaly_samples = []
    decode_method_printed = False

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            x = batch['video'].to(device)
            y = batch['target']
            ilen = batch['input_length']
            tlen = batch['target_length']

            out = model(x)
            returns_log = getattr(model, "returns_log_probs", False)
            logp = (out if returns_log else torch.log_softmax(out, dim=-1)).permute(1, 0, 2)
            logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))

            Tcur = logp.size(0)
            ilen = torch.clamp(ilen, max=Tcur)
            tlen = torch.clamp(tlen, min=1)

            # Confidence Filtering
            max_probs = torch.exp(logp).max(dim=2)[0]  # (T, B)
            confidence_mask = max_probs > confidence_threshold

            use_beam = (beam_width and beam_width >= 2 and ctc_beam_search_decode is not None)

            if not decode_method_printed:
                print(f"✓ evaluate_model: Using {'BEAM SEARCH' if use_beam else 'GREEDY'} decode"
                      f"{' (beam_width='+str(beam_width)+')' if use_beam else ''}")
                decode_method_printed = True

            if use_beam:
                pred_ids = ctc_beam_search_decode(
                    logp, phoneme_encoder.blank_id, ilen, beam_width=beam_width
                )
            else:
                pred_ids = ctc_greedy_decode(logp, phoneme_encoder.blank_id, ilen)

            # Confidence適用（簡易）
            for i in range(len(pred_ids)):
                valid_t = int(ilen[i].item())
                mask = confidence_mask[:valid_t, i].cpu().numpy()

                filtered = []
                for j, token_id in enumerate(pred_ids[i]):
                    if j < len(mask) and mask[j]:
                        filtered.append(token_id)
                pred_ids[i] = filtered if filtered else pred_ids[i][:3]

            preds = ids_to_phonemes(pred_ids, phoneme_encoder)

            off = 0
            for b_idx, tl in enumerate(tlen.tolist()):
                ids = y[off:off + tl].cpu().tolist()
                target = phoneme_encoder.decode_phonemes(ids)
                all_targets.append(target)

                pred = preds[b_idx]
                S, D, I = _levenshtein_sdi(target, pred)
                per_ref = 100.0 * (S + D + I) / max(1, len(target))

                if per_ref > 300 and len(anomaly_samples) < 20:
                    anomaly_samples.append({
                        'batch_idx': batch_idx,
                        'sample_idx': b_idx,
                        'target': target,
                        'predicted': pred,
                        'per': per_ref,
                        'target_len': len(target),
                        'pred_len': len(pred)
                    })

                off += tl

            all_predictions.extend(preds)

    if anomaly_samples:
        print("\n" + "=" * 70)
        print("⚠️  Anomaly Samples (PER > 300%)")
        print("=" * 70)
        for i, s in enumerate(anomaly_samples[:10], 1):
            print(f"{i:2d}) PER={s['per']:.1f}% | tgt_len={s['target_len']}, pred_len={s['pred_len']}")
            print(f"    Target: {' '.join(s['target'])}")
            print(f"    Predicted: {' '.join(s['predicted'])}")
        print("=" * 70)

    sample_results = []
    global_S = global_D = global_I = 0
    len_refs, len_hyps = [], []

    for p, t in zip(all_predictions, all_targets):
        S, D, I, Nref, Nhyp = _lev_sdi_counts(t, p)
        per_ref = 100.0 * (S + D + I) / max(1, Nref)
        per_disp = 100.0 * (S + D + I) / max(1, max(Nref, Nhyp))
        if per_ref > 0.0:
            sample_results.append({
                'per_ref': round(per_ref, 2),
                'per_display': round(per_disp, 2),
                'S': S, 'D': D, 'I': I,
                'len_ref': Nref, 'len_hyp': Nhyp,
                'predicted': p, 'target': t
            })
        global_S += S
        global_D += D
        global_I += I
        len_refs.append(Nref)
        len_hyps.append(Nhyp)

    sample_results.sort(key=lambda x: -x['per_ref'])

    metrics = evaluator.calculate_all_metrics(all_predictions, all_targets)
    metrics['sample_results'] = sample_results[:10]
    metrics['global_S'] = global_S
    metrics['global_D'] = global_D
    metrics['global_I'] = global_I
    metrics['avg_len_ref'] = float(np.mean(len_refs)) if len_refs else 0.0
    metrics['avg_len_hyp'] = float(np.mean(len_hyps)) if len_hyps else 0.0

    bins = [(1, 1), (2, 3), (4, 6), (7, 10), (11, 20), (21, 999)]
    bin_map = {}
    for lo, hi in bins:
        S = D = I = N = 0
        for p, t in zip(all_predictions, all_targets):
            if lo <= len(t) <= hi:
                s, d, i = _levenshtein_sdi(t, p)
                S += s
                D += d
                I += i
                N += len(t)
        per_bin = 100.0 * (S + D + I) / max(1, N)
        bin_map[f"{lo}-{hi}"] = per_bin
    metrics['length_bucket_per'] = bin_map

    if show_samples:
        print("\n[Top 10 Incorrect Samples by PER]")
        for i, s in enumerate(sample_results[:10], 1):
            print(
                f"{i:2d}) PER(ref)={s['per_ref']:>6.2f}%  PER(disp)={s['per_display']:>6.2f}%  "
                f"[S/D/I]={s['S']}/{s['D']}/{s['I']}  "
                f"len(ref/hyp)={s['len_ref']}/{s['len_hyp']} | "
                f"pred: {' '.join(s['predicted'][:20])} || tgt: {' '.join(s['target'][:20])}"
            )

    return {
        'ctc_compressed_acc': metrics.get('exact_match_consonant_exact_match_rate', 0.0),
        'edit_distance': metrics.get('consonant_errors', 0.0),
        'normalized_distance': (metrics.get('per_per', 0.0) / 100.0) if metrics.get('per_per') is not None else 0.0,
        'avg_per': metrics.get('per_per', 0.0),
        'raw': {
            **metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'sample_results': sample_results[:10],
            'anomaly_samples': anomaly_samples
        },
    }


if __name__ == "__main__":
    print("train.py (CTC neutral) loaded")
