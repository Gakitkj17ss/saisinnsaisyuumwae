#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
読唇術プロジェクト用ユーティリティ（Frame Attention対応版）
"""

import os
import json
import yaml
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
# ファイル先頭の import に追加
from dataset import create_dataloaders


class Config:
    """設定管理クラス"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        設定初期化
        
        Args:
            config_path: 設定ファイルパス（yaml or json）
        """
        # デフォルト設定
        self.default_config = {
            # モデル設定
            'model': {
                'input_channels': 1,
                'num_classes': 16, #6,16,32
                'model_type': 'deep_cnn',  # 'deep_cnn' or 'pattern_b_frame_attention'
                #'model_type': 'pattern_b_frame_attention',
                'dropout_rate': 0.3,
                
                # ===== Frame Attention 設定 =====
                'attention_type': 'softmax',  # 'sigmoid' or 'softmax'
                'temperature': 0.6,  # 0.3-1.0推奨
                'dual_attention': True,  # TrueでLSTM後にTemporal Attentionも併用
                
                'mode': 'consonant', #vowel,consonant,consonant_extended
                #Deep CNN用の追加パラメータ
                'lstm_layers': 2,     # 2, 3, 4 で実験可能
                'lstm_hidden': 384    # 大きめ推奨
            },
            
            # 訓練設定
            'training': {
                 'epochs': 150,
                 'optimizer': 'adamw',
                 'lr': 0.0001,
                 'weight_decay': 0.01,
                 'num_samples': 3,
                 'early_stopping_metric': 'per_per',
                 #'loss_type': 'expected_per',
                # 監視指標: 'val_loss' | 'per_per' | 'consonant_accuracy' | 'normalized_distance',combined_loss
                # 推奨: CTC≠PER の場合は 'per_per','val_softper'
                #'scheduler': 'cosine',  # 'reduce_on_plateau' → 'cosine'
                #'scheduler_params': {
                #    'T_max': 150,  # epochs数に合わせる
                #    'eta_min': 0.00001,
                #},
                'scheduler': 'reduce_on_plateau',  # 変更 reduce_on_plateau or cosine
                'scheduler_params': {
                    'mode': 'min',
                    'factor': 0.7,
                    'threshold':1e-3,
                    'patience': 8,
                    'min_lr': 0.000005
                },
                 'use_softper': True,
                 'separate_softper_loss': True,
                 # SoftPERのみ
                 'ctc_weight': 1.0,
                 'lambda_softper': 0.0,

                 # 通常CTC
                 #'ctc_weight': 1.0,
                 #'lambda_softper': 0.0,
                 'softper_tau': 0.1,
                 'early_stopping_patience': 10,
                 'gradient_clip': 5.0,
                 'use_length_aware_loss': True,
                 'use_acoustic': True,                    # 音響的評価を有効化
                 'save_confusion_matrix': True,           # 混同行列を保存
                 'confusion_matrix_interval': 10,         # 10エポックごと

                # ===== Gradual Unfreezing =====
                'gradual_unfreezing': False,  # NEW
                
                # カスタムスケジュール (オプション)
                # 指定しない場合はデフォルトスケジュール使用
                'unfreeze_schedule' : {
                    0:  "init",        # Epoch 0: Classifierのみ
                    5: "lstm",        # Epoch 15: LSTM層をunfreeze
                    10: "attention",   # Epoch 25: Attention層をunfreeze
                    15: "stage4",      # Epoch 35: CNN Stage4をunfreeze
                    20: "stage3",      # Epoch 45: CNN Stage3をunfreeze
                    25: "all",         # Epoch 55: 全層をunfreeze
                }
                                                            },
            # Soft Edit Distance設定
            'soft_edit_distance': {
                'enabled': False,
                'ctc_weight': 0,
                'soft_ed_weight': 1,
                'temperature': 0.3,
                'insert_cost': 1.0,
                'delete_cost': 1.0,
                'normalize': True,
                'warmup_epochs': 0
            },
            # データ設定
            'data': {
                'train_csv': '/home/bv20049/dataset/npz/zundadata/processed/final_train_cropp.csv',
                'valid_csv': '/home/bv20049/dataset/npz/zundadata/processed/final_valid_cropp.csv',
                'test_csv': 'data/test.csv',
                'batch_size': 32,
                'num_workers': 4,
                'max_length': 50,
                'input_size': 64,
                'include_n': False,
            },
            
            # 保存設定
            'save': {
                'checkpoint_dir': 'checkpoints/pattern_b_frame_attention',
                'log_dir': 'logs',
                'result_dir': '/home/bv20049/dataset/npz/zundadata/results/consonant_0202gatisoftpertukauze',
                'save_interval': 10
            },
            
            # その他
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'seed': 42,
            'debug': False,
            'analysis':{
                'analyze_ctc': False,
                'ctc_num_batches': 10  # 分析するバッチ数（省略可）
            }
        }
        
        self.config = self.default_config.copy()
        
        # 設定ファイル読み込み
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """設定ファイル読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = json.load(f)
            
            # 深いマージ
            self._deep_merge(self.config, loaded_config)
            print(f"✓ 設定ファイル読み込み完了: {config_path}")
            
        except Exception as e:
            print(f"⚠ 設定ファイル読み込みエラー: {e}")
            print("デフォルト設定を使用します")
    
    def save_config(self, save_path: str):
        """設定ファイル保存"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 設定ファイル保存完了: {save_path}")
    
    def _deep_merge(self, base_dict: dict, update_dict: dict):
        """辞書の深いマージ"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def print_attention_config(self):
        """Attention設定を表示"""
        print("\n" + "="*70)
        print("Attention Configuration")
        print("="*70)
        print(f"  Type:        {self.config['model']['attention_type']}")
        print(f"  Temperature: {self.config['model']['temperature']}")
        print(f"  Dual:        {self.config['model'].get('dual_attention', False)}")
        print("="*70 + "\n")
    
    def __getitem__(self, key):
        """設定値取得"""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """設定値設定"""
        self.config[key] = value
    
    def get(self, key, default=None):
        """設定値取得（デフォルト付き）"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default
        


def set_seed(seed: int = 42):
    """乱数シード固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ 乱数シード固定: {seed}")

def setup_logging(log_dir: str = 'logs', log_level: str = 'INFO'):
    """ログ設定"""
    os.makedirs(log_dir, exist_ok=True)
    
    # ログファイル名（タイムスタンプ付き）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'lip_reading_{timestamp}.log')
    
    # ログレベル設定
    level = getattr(logging, log_level.upper())
    
    # ロガー設定
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"ログ設定完了: {log_file}")
    
    return logger

def check_data_paths(config: Config) -> bool:
    """データパスの存在確認"""
    data_config = config['data']
    paths_to_check = ['train_csv', 'valid_csv']
    
    missing_paths = []
    for path_key in paths_to_check:
        path = data_config.get(path_key)
        if path and not os.path.exists(path):
            missing_paths.append(f"{path_key}: {path}")
    
    if missing_paths:
        print("⚠ 以下のデータファイルが見つかりません:")
        for path in missing_paths:
            print(f"  - {path}")
        return False
    
    print("✓ データファイル確認完了")
    return True

def load_loss_config():
    """
    loss_config.py から損失関数設定を読み込む
    
    Returns:
        dict: 損失関数の設定
            {
                'loss_type': str,
                'sed_temperature': float,
                'sed_learnable_indel': bool
            }
    
    Examples:
        >>> config = load_loss_config()
        >>> print(config['loss_type'])
        'differentiable_edit_distance'
    """
    try:
        from loss_config import get_loss_config, print_loss_config
        loss_config = get_loss_config()
        return loss_config
    except ImportError:
        print("⚠️  loss_config.py が見つかりません。デフォルト設定を使用します。")
        return {
            'loss_type': 'length_aware_ctc',
            'sed_temperature': 1.0,
            'sed_learnable_indel': False,
        }
    except Exception as e:
        print(f"⚠️  loss_config.py の読み込みエラー: {e}")
        return {
            'loss_type': 'length_aware_ctc',
            'sed_temperature': 1.0,
            'sed_learnable_indel': False,
        }


def apply_loss_config_to_args(args, loss_config=None):
    """
    loss_config の設定を args に適用
    コマンドライン引数が指定されている場合はそちらを優先
    
    Args:
        args: argparse.Namespace オブジェクト
        loss_config: 損失関数設定の辞書（Noneの場合は自動読み込み）
    
    Returns:
        dict: 最終的な損失関数設定
    
    Examples:
        >>> args = parser.parse_args()
        >>> final_config = apply_loss_config_to_args(args)
    """
    # loss_config が指定されていない場合は読み込む
    if loss_config is None:
        loss_config = load_loss_config()
    
    # コマンドライン引数で上書き
    if hasattr(args, 'loss_type') and args.loss_type:
        loss_config['loss_type'] = args.loss_type
        print(f"  → コマンドライン引数で上書き: loss_type={args.loss_type}")
    
    if hasattr(args, 'sed_temperature') and args.sed_temperature != 1.0:
        loss_config['sed_temperature'] = args.sed_temperature
        print(f"  → コマンドライン引数で上書き: sed_temperature={args.sed_temperature}")
    
    if hasattr(args, 'sed_learnable_indel') and args.sed_learnable_indel:
        loss_config['sed_learnable_indel'] = args.sed_learnable_indel
        print(f"  → コマンドライン引数で上書き: sed_learnable_indel={args.sed_learnable_indel}")
    
    return loss_config


def apply_loss_config_to_config_dict(config_dict, loss_config=None):
    """
    loss_config の設定を config 辞書に適用
    
    Args:
        config_dict: 設定辞書
        loss_config: 損失関数設定（Noneの場合は自動読み込み）
    
    Returns:
        dict: 更新された config 辞書
    
    Examples:
        >>> config = Config('config.yaml')
        >>> apply_loss_config_to_config_dict(config.config)
    """
    if loss_config is None:
        loss_config = load_loss_config()
    
    # training セクションに追加
    if 'training' not in config_dict:
        config_dict['training'] = {}
    
    config_dict['training']['loss_type'] = loss_config.get('loss_type', 'length_aware_ctc')
    config_dict['training']['sed_temperature'] = loss_config.get('sed_temperature', 1.0)
    config_dict['training']['sed_learnable_indel'] = loss_config.get('sed_learnable_indel', False)
    
    return config_dict


def print_loss_config_summary(loss_config):
    """
    損失関数設定のサマリーを表示
    
    Args:
        loss_config: 損失関数設定の辞書
    """
    print("\n" + "="*70)
    print("損失関数設定")
    print("="*70)
    print(f"Loss Type: {loss_config['loss_type']}")
    
    if loss_config['loss_type'] == 'differentiable_edit_distance':
        print(f"  Temperature: {loss_config['sed_temperature']}")
        print(f"  Learnable InDel: {loss_config['sed_learnable_indel']}")
        
        # 温度の説明
        temp = loss_config['sed_temperature']
        if temp <= 0.3:
            desc = "非常にminに近い（急峻な勾配）"
        elif temp <= 0.7:
            desc = "minに近い"
        elif temp <= 1.5:
            desc = "バランス良い（推奨）"
        elif temp <= 5.0:
            desc = "平均寄り"
        else:
            desc = "ほぼ平均（均等重み）"
        print(f"  → 挙動: {desc}")
    
    print("="*70 + "\n")


def get_loss_type_description(loss_type):
    """
    損失関数の説明を取得
    
    Args:
        loss_type: 損失関数のタイプ
    
    Returns:
        str: 説明文
    """
    descriptions = {
        'ctc': '標準CTC Loss',
        'length_aware_ctc': 'Length-Aware CTC Loss（短い系列の重み調整）',
        'differentiable_edit_distance': '微分可能な編集距離損失（PERと直接対応）',
    }
    return descriptions.get(loss_type, '不明な損失関数')
def check_gpu_availability():
    """GPU利用可能性確認"""
    print("\n" + "="*70)
    print("GPU Configuration")
    print("="*70)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"✓ GPU利用可能")
        print(f"  - GPU数:    {gpu_count}")
        print(f"  - 使用GPU:  {gpu_name}")
        print(f"  - メモリ:   {gpu_memory:.1f}GB")
        print("="*70 + "\n")
        return True
    else:
        print("⚠ GPU利用不可 - CPUで実行")
        print("="*70 + "\n")
        return False

class MetricsCalculator:
    """評価指標計算クラス"""
    
    @staticmethod
    def edit_distance(pred: List[int], target: List[int]) -> int:
        """編集距離（レーベンシュタイン距離）計算"""
        m, n = len(pred), len(target)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初期化
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # DP計算
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # 削除
                        dp[i][j-1],    # 挿入
                        dp[i-1][j-1]   # 置換
                    )
        
        return dp[m][n]
    
    @staticmethod
    def sequence_accuracy(pred: List[int], target: List[int]) -> float:
        """系列精度（完全一致率）"""
        return 1.0 if pred == target else 0.0
    
    @staticmethod
    def phoneme_accuracy(pred: List[int], target: List[int]) -> float:
        """音素レベル精度"""
        if len(target) == 0:
            return 1.0 if len(pred) == 0 else 0.0
        
        edit_dist = MetricsCalculator.edit_distance(pred, target)
        return max(0.0, 1.0 - edit_dist / len(target))
    
    @staticmethod
    def calculate_metrics(predictions: List[List[int]], targets: List[List[int]]) -> Dict[str, float]:
        """複数サンプルの評価指標計算"""
        if len(predictions) != len(targets):
            raise ValueError("予測と正解の数が一致しません")
        
        seq_accs = []
        phoneme_accs = []
        edit_distances = []
        
        for pred, target in zip(predictions, targets):
            seq_accs.append(MetricsCalculator.sequence_accuracy(pred, target))
            phoneme_accs.append(MetricsCalculator.phoneme_accuracy(pred, target))
            edit_distances.append(MetricsCalculator.edit_distance(pred, target))
        
        return {
            'sequence_accuracy': np.mean(seq_accs),
            'phoneme_accuracy': np.mean(phoneme_accs),
            'avg_edit_distance': np.mean(edit_distances),
            'total_samples': len(predictions)
        }

def create_directories(config: Config):
    """必要なディレクトリ作成"""
    dirs_to_create = [
        config.get('save.checkpoint_dir', 'checkpoints'),
        config.get('save.log_dir', 'logs'),
        config.get('save.result_dir', 'results')
    ]
    
    print("\n" + "="*70)
    print("Directory Setup")
    print("="*70)
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ {dir_path}")
    
    print("="*70 + "\n")

def save_results(results: Dict[str, Any], save_path: str):
    """結果保存"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # DataFrameに変換可能な部分
    if 'predictions' in results and 'targets' in results:
        df_data = {
            'text': results.get('texts', []),
            'target_phonemes': [str(target) for target in results['targets']],
            'pred_phonemes': [str(pred) for pred in results['predictions']],
            'accuracy': results.get('accuracies', [])
        }
        
        df = pd.DataFrame(df_data)
        csv_path = save_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"✓ 結果CSV保存: {csv_path}")
    
    # JSON形式でも保存
    # NumPy配列をリストに変換
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
            json_results[key] = [v.tolist() for v in value]
        else:
            json_results[key] = value
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 結果JSON保存: {save_path}")

def load_results(load_path: str) -> Dict[str, Any]:
    """結果読み込み"""
    with open(load_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✓ 結果読み込み完了: {load_path}")
    return results

class EarlyStopping:
    """Early Stopping実装"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.stopped_epoch = 0
    
    def __call__(self, current_loss: float, model: torch.nn.Module) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False

def print_model_info(model: torch.nn.Module, input_shape: tuple):
    """モデル情報表示"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*70)
    print("Model Information")
    print("="*70)
    print(f"総パラメータ数:        {total_params:,}")
    print(f"訓練可能パラメータ数:  {trainable_params:,}")
    print(f"入力形状:              {input_shape}")
    
    # メモリ使用量推定（概算）
    param_size = total_params * 4  # float32
    buffer_size = sum(p.numel() for p in model.buffers()) * 4
    total_size = (param_size + buffer_size) / 1024**2
    
    print(f"推定メモリ使用量:      {total_size:.1f}MB")
    print("="*70 + "\n")

def print_training_config(config: Config):
    """訓練設定を表示"""
    print("\n" + "="*70)
    print("Training Configuration")
    print("="*70)
    print(f"Epochs:              {config['training']['epochs']}")
    print(f"Batch Size:          {config['data']['batch_size']}")
    print(f"Learning Rate:       {config['training']['lr']}")
    print(f"Optimizer:           {config['training']['optimizer']}")
    print(f"Scheduler:           {config['training']['scheduler']}")
    print(f"Weight Decay:        {config['training']['weight_decay']}")
    print(f"Gradient Clip:       {config['training']['gradient_clip']}")
    print(f"Early Stop Patience: {config['training']['early_stopping_patience']}")
    print(f"Early Stop Metric:   {config['training'].get('early_stopping_metric', 'val_loss')}")
    print("="*70 + "\n")

def ctc_greedy_decode(log_probs: torch.Tensor, blank_id: int, input_lengths: torch.Tensor = None):
    with torch.no_grad():
        max_ids = torch.argmax(log_probs, dim=2)  # (T,N)
        T, N = max_ids.shape
        results = []
        for n in range(N):
            t_max = int(input_lengths[n].item()) if input_lengths is not None else T
            prev = None
            out = []
            for t in range(t_max):
                idx = int(max_ids[t, n].item())
                if idx == blank_id:
                    prev = None          # ★ blankでリセット（これが重要）
                    continue
                if idx != prev:
                    out.append(idx)
                prev = idx
            results.append(out)
        return results

# utils_pattern_b.py に追加
def ctc_beam_search_decode(log_probs, blank_id, input_lengths=None, beam_width=5):
    """
    改善版 CTC Beam Search (collapse対応)
    log_probs: (T, N, C)
    return: List[List[int]]
    """
    import math
    from collections import defaultdict
    
    T, N, C = log_probs.shape
    results = []
    
    for n in range(N):
        t_max = int(input_lengths[n].item()) if input_lengths is not None else T
        
        # beams: {prefix: (prob, last_char)}
        beams = {tuple(): (0.0, None)}
        
        for t in range(t_max):
            new_beams = defaultdict(lambda: (-float('inf'), None))
            
            for prefix, (log_p, last_char) in beams.items():
                for c in range(C):
                    log_prob_c = float(log_probs[t, n, c].item())
                    new_log_p = log_p + log_prob_c
                    
                    if c == blank_id:
                        # blank: prefixそのまま
                        key = prefix
                        if new_log_p > new_beams[key][0]:
                            new_beams[key] = (new_log_p, None)
                    else:
                        # CTC collapse: 同一文字連続は1つに
                        if last_char == c:
                            key = prefix
                        else:
                            key = prefix + (c,)
                        
                        if new_log_p > new_beams[key][0]:
                            new_beams[key] = (new_log_p, c)
            
            # 上位beam_width個を保持
            beams = dict(sorted(new_beams.items(), key=lambda x: x[1][0], reverse=True)[:beam_width])
        
        # 最良パス選択
        if beams:
            best_prefix = max(beams.items(), key=lambda x: x[1][0])[0]
            results.append(list(best_prefix))
        else:
            results.append([])
    
    return results

def ids_to_phonemes(id_seq_list, encoder):
    """
    ID列を音素列に変換する（バッチ対応）

    Args:
        id_seq_list (List[List[int]]): CTC decode後のID列リスト
        encoder: PhonemeEncoderインスタンス（decode_phonemesを持つ）

    Returns:
        List[List[str]]: 音素文字列のリスト
    """
    return [encoder.decode_phonemes(seq) for seq in id_seq_list]


def build_loaders_from_config(config):
    mode = config.get('model', {}).get('mode', 'consonant')

    res = create_dataloaders(
        train_csv_path=config['data']['train_csv'],
        valid_csv_path=config['data']['valid_csv'],
        batch_size=config['data'].get('batch_size', 16),
        num_workers=config['data'].get('num_workers', 0),
        cache_size=config['data'].get('cache_size', 100),
        augmentation_config=config.get('augmentation', None),
        max_length=config['data'].get('max_length', 40),
        mode=mode,
    )

    # 返り値が3 or 4 の両対応
    if isinstance(res, tuple) and len(res) == 4:
        train_loader, valid_loader, encoder, labels = res
    else:
        train_loader, valid_loader, encoder = res  # 旧仕様
        labels = getattr(encoder, 'vowels', None) or getattr(encoder, 'consonants', None) or []

    return train_loader, valid_loader, encoder, labels

def sync_num_classes_with_encoder(config: Dict[str, Any], encoder) -> None:
    nc = getattr(encoder, "num_classes", None)
    if callable(nc):
        nc = nc()
    if isinstance(nc, int) and nc > 0:
        config["model"]["num_classes"] = nc
        print(f"✓ num_classes を encoder に同期: {nc}")

def save_evaluation_report(
    metrics_dict,
    predictions,
    targets,
    labels,
    save_dir,
    mode='consonant',
    sample_results=None
):
    """評価結果を統合保存（テキスト/CSV/JSON）"""
    os.makedirs(save_dir, exist_ok=True)
    
    # === 1. テキストレポート ===
    txt_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("最終評価レポート\n")
        f.write("="*70 + "\n\n")
        
        # PER
        per_val = metrics_dict.get('per_per', metrics_dict.get('per', 0.0))
        f.write(f"PER (音素誤り率): {per_val:.2f}%\n")
        
        # 完全一致率
        if mode == 'consonant':
            exact = metrics_dict.get('exact_match_consonant_exact_match_rate', 0.0)
            f.write(f"完全一致率（子音列）: {exact:.2f}%\n")
        else:
            exact = metrics_dict.get('exact_match_vowel_rate', 0.0)
            f.write(f"完全一致率（母音列）: {exact:.2f}%\n")
        
        # 位置別正解率
        first_acc = metrics_dict.get('position_first_accuracy', 0.0)
        last_acc = metrics_dict.get('position_last_accuracy', 0.0)
        phoneme_type = '子音' if mode == 'consonant' else '母音'
        f.write(f"最初/最後の{phoneme_type}正解率: {first_acc:.2f}% / {last_acc:.2f}%\n\n")
        
        # グローバルエラー
        f.write("--- グローバルエラー分解 ---\n")
        global_S = metrics_dict.get('global_S', 0)
        global_D = metrics_dict.get('global_D', 0)
        global_I = metrics_dict.get('global_I', 0)
        avg_len_ref = metrics_dict.get('avg_len_ref', 0.0)
        avg_len_hyp = metrics_dict.get('avg_len_hyp', 0.0)
        total_ref = max(1, global_S + global_D + sum([len(t) for t in targets]))
        
        f.write(f"S/D/I: {global_S}/{global_D}/{global_I}\n")
        f.write(f"平均長(ref/hyp): {avg_len_ref:.2f} / {avg_len_hyp:.2f}\n\n")
        
        # 長さ別PER
        length_bucket = metrics_dict.get('length_bucket_per', {})
        if length_bucket:
            f.write("--- 長さ別PER (参照基準) ---\n")
            for k in ["1-1", "2-3", "4-6", "7-10", "11-20", "21-999"]:
                if k in length_bucket:
                    f.write(f"L={k:>6}: PER={length_bucket[k]:5.2f}%\n")
            f.write("\n")
        
        # サンプル結果
        if sample_results:
            f.write("--- Top10 エラーサンプル ---\n")
            for i, s in enumerate(sample_results[:10], 1):
                pred_str = ' '.join(s['predicted'][:20])
                tgt_str = ' '.join(s['target'][:20])
                f.write(f"{i:2d}) PER={s['per_ref']:>6.2f}%  [S/D/I]={s['S']}/{s['D']}/{s['I']}  "
                       f"pred: {pred_str} || tgt: {tgt_str}\n")
    
    print(f"✓ 評価レポート保存: {txt_path}")
    
    # === 2. サンプル結果CSV ===
    if sample_results:
        csv_path = os.path.join(save_dir, 'sample_results.csv')
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['rank', 'per_ref', 'per_display', 'S', 'D', 'I', 
                           'len_ref', 'len_hyp', 'predicted', 'target'])
            for i, s in enumerate(sample_results[:10], 1):
                writer.writerow([
                    i, s['per_ref'], s['per_display'], s['S'], s['D'], s['I'],
                    s['len_ref'], s['len_hyp'],
                    ' '.join(s['predicted']), ' '.join(s['target'])
                ])
        print(f"✓ サンプル結果CSV: {csv_path}")
    
    # === 3. 音素別メトリクスCSV ===
    from matrics_undefined import UnifiedEvaluationMetrics
    evaluator = UnifiedEvaluationMetrics()
    per_phoneme = evaluator.calculate_per_phoneme_metrics(predictions, targets, labels)
    
    csv_path = os.path.join(save_dir, 'per_phoneme_metrics.csv')
    import csv
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['phoneme', 'precision', 'recall', 'f1', 'support'])
        sorted_labels = sorted(labels, key=lambda l: -per_phoneme[l]['support'])
        for label in sorted_labels:
            r = per_phoneme[label]
            if r['support'] > 0:
                writer.writerow([label, f"{r['precision']:.2f}", 
                               f"{r['recall']:.2f}", f"{r['f1']:.2f}", r['support']])
    print(f"✓ 音素別メトリクスCSV: {csv_path}")

if __name__ == "__main__":
    # テスト用実行
    print("\n" + "="*70)
    print("utils_pattern_b.py モジュールテスト")
    print("="*70 + "\n")
    
    # 設定テスト
    config = Config()
    print(f"✓ Config作成成功")
    print(f"  - モデルタイプ:   {config['model']['model_type']}")
    print(f"  - クラス数:       {config['model']['num_classes']}")
    print(f"  - Attention Type: {config['model']['attention_type']}")
    print(f"  - Temperature:    {config['model']['temperature']}")
    print(f"  - Dual Attention: {config['model']['dual_attention']}")
    
    # Attention設定表示
    config.print_attention_config()
    
    # シード設定テスト
    set_seed(42)
    
    # GPU確認テスト
    check_gpu_availability()
    
    # 評価指標テスト
    pred = [1, 2, 3, 1]
    target = [1, 2, 1, 3]
    metrics = MetricsCalculator.calculate_metrics([pred], [target])
    print(f"✓ 評価指標テスト:")
    for key, value in metrics.items():
        print(f"  - {key}: {value}")
    
    print("\n" + "="*70)
    print("✓ utils_pattern_b.py モジュール正常動作確認完了")
    print("="*70 + "\n")