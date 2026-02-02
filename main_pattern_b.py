#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
読唇術プロジェクト メインスクリプト (Pattern B対応・評価指標統一版)
- Pattern B: CNN → LSTM → Temporal Attention
- Sigmoid/Softmax簡単切り替え
- vowelモデルは Attention なし(NoAttn)版に差し替え可能
"""

import os
import argparse
import torch
from pathlib import Path
import numpy as np

# モジュールインポート
print("モジュール読み込み中...")
from phoneme_analysis_unified import analyze_phonemes_unified
from matrics_undefined import UnifiedEvaluationMetrics
from phoneme_aware_per import run_evaluation
try:
    from enhanced_metrics import EnhancedEvaluationMetrics
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    print("Warning: enhanced_metrics not found. Using standard metrics only.")
    ENHANCED_METRICS_AVAILABLE = False
from dataset import create_dataloaders
from train import LipReadingTrainer, evaluate_model
from utils_pattern_b import (
    Config, set_seed, setup_logging, check_data_paths,
    check_gpu_availability, create_directories, save_results,
    print_model_info, MetricsCalculator, build_loaders_from_config, sync_num_classes_with_encoder
)
from ctc_analyzer import (
    analyze_blank_rate,
    analyze_consecutive_blanks, 
    analyze_phoneme_duration,
    analyze_blank_between_phonemes,
    visualize_ctc_analysis,
    print_analysis_summary
)
# ===== モード非依存の統一サマリー出力 =====
def _compute_first_last_accuracy(pred_seqs, tgt_seqs):
    n = 0; first_ok = 0; last_ok = 0
    for p, t in zip(pred_seqs, tgt_seqs):
        if len(t) == 0:
            continue
        n += 1
        if len(p) > 0 and p[0] == t[0]:
            first_ok += 1
        if len(p) > 0 and p[-1] == t[-1]:
            last_ok += 1
    if n == 0:
        return 0.0, 0.0
    return 100.0 * first_ok / n, 100.0 * last_ok / n

def print_unified_summary(final_raw, encoder):
    """
    final_raw: evaluate_modelの戻り値のraw部（predictions/targets必須）
    表示を '子音側のフォーマット' に統一して出力し、mmにも統一キーを追加して返す
    """
    from matrics_undefined import UnifiedEvaluationMetrics
    evalr = UnifiedEvaluationMetrics()

    preds = final_raw.get('predictions', [])
    tgts  = final_raw.get('targets', [])

    # 1) PER
    per = final_raw.get('per_per', final_raw.get('PER', None))
    if per is None:
        per = evalr.sequence_per(preds, tgts)  # ％を返す実装想定

    # 2) 完全一致率（系列）
    exact = final_raw.get('exact_match_consonant_exact_match_rate',
                          final_raw.get('exact_match_vowel_rate', None))
    if exact is None:
        # collapse済み前提のpreds/tgtsならそのまま、未collapseなら内部でcollapseする実装に依存
        exact = evalr.sequence_exact_match_rate(preds, tgts)  # ％を返す実装想定

    # 3) 最初/最後トークン正解率（モード非依存）
    # 文字列リストが前提のはずだが、もしintならencoderで変換
    if preds and preds[0] and isinstance(preds[0][0], int):
        preds_tok = [encoder.ids_to_symbols(x) for x in preds]
        tgts_tok  = [encoder.ids_to_symbols(x) for x in tgts]
    else:
        preds_tok, tgts_tok = preds, tgts

    first_acc, last_acc = _compute_first_last_accuracy(preds_tok, tgts_tok)

    # ---- 表示（子音側の体裁に統一）----
    print(f"PER (音素誤り率):     {per:.2f}%")
    print(f"完全一致率（系列）:    {exact:.2f}%")
    print(f"最初/最後のトークン:   {first_acc:.2f}% / {last_acc:.2f}%")

    # mmに統一キーも足して返す（保存jsonも揃う）
    final_raw.setdefault('per_per', per)
    final_raw['exact_match_sequence_rate'] = exact
    final_raw['position_first_accuracy'] = first_acc
    final_raw['position_last_accuracy']  = last_acc
    return final_raw

# =========================================================
# 引数と設定処理
# =========================================================
def parse_arguments():
    """コマンドライン引数パース"""
    parser = argparse.ArgumentParser(description='読唇術モデル訓練・評価 (Pattern B)')

    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'test'],
                        default='train', help='実行モード')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='設定ファイルパス')

    parser.add_argument('--train_csv', type=str, help='訓練用CSVパス')
    parser.add_argument('--valid_csv', type=str, help='検証用CSVパス')
    parser.add_argument('--test_csv', type=str, help='テスト用CSVパス')
    parser.add_argument('--checkpoint', type=str, help='チェックポイントパス')

    # Attention設定（子音モデルや注意ありモデルで有効）
    parser.add_argument('--attention-type', type=str, choices=['sigmoid', 'softmax'],
                        help='Attention type: sigmoid or softmax')
    parser.add_argument('--temperature', type=float, help='Attention temperature (0.1-1.0)')

    parser.add_argument('--use_softper', action='store_true',
                    help='Use SoftPER loss term (CTC + lambda*SoftPER)')
    parser.add_argument('--lambda_softper', type=float, default=None,
                        help='Weight for SoftPER term')
    parser.add_argument('--softper_tau', type=float, default=None,
                        help='Softmin temperature for SoftPER DP')

    # 訓練パラメータ
    parser.add_argument('--epochs', type=int, help='エポック数')
    parser.add_argument('--batch_size', type=int, help='バッチサイズ')
    parser.add_argument('--lr', type=float, help='学習率')

    parser.add_argument('--analyze_ctc', action='store_true', 
                    help='Analyze CTC predictions (blank rate, durations, etc.)')
    parser.add_argument('--ctc_analysis_output', type=str, default='ctc_analysis',
                    help='Output directory for CTC analysis results')

    # その他
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'auto'],
                        default='auto', help='使用デバイス')
    parser.add_argument('--seed', type=int, default=42, help='乱数シード')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    return parser.parse_args()


def setup_config(args):
    """設定セットアップ"""
    config = Config(args.config if os.path.exists(args.config) else None)

    # 引数上書き
    if args.train_csv: config['data']['train_csv'] = args.train_csv
    if args.valid_csv: config['data']['valid_csv'] = args.valid_csv
    if args.test_csv:  config['data']['test_csv']  = args.test_csv
    if args.epochs:    config['training']['epochs'] = args.epochs
    if args.batch_size:config['data']['batch_size']  = args.batch_size
    if args.lr:        config['training']['lr']      = args.lr

    if args.attention_type: config['model']['attention_type'] = args.attention_type
    if args.temperature:    config['model']['temperature']    = args.temperature

    if args.device == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config['device'] = args.device

    # SoftPER
    if 'training' not in config.config:
        config.config['training'] = {}

    if args.use_softper:
        config.config['training']['use_softper'] = True
    if args.lambda_softper is not None:
        config.config['training']['lambda_softper'] = float(args.lambda_softper)
    if args.softper_tau is not None:
        config.config['training']['softper_tau'] = float(args.softper_tau)

    config['debug'] = args.debug
    return config


# =========================================================
# モデル生成
# =========================================================
def create_model_from_config(config, num_classes_from_encoder=None):
    """
    mode=='vowel' の場合は Attentionなしの CompactVowelLipReader_NoAttn を使用
    mode=='consonant' の場合は model_typeに応じてモデル選択
    """
    mode = config['model'].get('mode', 'consonant')
    num_classes = int(num_classes_from_encoder or config['model']['num_classes'])

    if mode == 'vowel':
        from model_compact_vowel import CompactVowelLipReader_NoAttn
        return CompactVowelLipReader_NoAttn(
            num_classes=num_classes,
            dropout=config['model'].get('dropout_rate', 0.2),
        )
    
    else:  # consonant
        model_type = config['model'].get('model_type', 'deep_cnn')
        
        # ===== Deep CNN Model =====
        if model_type == 'deep_cnn':
            from model_deep_cnn import create_deep_cnn_model
            return create_deep_cnn_model(
                num_classes=num_classes,
                dropout=config['model']['dropout_rate'],
                lstm_layers=config['model'].get('lstm_layers', 3),
                lstm_hidden=config['model'].get('lstm_hidden', 256),
            )
        
        # ===== Pattern B (既存) =====
        elif model_type == 'pattern_b_frame_attention':
            from model_pattern_b import create_improved_pattern_b_model
            return create_improved_pattern_b_model(
                num_classes=num_classes,
                dropout_rate=config['model']['dropout_rate'],
                attention_type=config['model']['attention_type'],
                temperature=config['model']['temperature'],
                dual_attention=config['model'].get('dual_attention', False)
            )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    '''
    # main_pattern_b.py の create_model_from_config 内
    if mode == 'consonant':
        from model_consonant_transformer import create_consonant_transformer_model
        return create_consonant_transformer_model(
            num_classes=num_classes,
            dropout=config['model']['dropout_rate'],
            d_model=config['model'].get('d_model', 256),
            nhead=config['model'].get('nhead', 8),
            num_layers=config['model'].get('num_layers', 4)
        )
    '''
# =========================================================
# 学習メイン
# =========================================================
def train_model(config, args):
    """モデル訓練"""
    config.print_attention_config()

    print("\nデータローダー作成中...")
    train_loader, valid_loader, phoneme_encoder, labels = build_loaders_from_config(config.config)
    sync_num_classes_with_encoder(config.config, phoneme_encoder)
    
    mode = config.config['model'].get('mode', 'consonant')
    phoneme_type = '母音' if mode == 'vowel' else '子音'
    
    # ... データセット統計表示 (既存コード) ...

    print("\nモデル作成中...")
    model = create_model_from_config(
        config.config,
        num_classes_from_encoder=phoneme_encoder.num_classes()
    )

    # ★ デバッグ追加
    print(f"\n[DEBUG Model Output]")
    print(f"  encoder.num_classes(): {phoneme_encoder.num_classes()}")
    print(f"  config num_classes: {config.config['model']['num_classes']}")
    dummy_input = torch.randn(1, 40, 1, 64, 64).to(config['device'])
    model.to(config['device'])
    dummy_out = model(dummy_input)
    print(f"  model output shape: {dummy_out.shape}")  # (1, T, C) のCを確認

    # ... モデル情報表示 (既存コード) ...

    # ===== Trainer作成 =====
    use_length_aware = config.config['training'].get('use_length_aware_loss', True)
    use_gradual_unfreeze = config.config['training'].get('gradual_unfreezing', True)
    
    print(f"\n{'='*60}")
    print(f"Trainer Configuration")
    print(f"{'='*60}")
    print(f"Length-Aware Loss: {use_length_aware}")
    print(f"Gradual Unfreezing: {use_gradual_unfreeze}")
    print(f"{'='*60}\n")
    
    mode = config.config['model'].get('mode', 'consonant')

    tr_cfg = config['training']
    use_softper = tr_cfg.get('use_softper', False)
    lambda_softper = tr_cfg.get('lambda_softper', 0.05)
    softper_tau = tr_cfg.get('softper_tau', 0.2)
    separate_softper_loss = tr_cfg.get('separate_softper_loss', True)

    print("\n" + "="*70)
    print("SoftPER Wiring Check (main -> trainer)")
    print("="*70)
    print(f"config.training.use_softper   = {tr_cfg.get('use_softper', None)} -> {use_softper}")
    print(f"config.training.lambda_softper = {lambda_softper}")
    print(f"config.training.softper_tau    = {softper_tau}")
    print("="*70 + "\n")

    trainer = LipReadingTrainer(
    model=model,
    phoneme_encoder=phoneme_encoder,
    device=config['device'],
    save_dir=config['save']['checkpoint_dir'],
    result_dir=config['save']['result_dir'],
    early_stopping_metric=config['training'].get('early_stopping_metric', 'val_loss'),
    use_length_aware_loss=use_length_aware,
    gradual_unfreezing=use_gradual_unfreeze,
    mode=mode,                                    # ← 追加
    use_softper=use_softper,                      # ← 追加
    lambda_softper=lambda_softper,                # ← 追加
    softper_tau=softper_tau,                      # ← 追加
    separate_softper_loss=separate_softper_loss,  # ← 追加
)
    
    # ===== カスタムスケジュール設定 =====
    if use_gradual_unfreeze and 'unfreeze_schedule' in config.config['training']:
        custom_schedule = config.config['training']['unfreeze_schedule']
        
        # dict型に変換（YAMLからの読み込みで文字列キーになっている可能性）
        if custom_schedule:
            schedule_dict = {}
            for k, v in custom_schedule.items():
                epoch_num = int(k) if isinstance(k, str) else k
                schedule_dict[epoch_num] = str(v)
            
            print(f"\n🔧 Setting CUSTOM unfreeze schedule from config.yaml...")
            trainer.set_unfreeze_schedule(schedule_dict)
    
    # Optimizer設定
    trainer.setup_optimizer(
        optimizer_type=config['training']['optimizer'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler設定
    scheduler_params = config['training'].get('scheduler_params', {})
    trainer.setup_scheduler(
        scheduler_type=config['training']['scheduler'],
        **scheduler_params
    )

    # チェックポイント読み込み
    start_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"\nチェックポイント読み込み: {args.checkpoint}")
        start_epoch = trainer.load_checkpoint(args.checkpoint)

    # ===== 学習開始前の最終確認 =====
    print(f"\n{'='*70}")
    print(f"Training Start Confirmation")
    print(f"{'='*70}")
    print(f"Model: {config.config['model'].get('model_type', 'unknown')}")
    print(f"Gradual Unfreezing: {use_gradual_unfreeze}")
    if use_gradual_unfreeze and trainer.unfreeze_schedule:
        print(f"Active Schedule Epochs: {sorted(trainer.unfreeze_schedule.keys())}")
    print(f"Total Epochs: {config['training']['epochs']}")
    print(f"Early Stopping Metric: {config['training'].get('early_stopping_metric', 'val_loss')}")
    print(f"{'='*70}\n")

    # 学習ループ
    print("\n" + "=" * 70)
    print("訓練開始")
    print("=" * 70)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs=config['training']['epochs'],
        early_stopping_patience=config['training'].get('early_stopping_patience', 20)
    )
    
    # ========== 学習曲線保存 ==========
    history_plot_path = os.path.join(config['save']['result_dir'], 'training_history.png')
    trainer.plot_history(save_path=history_plot_path)
    
    # ===============================================
    # ✅ 統一された最終評価（途中評価と同一Evaluatorを使用）
    # ===============================================
    print("\n" + "=" * 70)
    print("最終評価指標計算中（validateと同一ロジック）...")
    print("=" * 70)
    
    final = evaluate_model(
        model, valid_loader, phoneme_encoder, config['device'],
        show_samples=True, num_samples=10
    )
    mm = print_unified_summary(final['raw'], phoneme_encoder)
    
    # サンプル表示
    evaluator = UnifiedEvaluationMetrics()
    print("\nサンプル結果（母音/子音共通）:")
    evaluator.print_sample_results(
        final['raw']['predictions'],
        final['raw']['targets'],
        num_samples=10,
        apply_collapse=True,
        show_correct=True,
        show_incorrect=True,
        vowel_mode=(mode == 'vowel')
    )
    
    # 音素別メトリクス表示
    evaluator.print_per_phoneme_metrics(
        final['raw']['predictions'],
        final['raw']['targets'],
        labels,
        mode=mode
    )
    
    # 最終評価サマリー
    print("\n" + "=" * 70)
    print(" 最終評価サマリー")
    print("=" * 70)
    
    per_val = mm.get('per_per', mm.get('PER', 0.0))
    print(f"PER (音素誤り率):     {per_val:.2f}%")
    
    if mode == 'consonant':
        print(f"子音完全一致率:       {mm.get('exact_match_consonant_exact_match_rate', 0.0):.2f}%")
        print(f"最初/最後の子音正解率: {mm.get('position_first_accuracy', 0.0):.2f}% / {mm.get('position_last_accuracy', 0.0):.2f}%")
    else:
        if 'exact_match_vowel_rate' in mm:
            print(f"母音完全一致率:       {mm['exact_match_vowel_rate']:.2f}%")
        if 'position_first_accuracy' in mm:
            print(f"最初/最後の母音正解率: {mm.get('position_first_accuracy', 0.0):.2f}% / {mm.get('position_last_accuracy', 0.0):.2f}%")

    # ★ 音響的PER評価
    if ENHANCED_METRICS_AVAILABLE:
        print("\n" + "=" * 70)
        print(" 音響的類似度を考慮した評価")
        print("=" * 70)
        
        evaluator_acoustic = EnhancedEvaluationMetrics(
            use_acoustic=True,
            mode=mode,
            phoneme_encoder=phoneme_encoder
        )
        
        acoustic_result = evaluator_acoustic.calculate_acoustic_per(
            predictions=final['raw']['predictions'],
            targets=final['raw']['targets'],
            apply_collapse=True
        )
        
        print(f"標準PER:           {acoustic_result['standard_per']:.2f}%")
        print(f"音響的PER:         {acoustic_result['acoustic_per']:.2f}%")
        diff = acoustic_result['standard_per'] - acoustic_result['acoustic_per']
        print(f"差分:              {diff:+.2f}% {'(類似音素の混同が多い)' if diff > 5 else ''}")
        print(f"\nエラー内訳:")
        print(f"  置換:     {acoustic_result['substitutions']:,} (重み付き: {acoustic_result['weighted_substitutions']:.2f})")
        print(f"  削除:     {acoustic_result['deletions']:,}")
        print(f"  挿入:     {acoustic_result['insertions']:,}")
        print(f"  総音素数: {acoustic_result['total_phonemes']:,}")

    # ========== 評価結果統合保存 ==========
    from utils_pattern_b import save_evaluation_report
    save_evaluation_report(
        metrics_dict=mm,
        predictions=final['raw']['predictions'],
        targets=final['raw']['targets'],
        labels=labels,
        save_dir=config['save']['result_dir'],
        mode=mode,
        sample_results=final['raw'].get('sample_results', [])
    )
    
    # ========== 音素特性を考慮したPER計算（追加） ==========
    print("\n" + "=" * 70)
    print("音素特性を考慮したPER計算中...")
    print("=" * 70)
    
    try:
        from phoneme_aware_per import run_evaluation
        
        per_results = run_evaluation(
            predictions_list=final['raw']['predictions'],
            targets_list=final['raw']['targets'],
            result_dir=config['save']['result_dir']
        )
        
        # ===== 修正：すでに%なので100倍しない =====
        print(f"✓ Phoneme-Aware PER: {per_results['overall']['phoneme_aware_per']:.2f}%")
        print(f"✓ Hard PER:          {per_results['overall']['hard_per']:.2f}%")
        print(f"✓ 改善率:            {per_results['overall']['improvement_rate']:.2f}%")
        
    except Exception as e:
        print(f"⚠ 音素特性を考慮したPER計算でエラー: {e}")
        import traceback
        traceback.print_exc()
    # ===============================================
    # Attention可視化
    # ===============================================
    try:
        print("\n" + "=" * 70)
        print("Attention可視化 + サンプル評価")
        print("=" * 70)
        from attention_visualizer import visualize_attention_with_samples

        has_attn_attr = hasattr(model, "attention_weights")
        if mode == 'vowel' and not has_attn_attr:
            print("（NoAttnモデルのためAttention可視化をスキップ）")
        else:
            attention_result = visualize_attention_with_samples(
                model=model,
                data_loader=valid_loader,
                phoneme_encoder=phoneme_encoder,
                device=config['device'],
                num_samples=5,
                save_dir=config['save']['result_dir']
            )
            print(f"\n✓ Attention可視化完了")
            print(f"  - 可視化画像: {len(attention_result['correct_samples']) + len(attention_result['incorrect_samples'])}枚")
            print(f"  - 正解率: {attention_result['accuracy']*100:.1f}%")
    except Exception as e:
        print(f"⚠ Attention可視化エラー: {e}")
        import traceback; traceback.print_exc()

    # ========== 音素別詳細分析（ルート直下に保存） ==========
    print("\n音素別詳細分析を実行中...")
    analyze_phonemes_unified(
        predictions=final['raw'].get('predictions', []),
        targets=final['raw'].get('targets', []),
        labels=labels,
        save_dir=config['save']['result_dir'],  # サブフォルダなし
        apply_collapse=True
    )

    # ========== CTC分析 ==========
    if config.config.get('analysis', {}).get('analyze_ctc', False):
        print("\n" + "="*60)
        print("Starting CTC Analysis...")
        print("="*60)
        
        # バッチ数を設定から取得
        num_batches = config.config.get('analysis', {}).get('ctc_num_batches', 10)
        
        model.eval()
        all_outputs = []
        all_decoded = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                # バッチが辞書形式の場合
                if isinstance(batch, dict):
                    videos = batch['video']
                    input_lengths = batch['input_length']
                else:
                    videos = batch[0]
                    input_lengths = batch[2]
                
                videos = videos.to(config['device'])
                outputs = model(videos)  # (B, T, C)
                
                # log_probsに変換
                log_probs = torch.log_softmax(outputs, dim=-1)  # (B, T, C)
                
                # ★ (B, T, C) → (T, B, C) に必ず変換
                log_probs = log_probs.permute(1, 0, 2)  # (T, N, C)
                
                all_outputs.append(log_probs.cpu())
                
                # ctc_greedy_decodeを使用
                
                from utils_pattern_b import ctc_greedy_decode, ids_to_phonemes
                
                decoded_ids = ctc_greedy_decode(
                    log_probs,
                    blank_id=phoneme_encoder.blank_id,
                    input_lengths=input_lengths
                )

                # ctc_greedy_decodeを使用
                from utils_pattern_b import ctc_greedy_decode
                
                decoded_ids = ctc_greedy_decode(
                    log_probs,
                    blank_id=phoneme_encoder.blank_id,
                    input_lengths=input_lengths
                )
                
                all_decoded.extend(decoded_ids)  # ID列を保存
                
                if batch_idx >= num_batches - 1:
                    break
        
        
        # 全バッチを結合
        all_outputs = torch.cat(all_outputs, dim=1)  # (T, N_total, C)
        
        # 分析実行
        results = {
            'blank_rate': analyze_blank_rate(all_outputs, blank_id=0),
            'consecutive_blanks': analyze_consecutive_blanks(all_outputs, blank_id=0),
            'phoneme_duration': analyze_phoneme_duration(all_outputs, all_decoded, blank_id=0),
            'blanks_between': analyze_blank_between_phonemes(all_outputs, blank_id=0)
        }
        
        # 結果出力
        print_analysis_summary(results)
        
        # 可視化保存
        ctc_output_dir = os.path.join(config['save']['result_dir'], 'ctc_analysis')
        os.makedirs(ctc_output_dir, exist_ok=True)
        visualize_ctc_analysis(results, 
                            os.path.join(ctc_output_dir, 'ctc_analysis.png'))
        
        print(f"\nAnalysis plot saved to: {ctc_output_dir}/ctc_analysis.png")

    print("\n" + "=" * 70)
    print("✓ すべての処理が完了しました")
    print("=" * 70)


# =========================================================
# 評価モード
# =========================================================
def evaluate_model_mode(config, args):
    """評価モード"""
    print("\n" + "=" * 70)
    print("評価モード")
    print("=" * 70)

    # DataLoader作成
    print("\nテストデータローダー作成中...")
    mode = config['model'].get('mode', 'consonant')
    _, test_loader, phoneme_encoder, labels = create_dataloaders(
        train_csv_path=config['data'].get('train_csv'),
        valid_csv_path=config['data']['test_csv'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        augmentation_config=None,
        max_length=config['data'].get('max_length', 40),
        mode=mode,
    )
    
    sync_num_classes_with_encoder(config.config, phoneme_encoder)
    
    # モデル作成・読み込み
    print(f"\nモデル作成中... (mode={mode}, num_classes={phoneme_encoder.num_classes()})")
    model = create_model_from_config(
        config.config,
        num_classes_from_encoder=phoneme_encoder.num_classes()
    )
    model.to(config['device'])
    
    if not args.checkpoint:
        raise ValueError("評価モードではチェックポイント(--checkpoint)が必須です")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"チェックポイントが見つかりません: {args.checkpoint}")

    print(f"\nチェックポイント読み込み: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=config['device'])
    
    # 出力次元チェック
    if 'model_state_dict' in checkpoint:
        head_keys = [k for k in checkpoint['model_state_dict'].keys() if k.endswith('classifier.3.weight')]
        if head_keys:
            ckpt_out = checkpoint['model_state_dict'][head_keys[0]].shape[0]
            if ckpt_out != phoneme_encoder.num_classes():
                raise ValueError(
                    f"Checkpoint出力次元({ckpt_out})とエンコーダ({phoneme_encoder.num_classes()})が不一致です。"
                )
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    # 評価実行
    print("\n最終評価を実行中...")
    final = evaluate_model(
        model, test_loader, phoneme_encoder, config['device'],
        show_samples=True, num_samples=10
    )
    mm = print_unified_summary(final['raw'], phoneme_encoder)
    
    # サンプル表示
    phoneme_type = '母音' if mode == 'vowel' else '子音'
    print(f"\nサンプル結果（{phoneme_type}モード）:")
    evaluator = UnifiedEvaluationMetrics()
    evaluator.print_sample_results(
        final['raw']['predictions'],
        final['raw']['targets'],
        num_samples=10,
        apply_collapse=True,
        show_correct=True,
        show_incorrect=True,
        vowel_mode=(mode == 'vowel')
    )
    
    # ★ 音素別メトリクス
    # 音素別メトリクス表示
    evaluator = UnifiedEvaluationMetrics()
    evaluator.print_per_phoneme_metrics(
        final['raw']['predictions'],
        final['raw']['targets'],
        labels,
        mode=mode
    )
    
    # 結果サマリー
    print("\n" + "=" * 70)
    print("テストセット評価結果")
    print("=" * 70)
    per_val = mm.get('per_per', mm.get('PER', 0.0))
    print(f"PER (音素誤り率):     {per_val:.2f}%")

    if mode == 'consonant':
        print(f"完全一致率（子音列）:  {mm.get('exact_match_consonant_exact_match_rate', 0.0):.2f}%")
    else:
        if 'exact_match_vowel_rate' in mm:
            print(f"完全一致率（母音列）:  {mm.get('exact_match_vowel_rate', 0.0):.2f}%")
    
    # 音素別分析
    analysis_dir = os.path.join(config['save']['result_dir'], 'phoneme_analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    analysis = analyze_phonemes_unified(
        predictions=final['raw'].get('predictions', []),
        targets=final['raw'].get('targets', []),
        labels=labels,
        save_dir=analysis_dir,
        top_k=5,
        plot_confusion=True,
    )

    print("\n--- 音素別分析 ---")
    print(f"Overall Acc: {analysis.get('overall_accuracy', 0.0)*100:.2f}%")
    print(f"Macro   Acc: {analysis.get('macro_accuracy', 0.0)*100:.2f}%")

    # ★ 音響的PER評価
    if ENHANCED_METRICS_AVAILABLE:
        print("\n" + "=" * 70)
        print(" 音響的類似度を考慮した評価")
        print("=" * 70)
        
        evaluator_acoustic = EnhancedEvaluationMetrics(
            use_acoustic=True,
            mode=mode,
            phoneme_encoder=phoneme_encoder
        )
        
        acoustic_result = evaluator_acoustic.calculate_acoustic_per(
            predictions=final['raw']['predictions'],
            targets=final['raw']['targets'],
            apply_collapse=True
        )
        
        print(f"標準PER:           {acoustic_result['standard_per']:.2f}%")
        print(f"音響的PER:         {acoustic_result['acoustic_per']:.2f}%")
        diff = acoustic_result['standard_per'] - acoustic_result['acoustic_per']
        print(f"差分:              {diff:+.2f}% {'(類似音素の混同が多い)' if diff > 5 else ''}")
        print(f"\nエラー内訳:")
        print(f"  置換:     {acoustic_result['substitutions']:,} (重み付き: {acoustic_result['weighted_substitutions']:.2f})")
        print(f"  削除:     {acoustic_result['deletions']:,}")
        print(f"  挿入:     {acoustic_result['insertions']:,}")
        print(f"  総音素数: {acoustic_result['total_phonemes']:,}")

        # ========== 音素特性を考慮したPER計算（追加） ==========
    print("\n" + "=" * 70)
    print("音素特性を考慮したPER計算中...")
    print("=" * 70)
    
    try:
        from phoneme_aware_per import run_evaluation
        
        per_results = run_evaluation(
            predictions_list=final['raw']['predictions'],
            targets_list=final['raw']['targets'],
            result_dir=config['save']['result_dir']
        )
    except Exception as e:
        print(f"⚠ エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ 評価完了")
    return mm

# =========================================================
# エントリポイント
# =========================================================
def main():
    args = parse_arguments()
    config = setup_config(args)
    set_seed(config['seed'])
    check_gpu_availability()
    create_directories(config)

    if not check_data_paths(config):
        print("エラー: データファイルが見つかりません")
        return

    if args.mode == 'train':
        train_model(config, args)
    elif args.mode in ['eval', 'test']:
        evaluate_model_mode(config, args)
    else:
        raise ValueError(f"未知のモード: {args.mode}")

    print("\n処理完了")

    

if __name__ == "__main__":
    main()