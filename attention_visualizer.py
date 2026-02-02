#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention可視化モジュール(修正版)

【修正内容】
- ラベル文字数に応じてAttention上位フレームを抽出・表示
"""

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK JP']
matplotlib.rcParams['axes.unicode_minus'] = False

import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import os
# ===== PER helper (Levenshtein-based) =====
def _levenshtein_sdi(ref, hyp):
    """
    ref, hyp: list[str](音素列)
    return (S, D, I)
    """
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[0]*(m+1) for _ in range(n+1)]  # 0:diag, 1:up(del), 2:left(ins)
    for i in range(1, n+1):
        dp[i][0] = i; bt[i][0] = 1
    for j in range(1, m+1):
        dp[0][j] = j; bt[0][j] = 2
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            a = dp[i-1][j-1] + cost
            b = dp[i-1][j] + 1
            c = dp[i][j-1] + 1
            if a <= b and a <= c:
                dp[i][j] = a; bt[i][j] = 0
            elif b <= c:
                dp[i][j] = b; bt[i][j] = 1
            else:
                dp[i][j] = c; bt[i][j] = 2
    i, j = n, m
    S = D = I = 0
    while i > 0 or j > 0:
        code = bt[i][j]
        if i > 0 and j > 0 and code == 0:
            if ref[i-1] != hyp[j-1]:
                S += 1
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or code == 1):
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I


def _sequence_per_percent(ref, hyp):
    """PER[%] = (S+D+I)/max(len(ref),len(hyp)) * 100"""
    S, D, I = _levenshtein_sdi(ref, hyp)
    denom = max(len(ref), len(hyp), 1)
    return 100.0 * (S + D + I) / denom
# ===== end helper =====


class AttentionVisualizer:
    """Attention重み可視化クラス"""
    
    def __init__(self, model, phoneme_encoder, device='cuda'):
        """
        Args:
            model: 訓練済みモデル
            phoneme_encoder: 音素エンコーダー
            device: 使用デバイス
        """
        self.model = model
        self.phoneme_encoder = phoneme_encoder
        self.device = device
        
        # モデルを評価モードに
        self.model.eval()
    
    def _text_to_phonemes(self, text) -> List[str]:
        """
        テキストを音素列に変換
        
        Args:
            text: 入力テキスト(文字列またはリスト)
        
        Returns:
            音素のリスト
        """
        # 0. すでに音素列(リスト)の場合はそのまま返す
        if isinstance(text, list):
            return text
        
        # phoneme_encoderの種類に応じて処理を分岐
        
        # 1. encode_textメソッドがある場合
        if hasattr(self.phoneme_encoder, 'encode_text'):
            return self.phoneme_encoder.encode_text(text)
        
        # 2. text_to_phonemesメソッドがある場合(文字列のみ受け付ける)
        if hasattr(self.phoneme_encoder, 'text_to_phonemes'):
            if isinstance(text, str):
                return self.phoneme_encoder.text_to_phonemes(text)
        
        # 3. デフォルト: 文字を分割して音素として扱う
        if isinstance(text, str):
            print(f"Warning: phoneme_encoderに音素変換メソッドがありません。文字列を分割します: {text}")
            return list(text)
        
        # 4. その他の型の場合はエラー
        raise TypeError(f"text must be str or list, got {type(text)}")
    
    def _phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """
        音素列をIDに変換
        
        Args:
            phonemes: 音素のリスト
        
        Returns:
            音素IDのリスト
        """
        # encode_phonemesメソッドがある場合
        if hasattr(self.phoneme_encoder, 'encode_phonemes'):
            return self.phoneme_encoder.encode_phonemes(phonemes)
        
        # phoneme_to_idがある場合
        if hasattr(self.phoneme_encoder, 'phoneme_to_id'):
            return [self.phoneme_encoder.phoneme_to_id.get(p, 0) for p in phonemes]
        
        # デフォルト: id2phonemeの逆引き
        if hasattr(self.phoneme_encoder, 'id2phoneme'):
            phoneme_to_id = {v: k for k, v in self.phoneme_encoder.id2phoneme.items()}
            return [phoneme_to_id.get(p, 0) for p in phonemes]
        
        raise AttributeError("phoneme_encoderに音素→ID変換メソッドがありません")
    
    def visualize_attention_with_evaluation(
        self,
        video: torch.Tensor,
        text,
        save_path: Optional[str] = None,
        evaluator=None
    ) -> Dict:
        """..."""
        self.model.eval()
        
        with torch.no_grad():
            if video.dim() == 4:
                video = video.unsqueeze(0)
            
            video = video.to(self.device)
            
            # Attention取得
            try:
                outputs = self.model(video, return_attention=True)
                
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    outputs, attention_weights = outputs
                    if attention_weights is not None:
                        attention_weights = attention_weights.cpu().numpy()
                else:
                    attention_weights = None
                    
            except (TypeError, AttributeError):
                outputs = self.model(video)
                attention_weights = None
            
            # model.attention_weightsから取得試行
            if attention_weights is None:
                if hasattr(self.model, 'attention_weights') and self.model.attention_weights is not None:
                    attention_weights = self.model.attention_weights.cpu().numpy()
            
            # attention_weightsがNoneの場合の処理追加
            if attention_weights is None:
                print("  ⚠ Attention weights取得失敗 - 可視化スキップ")
                # 予測だけ実行して返す
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
                log_probs = log_probs.permute(1, 0, 2)
                _, max_indices = torch.max(log_probs, dim=2)
                max_indices = max_indices.squeeze(1).cpu().numpy()
                
                pred_ids = []
                prev_id = None
                for idx in max_indices:
                    if idx != self.phoneme_encoder.blank_id and idx != prev_id:
                        pred_ids.append(int(idx))
                    prev_id = idx
                
                pred_phonemes = self.phoneme_encoder.decode_phonemes(pred_ids)
                target_phonemes = self._text_to_phonemes(text)
                is_correct = (pred_phonemes == target_phonemes)
                
                return {
                    'predicted': pred_phonemes,
                    'target': target_phonemes,
                    'is_correct': is_correct,
                    'attention_weights': None
                }
            
            # 残りの処理は既存通り
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            log_probs = torch.nn.functional.log_softmax(outputs, dim=-1)
            log_probs = log_probs.permute(1, 0, 2)
            
            _, max_indices = torch.max(log_probs, dim=2)
            max_indices = max_indices.squeeze(1).cpu().numpy()
            
            pred_ids = []
            prev_id = None
            for idx in max_indices:
                if idx != self.phoneme_encoder.blank_id and idx != prev_id:
                    pred_ids.append(int(idx))
                prev_id = idx
            
            pred_phonemes = self.phoneme_encoder.decode_phonemes(pred_ids)
            target_phonemes = self._text_to_phonemes(text)
            is_correct = (pred_phonemes == target_phonemes)
            
            result = {
                'predicted': pred_phonemes,
                'target': target_phonemes,
                'is_correct': is_correct,
                'attention_weights': attention_weights
            }
            
            if evaluator is not None:
                try:
                    eval_result = evaluator.evaluate_single(pred_phonemes, target_phonemes)
                    result.update(eval_result)
                except Exception as e:
                    print(f"⚠ 評価エラー: {e}")
            
            if save_path and attention_weights is not None:
                original_text_for_display = text if isinstance(text, str) else None
                self._plot_attention(
                    video=video.squeeze(0).cpu().numpy(), 
                    attention_weights=attention_weights, 
                    pred_phonemes=pred_phonemes, 
                    target_phonemes=target_phonemes, 
                    is_correct=is_correct, 
                    save_path=save_path, 
                    original_text=original_text_for_display
                )
            
            return result
    
    def _plot_attention(
        self,
        video: np.ndarray,
        attention_weights: Optional[np.ndarray],
        pred_phonemes: List[str],
        target_phonemes: List[str],
        is_correct: bool,
        save_path: str,
        original_text: str = None
    ):
        """Attention可視化(ローマ字/音素表示対応)"""
        import scipy.signal as signal
        
        num_frames = video.shape[0]
        has_attention = attention_weights is not None
        if has_attention:
            weights = attention_weights.squeeze()
        
        if has_attention:
            fig = plt.figure(figsize=(20, 16))
            gs = fig.add_gridspec(5, 1, height_ratios=[0.3, 1.0, 2.5, 1.0, 1.0], hspace=0.4)
            ax_title = fig.add_subplot(gs[0])
            ax_uniform = fig.add_subplot(gs[1])
            ax_consonant = fig.add_subplot(gs[2])
            ax_attention = fig.add_subplot(gs[3])
            ax_heatmap = fig.add_subplot(gs[4])
        else:
            fig, ax_uniform = plt.subplots(1, 1, figsize=(20, 4))
            ax_title = None
            ax_consonant = None
            ax_attention = None
            ax_heatmap = None
        
        # ========== タイトルエリア(元の文章:ローマ字/音素) ==========
        if ax_title is not None and original_text:
            ax_title.axis('off')
            # カタカナをローマ字に変換
            try:
                import pykakasi
                kks = pykakasi.kakasi()
                result = kks.convert(original_text)
                romaji = ''.join([item['hepburn'] for item in result])
            except:
                # pykakasi未インストール時は元テキストをそのまま
                romaji = original_text
            
            phoneme_str = ' '.join(target_phonemes)
            display_text = f"{romaji} / {phoneme_str}"
            ax_title.text(0.5, 0.5, f"元の文章: {display_text}", 
                        ha='center', va='center', 
                        fontsize=16, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8, edgecolor='blue', linewidth=2))
        
        # ========== 均等フレーム ==========
        num_display_uniform = min(10, num_frames)
        indices_uniform = np.linspace(0, num_frames - 1, num_display_uniform, dtype=int)
        thumbnails_uniform = []
        for idx in indices_uniform:
            frame = video[idx]
            if frame.shape[0] in [1, 3]:
                frame = np.transpose(frame, (1, 2, 0))
            if frame.shape[-1] == 1:
                frame = frame.squeeze(-1)
            thumbnails_uniform.append(frame)
        strip_uniform = np.concatenate(thumbnails_uniform, axis=1)
        if strip_uniform.max() > 1.0:
            strip_uniform = strip_uniform / 255.0
        ax_uniform.imshow(strip_uniform, cmap='gray' if len(strip_uniform.shape) == 2 else None)
        ax_uniform.set_title("Uniform Sampling (10 frames)", fontsize=12, fontweight='bold', pad=10)
        ax_uniform.axis('off')
        
        # ========== ピーク検出フレーム ==========
        if ax_consonant is not None and has_attention:
            num_target_phonemes = len(target_phonemes)
            
            if num_target_phonemes > 0:
                peaks, _ = signal.find_peaks(weights, height=weights.mean(), distance=5, prominence=0.003)
                
                if len(peaks) == 0:
                    top_indices = np.argsort(weights)[-num_target_phonemes:]
                elif len(peaks) < num_target_phonemes:
                    remaining = num_target_phonemes - len(peaks)
                    non_peak_mask = np.ones(len(weights), dtype=bool)
                    non_peak_mask[peaks] = False
                    extra = np.argsort(weights[non_peak_mask])[-remaining:]
                    extra_indices = np.where(non_peak_mask)[0][extra]
                    top_indices = np.concatenate([peaks, extra_indices])
                else:
                    top_indices = peaks[np.argsort(weights[peaks])[-num_target_phonemes:]]
                
                top_indices_sorted = sorted(top_indices)
                
                thumbnails_consonant = []
                for idx in top_indices_sorted:
                    frame = video[idx]
                    if frame.shape[0] in [1, 3]:
                        frame = np.transpose(frame, (1, 2, 0))
                    if frame.shape[-1] == 1:
                        frame = frame.squeeze(-1)
                    thumbnails_consonant.append(frame)
                
                strip_consonant = np.concatenate(thumbnails_consonant, axis=1)
                if strip_consonant.max() > 1.0:
                    strip_consonant = strip_consonant / 255.0
                
                ax_consonant.imshow(strip_consonant, cmap='gray' if len(strip_consonant.shape) == 2 else None)
                
                frame_width = strip_consonant.shape[1] / len(top_indices_sorted)
                img_height = strip_consonant.shape[0]
                
                for i, idx in enumerate(top_indices_sorted):
                    x_pos = (i + 0.5) * frame_width
                    
                    # 対応子音(上)
                    if i < len(target_phonemes):
                        ax_consonant.text(x_pos, -20, target_phonemes[i], 
                                        ha='center', va='bottom', 
                                        fontsize=13, color='blue', fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.95, edgecolor='blue', linewidth=2.5))
                    
                    # フレーム番号(下) - 位置を下げる
                    ax_consonant.text(x_pos, img_height + 25, f"F{idx}", 
                                    ha='center', va='top', 
                                    fontsize=10, color='white', fontweight='bold',
                                    bbox=dict(boxstyle='round,pad=0.4', facecolor='red', alpha=0.85, edgecolor='none'))
                
                title_lines = [
                    f"Target Consonants: {' '.join(target_phonemes)}",
                    f"Pred Consonants: {' '.join(pred_phonemes) if pred_phonemes else '(none)'}"
                ]
                
                ax_consonant.set_title("\n".join(title_lines), fontsize=12, fontweight='bold', color='green' if is_correct else 'red', pad=30)
                ax_consonant.set_ylim(img_height + 45, -30)
                ax_consonant.axis('off')
        
        # ========== Attention折れ線 ==========
        if ax_attention is not None and has_attention:
            frames_idx = np.arange(len(weights))
            ax_attention.plot(frames_idx, weights, linewidth=2.5, color='#2E86DE', marker='o', markersize=4)
            ax_attention.fill_between(frames_idx, weights, alpha=0.3, color='#54A0FF')
            if num_target_phonemes > 0 and len(peaks) > 0:
                selected_peaks = [p for p in peaks if p in top_indices_sorted]
                if selected_peaks:
                    ax_attention.scatter(selected_peaks, weights[selected_peaks], color='red', s=100, zorder=5, marker='*', label=f'Detected peaks ({len(selected_peaks)})')
            peak_idx = np.argmax(weights)
            ax_attention.scatter([peak_idx], [weights[peak_idx]], color='darkred', s=120, zorder=6, marker='*', label=f'Global peak: F{peak_idx}')
            ax_attention.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
            ax_attention.set_ylabel('Attention Weight', fontsize=11, fontweight='bold')
            ax_attention.set_title('Attention Weights', fontsize=11, fontweight='bold')
            ax_attention.grid(True, alpha=0.3, linestyle='--')
            ax_attention.legend(loc='upper right')
            ax_attention.set_xlim(-0.5, len(weights) - 0.5)
        
        # ========== ヒートマップ ==========
        if ax_heatmap is not None and has_attention:
            weights_2d = weights.reshape(1, -1)
            im = ax_heatmap.imshow(weights_2d, cmap='hot', aspect='auto', interpolation='nearest')
            ax_heatmap.set_xlabel('Frame Index', fontsize=11, fontweight='bold')
            ax_heatmap.set_ylabel('Attention', fontsize=11, fontweight='bold')
            ax_heatmap.set_title('Attention Heatmap', fontsize=11, fontweight='bold')
            ax_heatmap.set_yticks([])
            cbar = plt.colorbar(im, ax=ax_heatmap, orientation='horizontal', pad=0.1, fraction=0.05)
            cbar.set_label('Weight', fontsize=10)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {save_path}")

def visualize_attention_with_samples(
    model,
    data_loader,
    phoneme_encoder,
    device='cuda',
    num_samples=5,
    save_dir='results/attention_visualization',
    evaluator=None
) -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    visualizer = AttentionVisualizer(model, phoneme_encoder, device)
    
    if evaluator is None:
        try:
            from matrics_undefined import CTCAwareEvaluator
            evaluator = CTCAwareEvaluator()
        except ImportError:
            print("⚠ CTCAwareEvaluator not found. Using simple evaluation.")
            evaluator = None
    
    correct_samples, incorrect_samples = [], []
    total_samples, correct_count = 0, 0
    
    for batch in data_loader:
        videos = batch['video']
        targets = batch['target']
        target_lengths = batch['target_length']
        texts = batch.get('text', None)
        
        batch_size = videos.size(0)
        target_offset = 0
        
        for i in range(batch_size):
            if total_samples >= num_samples * 2:
                break
            
            video = videos[i]
            target_len = int(target_lengths[i].item())
            target_ids = targets[target_offset:target_offset + target_len].cpu().numpy()
            target_phonemes = phoneme_encoder.decode_phonemes(target_ids)
            
            # 元テキスト取得
            original_text = None
            if texts is not None:
                original_text = texts[i] if isinstance(texts, list) else texts[i].item() if hasattr(texts[i], 'item') else str(texts[i])
            
            save_path = os.path.join(save_dir, f'attention_sample_{total_samples:03d}.png')
            result = visualizer.visualize_attention_with_evaluation(
                video=video,
                text=original_text if original_text else target_phonemes,
                save_path=save_path,
                evaluator=evaluator
            )
            
            S, D, I = _levenshtein_sdi(result['target'], result['predicted'])
            denom = max(len(result['target']), len(result['predicted']), 1)
            per = 100.0 * (S + D + I) / denom
            
            sample_info = {
                'sample_id': total_samples,
                'predicted': result['predicted'],
                'target': result['target'],
                'is_correct': result['is_correct'],
                'save_path': save_path,
                'per': round(per, 2),
                'S': int(S), 'D': int(D), 'I': int(I),
            }
            
            if result['is_correct']:
                correct_samples.append(sample_info)
                correct_count += 1
            else:
                incorrect_samples.append(sample_info)
            
            total_samples += 1
            target_offset += target_len
        
        if total_samples >= num_samples * 2:
            break

    incorrect_samples.sort(key=lambda s: -s['per'])

    accuracy = correct_count / total_samples if total_samples > 0 else 0.0
    result = {
        'total_samples': total_samples,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'correct_samples': correct_samples,
        'incorrect_samples': incorrect_samples,
        'save_dir': save_dir
    }

    # ===== レポート出力 =====
    print(f"\n{'='*70}")
    print(f"Attention可視化 + サンプル評価レポート")
    print(f"{'='*70}")

    print(f"\n【全体統計】")
    print(f"  総サンプル数: {total_samples}")
    print(f"  正解数: {correct_count}")
    print(f"  不正解数: {total_samples - correct_count}")
    print(f"  精度: {accuracy*100:.1f}%")
    print(f"  保存先: {save_dir}")

    # 正解サンプル(最大5件)
    if correct_samples:
        print(f"\n【正解サンプル】 ({min(5, len(correct_samples))}件)")
        for i, s in enumerate(correct_samples[:5], 1):
            pred_str = ' '.join(s['predicted'])
            tgt_str  = ' '.join(s['target'])
            print(f"  {i}. ✓ PER={s['per']:.2f}%  予測={pred_str}, 正解={tgt_str}")
            print(f"     ファイル: {os.path.basename(s['save_path'])}")

    # 不正解サンプル(TOP10)
    if incorrect_samples:
        topn = min(10, len(incorrect_samples))
        print(f"\n【不正解サンプル】 (TOP {topn})")
        for i, s in enumerate(incorrect_samples[:topn], 1):
            pred_str = ' '.join(s['predicted'])
            tgt_str  = ' '.join(s['target'])
            print(f"  {i}. ✗ PER={s['per']:.2f}%  予測={pred_str}, 正解={tgt_str}")
            print(f"     ファイル: {os.path.basename(s['save_path'])}")
            # エラー分析(集合差)
            missing = set(s['target']) - set(s['predicted'])
            extra   = set(s['predicted']) - set(s['target'])
            if missing:
                print(f"     欠落音素: {missing}")
            if extra:
                print(f"     余分音素: {extra}")

    # Attention統計(案内)
    if total_samples > 0:
        print(f"\n【Attention統計】")
        print(f"  可視化画像を確認してください: {save_dir}")
        print(f"  注目度が高いフレームを確認できます")
    print(f"{'='*70}\n")

    # ===== JSON 保存(PERとSDIを含む)=====
    import json
    to_json = {
        'summary': {
            'total_samples': total_samples,
            'correct_count': correct_count,
            'incorrect_count': total_samples - correct_count,
            'accuracy': accuracy
        },
        'correct_samples': [
            {
                'sample_id': s['sample_id'],
                'predicted': s['predicted'],
                'target': s['target'],
                'file': os.path.basename(s['save_path']),
                'per': s['per'], 'S': s['S'], 'D': s['D'], 'I': s['I'],
            } for s in correct_samples
        ],
        'incorrect_samples': [
            {
                'sample_id': s['sample_id'],
                'predicted': s['predicted'],
                'target': s['target'],
                'file': os.path.basename(s['save_path']),
                'per': s['per'], 'S': s['S'], 'D': s['D'], 'I': s['I'],
                'missing_phonemes': list(set(s['target']) - set(s['predicted'])),
                'extra_phonemes': list(set(s['predicted']) - set(s['target'])),
            } for s in incorrect_samples
        ]
    }
    json_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(to_json, f, indent=2, ensure_ascii=False)
    print(f"✓ 評価結果をJSON保存: {json_path}\n")

    return result


if __name__ == "__main__":
    print("Attention可視化モジュール(修正版)")