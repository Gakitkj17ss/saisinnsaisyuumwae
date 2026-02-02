#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拡張評価指標モジュール（プロジェクト統合版・正確版IPA対応）
既存のUnifiedEvaluationMetricsを拡張し、音響的類似度を組み込む

【使い方】
既存コード:
    from matrics_undefined import UnifiedEvaluationMetrics
    evaluator = UnifiedEvaluationMetrics()

拡張版に切り替え:
    from enhanced_metrics import EnhancedEvaluationMetrics
    evaluator = EnhancedEvaluationMetrics(use_acoustic=True)  # True/Falseで切り替え
"""

import sys
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# 既存モジュール
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, '/mnt/project')
from matrics_undefined import UnifiedEvaluationMetrics

try:
    import Levenshtein
except ImportError:
    print("Warning: python-Levenshtein not installed. Installing...")
    import subprocess
    subprocess.run(['pip', 'install', 'python-Levenshtein', '--break-system-packages', '-q'])
    import Levenshtein

# 音素類似度（永田論文完全準拠版）
from phoneme_similarity_experiment import (
    AccurateIPAConsonantFeatures,
    AccurateIPAVowelFeatures
)

# EnhancedPhonemeSimilarity を正確版ベースで再定義
class EnhancedPhonemeSimilarity:
    """
    拡張版音素類似度（永田論文完全準拠版）
    子音: 24次元IPA特徴ベクトル
    母音: 3次元IPA特徴ベクトル
    """
    def __init__(self):
        self.ipa_con = AccurateIPAConsonantFeatures()
        self.ipa_vow = AccurateIPAVowelFeatures()

        # 子音グループ
        self.nasal = {'n', 'm', 'N'}
        self.fricative = {'s', 'h', 'z'}
        self.plosive = {'k', 't', 'p', 'g', 'd', 'b'}

        # 母音集合
        self.vowels = {'a', 'i', 'u', 'e', 'o'}

    def _is_vowel(self, p: str) -> bool:
        return p in self.vowels

    def acoustic_similarity(self, p1: str, p2: str) -> float:
        if p1 == p2:
            return 1.0

        # ===== 母音同士 =====
        if self._is_vowel(p1) and self._is_vowel(p2):
            sim = self.ipa_vow.similarity(p1, p2, weighted=True)

            # 母音の微調整ボーナス
            rounded = {'o', 'u'}
            if (p1 in rounded) and (p2 in rounded):
                sim = min(1.0, sim + 0.05)
            elif (p1 not in rounded) and (p2 not in rounded):
                sim = min(1.0, sim + 0.03)

            return max(0.0, min(1.0, sim))

        # ===== 子音同士 =====
        ipa_sim = self.ipa_con.similarity(p1, p2, weighted=True)

        bonus = 0.0
        if (p1 in self.nasal and p2 in self.nasal):
            bonus = 0.10
        elif (p1 in self.fricative and p2 in self.fricative):
            bonus = 0.10
        elif (p1 in self.plosive and p2 in self.plosive):
            bonus = 0.05

        # 有声/無声ペアの調整
        voicing_pairs = [('k','g'),('s','z'),('t','d'),('p','b')]
        for v1, v2 in voicing_pairs:
            if (p1==v1 and p2==v2) or (p1==v2 and p2==v1):
                bonus = -0.15
                break

        return max(0.0, min(1.0, ipa_sim + bonus))

def resolve_phoneme_list(phoneme_encoder, mode: str):
    """modeに応じて混同行列などで使う音素リストを返す"""
    mode = (mode or '').lower()
    if mode == 'vowel':
        v = getattr(phoneme_encoder, 'vowels', None)
        return list(v) if v else ['a', 'i', 'u', 'e', 'o']
    else:
        c = getattr(phoneme_encoder, 'consonants', None)
        return list(c) if c else ['k', 's', 't', 'n', 'h', 'm', 'y', 'r', 'w', 'g', 'z', 'd', 'b', 'p', 'N']


class EnhancedEvaluationMetrics(UnifiedEvaluationMetrics):
    """
    音響的類似度を組み込んだ拡張評価指標
    
    【特徴】
    - 既存のUnifiedEvaluationMetricsを継承
    - use_acousticフラグで通常版/拡張版を切り替え
    - 既存メソッドはすべて使用可能
    - 永田論文完全準拠のIPA特徴ベクトル使用
    
    【新規メソッド】
    - calculate_acoustic_per(): 音響的PER計算
    - analyze_confusion_patterns(): 混同パターン分析
    - build_confusion_matrix(): 混同行列作成
    - print_acoustic_metrics(): 音響的評価の表示
    """
    
    def __init__(self, use_acoustic: bool = True, mode: str = 'consonant', phoneme_encoder=None):
        super().__init__()
        self.use_acoustic = use_acoustic
        self.mode = mode
        self.phoneme_encoder = phoneme_encoder

        if use_acoustic:
            self.enhanced_sim = EnhancedPhonemeSimilarity()
            print(f"✓ Enhanced metrics enabled (Accurate IPA features) | mode={self.mode}")
        else:
            self.enhanced_sim = None

    def calculate_acoustic_per(self, 
                              predictions: List[List[str]], 
                              targets: List[List[str]],
                              apply_collapse: bool = True) -> Dict[str, float]:
        """
        音響的PER計算
        
        Returns:
            {
                'standard_per': 通常PER,
                'acoustic_per': 音響的PER (use_acoustic=Trueの時のみ),
                'substitutions': 置換数,
                'weighted_substitutions': 重み付き置換数,
                'deletions': 削除数,
                'insertions': 挿入数,
                'total_phonemes': 総音素数,
                'substitution_details': 置換詳細
            }
        """
        total_substitutions = 0
        weighted_substitutions = 0.0
        total_deletions = 0
        total_insertions = 0
        total_phonemes = 0
        substitution_details = []
        
        for pred, target in zip(predictions, targets):
            pred_processed = self.preprocess_sequence(
                pred, apply_collapse=apply_collapse, consonants_only=False
            )
            target_processed = self.preprocess_sequence(
                target, apply_collapse=apply_collapse, consonants_only=False
            )
            
            ops = Levenshtein.editops(pred_processed, target_processed)
            
            for op in ops:
                op_type, pred_idx, target_idx = op
                
                if op_type == 'replace':
                    p1 = pred_processed[pred_idx]
                    p2 = target_processed[target_idx]
                    
                    # 音響的類似度
                    if self.use_acoustic and self.enhanced_sim:
                        similarity = self.enhanced_sim.acoustic_similarity(p1, p2)
                        weight = 1.0 - similarity
                    else:
                        similarity = 0.0
                        weight = 1.0
                    
                    total_substitutions += 1
                    weighted_substitutions += weight
                    
                    substitution_details.append({
                        'pred': p1,
                        'target': p2,
                        'similarity': similarity,
                        'weight': weight
                    })
                    
                elif op_type == 'delete':
                    total_deletions += 1
                elif op_type == 'insert':
                    total_insertions += 1
            
            total_phonemes += len(target_processed)
        
        # 通常PER
        total_errors = total_substitutions + total_deletions + total_insertions
        standard_per = (total_errors / max(total_phonemes, 1)) * 100
        
        # 音響的PER
        acoustic_errors = weighted_substitutions + total_deletions + total_insertions
        acoustic_per = (acoustic_errors / max(total_phonemes, 1)) * 100
        
        result = {
            'standard_per': standard_per,
            'substitutions': total_substitutions,
            'deletions': total_deletions,
            'insertions': total_insertions,
            'total_phonemes': total_phonemes,
            'total_errors': total_errors,
            'substitution_details': substitution_details
        }
        
        if self.use_acoustic:
            result['acoustic_per'] = acoustic_per
            result['weighted_substitutions'] = weighted_substitutions
        
        return result
    
    def analyze_confusion_patterns(self, 
                                   predictions: List[List[str]], 
                                   targets: List[List[str]],
                                   top_k: int = 10) -> Dict:
        """
        混同パターン分析
        
        Returns:
            {
                'top_confusions': 混同トップK,
                'total_substitutions': 総置換数
            }
        """
        result = self.calculate_acoustic_per(predictions, targets)
        details = result['substitution_details']
        
        pair_counts = defaultdict(lambda: {
            'count': 0, 
            'total_similarity': 0.0,
            'total_weight': 0.0
        })
        
        for d in details:
            key = (d['pred'], d['target'])
            pair_counts[key]['count'] += 1
            pair_counts[key]['total_similarity'] += d['similarity']
            pair_counts[key]['total_weight'] += d['weight']
        
        sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1]['count'])
        
        top_pairs = []
        for (p1, p2), stats in sorted_pairs[:top_k]:
            avg_sim = stats['total_similarity'] / stats['count']
            avg_weight = stats['total_weight'] / stats['count']
            
            top_pairs.append({
                'pair': f"{p1}→{p2}",
                'pred': p1,
                'target': p2,
                'count': stats['count'],
                'avg_similarity': avg_sim,
                'avg_weight': avg_weight
            })
        
        return {
            'top_confusions': top_pairs,
            'total_substitutions': result['substitutions']
        }
    
    def build_confusion_matrix(self, 
                           predictions: List[List[str]], 
                           targets: List[List[str]],
                           phoneme_list: Optional[List[str]] = None) -> Tuple[np.ndarray, Dict]:
        """
        混同行列作成（子音/母音どちらもOK）
        phoneme_list を渡さない場合:
        - 予測/正解に出現したトークンから自動で語彙を作る（頻度順）
        """
        # 1) vocab自動推定
        if phoneme_list is None:
            # まず mode + encoder から確定（vowelなら5母音、consonantなら子音集合）
            if self.phoneme_encoder is not None:
                phoneme_list = resolve_phoneme_list(self.phoneme_encoder, self.mode)
            else:
                # encoderが無い場合は出現語彙から自動生成（保険）
                vocab = set()
                for pred, target in zip(predictions, targets):
                    vocab.update(self.ctc_collapse(pred))
                    vocab.update(self.ctc_collapse(target))
                phoneme_list = sorted(list(vocab))


        phoneme_to_idx = {p: i for i, p in enumerate(phoneme_list)}
        n = len(phoneme_list)
        matrix = np.zeros((n, n), dtype=int)

        total_errors = 0
        for pred, target in zip(predictions, targets):
            pred = self.ctc_collapse(pred)
            target = self.ctc_collapse(target)

            ops = Levenshtein.editops(pred, target)
            for op_type, pred_idx, target_idx in ops:
                if op_type == 'replace':
                    p1 = pred[pred_idx]
                    p2 = target[target_idx]
                    if p1 in phoneme_to_idx and p2 in phoneme_to_idx:
                        i = phoneme_to_idx[p2]  # true
                        j = phoneme_to_idx[p1]  # pred
                        matrix[i, j] += 1
                        total_errors += 1

        # 混同ペア上位
        most_confused = []
        for i in range(n):
            for j in range(n):
                if i != j and matrix[i, j] > 0:
                    sim = 0.0
                    if self.use_acoustic and self.enhanced_sim:
                        sim = self.enhanced_sim.acoustic_similarity(
                            phoneme_list[i], phoneme_list[j]
                        )
                    most_confused.append({
                        'true': phoneme_list[i],
                        'pred': phoneme_list[j],
                        'count': int(matrix[i, j]),
                        'similarity': float(sim)
                    })
        most_confused.sort(key=lambda x: -x['count'])

        summary = {
            'total_substitutions': total_errors,
            'most_confused_pairs': most_confused[:10],
            'phoneme_list': phoneme_list
        }
        return matrix, summary
    
    def print_acoustic_metrics(self, 
                          predictions: List[List[str]], 
                          targets: List[List[str]],
                          apply_collapse: bool = True,
                          show_confusion_analysis: bool = True,
                          top_k: int = 10):
        """
        音響的評価指標の表示
        """
        print(f"\n{'='*70}")
        print(f"{'拡張評価指標 (音響的類似度・正確版IPA)' if self.use_acoustic else '標準評価指標'}")
        print(f"{'='*70}")
        
        # 音響的PER
        result = self.calculate_acoustic_per(predictions, targets, apply_collapse)
        
        print(f"\n【音素誤り率 (PER)】")
        print(f"  標準PER:       {result['standard_per']:.2f}%")
        
        if self.use_acoustic and 'acoustic_per' in result:
            diff = result['standard_per'] - result['acoustic_per']
            print(f"  音響的PER:     {result['acoustic_per']:.2f}%")
            print(f"  差分:          {diff:+.2f}% {'(類似音素の混同が多い)' if diff > 5 else ''}")
        
        print(f"\n【エラー内訳】")
        print(f"  置換: {result['substitutions']:,}")
        if self.use_acoustic:
            print(f"    重み付き: {result['weighted_substitutions']:.2f}")
        print(f"  削除: {result['deletions']:,}")
        print(f"  挿入: {result['insertions']:,}")
        print(f"  総音素数: {result['total_phonemes']:,}")
        
        # PER計算を追加
        total_errors = result['substitutions'] + result['deletions'] + result['insertions']
        per = (total_errors / max(1, result['total_phonemes'])) * 100
        print(f"\n【計算されたPER】")
        print(f"  PER: {per:.2f}%")
        print(f"  総エラー数: {total_errors}")
        print(f"  総音素数: {result['total_phonemes']}")
        
        # 混同パターン分析
        if show_confusion_analysis and result['substitutions'] > 0:
            patterns = self.analyze_confusion_patterns(predictions, targets, top_k)
            
            print(f"\n【混同パターン Top {min(top_k, len(patterns['top_confusions']))}】")
            print(f"{'ペア':8s} {'回数':>6s} {'類似度':>8s} {'重み':>8s}")
            print("-" * 35)
            
            for p in patterns['top_confusions'][:top_k]:
                sim_str = f"{p['avg_similarity']:.3f}" if self.use_acoustic else "N/A"
                weight_str = f"{p['avg_weight']:.3f}" if self.use_acoustic else "N/A"
                print(f"{p['pair']:8s} {p['count']:6d} {sim_str:>8s} {weight_str:>8s}")
        
        print(f"{'='*70}\n")
    
    def save_confusion_matrix(self, 
                        predictions: List[List[str]], 
                        targets: List[List[str]],
                        output_path: str,
                        phoneme_list: Optional[List[str]] = None,
                        normalize: bool = False):
        """混同行列を画像として保存（高=緑、低=赤の配色）"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np
        import os

        matrix, summary = self.build_confusion_matrix(predictions, targets, phoneme_list)
        labels = summary['phoneme_list']

        # 正規化
        disp = matrix.astype(np.float32)
        if normalize:
            row_sum = disp.sum(axis=1, keepdims=True)
            disp = np.divide(disp, np.maximum(row_sum, 1.0))

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        
        # カスタムカラーマップ: 赤(低) → 黄 → 緑(高)
        colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('red_yellow_green', colors, N=n_bins)
        
        im = ax.imshow(disp, aspect='auto', cmap=cmap, vmin=0)

        ax.set_title("Confusion Matrix (True vs Predicted)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)

        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)

        # グリッド線
        ax.set_xticks(np.arange(len(labels)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

        # 数値表示
        threshold = disp.max() / 2.0 if disp.max() > 0 else 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = disp[i, j]
                if normalize:
                    txt = f"{v:.2f}" if v > 0.01 else ""
                else:
                    txt = f"{int(matrix[i, j])}" if matrix[i, j] > 0 else ""
                if txt:
                    color = 'white' if v > threshold else 'black'
                    ax.text(j, i, txt, ha='center', va='center', 
                        fontsize=9, color=color, fontweight='bold')

        # カラーバー
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Count (Higher = More Errors)' if not normalize else 'Ratio', 
                    rotation=270, labelpad=20)
        
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"✓ Confusion matrix saved: {output_path}")
        
        # サマリー表示
        if summary['most_confused_pairs']:
            print(f"\n最も混同されやすいペア:")
            for i, p in enumerate(summary['most_confused_pairs'][:5], 1):
                sim_str = f" (類似度={p['similarity']:.3f})" if self.use_acoustic else ""
                print(f"  {i}. {p['true']}→{p['pred']}: {p['count']:3d}回{sim_str}")


# ==============================================================================
# 便利関数: 既存コードを最小限の変更で拡張版に切り替え
# ==============================================================================
def get_evaluator(use_acoustic: bool = False, mode: str = 'consonant', phoneme_encoder=None):
    """
    評価器を取得（切り替え簡単）
    
    使い方:
        evaluator = get_evaluator(use_acoustic=True)   # 拡張版
        evaluator = get_evaluator(use_acoustic=False)  # 通常版
    """
    return EnhancedEvaluationMetrics(
        use_acoustic=use_acoustic,
        mode=mode,
        phoneme_encoder=phoneme_encoder
    )


if __name__ == "__main__":
    print("="*70)
    print("Enhanced Metrics Module Test (Accurate IPA Version)")
    print("="*70)
    
    # テストデータ
    predictions = [
        ['g', 'k', 'k'],  # k→g置換
        ['z', 's', 't'],  # s→z置換
        ['k', 'g'],       # k削除
    ]
    targets = [
        ['k', 'k', 'k'],
        ['s', 's', 't'],
        ['k', 'g', 'k'],
    ]
    
    # 通常版
    print("\n【通常版】")
    evaluator_standard = EnhancedEvaluationMetrics(use_acoustic=False)
    evaluator_standard.print_acoustic_metrics(predictions, targets, show_confusion_analysis=True)
    
    # 拡張版（正確版IPA）
    print("\n【拡張版 (正確版IPA)】")
    evaluator_enhanced = EnhancedEvaluationMetrics(use_acoustic=True)
    evaluator_enhanced.print_acoustic_metrics(predictions, targets, show_confusion_analysis=True)
    
    print("\n✓ Test completed")