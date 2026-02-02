#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matrics_undefined.py

統一評価指標モジュール（互換フル版）
- train.py / main_pattern_b.py の“過去API”を落とさず動かすための互換メソッドを全搭載
- 最終PER（per_per）は「CTC collapse後・全音素・参照長基準」で固定
    PER = 100 * (Σ(S + D + I)) / (Σ len(target))
- 追加機能：
  - 子音正解率（編集距離ベース）
  - 完全一致率（全音素/子音のみ/母音のみ）
  - 位置別正解率（子音のみ）
  - 音素別 Precision/Recall/F1（簡易）
  - サンプル表示（PER定義を統一 & S/D/I表示バグ修正済み）
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np


def _safe_token(seq, idx, fallback="∅"):
    return seq[idx] if 0 <= idx < len(seq) else fallback


class UnifiedEvaluationMetrics:
    """統一評価指標クラス（互換フル版）"""

    # 子音/母音の定義（あなたの環境に合わせた最小集合）
    CONSONANTS = {'k', 'g', 's', 'z', 't', 'd', 'n', 'h', 'b', 'p', 'm', 'y', 'r', 'w', 'N'}
    VOWELS = {'a', 'i', 'u', 'e', 'o'}

    def __init__(self):
        pass

    # =========================
    # 判定
    # =========================
    @classmethod
    def is_consonant(cls, phoneme: str) -> bool:
        return phoneme in cls.CONSONANTS

    @classmethod
    def is_vowel(cls, phoneme: str) -> bool:
        return phoneme in cls.VOWELS

    # =========================
    # CTC collapse / 前処理
    # =========================
    @staticmethod
    def ctc_collapse(sequence: List[str], blank_id: int = None) -> List[str]:
        """
        CTC collapse:
        - 連続同一トークンを圧縮
        - blank_id が渡された場合のみ blank を除去（通常 None でOK）
        """
        if not sequence:
            return []
        collapsed: List[str] = []
        prev = None
        for phoneme in sequence:
            # blank 除去（使う場合のみ）
            if blank_id is not None and phoneme == blank_id:
                prev = None
                continue
            if phoneme != prev:
                collapsed.append(phoneme)
                prev = phoneme
        return collapsed

    @classmethod
    def preprocess_sequence(
        cls,
        sequence: List[str],
        apply_collapse: bool = True,
        consonants_only: bool = False,
        vowels_only: bool = False,
    ) -> List[str]:
        """
        評価前処理を統一（collapse → filter）
        - train.py の最終PER（per_per）は consonants_only=False, vowels_only=False で使う
        """
        if apply_collapse:
            sequence = cls.ctc_collapse(sequence)

        if consonants_only and vowels_only:
            raise ValueError("consonants_only と vowels_only は同時に True にできません")

        if consonants_only:
            sequence = [p for p in sequence if cls.is_consonant(p)]
        elif vowels_only:
            sequence = [p for p in sequence if cls.is_vowel(p)]

        return sequence

    # =========================
    # Levenshtein（train.py と同一実装）
    # =========================
    @staticmethod
    def _levenshtein_sdi(ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
        """Levenshtein編集距離（S, D, I）: train.py と完全一致"""
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

    # =========================================================
    # PER（最終PERの定義はここ：per_per）
    # =========================================================
    def calculate_per(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        apply_collapse: bool = True,
        mode: str = "all",  # "all" | "consonant" | "vowel"
    ) -> Dict[str, float]:
        """
        グローバルPER（参照長基準）
        - mode="all" が train.py の per_per と一致する想定
        """
        total_S = total_D = total_I = 0
        total_phonemes = 0

        consonants_only = (mode == "consonant")
        vowels_only = (mode == "vowel")

        for pred, target in zip(predictions, targets):
            pred_p = self.preprocess_sequence(
                pred,
                apply_collapse=apply_collapse,
                consonants_only=consonants_only,
                vowels_only=vowels_only,
            )
            tgt_p = self.preprocess_sequence(
                target,
                apply_collapse=apply_collapse,
                consonants_only=consonants_only,
                vowels_only=vowels_only,
            )

            S, D, I = self._levenshtein_sdi(tgt_p, pred_p)
            total_S += S
            total_D += D
            total_I += I
            total_phonemes += len(tgt_p)

        total_errors = total_S + total_D + total_I
        denom = max(total_phonemes, 1)
        per = 100.0 * total_errors / denom

        return {
            "per": per,
            "substitutions": total_S,
            "deletions": total_D,
            "insertions": total_I,
            "total_phonemes": total_phonemes,
            "per_total_errors": total_errors,
        }

    # =========================
    # 子音正解率（編集距離ベース）
    # =========================
    def calculate_consonant_accuracy(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        apply_collapse: bool = True,
    ) -> Dict[str, float]:
        """
        子音正解率（編集距離ベース）
        correct_in_seq = max(0, len(target_consonants) - (S+D+I))
        """
        correct = 0
        total = 0
        total_errors = 0
        substitutions = deletions = insertions = 0

        for pred, target in zip(predictions, targets):
            p = self.preprocess_sequence(pred, apply_collapse=apply_collapse, consonants_only=True)
            t = self.preprocess_sequence(target, apply_collapse=apply_collapse, consonants_only=True)

            if len(t) == 0:
                continue

            S, D, I = self._levenshtein_sdi(t, p)
            edit_dist = S + D + I

            substitutions += S
            deletions += D
            insertions += I

            correct += max(0, len(t) - edit_dist)
            total += len(t)
            total_errors += edit_dist

        accuracy = (correct / total * 100.0) if total > 0 else 0.0
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "errors": total_errors,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
        }

    # =========================
    # 完全一致率
    # =========================
    def calculate_exact_match(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        apply_collapse: bool = True,
        consonants_only: bool = False,
        vowels_only: bool = False,
    ) -> Dict[str, float]:
        """完全一致率（辞書返し）"""
        exact_matches = 0
        total_samples = len(predictions)

        for pred, target in zip(predictions, targets):
            p = self.preprocess_sequence(
                pred, apply_collapse=apply_collapse,
                consonants_only=consonants_only, vowels_only=vowels_only
            )
            t = self.preprocess_sequence(
                target, apply_collapse=apply_collapse,
                consonants_only=consonants_only, vowels_only=vowels_only
            )
            if p == t:
                exact_matches += 1

        rate = (exact_matches / max(total_samples, 1)) * 100.0
        return {
            "exact_match_rate": rate,
            "exact_matches": exact_matches,
            "total_samples": total_samples,
        }

    # --- 互換API（main_pattern_b.py 等が呼ぶ想定）---
    def sequence_exact_match_rate(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        apply_collapse: bool = True,
        mode: str = "all",  # "all" | "consonant" | "vowel"
    ) -> float:
        """互換：系列完全一致率 [%]"""
        consonants_only = (mode == "consonant")
        vowels_only = (mode == "vowel")
        r = self.calculate_exact_match(
            predictions, targets,
            apply_collapse=apply_collapse,
            consonants_only=consonants_only,
            vowels_only=vowels_only
        )
        return float(r["exact_match_rate"])

    def exact_match_rate(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        apply_collapse: bool = True,
        consonants_only: bool = False,
        vowels_only: bool = False,
    ) -> float:
        """互換：完全一致率 [%]（古い呼び方対策）"""
        r = self.calculate_exact_match(
            predictions, targets,
            apply_collapse=apply_collapse,
            consonants_only=consonants_only,
            vowels_only=vowels_only
        )
        return float(r["exact_match_rate"])

    # =========================
    # 位置別正解率（子音のみ）
    # =========================
    def calculate_position_accuracy(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        apply_collapse: bool = True,
    ) -> Dict[str, float]:
        """位置別正解率（子音のみ）"""
        first_correct = first_total = 0
        middle_correct = middle_total = 0
        last_correct = last_total = 0

        for pred, target in zip(predictions, targets):
            p = self.preprocess_sequence(pred, apply_collapse=apply_collapse, consonants_only=True)
            t = self.preprocess_sequence(target, apply_collapse=apply_collapse, consonants_only=True)

            if len(t) == 0:
                continue

            # first
            first_total += 1
            if len(p) > 0 and p[0] == t[0]:
                first_correct += 1

            # last
            last_total += 1
            if len(p) > 0 and p[-1] == t[-1]:
                last_correct += 1

            # middle（target側が3以上のときのみ）
            if len(t) > 2:
                for i in range(1, len(t) - 1):
                    middle_total += 1
                    if i < len(p) and p[i] == t[i]:
                        middle_correct += 1

        first_acc = 100.0 * first_correct / max(first_total, 1)
        middle_acc = 100.0 * middle_correct / max(middle_total, 1)
        last_acc = 100.0 * last_correct / max(last_total, 1)

        return {
            "first_accuracy": first_acc,
            "first_correct": first_correct,
            "first_total": first_total,
            "middle_accuracy": middle_acc,
            "middle_correct": middle_correct,
            "middle_total": middle_total,
            "last_accuracy": last_acc,
            "last_correct": last_correct,
            "last_total": last_total,
        }

    # =========================
    # 音素別 Precision/Recall/F1（簡易）
    # =========================
    def calculate_per_phoneme_metrics(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        labels: List[str],
        apply_collapse: bool = True,
        mode: str = "all",  # "all" | "consonant" | "vowel"
    ) -> Dict[str, Dict]:
        """
        音素別のPrecision/Recall/F1（簡易）
        - 各系列を前処理した後、min_len でトークンを揃えて集計（粗い）
        """
        consonants_only = (mode == "consonant")
        vowels_only = (mode == "vowel")

        all_pred, all_true = [], []
        for p, t in zip(predictions, targets):
            pp = self.preprocess_sequence(
                p, apply_collapse=apply_collapse,
                consonants_only=consonants_only, vowels_only=vowels_only
            )
            tt = self.preprocess_sequence(
                t, apply_collapse=apply_collapse,
                consonants_only=consonants_only, vowels_only=vowels_only
            )
            L = min(len(pp), len(tt))
            all_pred.extend(pp[:L])
            all_true.extend(tt[:L])

        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        support = defaultdict(int)

        for pred, true in zip(all_pred, all_true):
            support[true] += 1
            if pred == true:
                tp[true] += 1
            else:
                fp[pred] += 1
                fn[true] += 1

        results = {}
        for label in labels:
            prec = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            rec = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            results[label] = {
                "precision": prec * 100.0,
                "recall": rec * 100.0,
                "f1": f1 * 100.0,
                "support": int(support[label]),
            }

        return results

    def print_per_phoneme_metrics(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        labels: List[str],
        mode: str = "consonant",
        apply_collapse: bool = True,
    ):
        """音素別メトリクスを表形式で表示"""
        res = self.calculate_per_phoneme_metrics(
            predictions, targets, labels,
            apply_collapse=apply_collapse, mode=mode
        )
        name = "母音" if mode == "vowel" else ("子音" if mode == "consonant" else "全音素")

        print("\n" + "=" * 70)
        print(f"Per-{name} Metrics")
        print("=" * 70)
        print(f"{name:<6} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)

        sorted_labels = sorted(labels, key=lambda l: -res[l]["support"])
        for lab in sorted_labels:
            r = res[lab]
            if r["support"] > 0:
                print(f"{lab:<6} {r['precision']:>10.2f}% {r['recall']:>10.2f}% "
                      f"{r['f1']:>10.2f}% {r['support']:>10}")

        valid = [r for r in res.values() if r["support"] > 0]
        if valid:
            avg_prec = float(np.mean([r["precision"] for r in valid]))
            avg_rec = float(np.mean([r["recall"] for r in valid]))
            avg_f1 = float(np.mean([r["f1"] for r in valid]))
            print("-" * 70)
            print(f"{'Macro avg':<6} {avg_prec:>10.2f}% {avg_rec:>10.2f}% {avg_f1:>10.2f}%")
        print("=" * 70)

    # =========================
    # サンプル表示（定義統一 & バグ修正済み）
    # =========================
    def print_sample_results(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        texts: Optional[List[str]] = None,
        num_samples: int = 5,
        apply_collapse: bool = True,
        show_correct: bool = True,
        show_incorrect: bool = True,
        vowel_mode: Optional[bool] = None,  # ★互換：旧コード対応
        mode: str = "consonant",            # "all" | "consonant" | "vowel"
        per_denom: str = "ref",             # "ref" | "max"
        **kwargs,                           # ★互換：未知の旧引数も握りつぶす
    ):
        """
        サンプル結果表示（互換フル）
        - 旧API: vowel_mode=True/False を受け付ける
        - 新API: mode="vowel"/"consonant"/"all"
        - 追加で渡された未知のkwargsは無視（main側の互換のため）
        """

        # ---- 互換変換 ----
        # main_pattern_b.py が vowel_mode を渡してくる場合はこちらを優先
        if vowel_mode is not None:
            mode = "vowel" if bool(vowel_mode) else "consonant"

        print(f"\n{'='*70}")
        print(f"サンプル結果（最大{num_samples}件） mode={mode}")
        print(f"{'='*70}")

        consonants_only = (mode == "consonant")
        vowels_only = (mode == "vowel")

        correct_samples = []
        incorrect_samples = []

        for i, (pred, target) in enumerate(zip(predictions, targets)):
            p = self.preprocess_sequence(
                pred, apply_collapse=apply_collapse,
                consonants_only=consonants_only, vowels_only=vowels_only
            )
            t = self.preprocess_sequence(
                target, apply_collapse=apply_collapse,
                consonants_only=consonants_only, vowels_only=vowels_only
            )

            text = texts[i] if texts and i < len(texts) else f"Sample{i}"
            is_correct = (p == t)

            S, D, I = self._levenshtein_sdi(t, p)

            if per_denom == "max":
                denom = max(len(t), len(p), 1)
            else:
                denom = max(len(t), 1)

            per = 100.0 * (S + D + I) / denom

            sample_info = {
                "index": i,
                "text": text,
                "pred": p,
                "target": t,
                "is_correct": is_correct,
                "per": float(per),
                "S": int(S),
                "D": int(D),
                "I": int(I),
            }

            if is_correct:
                correct_samples.append(sample_info)
            else:
                incorrect_samples.append(sample_info)

        # 正解例
        if show_correct and correct_samples:
            print("\n【✓ 正解例】")
            for k, s in enumerate(correct_samples[:num_samples], 1):
                print(f"\n{k}. [{s['text']}] (Sample {s['index']}) PER={s['per']:.2f}%")
                print(f"   Target: {' '.join(s['target'])}")
                print(f"   Pred:   {' '.join(s['pred'])} ✓")

        # 不正解例（PER降順）
        if show_incorrect and incorrect_samples:
            incorrect_samples.sort(key=lambda x: -x["per"])
            print("\n【✗ 不正解例】")
            for k, s in enumerate(incorrect_samples[:num_samples], 1):
                print(f"\n{k}. [{s['text']}] (Sample {s['index']}) PER={s['per']:.2f}%")
                print(f"   Target: {' '.join(s['target'])}")
                print(f"   Pred:   {' '.join(s['pred'])} ✗")
                print(f"   Errors: S={s['S']}, D={s['D']}, I={s['I']}")

        # サマリー
        total = len(predictions)
        correct_count = len(correct_samples)
        acc = 100.0 * correct_count / max(total, 1)
        print(f"\n{'='*70}")
        print(f"サマリー: 正解 {correct_count}/{total} ({acc:.1f}%), 不正解 {total-correct_count}/{total} ({100-acc:.1f}%)")
        print(f"{'='*70}")


    # =========================
    # train.py 互換：全部入りメトリクス
    # =========================
    def calculate_all_metrics(
        self,
        predictions: List[List[str]],
        targets: List[List[str]],
        apply_collapse: bool = True,
    ) -> Dict[str, float]:
        """
        互換の keys を返す
        - train.py が参照する per_per / consonant_accuracy を必ず含む
        - 旧コード向けに exact_match_* / position_* も付ける
        """
        metrics: Dict[str, float] = {}

        # PER（最終結果は per_per）
        per_metrics = self.calculate_per(predictions, targets, apply_collapse=apply_collapse, mode="all")
        metrics.update({f"per_{k}": v for k, v in per_metrics.items()})

        # 子音正解率
        acc_metrics = self.calculate_consonant_accuracy(predictions, targets, apply_collapse=apply_collapse)
        metrics.update({f"consonant_{k}": v for k, v in acc_metrics.items()})

        # 完全一致率（全音素）
        em_all = self.calculate_exact_match(
            predictions, targets, apply_collapse=apply_collapse,
            consonants_only=False, vowels_only=False
        )
        metrics.update({f"exact_match_all_{k}": v for k, v in em_all.items()})

        # 完全一致率（子音のみ）
        em_cons = self.calculate_exact_match(
            predictions, targets, apply_collapse=apply_collapse,
            consonants_only=True, vowels_only=False
        )
        metrics.update({f"exact_match_consonant_{k}": v for k, v in em_cons.items()})

        # 旧キー互換（もし参照されてたら困るので残す）
        metrics["exact_match_exact_match_rate"] = em_all["exact_match_rate"]
        metrics["exact_match_exact_matches"] = em_all["exact_matches"]
        metrics["exact_match_total_samples"] = em_all["total_samples"]

        # 位置別正解率（子音のみ）
        pos = self.calculate_position_accuracy(predictions, targets, apply_collapse=apply_collapse)
        metrics.update({f"position_{k}": v for k, v in pos.items()})

        return metrics

    # =========================
    # 表示（任意）
    # =========================
    def print_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """評価指標を見やすく表示"""
        print(f"\n{'='*70}")
        print(f"{prefix} 統一評価指標（CTC collapse適用）" if prefix else "統一評価指標（CTC collapse適用）")
        print(f"{'='*70}")

        # PER
        if "per_per" in metrics:
            print("\n【音素誤り率 (PER)】")
            print(f"  PER: {metrics['per_per']:.2f}%")
            print(f"  置換: {metrics.get('per_substitutions', 0):,}")
            print(f"  削除: {metrics.get('per_deletions', 0):,}")
            print(f"  挿入: {metrics.get('per_insertions', 0):,}")
            print(f"  総音素数: {metrics.get('per_total_phonemes', 0):,}")
            print(f"  総誤り数: {metrics.get('per_per_total_errors', metrics.get('per_total_errors', 0)):,}")

        # 子音正解率
        if "consonant_accuracy" in metrics:
            print("\n【子音正解率（編集距離ベース）】")
            print(f"  正解率: {metrics['consonant_accuracy']:.2f}%")
            print(f"  正解: {metrics.get('consonant_correct', 0):,}/{metrics.get('consonant_total', 0):,}")
            print(f"  エラー: {metrics.get('consonant_errors', 0):,}")
            print(f"    - 置換: {metrics.get('consonant_substitutions', 0):,}")
            print(f"    - 削除: {metrics.get('consonant_deletions', 0):,}")
            print(f"    - 挿入: {metrics.get('consonant_insertions', 0):,}")

        # 完全一致率
        if "exact_match_all_exact_match_rate" in metrics:
            print("\n【完全一致率】")
            print(f"  全音素: {metrics['exact_match_all_exact_match_rate']:.2f}%")
            print(f"    一致数: {metrics.get('exact_match_all_exact_matches', 0):,}/{metrics.get('exact_match_all_total_samples', 0):,}")
        if "exact_match_consonant_exact_match_rate" in metrics:
            print(f"  子音のみ: {metrics['exact_match_consonant_exact_match_rate']:.2f}%")
            print(f"    一致数: {metrics.get('exact_match_consonant_exact_matches', 0):,}/{metrics.get('exact_match_consonant_total_samples', 0):,}")

        # 位置別
        if "position_first_accuracy" in metrics:
            print("\n【位置別正解率（子音のみ）】")
            print(f"  最初: {metrics['position_first_accuracy']:.2f}%  ({metrics.get('position_first_correct', 0):,}/{metrics.get('position_first_total', 0):,})")
            print(f"  中間: {metrics['position_middle_accuracy']:.2f}%  ({metrics.get('position_middle_correct', 0):,}/{metrics.get('position_middle_total', 0):,})")
            print(f"  最後: {metrics['position_last_accuracy']:.2f}%  ({metrics.get('position_last_correct', 0):,}/{metrics.get('position_last_total', 0):,})")

        print(f"{'='*70}")


if __name__ == "__main__":
    print("matrics_undefined.py loaded (UnifiedEvaluationMetrics FULL compatible)")
