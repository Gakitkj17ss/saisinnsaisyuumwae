#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音素類似度実験モジュール（正確版追加）
- 永田先輩の論文完全準拠版を追加
- 間違えやすい音素置き換え実験
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from collections import defaultdict
import random


# ============================================================================
# 正確版: 永田論文完全準拠実装
# ============================================================================

class AccurateIPAConsonantFeatures:
    """
    IPA子音の調音特徴ベクトル（正確版・永田論文完全準拠）
    
    対象: 基本15子音 + 撥音N = 16子音
    ベクトル構造: 24次元
        - 声帯振動: 1次元 (0/1)
        - 調音場所: 11次元 (one-hot)
        - 調音方法: 8次元 (one-hot)
        - 気流通路: 1次元 (0=開放, 1=閉鎖)
        - 口蓋帆位置: 1次元 (0=上, 1=下/鼻音)
        - 破擦化: 1次元 (0/1)
        - 硬口蓋化: 1次元 (0/1)
    
    参考: 永田智行 修士論文 (2013) 表4-2, 4-3
    """
    
    def __init__(self):
        # 調音場所インデックス
        self.PLACE_BILABIAL = 0      # 両唇音 (下唇→上唇)
        self.PLACE_LABIODENTAL = 1   # 唇歯音
        self.PLACE_DENTAL = 2        # 歯音
        self.PLACE_ALVEOLAR = 3      # 歯茎音 (舌端→上歯茎)
        self.PLACE_POSTALVEOLAR = 4  # 後部歯茎音
        self.PLACE_RETROFLEX = 5     # そり舌音
        self.PLACE_PALATAL = 6       # 硬口蓋音 (前舌→硬口蓋)
        self.PLACE_VELAR = 7         # 軟口蓋音 (後舌→軟口蓋)
        self.PLACE_UVULAR = 8        # 口蓋垂音
        self.PLACE_PHARYNGEAL = 9    # 咽頭音
        self.PLACE_GLOTTAL = 10      # 声門音 (声帯)
        
        # 調音方法インデックス
        self.MANNER_PLOSIVE = 0      # 破裂音
        self.MANNER_NASAL = 1        # 鼻音
        self.MANNER_TRILL = 2        # ふるえ音
        self.MANNER_TAP = 3          # はじき音
        self.MANNER_FRICATIVE = 4    # 摩擦音
        self.MANNER_LAT_FRIC = 5     # 側面摩擦音
        self.MANNER_APPROXIMANT = 6  # 接近音
        self.MANNER_LAT_APP = 7      # 側面接近音
        
        # 特徴ベクトル定義 (24次元)
        # [voicing(1), place(11), manner(8), airstream(1), nasal(1), affricate(1), palatalized(1)]
        self.features = self._build_feature_vectors()
        
        # 重み設定 (永田論文 4.3.2節)
        self.weight_voicing = 1.2      # 声帯振動
        self.weight_place = 1.0        # 調音場所
        self.weight_manner = 1.0       # 調音方法
        self.weight_airstream = 1.2    # 気流通路
        self.weight_nasal = 1.2        # 口蓋帆位置
        self.weight_affricate = 0.8    # 破擦化
        self.weight_palatalized = 0.8  # 硬口蓋化
        
        # 閾値 (永田論文: 0.784以下は0)
        self.threshold = 0.784
    
    def _build_feature_vectors(self) -> Dict[str, np.ndarray]:
        """特徴ベクトルを構築"""
        features = {}
        
        # k: 無声軟口蓋破裂音
        features['k'] = self._make_vector(
            voicing=0, place=self.PLACE_VELAR, manner=self.MANNER_PLOSIVE,
            airstream=1, nasal=0, affricate=0, palatalized=0
        )
        
        # s: 無声歯茎摩擦音
        features['s'] = self._make_vector(
            voicing=0, place=self.PLACE_ALVEOLAR, manner=self.MANNER_FRICATIVE,
            airstream=0, nasal=0, affricate=0, palatalized=0
        )
        
        # t: 無声歯茎破裂音
        features['t'] = self._make_vector(
            voicing=0, place=self.PLACE_ALVEOLAR, manner=self.MANNER_PLOSIVE,
            airstream=1, nasal=0, affricate=0, palatalized=0
        )
        
        # n: 有声歯茎鼻音
        features['n'] = self._make_vector(
            voicing=1, place=self.PLACE_ALVEOLAR, manner=self.MANNER_NASAL,
            airstream=1, nasal=1, affricate=0, palatalized=0
        )
        
        # h: 無声声門摩擦音
        features['h'] = self._make_vector(
            voicing=0, place=self.PLACE_GLOTTAL, manner=self.MANNER_FRICATIVE,
            airstream=0, nasal=0, affricate=0, palatalized=0
        )
        
        # m: 有声両唇鼻音
        features['m'] = self._make_vector(
            voicing=1, place=self.PLACE_BILABIAL, manner=self.MANNER_NASAL,
            airstream=1, nasal=1, affricate=0, palatalized=0
        )
        
        # y: 有声硬口蓋接近音
        features['y'] = self._make_vector(
            voicing=1, place=self.PLACE_PALATAL, manner=self.MANNER_APPROXIMANT,
            airstream=0, nasal=0, affricate=0, palatalized=0
        )
        
        # r: 有声歯茎はじき音
        features['r'] = self._make_vector(
            voicing=1, place=self.PLACE_ALVEOLAR, manner=self.MANNER_TAP,
            airstream=1, nasal=0, affricate=0, palatalized=0
        )
        
        # w: 有声両唇軟口蓋接近音
        # ※2つの調音場所を持つが、主要な場所として両唇を使用
        features['w'] = self._make_vector(
            voicing=1, place=self.PLACE_BILABIAL, manner=self.MANNER_APPROXIMANT,
            airstream=0, nasal=0, affricate=0, palatalized=0
        )
        
        # g: 有声軟口蓋破裂音
        features['g'] = self._make_vector(
            voicing=1, place=self.PLACE_VELAR, manner=self.MANNER_PLOSIVE,
            airstream=1, nasal=0, affricate=0, palatalized=0
        )
        
        # z: 有声歯茎破擦音 (ザ・ズ等)
        features['z'] = self._make_vector(
            voicing=1, place=self.PLACE_ALVEOLAR, manner=self.MANNER_PLOSIVE,
            airstream=1, nasal=0, affricate=1, palatalized=0
        )
        
        # d: 有声歯茎破裂音
        features['d'] = self._make_vector(
            voicing=1, place=self.PLACE_ALVEOLAR, manner=self.MANNER_PLOSIVE,
            airstream=1, nasal=0, affricate=0, palatalized=0
        )
        
        # b: 有声両唇破裂音
        features['b'] = self._make_vector(
            voicing=1, place=self.PLACE_BILABIAL, manner=self.MANNER_PLOSIVE,
            airstream=1, nasal=0, affricate=0, palatalized=0
        )
        
        # p: 無声両唇破裂音
        features['p'] = self._make_vector(
            voicing=0, place=self.PLACE_BILABIAL, manner=self.MANNER_PLOSIVE,
            airstream=1, nasal=0, affricate=0, palatalized=0
        )
        
        # N: 有声歯茎鼻音 (撥音「ン」)
        features['N'] = self._make_vector(
            voicing=1, place=self.PLACE_ALVEOLAR, manner=self.MANNER_NASAL,
            airstream=1, nasal=1, affricate=0, palatalized=0
        )
        
        return features
    
    def _make_vector(self, voicing: int, place: int, manner: int,
                     airstream: int, nasal: int, affricate: int, palatalized: int) -> np.ndarray:
        """特徴ベクトルを生成 (24次元)"""
        vec = np.zeros(24, dtype=float)
        
        # 声帯振動 (1次元)
        vec[0] = voicing
        
        # 調音場所 (11次元 one-hot)
        vec[1 + place] = 1.0
        
        # 調音方法 (8次元 one-hot)
        vec[12 + manner] = 1.0
        
        # その他 (4次元)
        vec[20] = airstream
        vec[21] = nasal
        vec[22] = affricate
        vec[23] = palatalized
        
        return vec
    
    def get_vector(self, phoneme: str, weighted: bool = True) -> np.ndarray:
        """音素の特徴ベクトルを取得（重み付けオプション）"""
        if phoneme not in self.features:
            return np.zeros(24)
        
        vec = self.features[phoneme].copy()
        
        if weighted:
            # 重み適用
            vec[0] *= self.weight_voicing              # 声帯振動
            vec[1:12] *= self.weight_place             # 調音場所
            vec[12:20] *= self.weight_manner           # 調音方法
            vec[20] *= self.weight_airstream           # 気流通路
            vec[21] *= self.weight_nasal               # 口蓋帆
            vec[22] *= self.weight_affricate           # 破擦化
            vec[23] *= self.weight_palatalized         # 硬口蓋化
        
        return vec
    
    def similarity(self, p1: str, p2: str, weighted: bool = True) -> float:
        """
        2つの子音間の類似度 (Cosine類似度)
        
        Args:
            p1, p2: 子音記号
            weighted: 重み付けを使用するか
        
        Returns:
            類似度 (0.0-1.0)
        """
        v1 = self.get_vector(p1, weighted=weighted)
        v2 = self.get_vector(p2, weighted=weighted)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        sim = np.dot(v1, v2) / (norm1 * norm2)
        
        # 閾値処理 (永田論文 4.3.2節)
        if sim < self.threshold:
            sim = 0.0
        
        return float(sim)
    
    def get_confusable_pairs(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        混同しやすい子音ペアを抽出
        
        Args:
            threshold: 類似度の閾値
        
        Returns:
            [(p1, p2, similarity), ...] のリスト (類似度降順)
        """
        pairs = []
        phonemes = list(self.features.keys())
        
        for i, p1 in enumerate(phonemes):
            for p2 in phonemes[i+1:]:
                sim = self.similarity(p1, p2, weighted=True)
                if sim >= threshold:
                    pairs.append((p1, p2, sim))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs
    
    def get_similarity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        全子音ペアの類似度マトリクスを生成
        
        Returns:
            (matrix, labels)
            matrix: 16x16のnumpy配列
            labels: 音素リスト
        """
        phonemes = sorted(self.features.keys())
        n = len(phonemes)
        matrix = np.zeros((n, n))
        
        for i, p1 in enumerate(phonemes):
            for j, p2 in enumerate(phonemes):
                matrix[i, j] = self.similarity(p1, p2, weighted=True)
        
        return matrix, phonemes


class AccurateIPAVowelFeatures:
    """
    IPA母音の調音特徴ベクトル（正確版）
    
    対象: 日本語5母音 (a, i, u, e, o)
    ベクトル構造: 3次元
        - 舌の前後位置: 前舌(0.0) - 中舌(0.5) - 後舌(1.0)
        - 舌の高さ: 狭い/高い(1.0) - 中(0.5) - 開/低い(0.0)
        - 円唇性: 非円唇(0.0) / 円唇(1.0)
    
    参考: 永田智行 修士論文 (2013) 図4-2
    IPA母音図:
           前舌    中舌    後舌
    狭い    i              u
    中高    e              o  
    開            a
    """
    
    def __init__(self):
        # 特徴ベクトル定義 (3次元)
        # [舌の前後位置, 舌の高さ, 円唇性]
        self.features = {
            'i': np.array([0.0, 1.0, 0.0]),  # 前舌・狭い・非円唇
            'e': np.array([0.0, 0.5, 0.0]),  # 前舌・中高・非円唇
            'a': np.array([0.5, 0.0, 0.0]),  # 中舌・開・非円唇
            'o': np.array([1.0, 0.5, 1.0]),  # 後舌・中高・円唇
            'u': np.array([1.0, 1.0, 1.0]),  # 後舌・狭い・円唇
        }
        
        # 重み設定
        # 永田論文には母音の重み記載なし
        # 提案: 舌の高さを重視（聴覚的に顕著）
        self.weight_frontness = 1.0   # 舌の前後位置
        self.weight_height = 1.2      # 舌の高さ
        self.weight_roundness = 1.0   # 円唇性
        
        # 閾値 (母音用に緩和 - 子音より類似度が低めに出るため)
        self.threshold = 0.50
    
    def get_vector(self, phoneme: str, weighted: bool = True) -> np.ndarray:
        """母音の特徴ベクトルを取得（重み付けオプション）"""
        if phoneme not in self.features:
            return np.zeros(3)
        
        vec = self.features[phoneme].copy()
        
        if weighted:
            vec[0] *= self.weight_frontness
            vec[1] *= self.weight_height
            vec[2] *= self.weight_roundness
        
        return vec
    
    def similarity(self, p1: str, p2: str, weighted: bool = True) -> float:
        """
        2つの母音間の類似度 (Cosine類似度)
        
        Args:
            p1, p2: 母音記号
            weighted: 重み付けを使用するか
        
        Returns:
            類似度 (0.0-1.0)
        """
        v1 = self.get_vector(p1, weighted=weighted)
        v2 = self.get_vector(p2, weighted=weighted)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        sim = np.dot(v1, v2) / (norm1 * norm2)
        
        # 閾値処理
        if sim < self.threshold:
            sim = 0.0
        
        return float(sim)
    
    def get_confusable_pairs(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        混同しやすい母音ペアを抽出
        
        Args:
            threshold: 類似度の閾値
        
        Returns:
            [(p1, p2, similarity), ...] のリスト (類似度降順)
        """
        pairs = []
        phonemes = list(self.features.keys())
        
        for i, p1 in enumerate(phonemes):
            for p2 in phonemes[i+1:]:
                sim = self.similarity(p1, p2, weighted=True)
                if sim >= threshold:
                    pairs.append((p1, p2, sim))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs
    
    def get_similarity_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        全母音ペアの類似度マトリクスを生成
        
        Returns:
            (matrix, labels)
            matrix: 5x5のnumpy配列
            labels: 音素リスト
        """
        phonemes = sorted(self.features.keys())
        n = len(phonemes)
        matrix = np.zeros((n, n))
        
        for i, p1 in enumerate(phonemes):
            for j, p2 in enumerate(phonemes):
                matrix[i, j] = self.similarity(p1, p2, weighted=True)
        
        return matrix, phonemes


# ============================================================================
# 簡易版: 既存実装（後方互換性のため保持）
# ============================================================================

class IPAConsonantFeatures:
    """
    IPA子音の調音特徴ベクトル (簡易版)
    ※永田論文の近似実装 - 調音場所を数値化して簡略化
    ※後方互換性のため保持
    """
    def __init__(self):
        # 各子音の特徴定義（簡易版）
        self.features = {
            # 音素: [声帯振動, 調音場所, 気流通路, 接近度, 口蓋帆, 破擦化, 硬口蓋化]
            'k': [0, 3, 1, 1, 0, 0, 0],
            's': [0, 2, 2, 2, 0, 0, 0],
            't': [0, 2, 1, 1, 0, 0, 0],
            'n': [1, 2, 1, 1, 1, 0, 0],
            'h': [0, 4, 2, 2, 0, 0, 0],
            'm': [1, 1, 1, 1, 1, 0, 0],
            'y': [1, 3, 2, 3, 0, 0, 0],
            'r': [1, 2, 1, 4, 0, 0, 0],
            'w': [1, 1, 2, 3, 0, 0, 0],
            'g': [1, 3, 1, 1, 0, 0, 0],
            'z': [1, 2, 1, 1, 0, 1, 0],
            'd': [1, 2, 1, 1, 0, 0, 0],
            'b': [1, 1, 1, 1, 0, 0, 0],
            'p': [0, 1, 1, 1, 0, 0, 0],
            'N': [1, 2, 1, 1, 1, 0, 0],
        }
        
        self.weights = np.array([1.2, 1.0, 1.2, 1.2, 1.2, 0.8, 0.8])
        
    def get_vector(self, phoneme: str) -> np.ndarray:
        if phoneme not in self.features:
            return np.zeros(7)
        return np.array(self.features[phoneme], dtype=float)
    
    def similarity(self, p1: str, p2: str, weighted: bool = True) -> float:
        v1 = self.get_vector(p1)
        v2 = self.get_vector(p2)
        
        if weighted:
            v1 = v1 * self.weights
            v2 = v2 * self.weights
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        sim = np.dot(v1, v2) / (norm1 * norm2)
        
        if sim < 0.784:
            sim = 0.0
            
        return float(sim)
    
    def get_confusable_pairs(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        pairs = []
        phonemes = list(self.features.keys())
        
        for i, p1 in enumerate(phonemes):
            for p2 in phonemes[i+1:]:
                sim = self.similarity(p1, p2, weighted=True)
                if sim >= threshold:
                    pairs.append((p1, p2, sim))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs


# ============================================================================
# 既存の実験クラス（そのまま保持）
# ============================================================================

class PhonemeSubstitutionExperiment:
    """
    音素置き換え実験
    - 間違えやすい音素を置き換えて単語復元
    """
    def __init__(self, vocab_list: List[str]):
        """
        vocab_list: カタカナ単語のリスト
        """
        self.vocab = vocab_list
        self.ipa_features = IPAConsonantFeatures()
        
        # カタカナ→音素変換用エンコーダ
        from sys import path as syspath
        syspath.insert(0, '/mnt/project')
        from phoneme_encorder import ConsonantOnlyPhonemeEncoder
        self.consonant_encoder = ConsonantOnlyPhonemeEncoder()
        
    def substitute_phoneme(self, word: str, position: int, new_phoneme: str) -> str:
        """
        単語の指定位置の音素を置き換え
        
        Args:
            word: カタカナ単語
            position: 音素位置
            new_phoneme: 新しい音素
            
        Returns:
            置き換え後の単語(カタカナ)
        """
        phonemes = self.consonant_encoder.text_to_phonemes(word)
        if position >= len(phonemes):
            return word
        
        phonemes[position] = new_phoneme
        
        # 音素→カタカナ逆変換(簡易版)
        consonant_to_kana = {
            'k': 'カ', 's': 'サ', 't': 'タ', 'n': 'ナ', 'h': 'ハ',
            'm': 'マ', 'y': 'ヤ', 'r': 'ラ', 'w': 'ワ',
            'g': 'ガ', 'z': 'ザ', 'd': 'ダ', 'b': 'バ', 'p': 'パ', 'N': 'ン'
        }
        
        reconstructed = ''.join([consonant_to_kana.get(p, 'ア') for p in phonemes])
        return reconstructed
    
    def generate_confusion_examples(self, 
                                   n_examples: int = 100,
                                   similarity_threshold: float = 0.85) -> List[Dict]:
        """
        混同しやすい音素ペアで置き換え例を生成
        
        Returns:
            List[{
                'original': 元の単語,
                'modified': 置き換え後,
                'position': 位置,
                'original_phoneme': 元の音素,
                'new_phoneme': 新しい音素,
                'similarity': 類似度,
                'is_in_vocab': 語彙に存在するか
            }]
        """
        confusable_pairs = self.ipa_features.get_confusable_pairs(similarity_threshold)
        examples = []
        
        for _ in range(n_examples):
            word = random.choice(self.vocab)
            phonemes = self.consonant_encoder.text_to_phonemes(word)
            
            if len(phonemes) == 0:
                continue
            
            pos = random.randint(0, len(phonemes) - 1)
            orig_phoneme = phonemes[pos]
            
            candidates = [(p2, sim) for p1, p2, sim in confusable_pairs if p1 == orig_phoneme]
            candidates.extend([(p1, sim) for p1, p2, sim in confusable_pairs if p2 == orig_phoneme])
            
            if not candidates:
                continue
            
            new_phoneme, sim = random.choice(candidates)
            modified = self.substitute_phoneme(word, pos, new_phoneme)
            
            examples.append({
                'original': word,
                'modified': modified,
                'position': pos,
                'original_phoneme': orig_phoneme,
                'new_phoneme': new_phoneme,
                'similarity': sim,
                'is_in_vocab': modified in self.vocab
            })
        
        return examples
    
    def evaluate_recovery_rate(self, examples: List[Dict]) -> Dict:
        """
        単語復元率を評価
        
        Returns:
            {
                'total': 総数,
                'in_vocab_count': 語彙内に存在する数,
                'recovery_rate': 復元可能率
            }
        """
        total = len(examples)
        in_vocab = sum(1 for ex in examples if ex['is_in_vocab'])
        
        return {
            'total': total,
            'in_vocab_count': in_vocab,
            'recovery_rate': in_vocab / total if total > 0 else 0.0
        }


class EnhancedPhonemeSimilarity:
    """
    拡張版音素類似度評価指標
    - 永田先輩の研究をベースに改良
    """
    def __init__(self):
        self.ipa_features = IPAConsonantFeatures()
        
        self.nasal_consonants = {'n', 'm', 'N', 'ny', 'my'}
        self.fricative_consonants = {'s', 'h', 'z', 'sy', 'hy', 'sh'}
        self.plosive_consonants = {'k', 't', 'p', 'g', 'd', 'b', 'ky', 'ty', 'py', 'gy', 'dy', 'by'}
        
    def acoustic_similarity(self, p1: str, p2: str) -> float:
        """
        音響的類似度 (永田先輩の聴覚実験結果を模擬)
        
        Returns:
            0.0-1.0の類似度
        """
        if p1 == p2:
            return 1.0
        
        ipa_sim = self.ipa_features.similarity(p1, p2, weighted=True)
        
        class_bonus = 0.0
        if (p1 in self.nasal_consonants and p2 in self.nasal_consonants):
            class_bonus = 0.1
        elif (p1 in self.fricative_consonants and p2 in self.fricative_consonants):
            class_bonus = 0.1
        elif (p1 in self.plosive_consonants and p2 in self.plosive_consonants):
            class_bonus = 0.05
        
        voicing_pairs = [
            ('k', 'g'), ('s', 'z'), ('t', 'd'), ('h', 'b'), ('p', 'b'),
            ('ky', 'gy'), ('sy', 'zy'), ('ty', 'dy'), ('py', 'by')
        ]
        voicing_bonus = 0.0
        for v1, v2 in voicing_pairs:
            if (p1 == v1 and p2 == v2) or (p1 == v2 and p2 == v1):
                voicing_bonus = -0.15
                break
        
        final_sim = ipa_sim + class_bonus + voicing_bonus
        return max(0.0, min(1.0, final_sim))
    
    def sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """
        音素列間の類似度 (DTWベース)
        
        Args:
            seq1, seq2: 音素列
            
        Returns:
            0.0-1.0の類似度
        """
        if len(seq1) == 0 or len(seq2) == 0:
            return 0.0
        
        m, n = len(seq1), len(seq2)
        dtw = np.zeros((m + 1, n + 1))
        dtw[0, :] = np.inf
        dtw[:, 0] = np.inf
        dtw[0, 0] = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 1.0 - self.acoustic_similarity(seq1[i-1], seq2[j-1])
                dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
        
        max_len = max(m, n)
        normalized_distance = dtw[m, n] / max_len
        similarity = 1.0 - normalized_distance
        
        return max(0.0, min(1.0, similarity))
    
    def analyze_confusion_matrix(self, phonemes: List[str]) -> np.ndarray:
        """
        音素間混同行列を生成
        
        Args:
            phonemes: 対象音素リスト
            
        Returns:
            混同行列 (N x N)
        """
        n = len(phonemes)
        matrix = np.zeros((n, n))
        
        for i, p1 in enumerate(phonemes):
            for j, p2 in enumerate(phonemes):
                matrix[i, j] = self.acoustic_similarity(p1, p2)
        
        return matrix


# ===== 実行例 =====
if __name__ == "__main__":
    print("="*70)
    print("音素類似度実験（正確版追加）")
    print("="*70)
    
    # 1. 正確版 - 子音類似度
    print("\n【1. 正確版 - 子音類似度 (IPA完全準拠)】")
    accurate_consonant = AccurateIPAConsonantFeatures()
    
    test_pairs = [('k', 'g'), ('s', 'z'), ('t', 'd'), ('n', 'm'), ('k', 's'), ('b', 'p')]
    print("音素ペア間類似度:")
    for p1, p2 in test_pairs:
        sim = accurate_consonant.similarity(p1, p2, weighted=True)
        print(f"  {p1} vs {p2}: {sim:.4f}")
    
    # 2. 混同しやすいペア（正確版）
    print("\n【2. 混同しやすい子音ペア (Top 10)】")
    confusable = accurate_consonant.get_confusable_pairs(threshold=0.80)
    for p1, p2, sim in confusable[:10]:
        print(f"  {p1:4s} ↔ {p2:4s}: {sim:.4f}")
    
    # 3. 類似度マトリクス
    print("\n【3. 子音類似度マトリクス (16x16)】")
    matrix, labels = accurate_consonant.get_similarity_matrix()
    print(f"  形状: {matrix.shape}")
    print(f"  音素: {labels}")
    print(f"  非ゼロ要素数: {np.count_nonzero(matrix)} / {matrix.size}")
    
    # 4. 正確版 - 母音類似度
    print("\n【4. 正確版 - 母音類似度】")
    accurate_vowel = AccurateIPAVowelFeatures()
    
    vowel_pairs = [('a', 'e'), ('i', 'u'), ('e', 'o'), ('a', 'o'), ('i', 'e')]
    print("母音ペア間類似度:")
    for p1, p2 in vowel_pairs:
        sim = accurate_vowel.similarity(p1, p2, weighted=True)
        print(f"  {p1} vs {p2}: {sim:.4f}")
    
    # 5. 母音の混同しやすいペア
    print("\n【5. 混同しやすい母音ペア】")
    vowel_confusable = accurate_vowel.get_confusable_pairs(threshold=0.80)
    if vowel_confusable:
        for p1, p2, sim in vowel_confusable:
            print(f"  {p1:4s} ↔ {p2:4s}: {sim:.4f}")
    else:
        print("  閾値0.80以上のペアなし")
    
    # 6. 母音類似度マトリクス
    print("\n【6. 母音類似度マトリクス (5x5)】")
    vowel_matrix, vowel_labels = accurate_vowel.get_similarity_matrix()
    print(f"  形状: {vowel_matrix.shape}")
    print(f"  音素: {vowel_labels}")
    print(f"  非ゼロ要素数: {np.count_nonzero(vowel_matrix)} / {vowel_matrix.size}")
    
    # 7. 簡易版との比較
    print("\n【7. 簡易版との比較】")
    simple_consonant = IPAConsonantFeatures()
    
    comparison_pairs = [('k', 'g'), ('s', 'z'), ('t', 'd')]
    print("子音類似度の比較:")
    for p1, p2 in comparison_pairs:
        sim_simple = simple_consonant.similarity(p1, p2, weighted=True)
        sim_accurate = accurate_consonant.similarity(p1, p2, weighted=True)
        diff = sim_accurate - sim_simple
        print(f"  {p1} vs {p2}: 簡易={sim_simple:.4f}, 正確={sim_accurate:.4f}, 差={diff:+.4f}")
    
    print("\n✓ テスト完了")