#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音韻エンコーダーモジュール（統一版）
日本語テキストを音素列に変換

【対応モード】
- vowel: 母音のみ（a, i, u, e, o）+ blank = 6クラス
- consonant: 子音のみ（k, s, t, n, h, m, y, r, w, g, z, d, b, p, N）+ blank = 16クラス
"""

from typing import List, Dict, Optional
from collections import Counter
import json
import os


class VowelOnlyPhonemeEncoder:
    """
    カタカナ列 → 母音列[a,i,u,e,o] に変換
    - 撥音「ン」、促音「ッ」、記号・空白は無視
    - 長音「ー」は直前母音に吸収（=重複として圧縮）
    - 連続同一母音は1個に圧縮（例:「オオイ」→「オイ」）
    """
    def __init__(self):
        # 出力語彙（blankはCTC用）
        self.vowels = ['a', 'i', 'u', 'e', 'o']
        self.blank_token = '<blank>'
        self.blank_id = 0
        
        self.vocab = [self.blank_token] + self.vowels
        self.phoneme_to_id = {s: i for i, s in enumerate(self.vocab)}
        self.id_to_phoneme = {i: s for s, i in self.phoneme_to_id.items()}

        # --- カタカナ→母音核 辞書（主要音節＋拗音＋外来音の代表）
        #     ※母音だけが欲しいので、各仮名を最終母音に写像
        A = 'アァカガサザタダナハバパマヤャラワヮァァ'  # 最終母音 a
        I = 'イィキギシジチヂニヒビピミリヰ'            # i
        U = 'ウゥクグスズツヅヌフブプムユュルゥゥ'        # u
        E = 'エェケゲセゼテデネヘベペメレヱ'            # e
        O = 'オォコゴソゾトドノホボポモヨョロヲ'        # o
        # 小文字拗音（ャュョ）は前子音と結合し最終母音が a/u/o になる想定だが、
        # ここでは単独出現時の保険で a/u/o に割り当て済み。

        self.kana2vowel = {ch: 'a' for ch in A}
        self.kana2vowel.update({ch: 'i' for ch in I})
        self.kana2vowel.update({ch: 'u' for ch in U})
        self.kana2vowel.update({ch: 'e' for ch in E})
        self.kana2vowel.update({ch: 'o' for ch in O})

        # 外来音や拡張（ヴ/ティ/ディ/ファ/フィ/フェ/フォ 等）
        self.digraph_map = {
            # ヴ + 母音
            'ヴァ':'a','ヴィ':'i','ヴ':'u','ヴェ':'e','ヴォ':'o',
            # 子音+小母音（CV拡張）
            'キャ':'a','キュ':'u','キョ':'o','ギャ':'a','ギュ':'u','ギョ':'o',
            'シャ':'a','シュ':'u','ショ':'o','ジャ':'a','ジュ':'u','ジョ':'o',
            'チャ':'a','チュ':'u','チョ':'o','ヂャ':'a','ヂュ':'u','ヂョ':'o',
            'ニャ':'a','ニュ':'u','ニョ':'o','ヒャ':'a','ヒュ':'u','ヒョ':'o',
            'ミャ':'a','ミュ':'u','ミョ':'o','リャ':'a','リュ':'u','リョ':'o',
            'ファ':'a','フィ':'i','フェ':'e','フォ':'o','フュ':'u',
            'ティ':'i','ディ':'i','トゥ':'u','ドゥ':'u',
            'チェ':'e','シェ':'e','ジェ':'e',
            'ツァ':'a','ツィ':'i','ツェ':'e','ツォ':'o',
            # 小母音単独（保険）
            'ァ':'a','ィ':'i','ゥ':'u','ェ':'e','ォ':'o',
        }

        # 無視する記号類
        import re
        self.ignore_pattern = re.compile(r'[^\u30A0-\u30FF]')  # カタカナ以外
        self.skip_chars = set('ンッ・、。 「」『』（）()[]{}－ー　')  # ーは特別扱い

    def num_classes(self) -> int:
        """クラス数を返す（blank含む）"""
        return len(self.vocab)

    def katakana_to_vowel_sequence(self, text: str) -> List[str]:
        """カタカナ→母音列に変換"""
        # カタカナ以外を除去
        s = self.ignore_pattern.sub('', text)

        res = []
        i = 0
        L = len(s)
        while i < L:
            # 2文字の拗音・外来音を優先
            if i+1 < L:
                pair = s[i:i+2]
                if pair in self.digraph_map:
                    v = self.digraph_map[pair]
                    res.append(v)
                    i += 2
                    continue
            ch = s[i]
            if ch in self.skip_chars:
                if ch == 'ー' and res:
                    # 長音：直前母音を繰り返し → 後段で圧縮される
                    res.append(res[-1])
                # それ以外は無視
                i += 1
                continue
            v = self.kana2vowel.get(ch, None)
            if v is not None:
                res.append(v)
            i += 1

        # 連続同一母音を圧縮（例: オオイ→オイ）
        res_comp = []
        for v in res:
            if not res_comp or res_comp[-1] != v:
                res_comp.append(v)
        return res_comp

    def text_to_phonemes(self, text: str) -> List[str]:
        """テキスト→音素列（母音）"""
        return self.katakana_to_vowel_sequence(text)

    def encode_phonemes(self, phonemes: List[str]) -> List[int]:
        """音素列→ID列"""
        return [self.phoneme_to_id.get(p, 0) for p in phonemes]

    def decode_phonemes(self, ids: List[int]) -> List[str]:
        """ID列→音素列"""
        return [self.id_to_phoneme.get(int(i), '<unk>') for i in ids if int(i) != self.blank_id]


# phoneme_encorder.py に追加

class ConsonantOnlyPhonemeEncoder:
    """基本15子音のみ (既存・互換性維持用)"""
    def __init__(self):
        self.consonants = [
            "k", "s", "t", "n", "h", "m", "y", "r", "w",
            "g", "z", "d", "b", "p", "N"
        ]
        self.blank_token = "<blank>"
        self.blank_id = 0
        
        self.phoneme_to_id = {self.blank_token: 0}
        for i, c in enumerate(self.consonants, start=1):
            self.phoneme_to_id[c] = i
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}
        
        self.katakana_to_consonant = {
            "カ":"k","キ":"k","ク":"k","ケ":"k","コ":"k",
            "サ":"s","シ":"s","ス":"s","セ":"s","ソ":"s",
            "タ":"t","チ":"t","ツ":"t","テ":"t","ト":"t",
            "ナ":"n","ニ":"n","ヌ":"n","ネ":"n","ノ":"n",
            "ハ":"h","ヒ":"h","フ":"h","ヘ":"h","ホ":"h",
            "マ":"m","ミ":"m","ム":"m","メ":"m","モ":"m",
            "ヤ":"y","ユ":"y","ヨ":"y",
            "ラ":"r","リ":"r","ル":"r","レ":"r","ロ":"r",
            "ワ":"w","ヲ":"w",
            "ガ":"g","ギ":"g","グ":"g","ゲ":"g","ゴ":"g",
            "ザ":"z","ジ":"z","ズ":"z","ゼ":"z","ゾ":"z",
            "ダ":"d","ヂ":"d","ヅ":"d","デ":"d","ド":"d",
            "バ":"b","ビ":"b","ブ":"b","ベ":"b","ボ":"b",
            "パ":"p","ピ":"p","プ":"p","ペ":"p","ポ":"p",
            "ン":"N",
            "ア":None,"イ":None,"ウ":None,"エ":None,"オ":None,"ー":None,
        }
        
        self.palatalized_map = {
            "キャ":"k","キュ":"k","キョ":"k",
            "シャ":"s","シュ":"s","ショ":"s",
            "チャ":"t","チュ":"t","チョ":"t",
            "ニャ":"n","ニュ":"n","ニョ":"n",
            "ヒャ":"h","ヒュ":"h","ヒョ":"h",
            "ミャ":"m","ミュ":"m","ミョ":"m",
            "リャ":"r","リュ":"r","リョ":"r",
            "ギャ":"g","ギュ":"g","ギョ":"g",
            "ジャ":"z","ジュ":"z","ジョ":"z",
            "ビャ":"b","ビュ":"b","ビョ":"b",
            "ピャ":"p","ピュ":"p","ピョ":"p",
        }
    
    def num_classes(self) -> int:
        return len(self.phoneme_to_id)
    
    def text_to_phonemes(self, katakana_text: str) -> List[str]:
        phonemes = []
        i = 0
        while i < len(katakana_text):
            if i < len(katakana_text) - 1:
                two = katakana_text[i : i + 2]
                if two in self.palatalized_map:
                    c = self.palatalized_map[two]
                    if c:
                        phonemes.append(c)
                    i += 2
                    continue
            ch = katakana_text[i]
            cons = self.katakana_to_consonant.get(ch, None)
            if cons:
                phonemes.append(cons)
            i += 1
        return phonemes
    
    def encode_phonemes(self, phonemes: List[str]) -> List[int]:
        if not phonemes:
            return []
        return [self.phoneme_to_id[p] for p in phonemes if p in self.phoneme_to_id]
    
    def decode_phonemes(self, ids: List[int]) -> List[str]:
        return [self.id_to_phoneme.get(int(i), "<unk>") for i in ids if int(i) != self.blank_id]


class ExtendedConsonantPhonemeEncoder:
    """拡張版: 基本15 + 拗音16 = 31子音"""
    def __init__(self):
        self.consonants = [
            # 基本15
            "k", "s", "t", "n", "h", "m", "y", "r", "w",
            "g", "z", "d", "b", "p", "N",
            # 拗音16
            "ky", "sy", "ty", "ny", "hy", "my", "ry",
            "gy", "zy", "dy", "by", "py",
            "sh", "ch", "j", "ts"
        ]
        self.blank_token = "<blank>"
        self.blank_id = 0
        
        self.phoneme_to_id = {self.blank_token: 0}
        for i, c in enumerate(self.consonants, start=1):
            self.phoneme_to_id[c] = i
        self.id_to_phoneme = {v: k for k, v in self.phoneme_to_id.items()}
        
        # 1文字マップ (基本子音用)
        self.katakana_to_consonant = {
            "カ":"k","キ":"k","ク":"k","ケ":"k","コ":"k",
            "サ":"s","シ":"s","ス":"s","セ":"s","ソ":"s",
            "タ":"t","チ":"t","ツ":"t","テ":"t","ト":"t",
            "ナ":"n","ニ":"n","ヌ":"n","ネ":"n","ノ":"n",
            "ハ":"h","ヒ":"h","フ":"h","ヘ":"h","ホ":"h",
            "マ":"m","ミ":"m","ム":"m","メ":"m","モ":"m",
            "ヤ":"y","ユ":"y","ヨ":"y",
            "ラ":"r","リ":"r","ル":"r","レ":"r","ロ":"r",
            "ワ":"w","ヲ":"w",
            "ガ":"g","ギ":"g","グ":"g","ゲ":"g","ゴ":"g",
            "ザ":"z","ジ":"z","ズ":"z","ゼ":"z","ゾ":"z",
            "ダ":"d","ヂ":"d","ヅ":"d","デ":"d","ド":"d",
            "バ":"b","ビ":"b","ブ":"b","ベ":"b","ボ":"b",
            "パ":"p","ピ":"p","プ":"p","ペ":"p","ポ":"p",
            "ン":"N",
            "ア":None,"イ":None,"ウ":None,"エ":None,"オ":None,"ー":None,
        }
        
        # 拗音マップ (拡張版)
        self.palatalized_map = {
            "キャ":"ky","キュ":"ky","キョ":"ky",
            "シャ":"sy","シュ":"sy","ショ":"sy",
            "チャ":"ty","チュ":"ty","チョ":"ty",
            "ニャ":"ny","ニュ":"ny","ニョ":"ny",
            "ヒャ":"hy","ヒュ":"hy","ヒョ":"hy",
            "ミャ":"my","ミュ":"my","ミョ":"my",
            "リャ":"ry","リュ":"ry","リョ":"ry",
            "ギャ":"gy","ギュ":"gy","ギョ":"gy",
            "ジャ":"zy","ジュ":"zy","ジョ":"zy",
            "ヂャ":"dy","ヂュ":"dy","ヂョ":"dy",
            "ビャ":"by","ビュ":"by","ビョ":"by",
            "ピャ":"py","ピュ":"py","ピョ":"py",
            # 特殊
            "シェ":"sh",
            "チェ":"ch",
            "ジェ":"j",
            "ツァ":"ts","ツィ":"ts","ツェ":"ts","ツォ":"ts",
        }
    
    def num_classes(self) -> int:
        return len(self.phoneme_to_id)
    
    def text_to_phonemes(self, katakana_text: str) -> List[str]:
        phonemes = []
        i = 0
        while i < len(katakana_text):
            if i < len(katakana_text) - 1:
                two = katakana_text[i : i + 2]
                if two in self.palatalized_map:
                    c = self.palatalized_map[two]
                    if c:
                        phonemes.append(c)
                    i += 2
                    continue
            ch = katakana_text[i]
            cons = self.katakana_to_consonant.get(ch, None)
            if cons:
                phonemes.append(cons)
            i += 1
        return phonemes
    
    def encode_phonemes(self, phonemes: List[str]) -> List[int]:
        if not phonemes:
            return []
        return [self.phoneme_to_id[p] for p in phonemes if p in self.phoneme_to_id]
    
    def decode_phonemes(self, ids: List[int]) -> List[str]:
        return [self.id_to_phoneme.get(int(i), "<unk>") for i in ids if int(i) != self.blank_id]


def build_phoneme_encoder(mode: str):
    """
    mode: 'consonant' | 'consonant_extended' | 'vowel'
    """
    mode = (mode or "consonant").lower()
    
    if mode == "consonant":
        enc = ConsonantOnlyPhonemeEncoder()
        labels = enc.consonants
    elif mode == "consonant_extended":
        enc = ExtendedConsonantPhonemeEncoder()
        labels = enc.consonants
    elif mode == "vowel":
        enc = VowelOnlyPhonemeEncoder()
        labels = enc.vowels
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return enc, labels



# ===== テスト =====
if __name__ == "__main__":
    print("="*70)
    print("音韻エンコーダーテスト")
    print("="*70)
    
    # 母音テスト
    print("\n【母音モード】")
    vowel_enc = VowelOnlyPhonemeEncoder()
    test_texts = ["シュウリ", "キョウシツ", "ガッコウ"]
    for text in test_texts:
        phonemes = vowel_enc.text_to_phonemes(text)
        ids = vowel_enc.encode_phonemes(phonemes)
        decoded = vowel_enc.decode_phonemes(ids)
        print(f"{text:12s} → {phonemes} → {ids} → {decoded}")
    print(f"クラス数: {vowel_enc.num_classes()}")
    
    # 子音テスト
    print("\n【子音モード】")
    consonant_enc = ConsonantOnlyPhonemeEncoder()
    for text in test_texts:
        phonemes = consonant_enc.text_to_phonemes(text)
        ids = consonant_enc.encode_phonemes(phonemes)
        decoded = consonant_enc.decode_phonemes(ids)
        print(f"{text:12s} → {phonemes} → {ids} → {decoded}")
    print(f"クラス数: {consonant_enc.num_classes()}")
    
    print("\n✓ テスト完了")