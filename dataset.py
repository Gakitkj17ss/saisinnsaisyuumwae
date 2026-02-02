#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセットとDataLoader（統一版）
- phoneme_encoder.py の統一エンコーダを使用
- 20fps × 2秒 = 40フレーム想定（max_lengthは引数で変更可）
"""
import os
import random
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ★ 統一されたエンコーダをインポート
from phoneme_encorder import build_phoneme_encoder


# =========================
#  Data Augmentation
# =========================
class VideoAugmentation:
    """動画データ拡張（必要な時だけ有効化）"""

    def __init__(self, config=None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))

    def __call__(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_tensor: (T, C, H, W)
        Returns:
            (T, C, H, W)
        """
        if not self.enabled:
            return video_tensor

        if self.config.get("random_crop", False):
            video_tensor = self._random_spatial_crop(video_tensor)

        if self.config.get("random_brightness", False):
            video_tensor = self._random_brightness(video_tensor)

        if self.config.get("random_noise", False):
            video_tensor = self._random_noise(video_tensor)

        return video_tensor

    def _random_spatial_crop(self, video: torch.Tensor) -> torch.Tensor:
        """ランダムクロップ（全フレーム同じ位置）"""
        T, C, H, W = video.shape
        s0, s1 = self.config.get("crop_scale", [0.85, 1.0])
        scale = random.uniform(s0, s1)

        new_h, new_w = int(H * scale), int(W * scale)
        top = random.randint(0, H - new_h) if new_h < H else 0
        left = random.randint(0, W - new_w) if new_w < W else 0

        video = video[:, :, top : top + new_h, left : left + new_w]
        # 元サイズに戻す
        video = (
            F.interpolate(
                video.view(T * C, 1, new_h, new_w),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            .view(T, C, H, W)
            .contiguous()
        )
        return video

    def _random_brightness(self, video: torch.Tensor) -> torch.Tensor:
        """輝度変更"""
        factor = float(self.config.get("brightness_factor", 0.2))
        brightness = 1.0 + random.uniform(-factor, factor)
        return (video * brightness).clamp(0, 1)

    def _random_noise(self, video: torch.Tensor) -> torch.Tensor:
        """ノイズ追加"""
        std = float(self.config.get("noise_std", 0.03))
        noise = torch.randn_like(video) * std
        return (video + noise).clamp(0, 1)


# =========================
#  Dataset
# =========================
class CachedAttentionCTCDataset:
    """LRUキャッシュ付きデータセット（ptファイルから[T,1,64,64]を得る）"""

    def __init__(
        self,
        csv_path: str,
        phoneme_encoder,  # ★ 統一エンコーダを受け取る
        max_length: int = 40,
        cache_size: int = 100,
        augmentation_config=None,
        is_training: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.phoneme_encoder = phoneme_encoder
        self.max_length = int(max_length)
        self.cache_size = int(cache_size)
        self.is_training = bool(is_training)
        self.augmentation = VideoAugmentation(augmentation_config) if augmentation_config else None

        # LRU cache
        self._cache = OrderedDict()
        self._hits = 0
        self._miss = 0

        self._filter_existing_files()
        print(f"✓ Dataset loaded: {len(self.df)} samples (cache={cache_size}, max_len={max_length})")

    def _filter_existing_files(self):
        valid_idx = []
        for i in range(len(self.df)):
            try:
                vp = self.df.iloc[i]["video_path"]
                if isinstance(vp, str) and os.path.exists(vp):
                    valid_idx.append(i)
            except Exception:
                pass
        if len(valid_idx) < len(self.df):
            orig_len = len(self.df)
            self.df = self.df.iloc[valid_idx].reset_index(drop=True)
            print(f"⚠ 無効ファイルを除外: {orig_len - len(valid_idx)}件")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = row["video_path"]

        # cache
        if self.cache_size > 0 and video_path in self._cache:
            self._hits += 1
            self._cache.move_to_end(video_path)
            video = self._cache[video_path].clone()
        else:
            self._miss += 1
            video = self._load_video(video_path)
            if self.cache_size > 0:
                self._cache[video_path] = video
                if len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)

        # augmentation
        if self.is_training and self.augmentation and self.augmentation.enabled:
            video = self.augmentation(video)

        # labels
        text = str(row.get("text", ""))
        
        # ★ 統一されたエンコーダを使用
        phonemes = self.phoneme_encoder.text_to_phonemes(text)  # List[str]
        ids = self.phoneme_encoder.encode_phonemes(phonemes)    # List[int]

        return {
            "video": video,                           # (T,1,64,64)
            "target": ids,                            # List[int]
            "input_length": video.size(0),
            "target_length": len(ids),
            "text": text,
            "phonemes": phonemes,
        }

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        ptファイルから動画テンソルを読み込み、(T,1,64,64)へ整形
        許容入力: dict or Tensor
          - dictの場合: 'video'/'data'/'frames' いずれか
          - 形状: (T,1,64,64) / (1,T,1,64,64) / (T,64,64) / (T,H,W,C)
        """
        try:
            data = torch.load(video_path, map_location="cpu")

            # dict -> pick a likely key
            if isinstance(data, dict):
                for k in ("video", "data", "frames"):
                    if k in data:
                        data = data[k]
                        break
                else:
                    # 最初の値を取る
                    data = next(iter(data.values()))

            # (1,T,C,H,W) -> (T,C,H,W)
            if data.ndim == 5:
                if data.size(0) == 1:
                    data = data.squeeze(0)
                else:
                    raise ValueError(f"Unexpected 5D batch size: {tuple(data.shape)}")

            # (T,H,W) -> (T,1,H,W)
            if data.ndim == 3:
                data = data.unsqueeze(1)

            # (T,H,W,C) -> (T,C,H,W)
            if data.ndim == 4 and data.size(1) > 10:
                data = data.permute(0, 3, 1, 2)

            if data.ndim != 4:
                raise ValueError(f"Expect 4D tensor, got {tuple(data.shape)}")

            # トリム
            if data.size(0) > self.max_length:
                data = data[: self.max_length]

            # リサイズ to 64x64
            if data.size(-1) != 64 or data.size(-2) != 64:
                T, C, H, W = data.shape
                data = F.interpolate(
                    data.view(T * C, 1, H, W),
                    size=(64, 64),
                    mode="bilinear",
                    align_corners=False,
                ).view(T, C, 64, 64)

            # 最終形状確認
            assert data.shape[1:] == (1, 64, 64), f"Invalid shape: {tuple(data.shape)}"
            return data.float()

        except Exception as e:
            print(f"✖ 動画読み込みエラー: {video_path} -> {e}")
            return torch.zeros(self.max_length, 1, 64, 64, dtype=torch.float32)

    def get_cache_stats(self):
        total = self._hits + self._miss
        hit_rate = (self._hits / total * 100.0) if total > 0 else 0.0
        return {"cache": len(self._cache), "hits": self._hits, "miss": self._miss, "hit_rate": hit_rate}


# =========================
#  Collate
# =========================
def attention_ctc_collate_fn(batch):
    """可変長シーケンスをまとめてバッチ化（CTC用にtargetは連結）"""
    videos, targets, input_lengths, target_lengths, texts, phonemes = [], [], [], [], [], []

    # 1) 収集（targetsは連結）
    for item in batch:
        v = item["video"]
        videos.append(v)
        targets.extend(item["target"])
        input_lengths.append(item["input_length"])
        target_lengths.append(item["target_length"])
        texts.append(item["text"])
        phonemes.append(item["phonemes"])

    # 2) 動画を時間次元でパディング（右側ゼロ埋め）
    max_len = max(v.size(0) for v in videos)
    padded = []
    for v in videos:
        if v.size(0) < max_len:
            pad = torch.zeros(max_len - v.size(0), *v.shape[1:], dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        padded.append(v)

    return {
        "video": torch.stack(padded, dim=0),                 # (B, T, 1, 64, 64)
        "target": torch.tensor(targets, dtype=torch.long),   # (sum_L,)
        "input_length": torch.tensor(input_lengths, dtype=torch.long),   # (B,)
        "target_length": torch.tensor(target_lengths, dtype=torch.long), # (B,)
        "text": texts,
        "phonemes": phonemes,
    }


# =========================
#  DataLoader factory
# =========================
def create_dataloaders(
    train_csv_path: str,
    valid_csv_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
    cache_size: int = 100,
    augmentation_config=None,
    max_length: int = 40,
    mode: str = "consonant",
):
    """
    DataLoader を作成（modeで子音/母音を切替）
    Returns:
        train_loader, valid_loader, phoneme_encoder, labels
    """
    # ★ build_phoneme_encoderでmode判定
    from phoneme_encorder import build_phoneme_encoder
    encoder, labels = build_phoneme_encoder(mode)
    
    print(f"[DataLoader] mode={mode}, num_classes={encoder.num_classes}")

    train_dataset = CachedAttentionCTCDataset(
        csv_path=train_csv_path,
        phoneme_encoder=encoder,
        max_length=max_length,
        cache_size=cache_size,
        augmentation_config=augmentation_config,
        is_training=True,
    )

    valid_dataset = CachedAttentionCTCDataset(
        csv_path=valid_csv_path,
        phoneme_encoder=encoder,
        max_length=max_length,
        cache_size=cache_size,
        augmentation_config=None,
        is_training=False,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=attention_ctc_collate_fn,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=attention_ctc_collate_fn,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, valid_loader, encoder, labels

def get_dataset_statistics(train_loader, valid_loader, test_loader, encoder, mode='consonant'):
    """データセット統計を収集（母音/子音対応）"""
    stats = {
        'train_samples': len(train_loader.dataset),
        'valid_samples': len(valid_loader.dataset),
        'test_samples': len(test_loader.dataset) if test_loader else 0,
        'phoneme_counts': {},  # ← consonant_counts から変更
        'sequence_lengths': [],
        'mode': mode
    }
    
    # 音素カウントと系列長収集
    for loader in [train_loader, valid_loader]:
        for batch in loader:
            targets = batch['target']
            target_lengths = batch['target_length']
            
            offset = 0
            for tlen in target_lengths:
                tlen = int(tlen.item())
                ids = targets[offset:offset + tlen].cpu().tolist()
                phonemes = encoder.decode_phonemes(ids)
                
                # 音素カウント
                for p in phonemes:
                    stats['phoneme_counts'][p] = stats['phoneme_counts'].get(p, 0) + 1
                
                # 系列長
                stats['sequence_lengths'].append(len(phonemes))
                
                offset += tlen
    
    return stats


# ===== テスト =====
if __name__ == "__main__":
    print("="*70)
    print("データセット統合テスト")
    print("="*70)
    
    # エンコーダテスト
    print("\n【エンコーダテスト】")
    for mode in ['vowel', 'consonant']:
        enc, labels = build_phoneme_encoder(mode)
        print(f"\nMode: {mode}")
        print(f"  Classes: {enc.num_classes}")
        print(f"  Labels: {labels}")
        
        # サンプルテキスト
        test_text = "シュウリ"
        phonemes = enc.text_to_phonemes(test_text)
        ids = enc.encode_phonemes(phonemes)
        decoded = enc.decode_phonemes(ids)
        print(f"  {test_text} → {phonemes} → {ids} → {decoded}")
    
    print("\n✓ テスト完了")