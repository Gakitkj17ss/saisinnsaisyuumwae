#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep CNN Model with Residual Connections and Temporal Context
ResNet-style architecture for Japanese lip reading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """1D Conv for temporal patterns"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm1d(channels),
        )
        
    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        residual = x.transpose(1, 2)
        out = self.conv(residual)
        out = F.relu(out + residual)
        return out.transpose(1, 2)  # (B, T, C)


class MultiScaleTemporalAttention(nn.Module):
    """Multi-scale temporal attention (3/5/7 frames)"""
    def __init__(self, hidden_size):
        super().__init__()
        self.short_attn = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.mid_attn = nn.Conv1d(hidden_size, hidden_size, 5, padding=2)
        self.long_attn = nn.Conv1d(hidden_size, hidden_size, 7, padding=3)
        self.fusion = nn.Linear(hidden_size * 3, hidden_size)
        
    def forward(self, x):
        # x: (B, T, H)
        x_t = x.transpose(1, 2)  # (B, H, T)
        
        short = self.short_attn(x_t).transpose(1, 2)
        mid = self.mid_attn(x_t).transpose(1, 2)
        long = self.long_attn(x_t).transpose(1, 2)
        
        combined = torch.cat([short, mid, long], dim=-1)
        out = self.fusion(combined)
        return out

class TemporalAttention(nn.Module):
    """
    Contextual Frame-wise Attention
    - 出力は各時刻のスカラー重み (B,T,1)
    - スコア生成に時間方向Convを用いて近傍フレーム関係を明示的に考慮
    - 特徴は混ぜない（CTCの時間構造を保持）
    """
    def __init__(self, hidden_size, dropout=0.1, kernel_size=9, temperature=1.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.temperature = float(temperature)

        # (B,T,H) -> (B,H,T) で時間方向にConvしてスコア生成
        self.score_net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size // 2, kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size // 2, 1, kernel_size=1)  # (B,1,T)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, lengths=None, return_weights=False):
        """
        x: (B,T,H)
        lengths: (B,) optional
        return:
          out: (B,T,H)
          weights: (B,T) if return_weights
        """
        residual = x
        B, T, H = x.shape

        scores = self.score_net(x.transpose(1, 2)).transpose(1, 2)  # (B,T,1)
        scores = scores / self.temperature

        if lengths is not None:
            mask = torch.arange(T, device=x.device)[None, :] < lengths[:, None]  # (B,T)
            scores = scores.masked_fill(~mask[:, :, None], float("-inf"))

        weights = torch.softmax(scores, dim=1)  # (B,T,1)  (time-normalized)
        weights = self.dropout(weights)

        out = x * weights                       # frame-wise（混ぜない）
        out = self.layer_norm(out + residual)   # 残差で潰しすぎ防止

        if return_weights:
            return out, weights.squeeze(-1)     # (B,T)
        return out





class DeepCNNLipReader(nn.Module):
    """
    Deep CNN + LSTM + Temporal Context for consonant recognition
    
    Architecture:
    - 5 CNN stages with skip connections
    - Temporal Conv (local patterns)
    - Multi-scale Attention (3/5/7 frames)
    - 2-layer bidirectional LSTM
    - Frame attention
    - CTC output
    """
    
    def __init__(self, num_classes=16, dropout=0.3, lstm_layers=3, lstm_hidden=512):
        super().__init__()
        
        self.returns_log_probs = True
        
        # ===== Stage 1: 64x64 → 32x32, 1→64 =====
        self.stage1_main = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.stage1_skip = nn.Conv2d(1, 64, 1, stride=2, bias=False)
        
        # ===== Stage 2: 32x32 → 16x16, 64→128 =====
        self.stage2_main = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.stage2_skip = nn.Conv2d(64, 128, 1, stride=2, bias=False)
        
        # ===== Stage 3: 16x16 → 8x8, 128→256 =====
        self.stage3_main = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.stage3_skip = nn.Conv2d(128, 256, 1, stride=2, bias=False)
        
        # ===== Stage 4: 8x8 → 4x4, 256→384 =====
        self.stage4_main = nn.Sequential(
            nn.Conv2d(256, 384, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1, bias=False),
            nn.BatchNorm2d(384),
        )
        self.stage4_skip = nn.Conv2d(256, 384, 1, stride=2, bias=False)
        
        # ===== Stage 5: 4x4 → 2x2, 384→512 =====
        self.stage5_main = nn.Sequential(
            nn.Conv2d(384, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.stage5_skip = nn.Conv2d(384, 512, 1, stride=1, bias=False)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # ===== Temporal Processing =====
        self.temporal_norm = nn.LayerNorm(512)
        self.temporal_dropout = nn.Dropout(dropout * 0.3)
        
        # Temporal Convolution (local patterns)
        self.temporal_conv = TemporalConv(512, kernel_size=5)
        
        # Multi-scale Attention
        self.multiscale_attention = MultiScaleTemporalAttention(512)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        lstm_output_size = lstm_hidden * 2
        
        # Temporal Attention
        self.temporal_attention = TemporalAttention(
    lstm_output_size,
    dropout=dropout * 0.5,
    kernel_size=9,
    temperature=1.0
)
        
        # ===== Classification =====
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Attention weights storage
        self.attention_weights = None
        
        self._init_weights()

        with torch.no_grad():
            # 最終層の重みを小さく
            self.classifier[-1].weight.data *= 0.01
            # blankバイアスを負に、他を正に
            self.classifier[-1].bias.data.fill_(0.5)  # 非blankを少し優位に
            self.classifier[-1].bias.data[0] = -1.0   # blankを不利に
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*60}")
        print(f"DeepCNNLipReader")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - CNN stages: 5 (with skip connections)")
        print(f"  - Temporal Conv: kernel=5")
        print(f"  - Multi-scale Attention: 3/5/7 frames")
        print(f"  - LSTM layers: {lstm_layers} (bidirectional)")
        print(f"  - LSTM hidden: {lstm_hidden} (output={lstm_output_size})")
        print(f"  - Attention: Frame-wise")
        print(f"{'='*60}\n")

        # 訓練フェーズ管理
        self.training_phase = 0
        
        print(f"  - Gradual Unfreezing: enabled")
        print(f"{'='*60}\n")
    
    def freeze_all(self):
        """全パラメータを凍結"""
        for param in self.parameters():
            param.requires_grad = False
        print("🔒 All layers frozen")

    def unfreeze_classifier_and_lstm(self):
        """分類層とLSTMを解凍（初期段階用）"""
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.lstm.parameters():
            param.requires_grad = True
        print("🔓 Classifier + LSTM unfrozen")

    def unfreeze_classifier(self):
        """分類層のみ解凍"""
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("🔓 Classifier unfrozen")

    def unfreeze_lstm(self):
        """LSTM層を解凍"""
        for param in self.lstm.parameters():
            param.requires_grad = True
        print("🔓 LSTM unfrozen")

    def unfreeze_attention(self):
        """Attention層を解凍"""
        for param in self.temporal_attention.parameters():
            param.requires_grad = True
        for param in self.multiscale_attention.parameters():
            param.requires_grad = True
        print("🔓 Temporal Attention unfrozen")

    def unfreeze_stage(self, stage_num):
        """CNN特定ステージを解凍"""
        main = getattr(self, f'stage{stage_num}_main')
        skip = getattr(self, f'stage{stage_num}_skip')
        for param in main.parameters():
            param.requires_grad = True
        for param in skip.parameters():
            param.requires_grad = True
        print(f"🔓 CNN Stage{stage_num} unfrozen")

    def unfreeze_all_layers(self):
        """全層解凍"""
        for param in self.parameters():
            param.requires_grad = True
        print("🔓 All layers unfrozen")
    
    def count_trainable_params(self):
        """訓練可能パラメータ数をカウント"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def print_trainable_status(self):
        """訓練可能な層の状態を表示"""
        trainable, total = self.count_trainable_params()
        print(f"📊 Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    def _init_weights(self):
        """Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, lengths=None, return_attention=False):
        """
        Args:
            x: (B, T, 1, 64, 64)
            lengths: (B,) valid frame lengths
            return_attention: bool
        
        Returns:
            log_probs: (B, T, num_classes)
            attention_weights: (B, T) if return_attention=True
        """
        B, T, C, H, W = x.size()
        
        # ===== CNN Feature Extraction =====
        x = x.view(B*T, C, H, W)
        
        # Stage 1 with skip
        identity = self.stage1_skip(x)
        x = self.stage1_main(x)
        x = F.relu(x + identity)
        
        # Stage 2 with skip
        identity = self.stage2_skip(x)
        x = self.stage2_main(x)
        x = F.relu(x + identity)
        
        # Stage 3 with skip
        identity = self.stage3_skip(x)
        x = self.stage3_main(x)
        x = F.relu(x + identity)
        
        # Stage 4 with skip
        identity = self.stage4_skip(x)
        x = self.stage4_main(x)
        x = F.relu(x + identity)
        
        # Stage 5 with skip
        identity = self.stage5_skip(x)
        x = self.stage5_main(x)
        x = F.relu(x + identity)
        
        x = self.gap(x)  # (B*T, 512, 1, 1)
        x = x.flatten(1)  # (B*T, 512)
        x = x.view(B, T, -1)  # (B, T, 512)
        
        # ===== Temporal Processing =====
        x = self.temporal_norm(x)
        x = self.temporal_dropout(x)
        
        # Temporal Conv (local patterns)
        x = self.temporal_conv(x)
        
        # Multi-scale Attention
        x = self.multiscale_attention(x)
        
        # LSTM
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        x, _ = self.lstm(x)
        
        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        
        # Temporal Attention
        x, att_weights = self.temporal_attention(x, lengths=lengths, return_weights=True)
        self.attention_weights = att_weights.detach()
        
        # ===== Classification =====
        x = self.classifier(x)
        x = F.log_softmax(x, dim=-1)
        
        if return_attention:
            return x, att_weights
        
        return x


def create_deep_cnn_model(num_classes=16, dropout=0.3, lstm_layers=3, lstm_hidden=256):
    """Factory function"""
    return DeepCNNLipReader(
        num_classes=num_classes, 
        dropout=dropout,
        lstm_layers=lstm_layers,
        lstm_hidden=lstm_hidden
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DeepCNNLipReader Test")
    print("="*60 + "\n")
    
    model = create_deep_cnn_model(num_classes=16, dropout=0.3, lstm_layers=3, lstm_hidden=256)
    
    # Test input
    batch_size = 2
    frames = 40
    dummy_input = torch.randn(batch_size, frames, 1, 64, 64)
    lengths = torch.tensor([40, 35])
    
    # Forward pass
    print("Testing forward pass...")
    output = model(dummy_input, lengths, return_attention=False)
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Attention test
    output, attention = model(dummy_input, lengths, return_attention=True)
    print(f"✓ Attention shape: {attention.shape}")
    print(f"✓ Attention stats: mean={attention.mean():.4f}, std={attention.std():.4f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)