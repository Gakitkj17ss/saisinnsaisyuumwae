
# loss_soft_per.py
import torch
import torch.nn as nn

class SoftPERLoss(nn.Module):
    def __init__(
        self,
        blank_id: int,
        tau: float = 0.2,
        ins_cost: float = 1.0,
        del_cost: float = 1.0,
        eps: float = 1e-8,
        big: float = 1e4
    ):
        super().__init__()
        self.blank_id = int(blank_id)
        self.tau = float(tau)
        self.ins_cost = float(ins_cost)
        self.del_cost = float(del_cost)
        self.eps = float(eps)
        self.big = float(big)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: (T,B,C)
        targets: (sum(target_lengths),)
        input_lengths: (B,)
        target_lengths: (B,)
        """

        # --- SoftDP は AMP に弱いので fp32 固定（論文の定式化とは無関係の安全策） ---
        log_probs = log_probs.float()
        device = log_probs.device

        # --- device 統一（cuda/cpu混在を確実に潰す） ---
        targets = targets.to(device)
        input_lengths = input_lengths.to(device)
        target_lengths = target_lengths.to(device)

        T, B, C = log_probs.shape

        # --- batch内可変長：targets を (B,Umax) に展開 ---
        Umax = int(target_lengths.max().item())
        tgt2d = torch.full((B, Umax), self.blank_id, device=device, dtype=torch.long)
        off = 0
        for b in range(B):
            L = int(target_lengths[b].item())
            if L > 0:
                tgt2d[b, :L] = targets[off:off + L]
            off += L

        input_lengths = input_lengths.clamp(min=1, max=T)
        target_lengths = target_lengths.clamp(min=1, max=Umax)

        probs = log_probs.exp()  # (T,B,C)

        BIG = self.big

        # DP: prev[i] = D(i, j-1) を保持（jはフレーム側）
        prev = torch.full((Umax + 1, B), BIG, device=device, dtype=torch.float32)
        prev[0] = 0.0

        # 初期条件：D(i,0)= i * c_ins（本文に合わせて ins_cost を使う）
        for i in range(1, Umax + 1):
            prev[i] = prev[i - 1] + self.ins_cost

        # i が有効範囲か（i<=U）
        i_range = torch.arange(Umax, device=device).unsqueeze(1)   # (Umax,1)
        valid_tgt = i_range < target_lengths.unsqueeze(0)          # (Umax,B)

        def softmin3(a, b, c):
            xs = torch.stack([a, b, c], dim=0)
            return -self.tau * torch.logsumexp(-xs / self.tau, dim=0)

        for j in range(1, T+1):
            active = (j <= input_lengths).to(device)  # (B,)

            cur = torch.full((Umax+1, B), BIG, device=device)

            # --- 追加：blankコスト（モデル依存）---
            p_blank = probs[j-1, :, self.blank_id].clamp(min=self.eps)  # (B,)
            blank_cost = -torch.log(p_blank)  # (B,)  ★おすすめ

            # Delete cost (i=0) ← del_cost から blank_cost に変更
            cur0 = prev[0] + blank_cost
            cur[0] = torch.where(active, cur0, prev[0])

            p_corr_all = probs[j-1].gather(1, tgt2d)  # (B, Umax)

            for i in range(1, Umax+1):
                p_corr = p_corr_all[:, i-1]
                sub_cost = 1.0 - p_corr

                x_ins = cur[i-1] + self.ins_cost
                x_del = prev[i] + blank_cost      # ★ここも変更
                x_sub = prev[i-1] + sub_cost

                val = softmin3(x_ins, x_del, x_sub)

                valid_update = valid_tgt[i-1] & active
                cur[i] = torch.where(valid_update, val, prev[i])

            prev = cur

        final = prev.gather(0, target_lengths.unsqueeze(0)).squeeze(0)  # (B,)
        loss = final / (target_lengths.float() + self.eps)
        return loss.mean()

