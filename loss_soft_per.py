# loss_soft_per.py
import torch
import torch.nn as nn

class SoftPERLoss(nn.Module):
    def __init__(self, blank_id: int, tau: float = 0.2, ins_cost: float = 1.0, 
                 del_cost: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.blank_id = int(blank_id)
        self.tau = float(tau)
        self.ins_cost = float(ins_cost)
        self.del_cost = float(del_cost)
        self.eps = float(eps)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: (T,B,C)
        targets: (sum(target_lengths),)
        input_lengths: (B,)
        target_lengths: (B,)
        """
        T, B, C = log_probs.shape
        device = log_probs.device
        
        Umax = int(target_lengths.max().item())
        tgt2d = torch.full((B, Umax), self.blank_id, device=device, dtype=torch.long)
        off = 0
        for b in range(B):
            L = int(target_lengths[b])
            if L > 0:
                tgt2d[b, :L] = targets[off:off+L]
            off += L
        
        input_lengths = input_lengths.clamp(min=1, max=T)
        target_lengths = target_lengths.clamp(min=1, max=Umax)
        
        probs = log_probs.exp()
        BIG = 1e9
        prev = torch.full((Umax+1, B), BIG, device=device)
        prev[0] = 0.0
        
        # 修正: 初期化は削除コスト（参照i個、予測0個）
        for i in range(1, Umax+1):
            prev[i] = i * self.del_cost
        
        i_range = torch.arange(Umax, device=device).unsqueeze(1)
        valid_tgt = i_range < target_lengths.unsqueeze(0)
        
        def softmin3(a, b, c):
            xs = torch.stack([a, b, c], dim=0)
            return -self.tau * torch.logsumexp(-xs / self.tau, dim=0)
        
        for j in range(1, T+1):
            active = (j <= input_lengths).to(device)
            
            cur = torch.full((Umax+1, B), BIG, device=device)
            
            # 修正: cur[0]は挿入コスト（参照0個、予測j個）
            cur[0] = torch.where(active, torch.full((B,), j * self.ins_cost, device=device), prev[0])
            
            p_corr_all = probs[j-1].gather(1, tgt2d)
            
            for i in range(1, Umax+1):
                p_corr = p_corr_all[:, i-1]
                sub_cost = 1.0 - p_corr
                
                x_ins = cur[i-1] + self.del_cost  # 参照を1個追加=削除
                x_del = prev[i] + self.ins_cost    # 予測を1個追加=挿入
                x_sub = prev[i-1] + sub_cost       # 置換/一致
                
                val = softmin3(x_ins, x_del, x_sub)
                
                valid_update = valid_tgt[i-1] & active
                cur[i] = torch.where(valid_update, val, prev[i])
            
            prev = cur
        
        final = prev.gather(0, target_lengths.unsqueeze(0)).squeeze(0)
        loss = final / (target_lengths.float() + self.eps)
        
        return loss.mean()
