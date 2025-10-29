# src/models/astgcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """
    对称归一化 Ā = D^{-1/2} (A + I) D^{-1/2}
    A: (N, N) on device
    """
    N = A.size(0)
    A_hat = A + torch.eye(N, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(dim=1)  # (N,)
    deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt  # (N,N)

def pick_heads(hidden: int) -> int:
    """
    选一个能整除 hidden 的 num_heads（优先 8,4,2,1）
    """
    for h in (8, 4, 2, 1):
        if hidden % h == 0:
            return h
    return 1

class TemporalSelfAttention(nn.Module):
    """
    对时间维做自注意力：输入 (B*N, P, D), 输出同形状
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        # x: (B*N, P, D)
        out, _ = self.attn(x, x, x, need_weights=False)
        return out  # (B*N, P, D)

class SpatialGCN(nn.Module):
    """
    最简单的一阶图卷积：X_t' = Ā X_t W
    输入 (B, N, D) → 输出 (B, N, D_out)
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_norm):
        # x: (B, N, D)
        xw = self.lin(x)                # (B, N, D_out)
        Ax = torch.matmul(A_norm, xw)   # (N,N) @ (B,N,D_out) -> (B,N,D_out)  （广播在第0维）
        return Ax

class ASTGCN(nn.Module):
    """
    简化版 ASTGCN：时间注意力（按节点独立） + 空间GCN（按每个时间步）
    输入 X: (B, P, N, F)   输出: (B, Q, N, 1)
    """
    def __init__(self, num_nodes: int, in_feat: int, hidden: int, horizon: int):
        super().__init__()
        self.N = num_nodes
        self.P = None
        self.horizon = horizon

        # 先把通道 F 投影到 hidden 维，然后用 hidden 做注意力的 embed 维度
        self.in_proj = nn.Linear(in_feat, hidden)

        # 自动选择可整除 hidden 的 head 数
        heads = pick_heads(hidden)
        self.temporal = TemporalSelfAttention(dim=hidden, num_heads=heads)

        # 空间图卷积（按时间步逐帧做）
        self.spatial = SpatialGCN(in_dim=hidden, out_dim=hidden)

        # 解码到每个节点的 Q 步单通道
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, horizon)  # 输出每个时间步的 Q
        )

    def forward(self, X, A):
        """
        X: (B, P, N, F)
        A: (N, N)
        return: (B, Q, N, 1)
        """
        B, P, N, F = X.shape
        assert N == self.N, f"N mismatch: {N} vs {self.N}"

        A_norm = normalize_adj(A)  # (N,N)

        # 1) 通道投影到 hidden
        X = self.in_proj(X)        # (B, P, N, hidden)

        # 2) 时间注意力（对每个节点独立）：(B*N, P, hidden)
        X_t = X.permute(0, 2, 1, 3).contiguous().view(B * N, P, -1)
        X_t = self.temporal(X_t)   # (B*N, P, hidden)
        X = X_t.view(B, N, P, -1).permute(0, 2, 1, 3).contiguous()  # (B,P,N,hidden)

        # 3) 空间图卷积：对每个时间步 t
        outs = []
        for t in range(P):
            Xt = X[:, t, :, :]                     # (B,N,hidden)
            Xt = self.spatial(Xt, A_norm)          # (B,N,hidden)
            outs.append(Xt.unsqueeze(1))           # (B,1,N,hidden)
        X = torch.cat(outs, dim=1)                 # (B,P,N,hidden)

        # 4) 简单聚合时间维（平均池化或最后一步特征）
        X_agg = X.mean(dim=1)                      # (B,N,hidden)

        # 5) 解码到 Q 步
        Y = self.out(X_agg)                        # (B,N,Q)
        Y = Y.permute(0, 2, 1).unsqueeze(-1)       # (B,Q,N,1)
        return Y
