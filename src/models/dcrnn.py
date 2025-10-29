# src/models/dcrnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(A, add_self_loops=True, eps=1e-5):
    # 行归一化邻接 (扩散近似的一种简化)
    if add_self_loops:
        A = A + torch.eye(A.size(0), device=A.device)
    d = torch.clamp(A.sum(-1), min=eps)
    D_inv = torch.diag(1.0 / d)
    return torch.matmul(D_inv, A)

class DiffusionConv(nn.Module):
    """简化扩散卷积：X' = A_norm X W0 + A_norm^2 X W1"""
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.lin0 = nn.Linear(in_feat, out_feat, bias=False)
        self.lin1 = nn.Linear(in_feat, out_feat, bias=False)

    def forward(self, x, A_norm):
        # x: (B, N, F)
        x0 = torch.matmul(A_norm, x)           # (B, N, F)
        x1 = torch.matmul(A_norm, x0)
        return self.lin0(x0) + self.lin1(x1)

class DCRNNCell(nn.Module):
    def __init__(self, num_nodes, in_feat, hidden):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden = hidden
        self.diff = DiffusionConv(in_feat + hidden, hidden)

    def forward(self, x_t, h_prev, A_norm):
        # x_t: (B, N, F), h_prev:(B, N, H)
        z_in = torch.cat([x_t, h_prev], dim=-1)     # (B,N,F+H)
        h_t = torch.tanh(self.diff(z_in, A_norm))   # (B,N,H)
        return h_t

class DCRNN(nn.Module):
    """
    输入: (B, P, N, F) ; 输出: (B, Q, N, 1)
    """
    def __init__(self, num_nodes, in_feat, hidden=64, horizon=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden = hidden
        self.horizon = horizon
        self.cell = DCRNNCell(num_nodes, in_feat, hidden)
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x, A):
        # x:(B,P,N,F)  A:(N,N)
        B, P, N, F = x.shape
        assert N == self.num_nodes
        A_norm = normalize_adj(A)  # (N,N)
        A_norm = A_norm.unsqueeze(0).expand(B, -1, -1)  # (B,N,N)

        h = torch.zeros(B, N, self.hidden, device=x.device)
        # 编码 P 步
        for t in range(P):
            x_t = x[:, t, :, :]                     # (B,N,F)
            h = self.cell(x_t, h, A_norm)
        # 解码（自回归，使用前一步预测）
        outs = []
        y_t = self.readout(h)                       # (B,N,1) 先用 h 读出一步
        outs.append(y_t)
        for _ in range(self.horizon - 1):
            h = self.cell(y_t, h, A_norm)
            y_t = self.readout(h)
            outs.append(y_t)
        y = torch.stack(outs, dim=1)                # (B,Q,N,1)
        return y
