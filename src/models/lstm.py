# src/models/lstm.py
import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    """
    输入: (B, P, N, F)  -> 先把 N 个节点当作 batch 内样本维度：B*N
    输出: (B, Q, N, 1)
    """
    def __init__(self, num_nodes: int, in_feat: int, hidden: int = 64, num_layers: int = 1, horizon: int = 3):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=in_feat, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, horizon)

    def forward(self, x):
        # x: (B, P, N, F)
        B, P, N, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, P, F)
        x = x.view(B * N, P, F)                 # (B*N, P, F)
        out, _ = self.lstm(x)                   # (B*N, P, H)
        last = out[:, -1, :]                    # (B*N, H)
        y = self.fc(last)                       # (B*N, Q)
        y = y.view(B, N, self.horizon)          # (B, N, Q)
        y = y.permute(0, 2, 1).contiguous().unsqueeze(-1)  # (B, Q, N, 1)
        return y
