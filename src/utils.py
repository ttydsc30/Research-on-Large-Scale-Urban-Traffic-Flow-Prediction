# src/utils.py
import os
import random
import numpy as np
import torch

class StandardScaler:
    """Z-score 标准化，保存均值方差用于反归一化"""
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean()
        self.std = x.std() + 1e-8

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1e-5):
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


class EarlyStopping:
    """简单早停：监控验证集指标（越小越好），patience 次不提升则停止"""
    def __init__(self, patience=10, delta=1e-5, save_path=None):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best = None
        self.count = 0

    def step(self, metric_value, model_state=None):
        if self.best is None or metric_value < self.best - self.delta:
            self.best = metric_value
            self.count = 0
            if self.save_path and model_state is not None:
                torch.save(model_state, self.save_path)
            return True  # improved
        else:
            self.count += 1
            return False  # not improved

    def should_stop(self):
        return self.count >= self.patience
