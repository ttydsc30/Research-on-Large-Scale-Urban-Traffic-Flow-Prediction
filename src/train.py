# src/train.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils import set_seed, ensure_dir, EarlyStopping, mae, rmse, mape
from models.lstm import LSTMForecast
from models.dcrnn import DCRNN
from models.astgcn import ASTGCN

def load_cache(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "X_train": data["X_train"],
        "Y_train": data["Y_train"],
        "X_val":   data["X_val"],
        "Y_val":   data["Y_val"],
        "X_test":  data["X_test"],
        "Y_test":  data["Y_test"],
        "columns": data["columns"].tolist(),
        "mean": float(data["scaler_mean"]),
        "std":  float(data["scaler_std"]),
    }


def select_model(name, num_nodes, in_feat, hidden, horizon):
    name = name.lower()
    if name == "lstm":
        return LSTMForecast(num_nodes=num_nodes, in_feat=in_feat, hidden=hidden, horizon=horizon)
    elif name == "dcrnn":
        return DCRNN(num_nodes=num_nodes, in_feat=in_feat, hidden=hidden, horizon=horizon)
    elif name == "astgcn":
        return ASTGCN(num_nodes=num_nodes, in_feat=in_feat, hidden=hidden, horizon=horizon)
    else:
        raise ValueError(f"Unknown model: {name}")

def train_one_epoch(model, loader, criterion, optimizer, device, A=None):
    model.train()
    total = 0.0
    for X, Y in tqdm(loader, desc="Train", leave=False):
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        if A is not None:
            pred = model(X, A)
        else:
            pred = model(X)
        loss = criterion(pred, Y)
        loss.backward()
        optimizer.step()
        total += loss.item() * X.size(0)
        # 可选：逐步释放（对显存紧张有帮助）
        del X, Y, pred, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, A=None):
    model.eval()
    preds, gts = [], []
    for X, Y in tqdm(loader, desc="Eval", leave=False):
        X = X.to(device)
        Y = Y.to(device)
        if A is not None:
            pred = model(X, A)
        else:
            pred = model(X)
        preds.append(pred.cpu().numpy())
        gts.append(Y.cpu().numpy())
        del X, Y, pred
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return preds, gts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    parser.add_argument("--adjacency", type=str, required=True)
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm","dcrnn","astgcn"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--pred_out", type=str, default="predictions")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = load_cache(args.cache)
    X_train, Y_train = cache["X_train"], cache["Y_train"]
    X_val, Y_val     = cache["X_val"],   cache["Y_val"]
    X_test, Y_test   = cache["X_test"],  cache["Y_test"]

    Bp, P, N, F = X_train.shape
    Q = Y_train.shape[1]
    num_nodes = N
    in_feat = F
   
    # >>> 新增：图模型仅使用目标通道（第 0 个），丢弃时间/静态特征通道
    if args.model in ["dcrnn", "astgcn"] and F > 1:
        print(f"[INFO] {args.model} expects single target channel. Slicing X[..., :1] from F={F}.")
        X_train = X_train[..., :1]
        X_val   = X_val[..., :1]
        X_test  = X_test[..., :1]
        F = 1
        in_feat = 1

    # 邻接矩阵（按缓存节点数对齐）
    A = np.load(args.adjacency)
    if A.ndim == 3:  # 有的版本是 (1,N,N)
        A = A[0]

    # 让 A 的节点数与缓存保持一致
    if A.shape[0] != num_nodes:
        if A.shape[0] > num_nodes:
            # 若缓存是按“原顺序取前 N 列”，直接截取前 N×N 就能对齐
            print(f"[WARN] adjacency ({A.shape}) -> truncate to ({num_nodes},{num_nodes}) to match cache.")
            A = A[:num_nodes, :num_nodes]
        else:
            # 邻接比缓存还小：退化成单位阵，至少能跑通
            print(f"[WARN] adjacency smaller than cache nodes. Using identity matrix ({num_nodes}x{num_nodes}).")
            A = np.eye(num_nodes, dtype=np.float32)

    A = torch.tensor(A, dtype=torch.float32, device=device)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(Y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, drop_last=False)

    model = select_model(args.model, num_nodes, in_feat, args.hidden, Q).to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ensure_dir(args.save_dir)
    ensure_dir(args.pred_out)
    ckpt_path = os.path.join(args.save_dir, f"{os.path.basename(args.cache)}_{args.model}.pt")
    es = EarlyStopping(patience=args.patience, save_path=ckpt_path)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                                  A=A if args.model in ["dcrnn","astgcn"] else None)
        preds_val, gts_val = evaluate(model, val_loader, device,
                                      A=A if args.model in ["dcrnn","astgcn"] else None)
        val_mae = mae(gts_val, preds_val)
        improved = es.step(val_mae, model_state=model.state_dict())
        print(f"train_loss={tr_loss:.4f} | val_MAE={val_mae:.4f} | {'improved ✓' if improved else 'no improve'}")
        if es.should_stop():
            print("Early stopped.")
            break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    preds_test, gts_test = evaluate(model, test_loader, device,
                                    A=A if args.model in ["dcrnn","astgcn"] else None)
    test_mae = mae(gts_test, preds_test)
    test_rmse = rmse(gts_test, preds_test)
    test_mape = mape(gts_test, preds_test)
    print(f"[TEST] MAE={test_mae:.4f} RMSE={test_rmse:.4f} MAPE={test_mape:.2f}%")

    out_path = os.path.join(args.pred_out, f"pred_{os.path.basename(args.cache)}_{args.model}.npz")
    np.savez_compressed(
        out_path,
        preds=preds_test, gts=gts_test, columns=np.array(cache["columns"]),
        scaler_mean=np.array(cache["mean"]), scaler_std=np.array(cache["std"])
    )
    print(f"[OK] Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
