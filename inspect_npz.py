import numpy as np, glob, os

fpath = sorted(glob.glob("predictions/pred_*.npz"))[0]
d = np.load(fpath, allow_pickle=True)
P, G = d["preds"], d["gts"]
M, Q, N, C = P.shape
print("file:", os.path.basename(fpath))
print("shape:", P.shape, "(M,Q,N,1) = (样本,预测步,节点,通道)")
print("M=", M, "Q=", Q, "N=", N, "C=", C)

var_M = float(P.var(axis=(0,2,3)).mean())
var_Q = float(P.var(axis=(0,1,3)).mean())
print("variation across M (samples):", round(var_M, 4))
print("variation across Q (horizon):", round(var_Q, 4))

mean, std = float(d["scaler_mean"]), float(d["scaler_std"])
print("scaler_mean/std:", mean, std)
print("preds μ/σ (raw):", round(float(P.mean()),2), round(float(P.std()),2))
print("gts   μ/σ (raw):", round(float(G.mean()),2), round(float(G.std()),2))
