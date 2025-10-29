# src/visualizations.py
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from src.vis_utils import ensure_dir, load_pred_npz, list_pred_files

OUT_DIR = "viz/figures_png"

def error_heatmap(pred_path, out_png):
    preds,gts,ids = load_pred_npz(pred_path)
    err = np.abs(preds[:,0,:,0]-gts[:,0,:,0])  # (M,N)
    e = (err - err.min())/(err.max()-err.min()+1e-8)
    plt.figure(figsize=(12,6))
    plt.imshow(e.T, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Abs Error (norm)")
    plt.xlabel("Sliding Index (Test)"); plt.ylabel("Sensor Index")
    plt.title(f"Error Heatmap (1-step)\n{os.path.basename(pred_path)}")
    ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

def hour_dow_heat(pred_path, out_png):
    preds,gts,ids = load_pred_npz(pred_path)
    city = gts[:,0,:,0].mean(axis=1)  # (M,)
    # 合成时间索引（1h 粒度更合理；5m 粒度可重采样）
    freq = "1H"  # 不影响展示，只是 pivot 索引
    idx = pd.date_range("2019-01-01", periods=len(city), freq=freq)
    df = pd.DataFrame({"v":city}, index=idx)
    pt = df.pivot_table(values="v", index=df.index.dayofweek, columns=df.index.hour, aggfunc="mean")
    plt.figure(figsize=(10,4))
    plt.imshow(pt.values, aspect="auto")
    plt.yticks(range(7), ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
    plt.xticks(range(24), range(24))
    plt.colorbar(label="Avg Flow")
    plt.title(f"Hour x DayOfWeek (Test, 1-step)\n{os.path.basename(pred_path)}")
    ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

def cluster_centroids(pred_path, out_png, K=4):
    preds,gts,ids = load_pred_npz(pred_path)
    M = gts.shape[0]
    hrs = 24
    use = min(M, hrs*7)
    y = gts[:use,0,:,0]           # (use,N)
    y_hour = y[:hrs*(use//hrs),:].reshape(-1, hrs, y.shape[1]).mean(axis=0).T  # (N,24)
    km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(y_hour)
    plt.figure(figsize=(10,6))
    for k in range(K):
        centroid = y_hour[km.labels_==k].mean(axis=0)
        plt.plot(centroid, label=f"Cluster {k} (n={(km.labels_==k).sum()})")
    plt.legend(); plt.xlabel("Hour"); plt.ylabel("Flow")
    plt.title(f"Cluster Centroids (Daily Pattern)\n{os.path.basename(pred_path)}")
    ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

def mae_bar_compare(pred_paths, out_png):
    rows=[]
    for p in pred_paths:
        preds,gts,ids = load_pred_npz(p)
        mae = np.abs(preds[:,0,:,0]-gts[:,0,:,0]).mean()
        rows.append((os.path.basename(p), mae))
    rows = sorted(rows, key=lambda x:x[1])
    names = [r[0] for r in rows]; vals=[r[1] for r in rows]
    plt.figure(figsize=(10,5))
    plt.barh(names, vals)
    plt.xlabel("MAE (1-step, real unit)"); plt.title("Model/Granularity Comparison")
    ensure_dir(os.path.dirname(out_png)); plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    # 选一个（存在的）预测文件作为样例出图（你也可以手动指定）
    preds = list_pred_files()
    if not preds:
        raise SystemExit("No predictions found in predictions/*.npz")
    one = preds[0]
    error_heatmap(one, os.path.join(OUT_DIR, "Fig_error_heatmap.png"))
    hour_dow_heat(one, os.path.join(OUT_DIR, "Fig_hour_dow.png"))
    cluster_centroids(one, os.path.join(OUT_DIR, "Fig_cluster_centroids.png"))
    mae_bar_compare(preds, os.path.join(OUT_DIR, "Fig_mae_bar_compare.png"))
    print("[OK] Static figures saved to", OUT_DIR)
