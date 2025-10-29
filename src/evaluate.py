# src/evaluate.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import ensure_dir, mae, rmse, mape

def plot_one_sensor(preds, gts, columns, sensor_idx=0, horizon=3, save_png=None, dpi=300):
    y_true = gts[:, 0, sensor_idx, 0]
    y_pred = preds[:, 0, sensor_idx, 0]
    plt.figure()
    plt.plot(y_true, label="True")
    plt.plot(y_pred, label="Pred")
    plt.title(f"Sensor {columns[sensor_idx]} - 1-step Forecast")
    plt.xlabel("Time (sliding index)")
    plt.ylabel("Flow")
    plt.legend()
    if save_png:
        ensure_dir(os.path.dirname(save_png))
        plt.savefig(save_png, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_npz", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="viz/figures_png")
    parser.add_argument("--sensor_idx", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.pred_npz, allow_pickle=True)
    preds = data["preds"]  # (M,Q,N,1) in z-score
    gts   = data["gts"]
    columns = data["columns"].tolist()

    # 反归一化
    mean = float(data["scaler_mean"]) if "scaler_mean" in data.files else 0.0
    std  = float(data["scaler_std"])  if "scaler_std"  in data.files else 1.0
    preds_real = preds * std + mean
    gts_real   = gts   * std + mean

    # 指标（真实单位）
    _mae  = mae(gts_real, preds_real)
    _rmse = rmse(gts_real, preds_real)
    _mape = mape(gts_real, preds_real)
    print(f"[REAL] MAE={_mae:.4f} RMSE={_rmse:.4f} MAPE={_mape:.2f}%")

    # 出图也用真实单位（更直观）
    base = os.path.splitext(os.path.basename(args.pred_npz))[0]
    png_path = os.path.join(args.out_dir, f"{base}_sensor{args.sensor_idx}.png")
    plot_one_sensor(preds_real, gts_real, columns, sensor_idx=args.sensor_idx, save_png=png_path, dpi=300)
    print(f"[OK] saved {png_path}")


if __name__ == "__main__":
    main()
