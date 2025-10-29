# src/vis_utils.py
import os, glob, numpy as np, pandas as pd

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_pred_files(pred_dir="predictions"):
    return sorted(glob.glob(os.path.join(pred_dir, "pred_*.npz")))

def load_pred_npz(path: str):
    d=np.load(path, allow_pickle=True)
    preds=d["preds"]; gts=d["gts"]
    mean=float(d["scaler_mean"]) if "scaler_mean" in d.files else 0.0
    std=float(d["scaler_std"]) if "scaler_std" in d.files else 1.0
    cols=d["columns"].tolist()
    # 反标准化
    preds_r = preds*std + mean   # (M,Q,N,1)
    gts_r   = gts*std + mean
    return preds_r, gts_r, cols

def get_sensor_xy(metadata_csv: str, wanted_ids):
    md=pd.read_csv(metadata_csv)
    id_col=None
    for c in ["sensor_id","id","Id","ID","station_id"]:
        if c in md.columns: id_col=c; break
    if id_col is None: id_col=md.columns[0]
    lat_col = [c for c in md.columns if "lat" in c.lower()][0]
    lon_col = [c for c in md.columns if "lon" in c.lower() or "lng" in c.lower()][0]
    md[id_col]=md[id_col].astype(str)
    md=md.set_index(id_col).reindex([str(x) for x in wanted_ids])
    return md[lat_col].values, md[lon_col].values  # (N,)
