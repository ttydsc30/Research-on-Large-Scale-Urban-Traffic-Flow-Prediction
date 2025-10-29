# src/data_preprocessing.py
import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import ensure_dir, StandardScaler
from features import (
    build_time_features_compact,
    build_static_node_features_compact,
    # 新增：节假日两个工具函数（你已在 features.py 中实现）
    build_holiday_flags_from_calendar,
    build_holiday_flags_from_csv,
)

def read_h5_year(h5_path: str, key: str = "t") -> pd.DataFrame:
    """
    读取年度 HDF5；你的原始文件中 key 为 '/t'，这里传 't' 即可。
    返回 DatetimeIndex × N 的 DataFrame。
    """
    with pd.HDFStore(h5_path, mode='r') as store:
        # HDF 内部 key 一般为 '/t'，pd.read_hdf 传 't' 即可
        df = pd.read_hdf(h5_path, key=key)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = df.columns.astype(str)
    return df

def subset_by_district(values: pd.DataFrame, metadata: pd.DataFrame, district_codes=None) -> pd.DataFrame:
    if district_codes is None:
        return values
    md = metadata.copy()
    id_col, dist_col = None, None
    for c in ["sensor_id", "id", "Id", "ID", "station_id"]:
        if c in md.columns:
            id_col = c; break
    for c in ["district", "District", "DISTRICT", "zone"]:
        if c in md.columns:
            dist_col = c; break
    if id_col is None: id_col = md.columns[0]
    if dist_col is None: raise ValueError("metadata 中未找到 District 列，请检查列名。")
    sub_ids = md.loc[md[dist_col].isin(district_codes), id_col].astype(str).tolist()
    keep = [c for c in values.columns if c in set(sub_ids)]
    return values[keep]

def resample_to_hourly(df_5min: pd.DataFrame, how="mean") -> pd.DataFrame:
    return df_5min.resample("1H").mean() if how == "mean" else df_5min.resample("1H").sum()

def make_sliding_windows(values_2d: np.ndarray, P: int, Q: int, stride: int = 1):
    """
    values_2d: (T, N)
    返回:
      X: (M, P, N, 1)
      Y: (M, Q, N, 1)
    """
    T, N = values_2d.shape
    idx_list = list(range(0, T - P - Q + 1, stride))
    M = len(idx_list)
    X = np.zeros((M, P, N, 1), dtype=np.float32)
    Y = np.zeros((M, Q, N, 1), dtype=np.float32)
    for k, i in enumerate(tqdm(idx_list, desc="Building sliding windows", unit="win")):
        X[k, :, :, 0] = values_2d[i:i+P, :]
        Y[k, :, :, 0] = values_2d[i+P:i+P+Q, :]
    return X, Y

def split_by_ratio(T, ratios=(0.7, 0.1, 0.2)):
    a, b, c = ratios
    t1 = int(T * a); t2 = int(T * (a + b))
    return slice(0, t1), slice(t1, t2), slice(t2, T)

def pipeline(
    raw_dir: str,
    metadata_csv: str,
    save_dir: str,
    years=(2019,),
    district=None,
    district_map=None,
    P=12, Q=3,
    to_hourly=False,
    hourly_how="mean",
    add_static=False,
    time_feat_mode="basic",
    max_nodes=300,
    stride=12,
    holiday_csv=None,
    holiday_country=None,
):
    ensure_dir(save_dir)

    # 1) 读多年的 5min 并拼
    all_years = []
    for y in tqdm(years, desc="Reading HDF5 by year", unit="year"):
        h5_path = os.path.join(raw_dir, f"{y}.h5")
        df = read_h5_year(h5_path, key="t")  # 明确 '/t'
        all_years.append(df)
    values_5m = pd.concat(all_years, axis=0).sort_index()  # (T, N_all)

    # 2) metadata
    md = pd.read_csv(metadata_csv)

    # 3) 区域筛选
    if isinstance(district, str) and district_map and district in district_map:
        values_5m = subset_by_district(values_5m, md, district_map[district])

    # 4) 限制节点数（若需全量，传 --max_nodes 0 或删除此参数）
    if max_nodes is not None and max_nodes > 0:
        keep_cols = values_5m.columns[:max_nodes]
        values_5m = values_5m.loc[:, keep_cols]

    # 5) 缺失处理
    values_5m = values_5m.ffill().bfill().fillna(0.0)

    # 6) 重采样
    values = resample_to_hourly(values_5m, how=hourly_how) if to_hourly else values_5m
    gran = "1h" if to_hourly else "5m"

    # 7) 紧凑时间特征 (T, F_time)
    #    你的 build_time_features_compact 接收 DatetimeIndex；这里直接传 values.index
    time_feats = build_time_features_compact(values.index, mode=time_feat_mode).astype(np.float32)  # (T, F_time)

    # 7.1) 节假日 0/1（按需）——与时间特征在时间维度上拼接 → (T, F_time+1)
    hol_flags = None
    if holiday_csv:
        hol_flags = build_holiday_flags_from_csv(values.index, holiday_csv).astype(np.float32)  # (T,)
    elif holiday_country:
        hol_flags = build_holiday_flags_from_calendar(values.index, country=holiday_country).astype(np.float32)  # (T,)

    if hol_flags is not None:
        hol_col = hol_flags[:, None]  # (T,1)
        time_feats = np.concatenate([time_feats, hol_col], axis=1).astype(np.float32)

    # 8) 紧凑静态特征 (N, F_static)
    node_ids = values.columns.astype(str).tolist()
    static_feats = build_static_node_features_compact(md, node_ids) if add_static else np.zeros((len(node_ids), 0), np.float32)

    # 9) 目标值标准化
    scaler = StandardScaler()
    y_arr = values.values.astype(np.float32)  # (T, N)
    scaler.fit(y_arr)
    y_z = scaler.transform(y_arr)            # (T, N)

    # 10) 切分
    tr_slice, va_slice, te_slice = split_by_ratio(values.shape[0], (0.7, 0.1, 0.2))
    parts = {
        "train": (y_z[tr_slice], time_feats[tr_slice]),
        "val":   (y_z[va_slice], time_feats[va_slice]),
        "test":  (y_z[te_slice], time_feats[te_slice]),
    }

    out = {}
    for name, (y_part, tfeat_part) in parts.items():
        print(f"[{name}] y_part={y_part.shape}, tfeat_part={tfeat_part.shape}")
        # 构造目标滑窗（带 stride）
        X_seq, Y_seq = make_sliding_windows(y_part, P, Q, stride=stride)    # X:(M,P,N,1)
        M, _, N, _ = X_seq.shape

        # 时间特征滑窗 (M,P,1,Ft) → (M,P,N,Ft)
        Ft = tfeat_part.shape[1]
        if Ft > 0:
            t_win = np.zeros((M, P, 1, Ft), dtype=np.float32)
            for i in tqdm(range(M), desc=f"[{name}] Time feature windows", unit="win"):
                t_win[i, :, 0, :] = tfeat_part[i:i+P, :]
            t_win = np.repeat(t_win, repeats=N, axis=2)
        else:
            t_win = np.zeros((M, P, N, 0), dtype=np.float32)

        # 静态特征 (1,1,N,Fs) → (M,P,N,Fs)
        Fs = static_feats.shape[1]
        if Fs > 0:
            s_tile = np.broadcast_to(static_feats[None, None, :, :], (M, P, N, Fs)).astype(np.float32)
            X = np.concatenate([X_seq, t_win, s_tile], axis=3)  # (M,P,N, 1+Ft+Fs)
        else:
            X = np.concatenate([X_seq, t_win], axis=3)          # (M,P,N, 1+Ft)

        out[name] = (X, Y_seq)
        # 释放阶段性大对象（帮助内存）
        del X_seq, Y_seq, t_win

    # 11) 保存
    out_path = os.path.join(save_dir, f"cache_{gran}.npz")
    print(f"[Saving] {out_path}")
    np.savez_compressed(
        out_path,
        X_train=out["train"][0], Y_train=out["train"][1],
        X_val=out["val"][0],     Y_val=out["val"][1],
        X_test=out["test"][0],   Y_test=out["test"][1],
        scaler_mean=scaler.mean, scaler_std=scaler.std,
        columns=np.array(node_ids)
    )
    # 同步保存节假日标记（全时轴，便于 Dash 使用；若未启用则不写）
    if hol_flags is not None:
        np.save(os.path.join(save_dir, f"holiday_flags_{gran}.npy"), hol_flags.astype(np.int8))

    # DEBUG
    print(f"[OK] Saved {gran} cache to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--years", type=str, default="2019")
    parser.add_argument("--district", type=str, default=None, help="CA/GLA/GBA/SD 或 None")
    parser.add_argument("--P", type=int, default=12)
    parser.add_argument("--Q", type=int, default=3)
    parser.add_argument("--to_hourly", action="store_true")
    parser.add_argument("--hourly_how", type=str, default="mean", choices=["mean","sum"])
    parser.add_argument("--time_feat_mode", type=str, default="basic", choices=["none","basic","full"])
    parser.add_argument("--add_static", dest="add_static", action="store_true")
    parser.add_argument("--no_add_static", dest="add_static", action="store_false")
    parser.set_defaults(add_static=False)
    parser.add_argument("--max_nodes", type=int, default=300)
    parser.add_argument("--stride", type=int, default=12, help="滑窗步长，默认12（等于每60分钟取一个样本）")
    # 新增：节假日来源
    parser.add_argument("--holiday_country", type=str, default=None, help="使用 holidays 库自动生成(例如 CN/US)，留空则不启用")
    parser.add_argument("--holiday_csv", type=str, default=None, help="自备节假日CSV路径（与holiday_country二选一）")
    args = parser.parse_args()

    district_map = {"GLA": [7, 8, 12], "GBA": [4], "SD": [11]}
    years = tuple(int(x) for x in args.years.split(","))

    pipeline(
        raw_dir=args.raw_dir,
        metadata_csv=args.metadata,
        save_dir=args.save_dir,
        years=years,
        district=args.district,
        district_map=district_map,
        P=args.P, Q=args.Q,
        to_hourly=args.to_hourly,
        hourly_how=args.hourly_how,
        add_static=args.add_static,
        time_feat_mode=args.time_feat_mode,
        max_nodes=args.max_nodes,
        stride=args.stride,
        holiday_csv=args.holiday_csv,
        holiday_country=args.holiday_country,
    )
