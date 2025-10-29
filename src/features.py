# src/features.py
import numpy as np
import pandas as pd
from datetime import date
try:
    import holidays as holidaylib
except Exception:
    holidaylib = None

def build_time_features_compact(index: pd.DatetimeIndex, mode: str = "basic") -> np.ndarray:
    """
    返回紧凑时间特征矩阵 (T, F_time), float32
    mode:
      - "none": 返回形状 (T,0)
      - "basic": hour_sin, hour_cos, dow_sin, dow_cos (4维) —— 推荐先用
      - "full": hour, dow, month, is_weekend, + 上面4个周期特征 (共8维)
    """
    assert isinstance(index, pd.DatetimeIndex)
    T = len(index)
    if mode == "none":
        return np.zeros((T, 0), dtype=np.float32)

    hour = index.hour.values
    dow = index.dayofweek.values

    sin_hour = np.sin(2 * np.pi * hour / 24.0)
    cos_hour = np.cos(2 * np.pi * hour / 24.0)
    sin_dow  = np.sin(2 * np.pi * dow  / 7.0 )
    cos_dow  = np.cos(2 * np.pi * dow  / 7.0 )

    feats = [sin_hour, cos_hour, sin_dow, cos_dow]
    if mode == "full":
        month = index.month.values
        is_weekend = (dow >= 5).astype(int)
        feats = feats + [hour, dow, month, is_weekend]

    F = np.stack(feats, axis=1).astype(np.float32)  # (T, F_time)
    return F

def build_static_node_features_compact(metadata: pd.DataFrame, node_ids, use_cols=None) -> np.ndarray:
    """
    返回紧凑静态特征矩阵 (N, F_static), float32，不展开到时间维。
    node_ids: 与 values.columns 对齐的传感器 ID 列表（字符串）
    use_cols: 选择使用的静态列名；默认自动挑选数值列
    """
    md = metadata.copy()
    id_col = None
    for c in ["sensor_id", "id", "Id", "ID", "station_id"]:
        if c in md.columns:
            id_col = c
            break
    if id_col is None:
        id_col = md.columns[0]

    md = md.set_index(id_col)
    md = md.reindex(node_ids)

    if use_cols is None:
        use_cols = [c for c in md.columns if np.issubdtype(md[c].dtype, np.number)]
    if not use_cols:
        return np.zeros((len(node_ids), 0), dtype=np.float32)

    md = md[use_cols].fillna(md[use_cols].median())
    return md.values.astype(np.float32)  # (N, F_static)

def build_holiday_flags_from_calendar(index, country="CN"):
    """
    index: pandas.DatetimeIndex（比如你刚看到的 /t 的索引）
    country: 'CN'（中国）、'US'（美国）等
    return: np.ndarray shape (T,) 值域 {0,1}
    """
    import numpy as np, pandas as pd
    if holidaylib is None:
        raise RuntimeError("holidays 库未安装。请先: pip install holidays")

    years = sorted(set(index.year.tolist()))
    if country.upper() in ["CN", "CHINA"]:
        cal = holidaylib.China(years=years)
    elif country.upper() == "US":
        cal = holidaylib.US(years=years)
    else:
        try:
            cal = holidaylib.country_holidays(country=country, years=years)
        except Exception:
            cal = holidaylib.US(years=years)  # 回退
    flags = index.normalize().map(lambda d: 1 if date(d.year, d.month, d.day) in cal else 0)
    return flags.to_numpy(dtype="int8")

def build_holiday_flags_from_csv(index, csv_path, date_col=None):
    import numpy as np, pandas as pd
    df = pd.read_csv(csv_path)
    if date_col is None:
        cand = [c for c in df.columns if 'date' in c.lower() or 'day' in c.lower()]
        if not cand:
            raise ValueError("holiday CSV 未找到日期列，请通过 date_col 指定")
        date_col = cand[0]
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    holiset = set(df[date_col].dt.date.tolist())
    flags = index.normalize().map(lambda d: 1 if d.date() in holiset else 0)
    return flags.to_numpy(dtype="int8")