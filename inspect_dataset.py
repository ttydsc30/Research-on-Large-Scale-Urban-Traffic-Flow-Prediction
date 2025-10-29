# inspect_dataset.py
# Purpose: Inspect predictions/*.npz, data/raw/metadata.csv and data/raw/adjacency.npy,
# produce a human-readable TXT + JSON report and a sample columns→(lat,lon) mapping CSV.

import os, glob, json, math, re
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

# ---------- Heuristics for metadata columns ----------
ID_CANDIDATES  = ["sensor_id","node_id","id","station_id","Id","ID"]
LAT_CANDIDATES = ["lat","latitude","y"]
LON_CANDIDATES = ["lon","lng","longitude","x"]

def guess_col(cols, candidates, fuzzy_substr=True):
    cols_l = [c.lower() for c in cols]
    # exact first
    for cand in candidates:
        for i,cl in enumerate(cols_l):
            if cl == cand:
                return cols[i]
    if fuzzy_substr:
        for cand in candidates:
            for i,cl in enumerate(cols_l):
                if cand in cl:
                    return cols[i]
    return None

def just_digits(s):
    t = re.sub(r"\D+", "", str(s))
    return t if t else None

def load_metadata(meta_csv):
    md = pd.read_csv(meta_csv)
    id_col  = guess_col(md.columns, ID_CANDIDATES) or md.columns[0]
    lat_col = guess_col(md.columns, LAT_CANDIDATES)
    lon_col = guess_col(md.columns, LON_CANDIDATES)
    if lat_col is None or lon_col is None:
        raise ValueError(f"Cannot find lat/lon columns in {meta_csv}. Columns={list(md.columns)}")
    md[id_col] = md[id_col].astype(str)
    return md, id_col, lat_col, lon_col

def coords_match_for_columns(md, id_col, lat_col, lon_col, columns_list):
    """Return lat/lon arrays aligned to columns order; try normal and weak (digits) match."""
    ids = [str(x) for x in columns_list]
    md1 = md.set_index(id_col).reindex(ids)
    lats = md1[lat_col].to_numpy()
    lons = md1[lon_col].to_numpy()
    matched = int(np.isfinite(lats).sum())

    if matched >= 0.5 * len(ids):
        return lats.astype(float), lons.astype(float), {"mode":"exact", "matched": matched}
    # weak match by digits
    md2 = md.copy()
    md2["_id_digits"] = md2[id_col].apply(just_digits)
    idx_map = {just_digits(i): i for i in ids}
    lat_arr, lon_arr = [], []
    for i in ids:
        key = just_digits(i)
        if key and key in set(md2["_id_digits"]):
            row = md2.loc[md2["_id_digits"]==key].iloc[0]
            lat_arr.append(row[lat_col])
            lon_arr.append(row[lon_col])
        else:
            lat_arr.append(np.nan)
            lon_arr.append(np.nan)
    lats = np.array(lat_arr, dtype=float)
    lons = np.array(lon_arr, dtype=float)
    matched2 = int(np.isfinite(lats).sum())
    return lats, lons, {"mode":"weak-digits", "matched": matched2}

def summarize_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    info = {"file": npz_path, "keys": list(d.files)}
    # shapes
    def shp(k):
        return tuple(d[k].shape) if k in d.files and isinstance(d[k], np.ndarray) else None
    info["preds_shape"]   = shp("preds")
    info["gts_shape"]     = shp("gts")
    info["columns_len"]   = int(d["columns"].shape[0]) if "columns" in d.files else None
    info["columns_dtype"] = str(d["columns"].dtype) if "columns" in d.files else None
    # scaler
    info["scaler_mean"] = float(d["scaler_mean"]) if "scaler_mean" in d.files else None
    info["scaler_std"]  = float(d["scaler_std"])  if "scaler_std"  in d.files else None
    # quick health checks
    errs = []
    if info["preds_shape"] is None or info["gts_shape"] is None:
        errs.append("missing preds/gts")
    else:
        if len(info["preds_shape"]) != 4 or len(info["gts_shape"]) != 4:
            errs.append("preds/gts dim != 4 (expected M,Q,N,1)")
        else:
            if info["preds_shape"][2] != info["gts_shape"][2]:
                errs.append("preds N != gts N")
            if info["preds_shape"][3] != 1 or info["gts_shape"][3] != 1:
                errs.append("last dim != 1")
            if info["columns_len"] is not None and info["columns_len"] != info["preds_shape"][2]:
                errs.append("len(columns) != N")
    info["errors"] = errs
    # simple stats to infer scale
    try:
        preds = d["preds"]; gts = d["gts"]
        # sample a subset to avoid huge memory usage
        sl = (slice(0, min(preds.shape[0], 200)),
              slice(0, min(preds.shape[1], 3)),
              slice(0, min(preds.shape[2], 200)),
              0)
        ps = preds[sl]; gs = gts[sl]
        info["preds_mean_std"] = [float(ps.mean()), float(ps.std())]
        info["gts_mean_std"]   = [float(gs.mean()), float(gs.std())]
    except Exception as e:
        info["preds_mean_std"] = None
        info["gts_mean_std"] = None
        info["stat_error"] = str(e)
    return info

def summarize_adjacency(adj_path):
    if not os.path.exists(adj_path):
        return {"file": adj_path, "exists": False}
    A = np.load(adj_path)
    if A.ndim == 3:
        A = A[0]
    shp = tuple(A.shape)
    nnz = int(np.count_nonzero(A))
    total = A.size
    symm = bool(np.allclose(A, A.T, rtol=1e-5, atol=1e-8))
    vmin, vmax = float(A.min()), float(A.max())
    return {
        "file": adj_path, "exists": True, "shape": shp,
        "nnz": nnz, "sparsity": 1.0 - nnz/total,
        "symmetric": symm, "min": vmin, "max": vmax
    }

def main():
    ap = ArgumentParser()
    ap.add_argument("--pred_dir", default="predictions", help="folder containing *.npz")
    ap.add_argument("--meta", default="data/raw/metadata.csv")
    ap.add_argument("--adj",  default="data/raw/adjacency.npy")
    ap.add_argument("--outdir", default="viz/inspections")
    ap.add_argument("--sample_n", type=int, default=200, help="save first N columns→coords sample")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    txt_path  = os.path.join(args.outdir, "inspect_report.txt")
    json_path = os.path.join(args.outdir, "inspect_report.json")
    sample_csv= os.path.join(args.outdir, "columns_geo_sample.csv")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pred_dir": os.path.abspath(args.pred_dir),
        "meta": os.path.abspath(args.meta),
        "adj": os.path.abspath(args.adj),
        "npz_files": []
    }

    # 1) list & summarize npz
    npz_files = sorted(glob.glob(os.path.join(args.pred_dir, "*.npz")))
    if not npz_files:
        print(f"[WARN] No *.npz found in {args.pred_dir}")
    for f in npz_files:
        info = summarize_npz(f)
        report["npz_files"].append(info)

    # 2) adjacency
    report["adjacency"] = summarize_adjacency(args.adj)

    # 3) metadata + coordinate matching (using first valid npz)
    meta_summary = {}
    if os.path.exists(args.meta) and report["npz_files"]:
        md, id_col, lat_col, lon_col = load_metadata(args.meta)
        meta_summary["id_col"] = id_col
        meta_summary["lat_col"] = lat_col
        meta_summary["lon_col"] = lon_col

        # pick first npz that has columns
        use = None
        for item in report["npz_files"]:
            if item.get("columns_len"):
                use = item
                break
        if use is not None:
            d = np.load(use["file"], allow_pickle=True)
            cols = [str(x) for x in d["columns"].tolist()]
            lats, lons, mode = coords_match_for_columns(md, id_col, lat_col, lon_col, cols)
            meta_summary["match_mode"] = mode["mode"]
            meta_summary["matched"]    = mode["matched"]
            meta_summary["total"]      = len(cols)
            meta_summary["match_ratio"]= round(mode["matched"]/max(1,len(cols)), 4)

            # save sample mapping for first N
            n = min(len(cols), args.sample_n)
            sm = pd.DataFrame({
                "sensor_id": cols[:n],
                "lat": lats[:n],
                "lon": lons[:n]
            })
            sm.to_csv(sample_csv, index=False)
            meta_summary["sample_csv"] = os.path.abspath(sample_csv)
        else:
            meta_summary["note"] = "no npz with columns found"
    else:
        meta_summary["note"] = "metadata or npz missing"
    report["metadata"] = meta_summary

    # 4) write JSON + TXT
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = []
    P = lines.append
    P("==== Traffic Forecast Dataset Inspection ====")
    P(f"time       : {report['timestamp']}")
    P(f"pred_dir   : {report['pred_dir']}")
    P(f"metadata   : {report['meta']}")
    P(f"adjacency  : {report['adj']}")
    P("")
    P("---- NPZ files ----")
    if not report["npz_files"]:
        P("(none)")
    for it in report["npz_files"]:
        P(f"* {os.path.basename(it['file'])}")
        P(f"  keys        : {it['keys']}")
        P(f"  preds_shape : {it['preds_shape']}   gts_shape: {it['gts_shape']}")
        P(f"  columns_len : {it['columns_len']}   dtype: {it['columns_dtype']}")
        P(f"  scaler_mean : {it['scaler_mean']}   scaler_std: {it['scaler_std']}")
        P(f"  preds μ/σ   : {it['preds_mean_std']}  gts μ/σ: {it['gts_mean_std']}")
        if it['errors']:
            P(f"  [ERR] {it['errors']}")
        P("")
    P("---- Adjacency ----")
    adj = report["adjacency"]
    if not adj.get("exists", False):
        P(f"[WARN] {adj['file']} not found")
    else:
        P(f"file   : {adj['file']}")
        P(f"shape  : {adj['shape']}  symmetric: {adj['symmetric']}")
        P(f"sparse : {adj['sparsity']:.4f}  nnz: {adj['nnz']}  min/max: {adj['min']}/{adj['max']}")
    P("")
    P("---- Metadata Match ----")
    ms = report["metadata"]
    for k in ["id_col","lat_col","lon_col","match_mode","matched","total","match_ratio","sample_csv","note"]:
        if k in ms:
            P(f"{k:11s}: {ms[k]}")
    P("")
    P("Saved:")
    P(f"- {json_path}")
    P(f"- {txt_path}")
    if "sample_csv" in ms:
        P(f"- {ms['sample_csv']}")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))

if __name__ == "__main__":
    main()
