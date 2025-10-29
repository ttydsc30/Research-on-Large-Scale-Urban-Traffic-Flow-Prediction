# ===============================================================
# Dash Visualization for LargeST Traffic Forecasting
# Author: ChatGPT (customized for your project)
# ===============================================================
import os
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# ===============================================================
# Paths (auto detect)
# ===============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, "../../"))
PRED_DIR = os.path.join(ROOT_DIR, "predictions")
META_PATH = os.path.join(ROOT_DIR, "data/raw/metadata.csv")

print(f"[INFO] Root={ROOT_DIR}")
print(f"[INFO] Predictions={PRED_DIR}")
print(f"[INFO] Metadata={META_PATH}")

# ===============================================================
# Helper
# ===============================================================
def safe_percentile(a, q):
    try:
        return float(np.nanpercentile(a, q))
    except Exception:
        return None

# ===============================================================
# Load metadata
# ===============================================================
def load_metadata(path=META_PATH):
    if not os.path.exists(path):
        print(f"[WARN] metadata not found: {path}")
        return pd.DataFrame(), None, None, None
    df = pd.read_csv(path)
    # 自动识别列名
    id_col = next((c for c in df.columns if c.lower() in ["id", "sensor_id", "node_id"]), None)
    lat_col = next((c for c in df.columns if "lat" in c.lower() or c.lower() == "y"), None)
    lon_col = next((c for c in df.columns if "lon" in c.lower() or "lng" in c.lower() or c.lower() == "x"), None)
    if id_col is None or lat_col is None or lon_col is None:
        print("[WARN] metadata columns not recognized (id/lat/lon). Map may be empty.")
        return df, id_col, lat_col, lon_col
    df[id_col] = df[id_col].astype(str)
    print(f"[INFO] Metadata: {len(df)} sensors ({lat_col}, {lon_col})")
    return df, id_col, lat_col, lon_col

# ===============================================================
# Load predictions
# ===============================================================
def load_prediction_files(pred_dir=PRED_DIR):
    files = sorted(glob.glob(os.path.join(pred_dir, "pred_*.npz")))
    if not files:
        print(f"[WARN] no prediction files found under {pred_dir}")
    models = {}
    for f in files:
        name = os.path.basename(f).replace(".npz", "")
        try:
            d = np.load(f, allow_pickle=True)
            pred, gt = d["preds"], d["gts"]          # (M,Q,N,1)
            mean, std = d["scaler_mean"], d["scaler_std"]
            columns = [str(x) for x in d["columns"].tolist()]
            # 反标准化
            pred = pred * std + mean
            gt   = gt   * std + mean
            models[name] = {"pred": pred, "gt": gt, "columns": columns}
            print(f"[OK] Loaded {name}: {pred.shape}")
        except Exception as e:
            print(f"[WARN] Failed to load {f}: {e}")
    return models

def compute_metrics(pred, gt):
    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    mape = np.mean(np.abs((pred - gt) / (gt + 1e-5))) * 100
    return round(mae, 3), round(rmse, 3), round(mape, 2)

# ===============================================================
# Load all data once
# ===============================================================
meta, id_col, lat_col, lon_col = load_metadata()
models = load_prediction_files()
lat_map = meta.set_index(id_col)[lat_col].to_dict() if (not meta.empty and id_col and lat_col) else {}
lon_map = meta.set_index(id_col)[lon_col].to_dict() if (not meta.empty and id_col and lon_col) else {}

for name, data in models.items():
    cols = data["columns"]
    data["lat"] = np.array([lat_map.get(str(c), np.nan) for c in cols])
    data["lon"] = np.array([lon_map.get(str(c), np.nan) for c in cols])

# Optional: load holiday flags (saved by preprocessing as holiday_flags_1h.npy / 5m.npy)
HOL_FLAGS = None
hf1 = os.path.join(ROOT_DIR, "data/processed/holiday_flags_1h.npy")
hf5 = os.path.join(ROOT_DIR, "data/processed/holiday_flags_5m.npy")
for hp in [hf1, hf5]:
    if os.path.exists(hp):
        try:
            HOL_FLAGS = np.load(hp)
            print(f"[INFO] Loaded holiday flags: {hp} {HOL_FLAGS.shape}")
            break
        except Exception as e:
            print("[WARN] failed to load holiday flags:", e)

# 如果没有模型，给一个空页面提示
MODEL_KEYS = list(models.keys())
if not MODEL_KEYS:
    MODEL_KEYS = ["<no predictions found>"]
    models["<no predictions found>"] = {"pred": np.zeros((1,1,1,1)), "gt": np.zeros((1,1,1,1)), "columns": [], "lat": np.array([]), "lon": np.array([])}

# ===============================================================
# Dash App
# ===============================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "LargeST Traffic Forecast Dashboard"

# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
tabs = dbc.Tabs(
    [
        dbc.Tab(label="🗺 Traffic Heatmap", tab_id="map"),
        dbc.Tab(label="📈 Error Distribution", tab_id="error"),
        dbc.Tab(label="🕸 Network Graph", tab_id="network"),
        dbc.Tab(label="⚖ Model Comparison", tab_id="compare"),
    ],
    id="tabs",
    active_tab="map",
)

app.layout = dbc.Container(
    [
        html.H2("🚦 LargeST Multi-Model Traffic Visualization Dashboard", className="mt-3"),
        html.Hr(),
        tabs,
        html.Div(id="tab-content", className="mt-4"),
    ],
    fluid=True,
)

# ---------------------------------------------------------------
# Tab Rendering
# ---------------------------------------------------------------
@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(tab):
    if tab == "map":
        return render_map_tab()
    elif tab == "error":
        return render_error_tab()
    elif tab == "network":
        return render_network_tab()
    elif tab == "compare":
        return render_compare_tab()
    return html.Div("Select a tab above.")

# ===============================================================
# Tab 1: Dynamic Map Visualization (GT/PRED/ERROR)
# ===============================================================
def render_map_tab():
    return html.Div([
        html.Div([
            html.Label("Select Model:"),
            dcc.Dropdown(
                id="model-map",
                options=[{"label": k, "value": k} for k in models.keys()],
                value=list(models.keys())[0],
                style={"width": "40%"},
                clearable=False,
            ),
            html.Label("Display Mode:"),
            dcc.RadioItems(
                id="map-mode",
                options=[
                    {"label": "🟢 Ground Truth", "value": "gt"},
                    {"label": "🔵 Prediction", "value": "pred"},
                    {"label": "🔴 Error (|pred - gt|)", "value": "error"},
                ],
                value="pred",
                inline=True,
            ),
            html.Label("Animate by:"),
            dcc.RadioItems(
                id="animate-mode",
                options=[
                    {"label": "⏱ Time (Samples / M)", "value": "byM"},
                    {"label": "⏩ Horizon (Steps / Q)", "value": "byQ"},
                ],
                value="byM",
                inline=True,
            ),
            html.Br(),
            html.Div([
                html.Div([
                    html.Label("Sample index t"),
                    dcc.Slider(0, 1, 1, value=0, id="sample-slider", tooltip={"placement": "bottom"}),
                ], style={"flex":"1 1 420px", "minWidth":"300px"}),
                html.Div([
                    html.Label("Horizon step h"),
                    dcc.Slider(0, 1, 1, value=0, id="horizon-slider", tooltip={"placement": "bottom"}),
                ], style={"flex":"1 1 420px", "minWidth":"300px"}),
            ], style={"display":"flex","gap":"24px","flexWrap":"wrap"}),
            html.Br(),
            html.Button("▶ Play / Pause", id="play-btn", n_clicks=0),
            dcc.Interval(id="map-timer", interval=800, disabled=True),
            html.Div(id="map-stats", className="mt-2", style={"fontSize": "16px", "color": "#333"}),
        ]),
        dcc.Graph(id="map-graph", style={"height": "700px"})
    ])

# 播放/暂停
@app.callback(
    Output("map-timer", "disabled"),
    Input("play-btn", "n_clicks"),
    State("map-timer", "disabled")
)
def toggle_play(n_clicks, disabled):
    if not n_clicks:
        return True
    return not disabled

# 根据所选模型动态设置 slider 最大值和刻度
@app.callback(
    Output("sample-slider", "max"),
    Output("horizon-slider", "max"),
    Output("sample-slider", "marks"),
    Output("horizon-slider", "marks"),
    Input("model-map", "value")
)
def update_slider_max(model_name):
    data = models[model_name]
    M, Q = int(data["pred"].shape[0]), int(data["pred"].shape[1])
    m_marks = {0:"0", max(M-1,0): str(max(M-1,0))}
    h_marks = {0:"0", max(Q-1,0): str(max(Q-1,0))}
    mid_m = (M-1)//2 if M>2 else None
    mid_h = (Q-1)//2 if Q>2 else None
    if mid_m is not None: m_marks[mid_m] = str(mid_m)
    if mid_h is not None: h_marks[mid_h] = str(mid_h)
    return max(M-1,0), max(Q-1,0), m_marks, h_marks

# 自动推进帧
@app.callback(
    Output("sample-slider", "value"), Output("horizon-slider", "value"),
    Input("map-timer", "n_intervals"),
    State("animate-mode", "value"),
    State("model-map", "value"),
    State("sample-slider", "value"),
    State("horizon-slider", "value")
)
def advance_frame(n, mode, model_name, t, h):
    if n is None:
        raise Exception
    data = models[model_name]
    M, Q = data["pred"].shape[:2]
    if mode == "byM":
        t = (int(t or 0) + 1) % max(M, 1)
    else:
        h = (int(h or 0) + 1) % max(Q, 1)
    return t, h

# 地图渲染
@app.callback(
    Output("map-graph", "figure"), Output("map-stats", "children"),
    Input("model-map", "value"),
    Input("map-mode", "value"),
    Input("animate-mode", "value"),
    Input("sample-slider", "value"),
    Input("horizon-slider", "value"),
)
def update_map(model_name, mode, anim_mode, t, h):
    data = models[model_name]
    P, G = data["pred"], data["gt"]     # (M,Q,N,1)
    M, Q, N = P.shape[0], P.shape[1], P.shape[2]
    t = int(min(max(0, int(t or 0)), max(M-1,0)))
    h = int(min(max(0, int(h or 0)), max(Q-1,0)))

    if mode == "gt":
        vals = G[t, h, :, 0]
        title = f"Ground Truth — {model_name} | t={t}, h={h}"
    elif mode == "pred":
        vals = P[t, h, :, 0]
        title = f"Prediction — {model_name} | t={t}, h={h}"
    else:
        vals = np.abs(P[t, h, :, 0] - G[t, h, :, 0])
        title = f"Error (|Pred-GT|) — {model_name} | t={t}, h={h}"

    # 节假日提示（如果可用）
    if HOL_FLAGS is not None and len(HOL_FLAGS) > t:
        if int(HOL_FLAGS[t]) == 1:
            title += "  🎉 Holiday"

    # 过滤掉没有经纬度的点
    lat = data.get("lat", np.full((N,), np.nan))
    lon = data.get("lon", np.full((N,), np.nan))
    df = pd.DataFrame({
        "sensor": data["columns"] if data.get("columns") else np.arange(N).astype(str),
        "lat": lat,
        "lon": lon,
        "value": vals
    }).dropna()

    # 如果为空，返回提示图
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=title + " — (no geo points to plot)",
            height=700,
            margin=dict(l=10, r=10, t=60, b=10)
        )
        fig.add_annotation(
            text="No valid (lat, lon) for current model's nodes.\nCheck metadata mapping.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
        stats = "📊 MAE=-- | RMSE=-- | MAPE=--"
        return fig, stats

    # 颜色上限用95分位，避免极端值压缩色带
    v95 = safe_percentile(df["value"].values, 95)
    cmax = v95 if v95 and np.isfinite(v95) else None

    fig = px.scatter_mapbox(
        df,
        lat="lat",
        lon="lon",
        color="value",
        hover_data={"sensor": True, "value": True, "lat": False, "lon": False},
        color_continuous_scale="Viridis",
        zoom=8, height=700,
        mapbox_style="carto-positron",
        title=title
    )
    if cmax is not None and cmax > 0:
        fig.update_traces(marker=dict(cmin=0, cmax=cmax))
        fig.update_coloraxes(cmin=0, cmax=cmax)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))

    # 单帧统计
    mae = float(np.abs(P[t, h, :, 0] - G[t, h, :, 0]).mean())
    rmse = float(np.sqrt(((P[t, h, :, 0] - G[t, h, :, 0]) ** 2).mean()))
    denom = np.abs(G[t, h, :, 0]) + 1e-6
    mape = float((np.abs(P[t, h, :, 0] - G[t, h, :, 0]) / denom).mean() * 100)
    stats = f"📊 MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}% | M={M}, Q={Q}, N={N}"
    return fig, stats

# ===============================================================
# Tab 2: Error Distribution (with time & horizon controls)
# ===============================================================
def render_error_tab():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Select Model:"),
                dcc.Dropdown(
                    id="model-error",
                    options=[{"label": k, "value": k} for k in models.keys()],
                    value=list(models.keys())[0],
                    style={"width": "360px"},
                    clearable=False,
                ),
            ], style={"marginRight":"16px"}),

            html.Div([
                html.Label("Scope:"),
                dcc.RadioItems(
                    id="err-scope",
                    options=[
                        {"label":"Aggregate over all samples (M) and horizons (Q)", "value":"agg"},
                        {"label":"Specific time (sample t) & horizon (h)", "value":"point"},
                    ],
                    value="agg",
                    inline=False,
                ),
            ]),
        ], style={"display":"flex","gap":"24px","alignItems":"flex-start","flexWrap":"wrap"}),

        html.Div(id="err-sliders", style={"marginTop":"8px"}),

        html.Hr(),
        dcc.Graph(id="error-graph", style={"height": "560px"}),

        html.Hr(),
        html.H4("Global MAE over time (per sample)"),
        html.Div([
            html.Label("Select horizon h for the time series:"),
            dcc.Slider(0, 2, 1, value=0, id="err-ts-h", tooltip={"placement":"bottom"}),
        ], style={"margin":"6px 0 10px 0"}),

        dcc.Graph(id="error-ts", style={"height":"360px"}),
    ])

# 根据模型和 scope 构建滑块（M、Q 的范围以所选模型为准）
@app.callback(
    Output("err-sliders","children"),
    Input("model-error","value"),
    Input("err-scope","value"),
)
def build_err_sliders(model_name, scope):
    data = models[model_name]
    M, Q = data["pred"].shape[0], data["pred"].shape[1]
    if scope == "agg":
        return html.Div([
            html.Em(f"Aggregating over all samples (M={M}) and horizons (Q={Q}).", style={"opacity":0.8})
        ])
    else:
        return html.Div([
            html.Label("Sample index t (0..M-1):"),
            dcc.Slider(0, max(M-1,0), 1, value=0, id="err-t", tooltip={"placement":"bottom"}),
            html.Br(),
            html.Label("Horizon step h (0..Q-1):"),
            dcc.Slider(0, max(Q-1,0), 1, value=0, id="err-h", tooltip={"placement":"bottom"}),
        ])

# 误差直方图：聚合 / 指定时刻
@app.callback(
    Output("error-graph","figure"),
    Input("model-error","value"),
    Input("err-scope","value"),
    Input("err-t","value"),
    Input("err-h","value"),
)
def update_error_hist(model_name, scope, t, h):
    data = models[model_name]
    P, G = data["pred"], data["gt"]  # (M,Q,N,1)
    M, Q = P.shape[0], P.shape[1]
    if scope == "agg":
        mae_nodes = np.mean(np.abs(P - G), axis=(0,1,3))   # (N,)
        title = f"MAE Distribution — {model_name} (aggregated over M & Q)"
    else:
        t = int(min(max(0, t or 0), M-1))
        h = int(min(max(0, h or 0), Q-1))
        mae_nodes = np.abs(P[t, h, :, 0] - G[t, h, :, 0])  # (N,)
        title = f"MAE Distribution — {model_name} @ t={t}, h={h}"

    df = pd.DataFrame({"mae": mae_nodes})
    fig = px.histogram(df, x="mae", nbins=40, title=title)
    fig.update_traces(marker_color="#0077B6")
    fig.update_layout(margin=dict(l=10,r=10,t=48,b=10))
    return fig

# 全局 MAE 随时间（每个 sample 一个点）— 选择 horizon h
@app.callback(
    Output("error-ts","figure"),
    Input("model-error","value"),
    Input("err-ts-h","value"),
)
def update_error_timeseries(model_name, h):
    data = models[model_name]
    P, G = data["pred"], data["gt"]  # (M,Q,N,1)
    M, Q = P.shape[0], P.shape[1]
    h = int(min(max(0, h or 0), Q-1))
    mae_t = np.mean(np.abs(P[:, h, :, 0] - G[:, h, :, 0]), axis=1)  # (M,)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=mae_t, mode="lines", name=f"MAE @ h={h}"))
    fig.update_layout(
        title=f"Global MAE over time (samples) — {model_name} @ h={h}",
        xaxis_title="sample index (t)",
        yaxis_title="MAE",
        height=360,
        margin=dict(l=10,r=10,t=48,b=10),
    )
    return fig

# ===============================================================
# Tab 3: Network Graph
# ===============================================================
def render_network_tab():
    return html.Div([
        html.Label("Select Model:"),
        dcc.Dropdown(
            id="model-net",
            options=[{"label": k, "value": k} for k in models.keys()],
            value=list(models.keys())[0],
            style={"width": "60%"},
            clearable=False,
        ),
        dcc.Graph(id="net-graph", style={"height": "700px"}),
    ])

@app.callback(Output("net-graph", "figure"), Input("model-net", "value"))
def update_network(model_name):
    data = models[model_name]
    pred, gt = data["pred"], data["gt"]
    mae = np.mean(np.abs(pred - gt), axis=(0, 1, 3))
    lat = data.get("lat")
    lon = data.get("lon")
    df = pd.DataFrame({
        "lat": lat if lat is not None else np.array([]),
        "lon": lon if lon is not None else np.array([]),
        "mae": mae,
    }).dropna()
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Network Node Error Map — {model_name} (no geo points to plot)",
            height=700, margin=dict(l=10, r=10, t=60, b=10)
        )
        return fig
    v95 = safe_percentile(df["mae"].values, 95)
    cmax = v95 if v95 and np.isfinite(v95) else None
    fig = px.scatter_mapbox(
        df, lat="lat", lon="lon", color="mae",
        color_continuous_scale="Viridis", zoom=8, height=700,
        mapbox_style="carto-positron",
        title=f"Network Node Error Map — {model_name}",
    )
    if cmax is not None and cmax > 0:
        fig.update_traces(marker=dict(cmin=0, cmax=cmax))
        fig.update_coloraxes(cmin=0, cmax=cmax)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

# ===============================================================
# Tab 4: Model Comparison
# ===============================================================
def render_compare_tab():
    stats = []
    for name, d in models.items():
        mae, rmse, mape = compute_metrics(d["pred"], d["gt"])
        stats.append({"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape})
    df = pd.DataFrame(stats)
    fig = go.Figure()
    for metric in ["MAE", "RMSE", "MAPE"]:
        fig.add_trace(go.Bar(x=df["model"], y=df[metric], name=metric))
    fig.update_layout(
        barmode="group",
        title="Model Performance Comparison",
        yaxis_title="Error Value",
        xaxis_title="Model",
    )
    return html.Div([dcc.Graph(figure=fig)])

# ===============================================================
# Run
# ===============================================================
if __name__ == "__main__":
    app.run(debug=False, port=8050)
