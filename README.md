# TrafficForecast-LargeST 项目说明文档

本项目以 LargeST（CityMind Lab）多城市交通流量数据集为基础，构建多模型交通流量预测系统，并通过 Dash 仪表盘进行可视化展示。

## 一、项目概览

  
数据源：LargeST 数据集（包含路网结构、气象、节假日信息等）  
目标任务：基于历史流量预测未来流量（5min / 1h 粒度）  
核心模型：LSTM、DCRNN、ASTGCN  
主要输出：预测结果 .npz 文件与交互式可视化 Dash 仪表盘  

## 二、项目目录结构

  
TrafficForecast-LargeST/  
├─ data/  
│ ├─ raw/ # 原始数据（HDF5、metadata、adjacency）  
│ ├─ interim/ # 预处理中间结果（可选）  
│ └─ processed/ # 已处理数据缓存（.npz）  
│  
├─ src/ # 核心源码目录  
│ ├─ data_preprocessing.py # 数据预处理  
│ ├─ features.py # 特征提取  
│ ├─ train.py # 模型训练  
│ ├─ evaluate.py # 模型评估  
│ ├─ visualizations.py # 静态可视化  
│ ├─ utils.py # 工具函数  
│ └─ models/ # 模型定义  
│  
├─ viz/  
│ ├─ figures_png/ # 静态图表  
│ └─ dash_app/ # Dash 动态仪表盘  
│ └─ app.py  
│  
├─ predictions/ # 模型预测结果  
├─ report/ # 报告与附录  
├─ requirements.txt  
├─ run.sh  
└─ README.md  

## 三、环境配置

  
推荐使用 Anaconda：  
conda create -n largest python=3.10  
conda activate largest  
pip install -r requirements.txt  

主要依赖包括 numpy, pandas, torch, dash, plotly, h5py, tqdm 等。

## 四、项目运行流程

### Step 1：数据预处理

脚本：src/data_preprocessing.py

功能：读取 h5 文件、填充缺失值、滑窗切分，输出 cache_5m.npz

### Step 2：模型训练

脚本：src/train.py

功能：加载缓存训练 LSTM / DCRNN / ASTGCN 模型，输出 predictions/\*.npz

### Step 3：模型评估

脚本：src/evaluate.py

功能：计算 MAE、RMSE、MAPE 指标并绘图

### Step 4：静态可视化

脚本：src/visualizations.py

功能：生成热力图、误差图、模型对比图并保存至 viz/figures_png

### Step 5：Dash 动态可视化

脚本：viz/dash_app/app.py

功能：交互式仪表盘，可视化多模型动态结果

## 五、Dash 仪表盘模块说明

  
🗺 Traffic Heatmap：动态地图展示流量（支持 GT / Pred / Error 切换、时间与预测步播放）  
📈 Error Distribution：支持时间滑块与全局 MAE 时间序列  
🕸 Network Graph：误差空间分布  
⚖ Model Comparison：MAE / RMSE / MAPE 模型对比柱状图  

## 六、模型文件说明

  
LSTM：src/models/lstm.py，时间序列基线模型。  
DCRNN：src/models/dcrnn.py，图卷积递归神经网络，捕捉时空相关性。  
ASTGCN：src/models/astgcn.py，基于注意力机制的时空图卷积网络，性能最佳。  

## 七、预测文件说明

  
predictions/  
├─ pred_cache_1h_lstm.npz  
├─ pred_cache_1h_dcrnn.npz  
└─ pred_cache_1h_astgcn.npz  

每个 .npz 包含 preds、gts、columns、scaler_mean、scaler_std 等字段。

