# FourierTM 操作指南

基于 SimpleTM (ICLR 2025) 改进，三个创新点：改进 ASB + FFT 替换 SWT + 解耦双路注意力。

## 项目结构

```
FourierTM/
├── model/
│   ├── SimpleTM.py              # 原始模型（保留对照）
│   └── FourierTM.py             # 新模型（支持消融开关）
├── layers/
│   ├── SWTAttention_Family.py   # 原始 SWT + GeomAttention
│   ├── FourierAttention.py      # 新：FFT tokenization + 双路/合并注意力
│   ├── ASB.py                   # 新：改进 ASB 模块
│   ├── Embed.py                 # 共用
│   ├── StandardNorm.py          # 共用
│   └── Transformer_Encoder.py   # 共用
├── scripts/
│   ├── run_fouriertm_all.sh     # 一键跑 FourierTM Full × 8 数据集
│   ├── run_ablation_all.sh      # 消融实验：5 变体 × 8 数据集
│   ├── run_fouriertm_quick.sh   # 快速验证（ETT, itr=1）
│   ├── run_ablation_ett.sh      # ETT 消融（5 变体 × 2 数据集）
│   └── multivariate_forecasting/
│       ├── ETT/
│       │   ├── SimpleTM_h1.sh / SimpleTM_h2.sh / SimpleTM_m1.sh / SimpleTM_m2.sh
│       │   └── FourierTM_h1.sh / FourierTM_h2.sh / FourierTM_m1.sh / FourierTM_m2.sh
│       ├── ECL/          SimpleTM.sh + FourierTM.sh
│       ├── Traffic/       SimpleTM.sh + FourierTM.sh
│       ├── Weather/       SimpleTM.sh + FourierTM.sh
│       └── SolarEnergy/   SimpleTM.sh + FourierTM.sh
├── dataset/
│   ├── ETT-small/         # ETTh1/h2/m1/m2.csv
│   ├── electricity/       # electricity.csv
│   ├── traffic/           # traffic.csv
│   ├── weather/           # weather.csv
│   └── solar/             # solar_AL.txt
├── verify_fouriertm.py    # 本地验证脚本（9项测试）
└── run.py                 # 入口（已添加消融参数）
```

## 三个创新点

1. **改进 ASB**：FFT → 自适应软掩膜 + 全局/局部滤波器 + 频带注意力 + Per-channel 门控
2. **FFT 替换 SWT**：用 FFT 频带分割替代小波变换做多尺度 tokenization
3. **解耦双路注意力**：dot-product 和 wedge-product 拆分为独立路径 + 可学习门控融合

## 消融实验开关

三个命令行参数控制组件开关，无需改代码：

| 参数 | 选项 | 默认 |
|------|------|------|
| `--use_asb` | `0` / `1` | `1` |
| `--tokenization` | `fft` / `swt` | `fft` |
| `--attention_mode` | `dual_path` / `combined` | `dual_path` |

五个消融变体：

| 变体 | 命令 |
|------|------|
| SimpleTM baseline | `--model SimpleTM` |
| +FFT only | `--model FourierTM --use_asb 0 --tokenization fft --attention_mode combined` |
| +ASB only | `--model FourierTM --use_asb 1 --tokenization swt --attention_mode combined` |
| +DualPath only | `--model FourierTM --use_asb 0 --tokenization swt --attention_mode dual_path` |
| FourierTM (Full) | `--model FourierTM --use_asb 1 --tokenization fft --attention_mode dual_path` |

---

## 操作流程

### Step 0: 本地验证（已完成）

```bash
conda run -n base python verify_fouriertm.py
# 9/9 PASS
```

### Step 1: 准备数据集

**ETT（已有）：**

```bash
ls dataset/ETT-small/
# ETTh1.csv  ETTh2.csv  ETTm1.csv  ETTm2.csv
```

**大数据集（需下载）：**

```bash
# Electricity (321 channels)
mkdir -p dataset/electricity/
# 从 https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014 下载
# 或从 SimpleTM 论文提供的 Google Drive 链接

# Weather (21 channels)
mkdir -p dataset/weather/
# https://www.bgc-jena.mpg.de/wetter/

# Traffic (862 channels)
mkdir -p dataset/traffic/
# http://pems.dot.ca.gov/

# Solar-Energy (137 channels)
mkdir -p dataset/solar/
# https://www.nrel.gov/grid/solar-power-data.html
```

建议：从 https://github.com/thuml/Time-Series-Library 的 Google Drive 链接一次性下载所有数据集。

### Step 2: 上传到服务器

**方式 A: GitHub（推荐）**

代码已推送到私有仓库 https://github.com/1053581017/FourierTM

```bash
# 服务器上首次 clone
git clone https://github.com/1053581017/FourierTM.git
cd FourierTM/

# 后续更新（本地改完代码后）
# 本地: git add -A && git commit -m "描述" && git push mine main
# 服务器: cd FourierTM && git pull
```

**方式 B: scp 直传**

```bash
# 本地打包
tar czf FourierTM.tar.gz \
  --exclude='.git' --exclude='__pycache__' --exclude='checkpoints' --exclude='results' \
  -C "01_Projects/技术支持/" FourierTM/

# 上传 + 解压
scp FourierTM.tar.gz <user>@<server>:~/
ssh <user>@<server> "tar xzf FourierTM.tar.gz && cd FourierTM/"
```

### Step 3: 安装依赖

```bash
pip install torch numpy pandas scikit-learn pywt
# pywt 是 SWT 消融变体需要的
```

### Step 4: 跑 FourierTM Full（全部 8 个数据集）

```bash
# 一键跑全部（建议 tmux/nohup）
nohup bash scripts/run_fouriertm_all.sh > fouriertm_all.log 2>&1 &
tail -f fouriertm_all.log
```

或者按数据集单独跑（可分配到不同 GPU）：

```bash
# GPU 0: ETT
CUDA_VISIBLE_DEVICES=0 bash scripts/multivariate_forecasting/ETT/FourierTM_h1.sh &
# GPU 1: 大数据集
CUDA_VISIBLE_DEVICES=1 bash scripts/multivariate_forecasting/ECL/FourierTM.sh &
```

### Step 5: 消融实验（全部 8 个数据集）

```bash
nohup bash scripts/run_ablation_all.sh > ablation_all.log 2>&1 &
```

5 变体 × 8 数据集 × 4 pred_len = 160 组实验。

### Step 6: 查看结果

```bash
cat result_long_term_forecast.txt
# model_id 包含变体名，如 ETTh1_FourierTM、ECL_FFT_only 等
```

**验收标准：** 8 个数据集的 avg MSE，至少 4 个比 SimpleTM 好。

### Step 7: 调参（如需要）

| 参数 | 说明 | 建议范围 |
|------|------|---------|
| `--asb_n_bands` | ASB 频带数 | 0, 2, 4, 8 |
| `--learning_rate` | 主学习率 | 0.001 ~ 0.02 |
| `--m` | 频带层数 | 1, 2, 3 |
| `--alpha` | 注意力权重 | 0.0 ~ 1.0 |
| `--d_model`, `--d_ff` | 模型维度 | 32, 64, 128, 256 |

ASB 学习率固定为主 lr 的 0.1 倍（在 `exp_long_term_forecasting.py` 中设置）。

---

## SimpleTM 论文 Baseline（Table 1, avg MSE）

| 数据集 | MSE | MAE | 通道数 |
|--------|-----|-----|--------|
| ETTh1 | 0.422 | 0.428 | 7 |
| ETTh2 | 0.353 | 0.391 | 7 |
| ETTm1 | 0.381 | 0.396 | 7 |
| ETTm2 | 0.275 | 0.322 | 7 |
| ECL | 0.166 | 0.260 | 321 |
| Traffic | 0.444 | 0.390 | 862 |
| Weather | 0.243 | 0.271 | 21 |
| Solar | 0.184 | 0.247 | 137 |

---

## 常见问题

**Q: SWT 变体报错 `No module named 'pywt'`**
A: `pip install PyWavelets`，只有 `--tokenization swt` 的变体需要。

**Q: 如何只跑单个变体？**
A: 直接用 `python -u run.py` 加对应参数，参见上方消融表格。

**Q: 如何只跑某个数据集？**
A: 跑 `scripts/multivariate_forecasting/<Dataset>/FourierTM.sh`。

**Q: Traffic 太慢怎么办？**
A: Traffic 862 通道，d_model=1024，本身就很慢。可以先只跑 pred_len=96，或降 batch_size。

**Q: 大数据集在哪下载？**
A: 推荐从 [Time-Series-Library](https://github.com/thuml/Time-Series-Library) 的 Google Drive 一次下载全部。
