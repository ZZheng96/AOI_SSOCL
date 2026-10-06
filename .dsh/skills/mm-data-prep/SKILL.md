---
name: mm-data-prep
description: 数学建模数据处理与探索。当需要读取赛题数据、清洗缺失/异常值、特征变换、特征工程、生成 EDA 报告或处理时空/图像/点云数据时使用本 skill。
whenToUse: 拿到原始数据后第一步、建模前数据准备、数据质量存疑时
---

# 数据处理与 EDA

## 操作步骤

1. **读取**：`mmkit.data.readers.read_table` 自动按扩展名读 csv/excel/json/txt（自动编码探测）；
   批量读取：`read_dir_tables`；
2. **清洗**：`clean_df`（去重复/空白/缺失标记/类型推断）→ `fill_missing`（数值中位数/类别众数/插值/分组填充）→ `remove_outliers`（IQR/3σ/MAD）；
3. **变换**：`scale`（standard/minmax/robust）、`onehot`/`label_encode`、`bin_series` 分箱、`add_time_features` 时间特征；
4. **特征工程**：`lag_features`/`rolling_features`/`diff_features`（时序）、`interaction_features`、`agg_features`（分组聚合）、`fft_features`/`window_stats_series`（信号）；
5. **EDA**：`eda_report(df, target=...)` 一键生成分布/相关/缺失/类别图集到 `figures/eda`；
6. **落盘**：清洗后数据存 `data/processed/`，指标存 `data/results/`（`save_json`/`save_df`）。

## 质量门槛（进入建模前必须满足）

- 无未处理的缺失/异常（或已说明处理方式并在论文中交代）；
- 数值列类型正确、类别列已编码或保留；
- 关键变量分布与相关性已可视化（论文"数据探索"节素材）；
- 随机种子固定：`mmkit.utils.seed.set_seed(42)`。

## 资源

- 代码：`code/mmkit/data/`（readers/cleaning/transform/feature_eng/eda）
- 环境：`code/requirements*.txt`（核心档必装）
- 图像/三维数据：图像用 `cv2`/`skimage`（3D 档），点云见 `mm-3d-visual`

## 常见坑

- 中文编码（gbk/utf-8-sig）读错 → 用 `read_table` 自动探测；
- 对整列含文本的列做数值统计 → 先 `pd.to_numeric(errors="coerce")`；
- 时序数据打乱后做滑窗特征 → 数据泄漏；
- EDA 图直接贴论文 → 需按 `mm-plotting` 规范重新出图（300dpi、统一风格）。
