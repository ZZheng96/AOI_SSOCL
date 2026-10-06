# SolderSmtAllAlg 算法设计文档

## 1. 概述

SolderSmtAllAlg 是 SMT（表面贴装）焊锡全缺陷检测算法，一次 `run()` 调用同时检测四类缺陷：

| 缺陷类型 | label | 说明 |
|---------|-------|------|
| 多锡 | `excess` | 焊锡溢出焊盘区域 |
| 少锡 | `insufficient` | 焊盘上焊锡覆盖率不足 |
| 连锡 | `bridge` | 相邻焊盘间焊锡连通 |
| 虚焊 | `cold_solder` | 焊锡与焊盘结合不良（冷焊） |

### 性能指标

| 指标 | 数值 |
|------|------|
| 平均耗时 | ~17ms/pair（默认）/ ~14ms/pair（生产优化） |
| 输入 | 待检图 + 模板图(可选) + pad.json |
| 输出 | `AlgorithmResult` (parts列表 + 可视化图(可选)) |

---

## 2. 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    run(image, config, ...)               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─ Step0: 输入预处理 ──────────────────────────────┐  │
│  │  ROI裁剪 → auto_resize(可选) → 模板同步缩放      │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌─ Step1: 干扰排除 ────────────────────────────────┐  │
│  │  HSV分割: 阻焊层(绿) + 丝印(白) + 元件(暗)       │  │
│  │  → interference mask                              │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌─ Step2: 焊盘定位 ────────────────────────────────┐  │
│  │  pad.json → 模板对齐(ECC) → 坐标映射 → 外扩+合并  │  │
│  │  → pad_rects / outer_rects / group_rects         │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌─ Step3: 焊锡提取 ────────────────────────────────┐  │
│  │  HSV蓝色种子 ∩ 非干扰区 → 形态学膨胀 → 填充      │  │
│  │  → solder_mask                                    │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌─ Step4: 缺陷判定 ────────────────────────────────┐  │
│  │  少锡: 焊盘内焊锡覆盖率 < threshold               │  │
│  │  多锡: 焊盘外焊锡 ∩ diff_mask                     │  │
│  │  连锡: 相邻焊盘间焊锡连通域 ≥ min_area            │  │
│  │  虚焊: 3规则(边缘缺陷/toe金属/diff覆盖率)         │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌─ Step5: 输出 ────────────────────────────────────┐  │
│  │  坐标还原(auto_resize逆变换) → 可视化图           │  │
│  │  → AlgorithmResult                                │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块详解

### 3.1 干扰排除 (`_remove_interference`)

**目的：** 识别阻焊层、丝印、元件等非焊锡区域，避免误检。

**方法：** HSV颜色空间多通道分割
- 阻焊层：H=38~78 (绿色区间)
- 丝印：H=82~97, S=75~155, V=75~175 (白色区间)
- 元件：V<60 (暗色区域)

**输出：** `interference mask`（干扰区域=白色），后续焊锡提取时排除这些区域。

### 3.2 焊盘定位 (`align_template` + pad.json)

**目的：** 从 pad.json 获取焊盘框坐标，通过模板对齐映射到待检图。

**流程：**
1. 从 pad.json 提取 `label="pad"` 的矩形 → `pad_frames`
2. 有模板时：ECC对齐算法计算变换矩阵 → 坐标映射
3. 无模板时：直接使用 pad.json 坐标
4. 计算外扩框（`outer_rects`）和分组合并框（`group_rects`）

**模板对齐：** 使用 `cv2.findTransformECC()` 计算仿射/平移变换，支持缩放范围 `scale_min~scale_max`。

#### 模板模式 (template_mode)

算法设计了7种模板模式，但**当前版本固定为 `diff_grow_ROI`**（代码中硬编码），即 pad.json焊盘框 + grow焊锡提取 + 差分OR融合：

| 模式 | 焊盘来源 | 焊锡方法 | 差分 | 说明 |
|------|---------|---------|------|------|
| `grow_local` | 待检图自动检测 | grow | 无 | 无模板，全自动 |
| `grow_ROI` | pad.json | grow | 无 | 有pad标注，无模板 |
| `grow_pad` | 模板检测+映射 | grow | 无 | 有模板，无pad标注 |
| `diff_ROI` | pad.json | 纯差分 | 有 | 纯差分判定 |
| `diff_pad` | 模板检测+映射 | 纯差分 | 有 | 纯差分判定 |
| **`diff_grow_ROI`** | **pad.json** | **grow** | **有** | **推荐模式（当前固定）** |
| `diff_grow_pad` | 模板检测+映射 | grow | 有 | 有模板，无pad标注 |

> `diff_grow_ROI` 兼顾精度和速度：pad.json提供准确焊盘位置，grow提取焊锡区域，差分约束多锡/虚焊。其他模式保留供未来扩展。

### 3.3 焊锡提取 (`extract_solder`)

**目的：** 从待检图中提取焊锡区域（蓝色金属光泽）。

**流程：**
1. **蓝色种子提取：** `cv2.inRange(hsv, (h_low, s_min, v_min), (h_high, 255, 255))`
2. **V补偿：** H接近h_high时放低V门槛，捕捉低亮蓝色
3. **非干扰区约束：** `mask_blue = mask_blue & non_inter`
4. **形态学处理：** 开运算去噪 → 膨胀生长（`max_iters=0` 时不生长）
5. **填充：** 闭运算填孔

**关键参数：** `solder_blue_h_low/h_high/s_min/v_min` 决定蓝色范围，不同批次/光照需调整。

### 3.4 缺陷判定

#### 3.4.1 少锡 (`judge_insufficient`)

```
覆盖率 = count(solder_mask ∩ pad) / area(pad)
若 覆盖率 < insufficient_thresh(0.08) → 判定少锡
```

#### 3.4.2 多锡 (`judge_excess`)

```
多锡候选 = solder_mask ∩ group_rect - pad_rect
多锡候选 = 多锡候选 ∩ diff_mask  (有模板时)
若 候选面积占比 > excess_thresh(0.08) → 判定多锡
```

#### 3.4.3 连锡 (`judge_bridge`)

```
连通域 = solder_mask ∩ group_rect - pad_rect
若 连通域面积 ≥ bridge_min_area(20) 且连通≥2个pad → 判定连锡
```

使用 `cv2.connectedComponentsWithStats` 替代逐pad mask创建，性能更优。

#### 3.4.4 虚焊 (`judge_cold_solder`)

三规则联合判定（gull-wing引脚）：

| 规则 | 条件 | 说明 |
|------|------|------|
| Rule1 | rim焊锡覆盖率 < 0.5 且 toe金属覆盖率 < 0.3 | 边缘焊锡不足 |
| Rule2 | 焊锡暗区比例 > 0.30 且 V < 80 | 焊锡发暗 |
| Rule3 | diff_mask覆盖率 > 0.38 | 模板差异显著 |

**优化：** 当 `diff_ratio=0`（无差异）时跳过Hough裂痕检测，大幅减少虚焊检测耗时。

### 3.5 模板差分 (`compute_diff`)

**目的：** 有模板时计算待检图与模板的差异区域，约束多锡/虚焊判定。

**流程：**
1. 模板对齐后裁剪重叠区域
2. 灰度差分 → 二值化（`diff_thresh`）
3. 形态学开运算去噪
4. 差分mask映射回待检图坐标

**优化：** `skip_rb_score=True` 时跳过RB_score计算，`merge_compute_diff=True` 时合并两次diff计算为一次。

### 3.6 自动缩放 (`auto_resize_image`)

**目的：** 大图自动缩小，减少计算量。

**规则：**
- 图片宽或高 > `auto_resize_threshold`(450) 时触发
- 等比缩小至最大边 = `auto_resize_max_size`(400)
- pad_frames/group_frames 同步缩放
- 结果坐标在返回前自动还原到原图尺寸

**代价：** 缩小可能影响少锡检测灵敏度（insufficient召回率从100%降至90.9%）。

---

## 4. 文件结构

```
composite_smt_all/
├── __init__.py              # 包标记（空）
├── _solder_core.py          # 纯算法函数（无框架依赖）
└── solder_smt_all_alg.py    # 算法主体（继承IDetectionAlgorithm）

tuner/
├── pcba_tuner_real.py       # 调参工具（导入生产算法，推荐）
└── pcba_tuner.py            # 调参工具（独立版，无需算法包）
```

### 4.1 `_solder_core.py` - 纯算法函数

所有图像处理函数集中在此文件，只依赖 `os/cv2/numpy`，不依赖框架。便于独立测试和复用。

| 函数 | 作用 |
|------|------|
| `_remove_interference` | 干扰排除 |
| `classify_pads` | 焊盘聚类分类 |
| `_expand_rect` | 矩形外扩 |
| `_merge_overlapping_rects` | 重叠矩形合并 |
| `extract_solder` | 焊锡提取 |
| `judge_insufficient` | 少锡判定 |
| `judge_excess` | 多锡判定 |
| `judge_bridge` | 连锡判定 |
| `judge_cold_solder` | 虚焊判定 |
| `align_template` | 模板ECC对齐 |
| `compute_diff` | 模板差分 |
| `auto_resize_image` | 自动缩放 |
| `scale_pad_rects` | pad坐标缩放 |
| `map_rect_to_original` | 坐标映射 |

### 4.2 `solder_smt_all_alg.py` - 算法主体

| 组件 | 作用 |
|------|------|
| `SolderSmtAllAlg` | 算法类，继承 `IDetectionAlgorithm` |
| `default_config()` | 返回97个默认参数 |
| `run()` | 唯一入口，编排Step0~Step5 |
| `_build_output_image()` | PIL可视化图渲染 |
| `_load_toe_rim_annotations()` | toe/rim标注加载 |
| `_run_diff_detection()` | RB_score差分检测 |

### 4.3 `tuner/` - 调参工具

基于 tkinter 的可视化参数调试工具，支持 ROI 绘制、HSV 抽色、实时缺陷判定和参数持久化。详见 [tuner.md](tuner.md)。

| 文件 | 说明 |
|------|------|
| `pcba_tuner_real.py` | 导入生产算法函数，检测结果与产线一致（推荐） |
| `pcba_tuner.py` | 独立版本，自带简化检测逻辑，不依赖算法包 |

---

## 5. 数据流

```
输入:
  image (np.ndarray)          ← 待检图 BGR
  config (dict)               ← 参数 + pad_frames + pad_json_path
  original_template_image     ← 模板图 BGR (可选)
  roi_bbox                    ← ROI区域 (可选)

输出:
  AlgorithmResult
  ├── code: 0(成功) / 1(失败)
  ├── parts: List[DetectPartBox]  ← 缺陷框列表
  │   ├── label: "excess"/"insufficient"/"bridge"/"cold_solder"
  │   ├── x, y, width, height     ← 原图坐标
  │   ├── confidence              ← 置信度
  │   └── metadata                ← 方法/覆盖率等
  └── metadata
      ├── num_defects             ← 缺陷总数
      ├── output_image            ← 可视化图 (BGR np.ndarray)
      ├── pad_count               ← 焊盘数
      ├── solder_area             ← 焊锡面积
      └── defects_enabled         ← 各缺陷开关状态
```

---

## 6. 性能优化策略

| 优化项 | 效果 | 说明 |
|--------|------|------|
| 迭代次数100→0 | -90ms/pair | 跳过焊锡生长循环 |
| skip_rb_score | -18ms/pair | 跳过RB_score差分计算 |
| merge_compute_diff | -6ms/pair | 合并两次diff为一次 |
| uint8替代float32 | -7ms/pair | diff计算用uint8 |
| HSV/Gray预计算复用 | -2ms/pair | 一次cvtColor多处用 |
| crack跳过(diff_ratio=0) | -20ms/pair(多pad) | 无差异时不做Hough |
| auto_resize | -59% | 大图缩小至400x400 |
| connectedComponentsWithStats | -2ms/pair | 替代逐pad mask |
| ROI切片替代全图mask | -1.5ms/pair | 小区域直接切片 |
| enable_visualization=false | -3ms/pair | 生产环境跳过可视化图 |
| diff_min_area=0 | -1.5ms/pair | 跳过diff连通域过滤 |
| HSV预计算传参 | -0.5ms/pair | _remove_interference接受预计算HSV |
| 无模板跳过compute_diff | -5~10ms/pair(无模板) | has_tpl守卫避免无效计算 |

**总效果：** 176.7ms -> 17ms (默认) / 14ms (生产优化) (Pair1), 20062ms -> 695ms (全量41pair)

---

## 7. 调参须知

> **重要：** 以下参数必须根据实际图片和产线条件调整，默认值仅适用于测试集，直接用于生产环境可能检出率不达标。

### 7.1 必须调整的参数

| 参数 | 默认值 | 说明 | 调参方法 |
|------|--------|------|---------|
| `solder_blue_h_low` | 85 | 焊锡蓝色HSV H下限 | 用OpenCV拾取焊锡区域HSV值，取H范围 |
| `solder_blue_h_high` | 115 | 焊锡蓝色HSV H上限 | 同上 |
| `solder_blue_s_min` | 60 | 焊锡蓝色HSV S下限 | 同上 |
| `solder_blue_v_min` | 150 | 焊锡蓝色HSV V下限 | 同上 |

> 焊锡蓝色范围是**最关键参数**。不同批次焊锡、不同光源条件（色温/角度）下HSV值差异很大。错误的蓝色范围会导致焊锡提取不足或过度，直接影响所有四类缺陷的检测。

### 7.2 建议按需调整的参数

| 参数 | 默认值 | 说明 | 调参场景 |
|------|--------|------|---------|
| `insufficient_thresh` | 0.08 | 少锡覆盖率阈值 | 漏检降低，误检升高 |
| `excess_thresh` | 0.08 | 多锡面积占比阈值 | 漏检降低，误检升高 |
| `bridge_min_area` | 20 | 连锡最小面积 | 小焊盘降低，大焊盘升高 |
| `cold_solder_dark_ratio` | 0.30 | 虚焊暗区比例阈值 | 漏检降低，误检升高 |
| `cold_diff_ratio_thresh` | 0.38 | 虚焊diff覆盖率阈值 | 漏检降低，误检升高 |

### 7.3 速度 vs 精度权衡

| 参数 | 速度优先 | 精度优先 | 说明 |
|------|---------|---------|------|
| `enable_auto_resize` | `true` | `false` | 关闭可提升少锡检出率，但耗时增加~3倍 |
| `auto_resize_max_size` | 300 | 600+ | 值越大精度越高但越慢 |
| `skip_rb_score` | `true` | `false` | 开启RB_score差分可提升多锡/连锡精度 |
| `solder_grow_max_iters` | 0 | 5~20 | 增加生长迭代可扩大焊锡区域，但可能过度生长 |
| `enable_visualization` | `false` | `true` | 调试时开启查看可视化图，生产关闭提速~3ms |
| `diff_min_area` | 0 | 10 | 0=跳过连通域过滤提速，10=过滤小噪点 |

### 7.4 干扰排除参数

以下参数用于识别阻焊层/丝印/元件，不同PCB板颜色可能不同：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mask_h_low/high` | 38/78 | 阻焊层H范围（绿色），绿板用默认，蓝板需调整 |
| `silk_h_low/high` | 82/97 | 丝印H范围（白色），不同丝印颜色需调整 |
| `comp_v_max` | 60 | 元件暗度阈值，深色元件需降低 |

### 7.5 虚焊toe/rim相关参数

仅当pad.json包含toe/rim标注时生效：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `toe_metal_h_low/high` | 80/124 | toe金属HSV H范围 |
| `toe_metal_s_min/max` | 39/78 | toe金属HSV S范围 |
| `toe_metal_v_min/max` | 125/222 | toe金属HSV V范围 |
| `cold_solder_rim_ratio` | 0.5 | rim焊锡覆盖率阈值 |
| `toe_metal_ratio_thresh` | 0.3 | toe金属覆盖率阈值 |

> toe/rim的HSV范围取决于引脚金属的色泽，不同镀层（锡/金/银）需调整。

### 7.6 调参建议流程

1. 用 tuner 工具加载图片，绘制 pad/toe/rim 框，调节 HSV 范围和阈值（详见 [tuner.md](tuner.md)）
2. 先用默认参数跑一批图片，查看可视化结果图
3. 若焊锡mask提取不准 -> 调整 `solder_blue_*` 参数
4. 若干扰排除不准 -> 调整 `mask_*`/`silk_*`/`comp_*` 参数
5. 若少锡/多锡误检或漏检 -> 调整对应 `*_thresh` 阈值
6. 若虚焊误检或漏检 -> 调整 `cold_solder_*` 参数
7. 可通过 `debug_save_solder_mask=true` 保存焊锡mask中间图辅助调参

---

## 8. 局限性

1. **少锡检测灵敏度：** auto_resize缩小图片后，细微少锡可能漏检（召回率90.9%）
2. **蓝色范围依赖：** 焊锡蓝色HSV范围因批次/光照不同需手动调整
3. **pad.json必需：** 无pad.json时无法运行（返回code=1）
4. **虚焊Rule1依赖标注：** toe/rim标注缺失时仅用Rule2+Rule3
5. **模板对齐精度：** 缩放/旋转过大时ECC可能对齐失败
