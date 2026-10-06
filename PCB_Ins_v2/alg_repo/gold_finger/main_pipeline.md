# PCB 缺陷检测流水线文档

`main_pipeline.py` 是金手指（PCB 沉金触点）缺陷检测的总控流水线，对同一视场（FOV）的良品图与待测图做像素级对比，输出缺陷红框及可选类别标签。

**执行链路：**

```
图像配准 → 板面掩膜提取 → ECC 精配准 → 全局/局部光照归一化 → 邻域容差差分 → [掩膜形状兜底] → [缺陷分类]
```

---

## 文件结构

```
8.2/
├── main_pipeline.py          # 总控流水线（含掩膜提取、光照匹配、缺陷检测、分类调度）
├── peizhun.py                # 混合配准：相位相关 + ORB + NCC + 锚点候选 → 择优
├── seed_extract.py           # 种子点交互取色 + HSV 范围自动生成
├── defect_features.py        # 缺陷区域手工特征（41 维：几何/颜色/纹理）
├── forest_infer.py           # 随机森林纯 NumPy 推理器 + sklearn 导出
├── extract_features.py       # 从 X-AnyLabeling 标注提取训练特征集
├── train_classifier.py       # 随机森林训练 → 交叉验证 → 导出 .npz
├── profile_align.py          # 配准+掩膜内部耗时细分工具
├── profile_run.py            # 完整链路分级计时工具（多图组、无缓存）
├── defect_classifier.npz     # [选项] 在线分类模型
├── pipeline_results/         # 输出目录（含缓存、取色配置、结果图）
└── 修复记录_金手指边缘误检与配准.md  # 2026-08-02 修复记录
```

**依赖：** `opencv-python`、`numpy`。分类器训练额外需要 `scikit-learn`、`joblib`。

**运行：** 在 `main()` 中修改 `path_good` / `path_test`，然后 `python main_pipeline.py`。

---

## 配置参数

均在 `main()` 函数开头集中设置。

### 输入与输出

| 参数 | 说明 |
|------|------|
| `path_good` | 良品模板图路径 |
| `path_test` | 待测图路径 |
| `output_dir` | 输出目录，默认 `pipeline_results` |

### 板面颜色

| 参数 | 默认 | 说明 |
|------|------|------|
| `BOARD_COLOR` | `"seed"` | `"seed"`（推荐）/ `"gold"` / `"custom"` |
| `SEED_REPICK` | `False` | `True` 时忽略已保存配置，强制重新弹窗取色 |
| `CUSTOM_HSV_RANGES` | 见代码 | custom 模式下的 HSV 范围列表，多段取并集 |
| `MIN_REGION_AREA` | `None` | 掩膜最小连通域面积（原图像素），`None` = 按分辨率自适应 |

**板面类型行为：**

| 值 | 行为 |
|----|------|
| `seed` | 交互取色 → 保存 `_seed_ranges.json` → 内部转为 `custom` 运行 |
| `gold` | 内置阈值 `H∈[10,50], S∈[25,255], V∈[60,255]` |
| `custom` | 直接使用 `CUSTOM_HSV_RANGES` |

取消取色时自动回退到 `gold`。

**取色窗口操作**（`seed_extract.py`）：左键加点，右键删最近点，`+/-` 调容差，`z` 撤销，`r` 清空，Enter/空格确认，Esc 取消。取色 patch 尺寸自适应——在多个候选尺寸中选 HSV 分布最均匀的，自动收缩到点击目标色块内部，避免混入指缝背景。

### 检测参数（`detect_defects` 调用处）

| 参数 | 默认 | 说明 |
|------|------|------|
| `diff_thresh` | 20 | 差分判定阈值，越小越敏感。误检多调大，漏检多调小 |
| `min_defect_area` | 5 | 最小缺陷面积（像素） |
| `shift_tolerance` | 1 | 邻域容差半径（px），吸收配准残余偏移 |
| `edge_margin` | 1 | 掩膜边界向内屏蔽宽度（px） |
| `edge_band` | 自适应 | 块外轮廓阈值渐变带宽，`max(6, long_side/300)` |
| `edge_penalty` | 1.0 | 块最外沿阈值 = `diff_thresh × (1+edge_penalty)`，仅作用于块外轮廓带 |
| `edge_thresh_cap` | `None` | 抬高后阈值的绝对上限，担心漏边缘大缺陷时可设 80 |
| `display_img` | 未模糊掩膜后待测图 | 画框用的底图 |

### 掩膜形状检测参数（步骤 4.5）

| 参数 | 默认 | 说明 |
|------|------|------|
| `shape_tol` | `max(2, long_side/800)` | 容差带宽度（px），吸收配准残差 |
| `shape_min_area` | 自适应 | 形状缺陷最小面积，误报碎块时调大 |

---

## 流水线步骤

### 0. 读取与取色

读取两图（`cv2.imread`）。`BOARD_COLOR = "seed"` 时调用 `load_or_pick_seed_ranges()`：优先读取已保存的 `_seed_ranges.json`；不存在或 `SEED_REPICK=True` 则弹窗取色并保存。生成的多段 HSV 范围写入 `CUSTOM_HSV_RANGES`，`BOARD_COLOR` 内部切换为 `"custom"` 走统一路径。

### 1. 图像配准 — `peizhun.align_image(moving, template)`

将待测图对齐到良品模板。内部流程：

1. **尺寸统一**：待测图 resize 到模板尺寸（放大用 INTER_CUBIC，缩小用 INTER_AREA）
2. **相位相关**（阶段一）：中心 80% 区域 + Hanning 窗抑制 FFT 边界伪峰。响应 ≥ 0.2 且平移量 ≤ 裁剪区 25% 时直接返回，否则进入阶段二
3. **ORB 特征匹配兜底**（阶段二）：中心 ROI 内提取 ORB 特征 → KNN + ratio test 筛选 → RANSAC 估计仿射。校验：内点数 / 内点率 / 旋转 ≤ ±15° / 缩放 ∈ [0.85,1.15] / 平移 ≤ 25% 图像尺寸
4. **候选择优**（阶段三）：当相位与 ORB 均不可靠时，在以下候选中实测降采样灰度图配准残差，残差最小者生效：
   - 恒等变换（不退反进）
   - 相位平移
   - ORB 仿射
   - **NCC 平移候选**：中央大块归一化互相关搜索（半径 1/8 图像尺寸）
   - **锚点平移候选**：仅在检测到显著周期时启用。计算独特性权重图（与自身平移一个周期的差异），裁出独特性最强的窗口做全图模板匹配；锚点密度峰 ≥ 2× 全图均值才启用
5. **独特性加权残差**：当锚点候选可用时，残差用独特性权重加权——周期条纹权重趋零，十字基准等独特结构主导判断，避免"错开整数个周期后平均残差几乎不变"的假对齐

返回 `(ok, aligned, meta)`。`meta["method"]` 取值为 `phase` / `orb_affine` / `ncc_shift` / `anchor_shift` / `lowconf_*`；`meta["fallback_errors"]` 记录各候选实测残差。`ok=False` 时流水线仍继续执行。

### 2. 良品板面掩膜 — `get_pcb_alpha_mask(im_good, ...)`

仅在良品图上算一次，输出 `(H, W, 1)` float32 Alpha 软掩膜：

1. **自适应放大**：长边 ≤ 800 → ×8，≤ 2000 → ×2，否则 ×1。形态学参数按有效尺度（放大倍数 × 分辨率倍率）缩放
2. **HSV 颜色提取**：gold 内置阈值 / custom 多段范围取并集
3. **形态学清理**：闭运算（核尺寸上限 8 原图像素，不桥接指缝）+ 开运算去噪 → 面积过滤噪点（自适应/手动）→ 中值滤波 → 孔洞填充
4. **软边缘**：高斯羽化 + smoothstep 过渡（0.35~0.65），避免硬边缘在配准残差处产生条带误检

### 2.5. 待测掩膜 + ECC 精配准

此步骤在 **色彩归一化之前** 执行（归一化会平移背景色进入板面 HSV 范围，导致掩膜铺满全图）。

1. **提取待测掩膜**：在 `aligned_test` 上运行 `get_pcb_alpha_mask`，乘以 `valid`（非全黑区域）排除 warp 黑边
2. **面积合理性检查**：待测掩膜面积与良品掩膜比值 ∈ [0.5, 1.5]，超出则自动禁用掩膜交集与形状检测
3. **ECC 仿射精配准**（`refine_alignment_ecc`）：在两掩膜 alpha 场上做 `findTransformECC`（MOTION_AFFINE），收敛至亚像素精度。校验条件：
   - 相关系数 > 0.5
   - 掩膜残差真实降低（排除"错周期"局部最优）
   - |旋转| ≤ 5°、缩放 ∈ [0.9, 1.1]、|平移| ≤ 10% 图像尺寸
4. **合成单一仿射**（`compose_refined_affine`）：粗配准与 ECC 精修合成为一个 2×3 矩阵，对原图一次性 warp，避免二次重采样损失高频细节
5. ECC 通过后在新的 `aligned_test` 上重新提取待测掩膜

### 3. 光照归一化

统计范围：`valid` 非黑边 ∩ `alpha > 0.99` 核心区。

1. **全局线性匹配**（`match_color_linear`）：逐通道均值/方差匹配，增益 clip 到 [0.5, 2.0]
2. **低频光照场校正**（`match_illumination_local`）：掩膜加权大尺度低通估计两图各自亮度场，按 `ref_low / src_low` 比值逐像素校正，消除灯光角度/暗角/局部阴影等非均匀亮度差。增益 clip 到 [1/1.6, 1.6]

### 4. 缺陷检测 — `detect_defects`

1. **预处理**：两图 3×3 高斯模糊后乘以良品 alpha 掩膜（差分用模糊图，画框用未模糊图）
2. **邻域容差差分**：`diff = max(test - dilate(good), erode(good) - test)`，仅超出良品图邻域 min/max 包络的像素才计入差异
3. **取 BGR 三通道最大差异**，任一通道变化即捕捉
4. **"整块"掩膜构建**（`_build_block_mask`）：
   - 闭运算核从 15px 起翻倍尝试，直到连通域数量不再减少——桥接指条但不合并相距很远的独立金区
   - 在缩至 500px 的小图上执行，大核保持毫秒级
   - 垫一圈零边界计算距离场，金面延伸到图像边界时一并作为块外轮廓
5. **核心区过滤**：
   - `alpha > 0.99`（排除羽化带）+ 腐蚀 `edge_margin`
   - 块外轮廓带内取良品/待测掩膜交集，消除"良品是金面、待测抠到背景"的假差异。内部不取交集，避免缺陷处待测掩膜缺失导致真实缺陷被排除
6. **自适应阈值**：块外轮廓带（`edge_band` 像素内）用距离线性衰减的抬高阈值 `diff_thresh × (1 + edge_penalty × band)`；内部用固定 `diff_thresh`。`edge_thresh_cap` 可设绝对上限，保证边缘严重缺陷仍能报出
7. **形态学清理**：开运算 → 连通域面积过滤（`min_defect_area`）→ 向量化去噪
8. 返回 `(result_img, diff_mask, boxes, pixel_count)`

### 4.5. 掩膜形状差异检测 — `detect_mask_shape_defects`

补盲检测——步骤 4 在块外轮廓带内取了掩膜交集，交集之外的真实边缘缺料/多料成为盲区。

- 一方掩膜硬阈值（>0.5）超出另一方 `tol_px` 容差带（吸收配准残差）的部分判为缺料/多料
- `valid` 掩膜排除 warp 视野外区域
- 开运算 + 面积过滤后并入 `diff_mask` 与 `defect_boxes`

### 5. 缺陷分类（可选）

仅当 `defect_classifier.npz` 存在且有缺陷框时执行。

- `classify_defects(diff_mask, norm_test, model_path)` — 特征底图必须用完整的 `norm_test`（非 alpha 抠过的图），与训练时口径一致
- 每个连通域外扩 48px 取 ROI → 提取 41 维手工特征 → NumPy 森林推理
- `annotate_defect_labels` — 在原图对应红框上方写类别名与置信度

**框坐标回映**（`map_boxes_to_source`）：配准后（模板坐标系）的缺陷框经仿射逆变换 + 原图/模板尺寸比例缩放，映射回原始待测图坐标，最终画在原图上。

---

## 特征体系（41 维）

训练与推理共用 `defect_features.py`，保证口径一致。

| 组别 | 维度 | 内容 |
|------|------|------|
| 几何 (13) | `log_area_rel`, `rel_diameter`, `elongation`, `extent`, `solidity`, `circularity` + Hu 矩 ×7 | 相对尺寸/伸长率/实心度/圆度/Hu 矩（对数压缩），长边归一化 |
| 颜色 (15) | H 均值向量 (cos/sin) + S/V 均值与标准差 + Lab 均值 (L/a/b) + 内外环带色差 (dL/dA/dB/dS/dV/dHue) | HSV + Lab 区域统计，内外环带差抵消板间光照整体差异 |
| 纹理 (13) | 梯度均值/标准差 + Laplacian 标准差 + riu2 LBP 直方图 ×10 | 梯度强度、旋转不变均匀 LBP |

---

## 预处理缓存

文件：`pipeline_results/_preprocess_cache.npz`

缓存键格式：
```
{path_good}|{path_test}|{BOARD_COLOR}|{CUSTOM_HSV_RANGES}|{MIN_REGION_AREA}|norm_v9
```

缓存内容：`norm_test` / `alpha_mask` / `alpha_mask_test` / `valid` / `affine`

命中时跳过步骤 1~3，只重跑检测。换图 / 改板面类型 / 改 HSV / 改 `MIN_REGION_AREA` / 源图 mtime 变化 / 版本号升级，任一项变化即失效重算。`alpha_mask_test` 不可靠时以空数组占位，加载时还原为 `None`。

---

## 输出产物

| 文件 | 说明 |
|------|------|
| `1_aligned_and_normed.jpg` | 配准 + 光照归一化后的待测图 |
| `2_extracted_good.jpg` | 良品图乘以 alpha 掩膜 |
| `3_extracted_test.jpg` | 归一化待测图乘以 alpha 掩膜 |
| `4_aligned_boxes.jpg` | 配准空间预览：红框标注缺陷（模板坐标系） |
| `4_final_result.jpg` | 最终结果：原待测图 + 红框 + 分类标签 |
| `4_diff_mask.jpg` | 二值差分掩膜（含形状检测） |
| `_preprocess_cache.npz` | 预处理缓存 |
| `_seed_ranges.json` | 种子点取色配置 |

---

## 训练分类器

```
python extract_features.py [标注数据目录]
python train_classifier.py
```

- `extract_features.py` 解析 X-AnyLabeling JSON 标注（`image_test/`），逐多边形提取特征，输出 `features_dataset.npz`
- `train_classifier.py` 加载特征集 → 5 折分层交叉验证 → 全量训练（200 棵，`class_weight="balanced"`）→ 导出 `defect_classifier.joblib`（备档）+ `defect_classifier.npz`（在线推理，NumPy 毫秒级加载，不依赖 sklearn）
- 导出时自动做一致性自检：NumPy 森林预测与 sklearn 原模型对齐
- 同义标签归并（`HuaShang` → `HuaHen`），样本数 < 5 的类别自动剔除

---

## 性能剖析工具

- **`profile_align.py`**：细分 `align_image` 与 `get_pcb_alpha_mask` 内部各环节耗时（相位/ORB/NCC/锚点/周期估计/残差评估/warp/掩膜各步）
- **`profile_run.py`**：多图组完整链路分级计时（无 GUI、无缓存），输出每步绝对耗时与占比

---

## 核心 API 速查

```python
# 配准
ok, aligned, meta = align_image(moving, template, enable_fast_mode=True)
# meta["method"]: phase / orb_affine / ncc_shift / anchor_shift / lowconf_*
# meta["fallback_errors"]: {"identity": x, "phase": x, "orb": x, ...}

# 板面掩膜
alpha = get_pcb_alpha_mask(image, board_color="gold",
                           custom_hsv_ranges=None, min_region_area=None)
# → (H, W, 1) float32, 值域 [0, 1]

# 掩膜精配准
ok_ecc, warp = refine_alignment_ecc(alpha_good, alpha_test)
combined = compose_refined_affine(coarse_affine, ecc_warp)

# 光照归一化
norm = match_color_linear(source, reference, valid_mask=mask)
norm = match_illumination_local(norm, reference, valid_mask=mask)

# 缺陷检测
result, mask, boxes, n = detect_defects(
    im_good_masked, im_test_masked, alpha_mask,
    diff_thresh=20, min_defect_area=5,
    shift_tolerance=1, edge_margin=1,
    edge_band=12, edge_penalty=1.0, edge_thresh_cap=None,
    alpha_mask_test=alpha_mask_test, display_img=im_test_display)

# 掩膜形状兜底
shape_mask, shape_boxes = detect_mask_shape_defects(
    alpha_good, alpha_test, valid_mask=valid, tol_px=3, min_area=40)

# 框回映射
src_boxes = map_boxes_to_source(boxes, affine, src_shape, tpl_shape)

# 缺陷分类
classified = classify_defects(diff_mask, norm_test, "defect_classifier.npz")
# → [((x,y,w,h), label, prob), ...]

# 种子点取色
ranges = load_or_pick_seed_ranges(image, config_path, image_path=path, force_repick=False)
# → [((H_low,S_low,V_low), (H_high,S_high,V_high)), ...]
```

---

## 关键设计决策

1. **掩膜交集仅作用于块外轮廓带**：内部不取交集，否则异物/污渍导致待测掩膜局部缺失会连带排除真实缺陷像素
2. **ECC 精配准在掩膜 alpha 场上做**：平滑梯度场比周期性金手指图案更稳定，ECC 可可靠收敛到亚像素精度
3. **配准候选择优 + 独特性加权**：周期性图案上平均残差无法区分正确定位与错周期假对齐，只有基准标记等独特结构能裁决
4. **掩膜闭运算核上限 8px（原图尺度）**：防止指缝被桥接破坏掩膜拓扑；小孔由 flood fill 兜底
5. **色彩归一化在掩膜提取之后**：归一化会平移背景色进入板面 HSV 范围，导致待测掩膜铺满全图
6. **训练/推理特征口径统一**：`defect_features.py` 被 `extract_features.py`（训练）和 `main_pipeline.py`（推理）共用，几何特征以整图长边归一化保证跨分辨率一致
