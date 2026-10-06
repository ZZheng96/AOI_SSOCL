# SolderSmtAllAlg 可调参数说明

> 所有参数均通过 `config` dict 传入 `run()` 方法，或写在 `params.json` / 配置 JSON 的 `defaultParam` 中。
> 未提供时使用 `default_config()` 中的默认值。

---

## 1. 干扰排除 (Step1)

干扰排除用于识别阻焊层、丝印、元件等非焊锡区域，生成的 `interference mask` 会在后续焊锡提取时排除这些区域。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mask_h_low` | int | 38 | 阻焊层 HSV H 下限 |
| `mask_h_high` | int | 78 | 阻焊层 HSV H 上限 |
| `mask_s_min` | int | 30 | 阻焊层 HSV S 下限 |
| `mask_v_min` | int | 30 | 阻焊层 HSV V 下限 |
| `silk_h_low` | int | 82 | 丝印 HSV H 下限 |
| `silk_h_high` | int | 97 | 丝印 HSV H 上限 |
| `silk_s_low` | int | 75 | 丝印 HSV S 下限 |
| `silk_s_high` | int | 155 | 丝印 HSV S 上限 |
| `silk_v_low` | int | 75 | 丝印 HSV V 下限 |
| `silk_v_high` | int | 175 | 丝印 HSV V 上限 |
| `comp_v_max` | int | 60 | 元件 HSV V 上限（暗色元件） |
| `comp_h_low` | int | 0 | 元件 HSV H 下限 |
| `comp_h_high` | int | 179 | 元件 HSV H 上限 |
| `comp_s_min` | int | 0 | 元件 HSV S 下限 |
| `comp_s_max` | int | 255 | 元件 HSV S 上限 |
| `min_interference_area` | int | 30 | 干扰区域最小面积（像素），小于此值的连通域被忽略 |
| `interference_close_ks` | int | 5 | 干扰 mask 闭运算核大小 |

---

## 2. 焊盘框定位 (Step1+)

从 `pad.json` 加载焊盘框坐标，计算外扩框和分组合并框。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `morph_close_ks` | int | 9 | 形态学闭运算核大小（本地检测焊盘时使用） |
| `morph_open_ks` | int | 5 | 形态学开运算核大小 |
| `min_pad_area` | int | 20 | 焊盘最小面积（像素） |
| `n_classes` | str/int | "auto" | 焊盘聚类类别数，"auto" 自动推断 |
| `cluster_ar_thresh` | float | 0.5 | 第一轮聚类宽高比阈值 |
| `cluster_area_ratio` | float | 0.3 | 第一轮聚类面积比阈值 |
| `cluster_ar_thresh_r2` | float | 0.15 | 第二轮聚类宽高比阈值 |
| `cluster_area_ratio_r2` | float | 0.15 | 第二轮聚类面积比阈值 |
| `pad_expand_ratio` | float | 0.30 | 焊盘外扩比例（长边） |
| `pad_expand_short_ratio` | float | 0.30 | 焊盘外扩比例（短边） |
| `pad_expand_aspect_thresh` | float | 2.0 | 宽高比阈值，超过则使用短边外扩 |
| `pad_expand_match_aspect` | bool | true | 外扩时是否保持宽高比 |

---

## 3. 焊锡提取 (Step2)

从待检图中提取焊锡区域（蓝色金属光泽），是所有缺陷判定的基础。

### 3.1 蓝色种子提取

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `solder_blue_h_low` | int | 85 | 焊锡蓝色 HSV H 下限 |
| `solder_blue_h_high` | int | 115 | 焊锡蓝色 HSV H 上限 |
| `solder_blue_s_min` | int | 60 | 焊锡蓝色 HSV S 下限 |
| `solder_blue_v_min` | int | 150 | 焊锡蓝色 HSV V 下限 |

> **注意:** 不同批次/光照条件下焊锡蓝色可能偏移，需根据实际图片调整。
> 可用 OpenCV 颜色拾取工具获取目标焊锡区域的 HSV 值。

### 3.2 形态学与生长

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `solder_noninter_dilate_ks` | int | 15 | 非干扰区膨胀核大小 |
| `solder_growzone_open_ks` | int | 5 | 生长区开运算核大小 |
| `solder_grow_dilate_ks` | int | 3 | 生长膨胀核大小 |
| `solder_grow_max_iters` | int | 0 | 最大生长迭代次数（0=不生长） |
| `solder_max_growth_ratio` | float | 0 | 最大生长比例（0=不限制） |
| `solder_growth_check_interval` | int | 0 | 生长检查间隔（0=不检查） |

### 3.3 辅助开关（默认关闭，可按需开启）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `solder_enable_noninter_process` | bool | false | 是否启用非干扰区后处理 |
| `solder_enable_sv_ok` | bool | false | 是否启用 SV 约束过滤 |
| `solder_enable_texture_filter` | bool | false | 是否启用纹理过滤 |

### 3.4 SV/纹理参数（辅助开关开启时生效）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `solder_min_blue_ratio` | float | 0.1 | 最小蓝色比例 |
| `solder_v_min_blue` | int | 150 | 蓝色 V 下限 |
| `solder_v_min_interf` | int | 150 | 干涉区 V 下限 |
| `solder_v_min_red` | int | 80 | 红色 V 下限 |
| `solder_s_min_blue` | int | 60 | 蓝色 S 下限 |
| `solder_s_min_interf` | int | 30 | 干涉区 S 下限 |
| `solder_s_min_red` | int | 0 | 红色 S 下限 |
| `solder_s_max_red` | int | 100 | 红色 S 上限 |
| `solder_v_min_global` | int | 110 | 全局 V 下限 |
| `solder_wv_w_scale` | float | 45.0 | W-V 校正 W 缩放系数 |
| `solder_wv_v_scale` | float | 80.0 | W-V 校正 V 缩放系数 |
| `solder_wv_thresh` | float | 1.0 | W-V 校正阈值 |
| `solder_wv_w_weight` | float | 0.6 | W-V 校正 W 权重 |
| `solder_wv_v_weight` | float | 0.4 | W-V 校正 V 权重 |
| `solder_close_ks` | int | 7 | 焊锡 mask 闭运算核大小 |
| `solder_texture_ksize` | int | 7 | 纹理检测核大小 |
| `solder_texture_thresh` | float | 3.0 | 纹理检测阈值 |

### 3.5 W 校正 LUT 段边界

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `w_lut_a_hi` | int | 25 | LUT A 段上界 |
| `w_lut_b_hi` | int | 35 | LUT B 段上界 |
| `w_lut_c_hi` | int | 77 | LUT C 段上界 |
| `w_lut_d_hi` | int | 100 | LUT D 段上界 |
| `w_lut_w_a` | int | 60 | LUT A 段 W 值 |
| `w_lut_w_b` | int | 3 | LUT B 段 W 值 |
| `w_lut_w_c` | int | 4 | LUT C 段 W 值 |
| `w_lut_w_d` | int | 3 | LUT D 段 W 值 |

---

## 4. 模板对齐

有模板图时，使用 ECC 算法对齐模板和待检图，计算缩放和平移变换。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ECC_EUCLIDEAN` | bool | false | true=平移变换, false=仿射变换 |
| `crack_cpp_min_pads` | int | 4 | C++批量crack检测的最小pad数(0=始终用Python, 1=始终用C++) |
| `scale_min` | float | 0.5 | 允许的最小缩放比 |
| `scale_max` | float | 2.0 | 允许的最大缩放比 |
| `grow_outside_region` | bool | true | 焊锡生长是否允许超出组框 |
| `seed_outside_region` | bool | true | 焊锡种子是否允许超出组框 |

---

## 5. 缺陷判定 (Step3)

### 5.1 少锡 (insufficient)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `insufficient_thresh` | float | 0.08 | 少锡阈值：焊锡覆盖率低于此值判定为少锡 |

### 5.2 多锡 (excess)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `excess_thresh` | float | 0.08 | 多锡阈值：焊盘外焊锡占比超过此值判定为多锡 |

### 5.3 连锡 (bridge)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bridge_min_area` | int | 20 | 连锡最小面积（像素），小于此值忽略 |

### 5.4 虚焊 (cold_solder)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cold_solder_dark_ratio` | float | 0.30 | 暗区比例阈值 |
| `cold_solder_v_dark` | int | 80 | 暗区 V 值阈值 |
| `pin_type` | str | "gull-wing" | 引脚类型：`gull-wing` 或 `terminal` |
| `cold_solder_rim_ratio` | float | 0.5 | 边缘焊锡覆盖率阈值（Rule1） |
| `toe_metal_ratio_thresh` | float | 0.3 | toe 金属覆盖率阈值（Rule1） |
| `cold_diff_ratio_thresh` | float | 0.38 | diff 覆盖率阈值（Rule3） |
| `crack_min_length_ratio` | float | 0.5 | crack线段最小长度/pad短边比例 |
| `toe_metal_h_low` | int | 80 | toe 金属 HSV H 下限 |
| `toe_metal_h_high` | int | 124 | toe 金属 HSV H 上限 |
| `toe_metal_s_min` | int | 39 | toe 金属 HSV S 下限 |
| `toe_metal_s_max` | int | 78 | toe 金属 HSV S 上限 |
| `toe_metal_v_min` | int | 125 | toe 金属 HSV V 下限 |
| `toe_metal_v_max` | int | 222 | toe 金属 HSV V 上限 |

---

## 6. 模板差分 (Step4, 可选)

有模板时可启用模板差分检测（RB_score 方向性差异）。默认通过 `skip_rb_score=true` 跳过。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `skip_rb_score` | bool | true | true=跳过 RB_score 差分（加速），false=启用 |
| `merge_compute_diff` | bool | true | true=合并两次 compute_diff 计算为一次 |
| `skip_diff_heatmap` | bool | true | true=跳过热力图生成（加速） |
| `RB_thresh_ins` | int | 100 | 少锡 RB_score 阈值 |
| `RB_thresh_exc` | int | 100 | 多锡 RB_score 阈值 |
| `RB_red_ratio_min` | float | 0.03 | RB 红色比例最小值 |
| `diff_insufficient_thresh` | float | 0.05 | 差分少锡阈值 |
| `insufficient_use_diff` | bool | true | 少锡是否使用 diff 结果 |
| `bluer_h_thresh` | int | 30 | 偏蓝 H 阈值 |
| `diff_thresh` | int | 80 | 差分二值化阈值 |
| `diff_min_area` | int | 10 | 差分最小面积 |

---

## 7. 自动缩放 (auto_resize)

当图片尺寸过大时自动等比缩小，减少计算量。坐标在返回前自动还原到原图尺寸。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_auto_resize` | bool | true | 是否启用自动缩放 |
| `auto_resize_max_size` | int | 400 | 缩放后图片的最大边长（像素） |
| `auto_resize_threshold` | int | 450 | 触发缩放的阈值（图片宽或高超过此值时缩放） |

> **注意:** 缩小图片可能影响少锡检测灵敏度。如需更高检出率可关闭此功能。

---

## 8. 模式控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `template_mode` | str | "diff_grow_ROI" | 模板模式（实际固定为 `diff_grow_ROI`） |
| `provide_pad_frames` | bool | false | 是否提供焊盘框（实际始终为 true） |

---

## 9. 缺陷开关

控制启用的缺陷检测类型，可按需关闭不需要的检测项。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `detect_excess` | bool | true | 是否检测多锡 |
| `detect_insufficient` | bool | true | 是否检测少锡 |
| `detect_bridge` | bool | true | 是否检测连锡 |
| `detect_cold_solder` | bool | true | 是否检测虚焊 |

---

## 10. Debug 参数

调试用参数，生产环境不需要。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `debug_save_solder_mask` | bool | false | 是否保存焊锡 mask 中间图 |
| `debug_output_dir` | str | "_debug_solder_mask" | 中间图输出目录 |
| `_debug_pair_tag` | str | "unknown" | 中间图文件名标识 |

---

## 10b. 可视化与性能参数

控制可视化输出和性能优化开关。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_visualization` | bool | true | 是否生成可视化输出图（`output_image`），生产环境设 false 可提速 ~3ms |
| `skip_diff_morphology` | bool | false | true=跳过 diff mask 形态学去噪（可能影响检测，慎用） |
| `diff_min_area` | int | 10 | diff mask 连通域最小面积，设 0 可跳过连通域过滤提速 ~1.5ms |

### 性能优化建议

| 场景 | 配置 | 平均耗时 |
|------|------|---------|
| 调试/Tuner | `enable_visualization=true` | ~20ms |
| 生产环境 | `enable_visualization=false` | ~15ms |
| 极致性能 | `enable_visualization=false`, `diff_min_area=0` | ~14ms |

> **注意:** `skip_diff_morphology=true` 会导致部分检测结果变化（多锡漏检），仅在不影响质量的场景使用。

---

## 11. 数据输入参数

以下参数由调用方提供，不在 `default_config()` 中：

| 参数 | 类型 | 说明 |
|------|------|------|
| `pad_frames` | List[[x,y,w,h]] | 焊盘框列表，来自 pad.json |
| `group_frames` | List[[x,y,w,h]] | 分组框列表（可选） |
| `toe_frames` | List[[x,y,w,h]] | toe 标注框列表（可选，虚焊 Rule1 用） |
| `rim_frames` | List[[x,y,w,h]] | rim 标注框列表（可选，虚焊 Rule1 用） |
| `pad_json_path` | str | pad.json 文件路径（可选，自动加载 toe/rim） |
| `pad_frames_source` | str | pad.json 文件路径（可选，同 pad_json_path） |
| `loose` | bool | 宽松干扰排除模式（有模板时自动 true） |
