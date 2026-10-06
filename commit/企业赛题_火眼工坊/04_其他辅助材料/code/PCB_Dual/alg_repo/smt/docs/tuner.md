# PCBA 调参工具 (tuner)

## 概述

tuner 是基于 tkinter 的可视化参数调试工具，用于 AOI 焊锡缺陷检测算法的参数标定。支持 ROI 绘制、HSV 抽色、实时缺陷判定和参数持久化，所见即所得。

## 两个版本

| 文件 | 说明 | 适用场景 |
|------|------|---------|
| `pcba_tuner_real.py` | 导入生产算法函数 (`_solder_core.py`)，检测结果与产线一致 | 参数标定、缺陷验证（推荐） |
| `pcba_tuner.py` | 独立版本，自带简化检测逻辑，不依赖算法包 | 快速查看 HSV 范围、无算法包时使用 |

## 启动方式

```bash
# 需要设置 PYTHONPATH (pcba_tuner_real.py 依赖算法包)
cd Dev_solder_smt_Alg
PYTHONPATH=src:contract_reference python tuner/pcba_tuner_real.py

# pcba_tuner.py 无需算法包
python tuner/pcba_tuner.py
```

Windows:
```cmd
set PYTHONPATH=src;contract_reference
python tuner\pcba_tuner_real.py
```

## 依赖

- Python 3.8+
- tkinter (Linux: `sudo apt-get install python3-tk`)
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Pillow (`pip install Pillow`)
- Linux 中文字体: `sudo apt-get install fonts-noto-cjk` (可选，改善中文显示)

## 功能模块

### 1. ROI 绘制

绘制三类检测框，保存为 LabelMe 风格 JSON：

| 框类型 | 用途 | 快捷键 |
|--------|------|--------|
| pad | 焊盘区域，少锡/多锡/连锡检测的基础框 | 1 |
| toe | 引脚金属区域，虚焊 Rule2 的 toe 金属占比计算区域 | 2 |
| rim | 虚焊检测边缘框，rim-toe 剩余区域用于焊锡覆盖率判定 | 3 |

编辑模式 (快捷键 e)：
- 点击选中框 -> 拖动移动 / 拖角缩放
- Delete 删除选中框
- Ctrl+Z 撤销

### 2. HSV 抽色

4 类抽色对象，每类独立保存 HSV 范围：

| 对象 | 用途 | 对应参数前缀 |
|------|------|-------------|
| 阻焊 (mask) | 排除绿色阻焊层干扰 | mask_h_low 等 |
| 丝印 (silk) | 排除白色丝印干扰 | silk_h_low 等 |
| 元件 (component) | 排除器件本体干扰 | comp_h_low 等 |
| 焊锡 (solder) | 提取焊锡区域，缺陷判定的基础 | solder_blue_h_low 等 |

操作方式：
- Tab 键切换当前对象
- 6 条滑条调节 H/S/V 的上下限
- 实时预览 Mask 叠加效果
- s 键进入采样模式，点击图片自动填充 HSV 范围

### 3. 实时缺陷判定

调节阈值滑条时实时显示判定结果（仅 pcba_tuner_real.py）：

| 缺陷类型 | 判定条件 | 阈值参数 |
|---------|---------|---------|
| 少锡 (insufficient) | pad 框内焊锡占比 < 阈值 | insufficient_thresh |
| 多锡 (excess) | pad 外扩环焊锡占比 > 阈值 | excess_thresh |
| 连锡 (bridge) | pad 间焊锡连通区域面积 >= 阈值 | bridge_min_area |
| 虚焊 (cold_solder) | rim-toe 区域焊锡占比 < 阈值 | cold_solder_rim_ratio |

虚焊三规则阈值：

| 规则 | 条件 | 阈值参数 |
|------|------|---------|
| Rule1 | rim-toe 焊锡覆盖率 < 阈值 | cold_solder_rim_ratio |
| Rule2 | crack 检出 + toe 金属占比 > 阈值 | cold_solder_toe_metal_ratio |
| Rule3 | crack 检出 + diff 覆盖率 > 阈值 | cold_solder_diff_ratio |
| crack长度 | crack线段长度 ≥ pad短边 × 比例 | crack_min_length_ratio |
| C++加速 | C++批量crack检测的最小pad数(0=纯Python) | crack_cpp_min_pads |

### 4. 持久化

| 文件 | 内容 | 加载/保存按钮 |
|------|------|-------------|
| params.json | HSV 范围 + 判定阈值 + 引脚类型 | [Cfg]加载 / [Save]保存 |
| pad.json | LabelMe 风格 ROI (pad/toe/rim 框) | [ROI]加载 / [Save]保存 |

> **注意：** pair目录加载时只自动加载 `pad.json`，不加载 `*_OK.json` 等其他标注文件。pad.json通常来自模板图坐标系，保存时自动还原到来源坐标系。

一键加载 pair 目录：[Folder] 按钮可同时加载图片、模板、params.json 和 pad.json。

### pair 目录结构

```
pair1/
├── NG_image.png           # 待检图（可多张）
├── NG_image_OK.png        # 模板图（*_OK.png，可多张）
├── pad.json               # pad/toe/rim 标注
└── params.json            # 参数配置
```

- 待检图：不含 `_OK` 的图片
- 模板图：文件名含 `_OK` 的图片，**多个时取第一个（按文件名排序）**作为算法模板
- 所有图片可在 tuner 中通过导航按钮切换浏览

## 快捷键

| 快捷键 | 功能 |
|--------|------|
| q | 退出 |
| 1 / 2 / 3 | 画 pad / toe / rim 框 |
| e | 编辑模式（选中/移动/缩放/删除） |
| s | 采样模式（点击图片取 HSV） |
| Tab | 切换 HSV 抽色对象 |
| Delete | 删除选中的框 |
| Ctrl+Z | 撤销操作 |
| Ctrl+滚轮 | 缩放图片 |
| 滚轮 | 滚动画布 |

## 界面布局

```
+----------------------------------------------------------+
| [Open] [Folder] [Cfg] [ROI] [Save] [Save]  文件名信息     |
+----------+-------------------------------+----------------+
| 操作模式  |                               | NG 统计        |
| ①②③ e s  |                               |                |
|----------|        主图画布                |----------------|
| HSV 抽色  |    (图片 + ROI框 + Mask叠加)   | HSV 读数       |
| 4对象切换 |                               |                |
| 6条滑条   |                               |----------------|
|----------|                               | 辅助画布       |
| 判定阈值  |                               | (小图切换)     |
| 少锡/多锡  |                               |                |
| 连锡/虚焊  |                               |                |
|----------|                               |                |
| 检测开关  |                               |                |
| [Copy]   |                               |                |
+----------+-------------------------------+----------------+
```

## 参数对应关系

tuner 的 params.json 与算法 config 的参数映射：

| tuner params.json | 算法 config | 说明 |
|-------------------|------------|------|
| mask_h_low 等 | mask_h_low | 阻焊 HSV 范围 |
| silk_h_low 等 | silk_h_low | 丝印 HSV 范围 |
| solder_blue_h_low 等 | solder_blue_h_low | 焊锡 HSV 范围 |
| insufficient_thresh | insufficient_thresh | 少锡阈值 |
| excess_thresh | excess_thresh | 多锡阈值 |
| bridge_min_area | bridge_min_area | 连锡最小面积 |
| cold_solder_rim_ratio | cold_solder_rim_ratio | 虚焊 Rule1 阈值 |
| cold_solder_toe_metal_ratio | cold_solder_toe_metal_ratio | 虚焊 Rule2 阈值 |
| cold_solder_diff_ratio | cold_solder_diff_ratio | 虚焊 Rule3 阈值 |
| crack_min_length_ratio | crack_min_length_ratio | crack线段最小长度比例 |
| crack_cpp_min_pads | crack_cpp_min_pads | C++批量crack检测最小pad数(0=纯Python) |
| pin_type | pin_type | 引脚类型 |
