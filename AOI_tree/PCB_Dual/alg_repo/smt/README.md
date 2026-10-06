# SolderSmtAllAlg - SMT焊锡全缺陷检测算法

## 概述

SolderSmtAllAlg 是基于传统CV的SMT焊锡全缺陷检测算法，一次调用同时检测**多锡/少锡/连锡/虚焊**四类缺陷。

- **algorithm_code:** `SolderSmtAllAlg`
- **version:** 2
- **继承:** `IDetectionAlgorithm`
- **平均耗时:** ~10ms/pair（纯Python, warm状态, 含虚焊crack懒执行优化）
                 ~8-12ms/pair（C++加速模块, pad>3时启用批量crack检测）
                 首次运行（CLI单次调用）约20-30ms，含OpenCV/numpy首次初始化开销
                 生产环境（反复调用）为warm状态，达到上述平均耗时
- **检测召回率:** cold_solder 100%, bridge 100%, excess 100%, insufficient 90.9%

### 重要说明

1. **焊锡蓝色范围必须调参：** `solder_blue_h_low/h_high/s_min/v_min` 是最关键参数，默认值仅适用于测试集。不同批次焊锡、不同光源条件下HSV值差异很大，**必须根据实际图片调整**，否则所有缺陷检测都会受影响。
2. **模板模式固定为 `diff_grow_ROI`：** 算法设计了7种 template_mode，但当前版本硬编码为 `diff_grow_ROI`（pad.json焊盘框 + grow焊锡提取 + 差分融合），其他模式保留供未来扩展。
3. **pad.json必需：** 调用方必须提供 `pad_frames`（来自pad.json），否则算法返回 code=1。
4. **速度与精度可权衡：** `enable_auto_resize` 开启时速度快但少锡灵敏度降低；关闭后精度提升但耗时增加约3倍。

---

## 目录结构

```
Dev_solder_smt_Alg/
├── docs/
│   ├── design.md                          # 算法设计文档
│   ├── params.md                          # 可调参数说明
│   ├── pad.md                             # pad.json标注格式说明
│   └── tuner.md                           # 调参工具文档
├── src/
│   ├── algorithms/
│   │   ├── __init__.py                    # 算法注册（含SolderSmtAllAlg）
│   │   └── solder_smt/
│   │       └── composite_smt_all/
│   │           ├── __init__.py            # 包标记
│   │           ├── _solder_core.py        # 纯算法函数（14个函数）
│   │           └── solder_smt_all_alg.py  # 算法主体（SolderSmtAllAlg类）
│   └── resources/
│       └── config/
│           ├── alg/solder_smt/
│           │   └── solder_smt_all_alg.json  # 默认参数配置（97个参数）
│           └── alg_pool/
│               └── cv_algorithms.json       # 算法池注册
├── tuner/
│   ├── pcba_tuner_real.py                 # 调参工具（导入生产算法，推荐）
│   └── pcba_tuner.py                      # 调参工具（独立版，无需算法包）
├── test_src/
│   └── algorithms/solder_smt/
│       └── solder_smt_all_alg_test.py       # 单元测试（6个用例）
└── contract_reference/                      # 接口契约（只读参考）
    └── core/
        ├── entities/                        # IDetectionAlgorithm, AlgorithmResult等
        ├── utils/                           # merge_config等工具
        └── load_class.py                    # 动态导入工具
```

---

## 快速开始

> **所有命令汇总见 [docs/command.md](docs/command.md)**

### 1. 环境要求

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Pillow (PIL) - 可视化渲染用
- pytest - 运行测试用

### 2. 安装

将 `src/` 目录加入 `PYTHONPATH`，或将整个 `src/` 复制到项目根目录。

### 3. Python API 调用

```python
import cv2, json
from algorithms.solder_smt.composite_smt_all.solder_smt_all_alg import SolderSmtAllAlg

# 加载图片和pad.json
img = cv2.imread("NG_image.png")
tpl = cv2.imread("NG_image_OK.png")  # 可选，有模板时精度更高

# 从pad.json加载焊盘框
with open("pad.json", "r", encoding="utf-8") as f:
    pad_data = json.load(f)
pad_frames = []
for s in pad_data["shapes"]:
    if s["label"] == "pad" and s["shape_type"] == "rectangle":
        (x1, y1), (x2, y2) = s["points"]
        pad_frames.append((int(min(x1,x2)), int(min(y1,y2)),
                           int(abs(x2-x1)), int(abs(y2-y1))))

# 运行检测
alg = SolderSmtAllAlg()
cfg = {
    "pad_frames": pad_frames,
    "pad_json_path": "pad.json",  # 可选，用于加载toe/rim标注
}
result = alg.run(img, cfg, original_template_image=tpl)

# 查看结果
print(f"缺陷数: {result.metadata['num_defects']}")
print(f"耗时: {result.cost_time*1000:.1f}ms")
for p in result.parts:
    print(f"  {p.label}: ({p.x},{p.y}) {p.width}x{p.height} conf={p.confidence:.2f}")

# 可视化图
vis = result.metadata["output_image"]
cv2.imwrite("result.png", vis)
```

### 4. CLI 单图测试

Windows:
```cmd
set PYTHONUTF8=1
set PYTHONPATH=src;contract_reference
python -m algorithms.solder_smt.composite_smt_all.solder_smt_all_alg "NG_image.png" -t "NG_image_OK.png" --pad "pad.json"
# 模板图通常命名为 *_OK.png
```

Ubuntu:
```bash
export PYTHONUTF8=1
export PYTHONPATH=src:contract_reference
python -m algorithms.solder_smt.composite_smt_all.solder_smt_all_alg "NG_image.png" -t "NG_image_OK.png" --pad "pad.json"
```

输出示例：
```
缺陷数=1 模式=diff_grow_ROI 焊盘=12 耗时=9.6ms 启用=['excess', 'insufficient', 'bridge', 'cold_solder']
  cold_solder:(33,103)82x103 conf=1.00
```

### 5. CLI 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `image` | 输入图片路径（必填） | - |
| `-t, --template` | 模板图片路径 | 无（无模板时跳过差分） |
| `--pad` | pad.json 路径 | 自动查找图片同目录 |
| `--params` | params.json 路径（加载额外参数） | 无 |
| `--no-excess` | 关闭多锡检测 | 开启 |
| `--no-insufficient` | 关闭少锡检测 | 开启 |
| `--no-bridge` | 关闭连锡检测 | 开启 |
| `--no-cold` | 关闭虚焊检测 | 开启 |
| `--no-vis` | 关闭可视化图（提速~3ms） | 开启 |

示例 - 只检测虚焊，加载params.json：
```bash
python -m algorithms.solder_smt.composite_smt_all.solder_smt_all_alg \
    "NG_image.png" -t "NG_image_OK.png" --pad "pad.json" \
    --params "params.json" --no-excess --no-insufficient --no-bridge --no-vis
```

### 6. 批量测试

用 Python 脚本批量检测目录下所有图片：

```python
import os, cv2, json, time
from algorithms.solder_smt.composite_smt_all.solder_smt_all_alg import SolderSmtAllAlg

alg = SolderSmtAllAlg()
pair_dir = "test_images/cold_solder_pair/pair1"

# 加载 pad.json + params.json + 图片
img = cv2.imread(os.path.join(pair_dir, "NG_image.png"))
tpl = cv2.imread(os.path.join(pair_dir, "NG_image_OK.png"))
with open(os.path.join(pair_dir, "pad.json"), "r") as f:
    pad_data = json.load(f)
pad_frames = [(int(min(s["points"][0][0], s["points"][1][0])),
               int(min(s["points"][0][1], s["points"][1][1])),
               int(abs(s["points"][1][0] - s["points"][0][0])),
               int(abs(s["points"][1][1] - s["points"][0][1])))
              for s in pad_data["shapes"] if s.get("label") == "pad"]

cfg = {"pad_frames": pad_frames, "pad_json_path": os.path.join(pair_dir, "pad.json")}
# 加载 params.json (如果存在)
params_path = os.path.join(pair_dir, "params.json")
if os.path.exists(params_path):
    with open(params_path, "r") as f:
        cfg.update(json.load(f))

t0 = time.perf_counter()
result = alg.run(img, cfg, original_template_image=tpl)
t1 = time.perf_counter()
print(f"缺陷数={result.metadata['num_defects']} 耗时={(t1-t0)*1000:.1f}ms")
for p in result.parts:
    print(f"  {p.label}: ({p.x},{p.y}) {p.width}x{p.height}")
```

### 7. 运行单元测试

Windows:
```cmd
set PYTHONUTF8=1
set PYTHONPATH=src;contract_reference
python -m pytest test_src/algorithms/solder_smt/solder_smt_all_alg_test.py -v
```

Ubuntu:
```bash
export PYTHONUTF8=1
export PYTHONPATH=src:contract_reference
python -m pytest test_src/algorithms/solder_smt/solder_smt_all_alg_test.py -v
```

---

## 调参工具 (tuner)

tuner 是基于 tkinter 的可视化参数调试工具，支持 ROI 绘制、HSV 抽色、实时缺陷判定和参数持久化。详见 [docs/tuner.md](docs/tuner.md)。

### 启动

```bash
# Windows
set PYTHONPATH=src;contract_reference
python tuner\pcba_tuner_real.py

# Ubuntu
PYTHONPATH=src:contract_reference python tuner/pcba_tuner_real.py
```

### Ubuntu 字体说明

Ubuntu 上 tkinter 可能无法显示中文（显示为方框），需安装中文字体：

```bash
sudo apt-get install xfonts-wqy fonts-noto-cjk
xset fp rehash  # 刷新X字体缓存
```

tuner 会自动检测系统可用的中文字体，优先级：`wenquanyi bitmap song` > `Noto Sans CJK SC` > `song ti`。

> **注意：** 部分conda环境的tkinter编译时未启用fontconfig支持，无法识别TTF字体。若安装字体后仍显示异常，可尝试用系统Python运行：`/usr/bin/python3 tuner/pcba_tuner_real.py`

## C++加速模块 (可选)

将耗时从 ~30ms 降至 ~20-30ms (Ubuntu)。加速函数：`compute_diff_fast`（差分计算）、`detect_crack_hough_fast`（单pad crack检测）、`detect_crack_batch_fast`（批量crack检测, OpenMP并行）。

### Ubuntu 构建

```bash
# 1. 安装依赖
sudo apt-get install g++ python3-dev libopencv-dev
pip install pybind11>=2.10.0

# 2. 编译 (从Dev_solder_smt_Alg根目录运行, .so会生成在根目录)
python csrc/setup.py build_ext --inplace

# 3. 验证
python -c "from _fast_core import compute_diff_fast; print('OK')"
```

### Windows 构建 (可选)

```cmd
pip install pybind11>=2.10.0
python csrc\setup.py build_ext --inplace
```

### 不构建：纯Python回退
如果 `_fast_core` 模块不存在或编译失败，所有函数自动回退到纯Python实现，功能完全一致，只是稍慢。

### C++加速阈值

crack检测的C++批量加速对小数量pad有pybind11开销，通过 `crack_cpp_min_pads` 参数控制：

| 值 | 行为 | 适用场景 |
|---|------|---------|
| 0 | 始终用Python | pad少(≤3)的小图 |
| 4(默认) | pad≥4时用C++ | 平衡选择 |
| 1 | 始终用C++ | pad多的大图 |

可在params.json中设置，或通过tuner输入框调整。

---

---

## 输入要求

### 必需输入

| 输入 | 说明 |
|------|------|
| `image` | 待检图（BGR格式 np.ndarray） |
| `config["pad_frames"]` | 焊盘框列表 `[(x,y,w,h), ...]`，来自pad.json |

### 可选输入

| 输入 | 说明 |
|------|------|
| `original_template_image` | 模板图（OK图），有模板时可做差分对比，精度更高 |
| `config["pad_json_path"]` | pad.json文件路径，用于自动加载toe/rim标注 |
| `roi_bbox` | ROI区域，裁剪输入图后再检测 |

### pad.json格式

详见 [docs/pad.md](docs/pad.md)。核心要求：
- LabelMe格式JSON
- 至少包含 `label="pad"` 的矩形标注
- 可选 `toe`/`rim` 标注（虚焊Rule1用）

---

## 输出说明

### AlgorithmResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | int | 0=成功, 1=失败 |
| `parts` | List[DetectPartBox] | 缺陷框列表 |
| `metadata["num_defects"]` | int | 缺陷总数 |
| `metadata["output_image"]` | np.ndarray | 可视化图（BGR） |
| `metadata["pad_count"]` | int | 焊盘数 |
| `metadata["solder_area"]` | int | 焊锡面积（像素） |

### 缺陷框标签

| label | 颜色 | 说明 |
|-------|------|------|
| `excess` | 蓝色 | 多锡 |
| `insufficient` | 黄色 | 少锡 |
| `bridge` | 红色 | 连锡 |
| `cold_solder` | 品红 | 虚焊 |

---

## 参数调优

> **详细调参指南详见 [docs/design.md](docs/design.md) 第7节"调参须知"。**

### 必须调整的参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `solder_blue_h_low/high` | 85/115 | 焊锡蓝色HSV H范围，**必须根据实际图片调整** |
| `solder_blue_s_min` | 60 | 焊锡蓝色HSV S下限 |
| `solder_blue_v_min` | 150 | 焊锡蓝色HSV V下限 |

### 常用调优参数

| 参数 | 默认值 | 调优场景 |
|------|--------|---------|
| `insufficient_thresh` | 0.08 | 少锡漏检时降低，误检时升高 |
| `excess_thresh` | 0.08 | 多锡漏检时降低，误检时升高 |
| `bridge_min_area` | 20 | 连锡漏检时降低，误检时升高 |
| `cold_solder_dark_ratio` | 0.30 | 虚焊漏检时降低，误检时升高 |
| `enable_auto_resize` | true | 少锡漏检时关闭（速度变慢但精度更高） |
| `auto_resize_max_size` | 400 | 增大可提高精度但速度变慢 |

### 速度 vs 精度

| 参数 | 速度优先 | 精度优先 |
|------|---------|---------|
| `enable_auto_resize` | `true` | `false` |
| `auto_resize_max_size` | 300 | 600+ |
| `skip_rb_score` | `true` | `false` |
| `solder_grow_max_iters` | 0 | 5~20 |

### 调参流程

1. 用 tuner 工具加载图片，绘制 pad/toe/rim 框，调节 HSV 范围和阈值
2. 用默认参数跑一批图片，查看可视化结果图
3. 焊锡mask不准 -> 调整 `solder_blue_*` 参数
4. 干扰排除不准 -> 调整 `mask_*`/`silk_*`/`comp_*` 参数
5. 缺陷误检/漏检 -> 调整对应 `*_thresh` 阈值
6. 可设 `debug_save_solder_mask=true` 保存中间图辅助调参

调参工具详见 [docs/tuner.md](docs/tuner.md)。

### 完整参数文档

详见 [docs/params.md](docs/params.md)（11个分类，97个参数）。

---

## 性能数据

### 测试集结果（41个pair）

| 缺陷类型 | 召回率 | 误检 |
|---------|--------|------|
| cold_solder | 100% (6/6) | 0 |
| bridge | 100% (20/20) | 0 |
| excess | 100% (4/4) | 0 |
| insufficient | 90.9% (10/11) | 0 |

### 耗时

| 平台 | 模式 | warm平均 | 首次运行 | 说明 |
|------|------|---------|---------|------|
| Windows | 纯Python | ~10ms | ~20ms | Python 3.11, OpenCV 4.x |
| Ubuntu | 纯Python | ~15ms | ~30ms | OpenCV编译优化差异 |
| Ubuntu | C++加速 | ~10ms | ~30ms | pad>3时启用批量crack检测, pad<=3用Python |

> warm = 非首次调用（生产环境反复调用时的性能）
> 首次运行 = CLI单次调用（含OpenCV/numpy初始化）

### 平台差异说明

Ubuntu上纯Python模式耗时约为Windows的2-3倍，主要原因：

1. **OpenCV编译差异** - Windows的`opencv-python` pip包预编译了SIMD/AVX指令优化；Ubuntu上从源码编译的OpenCV可能未开启同等优化
2. **CPU调度策略** - Linux的CPU governor动态调频更激进，导致同一图片多次运行耗时波动大（30ms~50ms），Windows上波动小（15ms~20ms）
3. **无功能性差异** - 两个平台的检测结果完全一致，仅速度不同

建议Ubuntu环境编译C++加速模块（见上方"C++加速模块"章节）以获得与Windows相当的耗时。

---

## 算法设计

详见 [docs/design.md](docs/design.md)。

### 处理流程

1. **输入预处理** - ROI裁剪 + auto_resize（可选）
2. **干扰排除** - HSV分割阻焊层/丝印/元件
3. **焊盘定位** - pad.json + 模板ECC对齐
4. **焊锡提取** - HSV蓝色种子 + 形态学
5. **缺陷判定** - 少锡/多锡/连锡/虚焊并行检测
6. **输出** - 坐标还原 + 可视化图

---

## 接口规范

- 继承 `IDetectionAlgorithm`
- `algorithm_code` 与配置JSON的 `algorithmCode` 一致
- `run()` 为唯一入口，签名固定
- 不抛异常，错误返回 `code=1` 的 `AlgorithmResult`
- 使用 `merge_config` 合并默认与外部参数
- 可视化图放 `metadata["output_image"]`
- 所有坐标相对输入图像素坐标
