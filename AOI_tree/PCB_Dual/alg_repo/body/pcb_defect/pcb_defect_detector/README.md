# pcb_defect_detector

PCB 元件 **7 项缺陷检测算法**的对外封装包。

本包在算法引擎之上提供一层极简门面，**只暴露一个检测器类 `PCBDefectDetector` 与两个结果数据类**，调用方无需了解引擎内部机制。

## 7 项算法

| code | 中文名 | 原理 |
|------|--------|------|
| `component_wrong_part` | 错件 | 尺寸+HSV直方图+轮廓 三特征2/3投票 |
| `component_missing` | 缺件 | 本体像素比+差分+边缘+暗色通道 联合判定 |
| `component_shift` | 移位 | ROI位姿估计 dx/dy/theta 超公差 |
| `component_tombstone` | 立碑 | 宽高比+焊盘不对称+ROI模式联合 |
| `component_flipped` | 翻件 | NCC正面+HSV直方图 双特征2/2投票 |
| `component_reverse_polarity` | 极反 | 丝印+色带+灯芯 三路联合(无OCR) |
| `component_damage` | 破损 | SSIM差异+轮廓比对，先精对齐再判破损 |

## 安装

```bash
pip install -r requirements.txt
# 或直接把本目录加入 sys.path / 以源码方式使用
```

依赖：`opencv-python`、`numpy`、`scikit-image`（可选，缺失时破损检测自动降级）。

## 快速上手

```python
from pcb_defect_detector import PCBDefectDetector

det = PCBDefectDetector()

# 列出可用算法
print(det.list_algorithms())
# {'component_wrong_part': '错件', 'component_missing': '缺件', ...}

# 单算法检测（图片可为路径或 numpy 数组）
r = det.detect("component_missing", "template.png", "test.png")
print(r.status, r.defect_count, r.cost_time)
for d in r.defects:
    print(d.label, d.x, d.y, d.width, d.height, d.confidence)

# 保存可视化结果图
r.save_image("result.png")

# 一次跑全部 7 项
results = det.detect_all("template.png", "test.png")
for code, rr in results.items():
    print(code, rr.status, rr.defect_count)

# 自定义阈值覆盖（可选）
r = det.detect("component_shift", tpl, test, config={"proc_max_side": 0})
```

## 对外接口（全部）

| 对象 | 说明 |
|------|------|
| `PCBDefectDetector()` | 检测器门面 |
| `PCBDefectDetector.list_algorithms()` | 静态方法，返回 `{code: 中文名}` |
| `det.available_algorithms()` | 实际成功加载的 code 列表 |
| `det.detect(algorithm, template, test, config=None)` | 单算法检测，返回 `DetectionResult` |
| `det.detect_all(template, test, config=None, algorithms=None)` | 多算法检测，返回 `{code: DetectionResult}` |
| `det.detect_batch(algorithm, pairs, config=None)` | 批量检测多对图 |
| `DetectionResult` | `algorithm / status / defect_count / defects / cost_time / output_image / error / extra`，含 `is_ok / is_ng / has_error / to_dict() / save_image(path)` |
| `Defect` | `label / x / y / width / height / confidence / description / severity / extra`，含 `to_dict()` |
| `list_algorithms()` | 模块级快捷函数 |

### 结果 `status` 取值

- `OK`：未检出缺陷
- `NG`：检出缺陷
- `ERROR`：调用过程出错（见 `error` 字段）

## 设计要点 / 契约

1. **绝不向外抛异常**：任何内部异常（含未知算法、空图、读盘失败、引擎报错）都被吸收为 `status="ERROR"` 的 `DetectionResult`，调用方只需检查 `status`。
2. **图片输入双形态**：`str` / `Path` 路径或 `np.ndarray`（BGR 或灰度），由门面自动适配；中文路径安全。
3. **不落盘**：检测过程不会把传入的图片或中间结果写到磁盘，避免数据泄漏与磁盘占用。
4. **坐标契约**：缺陷框坐标相对输入测试图，左上角 `(x, y)` + 宽高，单位像素。
5. **内部引擎私有**：`pcb_defect_detector/_engine/` 为私有实现，不应从外部导入；统一使用顶层包接口。

## 目录结构

```
pcb_defect_detector/
├── README.md
├── requirements.txt
├── detect_cli.py                 # 命令行批跑工具
├── pcb_defect_detector/          # 公共包
│   ├── __init__.py               # 暴露 PCBDefectDetector / DetectionResult / Defect / list_algorithms
│   ├── detector.py               # 门面实现
│   ├── results.py                # DetectionResult / Defect 数据类
│   ├── image_io.py               # Unicode 安全 imread/imwrite
│   └── _engine/                  # 私有算法引擎
│       ├── core/  algorithms/  utils/  resources/config/...
└── tests/
    └── test_smoke.py             # 冒烟测试
```

## 验证

```bash
set PYTHONUTF8=1
python tests\test_smoke.py        # 冒烟测试
python detect_cli.py --help       # 命令行批跑工具
```
