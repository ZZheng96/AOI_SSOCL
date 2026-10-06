# pad.json 标注格式说明

> pad.json 是 LabelMe 格式的 JSON 文件，标注焊盘(pad)、toe、rim 等区域。
> 算法通过 `pad_frames` 参数（或 `pad_json_path`）读取焊盘框，用于缺陷检测。

---

## 1. 文件结构

```json
{
  "version": "6.3.1",
  "flags": {},
  "shapes": [ ... ],
  "source": "xxx_OK.json",
  "imagePath": "",
  "imageData": null,
  "imageHeight": 0,
  "imageWidth": 0
}
```

| 字段 | 说明 |
|------|------|
| `version` | LabelMe 版本号 |
| `shapes` | 标注形状列表（核心字段） |
| `source` | 来源文件名（含 `_OK` 表示模板标注，需坐标映射） |
| `imagePath` | 图片路径（可空） |
| `imageHeight` / `imageWidth` | 图片尺寸（可空） |

---

## 2. shape 结构

每个 shape 是一个矩形标注：

```json
{
  "label": "pad",
  "points": [
    [12.66, 63.24],
    [207.71, 219.67]
  ],
  "group_id": null,
  "description": "",
  "shape_type": "rectangle",
  "flags": {},
  "mask": null
}
```

| 字段 | 说明 |
|------|------|
| `label` | 标注类型（见下表） |
| `points` | 两个角点坐标 `[[x1, y1], [x2, y2]]`，浮点数 |
| `shape_type` | 固定为 `"rectangle"` |
| `group_id` | 分组ID（可空） |

---

## 3. 支持的 label 类型

| label | 必需 | 说明 | 用途 |
|-------|------|------|------|
| **`pad`** | **必需** | 焊盘矩形框 | 焊锡提取 + 所有缺陷判定的基础 |
| `toe` | 可选 | toe（趾部）区域 | 虚焊检测 Rule1：toe 金属覆盖率 |
| `rim` | 可选 | rim（边缘）区域 | 虚焊检测 Rule1：边缘焊锡覆盖率 |
| `silk` | 可选 | 丝印区域 | 干扰排除（可不用，算法自动检测） |
| `solder_mask` | 可选 | 阻焊层区域 | 干扰排除（可不用，算法自动检测） |
| `component` | 可选 | 元件区域 | 干扰排除（可不用，算法自动检测） |

> **注意:** 只有 `pad` 是必需的。`toe` 和 `rim` 仅在 `pin_type="gull-wing"` 且需要虚焊 Rule1 检测时使用。其他 label 由算法自动检测，标注仅为辅助。

---

## 4. 坐标系

- 坐标原点：图片左上角
- x 轴向右，y 轴向下
- `points` 中的坐标为像素坐标（浮点数），算法内部取整
- 如果 `source` 含 `_OK`（模板标注），坐标会通过模板对齐映射到待检图

---

## 5. 调用方式

### 方式1：通过 `pad_frames` 直接传入

调用方从 pad.json 中提取 `label="pad"` 的矩形，转为 `[x, y, w, h]` 列表：

```python
import json

def load_pad_frames(pad_json_path):
    with open(pad_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pads = []
    for s in data.get("shapes", []):
        if s["label"] == "pad" and s["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = s["points"]
            pads.append((
                int(min(x1, x2)),
                int(min(y1, y2)),
                int(abs(x2 - x1)),
                int(abs(y2 - y1))
            ))
    return pads

cfg = {
    "pad_frames": load_pad_frames("path/to/pad.json"),
    "pad_json_path": "path/to/pad.json",  # 可选，用于加载 toe/rim
}
result = alg.run(image, cfg, original_template_image=template)
```

### 方式2：仅通过 `pad_json_path` 传入

算法内部自动加载 pad.json，提取 pad/toe/rim：

```python
cfg = {
    "pad_json_path": "path/to/pad.json",
}
result = alg.run(image, cfg, original_template_image=template)
```

> **推荐方式1**，同时传入 `pad_frames` 和 `pad_json_path`。

---

## 6. 完整示例

以下是一个包含 2 个 pad、2 个 toe、2 个 rim 的最小 pad.json：

```json
{
  "version": "6.3.1",
  "flags": {},
  "shapes": [
    {
      "label": "pad",
      "points": [[12.66, 63.24], [207.71, 219.67]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    },
    {
      "label": "pad",
      "points": [[582.46, 60.76], [754.74, 216.70]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    },
    {
      "label": "toe",
      "points": [[121.57, 118.68], [202.26, 172.15]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    },
    {
      "label": "toe",
      "points": [[590.38, 116.21], [679.49, 172.64]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    },
    {
      "label": "rim",
      "points": [[96.82, 97.40], [196.32, 195.91]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    },
    {
      "label": "rim",
      "points": [[699.29, 96.41], [602.26, 196.41]],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {},
      "mask": null
    }
  ],
  "source": "template_OK.json",
  "imagePath": "",
  "imageData": null,
  "imageHeight": 286,
  "imageWidth": 792
}
```

---

## 7. pad_frames 格式

`pad_frames` 是一个列表，每个元素是 `[x, y, w, h]` 四元组：

```python
pad_frames = [
    (12, 63, 195, 156),    # pad 1: x=12, y=63, width=195, height=156
    (582, 60, 172, 156),   # pad 2: x=582, y=60, width=172, height=156
]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| x | int | 矩形左上角 x 坐标 |
| y | int | 矩形左上角 y 坐标 |
| w | int | 矩形宽度 |
| h | int | 矩形高度 |

---

## 8. toe/rim 匹配规则

toe 和 rim 通过**中心点**匹配到最近的 pad：

1. 计算 toe/rim 矩形的中心点
2. 检查中心点是否落在某个 pad 矩形内
3. 如果落在多个 pad 内，选择重叠面积最大的 pad
4. 匹配结果为 `{pad_index: (x, y, w, h)}` 字典

> 如果 toe/rim 没有匹配到任何 pad，则该标注被忽略。

---

## 9. 模板标注 vs 待检图标注

| source 含 `_OK` | 说明 | 处理方式 |
|-----------------|------|---------|
| 是 | 标注来自模板图 | 通过模板对齐映射到待检图坐标 |
| 否 | 标注来自待检图 | 直接使用原始坐标 |

算法通过 `source` 字段是否包含 `_OK` 或 `_template` 来判断标注来源。
