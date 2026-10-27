# OPT-DECODE 前后对比（2026-10-13，开发版，生产未验收）

改动：`AOI_Core/backend/pipeline/service.py::_decode_rgb`。PNG（8 位 RGB/L、单帧、无 tRNS、单 IDAT 流 rawmode 为 RGB/L）也走 cv2，用 `IMREAD_COLOR_RGB | IMREAD_IGNORE_ORIENTATION` 直接得到 RGB，省掉 cvtColor；尺寸或通道不符时回退 PIL。其余格式和模式（RGBA/P/16 位/1 位/APNG/CMYK/TIFF 等）仍走原来的 PIL `convert("RGB")`。

## 解码子步（bench_decode.py，6 张 TEST-01 去重 2500² PNG × 5 次，单位 ms）

| 子步 | before median/p95 | after median/p95 |
|---|---|---|
| PIL 全解码（旧路径） | 109.0 / 118.8 | 118.3 / 134.5 |
| cv2 imdecode BGR | 92.1 / 98.6 | 98.1 / 112.4 |
| cvtColor BGR→RGB | 5.9 / 8.4 | 6.5 / 8.4 |
| cv2 imdecode 直出 RGB | 89.0 / 94.4 | 94.5 / 101.9 |
| `_decode_rgb` 当前实现 | 109.7 / 125.3（PIL） | 99.1 / 109.0（cv2 RGB） |

after 那一轮机器负载更高（PIL 基线本身也慢了 9ms）。按同轮相对值看，`_decode_rgb` 比 PIL 快约 19ms，median 约快 16%。像素 maxdiff 全部为 0。

## 端到端同进程 A/B（bench_e2e_ab.py，TEMP 隔离、prepare 14 张正常图，6 张测试图 × 4 轮交替，n=24）

| 指标（ms） | old p50/p95 | new p50/p95 |
|---|---|---|
| decode | 103.9 / 116.5 | 94.8 / 105.3 |
| infer | 58.8 / 69.4 | 58.6 / 68.6 |
| render | 0.0 / 32.0 | 0.0 / 29.7 |
| e2e | 162.9 / **205.7** | 154.7 / **190.8** |
| wall | 168.7 / 229.9 | 160.5 / 215.6 |

精度：6 张图的 pixel、score、decision、slots、raw、boxes、heatmap 差异全部为 0。旧实现自重复的差异也是 0。24 组 A/B 的决策和分数全部一致（`AB_IDENTICAL True`）。

## 结论与边界

- 本 A/B 环境下，热态 e2e P95 从 205.7 降到 190.8ms（≤200ms）。decode 段 P95 降低约 11ms，median 降低约 9ms。
- 这个 A/B 用 TEMP 新 prepare 的 14 图模型，不是 TEST-01 的稳定版环境和快照。TEST-01 原值是 P95 direct 295.8 / http 274.7ms，那套环境推理段更重。按 decode 省 10～20ms 推算，TEST-01 口径下仍会超过 200ms，需要验收对话在稳定版复测。
- 剩余瓶颈是 PNG 的 zlib inflate（约 90ms/张，2500²×3）。下列做法会改变像素，或需要新依赖，所以没有采用：IMREAD_REDUCED（引擎全分辨率消费）、turbojpeg/pyspng（未安装）、和上一阶段重叠（没有前置阶段）。
- 进一步降低的方向是产线格式：JPEG/BMP，或原始帧经 `detect_frame` 直接送入，跳过解码。

产物：`bench_decode.py`、`decode_breakdown_before.json`、`decode_breakdown_after.json`、`bench_e2e_ab.py`、`e2e_ab.json`、`e2e_ab.log`、`smoke_full.log`。
