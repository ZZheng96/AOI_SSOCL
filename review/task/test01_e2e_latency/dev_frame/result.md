# 流/视频帧内存直检 A/B（V§77 / SYNC-25）

脚本：`bench_frame_ab.py`，原始数据：`ab.json`，日志：`run.log`。
隔离环境：TEMP 下 storage/engine/SQLite，engine_fast.yaml，使用 stable 归档中的 14 张正常图 prepare gold_finger。
帧源共 8 帧：TEST-01 去重测试图 6 张，用 cv2 读成 BGR ndarray；另加 2 张合成帧，分别为加高斯噪声 σ=4 和亮度 +12。
口径：两条路径都使用 persist=True、with_heatmap=False、target_type="stream"，与 RTSP 循环一致。共 3 轮，每轮按帧交替执行 old/new。

- old：`cv2.imwrite(tmp.jpg)` + `detect_image(tmp)` + `os.remove`，即原 `_rtsp_loop` 的做法。
- new：`detect_ndarray(frame)`。

## 耗时（单帧墙钟，n=24）

| 路径 | wall P50 | wall P95 | wall mean | infer P50 | infer P95 |
|---|---|---|---|---|---|
| old | 135.8 | 166.6 | 141.4 | 104.7 | 125.5 |
| new | 65.4 | 123.3 | 78.3 | 52.6 | 68.3 |

单位均为 ms。old 的 infer 包含 load_image（JPEG 解码），new 的 infer 只含内存推理。
P50 下降约 70ms（-52%），P95 下降约 43ms。new 的 P95 主要来自 2 帧异常帧（det881、synth_noise），这两帧固定在 ~115ms，属于帧内容相关的推理分支。第一次运行（与全量 smoke 并发）的结果同向：P50 134.6→67.0，P95 156.0→119.6。

## 精度对照

old 与 new 逐帧比较，8 帧中有 1 帧 decision 不同，score 绝对差最大 1.33：

| 帧 | old（JPEG 往返） | new（原始像素） |
|---|---|---|
| det878_bench_00 | anomaly 1.1003 | normal 0.8177 |
| det27_bench_07 | normal 0.6159 | normal 0.8203 |
| det900_bench_02 | normal -1.0 | normal 0.3307 |
| det881_bench_01 | anomaly 1.2513 | anomaly 0.9219 |
| det902_bench_04 | normal 0.2721 | normal 0.5885 |
| det911_bench_05 | normal 0.1615 | normal 0.1615 |
| synth_noise | anomaly 1.0872 | anomaly 1.0924 |
| synth_bright | normal 0.1602 | normal 0.1615 |

归因（`attribution` 段，6 张真实图，`route_equal_all=True`）：

- 输入同为无损 PNG 像素时，`detect_image(png)` 与 `detect_ndarray` 的 decision/score 逐帧完全相等。
- 输入同为 JPEG 像素时，`detect_image(jpg)` 与 `detect_ndarray(jpg 解码帧)` 也逐帧完全相等。

结论：两种入口的打分没有差异，全部差异都来自旧路径额外的 JPEG 有损重编码（cv2 默认质量 95）。新路径的结果等于直接检测原图，相当于去掉了一次人为的失真，因此不是退化。det878 在旧路径下判为 anomaly，是 JPEG 伪影导致的。这一帧在无损路径下本来就是 normal。

注意：新路径上线后，RTSP 帧的分数分布会与旧路径下积累的历史记录（基于 JPEG）有系统偏差。阈值/告警统计如果参考过旧流数据，需要知晓这一点。
