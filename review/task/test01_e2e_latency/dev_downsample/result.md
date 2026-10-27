# AC-066 下采样解码试验结果（2026-10-13，不采用）

详见验证记录 §69。本目录文件：
- `bench_decode_ds.py` / `decode_bench.json`：解码微基准，reps=4。json 中的 "jpg2500" 组实为 4096×3000 jpg，属误标。
- `bench_ds_ab.py` / `ds_ab.json` / `ds_ab.log` / `ds_ab.err`：TEMP 隔离下的精度 + e2e A/B（运行 100s，TEMP 已清理）。
- `smoke_stdout.log` / `smoke_stderr.log`：全量 smoke，364 tests / 87.321s / OK。

## 要点
- 引擎 fast profile 内部已经是低分辨率计算（DINO 448²、hierarchical work_side=512、对位 256²、clf ≤1024），下采样只能省 decode 和 render 两段。
- PNG 2500²（TEST-01 口径）：REDUCED/area 解码无收益，e2e 反而更慢（P95 221.6→235.1）。
- JPG 4096×3000：reduced4 的 e2e P95 从 156.9 降到 115.8，但 34 张精度集上框 IoU 约 0.2，判定翻转 4～6 张，good 误报增加；下采样后重新 prepare 会丢检。
- 结论：不采用，不改生产路径。200ms 目标需另寻办法（相机 JPEG/内存帧、decode 与 infer 流水并行、render/persist 异步化）。
