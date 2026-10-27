# round3_speed — HARD-01 单图纯推理 P95

环境：GTX 1660 SUPER，开发版 AOI_tree\AOI_Core，生产快照只读加载。样本：gold_finger v2 n=26（归档随机，图尺寸混杂，最大 4096×3000、5120²），smt_solder v4 n=18。指标：引擎 `latency_ms`（纯推理，不含 decode/render）。

## 结论

| 品类 | 基线 P95 | 默认（无损）P95 | GPU blob 开关 P95 |
|---|---|---|---|
| gold_finger | 397.7 | 316.8 | 181.8 |
| smt_solder | 69.2 | 63.6 | 不适用（无 blob） |

- 默认配置下的无损优化：gold_finger P95 降低 20%，仍超 200ms 红线。剩余瓶颈是大图 blob DoG（CPU 无损下限约 300ms，5120² 约 616ms）和 trad 特征（约 180–260ms）。
- 打开 GPU blob 开关（有损、默认关）后 gold_finger P95 为 181.8，达标。判定 26/26 一致，校准分 Δ=0，raw Δ≤7.45e-8。但历史 U22 记录过 GPU 手工核造成 AUROC 退化（0.7075→0.6112），所以开关保持默认关闭，等主 agent 做全量 AUROC 验证后再决定。
- 428.6 与基线 397.7 的差别来自抽样，本次基线另行抽取 26 张。

## 口径差：TEST-01 204.1ms 与 428ms

- 428.6 / 397.7：使用生产 gold_finger 快照（legacy trad+blob），归档原图尺寸混杂到 4096×3000 和 5120²，取引擎 dt_ms，只算推理。
- 204.1：TEST-01 在 TEMP 下新 prepare 的模型上测 40 张 2500² 图，是 e2e（decode+infer+render），其中 infer P95 约 87ms。
- 两个数字的模型、图尺寸和计时范围都不同，不可直接比较。blob 和 trad 的耗时随像素数线性增长，4096×3000 约为 2500² 的 2 倍，5120² 约为 4.2 倍。

## 分阶段耗时（gold_finger，P95 ms；run=执行，wait=线程池排队）

| 阶段 | base | opt1（默认） | opt2（GPU 开关） |
|---|---|---|---|
| latency_ms | 397.7 | 316.8 | 181.8 |
| slot:blob run | 397.0 | 316.3 | 144.5（P50 31.5） |
| slot:trad run | 232.4 | 263.4 | 181.3 |
| img_features run / wait | 124.2 / 216.9 | 177.9 / 3.0 | 109.7 / 2.5 |
| align_offset run / wait | 34.6 / 297.7 | 69.1 / 6.0 | 54.1 / 5.5 |
| forward | 0（legacy 无 DINO） | 0 | 0 |

smt_solder：forward P95 32.4，各槽位 ≤4ms，latency 69.2→63.6。

## 逐项优化

| 项 | 结果 | 精度 | 状态 |
|---|---|---|---|
| CPU 线程池 2→自适应 `max(2,min(4,cpu//2))` | 消除 img_features/align 约 200–300ms 排队 | 逐位一致（任务独立，trad RNG 仍由锁保护） | 默认 |
| blob σ=2 去重，3 个 blur 并行 | blob 约 −15% | 逐位一致 | 默认 |
| blob cv2.absdiff/cv2.max 替代 numpy | 少一次临时数组 | 逐位一致 | 默认 |
| blob 精确 top-k 预筛（步长 64 子集第 k 大作阈值） | partition 成本降到约 1/50 | 数学上精确，测试含对抗样例 | 默认 |
| gray 原地 /255 | 少一次分配 | 逐位一致 | 默认 |
| blob 条带 halo 并行 | 5120² 482→404ms，收益小（cv2 已多线程） | 逐位一致 | 未采用 |
| GLCM bincount 替代 graycomatrix | 更慢（graycomatrix 仅约 2.6ms） | — | 未采用 |
| graycoprops 归一化去重 | 理论约 9ms | 需另行逐位验证 | 未做 |
| GPU blob DoG（cv2 同核、reflect101、torch.topk） | 4096×3000 208→48ms，2500² 101→25ms；<1MP 自动回落 CPU；显存按 `mem_get_info` 自适应，峰值约 10·H·W·4B | 42 图 CPU/GPU 相对差 max 4.0e-7；bench 判定一致 | 开关，默认关 |
| blob 降采样 max_side | 已有开关 | U22：检出 6/20→3/20 | 保持默认关 |
| DINO fp16/channels_last/cudnn.benchmark | smt forward 仅约 32ms，已达标；gold_finger 不走 DINO | — | 未做（对 HARD-01 无影响） |
| heatmap 惰性 | blob 默认 with_heatmap=False，已惰性 | — | 已有 |

## 精度对比（compare.py，逐图）

- gold_finger base↔opt1：decision 不一致 0/26，Δscore/Δraw/Δslot 均为 0，n_boxes 差异 0。
- smt_solder base↔opt1_smt：0/18，全 0。
- gold_finger base↔opt2_gpu：decision 不一致 0/26，Δscore 0，max|Δraw| 7.45e-8，n_boxes 差异 0。

## 开关用法

- `AOI_BLOB_GPU=1` 环境变量，或在 blob 槽位 cfg 中设 `gpu: true`；可选 `gpu_min_px`（默认 1<<20）。环境变量优先，`0`/`false`/空值表示关闭。
- 配置文件未改动，避免改变 config 指纹。默认值由代码中的 `cfg.get("gpu", False)` 决定。

## 改动文件

- `AOI_tree\AOI_Core\algo\slots\blob.py`：无损改写和 GPU 开关
- `AOI_tree\AOI_Core\algo\eval\offline.py`：`_cpu_exec` worker 自适应（1 行）
- `AOI_tree\AOI_Core\tests\smoke\test_round3_speed.py`（新增，6 项）：与旧实现逐位一致、top-k 精确、GPU 默认关、GPU 相对差 <1e-5、小图回落、worker 数在 [2,4]
- 测试：新增 6 项，加上 shared_backbone/final_perf/async_persist/g1_latency_equivalence/snapshot_layout_compat/round2_release/final_sim_slots/cde_report/feature_registry，共 66 项全部通过
- 本目录工具：`bench_profile.py`、`compare.py`、`show.py`、`bench_*.json`；`probe_blob3.py`（条带并行验证）、`probe_blob_gpu.py`（GPU 对照）留作证据

## 建议状态

HARD-01：部分完成。默认无损路径 P95 为 316.8，仍未达标。有损 GPU 开关路径 P95 为 181.8，达标且小样本判定一致，但需要在 gold_finger 全量集上复核 blob/fused AUROC 和检出数（对照 U22）。复核不退化后再把开关设为默认开。若要不依赖 GPU，下一步应在 trad（约 180ms）和大图 blob CPU 实现上继续优化。
