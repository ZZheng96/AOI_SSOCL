# round5 L3 配对模板匹配（gold_finger）

## 改动
- `algo/slots/tpl.py`：新增 `pair_mode`（默认 false）。`score_image(..., pair_template=)` 只与这一张配对模板做相位相关配准、逐位置特征 L2 差分，再取 top-k 均值，不做全局 min，也不做 self_eps 排除。配对模板特征按路径缓存。新增 `resolve_pair_template(path, tpl_dirs)`，按 `X → X_repaired.*` 查找；输入本身是 `*_repaired` 时返回 None。头部红线改为：L3 下由用户或上游显式提供的修复图配对合法，非 L3 仍然禁止。
- `algo/eval/offline.py`：`predict / predict_frame(pair_template=)` 透传到 tpl。`slots.tpl.pair_auto_resolve` 和 `pair_dirs` 可选，用于按上游契约自动解析配对。`slots.tpl.pair_tpl_weight` 可让 tpl 主导融合，默认 None，即关闭。
- `backend/engine/__init__.py`：推理入口透传 `pair_template`。不传时走原逻辑。
- `tests/smoke/test_round5_l3_pair.py`：4 项合成测试（默认关闭时行为不变、有缺陷的配对分更高、文件名解析、缺配对时回退），全部 OK。
- 默认配置未开启 pair_mode，默认路径数值与 p95 不变。

## 结果（evalset 与 summary_orig.json 相同：47 张正常 + 47 张缺陷，未挑样本）
| 口径 | tpl AUROC | win_rate | FPR≤5% 时 recall | 融合(w_tpl=1.0) AUROC |
|---|---|---|---|---|
| 基线（全局模板库） | 0.4975 | 0.489 | 0.085 | 主融合 0.5115 |
| **pairA：缺陷图配自身修复图，正常图配自身** | **1.0** | **1.0** | **1.0** | **1.0** |
| pairC：正常图配另一张修复图 | 0.0 | 0.0 | 0.0 | 0.0 |
| 重拍模拟：正常图配自身扰动副本（±3px、±4% 亮度、σ=2 噪声） | 0.026 | — | 0.0 | — |

- pairA 复现了 demo4 s_r 配对差分的 AUROC 1.0。缺陷与修复图的差分分位为 P0/P5/P50 = 0.34/0.57/0.96，正常图为 0。
- 口径说明：gold_finger 的修复图是在缺陷图上直接做像素编辑得到的，所以差分只出现在缺陷区域。正常图和自身配对，差分为 0，这是一个平凡分数。一旦模板换成另一张图（pairC）或做重拍扰动，正常图差分升到 1.8～2.4，高于全部缺陷差分。**所以 1.0 只在“模板与检测图像素级同源”的条件下成立，不代表真实重拍工况下也能达到。** 真实产线需要用同工位重拍样本标定阈值，再复测一次（需真实环境）。
- 固定阈值：用 train 正常图的跨图配对 p95 定阈值（4.63），会高于所有配对缺陷分，recall 为 0。按 pairA 口径，只要阈值 >0 且 <0.34，就能完全分开，但这个阈值无法只用正常图正当标定，所以没有写进默认配置。
- 延迟：配对模式下冷提取模板特征的 p95 为 786ms（基线 558ms），因为多了一次模板前向。命中缓存时与原路径相近。默认关闭，不影响默认 p95。

## 复现
- `python %TEMP%\r5_pair_eval.py` 生成 `raw/summary_pair.json` 和 `raw/per_image_pair.json`
- `python %TEMP%\r5_pair_recap.py` 生成 `raw/summary_pair_recap.json`（临时脚本，已不保留）
- `python -m unittest tests.smoke.test_round5_l3_pair` 结果：4 OK
