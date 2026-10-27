# PCB-DUAL-03 结果记录（2026-10-13 本轮更新）

## 已交付
- scheduler 默认启用请求内 adapter.run_many；可用 batch_enabled=False 关闭，stop_on_error 自动逐项。保留 gate、冻结参数优先级、ROI/参数/version_id 快照、诊断与超时保护。
- SMT 旧四 flags 强制合并不等价（cold 依赖 insufficient 结果），已改为独立 flags/判定/渲染，仅复用局部前处理、对齐与焊锡提取。
- TH 保留每项 ready/code/ERROR/REVIEW/消息语义；同输入且实际阈值一致复用 inspect，重新过滤缺陷并渲染。不同有效配置不强行合并；无跨请求缓存。
- 未使用的 core.run_many 已移除，实际接入为 scheduler → adapter.run_many → core.run 的局部共享阶段。
- 批量结果 elapsed_ms 为该项实际 adapter 执行耗时；batch.elapsed_ms 明确为共享批次墙钟时间。超时不可取消、无重试；inflight_snapshot 提供 count/limit/key/age_ms/timed_out/cancellable，超时/重入/容量告警，批量超时 metadata 附快照。
- 已有 PCB-DUAL-02 状态与缺陷表实现经离屏定向核查：REVIEW/SKIP/ERROR 不整体 OK；算法、ROI、耗时及复核原因列存在并测试。

## 实际调度计数（真实算法 + mock 包装，不是纯接口替身）
| 选择 | scalar | batch | 说明 |
|---|---:|---:|---|
| 4 SMT | extract_solder 4 | 1 | adapter.run_many 1；core.run 仍逐项，独立判定不可无条件合并 |
| 5 TH 单框默认参数 | run_inspect 4 | 2 | 连锡未就绪不执行；有效阈值分成两组，不能虚报 5→1 |
| 5 TH 单框相同 {} 参数 | run_inspect 4 | 1 | 四盘内项同组，连锡 SKIP |
| 5 TH 多框 | run_inspect 1 | 1 | 仅连锡可执行，四盘内项 SKIP，本就没有冗余 inspect |

## 验证
- py -3.11 + QT_QPA_PLATFORM=offscreen，tests.test_adapter_batch（8）+ test_shift_batch（4）+ test_tuner_preview（7）：19/19 PASS。最新执行 2.405s，仅测试耗时，非性能证据。
- 小合成图逐项/批量递归字段与 ndarray 精确比较（仅排除非确定耗时）；覆盖 flags override、toe/rim、参数/输入/ROI隔离、缺图/缺ROI/异常、TH单框/多框、冻结 shared override、状态表及0.01秒超时回收/无重试。
- 真实脚本固定 seed=20261013、TEMP cwd、逐样本进度：3 SMT + 1 Shift + 2 TH。SMT源/适配器状态、框和公共metadata一致，四项 batch/scalar 精确比较通过；Shift本轮源对照 NG/NG、框及公共metadata一致；TH 2-8.png/3-2.png 与 templ_2/templ_3 真实配套，对四个可执行项 source/adapter 状态/框/输出数组一致，五项batch/scalar精确比较通过。
- TH 搜索范围补齐开发版 alg_repo/through_hole/detect/input/{standard_roi,roi}，找到了真实配套，旧“仅AOI_CV未发现所以永久BLOCKED”结论作废。使用 paired JSON solder_mode + 源默认参数；未注入整份 config.yaml，也未伪造人工 ROI。

## 边界与外部条件
- 无 GT，不宣称精度；真实 SMT toe/rim/Crack 全分支仍未具备配套。本批无全量数据集或性能专项，速度仅沿用0.95s设计预算，不能保证所有图<1s。
- 缺陷框 roi_id 仍依赖算法提供，缺省显示“未提供”，未伪造一对一 ROI 对应；快照中完整 ROI 可追溯。
- 已补TH强制code!=0/REVIEW/异常、scheduler批量异常及stop_on_error故障注入等价测试；独立容量饱和未测试，不将未测试分支称为全面验收。最后回归仍19/19 PASS，2.389s（此前2.405s记录为上一轮）。
- 不改 stable/AOI_Core/生产DB/生产快照，不提交 Git，不勾生产验收。稳定版、真实桌面及精度验收仍需独立授权。

## 文件
开发根目录 d:\CGAIC\AOI_SSOCL\AOI_tree\PCB_Dual：app/detect/scheduler.py、app/detect/adapters/{smt_adapter,through_hole_adapter}.py、alg_repo/smt/src/algorithms/solder_smt/composite_smt_all/solder_smt_all_alg.py、alg_repo/through_hole/detect/src/algorithms/pcb_through_hole/pcb_through_hole_solder_alg.py、tests/{test_adapter_batch,tuner02_real_cases}.py、docs/架构与设计.md。
证据：env.json、cmd.txt、raw/real_cases.json、raw/batch_tests.json。既有专项及协同追加更新。
