# AOI_tree 全量改进任务清单

更新时间：2026-10-14。本清单为持续实施台账，不是生产达标声明；每轮在原行更新状态/证据，并在末尾记录实际交付。范围为 AOI_Core、PCB_Dual、AOI_feature 及相关 EXPs 复现链。禁止恢复用户删除的 TEMP 脚本、提交 git、修改生产快照或擅跑全量数据。

## 2026-10-13 进度快照（A→G 完成，H 块进行中）

维护分工：本台账是实施进度台账（每轮更新）；`简明改进建议.md` 是验收规范（AC 定义与 G1～G6），只同步“现实情况”列，生产验收框不勾。

**进度总览（2026-10-13，按权威清单 `简明改进建议.md` 统计）**
- AC 共 80 项：已实现（模拟验证，待实机）24（AC-004/008/013/018/020/021/022/028/029/030/031/032/038/051/056/057/063/064/065/067/070/071/072/075，同权威清单）、部分实现 54、未实现 2。生产验收框勾选仍为 0。（2026-10-14 SYNC-28/29、V§85～§91 后按权威清单各行状态列重数；“基本实现”一律记部分实现；AC-062 两组结论冲突保守取部分实现；AC-060 两组冲突取较新 V§91 的部分实现；SYNC-30 V§94～§97 第三轮后重数，权威清单数量不变）
- 未实现 2 项：AC-045/046。
- 本开发对话近期完成代码部分（仍为部分实现）：AC-055/058/063/064/065/074/076（V§43～§55）。
- 本轮新增：Core 已接入“特征工程”统一页面，复用既有 feature-set/revision/report/rebuild API；生产特征只读，候选配置仅登记 draft，不自动切生产。语法检查通过；特征 API 定向 **6 tests / 3.849s / OK**；离屏 FeaturePage 构造通过。该页尚未完成真实桌面/API/GPU/生产验收，相关 AC 仍保持部分实现，生产验收仍为 0。
- 下一步候选：补 Core 页面契约/权限覆盖，完成 AOI_feature 提取链与 Core Trad 的真实等价验证；随后再做少量图精度/耗时评测。

**快照明细**
- 已完成分块：A（AC-001～009）、B（AC-010～017）、C/D/E（AC-018～027）、F（AC-063～065）、G（AC-028～037，V§29～§33）。
- H 块：AC-038 隔离守卫、AC-039 回放审计、AC-042 review_stage、AC-043 回放证据绑定、AC-040 只读发布发现、AC-041 学习态/RNG 严格预验证均已补最小实现/证据（V§34～§38），仍均为部分实现；AC-040/041 尚未生产验收。下一步 AC-044～050。
- AC-040 新增 `readonly_connection`/`discover_prepared` 只读 API；39 tests / 16.769s OK，历史缺 DB 不创建，prepared target/parent/conflict 仅提供发现与边界依据。
- AC-041 当前 v2 已覆盖 handler/cache/gate/config、局部 NumPy/CPU Torch、全局 Python/NumPy/Torch/CUDA RNG 字段；新增 RNG 严格预验证及 4 项测试，组合学习态相关 26 tests / 7.295s OK。真实 GPU、跨环境、完整回放仍缺。
- AC-050 候选事件归属修复：`CandidateRelease.status` 按 `candidate_id` 过滤；历史缺 `candidate_id` 仅按 `payload.candidate_event` 精确匹配；无法确认时标记 `orphan/unavailable`。新增候选串线和历史缺字段测试，**41 passed、77 warnings、19 subtests**。生命周期时间线、水位、结构化失败原因、取消续接仍未解决；生产验收仍为 0。
- AC-049/050 新增最小切片：`backend/evaluation/paired.py` 的 `paired_prediction_delta` 按同 seed/完整样本 ID 集合计算 candidate-baseline 的 AUROC/FPR/recall/gray delta；缺种子、ID 重复/缺失、标签/样本集不一致、NaN/Inf、单类别返回 `unavailable`，不虚构 CI。新增5项测试 **1.379s通过**；`CandidateRelease` 扩展 `reason_code`/`violations`/`retryable`，验收报告、构建失败/预算延期写结构化原因并由 `learning_trace` 保留。合并定向 **52 tests / 22.319s / OK**（命令总26.607s）。仍未实现六臂同预算训练、配对CI、多流多种子、完整生命周期时间线/水位/取消续接；AC-049/050仍部分实现，生产验收仍为0。
- AC-047 最小修复：候选构建开始记录 `candidate_guard`（feedback_id/detection_id/invalidated/revision），candidate_release 在验收/批准/激活前复核；作废或区域 revision 变化 fail-closed，旧候选无 guard 兼容。候选发布定向 **45 项通过（新增4项）**，`py_compile` 通过，生产指针不变。完整争议修订链、独立参考标签、撤错可信重建仍缺；生产验收仍为 0。
- AC-048 最小修复：`CandidateRelease` 增加结构化 `reason_code`；作废或人工修订候选在 validate/decide/activate 前 fail-closed，候选标记 `blocked`，要求从可信 parent 重建。定向 **45 tests / 25.369s / OK**，`py_compile` 通过；未改生产版本。完整撤错重建、参考标签重算、特征库头 CDF 阈值重放仍缺，AC-048 为部分实现，生产验收仍为 0。
- AC-076/073 重要发现与修复（V§43）：`POST /api/models/{id}/activate` 与 `/api/models/rollback` 曾接受 `allow_stale=true` 并转发 `engine.activate`，管理员可绕过特征表示 stale 门禁直接切生产指针；现显式 true 返回 409 `STALE_OVERRIDE_FORBIDDEN`（retryable=False，检查在 resolve/门禁/日志前，rollback 在 try 前），生产入口固定 `allow_stale=False`。定向 **9 tests / 4.1s / OK**。领域层 `engine.activate(allow_stale=True)` 仍可内部调用、无审计绑定恢复授权协议，普通快照激活不强制走 CandidateRelease。
- AC-055（V§43）：`feedback_candidate_queued`/`review_submit`/`feedback_invalidate` 三处 OperationLog 补结构化 `extra`（feedback/detection/source_event、category/learning_status、review_stage/origin_decision、invalidated/was_consumed、仅取 Detection 记录的 detection_model_version，缺失为 None）。新增4项，合计 **41 tests / 3.7s / OK**。未测 was_consumed=True；四时点完整关联、日志与业务同事务、候选批准/激活日志仍缺。AC-055/073/074/076 仍部分实现，生产验收 0。
- 全量 smoke：275 tests / 77.150s / OK（2026-10-13，含 §42/§43 改动）。
- **稳定版验收（2026-10-13）**：14 个关键文件 SHA-256 与开发版完全一致；编译检查通过；定向验收 64 用例全部通过（test_stale_override_forbidden 4/4、test_feedback_audit_extra 4/4、test_paired_evaluation 5/5、test_missing_metric_charts 4/4、test_candidate_release 47/47）。稳定版当前可作为验收基线。
- **稳定版第 8 轮（2026-10-14，验收对话，V§92）**：SYNC-28（29 文件，补遗漏 test_async_persist.py）定向 80 OK + 全量 485 OK；SYNC-29（8 文件，补遗漏 algo/feature_revision.py）定向 21 OK + 全量 **506 tests / 135.8s / OK**（与开发版一致）。TEST-05 缺权重挂起问题已由 SYNC-28 dino.py 修复（模拟验证通过）。PCB-ACCEPT：05 已同步复验（35 回归 + 6-case paired OK）、01 真实桌面部分覆盖 PASS、02 部分 PASS、03/04 BLOCKED（GT/阈值、实测授权）。
- **稳定版第 2 轮同步验收（2026-10-13）**：SYNC-02b～07 共 6 批合并同步（02b 文件前轮已一致；03/04/05/06 共 10 文件 + 07 共 2 文件备份后复制、哈希一致）；稳定版定向 65 tests OK + tests/ui 15 tests OK + SYNC-07 定向 12 tests OK，全量 smoke **283 tests / 79.304s / OK**。详见验证记录§50。
- **稳定版第 3/4 轮同步验收（2026-10-13）**：SYNC-08～10（6 文件）全量 **289 tests / 85.431s / OK**（§56）；SYNC-11～12（7 文件）全量 **299 tests / 83.382s / OK**，与开发版一致（§60）。
- **稳定版第 5 轮同步验收（2026-10-13）**：SYNC-13/14/15 合并同步（§65）；FIX-SNAP 使生产快照 v2 可降级加载（生产阻断解除），但复验发现降级 v2 slot 分数对不同图恒定、decision_trace 为空，转开发排查；全量 smoke **331 tests / 89.173s / OK**；TEST-01 复测 direct e2e P95 288.4→216.3ms 仍 FAIL。
- **稳定版第 6 轮同步验收（2026-10-13）**：SYNC-16～18、20～23 共 7 批 22 文件合并同步（§75）；定向 147 tests OK、tests/ui 19 OK、全量 smoke **378 tests / 100.712s / OK**。SYNC-19 编号未使用；TEST-02 口径复测（让行/锁粒度效果）待执行。
- **稳定版第 7 轮同步验收（2026-10-13）**：SYNC-24～27 共 4 批 13 文件合并同步（§80）；定向 57 tests OK、全量 smoke **405 tests / 106.532s / OK**（与开发版一致）。TEST-02 复测 c1 learning P95 1504.8→394.1ms（−74%）；TEST-01 第三轮 direct e2e P95 204.1ms（距红线约 2%），两者仍 FAIL。AC-072 用户已定值并可配置（SYNC-27）。当前无待同步批次。
- **实测任务进展补充（2026-10-13，验收对话）**：TEST-05 部分 FAIL（离线 vendor 加载 PASS；缺权重时 torch.hub 无超时挂起 180s，V§82）；TEST-06 FAIL/new_candidate（8 项过 2 败 6，唯一根因 selfsim RNG 口径 default_rng(42) vs seed(0)，其余 14 组逐位一致，V§83）；TEST-07 FAIL（FPR 100%，tau 低于全部样本得分、2.0 钳位饱和，V§81）。TEST-04（真实桌面）、TEST-09（授权实机视频，BLOCKED）未做。
- **TEST-04 完成（2026-10-14，验收对话，V§84）**：真实桌面联调 PASS（Gauge 空态/口径卡/open 告警/legacy 未知保留/评估看板预算达标线均与 API 一致）；附带确认生产 v2 与 TEMP 新模型同样在 test 集 FPR=100%、AUROC=0.5，精度问题为系统性。至此 TEST-01~08 除 TEST-09（BLOCKED，需授权实机视频）外全部执行完毕。
- **5 组并行（模拟验证，2026-10-13，V§85～§89，待同步 SYNC-28）**：final_sim/final_perf/final_learning/final_release/final_model 五组在 TEMP/合成数据/离屏环境补实现与复验；全量 smoke **485 tests / 147.314s / OK**（405→485，+80）。权威清单按建议状态更新为已实现 17、部分实现 58、未实现 5；AC-062 两组冲突保守取部分实现。HARD-01 纯推理 P95 428.6ms，本机 200ms 不可达；HARD-02 门禁已实现但精度未达标。全部为模拟验证，生产验收 0。
- **第二轮 2 组（模拟验证，2026-10-14，V§90～§91，待同步 SYNC-29，须在 SYNC-24～28 之后整文件同步）**：round2_learning（9 tests OK；AC-063 转已实现，019/042/044/052 部分实现，044 未实现→部分）、round2_release（12 passed/7.9s，回归 55+36 subtests；013/028～032 已实现，011/012/014/015/016/033/034/035 基本实现记部分实现，060 未实现→部分、061 部分）；全量 smoke 第二轮 **506 tests / 154.213s / OK**。权威清单重数为已实现 24、部分实现 54、未实现 2，生产验收 0。
- **实测任务进展（2026-10-13，验收对话）**：TEST-01（AC-066）实测 **FAIL**（2500² 热态 P95 direct 295.8/http 274.7ms，超 200ms 红线；§57）；重要发现——稳定版生产快照 gold_finger v2 缺 layout 统计无法加载，生产可用性阻断，须重新 prepare；TEST-02（AC-069）实测 **FAIL**（c1 learning P95 1505ms，c4 idle P95 406ms 亦破 G1，阻塞在引擎内部锁非队列；§62）；TEST-03（AC-070）实测 **PASS**（整卡峰值 926MiB，上限草案已出，OOM 子项未做；§58）；TEST-08（AC-072）草案已出（§59）；**2026-10-13 用户已定值**（FPR≤5%、Recall≥95%、回退≤2pt/版本、灰区≤15%，保留期见 test08 result.md），开发版已做成 `business_budget` 可配置项，设置页可编辑，统计/评估页与候选门禁 policy 已接入；保留期不自动删除，标签迟到未接入入训；AC-072 部分实现，待同步 SYNC-27，生产验收 0（V§79）。追加（2026-10-13，开发对话）：OPT-DECODE 让 PNG 也走 cv2 直出 RGB，像素和分数差异为 0。开发版同进程 A/B（TEMP，n=24）中，decode p95 116.5→105.3ms，e2e p95 205.7→190.8ms；稳定版 TEST-01 口径待复测，AC-066 未通过（V§64/SYNC-15）。追加 AC-069 证据（2026-10-13，开发对话）：即学改为“有上界让行（1.5s）+ 锁外预提取 + 锁内仅打分/状态/WAL”，推理透出真实 `lock_wait_ms`，并新增 `lock_stats()`。修复 consolidate 候选路径让行无上界会被饿死的问题。引擎层小测（≤20 图，TEMP v1 副本，单品类，前台闭环）中，前台 P95 idle 183.0ms，改动前模拟 1359.3ms，改动后 254.8ms，特征已缓存 127.2ms；30 次请求中等锁次数从 10 降到 1。全量 smoke 338 OK。剩余缺口：学习被限速、锁内残余 hold 约 0.4～0.54s、前台之间仍串行。稳定版 TEST-02 口径未复测，生产验收 0（V§66/SYNC-16）。**前台锁粒度追加（开发版，2026-10-13）**：前台的读图、预处理、切块、DINO 前向移到品类锁外预提取，前向套全局 GPU 信号量，进锁后按指纹校验，失效则重算；读写锁和 GPU 合批评估后不采用。test02 口径开发版 A/B 结果：c4_idle 的 lock_wait P95 从 77.7ms 降到 0.007ms，前台之间串行等锁已消除；wall P95 302.7→303.1ms 持平（瓶颈转到单 GPU 前向和锁外 persist/heatmap），c2_idle 294.0→248.9ms，c1_idle 245.9→258.3ms（噪声）。并发与串行 mismatch=0，前后 score 完全一致。全量 smoke 371 OK。剩余缺口：学习档锁内 hold 700～1300ms、单 GPU 前向耗时。稳定版未同步，生产验收 0（V§72/SYNC-22）。
- V§45（2026-10-13）：UI 缺失值收尾（SYNC-02/02b，tests/ui 15 OK）；AC-055 补 was_consumed=True 测试；AC-074 候选 decision/activate/rollback/recover 原先无 OperationLog，现成功/拒绝均写结构化日志（含 reason_code），定向 55 tests / 21.6s OK。待验收对话同步 SYNC-02b、SYNC-03。
- V§46（2026-10-13）：全量 smoke 279 OK；AC-063 定位容器补 coord/image_size/model_version/heatmap_postproc，定向 6 tests OK。待同步 SYNC-04。
- V§47（2026-10-13）：AC-058 实时流游标修正——过滤造成的 ID 洞原被误报“丢帧”，库重置后游标卡死；现只计全表缺失 ID、越界重订阅，3+15 tests OK。待同步 SYNC-05。
- V§48（2026-10-13）：全量 smoke 282 OK；AC-064 区域复核争议/噪声隔离（他人无理由推翻→disputed 不进训练，仲裁须 reason），5 tests OK。待同步 SYNC-06。
- V§49（2026-10-13）：AC-058 统计口径——overview scope_stats 新增窗口起点/分母（已落库检测，非相机帧）/工单/by_source/真实 e2e（不再以推理耗时冒充）/receipt_rate 不可用声明，旧字段不变，2+7=9 tests OK。待同步 SYNC-07。
- V§51（2026-10-13）：AC-064 多框复核 UI 面板——RegionReviewDialog 列框/确认/拒绝/标争议，event_id 幂等 + expected_revision 乐观锁，409 显示 detail 并刷新；FeedbackDialog 新增“多框复核…”入口（与图级反馈分开），tests/ui 4 OK。叠加框可视化、UI 绘框 add/revise、真实桌面未做。待同步 SYNC-08。（§50 已被验收对话占用，顺延 §51）
- V§52（2026-10-13）：AC-065 热图后处理参数版本化——新增 `heatmap_postproc_meta`：algo/阈值基准 per_image_minmax/min_area 单位 display_px/morph_kernel（设置 pipeline.overlay_box_morph_kernel，默认3）/version_id（hpp-+sha1，display_scale 不参与）；默认输出不变，2 tests OK。冻结版本选择与真实标注定位评估未做。待同步 SYNC-09。
- V§53（2026-10-13）：AC-063 热图二值 mask 留存——`render_heatmap_overlay(return_mask=True)` 返回开运算后二值 mask，设置 pipeline.overlay_save_mask（默认 True）时写无损 PNG `{stem}_mask.png`（与 heat/overlay 并行写盘）；heatmap_postproc 新增 raw_range（min-max 基准）/mask_path；孤儿扫描把 `_overlay.jpg` 同前缀 `_mask.png` 计为引用。4 tests OK，与 test_localization 合计 9 OK。原始浮点热图未存，human_box/训练状态仍缺，mask 未入下载白名单。待同步 SYNC-10。
- V§54（2026-10-13）：AC-063 human_box 来源与 region 确认/训练状态——`annotate_review` 只读派生视图（不改存储、不加 DB 列）：每 region 增 origin_source/source（人工确认/add/revise→human_box）/reviewer/review_revision/confirmed/train_status（eligible/excluded_candidate/excluded_rejected/excluded_disputed/excluded_visual）；region_review GET/POST 返回该视图；`trainable_region_boxes` 仅取 eligible。新测试 3 OK，全量 smoke 292 / 85.45s OK。无 region 级 used 状态、UI 未显示新列。待同步 SYNC-11。
- V§55（2026-10-13）：AC-064 原图叠加框（按 train_status/source 着色、选中高亮、单击框选行）+ 拖拽绘框 add / 选中重画 revise（强制 reason、带 expected_revision，409 刷新）；坐标换算/payload 为纯函数；修正对话框请求缺 `/api` 前缀；AC-063 to_dict 透出 annotate_review 视图、表格增训练状态/复核人列。test_region_review_draw 7 OK，全量 smoke 299 / 93.06s OK（离屏，实机待 TEST-04 类实测）。无手柄微调/删除框。待同步 SYNC-12。
- V§61（2026-10-13）：AC-058 UI 展示统计口径——工作台新增推理耗时（仅推理）/端到端耗时（均值/P95，缺 e2e 条数）/接收→落库率（无数据源显示“不可用”）三卡 + 口径行（窗口起点/分母=已落库检测非相机帧/工单或全局/按来源计数），tooltip 说明口径，缺数据显示“—”/“不可用”不以 0 冒充，不设阈值。test_ui_stats_caliber 5 OK，全量 smoke 304 / 84.25s OK（离屏 smoke，实机待测）。backlog/接收计数无数据源、stats_page 未改。待同步 SYNC-13。（§56～§60 被验收对话占用，顺延 §61）
- V§67（2026-10-13）：AC-058 统计页接入口径。stats_page 直接 import 工作台的 scope_caliber_texts/CALIBER_TIP/NA（未复制），推理耗时与端到端耗时（均值/P95/缺 e2e 条数）分卡展示，并加口径行和 tooltip；批次表推理与 e2e 分列。原实现的问题与修复：quality 推理均值被当作“平均延迟”，后端 0.0、`or 0` 被显示成“0 ms/0.0%”，现在缺失或分母为 0 时显示“—”，各比率写明分母（非相机帧、健康卡上限 3000 条）；CSV 缺失值留空；延迟图标注为推理日聚合。test_ui_stats_page_caliber 7 OK，tests/ui 19 OK，全量 smoke 345 / 93.80s OK（离屏）。后端 0.0 冒充缺失、timeseries 的 coalesce 回退与按工单拆分、receipt/backlog 数据源仍为缺口。待同步 SYNC-17。
- V§70（2026-10-13）：AC-064 复核对话框加入三项编辑能力：角点手柄微调后按 revise 提交（须 reason）、op=delete（仅限人工框，软删除 status=deleted/excluded_deleted，模型框返回 422）、op=undo（撤销本人最近一步，追加反向修订并保存 before 快照，被他人改过返回 409，撤销 add 时置为 retracted）。修复：模型框多次 revise 后来源被覆盖成 human_box、撤销 revise 后来源判断有误、delete 缺 region_id 时返回 404。test_region_review_edit 9 OK，全量 smoke 364 / 86.771s OK（离屏）。拖边/平移、重做、实机测试未做。待同步 SYNC-20。
- V§74（2026-10-13）：AC-058 后端统计缺失值改为返回 null：quality/batch/health 的比率和均值、工单 bad_rate 和 rate、_p95([]) 不再以 0.0 冒充；timeseries 的 e2e 改为 avg(latency_e2e_ms)，去掉回退到推理耗时；report 和 CSV 只统计 e2e，并加入“e2e缺失”计数；仓内 UI 遇到 None 时安全显示 NA 或“—”。AC-063 即时学习 submit_feedback 成功后回写 consumption[]（kind=instant，幂等，失败时不回写）。AC-064 复核对话框新增“学习消费”标签列；补测试，覆盖删除或撤销已消费框后 guard 返回 PARENT_REBUILD_REQUIRED_REGION_STALE（deleted/excluded）。test_stats_null_and_consume 7 OK，全量 smoke 383 / 104.433s OK（离屏）。待同步 SYNC-23。
- V§63（2026-10-13）：FIX-SNAP 旧快照 layout 统计缺失兼容——原因：gold_finger v2 保存早于 layout 统计持久化修复，slot_state 仅含旧 PCA 键（全 None），10-12 严格校验拒载整个快照。方案 b（显式降级）：`load_snapshot` 捕获 `LayoutStatsMissing`，登记 `pipe.degraded_slots`（layout_degraded/layout_stats_source=missing_in_snapshot/missing/reason/action）+ warning，layout 移出 `_score_slots`、权重置 0、余者按比例归一；trace fusion 带 degraded_slots；不回退默认常数。严格保护保留：candidate 加载、直接 `_restore_slot_state`、全部槽位降级均仍拒绝。方案 a（recomputed_on_load）未实现：快照无训练图/特征，不能用未知标签图重拟合。test_snapshot_layout_compat 4 OK / 0.63s，全量 smoke 308 / 83.5s OK；真实 v2 TEMP 副本加载 1.0s 成功、5 图推理正常（trace degraded=['layout']）。阈值偏移风险须重新 prepare；稳定版未复验。待同步 SYNC-14（生产阻断修复，优先）。
- 待做：横切 AC-051～080（含 LOC-3/AC-073）；OPT 少量图精度/耗时评测（cum_cap/tail_cap 复核、mechanism_gain_report 真实取分）。
- smoke 全量：239 OK / 72.8s（2026-10-13，历史基线）。生产验收仍为 0。
- **第三轮 4 组（模拟验证，2026-10-14，V§94～§97，待同步 SYNC-30，须在 SYNC-24～29 之后整文件同步）**：round3_speed（blob CPU 无损改写 + GPU DoG 开关默认关 + 离线 worker 数；66 项 OK；gold_finger P95 397.7→316.8ms 默认、开 GPU 181.8ms，26/26 判定一致；HARD-01 未实现→部分实现）；round3_decisions（设置页「待决策事项」governance_decisions、冻结后改阈值 409 THRESHOLD_FROZEN、deplock 依赖锁 warn/block、candidate_release_mode、calibration_isolation；33 passed；AC-019/076 部分实现，AC-045/046 仅登记开关仍未实现）；round3_accuracy（`_soft_bound` 替换 2.0 硬钳位；57 例+64 subtests OK；fused AUROC 0.935→0.970、FPR≤5% 最大 recall 0→0.80，固定阈值 FPR 仍 1.00，根因为 test 正常样本批次 Image_2026/t_fov 未覆盖，blocked on data）；round3_coverage（coverage.py 批次覆盖告警 + 预算参考工作点，不改判定；新增决策键 blob_gpu_enabled/approved_extra_good_batches；Ran 40 OK）。全量 smoke **536 tests / 135.936s / OK**（506→536，`review/task/final_smoke/stderr_r5.txt`）。权威清单计数不变（24/54/2）；本台账行状态计数 已实现 32 / 部分实现 46→47 / 未实现 4→3 / 其他 2（HARD-01 转部分实现）。生产验收 0。
- **第四轮 3 组（模拟验证，2026-10-14，V§98～§100，待同步 SYNC-31，须在 SYNC-30 之后同步）**：round4_sem_shead（sem 方向反、domain 下 shead 权重置零，经诊断都不是代码 bug，根因是 gold_finger 正常图为缺陷图的 `*_repaired` 孪生图，以及批间距离；未改算法代码；新增 test_round4_sem_shead 4 项，与 round3_accuracy 合计 8 OK）；round4_decisions_apply（开发版 `configs/default.yaml` 追加 governance_decisions，开发方代定 7 项，保留人工 6 项；稳定版是否采用由验收对话决定；20 OK）；round4_batch_rebuild（`split_extra_good` + staging 副本，修复 `approved_extra_good_batches` 在重建时实际不生效的问题（test/good 路径被 guard_bundle 拒绝）；新增 6 项、回归 47 项 OK；B 组新批次 eval 固定阈值 FPR 0.556→0.333、recall 0.79→0.26；成对难例 E2 AUROC 只有 0.54～0.58，精度瓶颈在模型对缺陷的敏感度；datasource_id 分支未接入）。全量 smoke **549 tests / 154.231s / OK**（536→549，`review/task/final_smoke/stderr_r6.txt`）。权威清单计数不变（24/54/2），HARD-02 维持部分实现。生产验收 0。
- **REMAIN-01/03 修复交付（2026-10-14，V§110，已登记 SYNC-33，待验收对话复验）**：REMAIN-01 协议 B dtype 对齐——AOI_feature `scoring.py` extract_features 输出与 GroupKNN.fit 全程改 float32，与生产 TradSlot.fit 逐位一致（原 float64 致 bank 差约 3.255e-03），新增协议 B 逐位一致测试；AOI_feature 4 文件归属确认（§4.7）。REMAIN-03 血缘最小闭环——Detection 新增 feature_revision_id/feature_lineage 列 + DB 迁移 v8；`_persist` 落库真实血缘（unbound/unknown 如实）；`persist.py` 快照 meta 不再恒写 unknown；`detection_stamp` 带 lineage 链；新增 GET production/pointer、POST bind/rollback 三个 HTTP 路由（admin+审计+409 fail-closed）；§4.6 归属确认。修复一处回归：`service.py` 相对导入越界导致 40 项导入 ERROR，改绝对导入后全量 smoke **555 tests / 141.169s / OK**（549→555，`stderr_r7.txt`）；定向 13+5 tests OK。权威清单计数不变（24/54/2）。稳定版复验（协议 B 八项全链路、TEMP 激活/回滚/重启实证、迁移 v8 升级路径）均待验收对话，不预判 PASS。生产验收 0。

## 2026-10-12 G1服务路径与巩固锁

HARD-01/AC-066 的工程测量已达标：8 张工程 2500 图完整服务（热图、持久化、告警、JSON）p95=190.11ms（engine_path 129.79ms），决策与 prior 一致；原生 2500、HTTP/并发和 G2 精度仍待正式验收，生产框不勾。AC-068：consolidate(activate=False) 的装载和重放已移出品类锁，backbone 哈希已缓存；快照哈希已移出锁、候选父哈希锁外预热，竞争场景复测锁等待max 160.3ms、overlap P95 319.0ms（2500工程图，非原生正式验收）。smoke 137 OK。详见验证记录开头。

## 2026-10-12 service.py 语法阻断修复续验

仅移除 `submit_feedback_record` 提前闭合的 docstring 三引号，M7a/M8a 说明恢复为文档字符串；py_compile 和 AST 安全门禁断言通过，`candidate_only=True`、`allow_legacy_online=False` 及显式 legacy 拒绝分支保留。四模块回归实际为 **41 tests / 15.057s / failures=8, errors=3 / exit 1**（后续复核：同四模块 45 tests OK，已关闭）：旧在线反馈预期与候选默认排队契约不符、TEMP SQLite 清理错误及候选重复请求 outputs 索引错误仍开放；学习态 7 项通过、隔离候选其余 15 项通过。仅解除导入语法阻断，不改功能/测试以追求全绿；SS-A/SS-B.STATE 及相关 AC 状态仍为部分实现、生产未验收，不勾生产验收。详细命令与失败项见验证记录开头；未回滚并行实现、未提交、未恢复删除文件、未改生产 DB/快照。

## 2026-10-12 候选验收/批准/发布实施台账

AC-039/040/043/050/053维持部分实现、生产未验收。新增独立CandidateRelease账本服务、候选REST与TEMP测试，engine候选发布必须明确approved；双域真实predict报告绑定candidate/parent/WAL/截止ID/config与数据hash，缺数据unavailable；计数预算/工件验收、批准幂等冲突、过期拒绝、锁外加载与短协调切换、可信父回滚、水位记录及显式prepared恢复已接通。首轮24 tests / 10.714s通过（其中新增8项）；本轮续做提交后业务Model镜像同步，候选activate/rollback及显式recover接通，业务提交失败明确报错且保留真实已发布版本、重复重试幂等，同品类唯一active，历史缺行只补目标快照身份、不伪造精度/数据血缘。业务DB访问不持品类推理锁，不调用全库扫描；engine严格只改rollback目标块、service不再重复覆盖候选唯一镜像。合并候选/Feature gate/异步反馈回归61 tests / 15.849s / OK（发布15项，其中本轮新增7项）。详细命令与风险见验证记录开头。仍为部分实现、生产未验收：真实独立数据/近重复与批次隔离、HTTP密钥全链、自动启动恢复、多进程缓存、完整学习态/RNG、G1及GPU硬预算尚未完成。旧反馈直接工作态未改，E_open/FeatureSet/UI未改，生产DB/快照未写、未提交git。

## 0. 权威验收编号同步（2026-10-12文档复核）

验收规范以`d:\CGAIC\commit\repo\review\简明改进建议.md`AC-001～AC-080及G1～G6为准；本台账原行保留实施证据，后续更新同时引用AC编号，不能沿用旧口径授予生产通过。权威清单80条：已实现17、部分实现58、未实现5（2026-10-13 SYNC-28 后复计，均为模拟验证），生产验收0（AC-068候选计数底座及AC-061最小revision为部分实现，非生产验收）；合并粒度与本台账不同，不直接相加。本轮已完成候选恢复与engine归因透传的代码/小型回归验证，但不改变生产验收统计，也未修改生产DB/快照。

- S1原37编号的一对一映射见权威§7.1（B6→AC-016、B7→AC-015，其余依原组顺序）。
- UI-P0.1～5→AC-051/056/057/050/053；UI-P1.1～5→AC-052/054/055/058/058；UI-P2.1～5→AC-060/061/040+067/074/059。
- LOC-1/2/3/4/5→AC-063/065/073/064/065；UI-ARCH→AC-073。
- SS-A/WAL/PUBLISH→AC-038/039/040/041；SS-B/STATE→AC-039/041；SS-C/BUDGET/ORDER→AC-043/068/042；SS-D/E/F/STAGE/AUDIT→AC-044～046/047～048/050/042/049。
- HARD-01/02/03→AC-066/067/068；并发GPU、显存、冷启动分别由AC-069/070/071追踪。
- REPORT-1～10→AC-075/076/077/078/079/076/079/076/076/080。

§7是首批历史审计与交付，必须结合§5最新底座和V§16～18解读：显式`learn_candidate`已有锁外父快照/WAL冻结及候选保存、`acceptance=unavailable`禁止该路径激活；§7.2“候选仍在激活pipe执行”等指旧反馈/consolidate路径，不能据此否定新增底座，也不能据新增底座声称旧路径已隔离。完整批准/可恢复激活、学习态RNG/跨进程一致、200ms/独立精度/资源仍开放，原证据不覆盖。

## 1. 来源、状态与证据规则

已读取并对照当前代码：
- S1 `d:\CGAIC\commit\repo\review\简明改进建议.md`：A1–A9、B1–B8、CDEF1–CDEF10、GHIJK1–GHIJK10，共37条。
- S2 `d:\CGAIC\commit\repo\review\功能前端映射审查与 AOI_feature 整合设计建议.md`：§11 P0/P1/P2各5条，§15定位专项及§4–§13详细约束。
- S3 `d:\CGAIC\commit\repo\review\SSOCL分阶段落地建议与系统改进清单.md`：§8 A–F任务包，§4–§7及§9约束。
- S4 `d:\CGAIC\commit\repo\AOI_SSOCL生产复现报告.md`：A/B/C/D/E证据及开放项。
- S5 `d:\CGAIC\commit\repo\AOI_SSOCL对比审查报告.md`：§7六项建议及§4证据限制。
- V `d:\CGAIC\AOI_SSOCL\review\AOI_tree首批改进与验证记录.md`：历史轻量、8图计时、20图精度、25图重反馈及本轮WAL修复。

状态：**已实现**=代码链存在但尚未满足对应完整验收；**部分实现**=只有部分链路/证据；**未实现**=已查相关生产入口但未找到完整要求的实现；**已验收**必须明确范围、命令、结果和证据，不能用单元测试授予生产性能/精度结论。未实现不是绝对全树不存在的证明，后续发现代码须更新。本轮没有任何完整建议获得生产“已验收”。

路径简写均相对 `d:\CGAIC\AOI_SSOCL\AOI_tree`：E=`AOI_Core/backend/engine/__init__.py`；P=`AOI_Core/algo/eval/offline.py`；H=`AOI_Core/algo/ssocl/feedback.py`；Z=`AOI_Core/algo/persist.py`；DB=`AOI_Core/backend/db/models.py`；SL=`AOI_Core/backend/self_learning/service.py`；DS=`AOI_Core/backend/pipeline/service.py`；U=`AOI_Core/ui`；F=`AOI_feature`；D=`PCB_Dual`。

优先级：P0生产可信性/安全/硬红线，P1算法与操作闭环，P2工程工作区及研究扩展。优先不代表可以跳过依赖。测试条件缩写：
- T0：隔离 TEMP 工件、固定种子、单元/小样本；真实算法与替身测试边界分开记录。
- T1：§2端到端时延与并发资源协议。
- T2：§2独立精度协议；标签/数据/阈值血缘齐全。
- T3：并发、故障注入、重复事件、重启/LRU与继续学习恢复。
- T4：API契约＋离屏UI＋角色权限＋真实操作联调。
- T5：真实框/掩码、覆盖/原图坐标/IoU与定位召回，不拿图级AUC替代。
- T6：新旧域分层持续学习、训练字节/步数/墙钟预算与长期恢复。

## 2. 强制验收协议（覆盖来源中的旧1秒口径）

### 2.1 检测硬红线

**2500×2500像素单图端到端 p95 ≤ 200ms，并且精度不退化，两者同时通过才可发布。** 旧“单图<1秒”、模型内段速度、均值、磁盘热且特征缓存命中、其他尺寸折算都不能代替此门禁。

1. 优先使用真实原生2500×2500图，记录数据清单/内容SHA256/尺寸/编码/存储介质。没有该尺寸时：其他真实尺寸仅诊断；缩放至2500的图只作合成尺寸工程检查，折算结果标为估算，均不得写实测通过。
2. 时间从输入请求可处理/开始读取到完整检测输出可用；包含读取、解码、预处理、冻结主干前向、槽位打分、融合、判定及结果包装/序列化。按生产入口需生成热图/文件/落库/返回时也计入，同时单列服务与客户端排队/传输。模型冷装载、首图预热单列；生产若会冷装载，需另报该路径，不藏在统计外。
3. 每张每轮强制禁止 `_item_cache` 或其他特征复用；预热也不能复用特征。模型常驻可允许，操作系统页缓存冷/热分别报告，不把热读当冷读。GPU在计时边界 synchronize；使用 perf_counter。记录I/O+解码、预处理、前向、打分+输出分段，不重复相加嵌套阶段。
4. 固定硬件/GPU、CPU线程、Python/Torch/sklearn、代码/主干/配置/模型hash和种子。以同图同快照交替重复基线/候选；报告逐图原始时间、N、P50/P95/P99/max、分位计算方法及置信区间。初筛只取少量图；正式样本量/重复次数与置信要求先冻结，未经授权不跑全量。不以8图线性插值p95作生产保证。
5. 分别测空闲和后台学习竞争，排队计时从请求到达开始；同品类锁等待不得藏在 `DetectionResult.latency_ms` 外。预算失败就拒绝候选，不减少缺陷覆盖或放宽精度标准来“提速”。

### 2.2 精度门禁

- 先按工件/批次/近重复分组隔离训练、正常建库、校准、门控开发集、最终独立测试集；固定manifest及哈希。初始化缺陷不得取最终测试集，历史回放也不得侵入测试。数据不足标“不可验收”。
- 业务图像级误报率预算、gray计分、旧域容差、微小/低对比度/边界/全局异常子集及置信要求在试验前确定。现阶段预算数值尚未获得，**阻塞正式验收，不代填默认值**。
- 评分方向、槽位选择、融合权重、阈值与定位后处理只用训练/校准/开发验证数据确定并冻结；禁止测试标签选方向（含max(AUC,1-AUC)）、调阈值/挑版本。测试只最终报告实际图级FPR、在冻结误报约束下召回及配对不退化结论，补报漏检数、AUROC/AP、灰区、域外误报、置信区间。
- 相等于失败基线不是达标。V§6中20图正常3/缺陷17，fused AUC=0（HEAD也0），只能证明该小样本指标相等；未形成固定FPR召回结论，绝不能称精度达标。查训练/部署域分布、常数分、方向/校准和冻结权重，不能用这20个测试标签倒选方向。

### 2.3 持续学习必须有界

- 每轮限制反馈事件数、正常/缺陷/回放字节与样本、LOO读取量、搜索候选数、优化步数、GPU/CPU占用及墙钟预算；积压/超预算进入deferred/暂停，不借异步隐藏总耗时。数值预算待业务确认，但有界机制本身必须实施。
- 微批候选在独立实例运行；生产推理不被长训练锁阻塞。取消/超时不能继续修改激活模型。每轮记录排队→训练→门控→保存→激活总时间和前台p95。
- V§7的25图重路径第12轮11.670592秒是一次观测，不是默认规模或全局最坏上界；未覆盖disc接受后的二次LOO、router、搜索、孵育、最大库/缓存、HTTP/DB和锁等待。S4所谓“转后台不阻塞检测”被当前E品类锁逻辑否定，已在 AOI_tree 文档中勘误（REPORT-3）。

## 3. S1逐条任务（37条）

| 原编号 / 要求 | 状态 | 当前代码证据与缺口 | 依赖 | 优先级 | 测试条件 |
|---|---|---|---|---|---|
| A1 缺陷尺度/域差决定整图或局部 | 已实现（功能） | algo/common/resolution.py ResolutionProfile+select_view_mode（global/global_local/domain_first，含bounds）；offline._fit_local_branch按档案或tiling.local.enabled(true/false/auto)判定，view_mode落盘 | — | P1 | smoke test_local_branch.test_modes/profile_from_cfg/bounds_direction |
| A2 全局上下文分支 | 已实现（功能） | 整图专家（Decider）恒保留，局部分支为_finalize叠加层，不改Decider；enabled=false时输出逐字段不变 | — | P1 | test_disabled_passthrough |
| A3 单级重叠确定性局部覆盖/ROI覆盖 | 已实现（功能） | LocalBranch单级重叠滑窗；ROI块必选（BudgetScheduler） | — | P1 | test_tiles_cover_and_labels/test_scheduler |
| A4 块尺寸、边缘覆盖、坐标反解 | 已实现（功能） | resolve_tile_size（数值/auto，bounds缺失回落512并标auto_bounds_missing_candidate_512）；view_signature落盘，不一致停用局部分支 | — | P1 | test_resolve/test_state_roundtrip_and_signature/test_persist_load |
| A5 训练/建库/校准/推理粒度一致 | 已实现（功能） | 块面积占比标签≥0.25正/≤0.05负（候选值），中间忽略；伪异常默认关，开启即报未实现 | — | P1 | test_tiles_cover_and_labels/test_pseudo_off_by_default |
| A6 分支图级误报预算/块图两层校准 | 已实现（功能） | branch_budget：图级α均分两分支、块级Šidák、并集基线、holdout同批统计fpr_union；未冻结默认值标来源；branch_ratio处理τ≤0；V§89 4000张合成图两层校准复验（模拟验证，真实分支预算未冻结） | — | P0 | test_sidak_and_union_thresholds/test_sources_unfrozen/test_branch_ratio_nonpositive_tau |
| A7 分段时延与预算细判 | 已实现（功能），性能未达标 | timing分段io/tile/forward/score/local/hier/decide/tta/total；budget模式热度优先+轮换，灰区/冲突全量重算。AC-009实测local分支p95 3.2~4.0s（63块/图，3000×4096），远超G1 | OPT | P0 | test_budget_mode_and_force_full/test_budget_conflict_rescore；AC-009报告 |
| A8 细判追加/确认与新增漏检统计 | 已实现（功能） | side_effects：added_by_local/confirmed/global_only/vetoed；否决默认关且仅降gray；added_by_local定位用局部热图（source=local_branch） | — | P1 | test_added_by_local/test_veto_only_to_gray/test_mask_boxes |
| A9 三方案独立验收 | 已执行，结论：局部分支不应默认开 | commit/repo/revise/ac009_three_way.py；gold_finger 20图（17缺陷/3良品）：整图p95 475ms(折算241.8)；full p95 3531ms；budget p95 4327ms；救回漏检0、新增误报0；三方案G1均不过。详见验证记录§25；V§89 scheme_compare 同数据三方案协议已建，块数0/24/6（合成，模拟验证），真实G1/G2缺 | OPT | P0 | revise/ac009_gold_finger.json |
| B1 统一注册表接入trad | 未实现 | F/feature_lib.py独立注册；algo/slots/trad.py固定ext/200维，未共用F配置；V§89 parity harness 可做组级切片比对（模拟验证），生产提取仍未统一 | B7 | P1 | T0提取一致性/T2 |
| B2 按品类/视图验证启用 | 部分实现 | profile及sanity槽权重存在；传统组级/局部贡献选择未生产化；V§91 基本实现：缺口为 group_selection_plan 未接入UI/API（模拟） | B1/B3 | P1 | T2，DINO基线 |
| B3 贡献档案/簇级消融 | 部分实现 | F/contribution.py、algo/eval/contribution.py单组/消融；缺版本×品类×缺陷×视图强绑定与CI；V§91 基本实现：缺口为档案生成流程未自动调用 stamp_profile_lineage（模拟） | B1/B7 | P1 | T2，固定/重建消融分开 |
| B4 验证集选择方向/权重/组维归一 | 已实现（模拟验证，生产未验收） | P slot_sanity来自init_defect；trad全向量L2且score_image跳首邻，未组级归一/独立验证；V§91 trad_protocol_audit 文档化审计已实现（权威AC-013转已实现），生产trad打分协议未改 | CDEF2、B3 | P0 | T2，禁止test选方向 |
| B5 单图/逐框解释 | 部分实现 | E/DS含raw/cal/weights；新增图级反馈持久详情/未知显示（验证§20）；逐框dominant_slot仍复制整图，trad参考范围/近邻未全呈现；V§91 基本实现，缺口见权威AC-014（模拟） | B1、LOC | P1 | T4/T5，区域证据；归因4项/组合14项契约通过，非生产验收 |
| B6 前端特征闭环和待验证配置 | 部分实现 | U监控显示feature metadata；F独立界面，无完整工程师revision发布入口；V§91 基本实现：缺口为 feature_ops_view/submit_toggle 未接入API/UI权限体系（模拟） | B1/B7、UI-P2.2 | P2 | T4权限/过期报告 |
| B7 库版本血缘与不兼容向量隔离 | 部分实现 | Z production_feature_metadata及最小FeatureSetRevision绑定实现/config/数据/模型与证据hash，显式绑定模型有前门禁（验证§24）；表示变化重提/库隔离/完整发布仍缺；V§91 基本实现，缺口见权威AC-015（模拟） | UI-P0.5、SS-B | P0 | T0/T3/T2重提/回滚 |
| B8 难例候选特征准入 | 已实现（smoke，生产未验收） | backend/evaluation/hard_admission.py：漏检/误报/冲突/漂移/open难例池+train参照→候选revision真实trad重打分，独立gain/redundancy/regression/G1/G2五项，全passed才可approved（feature_revision门禁消费，hard_pool_sha256）；未注册/auto_discovery→planned，open正样本>0.5→insufficient_evidence；API hard-pool/admission(engineer)+学习页按钮；验证记录§26 | B3/B7、GHIJK7 | P2 | tests/smoke/test_hard_admission.py；196 OK |
| CDEF1 槽CDF定义/尾部稳定性 | 已实现（smoke，生产未验收） | cde_report.calibration_contract：各槽bins/样本量/small_sample/尾部分位/self_match_bias，self模式标高风险；持久cde_report.json，slot-health API/学习页可见；验证记录§27 | 校准留出集 | P0 | test_cde_report [1/8]；205 OK；真实数据尾部稳定性未测 |
| CDEF2 建库/训练/校准/阈值数据隔离 | 部分实现 | guard_bundle隔离train/test；calibration_contract报告iso=self/holdout及自匹配偏差，stratified_split供独立切分；P fit默认仍self校准，未切换完整交叉拟合（§27.3）；V§90 offline.resolve_isolation 默认 self→auto（fit≥16 走 holdout 80/20，否则回退 self 并记原因），需真实数据确认不退化、已有快照需重建（AC-019，模拟） | 数据manifest | P0 | T2自匹配/留出 |
| CDEF3 固定融合基线与独立权重验证 | 已实现（smoke，生产未验收） | cde_report.fusion_baselines：同一校准分下equal/sanity/consensus/domain_prior/hard_rule_max AUROC对照，标production_mode，unverified默认排除；验证记录§27 | CDEF2、HARD-02 | P0 | test_cde_report [1/8]；固定FPR召回真实数据未测 |
| CDEF4 软路由准入与维度版本 | 已实现（smoke，生产未验收） | router_admission独立切分对比固定权重→permitted/trace_only/insufficient，写router_gate.admission；_retrain_router按router_slot_names校验n_slots/槽存在，不符不改；主融合仍固定权重 | CDEF6/3 | P1 | test_cde_report [2/8][8/8] |
| CDEF5 可测域差真正接入路由 | 已实现（模拟验证，生产未验收） | cde_report.domain_shift：median_shift_iqr/fpr_inflation→insufficient/shift/normal_drift＋slot_reliability；mechanisms如实标dynamic_routing=not_integrated，未接入路由；V§89 域差定义/尾部FPR膨胀合成复验，与权威AC-022一致，阈值为候选、真实新域缺 | CDEF3/4 | P1 | test_cde_report [3/8]；需新域正常样本 |
| CDEF6 路由熵正则修复与分层重评 | 部分实现 | 2026-10-12：loss改为mil-λH（test_entropy_reg_resists_collapse）；router_admission已用stratified_split分层评估，但仅fit内小样本，U15重评未做 | 独立train/val/test | P1 | T0防坍缩/T2，U15重评 |
| CDEF7 冷启动用途/头训练模式 | 部分实现 | cde_report.cold_start输出COLD_START_ROLES及disc/shead freeze/init_from；跨模式精度对照报告未做 | CDEF2 | P1 | T2三模式对照 |
| CDEF8 槽位能力清单 | 已实现（smoke，生产未验收） | cde_report.slot_health能力表＋运行时ok/reverse/unverified、open不进融合、cal_n/small_sample；GET /models/{cat}/slot-health(engineer)＋学习页「槽位健康与更新审计」Tab | UI-P2.1 | P2 | test_cde_report [4/8]＋API；UI未人工联调 |
| CDEF9 条件在线更新完整独立门禁 | 部分实现 | RegressGate审计条目（anchor_fpr/gray前后、缓存recall/ood_fpr，零额外前向）＋_audit_decide显式结局，持久regress_gate.json（learning_state路径补读），trace.update_log/slot-health可见；E仍直接改工作态，非候选统一验收；V§87 引擎级 legacy_online_guard 默认封禁旧在线即学（模拟验证），每轮独立FPR/召回/G1闭环缺 | SS-A/B/C | P0 | test_cde_report [5-7/8]；T2/T3/T6 |
| CDEF10 文档/实际与候选权重分开 | 已实现（smoke，生产未验收） | _to_result weights取fusion trace effective_weights，extra带candidate_router.applied=False与weights_semantics；无trace回退pipe.weights | CDEF4、UI-P0.5 | P1 | test_cde_report [8/8]；UI联调未做 |
| GHIJK1 分层记忆资产来源 | 已实现（模拟验证，生产未验收） | AC-028（记录§29）：NormalBank/DefectBank 逐条血缘（op/source/时间/版本/hash/image_id），lineage() 标 memory_not_weights，engine 学习态透出摘要，旧对象标 legacy；sem.bank/头/router/校准逐项血缘仍缺；V§91 memory_asset_inventory + test_bank_lineage_incubate 通过（模拟） | B7、SS-B | P1 | T0/T3 |
| GHIJK2 锚定不变/严格成簇/人工gate | 已实现（模拟验证，生产未验收） | AC-029（§29）：候选间相似（cos≥0.90）+窗口内连续命中 k 次才成簇，替代 0.01 分桶；人工样本独立回归门禁及 cluster_sim 真实标定待做；V§91 test_bank_lineage_incubate（lineage、incubation_pool）通过（模拟） | SS-C | P1 | T2/T6长期容量/成簇 |
| GHIJK3 四类反馈/无反馈不自动回流 | 已实现（模拟验证，生产未验收） | AC-030（§30）：entry 带四类 route；unlabeled_reflow 默认关闭（仅 skip/cache/active_queue），开启后须高置信正常+open 域内（None 拒）+严格成簇+熔断/锚定门控；真实数据阈值标定与生产验收待做；V§91 feedback.route_of + test_feedback_routes 通过（模拟） | SS-A/C | P0 | T0/T2无标签不污染 |
| GHIJK4 缺陷近邻快照gate/初始化框库 | 已实现（模拟验证，生产未验收） | AC-031（§30）：缺陷入库后复用 gate 审计做锚定 FPR 门控（fpr_tol=0，超限回滚并记 decision），entry 记 defect_gate/scope(similar_only/incubation_ready)/n_pure_images；init_defect 掩码 patch 预置（开关默认关，整批门控/撤销）；YOLO 框路径、域外 FPR holdout、缺陷侧召回门控仍缺；V§91 feedback.seed_init_defect（独立门控、失败整批撤销）+ test_feedback_routes 通过（模拟） | SS-C、可靠框 | P0 | T2正常锚定/未见缺陷 |
| GHIJK5 孵育调用/纯patch/独立批准 | 已实现（模拟验证，生产未验收） | AC-032（§29）：只用 add_box/add_patch 纯 patch，按独立图计数，holdout separation 批准且不劣于旧头才替换；修复负样本 4D 维度 bug（旧头从未训练成功）；按类型/域计数与域外 FPR holdout 仍缺；V§91 逐条血缘测试 #8 + test_bank_lineage_incubate 通过（模拟）；SYNC-34：按类型计数/孵育已补（HeadManager.heads 按 defect_type，boost 仍只用通用头，类型头仅出 per-type 分数），test_per_type_incubate 7 OK（模拟）；按域计数与域外 FPR holdout 仍缺 | GHIJK4、SS-A/C | P1 | T0/T2/T5样本独立计数 |
| GHIJK6 槽位与孵育头职责 | 已实现（§31） | AC-033：cde_report.SLOT_ROLES（disc/sem/inp 的职责/训练/更新/不承诺）+POST_FUSION_ROLES（defect_knn/incubate 训练/更新/定位/域敏感/不承诺）；slot_health 返回 post_fusion 运行态（in_fusion=False、样本/独立图/trained/separation/最近决策）；disc 如实标注 disc_ft 在线微调；V§91 capability_card 基本实现：缺口为能力卡未接入 API/UI（模拟） | CDEF8 | P2 | 前端展示归入横切 UI |
| GHIJK7 E_open人工难例闭环 | 部分实现（§31） | AC-052 队列/幂等处置/engineer门禁已有；AC-034 补：处置原因归类 cause∈{drift,noise,registration,new_defect,unknown}+矛盾组合 422；new_defect 仅登记 admission 候选（auto_online=False，需 gain/redundancy/regression/g1/g2/version）；hard_pool 按 cause 计数、drift 标漂移；UI 处置框加原因归类。仍缺：候选事件自动生成 revision、处置后效果关联；V§91 基本实现：缺口为 admission_registration_gate 为离线登记函数，未接入 E_open 实际处置流程/API（模拟）；SYNC-34：new_defect 处置已接入自动生成候选 revision draft（同类型归并计数，auto_online=False，仍需人工 admission），GET /review/open_alerts/autoregistered，test_open_autoregister OK（模拟）；处置后效果关联、候选组 extractor 仍缺 | UI-P1.1、B8 | P1 | 告警不自动上线 |
| GHIJK8 全快照/血缘/表示变更隔离 | 已实现（记录§32） | 组件清单/code_sha256/rng/缓存盘点入meta；backbone变更告警且candidate禁混库；组件级回滚待做；V§91 snapshot_completeness 修正 SNAPSHOT_REQUIRED 键名（python_version/torch_version/numpy_version，原先真实快照恒判 incomplete）；基本实现：缺口为快照 feature_revision_binding 仍为 unknown，需在 persist.write_snapshot_state 调 revision_binding（禁改，未做，模拟） | B7、SS-B | P0 | T3继续学习轨迹 |
| GHIJK9 独立发布门控/尾部/累计漂移 | 部分实现（记录§32） | 固定初始尾部+上行累计上限+连续3次冻结/unfreeze已实现并持久化；独立缺陷/域外/G1锚、按更新类型容忍待做 | SS-A/B/C、CDEF2 | P0 | T2/T3/T6慢性漂移 |
| GHIJK10 分机制增益/基线独立验收 | 部分实现（§33） | mechanism_gain_report：四口径+消融CI/副作用分组+PatchCore披露+机制事实标记；真实取分开关/API/生产验收未齐 | GHIJK1–9、§2 | P1 | T2/T6，报告副作用 |

## 4. S2前端契约与工程工作区任务

| 原编号 / 要求 | 状态 | 代码证据与缺口 | 依赖 | 优先级 | 测试条件 |
|---|---|---|---|---|---|
| UI-P0.1 (§11 P0-1) alert/warn统一 | 已实现 | U/pages/dashboard_page.py已有契约修复；test_detection_version IncompleteReportTest通过 | 无 | P0 | T0已有，T4真实聚合联调 |
| UI-P0.2 (§11 P0-2) Gauge禁二次判定 | 已实现 | U/widgets/gauge.py仅绘图几何截断，原始score/threshold保留；is_anomaly只读decision；唯一调用方监控页显式传decision，缺失旧记录不猜测 | UI-P0.3 | P0 | T0/T4离屏契约8项通过（含调用方/历史适配），真实产线联调仍待；验证记录§11 |
| UI-P0.3 三态与open语义 | 已实现 | Gauge及监控badge三态一致，未知/未判定为独立空态；open_alert仅控制开放集徽标，不升级为缺陷，历史条带缺判定显示中性色 | UI-P0.2 | P0 | T4离屏normal/gray/anomaly/缺decision×open独立通过；不是未知处置队列验收，验证记录§11 |
| UI-P0.4 queued/applied/candidate/activated | 部分实现 | auto_activate已生效：service.consolidate(activate=None)读设置中心（默认false→仅候选），routes_feedback不再把None当激活，工单"学习提升"回执区分"(已激活)/(候选,待激活)"；test_consolidate_auto_activate 2通过。外部PCB_Dual不传参→按配置只出候选（行为变化）。反馈仍即学（queued态未拆） | SS-A/B/F | P0 | T3/T4四态不混称 |
| UI-P0.5 检测模型/特征/config版本 | 部分实现 | E _to_result实际version，DS记录，Z生产trad元数据；test_detection_version 11通过，legacy未知保留 | B7 | P0 | T0已有/T3并发发布/T4 |
| UI-P1.1 未知信号明细队列 | 部分实现 | AC-052：反馈页已嵌入OpenAlertQueue，持久列表/详情/处置API复用Detection JSON；过滤/分页/责任人/事件及engineer权限已TEMP验证，UI离屏可提交；真实桌面、噪声漂移分类和学习效果仍缺；V§90 只读 `GET /api/review/open_alerts/summary?category=` 处置汇总接口，UI 面板未接（模拟） | 稳定Detection/资产ID | P1 | T4过滤/责任人/处置 |
| UI-P1.2 attribution/evidence透传 | 部分实现 | H真实mode/series/verdict/next；服务learning_trace及只读details API/实时历史可见UI已接，未知/有限JSON保留；真实handler→engine归因透传通过，当前无独立evidence/confidence/source_slot；已补timeout后最终成功/失败、归因/evidence回写，终态防pending覆盖/重复回调、pending来源事件去重，22项TEMP回归通过（设计建议与验证§21）；生产未验收 | SS-F、区域契约 | P1 | T0/T4有限JSON/未知/失败契约及真实handler→engine归因透传通过；生产未验收 |
| UI-P1.3 检测/反馈/学习/激活四时点 | 部分实现 | Detection/Feedback/Model及trace有基础；AC-055：反馈入队/复核提交/作废三处 OperationLog 已补结构化 extra（含检测时 model_version，缺失为 None 不用当前版本补，V§43，41 tests OK）；候选/批准/激活时点关联及日志与业务同事务仍缺 | SS-B、UI-P0.5 | P1 | T3/T4旧结果不覆盖 |
| UI-P1.4 工单/窗口/延迟/backlog口径 / AC-058 | 部分实现 | routes_detect/stats及U监控存在；overview scope_stats 已补 window_start/denominator=persisted_detections/workorder_id/excludes_archived_workorders/by_source/latency_basis=inference_only、真实 e2e（avg/p95/n_e2e_missing，不再以推理耗时冒充）、receipt_rate=None+unavailable（V§49）；工作台已展示推理/端到端耗时、接收→落库率（不可用）及窗口/分母/工单/来源口径行+tooltip，缺数据显示“—”/“不可用”（V§61，离屏 smoke，实机待测）；接收→落库率/backlog无数据源、stats_page未展示新口径 | 稳定事件时间 | P1 | test_stats_window_basis 2 OK（V§49）；test_ui_stats_caliber 5 OK（V§61）；T4待做 |
| UI-P1.5 after_id/has_more/gap / AC-058 | 部分实现 | routes_detect:311查询has_more；live 游标已修：过滤洞计 filtered_out 不计 gap、仅全表缺失ID计dropped、越界reset_required并UI重订阅（V§47） | UI-P1.4 | P1 | test_live_cursor_gap 3 OK（V§47）；T4待做 |
| UI-P2.1 基础/候选槽编排页 | 部分实现 | factory/configs可构造槽，fast默认三槽；无统一七槽版本健康发布页；V§91 release_ops.orchestration_table 离线编排表（AC-060 未实现→部分实现），未接 API/UI 编排页（模拟） | SS-A/B、CDEF8 | P2 | T4角色/启停diff |
| UI-P2.2 FeatureSetRevision/特征工程 | 部分实现 | 新增不可变manifest/五态注册表及API，F评价绑定revision；模型页显示未知绑定；完整特征工程工作区/Detection关联仍缺（验证§24）；V§88 EvalRun绑定校验、V§89 parity/scheme_compare 协议增量（模拟）；V§91 revision_binding(store, model_snapshot) 离线绑定函数（AC-061），未接 persist 写快照与 API（模拟） | B1/B7 | P2 | T0/T4不可变revision |
| UI-P2.3 评估绑定激活强门禁 | 部分实现 | 显式revision绑定模型API激活前复核独立评价/数值等价/生产实现/模型资产，force不可绕过（验证§24）；旧未绑定路径不变，完整统一批准发布仍缺 | SS-A/B/C、T2协议 | P0 | T3拒绝/过期父版本 |
| UI-P2.4 diff/候选对比/回滚记录 | 部分实现 | E build_variant/rollback存在；activate/rollback 生产入口已禁 allow_stale 绕过（409 STALE_OVERRIDE_FORBIDDEN，V§43）；缺发布事务/diff/批准审计完整链，普通快照激活不强制走 CandidateRelease；V§88 snapshot_prepare/commit 与 release_audit 哈希链（模拟），日志与业务仍不同事务 | SS-A/B、UI-P2.3 | P1 | T3/T4 |
| UI-P2.5 PCB/Core统一资产与事件 | 部分实现 | D服务化HTTP，三记录来源仍在；RTSP临时图资产生命周期待证；V§87 保留期 preview/execute 间接覆盖（TEMP库模拟） | SS-B、资产保留策略 | P1 | T3/T4反馈时原图可重现 |
| LOC-1 (§15.3A/15.4) 来源/坐标/阈值/版本定位契约 / AC-063 | 已实现（模拟验证，生产未验收） | algo/decision/localization.py：finalize_regions统一原图xyxy＋coord_space、source(engine_box/local_branch/heatmap_cc/human_box)、score/thr、postproc版本、status=candidate/confirmed=False；P _finalize输出fused_heatmap+localization，E/DS透传、to_dict落库；heatmap_cc仅visual_only不再覆盖引擎框；验证记录§28；V§46 容器补 coord/image_size/model_version；V§53 二值 mask 无损 PNG 留存（heatmap_postproc.mask_path）+raw_range 归一化基准，孤儿扫描保留 mask；V§54 region_review 查询返回 annotate_review 派生视图：人工确认/add/revise→source=human_box（origin_source 保留候选来源）+reviewer/review_revision/confirmed/train_status（eligible/excluded_candidate/rejected/disputed/visual），trainable_region_boxes 仅 eligible；V§55 schemas.to_dict 透出的 region_review 亦经 annotate_review，复核对话框表格增 训练状态/复核人 列；V§68 原始浮点热图 float16 .npy 留存（模型输出工作网格，pipeline.overlay_save_raw_heatmap 默认开，并行写盘）+heatmap_postproc 记 raw 路径/shape/dtype/缩放关系，recompute_from_postproc 可按新阈值复算（默认参数与在线逐位一致），孤儿扫描与下载白名单（防穿越）覆盖 mask/raw；V§71 region 级训练消费回写：build_feedback_candidate 成功后 CAS 写 region_review.consumption[]（region_id/region_revision/bbox/candidate_id/version/at），annotate_review 派生 used/used_by/stale_after_use/stale_reason，guard 快照框消费后被修订/删除/作废→PARENT_REBUILD_REQUIRED_REGION_STALE（发现：trainable_region_boxes 生产无调用方）；V§74 已补即时学习路径回写（kind=instant）和 UI“学习消费”标签（待同步SYNC-23）；复算仍无 API/UI 入口（AC-063 按权威清单仍部分实现）；V§87 已加复算 API `POST /detections/{id}/heatmap/recompute`（TEMP），UI 未接、实机未验；V§90 复核对话框已接「热图复算(只读预览)」（rel_thresh/min_area/morph_kernel，409/403/410/404 清空预览，不改区域/修订号），权威 AC-063 转已实现，真机未验 | UI-P0.5/B5 | P1 | test_localization 4项；209 OK；test_heatmap_postproc_version 4 OK+test_localization 合计 9 OK（V§53，待同步SYNC-10）；test_region_human_box_status 3 OK、全量 smoke 292 OK（V§54，待同步SYNC-11）；test_region_review_draw 7 OK、全量 smoke 299 OK（V§55，待同步SYNC-12）；test_raw_heatmap_retention 5 OK、全量 smoke 350 OK，raw 524,416B/条、写盘 p50≤1.01ms、e2e on/off 差在噪声内（V§68，待同步SYNC-18）；test_region_used_status 5 OK、全量 smoke 355 项中 354 OK（唯一 error 为并行 delete/undo 改动致 test_review_ops 旧断言失效，非本批；V§71，待同步SYNC-21）；T5真实框未测 |
| LOC-2 (§15.5) 后处理参数/小缺陷/相邻区域 / AC-065 | 已实现（smoke，生产未验收） | postproc_params（mask_thresh/min_area/morph/work_side/top_k取自cfg.hierarchical，内容hash version_id）随结果落盘；select_postproc只允许dev切分、禁止AUC目标；DS可视化heatmap_cc阈值取设置pipeline.overlay_box_rel_thresh（默认0.6），标visual_only不参与定位；V§52：DS热图后处理补版本化元数据heatmap_postproc_meta（algo/threshold_basis=per_image_minmax/min_area_unit=display_px/morph_kernel取pipeline.overlay_box_morph_kernel默认3/version_id=hpp-sha1，display_scale不参与）；无真实标注定位集，未选出冻结版本；V§89 合成标注上冻结 pp-2f54c2e670（dev 选、test 报告，loc 4项OK，模拟验证），真实标注仍缺 | LOC-1、T2划分 | P1 | test_eval_and_select；test_heatmap_postproc_version 2 OK（V§52，待同步SYNC-09）；T5/T1待真实标注 |
| LOC-3 (§15.6) 统一区域证据面板 | 部分实现 | U/image_viewer与monitor有原图/框/热图；单框证据/训练状态不足 | LOC-1 | P2 | T4真实多框联调 |
| LOC-4 (§15.7) 多区域反馈/标注复核 / AC-064 | 已实现（离屏模拟验证，生产未验收；V§85 补 redo/四边拖拽/整框平移，实机待 TEST-04 类） | backend/api/region_review.py：GET/POST /api/detections/{id}/regions，confirm/reject/add/revise，engineer权限、event_id幂等/409、revision乐观锁、revised_from修订链；仅confirmed进入trainable_region_boxes；Feedback.region加box_source=human_box | SS-E、LOC-1 | P1 | test_localization API项（403/409/422/404）+争议隔离（V§48）；UI面板 ui/widgets/region_review_dialog.py + FeedbackDialog“多框复核…”入口，tests/ui/test_region_review_dialog 4 OK（V§51）；V§55：原图叠加各 region 框（按 train_status/source 着色，候选橙虚线/人工可训绿实线/人工未训蓝实线，选中高亮）、拖拽画框→add、选中框重画→revise（带 reason+expected_revision），修正对话框缺 /api 前缀；test_region_review_draw 7 OK、全量 smoke 299 / 93.06s OK（离屏 smoke，实机待 TEST-04 类实测，待同步SYNC-12）；V§71 框级消费追溯 used/stale_after_use 经 API 透出，消费后修订/删除/作废候选发布 fail-closed（REGION_STALE），test_region_used_status 5 OK（待同步SYNC-21）；V§70：角点手柄微调→revise、delete（仅 human_box 软删除，excluded_deleted）、undo（本人最近一步反向修订，他人改过 409），test_region_review_edit 9 OK、全量 smoke 364 / 86.771s OK（离屏，待同步SYNC-20）；V§74：框列表新增“学习消费”列（未用于学习/已用于学习/已用后变更），删除或撤销已消费框后发布阻断有测试，test_stats_null_and_consume 7 OK、全量 smoke 383 OK（离屏，待同步SYNC-23）；真实桌面、拖边/平移、重做待做 |
| LOC-5 (§15.8) 定位指标与版本绑定 / AC-065 | 已实现（模拟验证，生产未验收） | eval_localization：IoU实例召回/精度/误定位率/mean_iou，结果绑定postproc version_id；V§89 合成标注 dev/test 分报（misloc 0.056、test small recall 0.833、后处理 6.35ms），绑定冻结 pp-2f54c2e670；无真实标注定位集与生产评价页 | LOC-1/2、可靠标注 | P1 | test_eval_and_select；T5不用AUROC替代 |
| UI-ARCH (§10/§12/§13) 角色工作区与阶段发布 | 部分实现 | U已有检测/学习/模型/数据页及权限；未知中心/引擎/特征/发布链不全；V§88 STALE_AUTHORIZATION_* 结构化错误码、V§85 三角色越权 401/403 且发布调用 0 次（模拟） | UI-P0/P1/P2 | P2 | T4角色与可操作闭环 |

## 5. S3持续学习任务包及必须单独追踪的子项

| 原编号 / 要求 | 状态 | 代码证据与缺口 | 依赖 | 优先级 | 测试条件 |
|---|---|---|---|---|---|
| SS-A (§8 A) 工作/候选隔离、短发布协调 | 部分实现 | 新增显式 `/feedback/candidate`→service→`learn_candidate` 单事件底座：完整快照重装载、父WAL严格重放、训练锁外执行、生产pipe/current不变；候选路径含候选OS锁、父/WAL/候选hash校验；AC-038（V§34）：service默认candidate_only=True/allow_legacy_online=False，生产入口（api/ui/self_learning）AST守卫禁止legacy在线即学，主干进程共享只读已测；engine.submit_feedback本身无引擎级守卫（bench/m01仍可直改），consolidate仍基于consumed反馈；V§87 已补引擎级守卫 `legacy_online_guard`/LegacyOnlineLearningBlocked，get_engine默认封禁（TEMP模拟，权威AC-038转已实现），AC-039～041仍部分实现 | 无，首要底座 | P0 | TEMP候选8项/37项回归通过；test_work_candidate_isolation_ac038 3项；V§17/§19训练与factory替身边界；真实训练、跨进程和全量验收待验 |
| SS-A.WAL (§4.1/4.3) 候选巩固缓存一致 | 部分实现 | 候选冻结父版本及WAL字节边界，完整重放并在保存前复查current/WAL；父资产缺失、重复冲突、WAL变化均 fail closed；共享资源预算及v1显式学习态/RNG已接入；CPU小头三轮冷恢复续训通过，生产GPU完整流仍待验 | SS-B | P0 | 候选WAL/父版本/失败测试已通过；真实恢复轨迹待验 |
| SS-A.PUBLISH (§4.2) staging/committed/父版本/批准/恢复 | 部分实现 | 候选TEMP staging经真实save/reload后rename为带父版本元数据的候选，不切current；候选恢复校验父current/父快照/WAL/候选hash；activate校验父版本/完成态并对`acceptance=unavailable` fail closed；rename与意图完成记账尚非原子，完整审批、跨进程恢复/独立验收未实现，见V§19 | SS-B、UI-P2.3 | P0 | 候选发布封锁测试通过；不宣称生产发布闭环 |
| SS-B (§8 B) 持久事件、幂等、水位/恢复 | 部分实现；请求幂等小样本已验证 | 普通反馈兼容原路径；候选事件使用独立`candidate:`命名空间和持久意图，重复成功幂等、payload冲突/失败拒绝；receipt/fsync、重复ID/冲突及training/prepared显式recover拒绝已测；候选不置生产consumed；跨文件事务非exactly-once，完整G3崩溃恢复及多进程发布仍开放，见V§16/§17/§19 | SS-A | P0 | TEMP候选8项+既有29项共37项通过；真实训练/崩溃/HTTP鉴权/多进程待验 |
| SS-B.STATE (§4.3) 学习态完整恢复 | 部分实现 | 候选父/WAL/工件完整性缺失即拒绝；新增v1字段契约恢复训练/反馈缓存、计数冷却、门控锚定、局部NumPy/Torch RNG，旧候选明确legacy不可恢复；WAL坏行/单条重放异常进入最近一次进程内审计；CUDA/跨环境/完整业务重放及审计持久化仍开放 | SS-B、B7 | P0 | T3相同后续流轨迹；V§36回放审计回归 |
| SS-C (§8 C) 独立新旧域门控与分层回放 | 部分实现 | H正常/框缓存有上限，_loo_records及anchor保护；独立回放返回replay_run_id、parent_version、WAL watermark、输入/config摘要，CandidateRelease可校验并绑定报告；完整跨域分层训练回放及G1/G2/G3/G6仍缺 | SS-A/B、CDEF2 | P0 | T6/T2退化拒绝；V§37回放证据定向回归 |
| SS-C.BUDGET (§5.3/9.2) 每轮资源上界 | 部分实现 | 候选入口全轮共享events/LOO/search/steps/reloads/文件bytes计数与合作取消，实际配置hash；TEMP验证§22。生产直接反馈未迁移，显存/最大库成本仍无硬上界。2026-10-12：已加墙钟检查点上界 max_wall_ms（默认60s，超限deferred；单步不可抢占，used.wall_ms取末检查点值+elapsed_ms另报），见验证记录“候选墙钟上界”。2026-10-12：巩固重放出锁减少前台阻塞（见HARD-03），不构成预算上界 | SS-A/C | P0 | T6各重路径/T1并发 |
| SS-C.ORDER (§5.1/5.2) 时序/首见/再检/未见分报 | 部分实现 | learning_curve有流和eval；relearn仅predict（不是训练回放）；分组/乱序时序未全验收；AC-043新增回放证据摘要但未持久化 | SS-B、数据manifest | P1 | T2/T6流顺序/迟到标签 |
| SS-D (§8 D/§6) L1伪标签真实训练 | 部分实现 | H无标签缓存/选样不是student损失；无teacher有效期/覆盖替换协议；V§90 label_audit.LabelRecord/resolve_labels 标签身份与覆盖解析（AC-044 未实现→部分实现），未落 DB、未接 student 训练（模拟）；V§95 设置页登记 ac045_l1_real_training_enabled/ac046_l1_pollution_exit_enabled 开关（默认 false，无运行时行为，AC-045/046 仍未实现） | SS-C验收通过 | P2 | T6同人工/训练预算对照 |
| SS-E (§8 E/§7) L2争议隔离/修订/可信重建 | 部分实现 | routes_feedback作废说明已学影响不撤销；无完整基线重建/消费修订 | SS-B/C、历史资产 | P1 | T3/T2错误标签撤销 |
| SS-F (§8 F) 用户可见接收/应用/发布 | 部分实现 | SL/U状态基础＋candidate，但不能证明应用到哪一候选/批准/最终发布 | SS-A–E随交付 | P1 | T4拒绝原因/四时点 |
| SS-STAGE (§2) review_stage独立于scenario | 部分实现 | V§35：`routes_review.py` 复用 `Detection.n_tiles` JSON 保存 review_stage（queued→submitted 最小契约，已提交/未知/显式不匹配返回409，旧记录兼容推断），定向4项OK；L0/L1/L2完整时序、与scenario编号分离的全链协议仍缺（同AC-042）；V§90 label_audit.audit_label_order + service.label_order_audit 只读标签时序审计，未接入候选构建过滤（模拟） | SS-C/D/E协议 | P1 | T0配置兼容/T4 |
| SS-AUDIT (§9.1/9.2) 固定/当前/候选/回放/伪标签消融 | 部分实现 | bench/learning_protocol存在；当前25图只是重路径工程触发，非不退化/半监督增益证明；SYNC-34：伪标签生成（teacher=激活版本，高置信双阈值）+人工优先合并已接入 consolidate 候选路径，只出候选不写库不激活，test_pseudo_label 5 OK（模拟）；伪标签消融增益未测 | SS-C验收、§2 | P1 | T2/T6多域/种子，需授权 |

## 6. S4/S5报告开放项与新硬红线任务

| 原编号 / 要求 | 状态 | 当前证据与缺口 | 依赖 | 优先级 | 测试条件 |
|---|---|---|---|---|---|
| HARD-01 / S4-A1/A2/S5§7-3 2500端到端≤200ms | 部分实现（2026-10-14 V§94：默认路径未达标，GPU 开关路径小样本达标） | S4模型p95=267.3、冷读287.8已超；热129.9口径不可替代。V8图p95=251.322尺寸不是2500验收。追加2026-10-12：gold_finger原生2500图=0；8张真实图TEMP缩放2500、隔离v1、清特征缓存/CUDA同步的Pipeline分段初筛p50=171.223/p95=184.385/max=187.081ms（线性插值），finalize p50=87.230ms最大；非完整DetectionEngine/服务端到端，未含_to_result、HTTP/DB/persist/热图渲染写盘/并发，不宣布200ms通过；详见V§10。续测真实DetectionEngine.predict_image_path及DetectionService.detect_image（热图/持久化/归档/告警/JSON），最终8图p50/p95/max分别197.554/247.830/262.393ms与308.947/428.110/472.379ms；冷载2063.641ms另列。finalize主要为归因future等待+分类器及对位；包含嵌套探针同步开销，HTTP/并发未测；原生2500仍缺失、精度门禁仍失败，非生产通过；可复用tests/bench/hard01_service.py，详见V§13。V§15继续：仅展示图先INTER_AREA缩放再RGB→BGR，2项数组/JPEG/框/归因/SQLite严格等价测试通过；固定8图旧新交替最终引擎p50/p95/max=172.155/188.224/191.893→171.455/185.275/189.024ms，正式服务262.213/434.157/495.527→257.156/402.423/471.613ms，尾部波动不能全归于优化。告警尖峰主要是log_action首次security冷导入（基线211.204ms），未跳过或异步；future等待约35ms仍保留。非原生/精度未复验/HTTP并发未测，服务200ms仍失败，状态不变。V§18轻量审计上下文拆分：security与audit共享同一ContextVar，服务显式传角色调用原DB同步事务，未改鉴权/字段/权限；新增5项审计+2项展示契约通过。两轮旧→新完整服务p95=437.831→327.246、452.654→316.123ms；首次告警229.940/219.878→1.934/1.928ms，新进程未导入security，仍真实提交并同字段审计。8张TEMP工程2500/清特征缓存/GPU同步，HTTP/原生/并发及精度正式门禁仍缺，p50也>200ms，HARD-01状态不变。V§23完整入口工程测量：直接空闲热P50/P95/P99/max=290.615/341.356/360.403/365.675ms，HTTP空闲=280.749/319.068/323.102/323.895ms；冷首请求2658.912/2201.558ms；工程学习竞争直接热P95=2203.575ms、HTTP=2202.696ms，重叠子集P95=3357.602/3519.675ms，均为非原生工程图且小样本；G1/AC-066仍未通过。追加2026-10-12（G1服务路径）：cv2解码+展示/热图并行imwrite（返回前join）+预取前移+向量化森林，7+3项等价测试通过；8张工程2500完整服务p50/p95/max=184.55/190.11/190.51ms，engine_path p95=129.79ms，8图决策与prior一致（aoi_hard01_service_zoivzjed/result.json）。工程口径首次≤200ms；原生2500、HTTP/并发、候选学习竞争下前台p95与G2精度未正式验收，生产框不勾。追加2026-10-13（V§69下采样解码试验，不采用，零生产改动）：PNG2500 REDUCED/area 解码无收益（95→94~99ms），e2e area1024 P95 235.1 vs full 221.6更慢；JPG4096×3000 reduced4 e2e P95 156.9→115.8ms，但34张精度集框IoU≈0.2、判定翻转4~6/34、good误报增加，重新prepare丢检，违反精度不退化；smoke 364/OK（review/task/test01_e2e_latency/dev_downsample/）。追加2026-10-13（V§76/SYNC-24 渲染异步）：异常图overlay/热图/mask/raw npy提交_io_exec，/api/file等待在途产物（超时503/失败500）；render P95 34→19ms，空闲e2e P95=180.6ms，负载下190～225ms不稳定，逐图精度一致，smoke 383 OK；剩余瓶颈decode（PNG inflate≈100ms）与infer抖动，需输入格式改造或SS-A算力隔离（review/task/test01_e2e_latency/dev_async/）。追加2026-10-13（V§77/SYNC-25 流/视频帧内存直检）：新增detect_ndarray，RTSP/视频抽帧不再JPEG往返，单帧P50/P95 135.8/166.6→65.4/123.3ms（n=24，不含热图）；1帧判定差异归因于旧路径JPEG重编码，同像素两入口逐帧一致；内存帧热图文件名净化；smoke 389 OK（review/task/test01_e2e_latency/dev_frame/）。追加2026-10-13（V§78/SYNC-26 学习锁内持有缩短）：RegressGate check→recenter锚定分按状态指纹复用（失配重算、回滚作废），accepted事件锁内hold 963.7→657.4ms（-32%），前台学习中wall P50/P95 18.7/34.8→17.7/29.8ms、max≈1000→≈720ms，精度逐图差0.0；rolled_back路径check≈600ms仍在锁内（锁外影子deepcopy更贵/check写gate状态，未拆）；smoke 395 OK（review/task/test02_lock_contention/dev_learnlock/）。追加2026-10-14（V§94/§97，待同步SYNC-30）：blob CPU 无损改写，gold_finger P95 397.7→316.8ms（TEMP，仍超红线）；GPU DoG 开关 181.8ms（26/26 判定一致，Δscore 0），默认关，设置页决策键 blob_gpu_enabled，env AOI_BLOB_GPU=0 强制关；需全量集复核 AUROC（U22 曾 0.7075→0.6112）；原生 2500/并发未测。生产验收0 | §2、SS-A算力隔离 | P0 | T1原生2500、逐图、并发 |
| HARD-02 / S4-B4/B5 精度不退化及固定FPR召回 | 部分实现（门禁已实现，精度未达标） | V20图fused AUC=0与HEAD相等；域差失败不能称精度达标；V§88 G2 门禁 fail-closed 错误码（RUN_MISSING/BINDING_MISMATCH/PARENT_STALE/POLICY_CHANGED）+Wilson 区间，9项OK（权威AC-067转已实现）；门禁只保证不放行，V§81 真实 FPR 100% 未改善；V§95 冻结后改阈值/gt_dataset_ref 返回 409 THRESHOLD_FROZEN；V§96 `_soft_bound` 替换 2.0 硬钳位，fused AUROC 0.935→0.970、FPR≤5% 最大 recall 0→0.80，固定阈值 FPR 仍 1.00（根因 test 正常样本批次 Image_2026/t_fov 未覆盖，blocked on data；sem 方向反转、sanity 置零 shead 未修）；V§97 评估输出 batch_coverage 告警与 budget_working_point 参考工作点（不改判定），approved_extra_good_batches 仅登记需人工重建（均待同步 SYNC-30，模拟） | CDEF2/3、数据预算确定 | P0 | T2不改方向/门控 |
| HARD-03 / S4-C1/E6 学习耗时有界不阻塞 | 部分实现 | 2026-10-13 与权威清单 AC-068（HARD-03 映射）一致改为部分实现，依据：候选全轮事件/LOO/search/步数/重载/文件bytes预算与墙钟检查点上界 max_wall_ms、deferred/cancelled 持久证据、巩固重放出锁（见下）；真实墙钟/显存/最大库上界仍未保证。S4单轮小时增长；V25图11.67s且未测所有支路，3000ms非上界。追加2026-10-12：consolidate(activate=False)的load_snapshot/_replay_journal移出品类锁，出锁后以缓存身份+版本+WAL(size,mtime_ns)复查才发布；backbone sha256按(path,size,mtime)缓存；新增4项测试，72 tests OK。续：save_snapshot拆为write(锁内约225ms)/finalize(锁外哈希约310ms)/set_version(写锁约8ms)；consolidate写私有.consolidate- stage后rename；候选父快照哈希改为锁外预热。native_fullchain rounds=32复测：锁等待max 454.7→160.3ms(consolidate)，候选锁等待≤5.5ms，overlap P95 319.0，idle P95 236.3，candidate p50 6098ms，262行决策一致，smoke 137 OK。剩余：锁内序列化约160ms、.candidate-*残留需对照candidate_events（.consolidate-*陈旧>10min已自动清理；学习墙钟已有检查点上界max_wall_ms，smoke 140 OK）。追加：新增 self_learning.learning_parallel（设置中心可改，默认 false＝学习在检查点让行前台推理，空闲300ms后继续，让行时长不计入墙钟预算）；串行复测 hot P95 236.7（≈无学习 idle 236.3，并行 overlap 为 319.0），单候选 4.0–4.6s，但学习完成随产线负载延后（本测推理结束后再跑36.4s）；smoke 142 OK。2026-10-12：崩溃残留 `.candidate-*` 在持品类OS锁后按 prepared 引用清理（_sweep_candidate_stages）；smoke 145 OK | SS-A/C.BUDGET | P0 | T6/T1最大预算路径 |
| REPORT-1 / S5§7-1 EXPs缺权重回退 | 已实现 | EXPs/src/backbone/dino.py:39–49已2026-10-11回退，与旧对比报告不同；离线自包含权重仍不足 | 权重manifest | P1 | T0干净机器无缓存/断网 |
| REPORT-2 / S5§7-1 权重获取/打包承诺 | 部分实现 | assets/checksums.sha256及Core回退；不恢复用户删除权重/演示快照，交付承诺需对齐 | REPORT-1 | P1 | T3干净离线部署 |
| REPORT-3 / S5§7-2 旧/新学习双轨勘误 | 已实现（AOI_tree 内文档） | 已在 EXPs/docs/算法引擎设计.md §7.3 加勘误，更正“全程异步不阻塞推理流”的说法，写明锁内序列化、learning_parallel 默认串行、并行 319ms 是全链路实测、G1 只约束纯推理折算。commit/repo 下的 S4/S5 原始报告不在改进范围内，没有修改；AOI_Core/docs/old 已过时，不维护 | SS-A、HARD-01/03 | P0 | 文档事实复核/T6 |
| REPORT-4 / S5§7-3 视频/多品类多种子 | 未实现 | 历史研究/复现报告不替代当前门禁；不按本轮授权跑全量 | SS-C、授权少样本筛选后 | P1 | T1/T2/T6标准档授权 |
| REPORT-5 / S5§7-4 泄漏幅度同流量化 | 未实现 | 旧/新双轨协议不同不能直接相减；未执行同流诚实对照 | 隔离旧代码＋固定数据 | P2 | T2小样本不污染线上 |
| REPORT-6 / S5§7-5 PCB服务部署前提 | 部分实现 | D feature_client/launcher存在，文档需与实际离线/服务故障行为一致 | UI-P2.5 | P2 | T3服务不可用/超时 |
| REPORT-7 / S5§7-6 运行残留打包清理 | 部分实现 | tools/cleanup_data.py存在；本轮未删除生产storage，需保留资产/版本血缘 | 生命周期批准 | P2 | T3预览/引用完整性 |
| REPORT-8 / S4-E8/layout恢复 | 部分实现 | Z已保存7拟合量并拒缺失历史layout；V§5测试证据，旧gold_finger/v2不可可靠加载 | 原训练数据可追溯 | P1 | T0合法往返/非法拒绝 |
| REPORT-9 / S4依赖/sklearn兼容 | 部分实现 | requirements-lock存在；当前pickle1.9.1/环境1.9.0仍警告；V§95 新增 deplock 依赖锁检查（dependency_lock_policy 默认 warn 附 dependency_lock_warning，block 时激活返回 409 DEPENDENCY_LOCK_MISMATCH），锁文件本身未修（待同步 SYNC-30，模拟） | 环境/资产冻结 | P1 | T0/T2固定依赖再测 |
| REPORT-10 / V§5 m14夹具语义及有限JSON | 小样本已验证，生产风险开放 | 历史18项因跳过detect/open_alert不代表输出契约通过。现已定位单缺陷候选路由样本标准差自由度不足；返回None及availability，训练非有限fail closed，检测严格JSON在落库前拒绝非法值。恢复M14真实请求，合成数据22项通过；5项定向单测通过，详见V§14 | 历史污染快照/记录未迁移；真实数据及线上重训仍需专项验证，禁止测试泄漏 | P1 | T0单/多缺陷、非法数值、严格JSON；T2真实数据 |

## 7. 候选/WAL专项审计结论与本轮交付

### 7.1 确认风险

`consolidate(activate=False)`之前仅load父快照并cache，保留WAL但不重放。因此已消费反馈在热缓存中暂时消失，而`_ensure_loaded`在LRU/重启会重放同一WAL恢复：同一current版本的预测随缓存生命周期变化。原测试只检查current不写/缓存版本号为3，没有检查恢复反馈。

本轮采取兼容现有在线语义的最小修复：load父快照后调用既有`_replay_journal(category, active_pipe, active)`，完成再cache。只重放匹配base版本且原图存在的日志；不追加/清空WAL，不切换current，不修改算法/门控；读取日志异常不发布半恢复cache。**该修复是缓存恢复一致性，不是候选训练隔离：当前反馈仍改变工作态，重放后也会恢复这些变化。** 没有把candidate独立训练或工作只读宣称已完成。

### 7.2 仍开放的WAL/发布风险（SS-A/B继续）

- 日志仍在学习之后写，不是预写事务；本轮SS-B已添加WAL fsync、写失败向上传递及持久完成receipt去重，但学习完成→WAL/receipt提交之间仍非原子，部分学习异常也不回滚。业务DB未consumed可同ID重试，在线工作态可能已改变；不得宣称exactly-once（V§16）。
- 重放按base过滤，原图缺失/算法失败可跳过；反馈后阈值/缓存/RNG恢复不完整时不保证学习轨迹一致。日志量未限，重放持锁/读全文件，可能突破恢复/前台预算。
- 候选训练仍在激活pipe执行；candidate保存可能包含早已生效的反馈。auto_activate默认行为仍保留；管理员activate不强制独立报告/批准/父版本一致。
- activate与feedback/consolidate不共享完整发布协调，内存先于current；候选恢复load失败时已保存候选可能成为未登记孤儿，管理锁内重放仍可长阻塞。
- 要真正隔离需SS-A独立候选、SS-B事件账务、SS-C独立门禁一起交付，不能简单删除WAL或忽略反馈来“保持旧分”。

### 7.3 实际修改与测试证据

修改：E候选分支增加重放；`AOI_Core/tests/smoke/test_candidate_isolation.py`扩为4项：候选父缓存/WAL调用、真实TEMP日志匹配base且不变、重放异常不替换缓存、activate=True原提交语义保持。

命令（cwd=`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
py -3.11 -u -m unittest tests.smoke.test_candidate_isolation tests.smoke.test_detection_version tests.smoke.test_feedback_ft_equivalence -v
```

结果：exit 0，19 tests，3.917s，全部通过（candidate 4＋version 11＋feedback equivalence 4）。真实WAL文件测试只用TEMP自建日志/资产，feedback是替身；版本测试包含实际save/load轻量对象，微调测试验证原数学/梯度/门控等价。未加载真实DINO/跑全量精度/2500时延，不能提升完整建议到已验收。sklearn1.9.1→1.9.0兼容警告仍在。

### 7.4 下一轮顺序

1. P0 SS-A/B：先定义工作态只读、事件候选归属/消费水位，再构造独立候选；并发/fault小样本证明拒绝不影响工作态。
2. P0 HARD-01/02：冻结数据/业务FPR协议，原生2500真实少样本分段初筛；精度方向仅由训练/开发集选择。当前失败基线先诊断，不放宽标准。
3. P0 SS-C.BUDGET：限制重路径规模/LOO/search/步数和墙钟，安全验证重复LOO复用的状态失效（库/头/校准/权重任何变化都不能复用过期评分），不删独立门控来提速。
4. 独立可并行小项：Gauge后端decision契约、m14合法夹具、attribution透传；各有测试后更新本清单。熵正则更正后须重评路由，不能立刻接管主融合。

每轮记录命令、样本/版本hash、耗时、失败/警告、未覆盖路径；长命令持续打印进度。当前结论：**硬红线、独立精度、学习全局上界、完整候选发布均未达标。**

## 8. FeatureSetRevision 最小切片实际交付（2026-10-12）

新增Core内容寻址manifest/五态注册表与API，AOI_feature评价绑定冻结血缘，显式revision模型激活前复核独立评价/等价/当前生产hash/资产且不可force，模型页未知绑定不猜测。active不是生产指针，旧生产未绑定流程不变；无自动上线、无生产trad源码更改，不改E_open/候选发布核心，不提交git。

8项新标准库TEMP测试＋11项既有metadata/版本回归通过；实际FeatureSet manifest为15组200维。命令、结果、失败及边界见验证记录§24，设计发现见S2§22。完整run_eval/真实生产等价与独立评估/HTTP鉴权/工作区和Detection/可恢复发布仍缺；AC-061仅部分实现，AC-010/062仍未实现完整生产要求，全部生产框不勾。

## 9. AC-050 候选事件归属修复（2026-10-13，生产未验收）

- `CandidateRelease.status` 按 `candidate_id` 过滤，避免不同候选状态串线；历史缺 `candidate_id` 的记录仅按 `payload.candidate_event` 精确匹配，无法确认时标记 `orphan/unavailable`，不猜测归属。
- 新增候选串线和历史缺字段测试：**41 passed、77 warnings、19 subtests**。证据仅覆盖 TEMP/小样本归属隔离与 fail-closed，不改变 AC-050 部分实现状态。
- 生命周期时间线、水位、结构化失败原因、取消续接仍未解决；生产验收仍为 **0**。未修改生产数据库/快照，未提交 git。

## 10. AC-048 CandidateRelease 结构化撤错前置守卫（2026-10-13，生产未验收）

- `CandidateRelease` 增加结构化 `reason_code`；作废或人工修订候选在 validate/decide/activate 前 fail-closed，候选标记为 `blocked`，要求从可信 parent 重建。
- 定向 **45 tests / 25.369s / OK**，`py_compile` 通过；未改生产版本。完整撤错重建、参考标签重算、特征库头 CDF 阈值重放仍未实现，AC-048 为部分实现，生产验收仍为 **0**。

## 10. AC-049/050 配对评估与结构化候选原因（2026-10-13，生产未验收）

- AC-049 新增 `backend/evaluation/paired.py` 的 `paired_prediction_delta`，按同 seed/完整样本 ID 集合计算 candidate-baseline 的 AUROC/FPR/recall/gray delta；缺种子、ID重复/缺失、标签/样本集不一致、NaN/Inf、单类别返回 `unavailable`，不虚构CI。新增5项测试 **1.379s通过**。
- AC-050 扩展 `CandidateRelease` 的 `reason_code`/`violations`/`retryable`；验收报告、构建失败/预算延期写入结构化原因，`learning_trace` 保留原因。合并定向 **52 tests / 22.319s / OK**（命令总26.607s）。
- 尚未实现六臂同预算训练、配对CI、多流多种子、完整生命周期时间线/水位/取消续接；AC-049/050仍为部分实现，生产验收仍为 **0**，不得表述为生产通过。

## 11. AC-076/073 禁止 allow_stale 绕过 + AC-055 反馈审计结构化关联（2026-10-13，生产未验收）

- 发现：activate/rollback 生产入口曾接受并转发 `allow_stale=true`，管理员可绕过 stale 门禁切生产指针。修复：`routes_models.py` 的 `_reject_stale_override` 对显式 true 返回 409 `STALE_OVERRIDE_FORBIDDEN`/retryable=False，生产入口固定 `allow_stale=False`，`RollbackRequest` 字段保留兼容；engine/service 签名与离线调用未改。**9 tests / 4.1s / OK**。
- AC-055：三处反馈 OperationLog 补结构化 extra，action/detail/返回值不变；**41 tests / 3.7s / OK**（新增4项，TEMP SQLite 真实落库）。
- 全量 smoke：275 tests / 77.150s / OK。AC-055/073/074/076 仍部分实现，生产验收 **0**；未改生产 DB/快照，未提交 git。详见验证记录§43。

## 12. Core 特征工程页契约与 AOI_feature parity（2026-10-14，生产未验收）

- 新增 `AOI_Core/tests/smoke/test_feature_parity.py`：仅使用 3 张确定性内存代表图，不读取全量数据；校验 15 组顺序、注册切片、200 维、有限值及默认配置逐图数值等价。首次测试发现常量图上 AOI_feature `selfsim` 产生 NaN，已在 `FeatureSet.extract_one` 增加与 Core 相同的 `nan/inf -> 0` 边界归一化。
- parity 定向 **3 tests / 0.810s / OK**。另明确记录：256×256 大图触发 selfsim 抽样时，AOI_feature 使用全局 NumPy RNG、Core 使用 `default_rng(42)`，当前契约测试将其作为已知差异边界，未冒充逐位等价；候选配置仍不得直接进入生产。
- Core UI 契约补充：特征工程页在 `NAV_ITEMS` 中；engineer/admin 可访问，operator 不可访问；离屏全页冒烟改为动态校验页面总数且要求当前 11 页全部切换。定向合并 **12 tests / 8.521s / OK**；UI 离屏冒烟 **PASS**。出现 Qt 字体目录 warning 和第三方弃用 warning，但无失败。
- 本轮已完成 AOI_feature/Core selfsim 随机抽样统一：两端使用显式 `default_rng(42)`，实验提取不污染全局 RNG；大图代表图 parity 纳入测试。此前 256×256 随机差异边界已关闭，但仍仅是少量内存图契约，不等于生产验收。
- 定向合并测试 **12 tests / 8.922s / OK**；测试期间仅有 Qt 字体目录、第三方弃用和常量图数值 warning，无失败。
- 当前仍未完成：真实 HTTP/桌面/GPU、生产 feature revision pointer 与 Detection 血缘闭环、G1 2500×2500 完整 e2e p95≤200ms、G2 正式业务精度、生产验收仍为 **0**。未修改生产快照/数据库，未提交 git。
- **2026-10-14 发布剩余验收认领池**：已在 `review/协同.md` 追加 REMAIN-01～REMAIN-14，覆盖 AOI_feature/Core 全链路 parity、真实 HTTP/API/桌面、生产 feature revision 与 Detection 血缘、G1/G2/G3/G5/G6、离线权重、PCB_Dual 缺口及 AC-001～080 总审计。任务仅发布，未认领、未执行、未授权同步；既有 TEST-01～09 和 PCB-ACCEPT-01～05 结论保持不变。要求少量固定种子样本、TEMP 隔离、长命令进度、result.md/env.json/cmd.txt/raw 工件，生产验收仍为 **0**。

- 2026-10-14 SYNC-32（V§106）：L3 配对模板匹配，`pair_mode` 默认关闭。gold_finger 缺陷图配自身修复图时 tpl AUROC 0.4975→1.0，与 demo4 一致；但跨图或重拍口径下不成立，固定阈值未达标。全量 smoke **553 tests / 138.283s / OK**。计数不变（24/54/2），HARD-02 维持部分实现，生产验收 0。
