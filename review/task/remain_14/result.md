# REMAIN-14 只读缺口审计：AC-001～080 + G1～G6 证据矩阵（草稿）

- 审计日期：2026-10-14
- 审计方：验收审计 agent（REMAIN-14 认领，协同.md「剩余验收任务认领池」）
- 性质：**只读草稿**。未修改任何生产代码、未运行任何测试、未改动台账/协同.md/验证记录原文。REMAIN-01～13 未完成项一律标「待复核/缺口」，无证据不勾完成；估算口径不写成实测；生产验收框 `[ ]` 仍全部未勾（只有用户能勾）。
- 骨架来源（唯一权威）：`d:\CGAIC\commit\repo\review\简明改进建议.md`（196 行，AC 编号/主题/实现状态列以其为准）。
- 证据来源：V=`d:\CGAIC\AOI_SSOCL\review\AOI_tree首批改进与验证记录.md`（2319 行，§1～§104）；T=`d:\CGAIC\AOI_SSOCL\review\AOI_tree全量改进任务清单.md`（312 行）；协同=`d:\CGAIC\AOI_SSOCL\review\协同.md`（1002 行）；task 产物目录 `d:\CGAIC\AOI_SSOCL\review\task\` 下各 result.md。
- 口径：「模拟待实机」=已有 TEMP/合成/离屏模拟或定向验证但缺真实环境验收；「待复核」=有交付/验收记录但独立复核或收尾子项未完成；「真实验收状态」中的 PASS/FAIL 均指定向/实测任务结论，不等于生产验收通过。

---

## 第一部分：80 行证据索引

| AC | 主题 | 实现状态 | 定向证据 | 真实验收状态 | 缺口/备注 |
|---|---|---|---|---|---|
| AC-001 | 分辨率选择 | 部分实现 | V§89 scheme_compare 协议（合成）；task\final_model | 模拟待实机 | 真实品类档案（最小缺陷像素/对比度/域差）仍缺 |
| AC-002 | 全局上下文 | 部分实现 | V§89 三方案对照含整图专家（合成） | 模拟待实机 | 真实结构/布局/大范围异常独立验证缺 |
| AC-003 | 单级局部覆盖 | 部分实现 | V§89 确定性覆盖测试（合成） | 模拟待实机 | 真实 ROI 覆盖目标缺陷未验 |
| AC-004 | 块与坐标 | 已实现 | V§89 随机尺寸/末行末列回移/跨块并集/原图反解（合成） | 模拟待实机 | 生产验收 0 |
| AC-005 | 粒度与标签 | 部分实现 | V§89 面积占比标签逻辑（合成） | 模拟待实机 | 训练/建库/校准/推理全链视图一致未验 |
| AC-006 | 块图误报预算 | 部分实现 | V§89 块/图两层校准+Šidák，4000 合成图 | 模拟待实机 | 真实分支预算冻结缺（依赖用户冻结 G2 数值） |
| AC-007 | 预算细判 | 部分实现 | V§86(final_perf) upload p50/p95 1086.7/1971.0→513.9/894.3ms；task\final_perf | FAIL | 纯推理 p95 428.6ms，200ms 本机不可达；未完整 ≤200ms |
| AC-008 | 细判副作用 | 已实现 | V§89 分报追加检出/误报确认/否决新增漏检（离线合成） | 模拟待实机 | 生产验收 0 |
| AC-009 | 三方案验收 | 部分实现 | V§89 三方案同数据对照协议（块数 0/24/6，合成） | 模拟待实机 | 真实 G1/G2 对照缺；旧 1 秒口径已废止 |
| AC-010 | 统一注册表 | 部分实现 | V§24 manifest 底座；V§89 parity harness 组级切片 | 模拟待实机 | 生产 trad 与评价共用实现未统一，不靠截维管理未达 |
| AC-011 | 特征启用 | 部分实现 | V§91 group_selection_plan（模拟） | 模拟待实机 | 未接入 UI/API；组级生产选择缺 |
| AC-012 | 贡献档案 | 部分实现 | V§91 profile_completeness/report_status（模拟） | 模拟待实机 | stamp_profile_lineage 未自动调用；强血缘/CI 不足 |
| AC-013 | 独立方向与归一 | 已实现 | V§91 trad_protocol_audit（文档化审计，模拟） | 模拟待实机 | 生产 trad 打分协议未改 |
| AC-014 | 单图与逐框解释 | 部分实现 | V§20 反馈归因持久契约 4 项契约/14 组合；V§91 box_evidence/box_attribution_audit | 模拟待实机 | dominant_slot 仍整图复制；审计结果未在 UI 展示；逐框/Trad 参考未齐 |
| AC-015 | 特征版本血缘 | 部分实现 | V§24 最小 revision/证据门禁；V§91 representation_change_plan | 模拟待实机 | 表示变更重提/重建/隔离只生成计划不执行 |
| AC-016 | 特征操作闭环 | 部分实现 | V§91 feature_ops_view/submit_toggle（模拟） | 模拟待实机 | 未接入 API/UI 权限体系；发布入口缺 |
| AC-017 | 难例特征准入 | 部分实现 | V§26 难例候选准入；V§89 062 等价证据+010 激活门禁（模拟） | 模拟待实机 | 真实难例池试算与独立增益/冗余评估缺 |
| AC-018 | CDF契约 | 已实现 | V§89 small_sample 标注/bootstrap 尾部稳定性/self_match_bias（合成） | 模拟待实机 | 真实品类尾部下限待真实正常集 |
| AC-019 | 校准隔离 | 部分实现 | V§87 holdout 开关；V§90 默认 auto（fit≥16 走 holdout 0.2）；V§95 设置页 calibration_isolation | 模拟待实机 | 真实数据 holdout vs self 精度/阈值对比缺；已有快照需重建才生效 |
| AC-020 | 固定融合基线 | 已实现 | V§89 sanity≤0.5 槽不进基线（合成）；V§98 domain 下 shead 置零诊断（非 bug） | 模拟待实机 | 独立集准入结果待真实数据；孪生图配对数据协议待改 |
| AC-021 | 软路由准入 | 已实现 | V§89 准入逻辑（分层三集/trace_only/n_params=k²+k，合成） | 模拟待实机 | 真实数据无 permitted 证据；主融合未接管 |
| AC-022 | 域差能力 | 已实现 | V§89 域差指标定义（normal_drift/shift、尾部 FPR 膨胀，合成） | 模拟待实机 | 阈值为候选、真实新域缺 |
| AC-023 | 路由训练修正 | 部分实现 | V§89 stratified_split/overfit_gap/collapsed 报告（合成） | 模拟待实机 | U15 真实数据 mil−λH 重评未做 |
| AC-024 | 冷启动训练角色 | 部分实现 | 无近期定向证据（engine_fast 冻结头有） | 待复核 | 头冻结预训练/微调/从零各配置独立验缺 |
| AC-025 | 槽位能力清单 | 部分实现 | 无近期定向证据（V§31/§33 相关） | 待复核 | 能力健康页缺；inp/open 说明未齐 |
| AC-026 | 更新准入 | 部分实现 | V§87 legacy_online_guard 默认封禁旧在线即学（模拟） | 模拟待实机 | 每轮快照/前后独立 FPR/召回/G1 闭环缺 |
| AC-027 | 实际/候选权重语义 | 部分实现 | V§20/§21 归因透传部分覆盖 | 待复核 | 全前端联调缺 |
| AC-028 | 记忆资产分层 | 已实现 | V§91 memory_asset_inventory + test_bank_lineage_incubate（模拟） | 模拟待实机 | 条目血缘缺；生产验收 0 |
| AC-029 | 正常入库与成簇 | 已实现 | V§29 严格成簇；V§91 bank lineage（模拟） | 模拟待实机 | 严格聚类门禁缺；人工样本独立回归缺 |
| AC-030 | 四类反馈 | 已实现 | V§30 四类分路；V§91 feedback.route_of（模拟） | 模拟待实机 | 自动回流高置信/一致性/域差/锚定门控未闭环 |
| AC-031 | 缺陷近邻止血 | 已实现 | V§30 锚定/FPR 门控；V§91 seed_init_defect 失败整批撤销（模拟） | 模拟待实机 | 初始化和独立门禁待真实证据 |
| AC-032 | 孵育头 | 已实现 | V§29 纯 patch/独立批准；V§91 血缘测试 #8（模拟） | 模拟待实机 | 独立与锚定通过再替换未真实验 |
| AC-033 | 槽与新头职责 | 部分实现 | V§31；V§91 capability_card（模拟） | 模拟待实机 | capability_card 未接入 UI |
| AC-034 | E_open难例学习 | 部分实现 | V§31 原因归类；V§91 admission_registration_gate（模拟） | 模拟待实机 | 未接入 E_open 实际流程；独立增益/回归/版本链缺 |
| AC-035 | 完整快照 | 部分实现 | V§32 快照组件清单；V§91 snapshot_completeness（feature_revision_binding 仍 unknown） | 模拟待实机 | 缓存/RNG/hash 盘点不全 |
| AC-036 | 发布/漂移回滚 | 部分实现 | V§32 固定初始尾部+累计上限+连续违规冻结；V§85 合成复验 | 模拟待实机 | 独立缺陷/域外/G1 锚与按更新类型容忍缺 |
| AC-037 | 分机制收益 | 部分实现 | V§33 mechanism_gain_report 四口径/消融 CI/披露；V§85 复验 | 模拟待实机 | 真实取分开关与 API 缺 |
| AC-038 | 工作/候选隔离 | 已实现 | V§34 隔离守卫+AST 守卫；V§87 legacy_online_guard（TEMP 模拟） | 模拟待实机 | 真实训练并发拒绝不改工作态未在真实环境验；生产验收 0 |
| AC-039 | 事件与水位 | 部分实现 | V§16/17/19 receipt/fsync/幂等；V§36 WAL 回放审计；V§85 复验 | 模拟待实机 | 跨文件事务非 exactly-once；完整 G3 与审计持久化缺 |
| AC-040 | 可恢复短锁发布 | 部分实现 | V§19/§38 只读发现 39 tests OK；V§88 snapshot_prepare/commit+release_audit 哈希链（模拟） | 模拟待实机 | 跨介质非原子；业务 Model 镜像/自动启动恢复/多进程/真实数据与鉴权缺 |
| AC-041 | 学习态/RNG恢复 | 部分实现 | V§38 RNG 严格预验证 26 tests OK（TEMP/CPU 小头契约） | 模拟待实机 | 真实 GPU、跨环境/跨版本、完整回放轨迹一致性缺 |
| AC-042 | L0时序和数据身份 | 部分实现 | V§35 n_tiles 最小契约；V§90 audit_label_order/label_order_audit（模拟） | 模拟待实机 | 未接候选构建前自动过滤及 UI/API；批次近重复隔离缺 |
| AC-043 | 独立回放门控 | 部分实现 | V§37 回放证据绑定（replay_run_id/父版本/WAL 水位）；V§85 复验 | 模拟待实机 | 跨域分层独立训练回放及 G1/G2/G3/G6 缺 |
| AC-044 | L1标签身份 | 部分实现 | V§90 LabelRecord/resolve_labels（模拟） | 模拟待实机 | 未落 DB 表、未接训练样本组装 |
| AC-045 | L1真实训练 | 未实现 | V§95 仅设置页登记开关 ac045_l1_real_training_enabled（默认 false，无运行时行为） | 缺口（开发前置） | teacher/student 真损失训练能力不存在；保持未实现直到有交付证据 |
| AC-046 | L1污染与退出 | 未实现 | V§95 仅登记开关 ac046_l1_pollution_exit_enabled（默认 false） | 缺口（开发前置） | 同预算研究门禁未执行；能力未实现 |
| AC-047 | L2争议隔离 | 部分实现 | V§40 candidate_guard 最小修复，候选发布定向 45 项通过 | 模拟待实机 | 完整争议修订链、独立参考标签、撤错可信重建缺 |
| AC-048 | 撤错可信重建 | 部分实现 | V§41 结构化 reason_code+fail-closed，定向 45 tests / 25.369s OK | 模拟待实机 | 完整撤错重建、参考标签重算、库/头/CDF/阈值重放未实现 |
| AC-049 | 六臂同预算评估 | 部分实现 | V§42 paired_prediction_delta，5 项测试 1.379s 通过 | 模拟待实机 | 完整独立六臂、配对 CI、多流多种子未验 |
| AC-050 | 学习可见状态 | 部分实现 | V§39/§42 候选事件归属修复，合并定向 52 tests / 22.319s OK | 模拟待实机 | 生命周期时间线、水位、取消续接缺 |
| AC-051 | alert/warn契约 | 已实现 | V§84 TEST-04 真实桌面：open 告警横幅/列表/详情与 API 逐字段一致；task\test04_desktop_ui | PASS（桌面覆盖场景，待复核） | 产线联调缺；alert/warn 同屏排序未覆盖；生产框未勾 |
| AC-052 | 未知信号中心 | 部分实现 | V§79(E_open队列续验) 21 项组合通过；V§90 open_alerts/summary（TEMP） | 模拟待实机 | 缺 UI 面板及与候选发布结果回链；处置未训练、真实桌面缺 |
| AC-053 | 检测版本事实 | 部分实现 | V§84 TEST-04 附带验证 legacy_unknown→「旧证据未知」保留 | 部分 PASS（待复核） | 全血缘未全；并发切版不重读 current 未验 |
| AC-054 | 归因证据透传 | 部分实现 | V§20 契约 14 组合；V§21 异步反馈终态回写 22 项 TEMP 回归 | 模拟待实机 | 独立 evidence/confidence/source_slot 缺；生产未验收 |
| AC-055 | 四时点追溯 | 部分实现 | V§43 三处 OperationLog 结构化 extra，41 tests OK；V§88 release_audit 批准/激活时点增量（模拟） | 模拟待实机 | 候选/批准/激活时点关联及日志与业务同事务缺 |
| AC-056 | Gauge唯一判定 | 已实现 | V§11 离屏 8 项；V§84 TEST-04 桌面 Gauge 空态契约 PASS | 部分 PASS（待复核） | Gauge 有值态未覆盖（需工单链路）；生产框未勾 |
| AC-057 | 三态与open分离 | 已实现 | V§84 TEST-04 桌面 open 徽标/不完备预警与 API 一致 PASS | 部分 PASS（待复核） | 真实闭环全场景缺；生产框未勾 |
| AC-058 | 统计和游标 | 部分实现 | V§47 游标修复；V§49/§61/§67/§74 口径/null 契约（smoke 至 383 OK）；V§84 口径卡桌面 PASS | 部分 PASS（待复核） | 接收→落库率/backlog 无数据源；timeseries 不能按工单拆分 |
| AC-059 | PCB资产统一 | 部分实现 | V§87 保留期 preview/execute 间接覆盖原图生命周期（TEMP） | 模拟待实机 | 三来源统一、RTSP 原图可追溯生命周期缺 |
| AC-060 | 七槽编排 | 部分实现 | V§91 orchestration_table（is_seven/missing_base/diff，模拟，数据层） | 模拟待实机 | 无统一 UI 编排页（final_sim 与 round2_release 结论冲突，取较新部分实现） |
| AC-061 | FeatureSetRevision | 部分实现 | V§24 manifest 19 项回归；V§88 EvalRun 绑定校验；V§91 release_ops.revision_binding（模拟） | 模拟待实机 | persist 未调用 revision_binding；生产 trad 非 revision 管理；REMAIN-03 生产血缘闭环有实现前置缺口 |
| AC-062 | Trad接入前等价 | 部分实现 | V§83 TEST-06：8 项过 2 败 6，verdict=new_candidate（唯一根因 selfsim RNG 口径）；V§89 parity harness；task\test06_trad_parity | FAIL（new_candidate） | 真实比对未通过；final_model/final_sim 结论冲突取保守；待 REMAIN-01 复测修复后版本 |
| AC-063 | 定位来源契约 | 已实现 | V§46/53/54/55/68/71/74 元数据/mask/raw 热图/human_box/消费回写；V§87 复算 API；V§90 复算 UI 只读预览（离屏） | 模拟待实机 | 真实桌面与真实留存 raw 热图端到端未验（REMAIN-11） |
| AC-064 | 多框复核 | 已实现 | V§48/51/55/70/71/74/85 争议隔离/面板/绘框/角点/删除/撤销/redo/拖边/平移（离屏 smoke 至 364 OK） | 模拟待实机 | 实机桌面验收缺（TEST-04 类）；与 used/stale 组合未单独断言 |
| AC-065 | 后处理与定位验收 | 已实现 | V§52 版本化；V§85 冻结 pp-2f54c2e670，dev recall 0.923/precision 0.970（合成标注，loc 4 项 OK）；task\final_sim\raw\ac065_loc_frozen.json | 模拟待实机 | 无真实标注定位集；生产验收 0 |
| AC-066 | 原生端到端200ms | 部分实现 | TEST-01：V§57 P95 295.8/274.7ms FAIL；V§65 复测 216.3ms FAIL；V§80 第三轮 204.1ms FAIL；V§94 gold_finger 纯推理 P95 397.7→316.8ms（GPU 开关 181.8ms，待全量集 AUROC 复核）；task\test01_e2e_latency、test01_e2e_latency\dev_*、round3_speed | FAIL | 默认路径仍 >200ms；GPU 开关未复核 AUROC 前不算通过；原生 2500 正式口径缺（REMAIN-04） |
| AC-067 | 独立精度强门禁 | 已实现 | TEST-07：V§81 FPR 10/10=100% FAIL；V§84 生产 v2 同样 FPR 100%/AUROC 0.5（系统性）；V§88 门禁链 9 项 OK（模拟）；V§96 软限幅 AUROC 0.935→0.970、FPR≤5% 下 recall 0→0.80；V§100 重建后 FPR 0.556→0.333；task\test07_precision_gate、round3_accuracy、round4_* | FAIL | 固定阈值 FPR 仍 100%/重建后 0.333，远高于 ≤5%；瓶颈在模型缺陷敏感度（E2 AUROC 0.54～0.58），开发排查中；门禁只保证不放行 |
| AC-068 | 学习资源上界 | 部分实现 | V§22 候选全轮预算与合作式取消（TEMP 小测） | 模拟待实机 | 真实墙钟/显存/最大库未保证 |
| AC-069 | GPU并发与前台 | 部分实现 | TEST-02：V§62 FAIL（c1 learning P95 1505ms/c4 idle 406ms）；V§66 学习让行；V§72 前台锁粒度（lock_wait P95 77.7→0.007ms）；V§78 锁内 hold 963.7→657.4ms；V§80 复测 c1 394.1ms 仍 FAIL；task\test02_gpu_concurrency | FAIL | 前台 P95 仍不满足 G1；锁内残余 hold/候选训练让行无上界/持续负载学习限速；TEST-02 口径复测待 REMAIN-06 |
| AC-070 | 显存上界 | 已实现 | TEST-03：V§58 整卡峰值 926MiB/6144 PASS（OOM 子项未做）；V§86(final_perf) 候选 vram_fraction=0.85、峰值 127.7MiB/上限 5222.1MiB；task\test03_vram | 部分 PASS（待复核） | OOM 后工作态/事件不破坏未验（REMAIN-07，破坏性测试需授权）；上限数值待用户定；双实例/多槽未测 |
| AC-071 | 冷启动预算 | 已实现 | V§13 冷载 2063.641ms；V§23 冷首请求 2658.912/2201.558ms；V§86(final_perf) COLD_START_BUDGET_DEFAULTS+warmup/cold_start_stats（本机模拟） | 模拟待实机 | 生产可冷载路径预算未在产线验；生产验收 0 |
| AC-072 | 业务预算与保留期 | 已实现 | TEST-08 用户定值 2026-10-13（FPR≤5%/Recall≥95%/回退≤2pt/灰区≤15%）；V§79 可配置化 SYNC-27；V§87 retention.py+late_excluded（TEMP）；V§95 gt_threshold_frozen；task\test08_business_budget | 部分 PASS（待复核） | 保留期不自动删除、删除审批未做；标签迟到入训仅开发版模拟（REMAIN-12） |
| AC-073 | 角色工作区与证据面板 | 部分实现 | V§43 STALE_OVERRIDE_FORBIDDEN 9 tests OK；V§85 三角色越权 401/403、发布调用 0 次（模拟）；V§88 check_recovery_authorization | 模拟待实机 | 完整工作区/权限与操作联调缺（REMAIN-08 桌面补验） |
| AC-074 | 对比与回滚审计 | 部分实现 | V§43/§45 候选四操作结构化 OperationLog 55 tests OK；V§88 release_audit 哈希链（模拟） | 模拟待实机 | 事务批准链缺、日志与业务不同事务；普通快照激活不强制走 CandidateRelease |
| AC-075 | 权重加载回退 | 已实现 | TEST-05：V§82 A/B PASS、C FAIL（torch.hub 无超时挂起 180s）；V§85 dino.py 改 vendor→hub 缓存→在线、离线缺权重立即 RuntimeError，deploy 4 项 OK（模拟）；SYNC-28 已同步稳定版；task\test05_offline_weights | 待复核（历史部分 FAIL，修复经模拟验证） | 干净机器真实断网复测未重做（REMAIN-10）；EXPs 与 Core 双入口一致性待核 |
| AC-076 | 部署与兼容资产 | 部分实现 | V§43 stale 门禁；V§85 19 个 pkl 0 加载错误但锁文件与环境不一致（torch 2.5.1+cu121 vs 2.14.0+cu126 等，未修仅记录）；V§95 deplock.py warn/block | 模拟待实机 | 锁文件本身不一致未修；真实干净部署缺 |
| AC-077 | 历史双轨纠偏 | 部分实现 | V§85 合成复验；权威清单已知事实节已写勘误 | 模拟待实机 | 后续运行报告遵循情况待持续核对 |
| AC-078 | 实机视频多域 | 部分实现 | TEST-09 BLOCKED（缺用户授权实机/视频）；V§85 合成 3 域×3 种子（最差域 pcb 弱缺陷 0.333，合成非实机） | BLOCKED | 需授权实机/视频 ≥3 品类×3 种子独立 G1/G2/G6（REMAIN-12） |
| AC-079 | 泄漏与残留治理 | 部分实现 | V§85 _protected_versions 保护在用/候选版本，dry-run 136 项 0 删除 | 模拟待实机 | 旧新代码同流量化泄漏未跑；未清任何资产 |
| AC-080 | 夹具与严格输出 | 部分实现 | V§12/§14 定向 5 项+合成 22 断言；V§85 复验 | 模拟待实机 | 真实数据重训/历史污染专项验证缺 |

**索引计数**：实现状态列 = 已实现 24 / 部分实现 54 / 未实现 2（与权威清单 §8 一致）。真实验收状态列 = FAIL 4（AC-007/062/066/067，另 AC-069 FAIL，共 5 项含 FAIL 结论）/ 部分 PASS 或 PASS 待复核 7（AC-051/053/056/057/058/070/072）/ BLOCKED 1（AC-078）/ 开发前置缺口 2（AC-045/046）/ 其余为模拟待实机或待复核。生产验收勾选 = 0/80。

---

## 第二部分：G1～G6 门禁结论

### G1 时延红线（原生 2500² 单图 e2e P95 ≤ 200ms 且 G2 同过）——**FAIL**
- 定义：权威 §2 G1；替代所有旧 1 秒口径。
- 当前证据：TEST-01 实测热态 P95 direct 295.8→216.3→204.1ms 三轮复测均 FAIL（V§57/§65/§80）；V§23 完整入口热 P95 341.4/319.1ms、学习竞争 2203ms；V§86(final_perf) 纯推理 P95 428.6ms、空闲 svc_e2e 609.0ms；V§94（TEMP 小样本纯推理）gold_finger 397.7→316.8ms，GPU DoG 开关 181.8ms 但 gold_finger 全量集 AUROC 未复核（U22 曾 0.7075→0.6112），复核前不算通过。
- 结论：**FAIL**。所有 ≤200ms 附近数字均为非原生/纯推理/小样本口径；原生 2500 正式复验待 REMAIN-04。

### G2 独立精度（业务冻结 FPR≤5%/Recall≥95%/旧域回退≤2pt/灰区≤15%）——**FAIL**
- 当前证据：TEST-07 FPR 10/10=100%（tau=0.828 低于全部样本得分，2.0 钳位饱和，V§81）；TEST-04 附带证实生产 v2 降级快照 test 集同样 FPR 100%/AUROC 0.5——精度问题为**系统性**（V§84）；V§96 软限幅后 fused AUROC 0.970、FPR≤5% 下最大 recall 0.80，但固定阈值 FPR 仍 100%；SYNC-31/round4 重建后固定阈值 FPR 0.556→0.333、recall 0.79→0.26，仍远高于 ≤5%；成对难例 E2 AUROC 仅 0.54～0.58，瓶颈在模型缺陷敏感度（V§98/§100）。
- 结论：**FAIL**。门禁链机制（V§88，POLICY_*/EVAL_RUN_* 错误码）模拟验证有效、只保证不放行；精度本身 blocked on 模型能力+批次覆盖，开发排查中；正式精度验收待 REMAIN-05。

### G3 恢复/故障（重复/冲突/并发/故障注入/取消/WAL/重启/跨进程恢复一致）——**部分（未验收）**
- 当前证据：receipt/fsync、重复 ID/冲突、training/prepared 显式 recover 拒绝已测（V§16/17/19）；坏 WAL 行进程内审计（V§36）；只读 discover_prepared 39 tests OK（V§38）；RNG 严格预验证 26 tests OK（V§38）；TEST-03 的 OOM 破坏子项未做（V§58）。
- 结论：**部分**。跨文件事务非 exactly-once；跨进程/重启/崩溃/断电/OOM 真实故障注入未执行（REMAIN-07/09）。

### G4 真实操作（API/权限/真实桌面与 HTTP 联调、四时点追溯）——**部分**
- 当前证据：TEST-04 真实桌面 PASS 覆盖场景（Gauge 空态、口径卡、open 告警、legacy 未知保留、评估看板预算线，V§84）；PCB-ACCEPT-01 真实桌面四轮（第一二轮发布链路 PASS、第三轮 TH A/B/C/E/F PASS + **D FAIL 清空复活缺陷**、第四轮压力 5/5 PASS，V§101/§103）。
- 结论：**部分**。Gauge 有值态、alert/warn 同屏、特征工程页真实 HTTP/桌面（REMAIN-02/08）、PCB 取消语义/REVIEW 触发/tooltip 未覆盖；PCB 侧存在 9 项已登记真实缺陷未修（协同 §4.5）。

### G5 定位（真实独立框/掩码、IoU/定位召回/误定位）——**部分（仅合成证据）**
- 当前证据：定位容器/来源/版本/mask/raw 热图链路完整（V§46～§74、§87、§90）；冻结后处理 pp-2f54c2e670 在**合成标注**上 dev recall 0.923/precision 0.970/misloc 0.056（V§85）；PCB 侧坐标契约 paired 对照通过但 SMT raw 输出图等价 FAIL 0/3 单列保留（V§86 PCB 节/§104）。
- 结论：**部分**。无真实人工标注定位集，IoU/实例召回真实指标缺（REMAIN-11）；PCB 定位 IoU 门禁待 GT 冻结（PCB-ACCEPT-03 BLOCKED）。

### G6 学习（固定事件流/预算/种子、真实训练回放与旧样重测分报）——**部分（未验收）**
- 当前证据：候选预算底座（V§22）、隔离候选/门禁链（V§34～§43、§85～§88）均为 TEMP/模拟；TEST-02 并发 FAIL（V§62，复测 394.1ms 仍 FAIL，V§80）；六臂同预算仅 paired_prediction_delta 最小切片（V§42）；**AC-045/046 L1 真实训练与污染退出未实现**（仅设置页登记开关，V§95）。
- 结论：**部分**。真实训练回放、多流多种子配对 CI、L1 能力均缺；待 REMAIN-06/09 及开发实现 AC-045/046。

---

## 第三部分：台账统计核对

台账口径：「已实现（模拟验证，待实机）24 / 部分实现 54 / 未实现 2」。

**核对方法**：以权威清单 §3～§6 共 80 行的「实现状态」列逐项重数（审计口径与权威一致："基本实现"记部分实现；冲突取保守/较新）。

**本审计逐项统计结果**：
- 已实现 **24**：AC-004/008/013/018/020/021/022/028/029/030/031/032/038/051/056/057/063/064/065/067/070/071/072/075 —— 与台账/权威 §8 完全一致。
- 部分实现 **54**：其余除 AC-045/046 外全部 —— 一致。
- 未实现 **2**：AC-045/046 —— 一致（V§95 仅登记 governance 开关，无运行时行为，维持未实现）。
- 生产验收通过 **0 / 80** —— 一致。

**结论：台账「24 模拟待实机 / 54 部分 / 2 未实现」统计在权威状态列口径下仍然准确。**

**与台账/权威的差异与注意点（非计数差异，为口径与冲突提示）**：
1. 台账内并存历史快照计数「已实现 17 / 部分 58 / 未实现 5」（台账 §0 节，2026-10-12 复核段），系 SYNC-28 前旧口径，与最新 24/54/2 不矛盾但易被误读，引用时须带时间戳。
2. AC-070：权威标「已实现」，但 TEST-03 协同表结论为「部分验收」（峰值采样 PASS、OOM 子项未做、上限待用户定）。建议理解为「模拟/部分实测已实现、真实验收部分 PASS 待复核」，不宜读作全子项通过。
3. AC-075：权威标「已实现」，TEST-05 曾部分 FAIL（C 场景 180s 挂起）；SYNC-28 dino.py 修复仅经模拟验证（V§85 deploy 4 项 OK），真实断网复测未重做（REMAIN-10）。状态列无误，真实验收列本审计标「待复核」。
4. AC-062：final_model（V§89）建议「机制已实现」与 final_sim（V§85）建议「未实现」冲突，权威取保守「部分实现」；TEST-06 真实比对 FAIL/new_candidate（V§83）未被后续 AOI_feature selfsim default_rng(42) 修复的少量内存图 parity 推翻（协同「最近开发交付」自述不覆盖 TEST-06 结论），待 REMAIN-01 全链路复测。
5. AC-060：final_sim 建议「未实现」与 round2_release 建议「部分实现」冲突，取较新「部分实现」。
6. AC-051/056/057：权威「已实现」且 TEST-04 桌面 PASS（覆盖场景），是 80 项中最接近真实验收的 3 项，但仍有未覆盖场景（Gauge 有值态、alert/warn 同屏、真实闭环），生产框仍不可勾。
7. 台账行级计数（「已实现 32 / 部分 47 / 未实现 3 / 其他 2」，SYNC-30 段）是台账自身行粒度（含 HARD-01/02 等非 AC 行），与权威 80 行口径不同，不能直接比较；两者各自自洽。

---

## 第四部分：遗漏 / 冲突 / 开发前置清单

### 4.1 REMAIN-01～13 未覆盖或待完成的开放项
| 编号 | 状态 | 关键开放点 |
|---|---|---|
| REMAIN-01 | 未认领 | Trad parity 全链路复测（依赖 selfsim RNG 修复版本登记与同步核实） |
| REMAIN-02 | 未认领 | 特征工程页真实 HTTP 联调（feature_page 等 8 文件尚未同步稳定版，见 4.3） |
| REMAIN-03 | 存在实现前置缺口 | 生产 Trad 读取并绑定 FeatureSetRevision 的闭环尚未实现；若实现不存在应交 FAIL/开发阻塞清单 |
| REMAIN-04 | 待复验，历史 FAIL 保留 | 原生 2500 direct+HTTP、空闲/学习负载 G1 正式复验 |
| REMAIN-05 | 待复验，历史 FAIL 保留 | 正式精度门禁（FPR≤5%/Recall≥95%/回退≤2pt/灰区≤15%）；当前 FPR 100%/0.333 未解 |
| REMAIN-06 | 待复验，历史 FAIL 保留 | 并发锁等待真实插桩（TEST-02 t_lock_acquired 口径缺陷须补）、budget/cancel 覆盖 |
| REMAIN-07 | 待补 OOM 与资源复核 | OOM 破坏性测试需专用隔离环境授权 |
| REMAIN-08 | 待补真实交互覆盖 | Gauge 有值态、特征页、11 页角色导航桌面补验 |
| REMAIN-09 | 部分实现前置缺口 | 跨进程/重启/崩溃恢复矩阵；AC-045/046 登记为开发阻塞而非测试 PASS |
| REMAIN-10 | 待复验缺权重分支 | 真实断网干净缓存复测（TEST-05 C 场景修复后） |
| REMAIN-11 | 待执行 | 真实人工标注定位集 IoU/实例召回；冻结定位公差缺 |
| REMAIN-12 | 部分外部前置阻塞 | 保留期/删除审批/迟到入训核查；实机视频授权（TEST-09 BLOCKED） |
| REMAIN-13 | 待复核（本轮增量已交付） | PCB：toe/rim 人工标定审核、Crack NG 正例、03 GT/阈值用户冻结、04 专项实测授权仍硬阻塞 |

### 4.2 证据相互矛盾处
1. **AC-060**：final_sim「未实现」 vs round2_release「部分实现」——取较新部分实现，统一编排页仍缺。
2. **AC-062**：final_model「机制已实现」 vs final_sim「未实现」 vs TEST-06 实测 FAIL/new_candidate——取保守部分实现，真实比对结论以 TEST-06 为准。
3. **AC-070**：权威「已实现」 vs TEST-03「部分验收」（OOM 未做）——真实验收口径降级为部分 PASS 待复核。
4. **AC-075**：权威「已实现」 vs TEST-05 部分 FAIL（修复仅模拟验证）——真实验收列待复核。
5. **SYNC-30 关键数字口径**：P95 316.8ms / AUROC 0.970 为开发版 TEMP 小样本纯推理，稳定版未复测性能（V§102 明确），不得写成稳定版实测。
6. **验证记录 §86 重号**：「86. PCB-ACCEPT-01/02/05」与「86. final_perf」同号，引用 final_perf 须写 V§86(final_perf)（协同已注记）。
7. **TEST-02 原始口径缺陷**：t_lock_acquired 实为线程开始时间，引擎内锁等待未拆分（协同 TEST-02 行），历史 FAIL 结论仍成立但复测须改口径（REMAIN-06 已含）。

### 4.3 必须先由开发实现/修复才能验收的项（开发前置）
1. **AC-045/046**：L1 真实训练、L1 污染与退出能力未实现（仅登记开关），REMAIN-09 相关子项只能登记开发阻塞。
2. **AC-062 / REMAIN-01**：AOI_feature selfsim RNG 口径已修（default_rng(42)）但对应批次未登记/未同步稳定版（协同 §4.6 未登记差异 8 文件：feature_page.py、test_feature_parity.py 等），验收前须开发对话确认归属并登记批次。
3. **REMAIN-03**：生产特征血缘绑定闭环（Trad 实际读取 FeatureSetRevision、Detection 记录真实 revision）实现可能不存在，须开发先交付。
4. **PCB 缺陷转开发 9 项**（协同 §4.5，全部未修）：①TH 清空焊点框 legacy roi 复活（优先）；②切模板静默清空标定；③调试页尺寸门禁不读模板级 gate；④API 直跑未就绪项 ERROR/NOT_READY 与 UI 口径不一致；⑤建模页关联标签不刷新；⑥阈值 spinner 两位小数精度；⑦恢复默认恢复到算法全局默认而非模板配方；⑧bridge 整批 [ERR] + 导入不递归；⑨TH 配准错配未 fail-closed（resp 不入门禁）。⑨修复后须复跑 `pcb_accept_02\raw\crack_mismatch_check.py`。
5. **AC-076**：requirements-lock.txt 与环境实际版本不一致（torch/numpy/Python 三项）未修，仅记录+deplock 告警。
6. **AC-067/G2 数据侧**：`approved_extra_good_batches` 稳定版取值待验收对话/用户决定（开发版 default.yaml 为开发方代定，按 SYNC-31 注记未同步）；datasource_id 重建分支未接登记。
7. **精度模型能力**：固定阈值 FPR 100%/0.333 根因在模型缺陷敏感度（E2 AUROC 0.54～0.58），非验收侧可解，开发排查中。

### 4.4 本审计未覆盖声明
- 未运行任何测试/命令复算（只读），所有测试计数引用自 V/台账/协同原文。
- 未逐一打开 task 目录产物文件，仅核对目录与 result.md 存在性作为证据索引。
- REMAIN-01～13 的结论以协同.md 认领池登记状态为准，本审计不替代其执行。

---

## 审计口径声明
1. 本产物为**只读草稿**，是最终汇总前的中间件；最终汇总依赖 REMAIN-01～13 完成。
2. 未复核项一律标「待复核」；无证据项标「无」或「缺口」；不以模拟/TEMP/离屏证据勾生产完成。
3. 速度数字严格区分实测（TEST-01/02/03 系列）与估算/小样本（V§86/§94 等）；估算口径未写成实测。
4. 生产验收框 `[ ]` 全部保持未勾；本审计不建议勾任何一项。
