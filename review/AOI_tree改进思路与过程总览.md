# AOI_tree 改进思路与过程总览

> 本文只做汇总和索引，不作为证据源。证据以验证记录（V）的 § 编号为准，状态以权威清单为准。统计时点为编写当日的文件内容，AC-072 行可能正在被并行更新。

## 1. 文档体系

| 文档 | 角色 | 维护规则（协同.md 第 16～21 行） |
|---|---|---|
| `commit/repo/review/简明改进建议.md`（权威清单） | 定义 AC-001～080、G1～G6 验收标准 | 只同步“实现状态”列；生产验收框 [ ] 只能由用户勾选 |
| `AOI_SSOCL/review/AOI_tree全量改进任务清单.md`（台账 T） | 任务级进度（A/B/CDEF/GHIJK/UI/LOC/SS/HARD/REPORT） | 开发版维护 |
| `AOI_SSOCL/review/AOI_tree首批改进与验证记录.md`（V） | 按 § 追加实现和测试证据（§1～§91） | 只追加，不改写 |
| `AOI_SSOCL/review/协同.md` | 开发版与稳定版之间的 SYNC 批次、TEST-01～09 认领与验收 | 稳定版（验收对话）负责同步并回写 |

## 2. 总体改进思路

- **fail-closed 优先**：缺失值不补 0（SYNC-02），快照缺字段时显式降级（§63），allow_stale 返回 409（§43），候选发布要经过 CandidateRelease/RegressGate。
- **可审计、可回滚**：WAL/receipt、FeatureSetRevision、审计 extra（§43/§45），定位元数据和 region used 回写（§46/§71）。
- **不以精度换时延**：凡会改变分数或判定的提速方案一律否决，例如 IMREAD_REDUCED（§64）和下采样（§69）。只接受差异为 0 的优化，或能解释差异来源的优化（§77）。
- **双对话协作**：开发版实现后登记 SYNC，由稳定版全量回归后回写（§44/50/56/60/65/75，共 6 轮）。
- **数字说话**：没有达标就不宣称达标，例如学习不能宣称 <1s（§7），缩放尺寸的 p95 不算正式验收（§10）。

## 3. 主题时间线

| 主题 | V§ | SYNC | 关键结论数字 | 状态 |
|---|---|---|---|---|
| 基线精度/域差 | §6 | — | gold_finger 20 图，fused AUROC=0、AP=0.6886；train 域 AUROC 0.7657 | 域差确认，G2 未过 |
| 持续学习耗时 | §7 | — | 单轮最慢 11670.6ms（LOO 7188ms），12 轮约 44.8s | 不宣称 <1s |
| G1 早期测量 | §10/§15/§18/§23 | — | 缩放尺寸 p95 184.4ms（非正式）；正式服务 402.4 → 327.2/316.1 → 热态 341.4/319.1ms；学习竞争下约 2.2s | FAIL |
| 局部分支 A9 | §25 | — | global p95 475ms（2500² 折算 241.8），full 3532，budget 4327ms；显存 130→906MiB；三方案 fpr 都是 1.0 | 不默认开启 |
| 候选发布/恢复/CDEF/G 块 | §16～§43（含 §27、§29～§33） | SYNC-01 | 审计 extra、allow_stale 409 等 | 部分实现 |
| UI/统计口径 | §11/§45/§47/§49/§61/§67 | 02/02b/03/05/07/13/17 | 缺失值不补 0、live 游标、口径卡、统计页 | 已同步 |
| 定位与复核 | §46/§48/§51～§55/§68/§70/§71/§74 | 04/06/08～12/18/20/21/23 | 定位元数据、dispute、多框复核、浮点热图、角点微调/撤销、region used | 已同步，实机未测 |
| TEST-01 端到端 | §57 → §65 | — | 热态 P95 direct 295.8 / e2e 288.4 → 235.3 / 216.3ms | 仍 FAIL |
| TEST-03 显存 | §58 | — | 峰值 926MiB；草案上限 ≤1024、告警 3000、拒绝 4500 | 部分验收，OOM 未做 |
| TEST-08 业务阈值 | §59 | — | 草案：FPR ≤1%、Recall ≥99%、G6 回退 ≤0.5pt、灰区 ≤10%；热存 90 天约 500GB | 待用户定值 |
| TEST-02 学习并发 | §62 | — | c1 idle 263 → learning 1505ms；c4 406/960 | FAIL |
| 快照兼容 FIX-SNAP | §63 → §65 | SYNC-14 | v2 显式降级，layout 权重 0；加载 0.86s | 已同步；阈值偏移未评测 |
| 解码优化 OPT-DECODE | §64 | SYNC-15 | decode p95 116.5→105.3，e2e 205.7→190.8ms，差异 0 | 已同步 |
| 学习让行 | §66 | SYNC-16 | learn P95 1359.3→254.8（cached 127.2）；等锁次数 10→1；学习次数 10→2 | 已同步（§75） |
| 下采样评估 | §69 | — | JPG 156.9→115.8 但 IoU≈0.2，判定翻转 4～6/34 | 否决 |
| 前台锁粒度 | §72 | SYNC-22 | lock_wait 77.7→0.007ms，wall 302.7→303.1 持平 | 已同步 |
| render 异步 | §76 | SYNC-24 | render P95 34→19ms；空闲 e2e P95 180.6ms；负载下 190～225ms | 待同步 |
| 帧直检 | §77 | SYNC-25 | P95 166.6→123.3ms；1/8 判定翻转，来源是旧路径 JPEG 重编码 | 待同步 |
| 学习锁缩短 | §78 | SYNC-26 | accepted hold 908.5/1018.9→684.4/630.4ms（-32%）；rolled_back 无变化 | 待同步 |
| 稳定版验收 | §44/§50/§56/§60/§65/§75/§80 | SYNC-24～27 | 64 → 283 → 289 → 299 → 331 → 378 → 405 OK | 7 轮通过 |
| 实测 TEST-04～07 | §81～§84 | — | TEST-04 桌面 PASS；TEST-05 部分 FAIL（hub 无超时挂起 180s）；TEST-06 FAIL（selfsim RNG 口径）；TEST-07 FAIL（FPR 100%，生产 v2 与 TEMP 新模型均 AUROC 0.5） | 精度问题为系统性 |
| final 5 组（sim/perf/learning/release/model） | §85～§89 | SYNC-28 | 全量 smoke 405→485 OK（147.3s）；HARD-01 纯推理 P95 428.6ms；G2 门禁 fail-closed + Wilson 区间 | 模拟验证，待同步 |
| round2 2 组（learning/release） | §90～§91 | SYNC-29 | 9 + 12 tests OK；全量 smoke 506 OK（154.2s） | 模拟验证，待同步（须在 SYNC-24～28 后整文件同步） |

注：SYNC-19 编号未使用；§73 编号丢失，内容补登为 §76。V§86 同时被 final_perf 与 PCB-ACCEPT 使用，引用 final_perf 时写作 V§86(final_perf)。

### 3.1 final + round2 两轮的思路和结论（2026-10-13～14）

- **思路**：把剩余“部分/未实现”项拆成可在 TEMP、合成数据、离屏 UI 中闭环的最小切片，并行 5 组 + 2 组；只补契约、门禁、审计和离线函数，不改生产快照、不跑全量数据。凡依赖真实产线、标注或用户定值的部分一律留作缺口。
- **final 组结论（V§85～§89）**：final 后权威清单为已实现 17 / 部分实现 58 / 未实现 5，smoke 增到 485。性能侧确认本机纯推理 P95 428.6ms，200ms 红线不可达；G2 门禁只保证不放行，精度没有改善（TEST-07 FPR 100%）。AC-062 两组结论冲突，保守取部分实现。
- **round2_learning 结论（V§90）**：复核对话框接入热图复算只读预览，AC-063 转已实现；离线隔离默认 self→auto（fit≥16 走 holdout 80/20）；新增 label_audit（标签时序审计、标签身份/覆盖解析）和 open_alerts 汇总只读接口。AC-019/042/044/052 为部分实现，其中 AC-044 由未实现转部分实现。
- **round2_release 结论（V§91）**：release_ops 补 12 个离线运维函数并修复 `SNAPSHOT_REQUIRED` 键名 bug（修复前真实快照恒判 incomplete）。AC-013/028～032 转已实现；AC-011/012/014/015/016/033/034/035 为“基本实现”，清单记为部分实现；AC-060 由未实现转部分实现。共同缺口：全部是离线函数，没有接 API/UI，也没有接入 persist 写快照。

### 3.2 第三轮：精度/速度收束（2026-10-14，V§94～§97，待同步 SYNC-30）

- **思路**：前两轮之后剩下两类主线问题：G1 速度和 G2 精度，另外还有一批只能由人工拍板的事项。本轮不再扩展新功能，分 4 组收束：速度组只做判定不变的无损改写，并加一个默认关闭的加速开关；精度组只修校准函数中可证明的缺陷；人工决策项集中到设置页「待决策事项」（`governance_decisions`），默认取保守值，不替用户决定；覆盖组在评估侧报告批次覆盖缺口，不改判定。
- **根因**：
  - 速度：blob 槽 CPU 路径是 gold_finger 的主要耗时。GPU DoG 能达标，但 U22 曾记录 AUROC 0.7075→0.6112，不能默认开启。
  - 精度：`np.clip(q,-1,2)` 硬钳位让高分样本挤在 2.0 上形成平局。更根本的原因是 test 正常样本来自 train/good 没有覆盖的采集批次（Image_2026、t_fov），阈值落在全部 test 正常样本得分之下，所以固定阈值 FPR 恒为 100%。次要发现：sem 方向反转（raw AUROC 0.30），sanity 把 shead 权重置零。
- **结果**（均为 TEMP/合成/小样本模拟验证，生产验收 0）：
  - V§94 round3_speed：gold_finger P95 397.7→316.8ms（默认路径无损，仍超红线），开 GPU 开关 181.8ms（26/26 判定一致，Δscore 0）；66 项 OK。HARD-01 在台账转为部分实现。
  - V§95 round3_decisions：设置页登记 11 项决策键（admin 可改）；冻结后改 G2/G6/灰区阈值或 `gt_dataset_ref` 返回 409 `THRESHOLD_FROZEN`；新增 deplock 依赖锁（warn 附告警，block 返回 409）；33 passed。
  - V§96 round3_accuracy：`_soft_bound` 替换硬钳位，fused AUROC 0.935→0.970，FPR≤5% 下最大 recall 0→0.80；固定阈值 FPR 仍为 1.00，blocked on data。57 例 + 64 subtests OK。
  - V§97 round3_coverage：新增 `coverage.py`，评估页显示批次覆盖告警和预算参考工作点（不改生产阈值）；决策键再加 `blob_gpu_enabled`、`approved_extra_good_batches`，共 13 项；Ran 40 OK。
  - 全量 smoke **536 tests / 135.936s / OK**（506→536）。权威清单计数不变（24/54/2）。

### 3.3 第四轮：根因诊断、决策落地、批次重建（2026-10-14，V§98～§100，待同步 SYNC-31，须在 SYNC-30 之后同步）

- **思路**：第三轮留下三件事：sem 方向反和 shead 置零的原因、13 项决策键一直是默认值、新批次 good 基线没有实际重建。本轮分 3 组处理：先做根因诊断，查清是代码还是数据的问题；把开发方可以代定的决策写入开发版 `default.yaml`，涉及 GT 和验收的项留给人工；把登记的批次真正接入重建，然后重测 FPR。
- **结果**（均为模拟验证，生产验收 0）：
  - V§98 round4_sem_shead：不是代码 bug，算法未改。gold_finger 的正常图是缺陷图的 `*_repaired` 孪生图，sem 实际衡量的是批间距离（domain 下 sem sanity 0.095）；shead 权重置零来自 sanity 规则 `w=max(0,auc-0.5)`（sanity 0.38），但 test 上 shead AUROC 为 0.956。domain 下 fused AUROC 0.703、FPR≤5% recall 0.00；改为等权后是 0.682 / 0.20，有升有降，因此没有加开关。8 OK。
  - V§99 round4_decisions_apply：开发版 `AOI_Core/configs/default.yaml` 末尾追加 `governance_decisions`，开发方代定 7 项，保留人工 6 项（见 7(a)）。`approved_extra_good_batches` 必须写成逗号分隔的字符串，写成 YAML 列表会回落为空。20 OK。
  - V§100 round4_batch_rebuild：修复登记后重建不生效的 bug。原来 test/good 路径会被 `guard_bundle` 拒绝，现改为 staging 副本 `approved_extra/train/good`；新增 `split_extra_good`（fit 与 eval 各占 50%），评估时剔除 fit 子集。B 组主结果：新批次固定阈值 FPR 0.556→0.333，E1+E2 recall 0.79→0.26；成对难例 E2 AUROC 只有 0.54～0.58。
  - 全量 smoke **549 tests / 154.231s / OK**（536→549，+13）。
- **结论**：HARD-02/G2 维持部分实现。登记→重建的路径已经接通，但精度瓶颈在模型对缺陷的敏感度，只补批次重建解决不了。后续需要更多同批次真实缺陷样本，并调整数据协议，把 sanity 和 shead 训练中与修复图配对的样本去掉。`datasource_id` 重建分支尚未接入登记。

## 4. 否决或未做的方案

| 方案 | V§ | 原因 |
|---|---|---|
| 局部分支默认开启（A9） | §25 | 时延和显存代价大，三方案 fpr 都是 1.0，没有精度收益 |
| 快照方案 a：用归档图重拟合 layout | §63 | 会混入缺陷样本；改用方案 b 显式降级 |
| IMREAD_REDUCED 解码 | §64 | 会改变分数 |
| 引入 turbojpeg/pyspng | §64 | 违反不新增依赖的约束 |
| 输入下采样（PNG/JPG/cpu profile） | §69 | PNG 没有收益；JPG 的 IoU≈0.2、判定翻转，重新 prepare 后丢检 |
| 读写锁 | §72 | predict 会写 _item_cache/_last_hms/trace/normal_bank/selector |
| GPU 合批 | §72 | 单 tile 收益小，会推高 c1，确定性难保 |
| 锁外影子 deepcopy / 拆分 CandidateBudget.check | §78 | deepcopy 需要 370～1290ms，比原来更贵；check 会写 gate 状态，fail-closed 风险高 |
| 学习让行不设上界 | §66 | 已知取舍：check 的让行目前没有上界，记为残余风险 |

## 5. 当前进度统计

| 来源 | 已实现 | 部分实现 | 未实现 | 其他 |
|---|---|---|---|---|
| 权威清单 AC-001～080 | 24（模拟验证，待实机） | 54 | 2 | — |
| 权威清单 G1～G6 | 0 | — | 全部未通过、未冻结 | — |
| 台账（84 行） | 32 | 47 | 3 | 2（A9 已执行但不开启；REPORT-10 小样本已验证） |
| 生产验收框 [x] | 0（全部未勾） | | | |

统计时点：2026-10-14，SYNC-28/29（V§85～§91）及 SYNC-30（V§94～§97）回写后。“基本实现”一律计入部分实现。第三轮仅 HARD-01 在台账由未实现转部分实现，权威清单数量不变。

**AC 已实现 24 条**：AC-004/008/013/018/020/021/022/028/029/030/031/032/038/051/056/057/063/064/065/067/070/071/072/075。

**AC 未实现 2 条**：AC-045/046。

**台账未实现 3 条**：B1、REPORT-4/5（HARD-01 于 V§94 转部分实现）。

**文档不一致**（以下问题已于 2026-10-13 修正）：
- 权威清单 §8 正文和台账第 10～11 行原写“部分 58 / 未实现 18（含 AC-069）”，与各行状态列不符；已按状态列重数为“已实现 4 / 部分实现 60 / 未实现 16”（AC-069、AC-072 计入部分实现）。AC-069 行状态列中过时的“待同步 SYNC-16”已改为“SYNC-16 已同步（V§75）”。
- 台账中 SS-STAGE 原标未实现，已按 §35 最小契约改为部分实现；HARD-03 原标未实现，已与 AC-068 一致改为部分实现。
- 台账第 29～31 行原写“当前无待同步批次”，已改为待同步清单：SYNC-24/25（均改 service.py，须按 24→25 顺序整文件同步）、SYNC-26/27（已在 协同.md 核对，状态为待同步）。

## 6. 剩余事项分类

### (a) 开发版可继续做的代码项

| 编号 | 缺口要点 |
|---|---|
| AC-002/003/004/005 | 联合专家协议、ROI 联合覆盖、全尺寸反解、面积标签全链路 |
| AC-011～016 | 组级生产选择、强血缘/CI、独立组级协议、逐框/Trad 参考、完整重提/重建、发布入口 |
| AC-019 | V§90 已改默认 auto（holdout），已有快照需重建，真实数据不退化未证 |
| AC-024/025/027 | 冷启动报告、能力健康页、全前端联调 |
| AC-028/029/031/032/033/035 | 条目血缘、严格聚类门禁、init/独立门禁、纯 patch 批准、能力说明、缓存/RNG/hash 盘点 |
| AC-038/041/042 | engine.submit_feedback 的引擎级守卫；L0/L1/L2 时序 |
| AC-047～050 | 争议修订链、撤错重建、六臂同预算和配对 CI、生命周期时间线和取消续接 |
| AC-052～055 | 分类/候选准入、evidence/confidence/source_slot、全血缘、日志与业务同事务 |
| AC-058/059/061 | 接收→落库率和 backlog 数据源、三来源记录和临时图生命周期、parent/Detection/发布回滚 |
| AC-063 | 复算 API/UI 已接（V§87/§90），转已实现；仅剩真机验证 |
| AC-069 | rolled_back 路径的 check 仍在锁内（§78），学习被限速（§66） |
| AC-073/074 | 领域层 allow_stale 审计绑定授权；pickle 版本警告 |
| §65 新发现 | 降级 slot 分数对不同图恒定；服务响应 decision_trace 为空（目前没有后续记录） |
| G1 剩余瓶颈 | PNG decode 98～107ms、infer 争用（§76） |

### (b) 需要真实产线、实机、标注或干净环境的实测项

| 编号 | 依赖 |
|---|---|
| AC-007 / G1（TEST-01） | 原生 2500² 产线条件下 p95 ≤200ms，目前仍 FAIL |
| AC-069（TEST-02 复测） | 按 TEST-02 口径复测，并修正 t_lock_acquired 口径，补测取消/限流 |
| AC-051/056/057（TEST-04） | 真实桌面，未认领 |
| AC-075 / AC-076（TEST-05） | 干净断网部署，未认领 |
| AC-001/018/020/021/022/030/036/037 | 品类档案、尾部验收、router 增益、动态域差、半监督闭环、独立缺陷/域外锚等，需要真实数据 |
| AC-026/039/040/043 | 旧反馈直接写工作态的实测；G3 和审计持久化；真实 GPU 跨环境回放；跨域独立回放 |
| AC-064/065 | 拖边/平移/重做的实机验证；真实标注定位集和冻结版本 |
| AC-068 / TEST-03 OOM 子项 | 真实墙钟、显存、最大库规模 |
| AC-071/080 | 生产冷启动预算；真实数据重训 |
| §63 残余 | v2 降级后的阈值偏移评测 |

### (c) 需要用户决策或授权的项

| 编号 | 待定内容 |
|---|---|
| AC-067 / AC-072（TEST-07/08） | 业务阈值数值（§59 草案已提交） |
| AC-070（TEST-03） | 显存上限（草案 ≤1024 / 3000 / 4500） |
| AC-062（TEST-06） | 提供旧 trad 协议工件 |
| AC-078（TEST-09） | 授权视频或实机 |
| AC-077/079 | 泄漏同流测试与清理审批 |
| SYNC-24～27 | 已同步（§80） |
| SYNC-28/29 | 稳定版同步验收（29 须在 24～28 之后整文件同步） |
| SYNC-30 | 第三轮 4 组（V§94～§97）待同步验收，全量 smoke 536 OK |
| PCB-TUNER / PCB-DUAL | 稳定版验收与用户授权 |
| 全部 AC、G1～G6 | 生产验收框只能由用户勾选 |

## 7. 剩余缺口（需真实环境/用户决策）

以下缺口无法通过模拟验证关闭。截至 V§100（SYNC-30、31 待同步），生产验收数为 0。按“待决策 / 需真实环境 / 暂缓”三类重排。

### 7(a) 待决策：设置页「待决策事项」（governance_decisions，config.py GOVERNANCE_DECISIONS_SPEC，共 13 项）

V§99 已在开发版 `AOI_Core/configs/default.yaml` 写入开发方代定的 7 项，其余 6 项保留人工决定（保持默认）。稳定版是否采用，由验收对话（SYNC-31）决定。

| 键 | 默认值 | 推荐值 / 决策条件 | V§99 处理 |
|---|---|---|---|
| gt_threshold_frozen | false | GT 确认后置 true，同时填 gt_dataset_ref；冻结后改 FROZEN_THRESHOLD_KEYS 或 gt_dataset_ref 返回 409 THRESHOLD_FROZEN | 保留人工 |
| gt_threshold_frozen_at | ""（只读） | 服务端在冻结时写入，无需人工填写 | 保留人工（只读） |
| gt_dataset_ref | "" | 填写确认后的 GT 数据集引用 | 保留人工 |
| ac045_l1_real_training_enabled | false | 保持 false（L1 真实训练未实现） | 开发方代定 =false |
| ac046_l1_pollution_exit_enabled | false | 保持 false | 开发方代定 =false |
| candidate_release_mode | manual | 保持 manual（可选 auto_on_approve） | 开发方代定 =manual |
| calibration_isolation | auto | 保持 auto（可选 self / holdout） | 开发方代定 =auto |
| dependency_lock_policy | warn | 锁文件修好前保持 warn（附 dependency_lock_warning），修好后改 block（不一致返回 409 DEPENDENCY_LOCK_MISMATCH） | 开发方代定 =warn |
| ac062_legacy_artifact_dir | "" | 提供旧 trad 协议工件后填写目录 | 保留人工 |
| sync28_29_accepted | false | SYNC-28/29（及 SYNC-30）稳定版验收通过后置 true | 保留人工 |
| production_acceptance_signed | false | 真实生产验收完成后由用户签署 | 保留人工 |
| blob_gpu_enabled | false | GPU 开关路径全量 AUROC 复核通过后再开启；env AOI_BLOB_GPU=0 可强制关闭 | 开发方代定 =false |
| approved_extra_good_batches | "" | 建议填 Image_2026、t_fov，并人工重建 good 基线 | 开发方代定 ="Image_2026,t_fov"（逗号分隔字符串；V§100 已接入重建） |

非 governance 的人工决策项：AC-070 显存上限（草案 ≤1024 / 3000 / 4500）；AC-077/079 泄漏同流测试与清理审批；全部 AC、G1～G6 生产验收框只能由用户勾选。

### 7(b) 需真实环境

| 项 | 缺口 |
|---|---|
| TEST-09 / AC-078 | 需要实机或授权视频 |
| TEST-05 | 需在干净的断网环境中运行 |
| AC-063/064 | 需在真机桌面复核 |
| HARD-01 | 默认路径未达 200ms 红线，GPU 开关路径仅小样本达标（V§94/§97）；需原生 2500 张及并发场景复测 |
| blob GPU | GPU 开关路径需全量 AUROC 复核后才能决定 blob_gpu_enabled |
| AC-069 | 学习锁需在产线负载下验证 |
| AC-019 / AC-065 | 需真实数据证明不退化；定位需真实标注集 |
| 新批次 good 重建 | 按 approved_extra_good_batches 重建后重测 FPR（V§95/§96）；V§100 模拟结果：B 组新批次 FPR 0.556→0.333，仍未达标，生产 DB 分支未用真实 engine 端到端跑过 |
| 精度瓶颈（HARD-02/G2） | 瓶颈在模型对缺陷的敏感度（V§98/§100，E2 AUROC 0.54～0.58）。需要更多同批次真实缺陷样本，并调整数据协议，把 sanity 和 shead 训练中与修复图（`*_repaired`）配对的样本去掉 |
| datasource_id 重建路径 | 按 datasource_id 重建的分支尚未接入 approved_extra_good_batches（V§100 边界） |
| 生产验收 | 全部生产验收框，当前为 0 |

### 7(c) 暂缓，不影响主线

| 项 | 说明 |
|---|---|
| 离线函数接入 API/UI | group_selection_plan、feature_ops_view / submit_toggle、capability_card、admission_registration_gate、orchestration_table、open_alerts summary 面板；batch_coverage / budget_working_point 仅作参考 |
| 禁改文件 / 快照写入 | persist.write_snapshot_state 需调用 revision_binding（AC-035/061）；已有快照需要重建 |
| label_audit | 尚未接入候选过滤 |
| AC-044 | 审计结果尚未落库 |
| sem 方向 / sanity 置零 | V§98 已诊断：非代码 bug（批间距离 + 修复图配对），改数据协议另列 (b) |
| AC-045/046 L1 实现 | 开关已登记（默认 false），实现暂缓 |
