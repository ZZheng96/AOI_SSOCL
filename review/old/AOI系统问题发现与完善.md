# AOI 系统问题发现与完善

> 评审对象：`AOI_tree/`（主干 AOI_Core + 树枝 PCB_Dual + research 复现包 + assets）
> 评审基准：《可自学习的AOI实时在线AI质检》赛题要求与工业级交付标准
> 评审时间：2026-10-03（现状复核）

---

## 一、总体结论

**AOI_tree 已形成较完整的工程原型和演示闭环，代码/配置/工具层的安全与交付阻断项均已落地；但赛题硬指标（2500×2500 / 2060 GPU / <200ms）与多品类自学习统计证据尚未在指定硬件与标准档数据集上复验。当前可认定为"具备参赛验收基础的工程原型"，不宜直接宣称为"完全符合赛题要求的工业级正式系统"。**

系统已具备图片检测、视频输入、自学习反馈、模型快照、公开数据集适配、泛化实验、热力图/缺陷框/槽位分数和桌面端+API 服务等基础能力；生产安全（鉴权/CORS/WebSocket/文档保护）、Docker 交付（CPU/GPU 双方案/持久化卷/健康检查/离线权重/非 root 运行）、依赖锁定（requirements-lock + checksums）、性能口径三层分离、视频/自学习协议化工具均已就绪。

**当前评级：工程原型较完整；P0-1/P0-2/P0-3 标准档复跑完成前不建议作为最终工业交付包或直接对外承诺赛题达标。**

---

## 二、赛题逐项对照（现状）

| 赛题要求 | 当前状态 | 结论 |
|---|---|---|
| 图片异常检测 | 图片路径、数据库图片、上传/内存帧入口齐备；多槽位融合和三态决策已接通 | 基本满足 |
| 视频异常检测 | 视频逐关键帧检测无帧数硬截断、`pulse_win` 可配置；协议工具已 PASS --quick | 待标准档多场景复跑 |
| 尺寸、缺件、逻辑、色彩、外观缺陷 | 传统 CV、布局/语义/颜色/开放集等槽位和 PCB_Dual 算法仓库已覆盖 | 能力存在，需专项数据证明 |
| 2500×2500、2060 及以下 GPU、<200ms | 三层预算口径已分离（竞赛 200ms / 红线 1000ms / CPU 挑战 2000ms） | 未在 2060 实机实测，不作正式承诺 |
| CPU 挑战目标 <2s | 有 CPU 实验方向 | 待固定输入/样本量/P95/环境 manifest 验收报告 |
| few-shot / zero-shot | 有 L0/L1a 配置、伪异常、迁移和公开数据适配 | 部分满足，缺统一 holdout 和重复实验支撑 |
| 用户反馈驱动优化 | Feedback、反馈前后留痕、学习、固化、激活、回滚和 trace 齐备；双臂对照协议已 PASS --quick | 闭环存在，稳定收益未证明 |
| 公开数据集训练和泛化 | 支持 MVTec、BTAD、MPDD、GYU-DET 等 | 基本满足，统计和证据链不足 |
| 可解释性文档 | 有热力图、缺陷框、槽位分数、类型归因、特征贡献和消融输出 | 有实现基础，正式说明书不完整 |
| 提交模型代码和使用说明 | 主干代码、用户手册、模型/实验文档、根 README 齐备 | 满足 |

---

## 三、待闭环事项

### P0 阻断项（必须在正式提交前闭环）

#### P0-1 性能指标未按赛题口径验收

- 三层预算口径与环境 manifest 工具已就绪。
- **仍待闭环**：未在 2060 GPU 实机实测。当前结论仅限开发机（GTX 1660S / CUDA 12.6）实测记录；`<200ms` 在指定硬件复验前不得写入交付承诺。

#### P0-2 视频链路证据链待标准档复跑

- 协议工具（FFV1 无损 + 原生分辨率）已 PASS --quick。
- **仍待闭环**：标准档（多品类/多 FPS/长视频/顺序错误）复跑；RTSP 实时流产线环境实测。

#### P0-3 自学习未证明反馈后稳定变准

- 双臂对照协议（固定 holdout、多种子、正确/错误反馈、回归门禁）与隔离红线已就绪，--quick 已 PASS。
- **仍待闭环**：多品类（≥3）多种子（≥3）标准档复跑；困难跨域集统计可解释改善证据。

### P1 高风险可交付性问题

| # | 问题 | 现状 |
|---|------|------|
| 1 | 跨域泛化统计不足 | 待标准档复跑 |
| 2 | few-shot/zero-shot 证据链不足 | 待标准档复跑 |

### P2 工业化加固项

- 反馈作废与已应用模型更新未形成自动事务回滚；需记录学习批次和父子模型版本。
- 模型、配置、代码、数据和运行时环境缺少强一致 hash，结果重建能力不足。
- `_archive` 中固定路径和过期脚本较多，影响交付包卫生。
- 可解释输出应补充各分数的定义、阈值来源、热力图可靠性、误报/漏报解释和适用边界。
- 应增加生产启动健康检查、模型加载状态、故障恢复、数据生命周期、第三方许可证和权重许可清单。

---

## 四、最终交付前的阻断清单

必须补齐并在结果目录落盘：

1. 2500×2500/2060/<200ms 和 CPU/<2s 的正式性能报告 —— 工具就绪（fullchain --competition），**待 2060 实机复测**
2. 异常视频、短时缺陷、顺序错误和长视频报告 —— 工具就绪（video_protocol），**待标准档复跑**
3. few-shot/zero-shot 固定协议与结果 —— **待标准档复跑**
4. 自学习前后、错误反馈、回滚和多品类重复实验 —— 工具就绪（learning_protocol），**待多品类标准档复跑**
5. 可解释性正式文档和已知限制清单 —— **待补充**

---

## 五、诚实边界声明

代码/配置/工具层落地不等同于赛题指标达标；未在 2060 GPU / 多品类标准数据集 / 产线 RTSP 环境复验前，对应赛题指标不得写入对外承诺。

**最终结论：具备参赛原型基础，赛题硬指标与多场景统计证据待复验，尚未达到工业级可交付状态。**

---

## 六、AOI_tree 数据流与工作原理

### 6.1 系统定位

`AOI_tree` 不是以树形数据结构为核心的算法，而是一套由数据登记、按品类建模、多槽位异常评分、融合决策、产线服务和反馈学习组成的 AOI 检测流水线。其主线为：

```text
数据登记 → 模型准备 → 图像切块 → 多槽位评分 → CDF 校准
→ 权重融合 → 阈值判定 → 定位/持久化 → 复核反馈 → 在线学习/新快照
```

其中：

- `AOI_Core` 是主检测系统；
- `PCB_Dual` 是传统视觉与特征模型的独立双引擎分支；
- `AOI_feature` 是独立的特征分析与 GroupKNN 工具；
- `EXPs` 是实验和离线评估代码，不能直接等同于生产 API。

### 6.2 模块级数据流

```text
文件夹 / MVTec / YOLO / 上传 / 视频
  → dataset_adapter / ingest
  → DataSource / Dataset / Image / Video
  → routes_models 选择正常图、初始缺陷图、验证图
  → DetectionEngine.prepare
  → factory.build：DINOv2 或 PDN + 启用槽位
  → Pipeline.fit
  → 快照、Model 注册、current version

图片 / 视频帧 / 产线输入
  → DetectionService
  → Pipeline.predict / predict_frame
  → 预处理与 single/tiles9/tiles36 切块
  → backbone 特征
  → sem / disc / shead / blob / trad / layout / color / inp / tpl / open 等槽位
  → 槽位分数与热力图
  → 正常分布 CDF 校准
  → 权重融合、缺陷库加分、模板规则、TTA
  → Decider：normal / gray / anomaly
  → 框、掩码、类型、Detection 落库
  → 告警或人工复核

检测结果 / 灰区复核
  → Feedback 记录与真值映射
  → DetectionEngine.submit_feedback
  → Pipeline.feedback
  → FeedbackHandler.feedback
  → 正常库/缺陷库、权重、校准或阈值更新
  → learning journal
  → consolidate
  → 新模型快照与版本切换
```

### 6.3 函数级关键调用链

#### 启动

```text
AOI_Core/main.py::main
  → start_backend
  → server.start_server_background
  → backend.api.app::lifespan
  → register_engine_snapshots
  → assign_legacy_datasets
  → PipelineService.start
```

#### 数据导入

```text
routes_data.api_import_images
  → ingest.import_folder
  → _register_batch
  → Dataset / Image
```

适配器导入为：

```text
probe_dataset
  → 前端确认
  → import_adapted
  → _register_batch
```

#### 模型训练

```text
routes_models.api_models_prepare
  → _prepare_engine
  → DetectionEngine.prepare
  → factory.build
  → Pipeline.fit
  → guard_bundle
  → _tile_and_extract
  → slot.fit
  → CDFCalibrator.fit
  → Decider.from_normal_scores
  → enable_ssocl
  → save_snapshot
  → set_current_version
```

#### 图片检测

```text
routes_detect
  → DetectionService.detect_image
  → DetectionEngine.predict_image_path / predict_ndarray
  → Pipeline.predict / predict_frame
  → _record
  → _finalize
  → _hier
  → _to_result
  → DetectionService._finish_detect
  → _persist
```

#### 产线消费

```text
PipelineService._loop
  → _tick
  → _apply_review_backpressure / _sync_rtsp
  → _consume_one
  → DetectionService.detect_image
```

#### 反馈闭环

```text
routes_review.api_review_submit
或 routes_feedback.api_create_feedback
  → submit_feedback_record
  → DetectionEngine.submit_feedback
  → Pipeline.feedback
  → FeedbackHandler.feedback
  → WAL
  → consolidate
```

### 6.4 原理上的关键边界

1. CDF 校准分数表示样本相对于训练正常分布的位置，不是严格的异常概率。
2. 当前 router 权重会记录，但实际融合路径使用的可能仍是 `self.weights`，二者存在脱节风险。
3. 最终判定并非单一线性融合，缺陷库加分、分数截断、模板强判和 TTA 都可能改变结果。
4. 多槽位共识只能表示证据一致性，不能直接等同于真值置信度。
5. 在线反馈样本通常是困难样本或灰区样本，不是正常总体的随机抽样，直接更新正常分布可能造成选择偏差。

---

## 七、问题清单：设计合理性、原理可行性与工程实现

### 7.1 设计合理性问题

1. `AOI_Core`、`PCB_Dual`、`AOI_feature` 的系统边界、运行时边界和数据口径不够清晰，容易把独立分支误认为同一检测引擎。
2. 主系统的 `normal / gray / anomaly` 与 PCB_Dual 的 `OK / NG / GRAY / ERROR` 不是严格同义关系，`is_anomaly` 也不能覆盖灰区和系统错误。
3. readiness 门槛与实际模型准备门槛不一致，可能出现界面提示数据不足但接口仍允许训练的情况。
4. 最终判定规则过于分散，融合、缺陷库、模板、TTA 和截断共同改写结果，缺少统一的 decision trace。
5. `n_tiles` 等字段承载了 tile 数、决策、槽位分数、框和引擎结果等多种语义，数据模型可维护性差。
6. 产线选图与检测落库之间没有原子认领状态，多个 worker 可能重复检测同一图片。
7. 反馈作废和模型学习效果没有统一事务边界，删除反馈记录不等于撤销已经产生的模型更新。
8. 快照仍依赖配置中的外部 backbone 权重和路径，跨机器恢复不完全可复现。
9. 传统视觉、特征模型、实验包和主系统的指标及结果格式未形成统一协议。

### 7.2 原理可行性问题

1. 正常样本过少时，CDF 分位点和阈值缺少统计稳定性；单张或极少量正常图只能证明流程可运行，不能证明阈值有效。
2. CDF 校准结果不应标注为异常概率；其范围可出现负值或大于 1，不能直接用于概率解释。
3. router 在初始缺陷上得到的收益不能直接推出生产域收益，尤其当实际融合没有使用 router 权重时。
4. 共识机制反映的是槽位相关性；多个槽位共享 backbone 或预处理时，可能共同产生系统性误报。
5. 在线权重学习将连续分数二值化，少量反馈可能显著改变结果，但没有直接优化 AUROC 或固定误报率下的召回率。
6. 反馈样本存在选择偏差，用灰区和误报样本重估正常分布可能造成阈值漂移。
7. TTA 采用较低分数覆盖异常结果时有漏检偏置，尤其对小缺陷、边缘缺陷和光照敏感缺陷。
8. 模板强判正常属于启发式否决，不是经过联合校准的证据融合；模板错位或模板版本变化可能覆盖真实异常。
9. 融合分数截断到 1.0 可能制造排序平局，损失困难样本之间的优先级信息。
10. PDN 的特征蒸馏损失不能证明异常排序、定位、阈值稳定性和跨域泛化与 DINO 等价。
11. 无标签共识不能作为正确性证明；需要有标签的独立 holdout 验证。
12. `EXPs` 中的简化 PRO 不是标准 AUPRO，不能直接与论文或工业基准结果比较。
13. 若在评估期间允许反馈学习改变内存状态，而报告没有固定快照版本，则批次评估可能不是在同一模型状态上完成。

### 7.3 工程实现与数据完整性问题

1. PCB_Dual 默认回滚路径把单个版本号当作版本列表使用，存在类型和流程错误风险。
2. PCB_Dual 双检中，传统侧的 `ERROR` 可能被另一侧的 `NG` 覆盖，实际优先级与设计意图不一致。
3. PCB_Dual 双检对传统结果缺失的输入边界处理不足，存在错误进入正常融合的风险。
4. 灰区查询依赖固定 JSON 空格格式，而另一处采用 JSON 字段解析，序列化变化可能导致查询不一致。
5. 数据集隔离主要依赖 `train / val / test` 路径字符串，不能防止同一图像、工件或批次通过不同路径泄漏。
6. 未识别图片在部分导入路径中默认归为 anomaly，可能污染训练、验证和测试标签。
7. PCB_Dual 超时只停止等待，不一定终止底层线程，长期运行可能造成资源堆积。
8. PCB_Dual PLC 取图没有原子 claim，归档移动异常还可能被忽略，容易重复处理。
9. PCB_Dual 模板 gate 通过修改和恢复全局状态实现，并发请求使用不同模板时存在相互覆盖风险。
10. 双引擎框结果直接拼接，未见跨引擎统一去重，可能重复记录同一物理缺陷。
11. AOI_feature 的 GroupKNN 统一跳过最近邻，对独立测试样本可能错误跳过真正最近的训练样本。
12. 全部槽位禁用时固定融合可能出现除零。
13. SQLite 新旧库的外键行为可能不一致，迁移逻辑也需要核验非 SQLite 配置下的启动路径。
14. `/detect/dual`、`/detect/traditional` 和 PLC 路径的落库行为不完全等价，外部系统若假设每次检测都有 Detection 记录，可能出现追溯缺口。

---

## 八、改进清单

### 8.1 P0：上线前必须完成

- [ ] 主要检测响应、历史记录、异步任务与部分统计已增加统一的 `final_status=PASS / ANOMALY / REVIEW / ERROR` 解释字段；未完成的任务返回 null，保留旧字段。尚待数据库独立状态列、全部统计和失败/复核口径集成验收。
- [x] 生产 readiness、准备和激活统一要求至少 3 张 train/normal；少样本实验需显式 `allow_experimental`，其快照不能自动激活。3 张仅为安全下限，**不构成统计有效性证明**。
- [ ] AOI_Core 主预测已记录原始/校准槽位分数、有效权重、规则原因、阈值及最终状态并沿返回/数据库 JSON 传递；各旁路入口及历史版本 trace 完整性尚待验证。
- [ ] 对 TTA、模板强判、缺陷库 boost 和分数截断分别做消融实验，确认对漏检率、误报率和定位率的影响。
- [ ] AOI_Core 本地图像增加条件 UPDATE 原子认领，claim 与工单/回队代次绑定，检测、统计及完成状态同事务写入；失败停止自动重试，须核验旧 worker 停止后凭令牌人工恢复。SQLite 模拟覆盖多工单/回队/失败恢复，但仍需真实多实例、断电和文件副作用验证。PCB_Dual PLC 文件使用 `processing/` rename 认领。
- [x] PCB_Dual 的 ERROR 优先级、双检缺失结果边界、默认回滚版本列表错误已修复。**算法线程超时仍无法强制终止，未完成资源隔离。**
- [ ] 独立 holdout 需固定模型快照、数据版本和配置。accuracy、robustness 和 A/B 评估已固定模型版本；仍缺数据/配置 hash 与完整统计复验。
- [ ] 在目标 2060 GPU、2500×2500 输入、真实 RTSP 环境完成 P95/P99 延迟、吞吐和故障恢复验证。

### 本轮实施状态与遗留风险（2026-10-03）

- AOI_Core 已修复空槽位融合、未知目录默认异常标签及 bundle 跨分区重复内容；本地图片 claim 与工单/回队代次绑定，持久化事务内完成状态更新，旧 SQLite 增量迁移。RTSP 按 `(source, workorder)` 建线程。崩溃遗留 claim 仅能在确认旧 worker 停止后人工恢复，尚非分布式 exactly-once。
- PCB_Dual 已修复双检错误优先级、默认回滚及 PLC 文件原子认领；异步模板 JSON 和标准图均提交时复制并校验 hash，按数据源/工单隔离 RTSP，模板 gate 采用请求级参数，传统框及跨引擎框聚合更稳定。超时仅限等待/状态，线程不能被强杀；已限制悬挂线程数且不再伪称重试，模型/外部资产仍需完整版本化。
- accuracy、robustness、A/B 固定模型快照，打分失败显式报错，学习回放不再从其他 test 图抽取反馈池，激活门控过滤作废及 test 反馈；快照增加内部工件 SHA-256、环境版本与可得的本地 backbone 权重 hash，加载前校验工件。AOI_feature 独立测试 GroupKNN 不再跳过真实首邻。
- **尚未完成**所有入口的统一状态数据库协议与全链 decision trace、进程级硬超时、模板全部外部依赖冻结、工件/批次级泄漏证明、跨机模型完整 provenance 及赛题目标硬件/标准数据集实验。当前轻量测试与语法验证不能替代以上验收。

### 8.2 P1：模型可信度和学习闭环

- [ ] 将 CDF 分数命名为相对异常分数，不再直接称为异常概率；如需概率，增加独立概率校准。
- [ ] 明确 router 是诊断组件还是生产组件；若用于生产，必须接入实际融合并纳入版本、日志和评估。
- [ ] 在线学习改为“反馈收集 → 影子模型 → 固定 holdout 验证 → 达标发布”，避免单条反馈直接改写生产模型。
- [ ] 为反馈引入样本来源、审核人、反馈置信度、学习批次和父子模型版本。
- [ ] 支持反馈撤销后的模型重放或精确回滚，而不是只删除反馈记录。
- [ ] 训练、验证和测试按工件/批次/采集会话隔离，并记录图像 hash、工件 ID、批次 ID 和采集 ID。
- [ ] 使用标准 AUROC、AUPRO、FPR@TPR、召回率、误报率、定位 IoU 和置信区间，替换或明确标注简化指标。
- [ ] 对 DINO、PDN、各槽位、TTA、模板规则和融合策略进行独立消融和多随机种子验证。

### 8.3 P2：数据模型和可维护性

- [ ] 拆分 `n_tiles` 等复合字段，分离检测结果、解释结果、引擎元数据和原始输入信息。
- [ ] 为模型快照写入代码版本、配置 hash、数据版本、backbone 权重 hash、依赖环境和随机种子。
- [ ] 对快照、权重、配置和数据建立强一致性校验，保证跨机器恢复和结果复现。
- [ ] 将模板参数改为显式请求上下文，禁止通过修改全局状态实现并发请求隔离。
- [ ] 统一所有检测入口的持久化协议，明确哪些请求必须生成 Detection、审计记录和原始结果。
- [ ] 对传统框和特征框执行统一坐标系转换、跨引擎去重和物理缺陷聚合。
- [ ] 将未知类别导入状态改为 `unclassified`，必须经过确认后才能进入训练或评估集。
- [ ] 统一数据库迁移和外键策略，增加旧库升级、非 SQLite 配置和并发写入测试。
- [ ] 补充启动健康检查、模型加载状态、资源占用、异常恢复、数据生命周期和第三方权重许可证清单。

### 8.4 P3：评审与交付证据

- [ ] 输出模块级架构图、函数级调用图和最终判定规则图。
- [ ] 输出每个分数、阈值、热力图和框的定义及适用边界。
- [ ] 输出误报、漏报、灰区、ERROR 和模板否决的典型案例。
- [ ] 输出多品类、多批次、多种子和跨域 holdout 结果。
- [ ] 输出自学习前后、错误反馈、反馈撤销、回滚和再次发布的完整审计记录。
- [ ] 输出性能、并发、断电恢复、重复消费、超时、模型加载失败和磁盘异常的演练记录。

---

## 九、建议的改造后目标架构

```text
统一数据与样本身份层
  → 数据质量门禁与工件级隔离
  → 版本化模型构建与固定快照
  → 特征/槽位并行评分
  → 统一校准与可审计融合
  → 统一状态机：PASS / ANOMALY / REVIEW / ERROR
  → 原子任务认领与幂等持久化
  → 反馈事件流
  → 影子学习与固定 holdout 验证
  → 达标后发布新版本
  → 可重放、可回滚、可复现
```

改造原则是：

1. 先统一数据、状态、版本和审计口径，再调算法。
2. 先固定 holdout 和生产目标，再决定在线学习策略。
3. 任何规则覆盖都必须记录原因并可通过实验关闭。
4. “流程可运行、指标有效、生产可复现”必须分别验收，不能互相替代。

---

## 十、复现链断点修复与实测验证（2026-10-11）

对照原提交包（04_其他辅助材料）做全量文件哈希对比后，确认本树曾存在 4 个"原版能复现、本树会断"的复现链断点，当日全部修复并实测通过：

| # | 断点 | 修复 | 实测验证 |
|---|------|------|---------|
| 1 | DINOv2 主干权重（88MB）被移出包，全树无 .pth | 从原提交包 assets 拷回 `AOI_tree/assets/dinov2/dinov2_vits14_pretrain.pth`，SHA256=b938bf1b… 与 `assets/checksums.sha256` 登记一致 | EXPs 与 AOI_Core 均命中"[dino] 使用随包 vendor 权重" |
| 2 | `EXPs/src/backbone/dino.py` 旧加载器：vendor 目录在但缺权重直接 raise（AOI_Core 侧 2026-10-06 已修，EXPs 未同步） | 同步回退逻辑：缺权重时降级 hub 缓存/在线下载 | 权重在包走 vendor 路径；逻辑与 AOI_Core 一致 |
| 3 | `EXPs/src/slots/trad.py` 硬依赖包外 `../../../demo4/src/features/traditional.py`（本树无 demo4） | 开发机 demo4 版提取器（SHA256 前缀 982CA052…，即产生全部随包快照的版本）内聚为 `EXPs/vendor/traditional.py`，trad.py 改候选路径（包内优先，demo4 兜底） | import 实测命中包内副本；bottle 重跑 trad 槽 AUROC=0.9143 与 U99 快照逐位一致 |
| 4 | PCB_Dual 金手指三对测试图未随包，`tests/eval_traditional.py` 静默跳过金手指用例 | 从开发机 PCB_Ins_v2 补齐 `alg_repo/gold_finger/test_images/`（102 文件） | 金手指 3 对全部检出（15/8/23 缺陷数，与历史复现一致） |

实测汇总（GTX 1660 SUPER / torch 2.5.1+cu121 / Python 3.11.9，数据集取包外 data_origin/data_local 与原口径相同）：

- **EXPs bottle 全量重跑**：fused AUROC=**1.0000**，sem/disc/shead/blob/trad/layout/inp 七个槽位分与随包 U99 快照**逐位一致**（仅 sanity 权重向量因 init_defect 抽样 RNG 漂移，结论无影响，即清单 G-9 已知现象）。产物 `commit/repo/repro_ssocl/outputs/m0/m0_mvtec_bottle_single_ssocl_verify.json`。
- **PCB_Dual 传统引擎量化基线**：缺陷 4/4 检出、正常 0/1 误报，SMT 13.6-26.1ms、金手指 510.5-906.0ms（优于历史复现 947.8-1373.4ms，与文档 535-1153ms 口径吻合）。
- **AOI_Core**：`tests/run_all.py` 冒烟全部通过。

结论：修复后本树的指标复现能力**不低于**原版（原版同样存在断点 3/4，需评审手工补齐），且在包完整性（金手指测试图随包、trad 提取器自包含、权重+校验和随包）上**优于**原版。注意 EXPs 复现仍需包外数据集（与原版相同口径）。

---

## 十一、生产模块复现验证与三处断链修复（2026-10-11，第二轮）

应"生产模块（AOI_Core/PCB_Dual/AOI_feature）必须独立复现指标"的要求，新增三个生产路径协议化 bench 工具（`AOI_Core/tests/bench/static_accuracy.py`、`learning_curve_bench.py`、`defect_classifier_bench.py`），对 38 项指标逐个走生产引擎/生产模块复现。过程中发现并修复三处**生产断链**：

| # | 断链 | 影响 | 修复 |
|---|------|------|------|
| 1 | `AOI_feature/run_eval.py` 引用 `algo.data`（AOI_SSOCL 已移除 AOI_Core/algo/data） | AOI_feature 全部功能（含 synthetic 冒烟）import 即崩 | `__init__.py` 增加 EXPs/src 兜底路径；run_eval.py try/except 双通道导入。synthetic 实测 AUROC=1.0000 ✅ |
| 2 | cpu 降配 profile：禁用主干后 `slots/open.py` E_open 对空 DINO 特征做 `T,D,G,_ = f.shape` 解包崩溃 | **CPU 降配工作点（文档 §3.2.6 三槽 blob/trad/layout）在生产引擎完全无法 prepare**（原版提交包同源隐患） | `backend/engine.apply_scenario_and_profile`：profile=cpu 时连带关闭 open 槽；`open.py fit` 空特征防御（全部为空时哨兵退化关闭） |
| 3 | cpu 降配 profile：`offline.py _anchor_feats` 对 None feats `torch.cat` 崩溃 | 同上链路第二处崩点 | 过滤 None、全空返回 None（`NormalBank.seed_core` 本就优雅跳过） |

修复后 CPU 降配首次端到端跑通：prepare v2 成功，2500² fullchain 模型段 mean 212.9ms / e2e 冷 302.3ms，优于文档 257/351ms。

生产复现关键结论（详见 commit/repo/AOI_SSOCL生产复现报告.md）：

- **静态精度逐位命中**：生产引擎 MVTec accuracy avg=**0.9793**（15/15≥0.9，与 U99 快照逐位一致）；fast 三数据集 0.9811/0.9679/0.9908 与 fast_summary **逐位一致**；24 品类 fast 全 ≥0.9。bottle 七槽位分逐位一致。
- **速度口径差距实证**：2500² 生产 fullchain 模型段 p50 174 / p95 267.3ms，P95 超 200ms 赛题预算（1660S）——与 §12.10 P0-1 自承认一致，文档 58/102/183ms 为研究脚本紧口径，生产口径系统性高约 1.7×，需 2060 实测或速度优化。
- **在线学习新诚实口径**与文档旧口径双轨登记（见 bench 报告 JSON）。

### 十一追加（同日晚间）：等待 C1 期间补做的两项验证与第 5 处断链

- **断链 5：PCB_Dual 持久化 settings.json 残留开发机绝对路径**（outputs_dir=E:\CPIPC\CGAIC\AOI_tree\...），换机后 ResultStore 初始化链式 mkdir 崩溃，run_dual 不可用。修复 `app/config.py` get_outputs_dir：绝对路径驱动器不存在时回退树内默认目录并自动改写设置。教训：运行时状态（outputs_dir 等）不应以机器绝对路径持久化进交付树。
- **E7 run_dual 服务化双引擎端到端实证**：launcher 自动拉起 AOI_Core 子进程 → FeatureClient 经 HTTP 准备 smt_solder 品类 → run_dual 同图并行传统+特征 → fuse_dual：NG 图传统检出「多锡」融合判 NG，OK 图双 OK。注意：模板必须带检测项调优参数（pair params.json），裸默认参数会漏检——建议模板创建向导提示参数标定步骤。
- **E8 accuracy 档快照 save/load 一致性实证**（m01 仅 fast 三槽的盲区补测）：全槽位（含 layout/tpl）prepare→全新引擎实例磁盘重载→5 探针图分数 max|Δ|=0.00e+00 逐位一致，persist 修复（断链 4）闭环。

### 十一追加 2（2026-10-12）：C1 学习曲线截断决策与最终结论

- **C1（GYU-DET 10 轮×30 反馈，诚实协议）按用户指令截断于 8/10 轮**。实测曲线 0.40→0.68→0.68→0.80→0.88→0.88→0.84→0.88，增益 +0.48（文档旧口径 +0.163 的 2.9 倍），第 4 轮（120 条反馈）起即稳定超越文档终点 0.7075；缺陷检出率峰值 0.9655。结论成立，剩余 2 轮不影响判定。
- **耗时根因（重要发现）**：LOO 诚实打分使在线学习成为 CPU 密集负载——每条反馈触发对增长记忆库的逐条重打分 + 门控校验（U69/U62 实时触发），单轮耗时逐轮递增（第 1 轮约 20 分钟 → 第 6-8 轮约 45-55 分钟），GPU 占用仅约 8%。这是消除自匹配泄漏的直接算力代价。产线不受影响（feedback_timeout_ms=3000 超时转后台），但**离线大批量回放/评测场景建议：① LOO 重打分增量缓存（仅重算受新样本影响的近邻）；② 门控校验降频（每 N 条而非每条）；③ 评测端 GPU 批打分**。
