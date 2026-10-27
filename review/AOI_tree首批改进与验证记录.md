# AOI_tree 首批改进与验证记录

- 验证时间：2026-10-12
- 验证根目录：`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`
- 首批验证未编辑生产代码；续验仅修改 `algo/persist.py`，对缺必要 layout 拟合态的历史快照明确拒绝加载，不猜测默认统计。HEAD 基线仍无有效可比结果。

## 2026-10-12 G1服务路径优化与巩固锁收窄（AC-066/068，生产未验收）

- 修改：
  - offline.py：clf/align 预取前移到 DINO 前向之前。
  - defect_classifier.py：向量化森林，带自检回退；修复 classes_ 映射。
  - pipeline/service.py：受限条件下 cv2 解码（JPEG/BMP 且 RGB/L，忽略 EXIF 方向）；热图/叠加 JPEG 并行写盘，返回前 join。
  - engine.consolidate(activate=False)：装载和重放移到品类锁外，持锁复查后再发布。
  - persist：backbone 哈希按 (path, size, mtime_ns) 缓存。
  - 设计说明以本记录为准（docs/old 为过时文档，不再维护）。
- 基准命令（Core 目录）：`python -m tests.bench.hard01_service --prior-result <aoi_hard01_2500_effattl5\result.json>`。结果：`%TEMP%\aoi_hard01_service_zoivzjed\result.json`。8 张工程 2500 图：engine_path p50/p95/max = 120.48/129.79/132.94ms；service_heatmap_persist = 184.55/190.11/190.51ms（上轮 229.74，最早 316–341）。冷载 2000ms 另列。8 图在两种模式下均为 anomaly/2.0，与 prior 一致。
- 口径：latency_e2e_ms 现不含 JPEG 写盘，上表为外部墙钟。图像非原生 2500，HTTP 与并发未在本轮复测。
- 测试：`python -m unittest tests.smoke.test_candidate_isolation tests.smoke.test_ext_hash_cache tests.smoke.test_g1_decode_equivalence tests.smoke.test_g1_latency_equivalence tests.smoke.test_candidate_learning_state tests.smoke.test_isolated_candidate_path tests.smoke.test_candidate_release`，结果 **72 tests / 26.711s / OK**。其中新增 3 项巩固锁测试：装载/重放期间另一线程可获取品类锁、WAL 增长时不覆盖缓存、缓存被替换时不覆盖；1 项哈希缓存失效测试；解码等价 7 项；时延等价 3 项。
- 续做：快照拆分与锁收窄
  - persist：save_snapshot 拆成三步。write_snapshot_state 在锁内执行，约 225ms，负责写状态文件并返回 meta 草稿。finalize_snapshot 在锁外执行，约 310ms，主要是 artifact 全文哈希加写 meta.json。set_snapshot_version 在 _write_lock 内执行，约 8ms，只修改 meta.json 中的 version。完整性校验不覆盖 meta.json 本身，所以改 version 是安全的。
  - consolidate(activate=False)：先写入私有 `.consolidate-*` stage，再在写锁内分配版本号并 rename；失败时清理 stage。
  - learn_candidate：进锁前先在锁外调用 `_candidate_files(cached=True)` 预热父快照哈希；锁内仍按 (path, size, mtime_ns) 复核。这样首个候选不再在推理锁内做约 240ms 的冷哈希。
  - 真实 v3 快照验证：拆分产物的 meta 字段集合与组合接口一致，load_snapshot 完整性校验通过。只有 learning_state.pkl 的哈希不同，因为它保存的是运行态。
  - 竞争基准命令：`python -m tests.bench.native_fullchain --engineering-prior <...effattl5\result.json> --mode direct --warmup 5 --rounds 32 --learning-path candidate --candidate-upgrade-parent`。结果见 `%TEMP%\aoi_native_fullchain_dziezfss\result.json`。
    - 锁等待 max：454.7 → 247.5（拆分后，max 来自候选冷哈希）→ 160.3ms（预热后，max 来自 consolidate）。
    - 候选锁等待 ≤5.5ms。
    - overlap 组 P50/P95/P99/max 为 258.1/319.0/382.6/445.2；idle 组为 212.9/236.3/298.1/344.3。
    - candidate p50 为 6098ms，consolidate 为 1684ms。
    - 262 行决策与 DB 逐行一致，learning_errors 为空。
  - smoke：137 tests OK，其中新增 2 项测试：hash/finalize 在品类锁外执行，失败时清理 stage。
  - 剩余：锁内 write_snapshot_state 约 160～225ms，其中 capture/validate 约 85、npz 约 57、pickle 约 62ms，learning_state.pkl 为 91.66MB。学习期间前台 P95 仍大于 200ms。已处理崩溃残留的 `.consolidate-*`：每次执行 consolidate 前，会清理 mtime 超过 10 分钟的残留目录，避免误删另一个引擎正在写入的暂存，对应 smoke 测试为 `test_stale_consolidate_stage_swept_fresh_kept`。`.candidate-*` 已按 candidate_events 的 prepared 引用清理（见下文“孤儿暂存清理”）。
- 未做：旧 submit_feedback 竞争 p95=2203ms（V§23）属于非生产路径。未改生产 DB/快照，未提交 git。
- 续做：候选墙钟上界（SS-C.BUDGET / AC-068）
  - `CandidateBudget` 新增 `max_wall_ms` 配置，默认 60000，是实测 candidate p50≈6.1s 的约 10 倍。计时用 `time.monotonic`，也可注入 clock。每个 `check()` 检查点都校验，超限时抛出 `CandidateDeferred(state="deferred")`，沿用已有的 deferred 与临时目录清理路径。wall_ms 不允许手工 charge。
  - 语义：`wall_clock_guaranteed="checkpoint"`。超时最迟在下一个检查点生效，单次前向、fit 或 save 不可中断，也不强杀线程。最后一个预算检查点是 `before_prepared`，所以 prepared 之后的提交不会因墙钟而中断，不破坏提交原子性。
  - 发现：如果在 `snapshot()` 中刷新 `used.wall_ms`，最后一个检查点之后的提交耗时可能让 used 超过 limit，导致 candidate_release 的审计（used≤limits）误拒已经完成的候选。因此 used.wall_ms 只在检查点更新，实际总耗时另报为 `elapsed_ms`。
  - 测试：新增 `test_budget_wall_clock`，覆盖注入时钟时的边界、禁止手工计费，以及端到端 `max_wall_ms=0` 时拒绝且生产与 `.candidate-*` 不受影响。不要 patch 全局 `time.monotonic`，否则会影响整个进程。显存上界仍为 False。
- 续做：学习与推理并行开关（learning_parallel）
  - 口径澄清：上文 overlap P95 319.0ms 不是折算值，而是 2500² 工程放大图在全链路服务（含解码/热图/DB/归档）上的实测墙钟，体现学习与推理争用 GPU/CPU。G1 的 200ms 只约束纯推理（折算到 2500² 大图），与此处口径不同。
  - 配置：`self_learning.learning_parallel`，默认 false，已加入 config `_SCHEMA`、设置中心 `_CONFIG_SCHEMA`、default/docker.yaml。引擎属性 `learning_parallel=None` 时热读取配置，bench 可用 `--learning-parallel` 覆盖。
  - 机制：predict_image_path/predict_ndarray 以 `_foreground()` 标记前台，先标记再等品类锁。串行模式下，`_learning_yield()` 在 CandidateBudget.check 的每个检查点，以及 consolidate 锁外 load_snapshot/重放前，等待前台空闲（最近一次推理结束 `learning_idle_ms=300`）。cancel 可打断等待。让行时长累计到 `_paused`，从 used.wall_ms 中扣除，避免产线持续负载把学习误判为墙钟超时；snapshot 另报 `yield_ms`。
  - 发现（死锁）：若学习线程持品类锁/写锁时让行，前台推理已计入前台、正等这把锁，双方互等。因此持任一锁（RLock._is_owned）时不让行。锁内段（序列化约160ms）仍会阻塞前台，这是串行模式的残余。
  - 发现：部分测试绕过 `__init__` 构造引擎，故前台计数等字段设为类级默认值。
  - 实测（rounds=32，同一事件集 sha 9b82565c…）：`%TEMP%\aoi_native_fullchain_j5r71khx\result.json`。hot P50/P95/P99/max = 204.2/236.7/307.8/329.7，与并行时 idle 组 212.9/236.3/298.1/344.3 持平（并行 overlap 为 258.1/319.0/382.6/445.2）。锁等待 max 0.08ms。代价：第 1 个候选在推理期间基本全程让行，墙钟 57.8s，其中预算口径已扣除让行；其余 7 个候选和 consolidate 在推理结束后跑完，分别为 4.0–4.6s 和 1.09s，比并行时 p50 6.1s 更快。学习总完成时间在推理结束后又延后 36.4s。262 行一致，learning_errors 为空。
  - 取舍：产线连续出图且间隔都不到 300ms 时，串行模式下学习会一直延后（不会饿死推理，但学习可能长时间不推进）。需要尽快学完时可在设置中心打开并行。
  - 测试：`test_learning_yields_to_foreground` 覆盖空闲不等待、前台期间阻塞、结束+idle 后放行、持锁不等、并行不等、cancel 打断、墙钟扣除让行。smoke 142 OK。
- 续做：`.candidate-*` 孤儿暂存清理（HARD-03）
  - 分析：候选失败路径本来就会 rmtree stage，只有进程崩溃才会残留。training/failed/rejected 状态的 stage 在 recover 时不会再用（training 只能拒绝），所以可以删。只有 `state='prepared'` 需要保留，recover 依赖 result JSON 中的 `stage`。
  - 实现：`_sweep_candidate_stages` 在 `_learn_candidate_locked` 开头调用，此时调用方已持有品类候选 OS 锁，不会和其他执行者竞争。它会删除不被任何 prepared 行引用的 `.candidate-*`，路径用 normcase/abspath 比较。测试为 `test_sweep_orphan_candidate_stages`。
- 续做：路由熵正则符号（CDEF6，部分实现）
  - 根因：`entropy_reg` 返回 H≥0，loss 却写成 `mil + λH`，最小化 loss 会压低熵，实际效果是鼓励坍缩。小样本实测：旧符号（等价 λ=-5）熵为 0.004，λ=0 和修正后 λ=5 的熵为 1.0986（=ln3）。
  - 修正：改为 `mil - λH`。融合分只用 `fuse(slot_scores, self.weights)`（offline.py:903），router 只用于追溯和 router_gate 记录，所以判定与精度不变。受影响的只有 router_w 追溯值，以及 router_gate 记录中的 enabled 和 router_auc。分层 train/val/test 重评尚未做。
  - 测试：`test_entropy_reg_resists_collapse` 断言 λ>0 时熵 > 0.9·ln3，旧符号会失败。smoke 145 OK。
- 续做：auto_activate 配置生效（UI-P0.4，部分实现）
  - 缺陷：设置中心 `self_learning.auto_activate`（default/docker.yaml 都是 false，含义是“人工 A/B 后再激活”）实际上没有生效。`routes_feedback` 用 `req.auto_activate is not False` 判断，None 也被当成激活；`service.consolidate` 默认 activate=True；工单“学习提升”和外部 PCB_Dual `feature_client.consolidate` 都不传参。结果是三条路径都会直接激活。
  - 修正：`consolidate(activate=None)` 改为读取配置，显式传 True/False 时优先于配置。ConsolidationBatch.status 和返回的 activated/message 随之区分 candidate 与 activated。工单回执追加“(已激活)”或“(候选,待激活)”，不再统称“已更新”。
  - 行为变化：在默认配置下，PCB_Dual 和工单学习只会生成候选版本。返回的 version 是新候选号，但生产版本不会切换，需要走候选验收和激活流程，或在设置中心打开 auto_activate。`m02_e2e` 后续断言 v2 is_active，所以在 update 设置中显式加了 `auto_activate: True`。
  - 测试：`test_consolidate_auto_activate` 共 2 项，分别验证 None 跟随配置、显式参数优先于配置。smoke 147 OK。
- 续做：文档勘误（REPORT-3）
  - 在 AOI_tree 内检索“全达标/不阻塞”。仅 `EXPs/docs/算法引擎设计.md` §7.3 写有“更新全程异步后台执行，不阻塞推理流”，与锁证据矛盾，已就地追加勘误块。其余命中项（通知推送、归档失败不阻塞、tuner 窗口）与学习锁无关，不需要改。docs/old 不维护。

## 2026-10-12 service.py 语法阻断最小修复（生产未验收）

- 仅删除 `backend/self_learning/service.py` 中 `submit_feedback_record` 提前结束 docstring 的三引号，使 M7a/M8a 说明完整位于原 docstring 内；没有回滚并行实现、修改功能或测试。未发现需追加修复的紧邻语法问题。
- 使用 `C:\Program Files\Python311\python.exe -m py_compile backend/self_learning/service.py` 通过（exit 0）；首试 `python` 别名 exit 9009，改用已安装真实解释器。AST 静态断言通过：`candidate_only=True`、`allow_legacy_online=False`，且函数首个业务分支仍以 `not candidate_only and not allow_legacy_online` 抛出 `FeedbackConflict`；未放宽安全门禁。
- 实际测试（Core 目录，`QT_QPA_PLATFORM=offscreen`、`PYTHONDONTWRITEBYTECODE=1`）：`& "C:\Program Files\Python311\python.exe" -u -m unittest tests.smoke.test_feedback_evidence tests.smoke.test_feedback_idempotency tests.smoke.test_candidate_learning_state tests.smoke.test_isolated_candidate_path -v`。逐 case 和 TEMP 阶段均有进度；**41 tests / 15.057s / FAILED (failures=8, errors=3) / exit 1**，不能沿用此前旧工作区全通过结论。
- 当前阻断已由 SyntaxError 转为测试契约差异：旧反馈证据测试预期 applied/pending/failed 与归因，而安全默认返回 candidate_queued；legacy 测试仅传 `allow_legacy_online=True` 未传 `candidate_only=False`，不应据此取消候选默认值。幂等 service 测试预期 consumed=True 失败并伴随 TEMP SQLite 未释放 WinError32；候选 service 重复请求测试取 `outputs[1]` 报 IndexError。按本轮最小语法修复边界记录，不修改上述行为或测试。（后续复核：同四模块现为 45 tests OK，上述失败已在后续轮次关闭；smoke 全量 147 OK。）
- `test_candidate_learning_state` 7 项全部通过；`test_isolated_candidate_path` 其余 15 项通过，service 重复请求 1 项错误。保留 sklearn 资产/运行版本警告，未升级依赖。仅完成语法阻断解除与安全门禁静态确认，相关 SS-A/SS-B.STATE/AC 仍按部分实现与生产未验收处理，不勾生产验收。未提交、未恢复删除文件、未修改生产 DB/快照。

## 2026-10-12 候选学习态续验（AC-041/043，生产未验收）

- 改动persist、FeedbackHandler、新learning_state、router及engine候选load；未编辑self_learning/service.py或routes。已有slot头/校准/阈值/权重/router/双库/gate/WL持久化，原handler.json不是续训态。新增v1显式字段payload：正常/框训练缓存、反馈/prequential/蓄水池缓存、计数与失败冷却、权重先验/去重、锚定路径及内存特征/基准、正常训练路径/CDF历史、候选流窗口、局部NumPy与CPU Torch generator；未孵育HeadManager也保存。LOO分数派生缓存失效重算；线程/锁/运行stats/日志/中途回滚临时副本不作为续训状态；Adam/SGD每次微调新建，无跨轮optimizer对象。
- 候选load严格要求manifest v1、哈希和必要工件；无manifest旧候选明确legacy不可恢复，不猜。显式字段集合、计数/有限数、训练buffer维度、head键及shape检查；所有新pickle加载走safe_pickle白名单。白名单是可信路径控制，不是受限unpickler，字段检查在反序列化后，不能据此宣称抵抗白名单内恶意pickle代码执行；快照必须来自可信生产者。
- 候选训练Dropout使用候选CPU generator，不恢复全局Python/NumPy/Torch/CUDA RNG；router候选不调用manual_seed，生产路径保留。生产加载仍清流窗口。候选局部Dropout与生产全局Dropout的随机流不承诺相同，只证明候选续训左右轨迹；非普通Dropout随机层明确拒绝，未知自定义随机算子/CUDA非确定性/跨设备跨版本bitwise未支持。factory初始化可能消耗全局RNG，未采用全局保存恢复（避免其他线程回退），不声称加载对全局RNG完全无影响。
- 最终命令：Core目录 `py -3.11 -m unittest tests.smoke.test_candidate_learning_state tests.smoke.test_feedback_ft_equivalence tests.smoke.test_candidate_isolation -v`；**15 tests / 3.866s / OK / exit0**。新增7项TEMP真实SheadHead/Handler：三次save/load后两epoch真实loss/backward/Adam.step梯度与参数逐值相等、蓄水池下一轮路径相同、门控否决回滚及冷却续存、真实feedback六条续流计数/preq/缺陷库相等、unknown/legacy/缺字段/恶意覆盖字段/wrong buffer shape拒绝、必要工件缺失及白名单外拒绝、router不seed全局。feature/factory和反馈打分使用轻量替身，正常回流关闭，非生产数据/完整跨域回放证明。
- 扩展4模块曾运行 **30 tests / 9.681s，29通过、1错误**：test_real_service_candidate_queue_and_trace导入被并行工作区service.py:169中文冒号SyntaxError阻断，按隔离要求未修改该文件。其余15项隔离候选（故障/预算/父WAL/OS锁等）通过。保留sklearn资产1.9.1/运行1.9.0警告；未更改依赖。初轮测试方法名错误、cal_hist要求所有槽位而真实默认空字典、夹具anchor文件不存在已最小修正。
- 所有新工件TEMP，未写生产DB/快照、未自动发布、未提交、未恢复删除文件。AC-041/043及SS-B.STATE保持部分实现和生产未勾选；完整disc续训、正常回流/跨域数据、GPU/G1/G2/G3/G6与自定义随机模块仍须后续独立验收。

## 2026-10-12 E_open队列续验（AC-052，生产未验收）

- 先查正确AOI_SSOCL子仓库：open_alerts.py、open_alert_queue.py、test_open_alert_queue.py已存在且未跟踪，pipeline/service.py已有工作区修改；不重建上一轮实现。原队列测试1 test / 0.542s / OK。已读API、反馈页嵌入UI、检测engine→service落库、SL排除open_alert反馈的巩固入口及AC-052。
- 本轮仅补处置API `require_role("engineer")`，增加测试：真实ApiKeyMiddleware缺/错密钥401、operator读200/写403、engineer写及admin重试幂等；真实_to_result→_finish_detect→_persist关联模型7、原分3.2/阈值2.5、图像路径；分页/品类/工单/状态过滤、空白拒绝、五类处置、事件冲突/旧revision、责任人时间原因、反馈未消费/not_requested及关闭连接重建客户端后持久状态。未新增UI页面或迁移。
- 命令（Core目录）：`$env:PYTHONDONTWRITEBYTECODE='1'; $env:QT_QPA_PLATFORM='offscreen'; py -3.11 -u -m unittest tests.smoke.test_open_alert_queue tests.smoke.test_detection_version tests.ui.test_gauge_contract -v`。最终 **21 tests / 1.454s / OK / exit0**，含队列2、版本11、Gauge8；有逐case及阶段进度。新增测试先两次失败：_to_result参数签名错误、Mock._dominant_slot进入严格JSON；修正测试夹具为真实轻量engine实例后通过，不放宽生产JSON契约。保留Starlette/httpx弃用及sklearn资产1.9.1/环境1.9.0警告，未安装/改依赖。
- 范围：自建TEMP SQLite、合成检测记录及管线契约、Qt offscreen；图像路径是证据引用，未验证真实图像下载/渲染、真实模型推理精度或完整生产app联调。未执行真实并发竞争/多进程重启。API仍沿用无密钥开发透传admin口径；责任人是人工填报，不是身份认证。未知历史版本/原分/阈值不从当前模型猜测。
- 处置只留存Feedback(consumed=False, learning_status=not_requested)，不会训练/发布；模板候选只是人工状态，尚未准入模板库。噪声/漂移分类、候选训练/批准/效果关联、完整资产血缘及真实桌面/G1/G2仍开放。AC-052及相关台账仅部分实现、生产框不勾。未写生产DB/快照、未提交、未恢复删除脚本；只定点编辑现有文档以保留并行工作。

## 2026-10-12 候选独立验收与发布最小闭环（AC-039/040/043/050/053）

- 新增 `backend/self_learning/candidate_release.py`、`backend/api/routes_candidates.py` 与 `tests/smoke/test_candidate_release.py`；引擎候选 activate/rollback及app注册接入。状态为 unavailable/validated/approved/rejected/activated/rolled_back，持久报告、决定和prepared/committed/aborted事件位于引擎receipt SQLite。只有明确批准且父current/WAL/工件/报告/数据字节未变、未过期才发布；旧force不能绕候选门禁。
- 父快照独立重放匹配父WAL，再与候选按新旧域实际predict对比，记录逐图FPR/recall/gray/predict时间、预算和图像SHA256，绑定candidate_id、父版本、WAL、截止ID、配置hash。缺双域标签或冻结政策为unavailable，不生成假指标。回滚只允许已激活候选可信父，独立重放不写事件收据，不清WAL；记录目标和事件水位。
- 装载和回滚重放在协调锁外，锁内复查状态/父/WAL/工件并预写事件后切current和缓存。显式recover补记指针已切的事件，未切则aborted，水位冲突需人工处理；未决事件阻止再次发布。检测返回实际推理版本。
- 实际命令：`& "C:\Program Files\Python311\python.exe" -u -m unittest tests.smoke.test_candidate_release.CandidateReleaseTest tests.smoke.test_isolated_candidate_path.IsolatedCandidateTest -v`，最终 **24 tests / 10.714s / OK / exit 0**（8新增发布测试、16既有候选回归）。覆盖未批准、拒绝/缺数据、重复冲突、父/数据/WAL/工件变化、过期、激活版本、回滚水位、重启及两个发布故障点恢复。逐case和预算/恢复阶段输出进度。TEMP真实SQLite和快照，factory/feedback/predict替身，不是真实算法精度。
- 初轮5项因SQLite连接未关闭全部在TEMP清理失败；Python连接上下文只管理事务，不会close，已finally关闭后复跑通过。曾无效python别名exit9009及旧E盘解释器不存在；采用上述真实解释器。sklearn 1.9.1工件/1.9.0运行版本警告、Pydantic与Starlette TestClient弃用警告仍存在，未升级依赖。
- 最小HTTP TestClient仅测试人为注入角色的403/200/409，无生产lifespan；**没有完成真实HTTP认证密钥/代理/桌面联调、验收后台任务全链业务SQLite、真实新旧域G2、原生2500 G1、GPU/墙钟硬预算、多进程缓存一致、完整学习态/RNG证明**。现有require_role鉴权关闭时允许admin，账本actor为既有角色而非实名。候选新接口以engine current+receipt账本为权威，首轮未同步业务Model.is_active，本轮最小镜像续做见下；没有自动启动恢复，须管理员显式recover。数据只检查登记split/path及hash，不证明批次/近重复隔离。生产DB/快照未改，全部生产AC保持未勾选。

### 2026-10-12 续做：最小可恢复业务Model镜像（部分实现，生产未验收）

- 改动 `backend/self_learning/candidate_release.py`、`backend/self_learning/service.py`、`tests/smoke/test_candidate_release.py`；`backend/engine/__init__.py`仅rollback目标块增加已提交同目标重试分流，E_open/Feature代码未改。候选REST无需另行同步：switch已完成账本commit后才同步，同目标已提交激活/回滚重试不新增发布事件；recover也检查最近实际提交目标并修复镜像。没有自动发布或启动恢复，验收/批准/过期/父WAL门禁不降低。
- 发现与策略：发布账本与业务SQLite是两个提交域，不能把发布成功等同业务镜像成功。同步失败抛明确ValueError（候选REST既有409），错误保留“发布已提交、实际版本vN、请显式recover”；业务事务回滚不撤销真实指针、不伪称成功。最新prepared或实际指针与提交目标不符时拒绝猜测；prepared恢复失败不会更新镜像。账本提交后/镜像前新增release_committed故障点；无prepared的显式recover也可修复提交后窗口。
- 同品类按精确`vN + engine_snapshot`选择最小ID目标，其他所有格式/重复行active清零；保留已登记目标version/path/metrics。历史缺行只补实际目标快照身份及发布事件标记，不填精度、数据集血缘或独立验收指标，不做全库register扫描/孤儿治理。其他品类不修改。业务DB访问只持管理写锁，不持品类推理锁，不持发布SQLite连接；多进程外部指针/缓存协议仍待验收。
- TEMP真实业务SQLite与真实快照：新增7项覆盖业务commit失败和重启显式恢复、回滚同步失败/幂等、提交后crash/历史缺行、prepared指针窗口/外部指针拒绝、发布账本SQLite触发器故障恢复、重复Model行service不覆盖唯一镜像、另线程可取得推理锁。业务commit故障注入Session.commit，发布故障使用真实SQLite RAISE(ABORT)触发器；其他品类隔离嵌入镜像断言。
- 实际合并命令：`python -u -m unittest tests.smoke.test_candidate_release.CandidateReleaseTest tests.smoke.test_isolated_candidate_path.IsolatedCandidateTest tests.smoke.test_candidate_isolation tests.smoke.test_feature_revision tests.smoke.test_feedback_evidence tests.smoke.test_feedback_idempotency -v`；**61 tests / 15.849s / OK / exit 0**，其中发布15项、隔离候选16项、既有consolidate隔离4项、Feature门禁8项、异步反馈证据8项、幂等反馈10项。逐case及预算/恢复阶段进度输出，后台终端执行。初轮新增5项与既有发布8项共13 tests / 5.616s通过；后补账本故障/重复行2项后合并通过。
- 仍有既有sklearn版本与Starlette弃用警告，未升级依赖。所有DB/快照均TEMP，未写生产存储、未恢复删除脚本、未提交；生产AC保持未勾选。本轮只补业务镜像协议，不证明真实新旧域精度、G1/GPU预算、生产HTTP鉴权、多进程缓存或完整学习态/RNG达标。

## 1. 轻量验证结果

| 项目 | 结果 | 证据 |
|---|---|---|
| `test_detection_version` | 通过，11 tests | 版本追溯、落库使用实际推理版本、特征元数据契约均通过 |
| `test_feedback_ft_equivalence` | 通过，4 tests | 反馈微调数学等价、梯度/损失、Dropout、空 anchor、BatchNorm 与设备传输优化均通过 |
| m01 engine smoke | 通过 | 合成数据准备、save/load、一轮反馈、v1/v2 巩固与学习曲线均通过；预测示例 75.8ms，反馈示例 485.19ms（单样本 smoke，不等同 p50/p95） |
| m15 feedback smoke | 通过，9 项断言 | uncertain 反馈、未准备模型降级、主动选样空态、导出端点通过 |
| m14 template smoke | 原断言失败，已定位为测试契约不符 | 缺陷注册为 test 而非 train_anomaly，prepare 的 init_defect 为空，train_auroc=None 合法；未扩展修复 smoke |
| m02 e2e smoke | NaN 修复后完整通过 | 续验 `python -u -m tests.smoke.m02_e2e` exit 0，M2/M2b/M5a/M6a/M7a/M8a 全链路通过，贡献 latest JSON 不再因 NaN/Inf 失败 |

测试过程中的长步骤均打印了阶段/样本进度。测试生成的数据和临时存储位于系统 Temp，不写入生产存储。

## 2. 已确认的修复证据

- 先前 anchor Tensor 布尔判断问题已由 `test_shead_empty_anchors` 覆盖并通过：None/空 anchor 不再触发 Tensor truth-value evaluation。
- 先前 Dropout 语义问题已由 `test_shead_dropout_loss_gradient_transfers_and_gate` 覆盖并通过；同时验证了 dropout、梯度、loss、门控结果与旧实现严格一致。
- feedback 特征搬运优化通过等价性测试：旧实现每轮重复搬运拼接后的 `(40, 4)`，新实现不再重复该搬运，参考 anchor 仅拼接/搬运一次；结果、梯度、参数、统计及回滚日志保持一致。
- 推理版本追溯及生产特征快照元数据契约测试均通过，包含未知字段兼容与 legacy 缺失字段报告。

## 3. 首批真实图片/缓存模型基准状态（历史；续验结果见 §5）

已发现真实图片：`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core\storage\archive\gold_finger\20261011`；已发现快照：`storage\engine\snapshots\gold_finger\v1/v2`、`smt_solder\v1-v4`，以及随包 `assets\disc_pretrain.pt`。

本轮未宣称达到“单图 <1 秒”或持续学习时间有界，原因是固定 seed 的 HEAD/工作区同样本对照基准未能形成有效可比数据：

1. 临时复制快照后以 `gold_finger` 加载时，运行时报告“品类尚未准备模型”，未得到可用的完整样本结果。
2. 直接尝试真实图片推理时，工作区路径在 layout 槽位触发 `AttributeError: LayoutSlot has no attribute count_median`，无法完成端到端计时；该过程未修改生产代码。
3. HEAD 临时 worktree 基准因其检出内容缺少可导入的 `backend` 包（`ModuleNotFoundError`）而无法执行。

因此没有伪造 p50/p95、分数或 decision 对照结果。当前仅有 m01 smoke 中单次预测 75.8ms 和单次 feedback 485.19ms 的非基准证据；不能外推为 p50/p95 或达标声明。完整基准仍需先解决快照类别/路径装载一致性、layout 快照兼容性，以及 HEAD worktree 的完整工程依赖问题，再以相同快照和相同图片重复运行，端到端计时必须包含 IO decode 且不得缓存特征。

## 4. 剩余风险与后续门禁

- 候选 router 目前已隔离并在元数据契约中验证 `candidate_router.enabled=False` 不污染实际 `actual_router`；仍需在真实模型上做候选/实际路由长期对照。
- m02 曾暴露的常量输入相关性 NaN JSON 失败已修复，续验全链路通过；运行时仍可打印 ConstantInputWarning，不等同于 JSON 非法。全量 LOO/真实精度未在本轮执行。
- m14 的 `train_auroc=None` 属于无 init_defect 时的合法契约，当前 smoke 的非空断言不符；保留测试失败事实，本轮不扩展修复。
- sklearn 运行时出现 1.9.1 训练对象在 1.9.0 加载的兼容性警告；本轮未改变环境，真实精度基准前应固定依赖版本。
- 本轮 smoke 使用合成数据兜底，不能代替真实数据准确率回归。

## 5. 续验：历史快照拒绝与真实小样本性能（2026-10-12）

### 5.1 layout 根因及最小修复

- 检查全部 6 个现存快照的 slot_state：gold_finger/v2 启用 blob/trad/layout，但 layout 的 count_median、count_std、area_hi、spacing_median、spacing_std、sym_median、sym_std 全部缺失；gold_finger/v1 与 smt_solder/v1-v4 仅含 sem/disc/shead，不存在该 layout 缺失问题。
- `layout.py:80-96` 的 fit 从正常训练图连通域数量、面积、间距与对称性计算统计；旧快照的 norm/centroids/patterns/PCA 键无法可靠推导这 7 个量。当前 persist save 已包含这些字段，但 load 原先跳过缺字段，直到 score 才 AttributeError。
- 最小修复：`algo/persist.py::_restore_slot_state` 对 layout 的缺失/None 必要统计抛 ValueError，列出缺字段并要求原训练数据重新 prepare。不补默认 0、不关闭槽位、不用未知标签归档图重拟合。完整当前状态按原值恢复，打分逻辑不变。
- 临时脚本测试通过：10 个无效状态（空、旧 PCA、None、逐一缺失 7 键）均明确拒绝；完整 fit→提取→restore 的 3 图 layout score 精确一致。另以现存 gold_finger/v2 调用完整 load_snapshot，确认在 load 阶段明确拒绝而不是预测时崩溃。

### 5.2 真实图片计时结果

使用现成可完整加载的 gold_finger/v1，复制到 TEMP 后单独设置 v1 激活指针，生产 current.json 未修改。实际槽位 sem/disc/shead；并非 v2 CPU/layout 模型结果。

- 隔离目录：`C:\Users\imia\AppData\Local\Temp\aoi_real_verify_p9l0tfzm`。
- 脚本：`C:\Users\imia\AppData\Local\Temp\aoi_real_small_verify.py`；结果：隔离目录下 `result.json`。
- seed=42（random/NumPy/Torch/CUDA），GPU NVIDIA GeForce GTX 1660 SUPER，CUDA；模型装载另计 1.811s。
- 从 gold_finger/20261011 归档复制 det4_bench_00 至 det11_bench_07 共 8 张图，SHA256 确认内容互不重复。图片未缩放改写；每次预测前清空 `_item_cache`，计时围绕 predict_image_path 且 CUDA synchronize，包含磁盘读取、decode、前处理、特征提取、推理与结果包装。不包含模型加载、HTTP/DB/热图文件写入；操作系统文件缓存未强制清空。
- 一次首图预热（同样无特征缓存）312.485ms，单独报告，不混入下面 8 图分位统计。

| 样本 | 端到端 ms | score | decision |
|---|---:|---:|---|
| det4_bench_00 | 231.978 | 1.721289 | anomaly |
| det5_bench_01 | 236.819 | 2.000000 | anomaly |
| det6_bench_02 | 238.886 | 1.568377 | anomaly |
| det7_bench_03 | 245.682 | 2.000000 | anomaly |
| det8_bench_04 | 251.814 | 1.555363 | anomaly |
| det9_bench_05 | 242.714 | 1.665980 | anomaly |
| det10_bench_06 | 250.409 | 1.578137 | anomaly |
| det11_bench_07 | 245.545 | 1.584644 | anomaly |

- **8 图 p50=244.129ms，p95=251.322ms**（NumPy percentile 线性插值）。仅是这台机器/此快照/8 图的工程时延证据，不能外推为全品类持续达标。
- **单轮 feedback wall elapsed=160.850ms，内部 feedback_ms=159.22ms**，清特征缓存后调用 submit_feedback(verdict=none,label=None)，真实动作为 high_conf_cache，fuse_blocked=True，未转后台。归档图缺可靠真值，不虚构正常/缺陷标签；此轮不是判别头微调最坏路径，不能证明重训练耗时上界。
- 无有效 HEAD 同快照同图基线，且归档图无可靠真值：**准确率不退化未证实**。全部预测 anomaly 仅记录结果，不解释为准确率。

### 5.3 m14 契约与 m02 复跑

- m14 在 tests/smoke/m14_template.py:95-97 把缺陷放 split=test；routes_models.py:735-740 仅选 train_anomaly 构造 init_defect（避免 test 泄漏）。offline.py:410-425 明确只在 def_items 非空时计算 train AUROC，否则保持 None；engine learning_insight 原样透出。故这是 smoke 断言/夹具与契约不符，不是生产返回值必须补假的 bug。本轮不改 m14。
- m02 完整复跑退出码 0：M2 + M2b + M5a + M6a + M7a + M8a 全部通过，包含 contribution latest、学习回放、反馈消费、巩固、版本切换、复核、导入清理等；常量相关性警告仍出现但不再导致非法 JSON。隔离 DB/storage：`C:\Users\imia\AppData\Local\Temp\aoi_m2_q8pjoebc`，合成数据：`C:\Users\imia\AppData\Local\Temp\aoi_synth_favlbnpx`。
- 所有长步骤与每张真实图均打印进度；未跑全数据，未修改生产快照/数据，未提交 git。

## 6. 带真值静态精度小样本基线与 HEAD 对照（2026-10-12）

### 6.1 协议、隔离与命令

- 已确认 `D:\CGAIC\data_local\gold_finger` 存在，包含 train/good、test/good、test/defect 与部分 ground_truth 掩码。本轮使用目录给出的图像级正常/缺陷真值，不计算像素级指标，也不使用文件名配对信息。
- 使用现有 `tests/bench/static_accuracy.py`，dataset=datalocal、category=gold_finger、seed=42、max-test=20、profile=fast、scenario=L1a。实际 init_normal=68（正常训练池不足 100）、init_defect=30、test=20，其中正常 3、缺陷 17；init_defect 与 test 无交集。两端完整 bundle 的 JSON(sort_keys=True) SHA256 均为 `17ff84f43c23d3352bd14ac4407982fab9ba9e7f8035ae63786173d16105e31a`。
- Python 为 `C:\Program Files\Python311\python.exe`（py -3.11），Torch=2.5.1+cu121，device=cuda。WindowsApps 的 python.exe 别名没有执行实际 Python 输出，因此命令显式改用 py -3.11；未安装/升级依赖。
- 工作区命令（cwd=`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
py -3.11 -u -m tests.bench.static_accuracy --dataset datalocal --categories gold_finger --seed 42 --max-test 20 --profile fast --storage "$env:TEMP\aoi_static_20261012_workspace_1740" --out "$env:TEMP\aoi_static_20261012_workspace_1740.json"
```

- HEAD=`7f8b655cfaa6aa5e14d4e647bd908631e903052b`。通过 `git ls-tree -r --name-only HEAD` 列出必要的 algo/backend/configs、assets/disc_pretrain.pt、EXPs/src/data，再逐文件 `git show HEAD:<路径>` 提取共 103 文件到 `C:\Users\imia\AppData\Local\Temp\aoi_static_head_20261012_nbzmfpzs`。未检出/恢复工作区文件，未触碰用户删除的旧 TEMP 脚本。HEAD 没有本轮未跟踪的 static_accuracy.py，因此仅复制当前基准脚本作为两端相同 harness，并补 TEMP tests 包标识；生产引擎、配置、协议加载器均来自 HEAD。
- HEAD 使用只读共享冻结 DINO vendor（`DINOV2_LOCAL_DIR=D:\CGAIC\AOI_SSOCL\AOI_tree\assets\dinov2`）；HEAD 的 disc_pretrain.pt 与工作区 SHA256 相同。未复制共享 defect_clf.pkl，HEAD 按自身逻辑从单类缺陷初始化后报告“类型判别器单类跳过”；工作区加载共享全局类型判别器。这是执行资产差异，因此此处对照仅用于图像分数指标，不宣称类型归因完全等价。
- 新建隔离提取/执行脚本 `C:\Users\imia\AppData\Local\Temp\aoi_static_head_extract_20261012.py`，运行命令 `py -3.11 -u "C:\Users\imia\AppData\Local\Temp\aoi_static_head_extract_20261012.py"`。其子进程 cwd 为隔离副本的 `AOI_tree\AOI_Core`，实际命令：

```powershell
& "C:\Program Files\Python311\python.exe" -u -m tests.bench.static_accuracy --dataset datalocal --categories gold_finger --seed 42 --max-test 20 --profile fast --storage "C:\Users\imia\AppData\Local\Temp\aoi_static_head_20261012_nbzmfpzs\storage" --out "C:\Users\imia\AppData\Local\Temp\aoi_static_head_20261012_nbzmfpzs\head_result.json"
```

### 6.2 真实结果

两次均 exit 0；终端输出引擎/数据、正常与缺陷提取、每槽 fit、shead 每 10 epoch、fusion/router、eval 20/20 与 summary 阶段进度。HEAD 提取也每 20 文件打印进度。

| 指标 | 当前工作区 | 隔离 HEAD |
|---|---:|---:|
| test 数量（正常/缺陷） | 20（3/17） | 20（3/17） |
| fused AUROC | 0.000000 | 0.000000 |
| fused AP | 0.6886341781511148 | 0.6886341781511148 |
| sem AUROC | 0.0392156862745098 | 0.0392156862745098 |
| disc AUROC | 0.6666666666666666 | 0.6666666666666666 |
| shead AUROC | 0.000000 | 0.000000 |
| fused_max AUROC | 0.029411764705882353 | 0.029411764705882353 |
| benchmark ms_per_image | 234.33982133865356 | 191.31317138671875 |
| 品类总 elapsed_s | 38.5 | 32.4 |

- 工作区 JSON：`C:\Users\imia\AppData\Local\Temp\aoi_static_20261012_workspace_1740.json`；新准备的 v1 快照仅写到同名 TEMP storage。
- HEAD JSON、完整 stdout/stderr 与提取 manifest：隔离根目录中的 `head_result.json`、`head_run.log`、`manifest.json`。HEAD 新准备 v1 同样只在隔离 storage。
- `ms_per_image` 是现有 Pipeline.evaluate 全循环 wall time / 20，含 predict 及样本收集与阶段输出，不含 prepare；并非逐图计时/p50/p95，也未新增 CUDA synchronize、缓存清空或重复交替运行。不得据两次均值宣称工作区变慢已被稳定证实或证明持续单图 <1 秒。

### 6.3 发现、警告与边界

- **本轮 HEAD 工程已成功运行，无工程阻塞；此前 §3/§5 的“无有效 HEAD 基线”只适用于历史快照/归档图协议，本轮新增的是 fresh prepare 带标签小样本协议对照。**
- 该固定 20 图上所有报告精度指标完全一致，未观察到相对 HEAD 的图像级 AUROC/AP 退化；但两端 fused AUROC 都是 0，不能解释为达到精度目标。train 域 fused AUROC=0.7657 与 test 域 0 差异明显，值得后续独立调查；本轮没有定位根因，不据此修改算法或重选样本。
- 正常仅 3 图、缺陷 17 图，AP 的无信息阳性比例为 0.85；实际 AP≈0.6886 不应脱离类别比例单独解读。小样本不能证明全量/全品类准确率不退化。
- 两端都出现 xFormers not available（SwiGLU/Attention/Block）警告；工作区额外出现 sklearn DecisionTreeClassifier/RandomForestClassifier 训练版本 1.9.1、当前加载 1.9.0 的 InconsistentVersionWarning。没有致命异常；兼容性警告仍是后续依赖固定门禁，未隐瞒或自动改环境。
- 本轮只写指定验证文档和新建 TEMP 隔离工件，未修改生产代码/快照/数据，不恢复旧 TEMP 脚本，不写 `d:\CGAIC\review`，不提交 git。

## 7. 持续学习重路径单轮计时（2026-10-12）

### 7.1 协议与隔离

- 优先检查 m01_engine、m15_feedback、self_learning/service 与 ssocl/feedback：m01 单条正常反馈、m15 未准备引擎的 uncertain 落库都不能覆盖头微调；本轮实际调用 DetectionEngine.submit_feedback，不用无标签缓存路径冒充持续学习。
- Python=`C:\Program Files\Python311\python.exe`，Torch=2.5.1+cu121，CUDA / NVIDIA GeForce GTX 1660 SUPER。未安装/升级依赖。random/NumPy/Torch/CUDA seed=42。
- 新建独立脚本 `C:\Users\imia\AppData\Local\Temp\aoi_feedback_worst_20261012_new.py`；成功运行隔离目录 `C:\Users\imia\AppData\Local\Temp\aoi_feedback_worst_20261012_09nu96gl`，完整阶段数据、回滚与门控历史保存在其 `result.json`。没有恢复用户删除的 TEMP 脚本。
- 共 **25 张合成图**（256×256，复用 tests.testdata._texture/_add_defect）：10 正常初始化 + 3 缺陷初始化 + 6 框标注缺陷反馈 + 6 正常反馈。train seed=0..9，初始缺陷100..102，反馈缺陷200..205，反馈正常300..305；不使用全量数据。缺陷反馈框为归一化 `[0.15,0.15,0.85,0.85]`，用于工程触发，不作为定位精度标注。
- scenario=L1a/profile=fast，真实 sem/disc/shead 与冻结 DINO；10 张正常图成为锚定集。prepare 另计 **6535.020ms**。独立 storage 下新建/激活快照，生产快照和 current 指针未修改。
- **小样本触发配置并非默认规模**：shead min_pos/min_neg 从默认20/40降到6/6；disc 前11轮临时提高触发阈值至1000000/1000000，仅延迟训练、不伪造缓冲，第12轮恢复默认60/120 patch门槛。两头真实采集反馈特征，epochs=3、lr=1e-4、优化器/损失与门控保留默认值。该协议测的是可触发的重路径，不是默认20+40反馈或最大24条回归缓存的全局耗时上界。
- 每轮清空 `_item_cache`，以 perf_counter + CUDA synchronize 计时；内部方法包装仅用于进度/计时，不替换算法。终端打印prepare、每轮和每阶段起止。timeout_ms=3600000，等待真实worker完成，未转后台；不把异步返回耗时当训练总耗时。统计包含新反馈图读取/提特征、学习与WAL；不含HTTP/DB以及服务入口反馈前后两次predict。

### 7.2 第12轮：本次观测最慢单轮

| 阶段（包含内部子阶段） | ms | 实际覆盖/结果 |
|---|---:|---|
| 正常回流 `_reflow` | 2572.063 | CDF/正常库/sem候选更新、缺陷召回保护、anchor gate；上行drift=0.1187，回滚；召回1.0→1.0 |
| 上项内部 anchor `check` | 1087.609 | **已包含在2572.063内，不另加** |
| 权重检查 `_maybe_apply_weights` | 2243.975 | 内含 LOO AUC重打分2243.686ms；初始train AUC=1.0，高质量冻结保护，无权重写回/单纯形搜索 |
| 双侧阈值 `_maybe_pair_thresh` | 2767.856 | 内含 LOO重打分2767.303ms；本轮没有新的阈值写回 |
| shead `_maybe_head_ft` | 1322.911 | 6正/6负图级特征、3epochs训练、锚定正则及定向门控；signed=-0.2061，接受；原始shead锚定均值移动-0.0201 |
| disc `_maybe_disc_ft` | 2478.750 | 真patch缓冲3epochs训练；训练前LOO AUC2177.390ms；锚定漂移绝对值0.3701超定向限，恢复旧disc参数，冷却10反馈 |
| 其余IO/提特征、收集、日志、线程/WAL等残差 | 285.037 | 总wall减以上5个顶层阶段；不是独立方法计时 |
| **单轮总wall** | **11670.592** | 引擎内部 feedback_ms=11669.29；action=reflow_rolled_back |

- 嵌套阶段不能相加重复计数：第12轮 `_loo_records` 共3次，独立子调用合计 **7188.379ms**（权重2243.686 + 双侧阈值2767.303 + disc训练前2177.390），均已包含在对应顶层阶段里。
- 12轮wall分别为：779.030、775.077、797.729、826.909、846.327、863.849、2163.842、5301.158、6074.034、7048.942、7686.116、11670.592ms；反馈序列总计约 **44833.605ms**，加prepare约 **51368.625ms**（不含进程导入/数据生成）。第12轮同时执行LOO、anchor门控、shead与disc真实微调，不是历史160.850ms的无标签缓存结果。
- 6条缺陷全部通过入库锚定保护；前5条正常反馈接受，第6条正常回流回滚。shead接受1次；disc训练后拒绝1次。整个序列无router微调、无weight_apply、无height_recal和单侧thresh_recal；双侧阈值在第10轮成功写回1次。

### 7.3 重要发现与未覆盖项

- **小样本单轮已约11.67秒**，不能宣称重学习<1秒或持续学习耗时有界；主要开销来自重复LOO重打分/锚定推理，而非仅头训练。阶段包装增加同步与终端输出开销，本轮是一次可复现工程观测，不是p50/p95或严格全局最坏上界。
- gate.history中的 `ok=false` 不能直接等同最终回滚：正常真值回流采用定向保护，下行允许接受。本次前3条正常回流的对称check虽false，实际被接受；最终回滚必须结合action/rollback_log。第12轮上行被拒，shead下行被接受，disc超限被拒。
- disc门控失败时不会执行训练后LOO AUC（源码仅在anchor门控ok且训练前AUC存在时执行）；因此本轮覆盖disc真实训练和回滚，**未覆盖disc接受后第二次LOO AUC**。完整接受支路可能更慢，不能把11.67s作为其上界。未人为放宽保护门控来强行接受。
- router对象存在，但prepare报告actual enabled=False（base/router AUC均1.0、收益不足）；本轮反馈为correct，没有漏检difficult队列，因此router微调未触发。单纯形权重搜索被高质量初始模型冻结保护阻止，未覆盖；未关闭保护伪造触发。HeadManager孵育、最大库/最大缓存、默认20/40 shead规模、并发锁等待、完整HTTP/DB/pre-post predict、巩固与生产数据精度均未测。
- 首次prepare被数据域隔离保护拒绝：init_normal放在扁平文件名train_00.png不等于train域；已改新脚本数据目录为train/good，无绕过guard_bundle。xFormers缺失与sklearn 1.9.1→1.9.0资产兼容警告仍出现，不是致命失败，未改依赖。
- 本轮未修改生产代码/快照/原始数据，未恢复删除脚本，不写 `d:\CGAIC\review`，未提交git。后续改进应优先验证同反馈内部重复LOO可否安全复用（模型/库/校准/权重变动时失效），并在既有数学等价与回滚保护测试下评估；这里只记录方向，未改算法。

## 8. 候选巩固隔离修复（2026-10-12）

- 发现：`SelfUpdateRequest.auto_activate` 原本虽已存在，但 `/self_learning/update` 没有消费该字段，`engine.consolidate()` 总是切换 `current.json`；候选训练/巩固因此可能直接进入生产。
- 修复：`engine.consolidate(..., activate=False)` 生成快照但不切换激活指针；候选路径不会清空学习 WAL，随后重新加载原激活版本放回内存缓存，避免磁盘仍指向旧版本但进程内实际使用候选版本的撕裂状态。
- API 语义：`auto_activate=false` 返回 `activated=false`、批次状态 `candidate`，候选需经过独立验收后再走管理员激活接口；默认值保持原有自动激活行为，避免改变既有调用方。
- 回归：`tests.smoke.test_candidate_isolation` 通过；`compileall` 通过。当前仍需补候选评估结果与批准记录的强制关联，不能把“生成候选”解释为“已验收”。

## 9. 本轮复核收尾与验收边界（2026-10-12）

### 9.1 WAL/缓存一致性补丁

- 复核确认 §8 原候选分支只加载父快照并缓存、保留 WAL，会使同一 current 版本在热缓存与 LRU/重启恢复之间出现行为差异：已消费反馈在前者暂时消失，后者又被 `_replay_journal` 恢复。
- 已在 `AOI_Core/backend/engine/__init__.py` 的 `consolidate(..., activate=False)` 中改为：父快照加载后先重放匹配 base 的 WAL，再发布热缓存。激活指针、WAL 内容和算法门控语义不变；重放异常不发布半恢复缓存。
- `AOI_Core/tests/smoke/test_candidate_isolation.py` 扩为 4 项，覆盖候选父缓存重放、真实 TEMP WAL 按 base 过滤且不追加/清空、重放异常不替换缓存、activate=True 原提交语义。与版本追溯 11 项、反馈微调等价 4 项合计 **19 tests，3.917s，exit 0**。
- WAL 测试中的 `feedback` 是替身，仅验证日志读取/过滤/调用契约，不等同真实训练恢复轨迹复现；未加载真实 DINO，未跑全量数据。

### 9.2 未通过/未完成的生产门禁

- 新硬红线优先于本记录早期的“<1 秒”历史口径：原生 2500×2500 单图端到端 p95≤200ms 尚无合格证据；既有 8 图 p95=251.322ms 且尺寸/样本量不足，不能替代正式验收。
- 独立测试集固定图级误报率下召回尚未完成。20 图工作区与 HEAD 的 fused AUC 均为 0，不能因为相等就称精度达标；不得用该测试集标签反选方向、权重或阈值。
- 持续学习耗时仍无全局上界。25 图重路径第 12 轮约 11.670592s 只是一次观测，未覆盖 disc 接受后二次 LOO、router、搜索、孵育、最大缓存及完整服务锁等待；`timeout_ms` 不构成训练中止保证。
- `consolidate(activate=False)` 的修复仅解决恢复一致性，不代表工作态/候选独立隔离、事件幂等水位或发布事务完成。仍需 SS-A/B/C 后续工作。

### 9.3 仍开放的工程风险

预写 WAL/fsync 与学习事务、重复 `feedback_id` 去重、原图缺失/单条失败的恢复策略、全文件读取和恢复耗时上界、激活与 current/缓存/DB 的发布协调、后台学习持品类锁等，均未在本轮扩大范围修复。sklearn 1.9.1 训练对象在 1.9.0 加载的兼容性警告仍存在；未升级依赖。

本轮未修改生产快照或原始数据，未恢复已删除 TEMP 脚本，未跑全量数据，未提交 git。

## 10. HARD-01：2500×2500 工程尺寸分段初筛（2026-10-12）

### 10.1 输入、隔离与配置

- 已阅读 `backend/engine/__init__.py`、`algo/eval/offline.py`、`backend/pipeline/service.py`、`backend/api/routes_detect.py` 与 `tests/bench/fullchain.py`。现有 fullchain 的 competition 模式要求至少100张，不用于本次≤8张初筛；本轮不启动HTTP服务，不跑全量。
- 对 `d:\CGAIC\data_local\gold_finger` 检查尺寸，**原生2500×2500数量为0**。从 `test/defect` 取1289.jpg至1296.jpg共8张真实图，原图均4096×3000；只在TEMP转RGB、LANCZOS缩放到2500×2500并保存JPEG quality=95。这是工程尺寸初筛，不是原生输入正式验收，也不是按尺寸折算的估算。
- 新建独立脚本 `C:\Users\imia\AppData\Local\Temp\aoi_hard01_probe_20261012.py`，未使用或恢复此前已删除脚本。隔离目录 `C:\Users\imia\AppData\Local\Temp\aoi_hard01_2500_effattl5`；结构化证据为其 `result.json`，含逐图原始SHA256、原尺寸、输入文件名、逐段时间与统计。
- 快照目录复制到TEMP storage，仅将副本current设置为v1。使用可加载的gold_finger/v1，不触碰生产v2/current。入口基础配置为 `configs/engine_fast.yaml`，实际加载快照v1/cfg.yaml：dinov2_vits14、output_layer=9、grid=32；single模式、tile_size=512、stride=448，每图1 tile；sem/disc/shead启用，open配置启用，blob/trad/layout/tpl/inp关闭；preprocess与TTA关闭，hierarchical.work_side=512、mask_thresh=0.5、min_area=16。2500是读取输入尺寸，不意味着主干对2500网格直接前向。
- GPU NVIDIA GeForce GTX 1660 SUPER；Windows 10 build19045、Python3.11.9、Torch2.5.1+cu121、CUDA12.1。模型常驻；模型加载另计2969.625ms；首图一次预热251.569ms另计。每张图及预热前清空 `_item_cache`，不复用特征；每段计时边界CUDA synchronize、perf_counter。操作系统页缓存未清空，首图预热可能使首张文件热读；不能称冷读基准。
- 快照cfg的seed字段为42，但本轮脚本没有显式设置random/NumPy/Torch/CUDA seed，也未记录完整模型/代码/配置hash、CPU线程和存储介质，不能满足正式复现清单；不把配置字段冒充执行种子。

运行命令（cwd=`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`）：

```powershell
& "C:\Program Files\Python311\python.exe" -u "C:\Users\imia\AppData\Local\Temp\aoi_hard01_probe_20261012.py"
```

执行exit 0。终端打印模型加载、首图预热、逐图分段时间与p50/p95汇总。分段是在每图结束时输出，不是每个子阶段起止的独立进度事件。

### 10.2 实测数值与计时口径

**必须纠正结果JSON字段名称的范围：`engine_total_ms` 实际是手工调用Pipeline内部步骤的同步wall time，不是完整DetectionEngine.predict_image_path或服务端到端。** 覆盖load_image读取/解码、_preprocess、compute_tiles/extract_tiles、_prefetch_cpu/_get_item/_prefetch_clf_imgfeat、_record、_finalize与_tta_refine。含Pipeline内部判定/分层定位/类型归因及rec包装，不含引擎 `_to_result` 的DetectionResult包装，也未调用trace_api.record。`item_feature_ms` 已包含DINO主干extract_tiles前向；`forward_scoring_ms` 是_record槽位打分/校准/融合等，不能说全部forward仅约9ms。并行future的等待成本可能在后续消费阶段体现。

单位均ms；preprocess关闭、TTA关闭的极小耗时不代表完整生产预处理/输出成本。

| 原图（TEMP输入2500×2500） | Pipeline总wall | 读取/解码 | item/特征含主干前向 | _record打分/融合 | _finalize |
|---|---:|---:|---:|---:|---:|
| 1289.jpg | 168.313 | 38.578 | 31.490 | 10.199 | 87.912 |
| 1290.jpg | 171.924 | 41.244 | 32.986 | 8.804 | 88.733 |
| 1291.jpg | 187.081 | 44.248 | 35.308 | 10.588 | 96.769 |
| 1292.jpg | 167.656 | 41.993 | 33.098 | 9.536 | 82.840 |
| 1293.jpg | 170.523 | 42.248 | 33.829 | 8.981 | 85.296 |
| 1294.jpg | 174.426 | 45.369 | 34.161 | 8.212 | 86.549 |
| 1295.jpg | 165.581 | 39.622 | 32.784 | 9.924 | 83.113 |
| 1296.jpg | 179.379 | 40.327 | 33.209 | 8.527 | 97.125 |

| 阶段 | p50 | p95 | max | mean |
|---|---:|---:|---:|---:|
| Pipeline总wall（JSON engine_total_ms） | 171.223 | 184.385 | 187.081 | 173.110 |
| 读取/解码 | 41.618 | 44.977 | 45.369 | 41.704 |
| _preprocess（关闭） | 0.014 | 0.015 | 0.015 | 0.013 |
| item/特征，含主干前向 | 33.153 | 34.906 | 35.308 | 33.358 |
| _record打分/校准/融合 | 9.258 | 10.452 | 10.588 | 9.346 |
| _finalize | 87.230 | 97.001 | 97.125 | 88.542 |

- N=8，一次测量；NumPy percentile默认线性插值。没有置信区间或交替重复基线，不以小样本p95作生产保证。逐图tiling/extract为0.019–0.048ms，TTA关闭路径0.009–0.014ms，完整精度的原始值见JSON；各分位不能相加成为总分位。
- 8张均score=2.0、decision=anomaly，仅记录预测；未执行独立精度或不退化验收，不能据目录标签的8张阳性推断整体准确率。

### 10.3 瓶颈、警告与未覆盖范围

- 本次Pipeline路径最大的分段是 `_finalize`（p50=87.230ms，p95=97.001ms），其次读取/解码（p50=41.618ms）和含主干前向的item/特征（p50=33.153ms）。源码显示finalize含缺陷库拦截、判定、_hier定位/类型归因、对位估计与在线缺陷率观察，**不是纯结果包装**；当前没有子阶段证据，不能认定某个操作是可靠可无损删除的瓶颈。
- 只提出后续建议：在同快照、同图、禁缓存下细分finalize的intercept/_hier/归因/align及future等待，再以数值等价验证决定是否可优化；先不修改任何生产代码，不禁用定位/归因以伪造提速。
- 未测DetectionResult包装/序列化、HTTP/multipart/请求排队与网络、品类锁等待、DB查询/落库、persist/告警/通知、热图叠加渲染与文件写入、后台学习竞争、并发和生产冷装载请求。内部mask/boxes生成不能代替服务热图输出测量；这些未测项不估算填数。
- 加载期间有xFormers缺失（SwiGLU/Attention/Block）与sklearn DecisionTree/RandomForest训练1.9.1、运行1.9.0兼容警告；未改依赖。TEMP中的v1加载成功，生产快照和原图均未修改。
- **结论：仅获得2500×2500缩放真实图的Pipeline工程尺寸初筛，p50=171.223ms、p95=184.385ms；不是完整端到端正式验收，不宣布200ms通过。HARD-01继续未实现/未验收，原生数据、完整输出链、精度与并发门禁仍待补。** 本轮仅追加指定两份文档证据，未修改生产代码/快照/原始数据，未提交git。

## 11. UI-P0.2 / UI-P0.3 Gauge 判定契约（2026-10-12）

- 最小前端修复：`ui/widgets/gauge.py` 保留原始 score/threshold，仅绘图圆弧/刻度几何截断到 `[0,1]`；中央数值显示原始 score（三位小数为展示格式，不参与判定）。正常绿色、gray 黄色待复核、异常红色仅由后端 `decision` 决定；缺失/空值/未知 decision 显示“未知 / 未判定”，`is_anomaly` 返回 None。
- 全部调用方：工程内检索只有 `MonitorPage._show_detection` 一处调用，现显式传 decision。历史适配 `_row_to_det` 不再从 `is_anomaly` 猜测 decision；监控徽标及历史条带缺判定不再默认正常。`open_alert` 只控制开放集徽标，不改变 Gauge 或判定徽标，未知模式不等于缺陷。
- 防串记录：每次 set_score 清空缺失 decision/threshold，不能继承上一记录的异常判定或阈值刻度；set_threshold 保留原值且不重判。
- 命令：在 `AOI_Core` 执行 `python -m unittest tests.ui.test_gauge_contract -v`；Python 3.11 / PySide6 6.8.3 / `QT_QPA_PLATFORM=offscreen`，**8 项测试全部通过，0.099s**。覆盖三态与分数矛盾、负分/大于1分及阈值、圆弧截断与原始文字、阈值更新、缺/非法 decision、阈值缺失、唯一调用方、dict/JSON 历史适配及三态/未判定×open 独立。
- 边界：使用真实 QWidget 离屏绘制及无网络监控呈现夹具，不启动服务、不加载模型、不改变算法生产精度或生产快照；不是全量页面/真实产线验收，也不代表 UI-P1.1/GHIJK7 未知人工处置闭环完成。设计发现同步写入现有 `docs/old/重构设计_检测引擎.md` §6.4。

## 13. HARD-01 真实引擎与正式检测服务续测（2026-10-12）

### 13.1 复用验收与隔离边界

- 先实际读取§10的TEMP result.json，确认184.385ms仅手工Pipeline步骤。新增必要的可复用 `AOI_Core/tests/bench/hard01_service.py`，不恢复已删脚本，不改变算法、槽位、精度门控或生产快照/DB。
- 同样1289–1296共8张工程2500输入，复制历史TEMP快照和输入到新TEMP；独立SQLite、配置、engine、heatmaps、archive，v1/current沿用副本。保留默认overlay_max_dim=1280及自动归档/告警，仅禁止外部webhook发送，避免误通知生产；异步webhook网络耗时未覆盖。
- 实际调用 `DetectionEngine.predict_image_path`；服务实际调用 `backend.pipeline.service.DetectionService.detect_image(persist=True, with_heatmap=True)`。必须注明：服务with_heatmap=True源码走predict_ndarray/predict_frame以复用解码，不强行改成路径入口。两个入口各测同8图一次，预热各1次另列，每次清_item_cache，CUDA同步，seed(random/NumPy/Torch/CUDA)=42。
- 主线程真实调用探针记录调用路径、calls、inclusive_ms及exclusive_ms（父段扣除直接子段）；后台图像特征计算不重复计入主线程总wall，其future等待在实际消费点记录。**父子inclusive不可相加，各阶段p95也不可相加**。线程池工作线程保持原调用不计入主线程栈；GPU同步/探针自身有开销，本结果是插桩工程初筛，不是无插桩生产保证。
- 完整服务wall从调用前至JSON dumps后，包含服务解码、模型、_to_result、_finish_detect包装、热图转换/1280叠加/两文件写入、SQLite事务提交、StatsDaily、自动归档和告警。HTTP/FastAPI路由JSON响应、请求排队/上传/网络与后台学习竞争未测；引擎路径JSON仅概要字段，非完整DetectionResult含mask序列化。服务JSON是实际完整返回dict。
- Python3.11.9/Torch2.5.1+cu121/CUDA12.1/GTX1660 SUPER；torch线程8、OpenCV线程16。页缓存不清空；不是冷读。cfg与历史result hash、输入sha256及逐段原始值记录于JSON。xFormers缺失与sklearn1.9.1→1.9.0警告仍在，未改依赖。

命令（cwd=AOI_Core）：

```powershell
& "C:\Program Files\Python311\python.exe" -u -m tests.bench.hard01_service --prior-result "C:\Users\imia\AppData\Local\Temp\aoi_hard01_2500_effattl5\result.json"
```

两次执行均exit=0、分5阶段及逐图start/done进度。第一次报告 `C:\Users\imia\AppData\Local\Temp\aoi_hard01_service_08i843c_\result.json` 的cold_segments引用被后续预热累加（冷载总wall正确，子段不用于验收）；将探针cold_segments改为即时复制后**同样样本全量复测**，最终采用 `C:\Users\imia\AppData\Local\Temp\aoi_hard01_service_r1z6thng\result.json`。这是验收脚本修正，不是算法优化；两次差异不能声称提速。

### 13.2 最终执行结果（ms）

| 入口 | p50 | p95（线性插值） | max | mean |
|---|---:|---:|---:|---:|
| 实际engine.predict_image_path+概要JSON | 197.554 | 247.830 | 262.393 | 203.904 |
| 正式服务热图+persist/archive/alarm+完整返回JSON | 308.947 | 428.110 | 472.379 | 329.080 |

- 模型冷载2063.641ms另列；engine预热294.633ms、服务预热315.097ms另列。无竞争品类锁等待纳入总wall，服务p50=0.0116ms、p95=0.0160ms，不能推断学习并发等待有界。
- 8图逐图engine/service wall分别：198.405/297.304、198.679/300.597、262.393/269.601、196.702/472.379、181.608/304.982、176.433/345.896、220.786/328.966、196.223/312.912。
- 校验9条Detection（8图+1服务预热）、StatsDaily.n_inspected=9，每张服务响应persisted及detection_id有效、heatmap/overlay文件存在。两入口8图均anomaly、score=2.0；阳性8图不是精度验收。
- 第一次参考wall：engine p50/p95=186.174/213.664ms，服务282.608/414.167ms；最终复测有明显主机/线程调度抖动，不隐藏较慢结果。

### 13.3 _finalize 真实子调用与输出瓶颈

下表采用最终服务入口，inclusive含子段、exclusive仅本段；future_wait与classifier_predict同级不重叠。

| 真实调用 | inclusive p50/p95 | exclusive p50 | 说明 |
|---|---:|---:|---|
| _finalize | 113.221 / 126.317 | 0.130 | 不是纯包装，主要成本在子调用 |
| _hier | 90.636 / 98.724 | 0.590 | 包含归因、mask及框 |
| _attribute_types | 73.759 / 83.833 | 0.080 | 包含下方future与classifier |
| _attribute_types/future.result | 45.685 / 57.029 | 45.685 | 图像侧特征异步future消费等待，不是第二次特征提取 |
| classifier.predict | 25.413 / 41.130 | 25.413 | 真实监督分类，不可直接删除 |
| build_heatmap_mask | 11.674 / 15.568 | 10.581 | resize子段已经扣除 |
| boxes_from_mask | 1.628 / 1.899 | 1.628 | 连通域 |
| _align_offset | 15.744 / 37.969 | 4.004 | resize p50=11.741已在其中 |

- 拦截、head boost、在线决策观察均已单独实测且很小；_hier未出现_slot_heatmaps调用，使用_record热图缓存，不能把它认定成重复槽位前向。类型分类器复用_clf_imgfeat future，主线程等待虽重但不是可严格等价删除的冗余计算。因此本轮**未修改生产算法、未禁用覆盖项，也没有算法等价优化/等价测试的适用变更**。
- 服务额外成本：_to_result p50/p95=0.107/0.129ms，热图叠加25.213/32.665ms、两次JPEG写入合计12.648/18.482ms、persist（含auto_archive）10.552/12.923ms，其中归档5.795/8.357ms；JSON实际dict包装0.01–0.25ms级，不是主瓶颈。
- 最大服务样本1292在_record_alarm为201.543ms（总体alarm p50=0.025ms、p95=131.017ms），冷启动连续异常第一次触发旁路日志/通知准备可能是原因，但本探针仅定位到真实函数，尚未再细分，不能把推测写成根因。首次执行相同样本也在alarm约195ms，需后续保留该触发覆盖进一步细分。
- 现有响应字段total_latency_ms在_finish_detect开头计算，latency_e2e_ms在persist/alarm之前计算；均不等于完整返回wall，不能以字段宣称2500端到端通过。新增验收wall弥补测量，不改已有API语义。
- **HARD-01仍未实现/未正式验收：原生2500真实图缺失，实际入口p95已超200ms，本轮非生产通过；精度门禁仍失败（历史20图fused AUC=0），本轮不放宽门禁也不声称精度不退化。** HTTP、独立最终精度集、锁竞争/学习并发、正式硬件及无探针复验仍待补。重要发现同步写入检测引擎设计§8.4，未提交git。

## 12. REPORT-10：m14 初始化缺陷契约修复与验证（2026-10-12）

- 修复文件：`AOI_Core/tests/smoke/m14_template.py`。原夹具把全部缺陷登记为 `split=test`，而 `routes_models.py` 的 prepare 只从 `split=train_anomaly` 取 `init_defect`；因此 `offline.py` 按契约返回 `train_auroc=None`，原 smoke 的非空断言错误。
- 最小修复：缺陷池拆为独立的 1 张 `train_anomaly` 与 3 张 `test` 缺陷，显式断言两组路径不相交；新增先无初始化缺陷 prepare 并断言 `train_auroc is None`，再登记合法 `train_anomaly` 并断言后续 insight 的训练指标有定义。test 样本只用于测试/检测，不参与初始化。
- 长任务 `wait_task` 增加阶段进度输出。未修改生产快照；数据、数据库、引擎存储均为 TEMP。
- 历史命令：`$env:PYTHONDONTWRITEBYTECODE='1'; py -3.11 -u -m tests.smoke.m14_template`（cwd=`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`）。历史结果虽exit 0、18项断言通过，但跳过detect/open_alert，**不能代表检测输出契约通过，也不是完整M14通过**。本轮补验见§14。

## 14. REPORT-10：合成检测非有限 JSON 根因及边界修复（2026-10-12）

- 修复前复现：`py -3.11 -u -c "from tests.smoke import m14_template as m; m._SYNTHETIC=False; print('REPRO: enforce real detection assertions', flush=True); m.main()"`，exit 1，Starlette严格JSON报`Out of range float values are not JSON compliant`。临时只读响应观察进一步确认唯一非有限字段是`candidate_router.train_info.weight_std`及其`n_tiles`副本；观察脚本验证后删除，未清洗响应。
- 根因：M14仅1张train_anomaly，`train_router_on_matrix`对缺陷样本路由权重使用`torch.std(dim=0)`，默认样本标准差correction=1，自由度为0，诊断值NaN。经router_gate.train_info→引擎extra.candidate_router→检测响应/落库n_tiles外泄。不是模板分数、融合分数或传统特征训练退化，也不是合法train_auroc=None。
- 修复：单样本weight_std=None，availability.weight_std包含insufficient、原因与n_samples；多样本保留原样本标准差，不改correction、不置零、不改score精度、融合权重或判定。路由训练拒绝空/非有限输入、非有限loss/参数/最终输出；异常向上抛出，不激活退化候选。检测服务在落库/统计/告警之前使用`json.dumps(out, allow_nan=False)`拒绝非法输出，验证不替换任何字段。
- 恢复M14真实detect、tpl槽位、报告窗口及open_alert落库断言；真实响应与查询记录执行严格JSON序列化，不再按synthetic skip。
- 验证命令（同工程cwd）：`py -3.11 -u -m unittest tests.smoke.test_router_finite tests.smoke.test_contribution -v`，5 tests，1.101s，OK；`py -3.11 -u tests/smoke/m14_template.py`，exit 0，22项断言通过，含真实检测响应及落库窗口。终端保留任务百分比、训练epoch及阶段进度。
- 单测覆盖：1正样本指标不可定义、多正样本与原样本标准差精确相等、NaN/±Inf训练拒绝、非法最终score/嵌套候选诊断在持久化前拒绝、合法score原精度和decision不变；所有合法输出包含allow_nan=False验证。
- 边界：仅合成小样本和定向单测，不代表生产精度/性能验收；未跑全量、未改生产快照、未恢复删除TEMP脚本、未提交git。历史污染快照/数据库不做静默迁移，旧非法结果会被严格检测边界拒绝；历史查询读取及在线重训等路径仍需后续专项验证。sklearn版本、xFormers与贡献相关性常量警告仍存在，但本轮M14不再出现std自由度警告。
- 局限：本机未探测到符合条件的真实数据集，未覆盖真实图片的 detect 响应；xFormers 缺失、sklearn 1.9.1→1.9.0 兼容警告仍存在。

## 15. HARD-01：正式输出链等价小优化与告警尖峰定位（2026-10-12）

- 最小生产改动仅在 `backend/pipeline/service.py`：展示图先按原尺寸/INTER_AREA缩至原overlay_max_dim，再RGB→BGR。通道置换与逐通道面积缩放可交换，减少2500²全图颜色转换及临时数组；原图推理、score、decision、mask、框坐标缩放、归因、热图JPEG、同步落库/归档/告警均不删减。未改变延迟字段口径。
- `tests/smoke/test_display_order_equivalence.py`新增2项：2500²及非方形/整比/非整比缩放数组逐元素精确相等；实际_finish_detect旧/新收尾比较score、decision、所有非时间/自增ID响应字段、热图/overlay JPEG字节、框、类型、对位、Detection.n_tiles及StatsDaily。3种尺寸、6条真实TEMP SQLite记录；2 tests OK（0.369s，后续修正旧实现全局引用后复测也OK）。另router_finite/contribution 5 tests OK（1.082s）。时间值不要求相等。
- 基准增加 `--legacy-display`，只在测试进程重建旧收尾，不回写生产。固定1289–1296八图/同v1副本/seed42、每次清_item_cache、CUDA同步、正式detect_image(persist=True,with_heatmap=True)+JSON、每轮独立TEMP SQLite。执行顺序旧→新→旧→新，后因发现旧函数复制globals会绕过部分探针，修正为共享生产模块globals，再追加旧→新；早期6ba4b5wo分段不可与候选相加比较，最终采用下表。全部运行exit0、逐图进度、每轮9条Detection（含预热）和StatsDaily及热图文件断言通过。

| 最终同探针轮次 | 引擎p50/p95/max ms | 正式服务p50/p95/max ms |
|---|---|---|
| 旧展示顺序 hn2iplb5 | 172.155 / 188.224 / 191.893 | 262.213 / 434.157 / 495.527 |
| 新展示顺序 ktj7gbco | 171.455 / 185.275 / 189.024 | 257.156 / 402.423 / 471.613 |

- 前面实际未修改基线cwzhp7o6服务p50/p95/max=274.346/414.296/483.906；候选uq48chgm=269.872/395.941/454.034；下一候选3mfuxmo_=265.881/403.139/470.650。**不能将跨轮总体差异全部归因于此小优化**：引擎并未修改但也有波动，首次导入和线程调度影响尾部。仅确认输出严格等价和减少大图转换工作，尚非稳定提速验收。
- 真实瓶颈证据（最终旧→新，inclusive p50）：future等待服务36.286→35.455ms，对位14.453→15.930ms，热图渲染23.526→22.925ms，两次imwrite合计12.325→11.892ms，persist8.987→10.320ms（包含archive5.031→5.243ms）。嵌套父子不能相加。JSON约亚毫秒，不是主要瓶颈。
- 告警尖峰根因已细分：cwzhp7o6 `_record_alarm`最大213.349ms，其中`log_action/import_security`211.204ms；最终旧202.646/197.706ms、新219.301/215.469ms。滑窗record本身很小，日志首次读current_actor触发security→FastAPI/Starlette冷导入占尖峰主体。正式HTTP启动通常已导入security，当前直接服务基准没有HTTP，因此不把它伪称稳定生产告警成本，也不预热/异步/跳过告警来隐藏本轮时间。future是归因实际图像特征消费等待，对位含预处理和phaseCorrelate，未发现可删除的重复计算，保持原逻辑。
- 证据文件：`C:\Users\imia\AppData\Local\Temp\aoi_hard01_service_hn2iplb5\result.json`、`C:\Users\imia\AppData\Local\Temp\aoi_hard01_service_ktj7gbco\result.json`，包含输入SHA256、配置/快照哈希、逐图和全部子阶段。所有输入/快照副本/DB/输出在TEMP；未修改生产快照或数据库、不恢复已删probe、不提交git。
- 结论：正式服务p95仍402.423ms>200ms，HARD-01仍未实现。非原生2500、仅8张缺陷图、线性插值p95、含插桩同步开销、OS缓存未清、webhook外网禁用、HTTP/并发未测，精度门禁未复验；不能宣称原生性能或生产精度通过。

## 16. SS-B：反馈请求持久幂等最小交付（2026-10-12）

### 16.1 调用链与修复

- 实际路径：`routes_feedback` 普通及传统CV反馈、`routes_review`复核共同调`self_learning.submit_feedback_record`；该服务传DB `feedback_id`到`DetectionEngine.submit_feedback`。UI普通反馈、PCB来源事件、m01无ID调用保持合法；未恢复archive脚本。
- 原服务按source_event_id先查后插，既无唯一约束也不比较payload；失败记录再次提交直接返回unavailable，不能重试学习。传统CV路由另有绕过服务的早返回，也不比较payload。引擎相同feedback_id每次调用都学习并追加WAL；回写DB失败后重试可能再次学习；重复WAL逐行恢复会再次累加。
- 现在业务DB v7迁移和新模型给source_event_id唯一约束，空字符串按无ID处理；已有重复非空事件迁移失败，不删除或静默合并历史反馈，需人工审计。单进程提交串行，跨进程唯一竞争失败后重新读取原事件。比较detection/type/label/defect/comment/region/operator，同ID冲突或已作废为409；未consumed重用原feedback_id学习重试，不新增StatsDaily。传统路由固定到事件原检测并走相同冲突/重试逻辑。
- 引擎storage新增`feedback_receipts.sqlite3`完成凭据（不是内存set），feedback_id优先、无feedback_id可用event_id；均无ID仍每次学习。SQLite BEGIN IMMEDIATE串行检查/学习/持久receipt：只有pipe.feedback成功且WAL追加flush+fsync成功才写receipt，随后调用on_complete置consumed。重复成功事件跳过学习，但再次尝试DB回写。receipt不随WAL清空；重启恢复仍从匹配base WAL重建工作态，不因receipt存在跳过必要恢复；同一恢复流重复ID只学习一次，冲突ID拒绝，成功恢复补receipt。async响应为pending而非applied。

### 16.2 TEMP小型验证

命令（上述Core cwd；逐项verbose进度）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
py -3.11 -u -m unittest tests.smoke.test_feedback_idempotency tests.smoke.test_candidate_isolation tests.smoke.test_detection_version tests.smoke.test_feedback_ft_equivalence -v
```

结果：exit0，29 tests，3.952s；SS-B新增10项＋candidate4＋version11＋feedback数学等价4全部通过。覆盖重复/不同payload、失败同ID重试、无ID旧行为、12并发同event、async完成后重复、DB回写失败重试不再学、TEMP真实WAL重放及重复行、receipt跨新引擎实例及WAL清空保留、WAL失败不置consumed、业务TEMP SQLite仅一Feedback/Stats计数、历史重复迁移fail closed不删除。学习pipe为替身，未跑真实训练/断电/HTTP多进程，不把替身恢复当完整轨迹恢复证明。sklearn1.9.1→1.9.0已有兼容警告仍在；git diff --check受仓库CRLF/既有及本轮行尾提示影响非零，未因此改其他任务文件。

### 16.3 真实边界（SS-B仍部分实现）

1. 这是完成凭据去重，不是exactly-once事务。pipe.feedback可能部分修改后抛错；WAL或receipt提交失败时内存可能已学习但无凭据，同进程重试可能再次学习。学习完成→WAL落盘的崩溃窗口仍在；业务DB不先consumed，事件仍保留可重试，不声称候选原子发布。
2. WAL已落盘→receipt提交前崩溃，新进程成功重放可补receipt并防后续重复；存活进程receipt失败后重试仍存在重复窗口。WAL缺图/重放失败仍按旧机制跳过，而已存在receipt/consumed不会自动撤销；恢复状态需人工核查，不能保证模型效果已恢复。
3. receipt是storage范围的终身反馈身份，不是版本线消费水位；显式回滚不自动重新学旧ID。手工删除receipt/复用业务DB ID或独立多进程pipe可能破坏语义；跨进程DB锁只保证请求身份串行，不解决各进程内存同步、activate与consolidate的发布竞态。
4. receipt数据库串行持锁覆盖训练，可能阻塞跨品类反馈；记录无压缩/清理及身份epoch。历史无ID事件无法回溯去重；WAL清空前历史已巩固事件不能从日志补receipt。完整解法仍依赖SS-A独立候选、持久事件/候选消费水位、跨快照/DB可恢复提交与完整学习态恢复，后续不得用先consumed或删除WAL掩盖问题。

未修改`backend/pipeline/service.py`；测试全用TEMP，未改生产DB/storage/快照、未恢复已删脚本、未提交git。

## 17. SS-A/B：显式候选反馈隔离最小纵向切片（2026-10-12）

### 17.1 调用链、兼容语义与实现

- 新增实际路由`POST /feedback/candidate`，仍受工程师角色依赖保护；请求必须提供非空`source_event_id`和`parent_version`。路由调用`submit_feedback_record(..., candidate_only=True)`落库，再由后台任务调用`build_feedback_candidate`，最终进入`DetectionEngine.learn_candidate`。任务进度为独立候选装载/学习、候选完成（禁止激活）、登记快照三个阶段。
- 旧`/feedback`、传统反馈及旧更新入口未改为候选模式。`auto_activate`原语义仍只控制巩固后的生产激活，不等于反馈学习隔离；旧在线反馈仍可能直接修改生产pipe。候选通过显式参数和`candidate:`事件命名空间避免误走旧生产学习路径，并保持候选Feedback不置生产`consumed`。
- `learn_candidate`先在storage SQLite写入持久training意图，短暂获取品类/写协调锁捕获父版本与父WAL边界，然后锁外用`algo.persist.load_snapshot`完整装载父Pipeline、严格重放冻结父WAL并执行候选反馈；不使用零散deepcopy，也不跳过父事件。候选保存到TEMP staging，真实重新load验证后以新版本目录发布，元数据记录`parent_version`、`candidate_state=completed`、`acceptance=unavailable`；不切current、不写生产WAL、不进入生产缓存。
- 发布前复查current和WAL字节未变化；父版本变化、WAL变化、缺图、重复事件payload冲突、保存失败或学习失败均fail closed。重复已完成事件返回原结果并不新增版本；同事件失败/未完成不会被静默重试。
- 激活入口对候选先检查父版本和完成态，再因独立验收/批准尚未实现而明确拒绝。因此本切片是可保存、可验证的隔离底座，不是已批准的生产发布闭环。

### 17.2 TEMP验证与边界

- 命令（Core cwd，逐项verbose输出）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
py -3.11 -u -m unittest tests.smoke.test_isolated_candidate_path tests.smoke.test_feedback_idempotency tests.smoke.test_candidate_isolation tests.smoke.test_detection_version tests.smoke.test_feedback_ft_equivalence -v
```

- 结果：exit0，37 tests，4.429s，OK；新增候选8项，既有SS-B幂等10项、候选巩固4项、版本11项、反馈数学等价4项通过。覆盖参数/库/阈值/current/WAL不污染、并发预测不等待候选训练、重复事件、payload冲突、学习/save失败、父版本/WAL变化、父WAL完整重放、缺失资产、真实save/reload、服务和路由调用链。所有工件位于TEMP，未改生产快照或数据库。
- 测试边界：Pipeline反馈动作、factory构建、后台task提交及模型登记使用替身；WAL/receipt/业务SQLite/快照保存与重载为真实路径。因此不能将37项通过解释为真实算法训练、DINO/完整学习态/RNG恢复、HTTP鉴权/UI、2500端到端性能、崩溃恢复或多进程发布验收。
- 资源与一致性边界：候选训练不持有品类推理锁，但捕获WAL/SQLite意图仍在短协调区间；16条父WAL与64KiB读取预算是初步保护，不是完整墙钟、GPU、训练步数或内存上界。候选Pipeline生命周期、rename与SQLite完成记账之间的原子恢复、进程退出遗留training、反馈作废/修订并发及模型登记失败后的清理尚未形成恢复协议；sklearn快照1.9.1与环境1.9.0兼容警告仍存在。

未修改`backend/pipeline/service.py`及security；未恢复用户删除的TEMP脚本；未提交git。

## 18. HARD-01：轻量审计依赖分离与同步输出链续测（2026-10-12）

### 18.1 最小改动与等价边界

- 原链为`DetectionService._record_alarm → db.database.log_action(user=None) → security.current_actor → FastAPI/Starlette`。告警需要操作者上下文与同步OperationLog事务，不需要认证中间件。
- 新增`backend/core/audit.py`，仅持有标准库ContextVar及轻量log_action适配器。security改为导入并兼容导出同一current_actor对象；中间件鉴权、角色排名、set/reset及路径权限逻辑均未改变。pipeline仅把log_action导入改到audit；适配器在同一位置读取current_actor.get()，显式传给**原数据库log_action**。数据库文件、事务、模型、检测/学习引擎、routes未由本任务修改。
- 等价理由：原user=None分支只导入ContextVar并get，无其他认证副作用；显式传入读取值使原函数走同一个OperationLog(user/action/detail/extra)和session_scope提交/失败回滚。显式空字符串用户保留，不使用truthy兜底。原security.current_actor仍为同一对象，因此旧DB调用点与新告警链都读取同一请求身份；普通threading线程仍默认operator。不是预热、绕过鉴权、跳过告警或异步落库；其他旧DB审计入口仍保持原导入方式，未全局改造。

### 18.2 契约测试

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
py -3.11 -u -m unittest tests.smoke.test_audit_contract tests.smoke.test_display_order_equivalence -v
```

- exit0，7 tests，1.603s。新增5项使用TEMP真实SQLite或冷子进程：新旧3角色及显式空user/extra审计字段完全相等；alarm通知回调内独立会话已经能读到提交记录；滑窗命中/冷却/响应相等；commit失败真实回滚且不通知；中间件operator/engineer/admin允许与403、错误key401、请求上下文reset、异步转同步线程角色传播及直接spawn线程默认operator；全新进程调用轻量审计不导入security/FastAPI/Starlette。另展示2项数组/JPEG/完整响应与Detection/Stats契约继续通过。
- 时间戳与自增ID是每次写入新生成值，不要求不同事务间精确相等；生成方式仍由未修改的OperationLog模型负责。

### 18.3 实际<=8工程图计时（不是正式验收）

- 基准`tests/bench/hard01_service.py`新增`--legacy-audit`，只在测试进程将服务log_action绑定回原database.log_action，不改生产代码；记录audit_rows及security_imported_at_end，断言真实告警记录存在。固定TEMP源`aoi_hard01_2500_effattl5/result.json`的1289–1296八图、同v1快照/seed42，每调用清_item_cache、GPU同步、包含实际引擎/服务热图JPEG/同步Detection与Stats/归档/同步告警及返回JSON。仍按原基准分别记录引擎与服务预热；**没有增加审计或security预热**，首次触发告警在计时第4张服务图内。
- 每轮独立TEMP config/SQLite/engine/输入/输出；webhook外网禁用，不发送生产通知。逐阶段及逐图start/done进度，两次旧→新四轮全部exit0，每轮9条Detection（含既有服务预热）、Stats计数及热图文件通过，真实审计恰一条：user=operator/action=alarm/detail=`category=gold_finger window_hits=5 window=20 k=5`/extra=null，新旧一致。

| 轮次/审计入口（TEMP后缀） | 引擎p50/p95/max ms | 完整服务+JSON p50/p95/max ms |
|---|---|---|
| 1旧 45gys2kq | 185.707 / 207.124 / 209.353 | 280.089 / 437.831 / 503.621 |
| 1新 qq5c71fc | 186.056 / 212.163 / 224.815 | 291.018 / 327.246 / 335.167 |
| 2旧 n0jep1f9 | 184.229 / 220.956 / 231.251 | 293.184 / 452.654 / 523.607 |
| 2新 t7t0u_7p | 183.893 / 197.286 / 197.816 | 294.290 / 316.123 / 322.349 |

- 第一次实际告警inclusive：旧_record_alarm=229.940ms/log_action=229.854ms/import_security=224.070ms；新=1.934/1.821ms。第二次旧=219.878/219.791/import_security=215.861ms；新=1.928/1.844ms。新链仍包含真实session_commit分别1.616/1.687ms，未删除或移到计时外。旧进程末security_imported=true、新=false；这是去掉不必要认证依赖，而非把首次导入挪到启动。
- 证据：`C:\Users\imia\AppData\Local\Temp\aoi_hard01_service_45gys2kq\result.json`、`aoi_hard01_service_qq5c71fc\result.json`、`aoi_hard01_service_n0jep1f9\result.json`、`aoi_hard01_service_t7t0u_7p\result.json`（后三者同TEMP父目录）。含输入SHA256、快照配置哈希、逐图嵌套分段、同步审计字段及全部计时；父子inclusive不可相加。
- **HARD-01仍未实现：新完整服务p95=327.246/316.123ms>200ms，p50也>200ms。**仅8张缩放真实缺陷图而非原生2500；p95为线性插值、OS页缓存未清、含探针CUDA同步开销，HTTP/并发及正式硬件验收缺失。引擎p95偶尔<200不能代表完整输出链达标；总体p50并未稳定下降，跨轮尾部差异不能全算作改动收益。历史精度门禁失败（20图fused AUC=0）未在本任务复验/放宽；sklearn 1.9.1→1.9.0快照兼容警告仍在。
- 设计发现同步追加到检测引擎设计独立§18补充；未覆盖其他agent章节，未改生产资产、不恢复任何已删除TEMP脚本、不提交git。

### 18.1 主线合并回归

- 在 AOI_Core 目录以 Python 3.11、QT_QPA_PLATFORM=offscreen 运行候选独立路径、反馈幂等、候选巩固、版本追溯、微调等价、审计、展示等价、router有限值、贡献及Gauge契约的10个unittest模块。
- 结果：57 tests，5.413秒，exit 0，全部通过。此为小型契约/恢复/数学等价回归，不替代真实候选训练、HTTP鉴权、并发GPU竞争、端到端200ms或独立精度验收。
- 仍有sklearn 1.9.1对象在1.9.0加载的兼容性警告；常量输入相关性测试产生预期警告。

## 19. 候选training / 孤儿快照fail-closed恢复收尾（2026-10-12）

实际读取engine后确认上一轮已部分加入OS锁、SHA256、prepared意图及故障接缝；本轮没有覆盖这些变化。§19原不存在，因此新增本节，不替换§17/18历史证据。

- 修复prepared后finally仍无条件删除stage的问题：prepared意图已提交后保留工件；候选pipeline执行器在finally关闭。Windows对只读描述符fsync实际报EBADF，同步文件改用r+b（不改字节）；拒绝目录/文件符号链接。
- `learn_candidate(..., recover=True)`为显式恢复入口（仅engine，尚无专用恢复HTTP/UI）。普通重试不将training猜测为failed、不补prepared；持有同品类OS锁后，显式recover将不完整training拒绝为rejected，保留意图，不恢复训练、不自动删除未知资产。新事件在training/prepared未处置时被拒绝。
- prepared恢复或completed幂等重试必须核对父current、父快照文件SHA256、冻结WAL SHA256、候选完整文件清单，并真实load_snapshot；失败明确拒绝。通过只rename工件/补记completed，不切current、不替换生产缓存、不生产consume；acceptance=unavailable仍禁止激活。没有持久意图或完整hash的孤儿不自动认领。
- TEMP新增4项候选测试（含8处checkpoint和4种完整性变更subTest）：保存前后、prepared前后、rename前后及completed前后异常，新引擎恢复、重复ID、普通重试不猜training、原current/WAL字节不变、候选/父hash/WAL/current变化拒绝。第二engine及真实子进程在活训练持OS锁时recover失败，持久training不变。候选训练动作/factory仍替身，快照/SQLite/WAL/load真实。
- 首轮：17 tests / 3.172s / exit1，failures=2、errors=13，主因为Windows只读fsync EBADF；修复后执行下列全组合，不隐藏失败记录。

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:QT_QPA_PLATFORM='offscreen'
py -3.11 -u -m unittest tests.smoke.test_isolated_candidate_path tests.smoke.test_feedback_evidence tests.smoke.test_feedback_idempotency tests.smoke.test_candidate_isolation tests.smoke.test_detection_version tests.smoke.test_feedback_ft_equivalence tests.smoke.test_audit_contract tests.smoke.test_display_order_equivalence tests.smoke.test_router_finite tests.smoke.test_contribution tests.ui.test_gauge_contract -v
```

工作目录`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`；11模块 **66 tests / 10.761s / exit0 / OK**。逐测试及TEMP阶段有进度。sklearn 1.9.1→1.9.0与常量相关性警告仍存在。

边界：故障注入是Python异常/BaseException＋新engine，不是杀进程/断电；子进程仅证明活候选互斥，不能证明所有生产写入口跨进程一致。目录rename和SQLite无跨文件原子事务/目录fsync保证；其他prepare/consolidate/feedback不持候选OS锁，外部写者检查后变化仍有竞态。没有完整学习态/RNG继续流等价、GPU真实训练/预算、独立批准发布、取消/资产清理或业务DB与快照原子恢复证据。全部AC-038～041/054生产框仍不勾。未改生产DB/快照、未恢复已删脚本、未提交git。

## 20. AC-054 / AC-014：反馈归因持久证据契约（2026-10-12）

### 20.1 实际修改与字段事实

- 修改Core的`backend/self_learning/service.py`、`backend/api/routes_feedback.py`、`ui/api_client.py`、`ui/widgets/feedback_dialog.py`、`ui/pages/feedback_page.py`；新增`tests/smoke/test_feedback_evidence.py`。复用既有learning_trace，新增反馈details只读API与实时/历史可见证据弹窗，来源事件幂等保留原证据，非有限JSON置null，失败/异步/未知不冒称学习完成或模型激活。
- 真实算法归因仅`mode / series / verdict / next`；独立evidence、confidence及区域source_slot不存在时显示未知，series的top_slot/max_slot不造逐框来源。不混actual weights/router/gate，不改判定和训练。
- 未编辑persist、db、算法或engine，不修改生产DB/快照，不恢复已删TEMP脚本，不提交git。设计发现及主线engine一行白名单diff见既有映射设计文档独立§20。

### 20.2 已执行小型验证

工作目录：`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`；Python为`C:\Program Files\Python311\python.exe`，offscreen由测试设置。

```powershell
python -u -m tests.smoke.test_feedback_evidence
python -u -m unittest tests.smoke.test_feedback_idempotency tests.smoke.test_feedback_evidence -v
```

- 第一条：4 tests，0.735s，OK，exit0；第二条：14 tests，1.066s，OK，exit0。逐测试进度输出，均为短测试。
- 新测试4项：实时/持久历史/列表一致及来源重复不再提交engine；实际offscreen证据QDialog含series/next和未知项；旧consumed无trace记录不调用模型推理；嵌套NaN/Inf转null并通过`json.dumps(..., allow_nan=False)`，四个既有字段保留；engine异常返回unavailable且提示无“已学习”，失败后async重试为pending且不冒称完成。
- TEMP真实SQLite仅自建`evidence.sqlite`；engine及训练、通知和审计为替身。路由函数直接调用，没有HTTP鉴权/真实网络端到端测试；离屏验证构造实际详情控件，但不等同人工桌面交互验收。未运行真实GPU学习或证明真实算法归因质量。

### 20.3 剩余依赖与验收状态

历史本节初测时engine同步白名单尚未合入；现已完成，见§20.4。后台async完成callback没有最终entry，不能安全持久最终证据，本轮显示pending/未知，不读共享日志猜测、不用当前模型补旧结果。既有错检追溯重判功能未改，新反馈历史详情不走它。AC-014逐框/Trad参考证据仍缺，AC-054/014保持部分实现，生产`[ ]`均未勾；本记录仅证明最小服务/API/UI契约。

### 20.4 主线engine透传收尾（2026-10-12）

同步entry白名单已增加attribution/evidence，只在真实字段存在时透传。已读handler wrong FP/FN及attribute_failure，真实返回mode/series/verdict/next，没有独立evidence/confidence/source_slot；不生成这些字段、不改训练。新增1项通过真实FeedbackHandler.feedback→真实attribute_failure→新engine.submit_feedback，证明series中关键fused=.2、top_cal=.6保留和缺字段不造值；图像/打分fixture，入库与router训练替身。独立evidence用注入仅验证透传存在条件和非有限→null、有限valid_score=.73不丢失。服务feedback_json限于证据包装，没有修改算法score或生产检测nonfinite拒绝路径，合并router有限值测试继续通过。合并结果66 tests/10.761s/exit0见§19。该证据不是完整真实模型训练归因质量验收；当时async最终entry回写仍缺，续修见§21。

## 22. AC-068～070：候选全轮预算与合作式取消（2026-10-12）

- 新增algo/ssocl/candidate_budget.py，入口唯一预算经ContextVar贯穿父WAL、新事件和所有同步重支路。累计events、LOO（含二次）、search叶候选、shead/disc/router优化步、孵育max_iter保守预留、snapshot重载次数/文件bytes及输入资产bytes/像素。不会裁剪LOO或搜索以降低门控标准；生产上下文无预算时no-op。
- engine.ssocl_cfg.candidate_budget实际生效limits/hash写training意图、成功meta/result与失败证据。预算异常专门穿透AHL宽泛except。超限deferred、合作取消cancelled；失败原ID不静默重训，用新事件ID。prepared前最后检查，提交后完成rename/记账；恢复预算拒绝保留prepared工件。current/生产pipe/WAL/consumed均不由候选改变。
- 新增4项TEMP测试（大量subTest）：共享父+新事件/优化步总量、7类零额度拒绝、取消初始/重载后/提交前及提交后的边界、真实router步与LOO前置拒绝、AHL异常穿透、像素拒绝、业务SQLite超限后consumed=false。快照/WAL/SQLite真实，feedback/factory及任务调度仍替身；不能证明真实训练质量/GPU竞争。
- 命令（Core cwd）：`$env:PYTHONDONTWRITEBYTECODE='1'; py -3.11 -u -m unittest tests.smoke.test_isolated_candidate_path tests.smoke.test_feedback_ft_equivalence tests.smoke.test_router_finite tests.smoke.test_feedback_idempotency tests.smoke.test_candidate_isolation -v`。最终结果：38 tests / 9.091s / exit0 / OK；逐测试与TEMP阶段进度可见。
- 失败过程保留：首次24 tests errors13是源码抽取等价夹具无__name__，相对导入改绝对导入后38项通过；随后扩展服务测试暴露ssocl_cfg共享配置对象被夹具污染（38项 failures3/errors3），夹具深拷贝隔离。下一轮发现现有实际配置max_input_pixels兼容项（failures5/errors18），补真实handler像素预检后38 tests/9.043s/exit0。WindowsApps裸python exit9009和失效外部venv exit103改用py -3.11，不改环境。sklearn1.9.1→1.9.0警告仍在。
- 补充格式检查：包含engine的限定文件`git diff --check`返回exit2，报告engine非候选区域多处trailing whitespace；工作区存在并行修改，未全文件格式化或覆盖无关区域。排除engine后上述feedback/incubate/router/测试及两份实施文档的检查exit0（Git仍提示LF→CRLF）。该格式检查失败不冒称全量通过。
- 重要设计边界：snapshot_bytes只是重载文件工作量、patch_bytes是资产文件量，非实际驻留内存；steps含sklearn500步保守预留，不是精确执行计数。ContextVar不自动传播executor线程；future异步重训练必须显式传播。墙钟/显存硬上界仍false，单次fit/核/加载内部不可抢占，未强杀线程。AC-068仅部分实现，AC-069/070未实现，全部生产框未勾选。
- 不修改异步反馈范围：build_feedback_candidate宽泛catch仍将服务trace记candidate_failed并重新抛出专用异常；engine保留deferred/cancelled，真实TEMP验证consumed=false。统一服务状态与原反馈重试入口列为后续冲突，不越界改service/routes。未改生产DB/快照、未恢复删除脚本、未提交git。

## 21. AC-054：异步反馈最终状态与证据回写（2026-10-12）

- 先重新检查工作区和已有test_feedback_evidence，确认原callback只置consumed，timeout后的pending trace会永久缺失结果。TaskManager为候选/巩固任务管理器，普通反馈使用engine自身线程，未将同一学习再投递TaskManager。
- engine新增可选`on_result(result, error)`，在品类锁释放后回传本次真实entry的白名单结果或错误；原无参成功on_complete保持兼容。同步返回及回调使用同一包装，attribution/evidence只在存在时透传，不读共享feedback_log、不新增归因字段、不改变训练或生产锁。
- service先持久pending，再由每次提交的结果锁保护终态回写；callback在返回前、重复callback、timeout pending返回不得覆盖已持久applied/failed。成功才consumed，失败trace含error且不置完成；后台线程新建session，不捕获请求session。成功post在学习完成后单独捕获，历史详情仍只读；不声称该post是锁内版本绑定的生产四时点完整证据。
- 相同来源事件pending直接返回原记录，不再次提交学习/计数；已完成继续返回原证据。failed保持既有显式重试语义；WAL/receipt幂等逻辑不变。
- 命令（Core cwd）：`& "C:\Program Files\Python311\python.exe" -u -m unittest tests.smoke.test_feedback_evidence tests.smoke.test_feedback_idempotency tests.smoke.test_feedback_ft_equivalence -v`。
- 结果：**22 tests / 3.642s / exit0 / OK**（evidence8、幂等10、微调数学等价4）。新增确定性Event阻塞/释放真实engine线程与TEMP WAL/receipt/SQLite，分别验证timeout后成功及失败、pending重复不再学、真实结果/嵌套非有限evidence包装、callback先于async返回、重复callback终态不翻转；旧实时/历史/offscreen/未知测试继续通过。首次裸python命令被WindowsApps别名拦截exit9009，改用上述已安装解释器，未更改环境；Qt字体目录警告不影响测试。
- 边界：反馈训练pipe、图像推理、通知/审计仍为替身；真实handler归因fixture已有测试但非真实GPU质量验证。没有生产DB/快照变更、删脚本恢复或git提交。进程退出仍可能遗留pending，回写DB失败没有持久outbox自动补偿；跨进程业务状态与引擎receipt的原子恢复、后推版本竞态及HTTP/真实桌面仍开放。AC-054保持部分实现，生产框不勾。

## 23. HARD-01 / G1：完整入口工程测量补充（2026-10-12）

- 使用 `AOI_Core/tests/bench/native_fullchain.py` 的TEMP隔离入口，范围从路径读盘/解码开始，覆盖预处理、推理、融合、决策、严格JSON、热图写盘、同步Detection/Stats提交、归档、告警与完整返回；直接模式不含HTTP传输，HTTP模式为实际`create_app()`+uvicorn回环，墙钟含客户端传输/排队/JSON解析。两种模式均禁用品类特征缓存；CUDA仅在外层请求边界同步，OS页缓存未清。
- 本轮只有8张既有缩放工程图（每张实际2500×2500），不具备`native=true`原生血缘，不能替代原生2500正式验收；每模式1冷请求、5预热、24热请求，分位数为NumPy线性插值，小样本仅工程定位。
- 直接空闲：热P50/P95/P99/max=`290.615/341.356/360.403/365.675ms`；HTTP空闲=`280.749/319.068/323.102/323.895ms`；冷首请求分别`2658.912/2201.558ms`。两者均超过200ms，G1不通过。
- 直接学习竞争：热24次P95/P99/max=`2203.575/3568.518/3972.774ms`；重叠8次=`3357.602/3849.740/3972.774ms`，非重叠16次P95/P99/max=`362.633/375.173/378.308ms`。HTTP学习竞争最新隔离结果：热P95/P99/max=`2202.696/3760.200/4221.204ms`；重叠8次=`3519.675/4080.898/4221.204ms`，非重叠16次=`370.089/370.494/370.596ms`。两次学习均8事件重叠、`learning_errors=[]`、持久化30条、告警审计1条。
- 竞争请求的品类锁等待与学习事件耗时均为秒级（学习事件约1.3～2.0s；锁等待最高约3.5～3.7s），证明旧`submit_feedback`与前台同品类推理串行；不能把3秒join等待解释为学习总耗时上界。事件清单使用工程图并统一标记`wrong/label=1`，不是业务错检或正式G2学习协议。
- HTTP隔离启动首次因空DB快照孤儿治理返回400；按正式快照导入语义登记隔离`Model(origin="uploaded")`后成功，未绕过lifespan，未改生产DB/快照。HTTP传输/队列尚未拆分，报告的`http_transport_ms`保持空值不伪造。
- 本轮没有新的算法或生产优化；既有审计/展示等价回归7 tests/1.426s/exit0。原生资产、G2独立精度、正式样本/重复/置信协议、真实GPU并发门禁仍缺，因此G1、AC-066、AC-069、AC-071均保持未通过/未验收。

## 24. AC-061 / AC-015 / AC-062：FeatureSetRevision 最小不可变底座（2026-10-12）

### 24.1 实际交付

- Core `algo/feature_revision.py`使用标准库规范JSON SHA256作为不可变manifest ID，绑定有序组/维度/连续slice/参数、预处理、实现hash、schema、seed、数据split及模型描述。登记幂等、返回深拷贝；配置变化产生新ID。五态draft→evaluated→approved→active→retired仅允许逐态推进，history含actor/UTC时间并与当前状态一致；注册表原子替换及排他锁保护写入。
- 原评价与批准时另交的独立评价分别冻结绝对路径/hash，血缘须与revision一致。独立评价必须completed、independent=true、acceptance=passed且AUROC有限合法；数值等价证据须八项检查通过、有样本及有限误差/容差，baseline hash匹配当前生产trad两文件。拒绝bool冒充数值、NaN/Inf、缺证据及证据变化。
- AOI_feature manifest探针保存/恢复NumPy RNG；run_eval绑定split输入hash与n_perm/seed协议，完成后复核配置/实现/数据，评价工件禁止覆盖。贡献评价明确independent=false/acceptance=unavailable，只到evaluated，不自动批准。实验fitted工件与生产快照仍unknown。
- 模型API提供注册/查询/迁移；显式revision绑定模型的激活在原gate前重新检查active、模型资产快照、独立评价、等价证据及当前生产实现，force不可绕过。active仅门禁状态，不切current、不自动上线。旧生产模型保留unknown绑定与原激活语义，模型页显示未知而不推断。

### 24.2 已执行的轻量验证

Core目录下执行：

```powershell
py -3.11 -u -m tests.smoke.test_feature_revision
py -3.11 -u -m unittest tests.smoke.test_feature_revision tests.smoke.test_detection_version -v
```

- 第一条8 tests / 0.483s / exit0 / OK；第二条19 tests / 1.465s / exit0 / OK；收尾重跑第二条（PYTHONDONTWRITEBYTECODE=1）19 tests / 1.534s / exit0 / OK。终端有逐测试进度，全部自建TEMP工件，无生产注册表/DB/快照修改。
- 覆盖ID/深拷贝/manifest篡改、逐态迁移及冻结、血缘错误、独立评价另交、原工件修改、数值非法/缺项/容差、active复核、遗留锁、无current副作用；API前门禁与UI force拒绝仅静态AST/源码契约验证，不是HTTP权限或桌面联调。
- 实际FeatureSet.load→revision_manifest→TEMP register成功：15组、200维、draft；不是完整run_eval或生产提取数值等价测试。生产两文件在正确AOI_SSOCL子仓库的`git status --short -- algo/slots/trad.py algo/vendor/traditional.py`为空，`git diff --exit-code -- algo/slots/trad.py algo/vendor/traditional.py`exit0，未改变生产trad源码。
- 裸python为WindowsApps无效别名，初次exit9009，改用现有py -3.11；未安装依赖。sklearn资产1.9.1/环境1.9.0及skimage min_size弃用警告保留，未改算法/依赖。
- 收尾编辑器diagnostics返回空列表。限定代码`git diff --check`exit2，persist/routes_models/model_page的工作区增量报告trailing whitespace（涉及此前改动和本轮行）；未全文件格式化覆盖并行工作，不冒称格式检查通过。此前单独生产两文件diff exit0。

### 24.3 重要发现与未验收范围

实验GroupKNN独立测试默认不跳最近邻，生产TradSlot.score_image固定跳首邻；向量探针一致不等于kNN/报告/库兼容全协议等价。独立评价另交解决了冻结非独立贡献评价后无法批准的问题，但TEMP自建passed证据只证明拒绝/状态契约，不代表真实独立生产评估通过。

注册表为受信本地存储，不是签名防恶意篡改系统；actor来自请求、protocol_hash只核格式，人工checks/误差声明由门禁消费而非重新执行计算。遗留锁需核查；评价工件独占创建并非原子写，写失败可能遗留冲突文件。完整run_eval集成、真实生产协议数值等价、独立G1/G2、HTTP鉴权、UI工作区/Detection关联及可恢复发布仍缺。AC-061仅部分实现，AC-010/062完整要求仍未实现；所有生产验收框保持未勾。未改E_open/候选发布核心，未提交git。

## 25. A 块（AC-001~009）局部分支与三方案对照（2026-10-12）

### 25.1 功能交付（AC-001~008）

- 整图专家（Decider）始终保留；局部分支在 `offline.py::_finalize` 中作为叠加层接入。trace 写入 `branch`（selected/local_score/side_effect/rescored_full）。
- `view_signature`（tile/stride/roi/backbone/grid/preprocess）随快照落盘。推理时签名不一致，局部分支就停用并给出原因。
- 局部块标签按面积划分：≥0.25 为正、≤0.05 为负，中间段丢弃。阈值标注为 `candidate_unfrozen`。
- 图级 α 在整图和局部之间均分，块级阈值用 Šidák 校正，取 `max(sidak, empirical_max_block)`。fpr_global/local/union 在同一批 holdout 上统计；holdout 张数少于 1/α_l 时标 `small_sample`。
- budget 模式按热度优先选块并轮换；遇到灰区或冲突时全量重算。side_effect 分四类：added_by_local / confirmed / global_only / vetoed。否决默认关闭，打开后也最多降到 gray。
- 新增 `branch_budget.branch_ratio(s,tau)`：tau≤0 时改用 `1+(s-tau)/|tau|`，非有限值返回 0。修复的问题：原始分为负时 s/tau 符号翻转，导致并集判定反向。
- smoke `tests/smoke/test_local_branch.py` 21 项全部通过，总计 168 OK。

### 25.2 AC-009 三方案对照（小样本）

脚本：`d:\CGAIC\commit\repo\revise\ac009_three_way.py`。只 fit 一次，推理时切换方案，每个方案开始前清空特征缓存并重置显存峰值统计。结果在 `ac009_gold_finger.json`。
数据：datalocal/gold_finger，fast 档，seed 42，test 20 张（17 缺陷、3 良品），图像 3000×4096，每图 63 块。fit 114.7s，其中局部建库 90.4s。

| 方案 | 块/图 | 显存MiB | recall_alarm | fpr_alarm | p50 ms | p95 ms | 2500²折算p95 | G1 |
|---|---|---|---|---|---|---|---|---|
| global | 0 | 130 | 0.29 | 1.0 | 164 | 475 | 241.8 | 否 |
| full | 80.4 | 906 | 0.41 | 1.0 | 1573 | 3532 | 1796 | 否 |
| budget | 52.8 | 906 | 0.35 | 1.0 | 896 | 4327 | 2201 | 否 |

- 三个方案的 auroc_union_ratio 都是 0.0，fpr 都是 1.0。已核对标签口径正确（`datalocal.py` 中 test/good 为 0）。原因是 gold_finger 的 test/good 与 train/good 存在域差：view_mode 判为 `domain_first`，3 张良品分数都高于全部缺陷。这组数据只能用来对比功能和耗时，不能当作精度结论。
- 救回漏检、新增误报、否决新增漏检都是 0。局部分支只把 recall_alarm 提高了 0.06~0.12（主要是把 normal 拉到 gray），没有改变 anomaly 判定。
- 耗时：局部逐块打分约 25ms/块。budget 的 p95 主要来自 6 次灰区/冲突全量重算，反而比 full 更差。显存从 130 升到 906MiB。

### 25.3 结论与遗留

- 结论：局部分支保持 `auto`/默认关闭，不默认开启。只有当档案判定存在小缺陷、且没有域差时才考虑开启。
- 未查明：full 方案的平均块数（80.4）超过 fit 时的 63。推测是部分 test 图尺寸不同、导致网格更大（个别图达到 144 块），尚未确认，放到 OPT 阶段核查。
- 放到 OPT 阶段：块 batch 化前向、限制重算预算（重算次数上限或只重算热区）、缩小 max_tiles。global 方案本身折算后也有 241.8ms 超过 G1，与 HARD-01 同口径，统一放到最终性能优化处理。
- 本次运行只用了 TEMP 存储，没有改动生产 DB 或快照。

## 26. B8 / AC-017：难例候选特征准入（2026-10-12，生产未验收）

### 26.1 实现

- `backend/evaluation/hard_admission.py`：
  - `build_hard_pool`：从未作废反馈中收集样本，来源分为 miss / fp / open_defect / open_normal / confirmed_defect，另外用 train normal 作参照。
  - 排除 test 图。同一张图标签矛盾时放进 conflicts，不参与打分。engine_update 回滚或 regress_gate 记为 drift。
  - 池内容取 sha256，作为证据绑定的依据。
- `admission_metrics` 是纯函数。它用当前 trad bank 和候选 revision 的组集合，按 score_image 的同一口径（跳过最近邻、取 [1:4]；legacy 用 L2，grouped 用 Σw·‖·‖/scale）重新打分，独立判定五项：
  - gain：AUROC 增益 ≥ 0.005，且组裁剪后不退化。
  - redundancy：与基线分数的相关系数 ≤ 0.95。
  - regression：按来源配对，退化不超过 0.02。
  - G1：bench p95 加上额外耗时，不超过预算；没有 bench 记为 unknown。
  - G2：锚定集 AUROC 不下降超过 0.02；任一侧少于 4 张记为 skipped。
- 五项全部 passed，acceptance 才是 passed。另外几种情况：
  - 含未注册组或 origin=auto_discovery：planned。
  - open 正样本占比 > 0.5、正负任一侧少于 4、grouped 缺权重：insufficient_evidence。
- `evaluate_feature_admission` 写 EvalRun(run_type=feature_admission) 和证据 JSON。不激活、不修改快照。
- `algo/feature_revision.py::_evaluation`：manifest 带 `admission_required` 时（新 draft 默认为 True），进入 approved/active 前要求五项和 acceptance 都为 passed，并且 hard_pool_sha256 合法。
- API：
  - `GET /models/feature-set/{cat}/hard-pool`
  - `POST /models/feature-revisions/{rid}/admission`：后台任务；limit 夹到 4..200。
  - 两者都要求 engineer；operator 访问返回 403，rid 不存在返回 409。
- 学习页增加"难例准入试算"按钮。档案不可用时按钮禁用。

### 26.2 验证

- 新增 `tests/smoke/test_hard_admission.py` 4 项，`test_feature_set.py` 增加路由测试 1 项。smoke 全量 196 OK。
- 覆盖：通过路径，以及 gain、redundancy、G1 失败；open 占比、planned、样本不足；grouped 打分与 `TradSlot._dist` 数值一致；端到端生成证据后，未通过的 revision 被拒批、通过的可批；把 g2 篡改为 skipped 后被拒批。

### 26.3 发现与边界

- 难例池目前全部来自人工反馈。生产 DB 中反馈很少时，结果多为 insufficient_evidence。这是设计上的 fail-closed 行为，不放宽阈值。
- G1 是增量估算（benchmark p95 加上 trad 组数变化带来的耗时差），不是候选 revision 的实测耗时。真正上线前，仍需按 HARD-01 口径实测。
- grouped 模式下，未拟合的组权重用均值、scale 用中位数补齐，并写入 notes。这些组的分数只能作参考，不能等同于重新拟合后的结果。
- 本节只做了 TEMP 测试，没有改动生产 DB 或快照，也没有在真实数据上试算。

## 27. C/D/E 块（AC-018~027 / CDEF1-10）：校准契约、融合基线、路由准入、域差、槽位健康与更新审计（2026-10-12，生产未验收）

### 27.1 实现

- `algo/fusion/cde_report.py`（纯函数）：
  - `calibration_contract`：各槽 CDF 的 bins、样本量和 small_sample 标记，尾部分位，以及 self_match_bias。self 模式标记为高风险，holdout 给出留出侧对照。
  - `fusion_baselines`：用同一份校准分，对比 equal、sanity、consensus、domain_prior、hard_rule_max 几种融合的 AUROC，并标注 production_mode。unverified 槽默认不参与，权重为空的模式不出报告。
  - `stratified_split`：按标签分层切分 train/val/test，保证覆盖完整、比例正确。
  - `router_admission`：在独立切分上比较 router 和固定权重，只有稳定胜出才给 permitted，否则给 trace_only；样本不足时给 insufficient。
  - `domain_shift`：
    - 指标是 median_shift_iqr 和 fpr_inflation，两项都没超阈值记为 normal_drift，任一项超阈值记为 shift，样本数小于 min_n 记为 insufficient。
    - 输出 slot_reliability。mechanisms 里如实写明 `dynamic_routing=not_integrated`。
  - `slot_health`：槽能力表加运行时状态，包括 ok、reverse 和 unverified。open 槽的 in_fusion 固定为 False。
- `Pipeline._build_cde_report`（offline.py）：只读取 fit 阶段已经算好的分数，不做额外前向。
  - 实际生效模式的优先级是 domain_prior > consensus > sanity/equal。
  - 准入结论写回 `router_gate["admission"]`，并生成 `cde_report`，里面带 cold_start、weights_semantics 和 effective_weights。
- `_retrain_router`：槽位名优先用 `router_slot_names`。维度必须等于 `router.n_slots`，并且每个槽都要存在，否则返回 False，不改动 router。
- 更新审计（AC-026）：
  - `RegressGate` 新增 base_scores、audit（上限 500 条）、`_seq` 和 `context_fn`。
  - `check` 复用门控已算好的锚定分，生成 anchor_fpr/gray 前后对比，以及缓存的 recall/ood_fpr，不做额外前向。
  - 结局（accepted/rolled_back/rejected）由门控调用点显式调用 `FeedbackHandler._audit_decide` 写入。
  - `rollback_log` 结构不变。
  - `update_audit_report()` 统计各结局数量。
- 持久化：
  - 保存时写入 `cde_report.json`，`regress_gate.json` 增加 audit、audit_seq 和 base_scores 字段。
  - 加载时恢复 context_fn。旧快照缺这些字段时回退为空。
- 输出语义（AC-027）：`_to_result` 的 weights 取 fusion trace 里的 effective_weights，extra 里附带 `candidate_router.applied=False` 和 `weights_semantics`。
- 对外暴露：
  - `trace.update_log` 增加 audit 字段。
  - 新增 `GET /models/{category}/slot-health`，要求 engineer 权限，operator 返回 403。未准备好时返回 `enabled=False`。
  - `api_client.slot_health`。
  - 学习页新增 Tab「槽位健康与更新审计」，切换到这个 Tab 时才加载。

### 27.2 验证

- 新增 `tests/smoke/test_cde_report.py`，共 9 项：
  - 纯函数 4 项。
  - 审计 2 项。
  - persist 往返和旧快照兼容 1 项。旧快照通过重写 artifact_hashes 清单来模拟。
  - router 维度检查和 effective_weights 1 项。
  - slot-health API 1 项。
- smoke 全量 205 项全部 OK，耗时约 70s。
- 没有在真实数据上评测精度或耗时。

### 27.3 发现与边界

- learning_state 只用严格的 GATE_FIELDS 恢复门控，不包含 audit 和 base_scores。快照里有 learning_state.pkl 时，不会走 regress_gate.json 的恢复路径，结果审计在加载后被清空。现已改为 restore 之后从 regress_gate.json 补读这几个字段，learning_state 的结构不变。
- 候选恢复（candidate learning_state）同样不带 audit 和 base_scores，候选态审计以工作态快照为准。
- router 目前只输出 trace，主融合仍用固定权重。准入结论只是证据，不会自动启用 router。
- 域差 `dynamic_routing` 尚未接入，new_fused 为 None 时只输出定义和训练侧统计。真正判定域差，需要新域的正常样本。
- 快照有 SHA256 清单校验，手工修改快照内文件会被拒绝加载（这是预期的 fail-closed 行为）。
- 审计指标 recall 和 ood_fpr 来自缓存值，不是每次更新时重新实测，只能作为趋势参考。

## 28. F 块（AC-063~065 / LOC-1/2/4/5）：定位契约、区域复核与定位评估（2026-10-12，生产未验收）

### 28.1 实现

- `algo/decision/localization.py`（新建，纯函数）：
  - `postproc_params`：从 cfg.hierarchical 取 mask_thresh、min_area、morph、work_side、top_k，按内容 hash 生成 `version_id`（同参数同版本）。
  - `finalize_regions`：统一为原图 xyxy 坐标，越界裁剪，按峰值降序，可截 top_k；每个区域带 source（engine_box/local_branch/heatmap_cc/human_box）、score、thr、postproc 版本，默认 `status=candidate`、`confirmed=False`。
  - `localization_summary`：输出 coord_space、postproc（完整 dict）、regions、status（ok/no_region）。
  - 复核：`new_review`、`apply_op`（confirm/reject/add/revise，revision 递增、事件记录、revise 记 revised_from），`trainable_region_boxes` 只返回 confirmed。
  - 评估：`iou`、`eval_localization`（实例召回、精度、误定位率、mean_iou）、`select_postproc`（只允许 dev 集，只接受定位指标）。
- `offline.py` 的 `_hier`/`_finalize` 生成 fused_heatmap 和 localization；local_branch 新增的区域在 finalize 之后覆盖 source。
- 引擎 `DetectionResult.fused_heatmap`，`_to_result` 透出 localization；`pipeline/service.py` 的 `_finish_detect` 用引擎框绘制，热图连通域框单独作为 `heatmap_boxes`（source=heatmap_cc，status=visual_only），不再覆盖 defect_boxes。
- `schemas.to_dict` 透出 localization 和 region_review。
- `backend/api/region_review.py`（新建），挂在 routes_review 下：
  - `GET /api/detections/{id}/regions`：返回 coord_space 和当前复核态（没有复核记录时由 localization 生成初始态）。
  - `POST /api/detections/{id}/regions`：engineer 权限；operator、event_id 必填；event_id 幂等（同 payload 返回 idempotent，不同 payload 返回 409）；expected_revision 不符返回 409；非法 bbox 返回 422；region 不存在返回 404；写库使用 n_tiles 乐观锁。
- `self_learning/service.py`：Feedback.region 增加 `box_source="human_box"`，幂等比较只看框本身，不受影响。

### 28.2 验证

- 新增 `tests/smoke/test_localization.py` 共 4 项：后处理版本与 finalize、复核四种操作、评估与选择禁止项、服务/API（真实 ApiKeyMiddleware，覆盖 403/409/422/404 及 DB 落库）。
- smoke 全量 209 项 OK，耗时 68.9s。
- 没有在真实标注定位数据上评测 IoU 或召回，也没有测耗时影响。

### 28.3 发现与边界

- fused_heatmap 是 numpy 数组，只保留在 rec 和 DetectionResult 中，不落库；落库的只有 localization 摘要。
- summary 里的 postproc 是完整参数 dict，region 里的 postproc 只是 version_id 字符串。首次测试就因为把两者混淆而失败，调用方要注意区分。
- heatmap_cc 只用于可视化。旧逻辑用热图连通域框覆盖引擎框，结果会让来源、坐标和阈值都对不上；现在两者分开返回。
- 复核态写在 Detection.n_tiles["region_review"]，没有新建表；修订链只保存在 JSON 中。
- select_postproc 在代码上禁止测试集和 AUC，但目前没有可靠的框/掩码标注集，因此还没有选出冻结的后处理版本。LOC-5 只有评估函数，还没有生产定位评价页。
- LOC-3 统一区域证据面板（AC-073）和复核 UI 尚未实现，留到横切阶段。

## 29. G 块第一批（AC-028/029/032）：库血缘、严格成簇、孵育纯 patch 与独立批准（2026-10-12，生产未验收）

### 29.1 实现

- `algo/ssocl/banks.py`：
  - 血缘（AC-028）：NormalBank 新增 `ext_meta`（与 ext 逐条对齐）和 `core_meta`（seed_core 记 source=train_good_anchor）；DefectBank 新增 `sample_meta`。每条记 op/source/时间/版本/内容 hash（fp16 sha256 前 16 位）/patch 数/image_id，add_box 另记 box。两库 `lineage()` 都标 `memory_not_weights=True`。
  - bank 条目仍是三元组 `[f, ts, x]`，血缘放在并行列表里，因为 `feedback._loo` 会临时替换 `ext`/`samples`。meta 缺失或长度不齐时，`_aligned_meta` 把全部条目标为 legacy，不猜测对应关系。
  - snapshot/rollback 覆盖 ext_meta/sample_meta、簇状态和 seq。
  - 严格成簇（AC-029）：`_cluster_vote` 用“新样本 patch 到簇代表 patch 的最近邻 cos 均值 ≥ cluster_sim(0.90)”判定是否属于同一簇（样本对样本），在 cluster_window(20) 内连续命中 cluster_k 次才 accepted；超出窗口的簇失效。簇代表最多抽样 256 个 patch。
  - 孵育池（AC-032）：`DefectBank.incubation_pool()` 只返回 op ∈ {add_box, add_patch} 且非 legacy 的条目，id 优先用 image_id，缺失时用 hash。
- `algo/ssocl/incubate.py` 重写：触发和重孵育都按独立图计数；用最新 `holdout_images`(2) 张独立图做 holdout；新头 separation 必须 ≥ `approve_min_sep`(0.2)，且不劣于旧头才替换，否则保留旧头，并在 log 中记录 approved/reason（below_min_sep / worse_than_current / holdout_pass / no_holdout / train_error）。被拒后需再新增 ≥max(1, n_hold) 张独立图才重试。
- `algo/ssocl/feedback.py`：缺陷入库（add/add_patch_feats/add_box）和两处 add_ext 都透传 `image_id=item["path"]`。
- `backend/engine/__init__.py`：学习状态新增 `lineage` 摘要（按 op/source 计数、legacy 数、最近 3 条、core hash、pending 簇数）；incubate 的 progress 改为按纯 patch 独立图计数，并透出 separation 和 last_decision。

### 29.2 验证

- 新增 `tests/smoke/test_bank_lineage_incubate.py` 共 11 项：不相似候选保持 pending、相似候选第 k 次 accepted、超窗口打断连续性、血缘字段与回滚、旧对象标 legacy 后仍可写、整图 add 不进孵育池、同图多框只算 1 张、legacy 样本被排除、按图计数而不是按框计数、4D 锚定负样本可训练并通过批准、门槛不可达时拒绝并按图数节流重试、新头劣于旧头时拒绝并保留旧头。
- smoke 全量 220 项 OK，耗时 64.7s。
- 没有在真实品类上评测孵育头对 AUROC 或 FPR 的影响（留到 OPT 阶段）。

### 29.3 发现与边界

- 旧孵育头实际从未训练成功：`_anchor_feats` 返回 (T,D,G,G)，旧代码直接和 (N,D) 正样本 concat，维度不符，异常被 except 吞掉。修复后孵育头会真正参与拦截，可能影响精度，所以必须保留 holdout 批准门禁；OPT 阶段要对照有头和无头两种情况。
- holdout 只用了“纯缺陷 patch vs 锚定 patch”，没有覆盖域外正常图的误报，也没有按缺陷类型或域分组计数（AC-032 中“按类型/域计数”的部分仍缺）。独立图数不足以留出时会以 no_holdout 放行，这是有意保留的冷启动行为，会写入 log。
- 严格成簇比旧的 0.01 分桶计数严格，无标注自动回流会更难进入扩展库；带标注回流路径（`_reflow_labeled`）不受影响。cluster_sim=0.90 没有在真实数据上标定。
- DefectBank 仍没有容量上限或淘汰机制（原状），血缘列表随条目线性增长。
- 旧快照加载后，血缘一律显示为 legacy，legacy 条目不进孵育池。因此旧快照上已有的孵育头会保留，但不会因为 legacy 样本触发重孵育。

## 30. G 块第二批（AC-030/031）：四类反馈分路、无反馈默认不回流、缺陷入库锚定 FPR 门控与 init_defect 掩码预置（2026-10-12，生产未验收）

### 30.1 实现

- `algo/ssocl/feedback.py`：
  - 分路（AC-030）：静态方法 `route_of(verdict, label)` 把反馈映射为四类路由，entry 初始化时就写入 `route`（之前写在 deferred 早退之后，导致部分条目缺失）。
  - 无反馈默认不回流：`unlabeled_reflow` 默认关闭，这时无标签样本只会进入 low_value_skip / high_conf_cache / active_queue。打开后要同时满足以下条件才调用 `_reflow`：高置信正常（fused ≤ ua_max_fused=0.2）、open 在域内（≤ ua_max_open=0.5，open 为 None 时保守拒绝）、严格成簇（§29）、熔断和锚定门控。不满足时分别记为 not_high_conf_normal / domain_unverified。
  - 缺陷入库门控（AC-031）：`_add_defect_sample` 末尾调用 `_defect_bank_gate(source)`，复用 RegressGate 的 check 和审计结果（anchor_fpr before/after），不增加前向。drift 超限，或锚定 FPR 上升超过 `defect_gate.fpr_tol`（默认 0）时回滚 DefectBank 快照，并写入 `audit[-1]["decision"]`。
  - 三处缺陷入库（correct_defect / false_negative / review_defect）都会在 entry 中写入 `_defect_scope`：defect_gate（accepted/rolled_back）、scope（独立纯 patch 图数 < incubate.min_samples 时为 similar_only，否则为 incubation_ready）、n_pure_images。
  - `seed_init_defect(patch_sets)`：把 init_defect 的掩码 patch 以 add_patch 方式整批预置，最后统一过一次门控，失败时整批撤销。action 取值为 init_defect_seed_skip / accepted / rolled_back。
- `algo/eval/offline.py`：
  - 新增 `_masked_defect_patches`：掩码按 >127 二值化后，在 G×G 网格上求覆盖率，取覆盖率 > 0.3 的 patch。K<4 时跳过，K>64 时用 linspace 下采样，结果存为 fp16。
  - fit 中保存 `_init_defect_patches`。`enable_ssocl` 只在 `init_defect_seed.enabled=True`（默认 False）时才预置，结果记在 `self.init_defect_seed`。处理完后清空 patches，避免进入快照。

### 30.2 验证

- 新增 `tests/smoke/test_feedback_routes.py` 共 8 项，全部 OK。覆盖内容：
  - route_of 映射。
  - 默认不回流时的三种去向。
  - 开启后被拒的 not_high_conf_normal 和 domain_unverified（包括 open=None）、通过时调用 _reflow。
  - FPR 上升时回滚并写入 decision、tol 内放行、drift 超限回滚。
  - scope 两种取值。
  - seed 的 skip / accepted / 整批 rolled_back。
  - 掩码 patch 的形状、dtype、取值，以及无掩码时返回 None。
- smoke 全量 228 项 OK，耗时 67.5s。
- 没有在真实品类上评测 init_defect_seed 开/关对精度的影响（留到 OPT 阶段）。

### 30.3 发现与边界

- torch 高级索引陷阱：`f[t, :, ii, jj]` 会把高级索引维度放到最前，结果是 (k,D) 而不是 (D,k)，之后再 `.T` 就错了。必须先取 tile：`f[t][:, ii, jj].T`。
- FeedbackHandler 没有 `self.cfg`，新配置只能在 `__init__` 中读入属性。读取时用 getattr 兜底，兼容旧 pickle。
- intercept_score 只增不减，所以缺陷入库的主要风险是正常样本误报。门控只看锚定正常集的 FPR，没有域外 FPR holdout。fpr_tol=0 偏严格，可能误拒有效样本，需要在 OPT 阶段标定。
- 无掩码的 init_defect 不生成 patch，YOLO 框路径尚未接入。检出缺陷时不重复前向，所以缺陷侧没有召回门控。
- 无 open_score 的部署（open 槽位未启用）在打开 unlabeled_reflow 后也不会回流，这是有意的保守行为。

## 31. G 块第三批（AC-033/034）：槽与新头职责、E_open 难例原因归类

### 31.1 实现
- AC-033：`algo/fusion/cde_report.py` 新增两张职责表。
  - `SLOT_ROLES`：disc / sem / inp 三项，记录职责、训练、更新和「不承诺」。
  - `POST_FUSION_ROLES`：defect_knn / incubate 两项，在上述字段外还记录定位和域敏感。
- AC-033：`slot_health` 的行数据合并职责字段，并新增 `post_fusion`（来自 `post_fusion_health(pipe)`）。
  - 运行态固定为 in_fusion=False。
  - defect_knn 给出 active、n_samples、n_pure_images。
  - incubate 给出 enabled、trained、min_samples、n_pure_images、separation、last_decision。
  - 缺少属性时用 getattr 兜底。
- AC-034：`backend/api/open_alerts.py` 的 OpenDisposition 新增可选字段 `cause`，取值 drift / noise / registration / new_defect / unknown。
  - new_defect 只能配 confirmed_defect。
  - drift / noise / registration 不能配 confirmed_defect。
  - 违反以上组合返回 422。
- AC-034：cause=new_defect 时，item 和 Feedback.learning_trace 中写入 `admission` 候选，内容为 kind=feature_admission、status=candidate、auto_online=False，`required_checks` 为 gain / redundancy / regression / g1 / g2 / version。
  - 它不触发学习，也不注册特征，Feedback 仍为 candidate_only。
  - 实际准入走已有的 `hard_admission.evaluate_feature_admission`（AC-017）。
- AC-034：不传 cause 时，事件 payload 与旧格式一致，旧事件的幂等比较不受影响。
- AC-034：`hard_admission.build_hard_pool` 读取 trace.open_cause。
  - cause=drift 的样本标记为 drift。
  - counts 新增 `open_causes`，按原因单列计数。
- AC-034：UI `open_alert_queue` 处置框新增「原因归类」下拉框。默认「未归类」，此时不发送 cause。

### 31.2 验证
- `test_bank_lineage_incubate.TestPostFusionRoles`：
  - disc / sem / inp 都有职责字段，post_fusion 不进融合。
  - 样本为 add_box + add 时，n_samples=2、n_pure_images=1。
  - 空 pipe 时为 inactive。
  - 与 test_cde_report 合跑 21 项 OK。
- `test_open_alert_queue`：
  - 三类矛盾组合或非法 cause 返回 422。
  - new_defect 生成候选，auto_online=False，含 version 检查，重复提交幂等。
  - drift 处置不生成 admission。
  - 两条 trace 的 open_cause 正确且都是 candidate_only，所有 Feedback 都未 consumed。
- `test_hard_admission.test_open_cause_counted`：
  - 三种原因计数正确。
  - open_defect / open_normal / drift 计数为 1 / 2 / 1，drift 标记只落在 cause=drift 的图上。
- 全量 smoke：230 项 OK，耗时 65.6s。

### 31.3 发现与边界
- disc 不是冻结头：feedback 中 disc_ft 默认开启，会用带框 patch 在线微调，gate 否决时回滚，DINO 不动。职责表已按这个事实描述，不写成「disc 不更新」。
- AC-017 的准入门禁（增益 / 冗余 / 分来源回归 / G1 / G2，open 正样本占比 >50% 判为 insufficient_evidence）已经存在。AC-034 的工作是把人工原因接入难例池，没有另造门禁。
- 尚未完成：
  - 准入候选只是登记，不会自动生成 FeatureSetRevision 草案，需要工程师手动发起试算。
  - 处置后的检出 / 误报效果还没有关联回该告警。
  - noise / registration 样本仍按 open_normal 进入负池，没有单独剔除或降权。

## 32. G 块第四批（AC-035/036）：快照组件清单与禁混库、固定初始尾部与累计门禁

### 32.1 实现
- AC-035 `algo/persist.py`：
  - `SNAPSHOT_COMPONENTS` 把快照工件归为 9 个组件（normal_bank / defect_bank / sem_and_slots / incubate_heads / calibration / weights / regress_baseline / learning_state / config）。meta 里的 `components` 记录每个组件的 files 和 present。
  - `CACHE_INVENTORY` 盘点 4 类不随快照传播的缓存：backbone 特征缓存、ext hash、normal_bank 流统计、global_defect_clf。
  - meta 新增 `code_sha256`（algo 包全部 .py 的聚合哈希）和 `rng`（policy / seed / feature_random_seed）。
  - 加载时的处理：
    - 比较 backbone 哈希，结果写入 `pipe.backbone_representation`。
    - 哈希不一致时发出告警；candidate=True 时直接 raise「禁止混库」。
    - 代码哈希不一致只记录到 `pipe.code_drift`，不阻断。
- AC-036 `algo/ssocl/feedback.py` RegressGate：
  - setup 时固定 `init_mean / init_std / init_p95 / init_tail`，recenter 不修改这些值。
  - `cumulative()` 计算两项：相对初始基准的累计漂移（以 init_std 为单位，带符号），以及锚定集超过 init_p95 的比例。
  - 判定规则：上行累计 >`cum_cap`=4，或尾部比例比 init_tail 增加超过 `tail_cap`=0.10，即判越限。
  - 连续 `freeze_after`=3 次违规后冻结。冻结期间所有更新都回滚，必须调用 `unfreeze()` 人工恢复；`rebase=True` 会以当前分布重置初始尾部。
  - 越限或冻结时，check 把 drift 抬到 >tol，并把 new_mean 取为 ≥base_mean，所以各调用点（包括 `_reflow` 的定向门控）不用改就会统一走回滚分支。
  - history 和 audit 都记录 cum_drift / tail_rate / frozen。
  - persist 写入 regress_gate.json 时附带 `PERSIST_FIELDS`。learning 和非 learning 两条加载分支都用 `_restore_gate_ac036` 回填。旧快照没有这些字段时保持默认值（init_*=None），不做累计判定。

### 32.2 验证
- `test_feature_registry`：manifest、code hash、rng、缓存盘点、representation_drift 三种情形都通过。
- `test_regress_gate_ac036`（4 项）：
  - 每步 +0.02 的小步上行漂移，单步门控能放行，但累计到第 N>1 步时被回滚；recenter 后 init_mean 不变。
  - 下行漂移不触发累计拦截。
  - 连续 3 次违规后冻结；冻结期间正常分布也回滚，且 new_mean≥base_mean；unfreeze 后恢复。中途通过一次会让违规计数清零。rebase 会重置 init。
  - status 能 JSON 往返；旧快照和旧 pickle 对象缺少新属性时 check 照常可用。
- 全量 smoke：234 项 OK，耗时 67.7s。

### 32.3 发现与边界
- 小锚定集的「超 p95 比例」不等于 0.05。例如 4 张图时为 0.25，所以尾部门禁改用相对 init_tail 的增量。如果直接用绝对值 0.10，test_cde_report 里原来能通过的检查会被误拦。
- 累计门禁只拦上行。锚定集全是正常图，下行是 recenter（虚高 CDF 重估）的预期方向，由单步门控约束。如果之后要监控「正常分下移掩盖缺陷」，应该用独立缺陷锚，不应该用双向上限。
- 尚未完成：
  - 独立缺陷 / 域外误报 / G1 / 显存锚点进入门禁。
  - 按更新类型设置不同容忍度。
  - 组件级回滚（目前仍是全快照或调用点级回滚）。
  - 冻结状态在 UI 中展示。
- `cum_cap` / `tail_cap` 是经验值，还没有用真实数据集的持续学习轨迹标定，需要在 OPT 阶段用少量图复核。

## 33. G 块第五批（AC-037 / GHIJK10）：分机制收益对照与事实标记

### 33.1 实现
- `algo/fusion/cde_report.py`：
  - `MECHANISM_FACTS` 按代码的实际行为标注机制状态：
    - auto_reflow：conditional。
    - clustering：implemented_limited（余弦成簇，不是无监督发现）。
    - multi_head：not_implemented（只有单头）。
    - auto_feature_registration：not_implemented（只登记 admission 候选）。
  - `mechanism_gain_report(variants, tau, ablations, groups, n_defect_used)` 是纯函数：
    - variants 的口径为 patchcore / fixed_fusion / current / new_head，必须包含 current，缺失的口径列入 `missing_variants`。
    - 对每个口径输出 AUROC、FPR@τ、Recall@τ。
    - 与 current 等长的口径输出配对 bootstrap ΔAUROC 的 95% CI。
    - ablations 中每个机制给出 on/off 的 gain、CI，以及 FPR/Recall 副作用，可按 groups 分报。verdict 由 CI 判定，取 gain / harm / inconclusive。
    - disclosure 字段写明当前策略是否低于 PatchCore、PatchCore 不消耗缺陷样本、当前策略消耗的缺陷数；当前策略低于 PatchCore 且额外消耗缺陷样本时，强制给出 note。

### 33.2 验证
- `test_cde_report.MechanismGainTest`（2 项）：
  - 三种 verdict、分组副作用、missing_variants、JSON 序列化。
  - 低于 PatchCore 时的披露、样本数不等时跳过配对 CI、缺少 current 时 ValueError。
- 全量 smoke：236 项 OK，耗时 72.3s。

### 33.3 发现与边界
- 目前只有报告函数，还没有接入真实取分流程（关闭记忆 / 近邻 / 孵育 / open / 回滚的推理开关）和 API。调用方需要在同一评测集上产出各口径的分数。真实数据对照放到 OPT 阶段用少量图做，结论不外推到生产验收。
- 配对 CI 要求各口径样本顺序一致。等长但顺序不同时，函数无法检测，只能由调用方保证。

## 34. H 块第一批（AC-038）：工作/候选隔离守卫

### 34.1 实现
- 核实已有隔离底座（未改动）：`engine.learn_candidate` 品类 `.candidate.lock` 非阻塞 OS 锁（冲突即拒）；持久意图 candidate_events（training/prepared/completed/rejected）；训练前后复核父版本、WAL 边界（≤64KiB）与父快照 hash；`.candidate-*` 暂存区与 CandidateBudget；`service.submit_feedback_record` 默认 `candidate_only=True`、`allow_legacy_online=False`，否则 FeedbackConflict；`_shared_dinov2` 进程级共享、eval、requires_grad=False。
- 新增 `tests/smoke/test_work_candidate_isolation_ac038.py` 回归守卫：
  - AST 扫描 backend/api、ui、backend/self_learning：禁止 `candidate_only=False`、`allow_legacy_online=True`，禁止在 service.py 之外调用 `*engine*.submit_feedback`。
  - service legacy 分支签名默认值与拒绝行为。
  - 两个 FrozenDINO 共享同一 `.model`、eval、无可训练参数（无权重时 skip）。

### 34.2 验证
- 新测试 3 项 OK（5.4s）；全量 smoke 239 项 OK（72.8s）。

### 34.3 发现与边界
- `engine.submit_feedback` 本身仍会直接修改工作态，只靠调用方守卫；bench/m01 测试仍直接调用，属于实验路径。
- consolidate 仍基于 consumed 反馈构建，不是事件水位驱动（归入 AC-039）。
- “teacher 固定”目前只体现在主干冻结共享层面；L1 teacher/student 协议未实现（AC-044/045）。
- 守卫是静态 AST 扫描，动态拼参（如 `**kwargs`）无法覆盖。

## 35. H 块第二批（AC-042）：复核阶段最小状态契约

### 35.1 实现
- `backend/api/routes_review.py` 复用 `Detection.n_tiles` JSON 保存 `review_stage`，不新增数据库列，兼容旧记录：无字段时按有效反馈推断为 `queued` 或 `submitted`。
- 复核提交只允许 `queued → submitted`；已提交、未知阶段、显式阶段不匹配均返回 409。反馈成功落库后才持久化 `submitted`，避免失败时提前推进。
- 新增 `tests/smoke/test_ac042_review_stage.py`，覆盖旧调用兼容、重复提交、未知阶段和非法显式阶段。

### 35.2 验证
- 定向测试：4 项 OK。
- 相关 `test_feedback_routes`：8 项 OK。
- `py_compile`：routes_review.py 与新增测试通过。

### 35.3 发现与边界
- 当前只实现复核接口的最小 `queued/submitted` 契约，尚未覆盖完整 L0/L1/L2、候选评估/批准/激活状态链。
- 状态暂存于 `n_tiles` JSON，历史记录不批量迁移；时间释放、延迟乱序标签审计和批次近重复隔离仍属于 AC-042 后续验收。

## 36. H 块第三批（AC-039）：WAL 回放审计边界

### 36.1 核对结果
- `AOI_tree/AOI_Core/backend/engine/__init__.py::_replay_journal` 在装载快照后按 `base == version` 过滤 WAL，逐行记录 `invalid_json`、`invalid_record`、`conflict`、`missing_image`、`replayed`、`replay_failed` 等状态；相同身份且 payload 不同会拒绝回放，成功的反馈 receipt 另行写入已有 receipt 表。
- 回放审计只赋给进程内的 `_last_journal_replay_audit` 列表；代码没有将该审计列表写入 WAL、receipt SQLite、快照或独立审计表。进程退出、重启或对象替换后，逐行回放结果不可由该审计列表恢复。
- 因此 AC-039 当前是“有回放并记录进程内诊断”，不是可持久追溯的 WAL 回放审计；receipt 只记录反馈身份幂等，不等同于回放审计结果，也不能证明每条跳过/失败原因已持久保存。

### 36.2 验证与边界
- 已有反馈幂等验证：**11 passed / 3.51s**。
- 合并定向回归：**51 tests / 16.721s，OK**。
- 全量 **239 / 72.841s** 是此前基线结果，不是本次修改后的最新结果；本次按要求不跑测试，不能把该基线写成当前回放审计的复验结果。
- 该实现仍未提供跨进程/重启后的审计连续性、WAL 截止水位对应的持久回放清单，AC-039 不具备生产验收依据。


## 37. H 块第四批（AC-043）：独立回放证据边界

### 37.1 核对结果
- `AOI_tree/AOI_Core/backend/engine/__init__.py::_replay_evidence` 校验父版本仍为 current，生成 `replay_run_id`、父版本、WAL 水位、输入数量与输入摘要 SHA256、配置 hash；返回值仅在内存中生成，不写生产 DB、快照或独立 `replay_evidence` 持久化表。
- `AOI_tree/AOI_Core/backend/self_learning/candidate_release.py::_validate_replay_evidence` 只在调用方提供 `replay_evidence` 时校验字段完整性、父版本/current 一致、WAL 水位一致和非空 `replay_run_id`；证据绑定是可选的，缺少证据不会由该校验函数强制拒绝。
- 候选报告可以把证据对象嵌入持久化报告，但这属于候选报告的可选绑定，不是独立回放证据账本；没有独立 run 状态、输入明细/产物持久化、重启后可查询的证据记录，也没有强制证明回放实际独立于生产工作态执行。

### 37.2 测试核对与验收结论
- `AOI_tree/AOI_Core/tests/smoke/test_candidate_release.py` 已有测试覆盖：证据生成的 run_id/父版本/WAL 水位/输入计数、receipt 库不存在 `replay_evidence` 表，以及父版本或 WAL 水位不匹配时拒绝。
- 上述测试证明当前“可生成、可选绑定、可校验水位”的契约，不证明证据独立持久化或重启可追溯。反馈幂等 **11 passed / 3.51s** 与合并定向 **51 tests / 16.721s OK** 为已有实际结果；全量 **239 / 72.841s** 仅为此前基线，均不能改写为本次最新全量结果。
- AC-043 仍未达到生产验收：独立回放证据既非强制绑定，也未独立持久化；当前只能标记为部分实现/生产未验收。

## 38. H 块第五批（AC-040/041）：只读发布发现与学习态 RNG 严格预验证

### 38.1 AC-040 只读连接与 prepared 发现
- 新增两个只读 API：`readonly_connection` 用于以只读方式访问发布账本，`discover_prepared` 用于发现待恢复的 prepared 记录；历史账本数据库缺失时只读检查不创建数据库、不补写历史状态。
- 定向回归：**39 tests / 16.769s / OK**。覆盖只读访问、历史缺 DB 不创建、prepared target/parent/conflict 发现及边界；测试使用 TEMP 账本/快照，不写生产数据库或快照。
- 实际边界与建议：发现结果应继续以 prepared 的 target、parent 和冲突信息作为人工恢复/拒绝依据；缺 target、parent 不一致、冲突或外部指针不应猜测修复，也不能把“发现到 prepared”当成已提交、已激活或生产恢复成功。业务镜像、跨介质原子性、自动启动恢复、多进程一致、真实数据与鉴权仍未验证。

### 38.2 AC-041 学习态与 RNG 严格预验证
- 当前 v2 学习态已具备 handler/cache/gate/config、局部 NumPy RNG、CPU Torch generator、全局 Python/NumPy/Torch RNG 与 CUDA RNG 字段；本轮新增 RNG 严格预验证及 4 项测试，校验字段、类型、形状、有限值和当前 CUDA 环境兼容性，非法或不兼容状态 fail closed。
- 组合学习态相关回归：**26 tests / 7.295s / OK**。这是 TEMP/CPU 小头和替身边界内的恢复契约验证，不是完整生产学习态验收。
- AC-041 仍为部分实现、生产未验收：真实 GPU、跨环境/跨版本一致性、完整有效事件回放、重启/LRU 后全轨迹等价、完整业务状态与审计持久化仍缺；不得将本轮预验证或 CPU 结果表述为生产通过。

## 39. AC-050 候选事件归属修复（生产未验收）

### 39.1 修复与边界
- `CandidateRelease.status` 查询现在按 `candidate_id` 过滤，避免不同候选之间发生状态串线。
- 对历史缺少 `candidate_id` 的记录，仅按 `payload.candidate_event` 做精确匹配；无法确认归属的记录标记为 `orphan/unavailable`，不猜测归属，也不继续推进候选状态。
- 新增候选串线和历史缺字段覆盖，验证候选状态隔离、精确事件匹配及无法确认时的 fail-closed 结果。

### 39.2 验证结果与未解决项
- 定向回归：**41 passed、77 warnings、19 subtests**。本结果为 TEMP/小样本证据，不等同生产验收；warnings 仍按既有依赖/运行环境问题保留，未通过升级依赖消除。
- 本次修复只解决候选事件归属与状态查询的串线风险。生命周期时间线、水位、结构化失败原因、取消续接仍未解决；AC-050 仍为部分实现，生产验收仍为 **0**。
- 未修改生产数据库/快照，未提交 git。

## 40. AC-047 候选守卫最小修复（2026-10-13，生产未验收）

### 40.1 修复与边界
- service 在候选构建开始记录 `candidate_guard`，包含 `feedback_id`、`detection_id`、`invalidated` 与区域 `revision`；`candidate_release` 在验收、批准、激活前复核该守卫。
- 反馈作废或区域 revision 发生变化时，候选发布路径 fail-closed；旧候选缺少 `candidate_guard` 时保持兼容，不臆造守卫信息。
- 本次只建立候选构建到发布前的最小一致性守卫，不等同于完整争议处理或撤错重建。完整争议修订链、独立参考标签、撤错后的可信重建仍缺。

### 40.2 定向验证与验收结论
- 候选发布定向回归：**45 项通过**，其中本轮新增 **4 项**；覆盖守卫记录及验收/批准/激活前复核、作废/区域 revision 变化 fail-closed、旧候选无守卫兼容。
- `py_compile` 通过；生产指针保持不变。证据仍限于定向 TEMP/小样本边界，未修改生产数据库/快照，未提交 git。
- AC-047 仍为部分实现，生产验收仍为 **0**：完整争议修订链、独立参考标签、撤错可信重建缺失，不能据此宣称 L2 争议隔离或撤错闭环完成。

## 41. AC-048 CandidateRelease 结构化撤错前置守卫（2026-10-13，生产未验收）

### 41.1 修复与边界
- `CandidateRelease` 增加结构化 `reason_code`，用于记录候选验收/批准/激活拒绝原因；作废或人工修订候选在 `validate`、`decide`、`activate` 前均 fail-closed，候选标记为 `blocked`，要求从可信 parent 重新构建，不通过只改数据库状态绕过。
- 本轮不改生产版本；该切片只建立撤错影响尚未可信重建时的拒绝与阻断边界，不等同于已完成完整撤错重建。

### 41.2 定向验证与验收结论
- 定向回归：**45 tests / 25.369s / OK**；`py_compile` 通过。证据仍限于 TEMP/小样本边界，未修改生产数据库/快照或生产指针。
- 尚未实现完整撤错重建、参考标签重算、特征库头 CDF 阈值重放；AC-048 更新为**部分实现**，生产验收仍为 **0**。

## 42. AC-049/050 配对评估与候选失败原因结构化（2026-10-13，生产未验收）

### 42.1 AC-049 配对指标最小实现与边界
- 新增 `backend/evaluation/paired.py` 的 `paired_prediction_delta`：在相同 seed 与完整样本 ID 集合上对齐 candidate-baseline，计算 AUROC、FPR、recall、gray delta；缺少 seed、样本 ID 重复或缺失、标签/样本集不一致、NaN/Inf 或单类别时返回 `unavailable`，不虚构置信区间。
- 新增 **5 项测试，1.379s 通过**。证据仅覆盖配对输入校验和小样本指标计算，不等同于六臂同预算训练或生产精度验收。

### 42.2 AC-050 结构化候选原因最小实现与边界
- 扩展 `CandidateRelease` 的 `reason_code`、`violations`、`retryable`；验收报告以及构建失败/预算延期写入结构化原因，`learning_trace` 保留原因，避免只用自由文本或状态名表达失败。
- 合并定向 `tests.smoke.test_paired_evaluation` 与 `test_candidate_release`：**52 tests / 22.319s / OK**（命令总耗时 **26.607s**）。本结果为 TEMP/小样本回归，不等同生产验收。

### 42.3 未实现范围与验收结论
- 尚未实现六臂同预算训练、配对 CI、多流多种子、完整生命周期时间线/水位/取消续接；AC-049/050 均仍为**部分实现**，生产验收仍为 **0**。
- 未修改生产数据库/快照或生产指针，未提交 git。以上新增实现不得表述为生产通过。

## 43. AC-076/073 生产入口禁止 allow_stale 绕过 + AC-055 反馈审计结构化关联（2026-10-13，生产未验收）

### 43.1 重要发现：生产入口可绕过特征表示 stale 门禁
- `POST /api/models/{id}/activate` 与 `POST /api/models/rollback` 曾接受请求体 `allow_stale=true`，并原样转发给 `engine.activate`。管理员可借此绕过特征表示 stale 门禁，直接切换生产指针；stale 门禁在 HTTP 生产入口上形同可选项。
- 设计含义：领域层的“允许 stale”开关不能暴露为生产 API 参数；绕过门禁需要独立、审计绑定的恢复授权协议，不能靠请求字段完成。

### 43.2 修复与边界
- `routes_models.py` 新增 `_reject_stale_override`：请求显式给出 `allow_stale=true` 时返回 **409**，`reason_code=STALE_OVERRIDE_FORBIDDEN`，`retryable=False`。该检查位于 resolve、门禁与操作日志之前；rollback 的检查放在 `try` 之前，避免被通用异常处理吞成 500。
- service/engine 调用在生产入口固定传 `allow_stale=False`；`schemas.RollbackRequest` 保留 `allow_stale` 字段用于兼容旧客户端解析，但值为 true 时拒绝。
- 未改 engine/service 签名及离线调用。边界：领域层 `engine.activate(allow_stale=True)` 仍可被内部代码调用，尚无审计绑定的恢复授权协议；普通快照激活仍不强制走 CandidateRelease。
- 验证：`test_stale_override_forbidden` 新增 4 项，与 `test_representation_rebuild` 合计 **9 tests / 4.1s / OK**（TEMP/定向回归）。

### 43.3 AC-055 反馈审计 OperationLog 结构化关联
- `feedback_candidate_queued`、`review_submit`、`feedback_invalidate` 三处 OperationLog 补充 `extra`：`feedback_id`、`detection_id`、`source_event_id`、`category`/`learning_status`、`review_stage`/`origin_decision`、`invalidated`/`was_consumed`、`detection_model_version`。
- `detection_model_version` 只取 `Detection.n_tiles.model_version`，缺失记 `None`，不用 `current_version` 补，避免用当前态覆盖检测时版本。`action`、`detail` 与接口返回值不变。
- 验证：新增 `test_feedback_audit_extra` 4 项（TEMP SQLite 真实落库），与 `audit_contract`/`feedback_evidence`/`ac042`/`feedback_routes`/`feedback_idempotency` 合计 **41 tests / 3.7s / OK**。
- 边界：未测 `was_consumed=True` 分支；检测/反馈/候选/激活四时点完整关联未完成；日志写入与业务写入不在同一事务；候选批准/激活日志未覆盖。

### 43.4 验收结论
- 全量 smoke：275 tests / 77.150s / OK（2026-10-13，含 §42/§43 改动；上次 239 项）。
- AC-055/073/074/076 均仍为**部分实现**，生产验收仍为 **0**。未修改生产数据库/快照或生产指针，未提交 git。

## 44. 稳定版独立验收（2026-10-13，本对话）

### 44.1 验收范围与方法
- 验收对象：`d:\CGAIC\AOI_SSOCL\stable_AOI_tree\AOI_Core`（稳定版）。
- 验收前提：开发版（`d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`）已完成 AC-049/050/055/076 及 SYNC-01/SYNC-02 批次同步。
- 验收方法：14 个关键文件 SHA-256 对比确认一致性；`py_compile` 编译检查；稳定版独立运行定向测试，不引用开发版结果。
- 验收纪律：稳定版验收通过 ≠ 生产验收；所有测试均为 TEMP/小样本/替身边界。

### 44.2 一致性核查
- 14 个关键文件 SHA-256 与开发版完全一致：
  - `backend/api/routes_models.py`、`routes_feedback.py`、`routes_review.py`、`routes_learning.py`
  - `backend/self_learning/candidate_release.py`、`backend/self_learning/service.py`
  - `backend/evaluation/paired.py`
  - `ui/pages/learning_page.py`、`ui/widgets/charts.py`
  - `tests/smoke/test_stale_override_forbidden.py`、`test_feedback_audit_extra.py`、`test_paired_evaluation.py`、`test_candidate_release.py`
  - `tests/ui/test_missing_metric_charts.py`
- `py_compile`：5 个关键文件全部通过（exit=0）。
- `requirements.txt` 与 `requirements-lock.txt` 两树逐字节一致。

### 44.3 稳定版定向验收测试
- 环境：`PYTHONDONTWRITEBYTECODE=1`、`QT_QPA_PLATFORM=offscreen`。
- 四文件合并测试：**17 tests / 1.824s / OK**。
  - `test_stale_override_forbidden`：4/4 OK（activate/rollback 的 allow_stale=true 均被 409 拒绝，默认路径不变）。
  - `test_feedback_audit_extra`：4/4 OK（三处 OperationLog 结构化 extra 真实落库）。
  - `test_paired_evaluation`：5/5 OK（配对差值安全项）。
  - `test_missing_metric_charts`：4/4 OK（缺失值不画零柱、NaN 断线）。
- 追加 `test_candidate_release`：**47 tests / 23.252s / OK**（含 IsolatedCandidateTest 套件）。
- 合计：**64 用例全部通过，0 失败 0 错误**。
- 非阻断性警告：sklearn 版本不一致（1.9.1 pickle ↔ 1.9.0 运行时）、Starlette/Pydantic 弃用警告，与开发版一致，不影响结果。

### 44.4 稳定版验收结论
- 稳定版当前与开发版完全同步，可作为验收基线。
- AC-049/050/055/073/074/076 在稳定版均为**部分实现**，生产验收仍为 **0**。
- 未修改生产数据库/快照或生产指针，未提交 git。

## 45. UI 缺失值收尾 + AC-055 was_consumed + AC-074 候选审计（2026-10-13，开发对话，生产未验收）

### 45.1 AC-061～070 UI 缺失值（SYNC-02/02b）
- 学习/统计/评估页缺失值统一为 NaN（`num_or_nan`），LineChart/LatencyChart `connect="finite"` 断线、BarChart 跳过缺失柱；趋势 reviewed==0、延迟 n_inspected==0 的日期置 NaN，真实 0 保留。
- `routes_learning` contribution latest 透出 `trad_groups`（此前组级表恒空）。
- 重要发现：`test_gauge_contract` 伪页面缺 `lbl_trace`/`btn_explain`，是稳定版 tests/ui 13 个既有 error 的来源，已修夹具。tests/ui **15 tests / 3.421s / OK**。
- 边界：后端 `routes_stats` 无数据日仍返回 0.0，只在 UI 端识别（改 None 影响面大，暂不改）。

### 45.2 AC-055 was_consumed=True 分支
- 新增 `test_invalidate_consumed_feedback`：已消费反馈作废返回 `(invalidated, was_consumed)=(True, True)`，note 含“回滚”，extra 结构化且无禁用字段；重复作废仍如实返回 was_consumed=True，不新增日志。`test_feedback_audit_extra` **5 tests / 0.882s / OK**。

### 45.3 AC-074 候选写操作结构化审计
- 重要发现：`routes_candidates` 的 decision/activate/rollback/recover 四个端点此前**不写任何 OperationLog**，批准人/理由/激活/回滚无审计。
- 修复：新增 `audited()`；成功写 `candidate_decision|activate|rollback|recover`，extra 含 category/version/(approved/report_hash/reason)/ok/state/idempotent；409 拒绝写 `<action>_rejected`，extra 含 reason_code/retryable；日志写失败只 warning，不掩盖业务结果。返回值与错误契约不变。
- 新增 `tests/smoke/test_candidate_audit_log.py` 3 项（成功/幂等、拒绝含 reason_code、审计失败不影响结果）。合并 `test_candidate_release`、`test_feedback_audit_extra`：**55 tests / 21.577s / OK**。
- 边界：日志与业务不同事务；普通快照激活仍不强制走 CandidateRelease；AC-055/074 仍部分实现，生产验收 **0**。未改生产 DB/快照，未提交 git。


## 46. AC-063 定位容器元数据 + 全量 smoke（2026-10-13，生产未验收）

- 全量 smoke（AC-063 前，含 SYNC-03）：**279 tests / 80.481s / OK**。
- AC-063：`_finish_detect` 的 `n_tiles.localization` 新增 `coord="image_xyxy"`、`image_size=[w,h]`（原图尺寸，非显示缩放）、`model_version`、`heatmap_postproc={rel_thresh,min_area,display_scale}`（仅 with_heatmap 时）。只增字段，regions/heatmap_boxes 语义不变。
- `test_localization` 补 4 条断言；与 `test_display_order_equivalence` 合跑 **6 tests / 0.621s / OK**（源码替换式等价测试未受影响）。
- 边界：未保存原始/二值 mask、形态学参数，human_box/source_slot/确认训练状态仍缺，AC-063 仍部分实现；生产验收 **0**。待验收对话同步 SYNC-04。

## 47. AC-058 实时流游标：ID 洞≠丢帧、库重置重订阅（2026-10-13，生产未验收）

- 重要发现：`/detections/live` 原以 `first_id > after_id+1` 判 gap，但查询已按产线类型/品类/工单过滤，**被过滤的他类记录也被当作“丢帧”**；且 `reset_required=gap` 而 UI 从不重置 `_last_id`，库重建后游标大于全表最大 ID 时将永远拉不到新数据。
- 修复（后端）：gap 区间内用全表 `count(id)` 扣除存在但被过滤的记录（新字段 `filtered_out`），只有全表缺失 ID（删除/未落库）计 `dropped_count`；`reset_required` 改为“无新数据且游标 > 全表 max(id)”，此时 `next_after_id=0`。额外开销：仅出现 ID 洞时 1 次主键区间 count，空批次时 1 次 max(id)，均为索引查询。
- 修复（UI monitor_page）：reset_required → `_last_id=0` 重订阅；gap 文案改为“检测记录缺失 N 条（删除或未落库，非相机丢帧）”，KPI“丢帧”改“缺失”。
- 新增 `tests/smoke/test_live_cursor_gap.py` 3 项（过滤洞不计 gap / 删除计 gap / 游标越界要求重置、追平不重置）**3 OK**；tests/ui **15 OK**。
- 边界：工单/时间窗口统计口径、接收→落库率、backlog 未做，AC-058 仍部分实现；生产验收 **0**。待同步 SYNC-05。
- 补记：含 AC-058/063 改动的全量 smoke **282 OK / 80.95s**。

## 48. AC-064 区域复核争议/噪声隔离（2026-10-13，生产未验收）

- 重要发现：原 `apply_op` 后写覆盖前写，乙可无理由把甲已确认的框直接改为 rejected（或反之），冲突标注无痕进入/退出训练，属“噪声标注未隔离”。
- 修复（`algo/decision/localization.py`）：新增 `dispute` 操作；**他人无理由推翻**已 confirmed/rejected 结论时自动转 `status=disputed, confirmed=False`，记录 `dispute={prev_status, prev_operator, requested}`；disputed 区域仲裁（confirm/reject/revise）必须填 `reason`，否则 422。同一操作人改判、或带理由的推翻照常生效。`trainable_region_boxes` 只取 confirmed，disputed 自然隔离。API `RegionOp.op` 增加 `dispute`。
- 测试：`test_localization` 新增 `test_dispute_isolation`，**5 tests / 3.27s OK**。
- 边界：UI 多框复核面板未接；图级/框级标签分离仍靠 Feedback.region 与 region_review 两套字段，未做统一报表；AC-064 仍部分实现。待同步 SYNC-06。

## 49. AC-058 统计口径（窗口/分母/e2e）（2026-10-13，生产未验收）

- 重要发现：`api_stats_overview` 原已计算 e2e 延迟但未返回，且缺 `latency_e2e_ms` 时会**用推理耗时冒充 e2e**；返回中也没有时间窗口起点、分母含义、工单过滤与来源构成，无法区分“已落库检测记录”与“相机帧”。
- 修复（`backend/api/routes_stats.py` 的 `scope_stats`，只增字段）：`latency_basis="inference_only"`（旧 `avg_latency_ms` 口径声明）；`avg_latency_e2e_ms`/`p95_latency_e2e_ms` 只统计真实写入的 `latency_e2e_ms`，缺失计入 `n_e2e_missing`；`window_start`（range=all 时 None）；`denominator="persisted_detections"`；`workorder_id`、`excludes_archived_workorders`；`by_source` 按 `target_type` 计数（注意 `Detection.target_type` 模型默认 `"image"`）；`receipt_rate=None` + `receipt_rate_availability="unavailable"`（回执率暂无数据源，不补 0）。旧字段 `avg_latency_ms` 等不变，UI 不受影响。
- 测试：新增 `tests/smoke/test_stats_window_basis.py` 2 项 **2 OK / 0.32s**；与 `tests/ui/test_missing_metric_charts` 回归合跑 **9 tests OK**。
- 边界：接收→落库率、backlog 仍无数据源；UI 尚未展示新口径字段；AC-058 仍部分实现，生产验收 **0**。待同步 SYNC-07。

## 50. 稳定版第 2 轮同步验收（SYNC-02b～07，2026-10-13，本对话）

### 50.1 范围与方法
- 批次：SYNC-02b（UI 缺失值收尾/夹具修复）、SYNC-03（候选审计日志）、SYNC-04（定位容器元数据）、SYNC-05（实时流游标）、SYNC-06（区域争议隔离）、SYNC-07（统计口径字段）。
- 方法：SHA-256 比对开发/稳定版 → 备份稳定版原文件（`review\sync_backup\SYNC-03/04/05/06/07\`）→ 复制 → 哈希复核 → 稳定版独立运行测试，不引用开发版结果。
- SYNC-02b 的 5 个文件比对时已与稳定版一致（随前轮同步），仅复验未复制。

### 50.2 稳定版验证结果
- 定向合并：candidate_audit_log + feedback_audit_extra + localization + display_order_equivalence + live_cursor_gap + candidate_release **65 tests / 24.882s / OK**。
- tests/ui discover：**15 tests / 3.975s / OK**。
- SYNC-07 追加定向：test_stats_window_basis + test_missing_metric_charts + test_live_cursor_gap **12 tests / 0.810s / OK**。
- 全量 smoke：**283 tests / 79.304s / OK**（同步 SYNC-03～06 后；SYNC-07 为只增字段的 additive 改动，由上述定向测试覆盖）。

### 50.3 结论
- SYNC-02b～07 全部同步完成且稳定版复验通过，稳定版与开发版再次一致，可作为新基线。
- 对应 AC-055/058/063/064/074 仍为部分实现，生产验收 **0**；未修改生产 DB/快照/生产指针，未提交 git。

## 51. AC-064 多框复核 UI 面板（2026-10-13，开发对话，生产未验收）

- 编号说明：§50 已由验收对话使用（稳定版第 2 轮同步验收），本节顺延为 §51。
- 重要发现：region_review API（修订链/乐观锁/幂等/争议隔离）已具备，但 UI 只有图级反馈，无框级复核入口，操作员无法逐框确认/拒绝，候选框与人工真值在界面上不可区分。
- 实现：
  - NEW `AOI_Core/ui/widgets/region_review_dialog.py`：`RegionReviewDialog` 表格列出区域（区域ID/bbox/来源/峰值/状态/操作人），状态显示 候选(不训)/已确认(可训)/已拒绝/争议(隔离)；操作员、理由输入框，确认/拒绝/标争议/刷新按钮。提交走 GET/POST `/detections/{id}/regions`，每次生成 uuid 作 `event_id`，携带 `expected_revision`；409（乐观锁或规则拒绝）显示后端 detail 并刷新修订号；未选区域或未填操作员拒绝提交。状态栏显示“共N框，可训M框，修订号R”。
  - DIFF `AOI_Core/ui/widgets/feedback_dialog.py`：新增“多框复核…”按钮，作为框级复核入口，与图级反馈分开；无 detection id 时禁用。
- 测试：NEW `AOI_Core/tests/ui/test_region_review_dialog.py`，**4 tests / 0.011s / OK**（offscreen）。FakeClient 复用真实 `localization.apply_op` 规则与修订号乐观锁；覆盖候选不训、确认后进入 trainable、他人无理由推翻→disputed 退出训练、缺操作员拒绝、修订号过期提示冲突且刷新后成功。另以 FeedbackDialog 离屏实例化验证按钮可用。
- 边界：真实桌面交互未验；按 bbox 在图上叠加框可视化未做；add/revise 的 UI 绘框未做（目前仅 API）；AC-064 仍部分实现，生产验收 **0**。待同步 SYNC-08。

## 52. AC-065 热图后处理参数版本化（2026-10-13，开发对话，生产未验收）

- 重要发现：DS 可视化热图框的 `rel_thresh=0.6`、`min_area=16` 本已可经设置 `pipeline.overlay_box_rel_thresh`/`overlay_box_min_area` 配置，但定位容器 `loc["heatmap_postproc"]` 只记录 `{rel_thresh,min_area,display_scale}`：无算法名与版本号、未声明阈值基准（逐图 min-max 相对阈值）、形态学开运算核硬编码且未记录、面积单位（显示像素）不明。同一结果无法追溯是哪套后处理产生，参数改动也不可比对。
- 实现（`AOI_Core/backend/pipeline/service.py`）：
  - 新增常量 `HEATMAP_POSTPROC_ALGO="minmax_rel_thresh+morph_open+cc_bbox/v1"` 与函数 `heatmap_postproc_meta(rel_thresh, min_area, morph_kernel, display_scale)`，返回 `algo/rel_thresh/threshold_basis="per_image_minmax"/min_area/min_area_unit="display_px"/morph_kernel/version_id/display_scale`；`version_id="hpp-"+sha1前12位`，**仅由算法参数决定，display_scale 不参与**（显示缩放不改变版本）。
  - `render_heatmap_overlay` 新增 `morph_kernel` 参数（≤1 跳过开运算）；流水线从设置 `pipeline.overlay_box_morph_kernel` 读取，默认 3，与原硬编码一致。
  - `loc["heatmap_postproc"]` 改用上述元数据（只增字段，原 rel_thresh/min_area/display_scale 保留）。
- 测试：NEW `AOI_Core/tests/smoke/test_heatmap_postproc_version.py`，**2 tests OK**。覆盖 version_id 确定性且对 rel_thresh/min_area/morph_kernel 敏感、display_scale 不改变版本；morph_kernel=3 去掉单像素噪点、=1 保留，min_area=500 过滤全部。
- 精度/耗时：默认参数下输出与改动前一致；仅多一次 sha1（微秒级），不影响单图耗时红线。
- 边界：仍为 visual_only 可视化框，不参与定位；无真实标注定位集，未按 dev 切分选出冻结参数版本；top-k/多尺度/相邻分离/小缺陷/边界裁剪等验收项未做；AC-065 仍部分实现，生产验收 **0**。待同步 SYNC-09。

## 53. AC-063 热图二值 mask 留存（2026-10-13，开发对话，生产未验收）

- 重要发现：AC-063 要求保留原始与二值 mask，但 DS 流水线只落盘 heat/overlay 两张 JPEG（伪彩可视化，有损），开运算后的二值 mask 用完即弃，`heatmap_postproc` 也未记录 min-max 归一化基准，热图框的二值结果无法事后复现或复核。
- 实现：
  - `AOI_Core/backend/pipeline/service.py`：`render_heatmap_overlay` 新增 `return_mask=False`，为 True 时额外返回形态学开运算后的二值 mask（显示分辨率，0/255）；流水线在设置 `pipeline.overlay_save_mask`（默认 True）时将 mask 以无损 PNG `{stem}_mask.png` 写入 heatmaps 目录，与 heat/overlay JPEG 一起提交 `_io_exec` 并行写盘（返回前 join，不增加串行耗时）。`loc["heatmap_postproc"]` 新增 `raw_range=[热图原始 min,max]`（min-max 归一化基准，可复现阈值）与 `mask_path`；`out` 也带 `mask_path`。
  - `AOI_Core/backend/api/routes_data.py` `_referenced_paths`：`overlay_path` 以 `_overlay.jpg` 结尾时，同前缀 `_mask.png` 计为被引用，避免孤儿扫描误删 mask。未加 DB 列，mask 路径存于 `n_tiles.localization.heatmap_postproc.mask_path`。
- 测试：`AOI_Core/tests/smoke/test_heatmap_postproc_version.py` 2→**4 tests OK**，新增 return_mask 输出二值且开运算去噪、与热图框一致；孤儿扫描保留 mask 兄弟文件。与 `test_localization` 同跑 **9 tests OK**。
- 精度/耗时：默认参数下框与判定不变；PNG 二值编码约 1~3ms 量级（估算，未实测），在 IO 线程并行，不进串行路径。
- 设计取舍：原始浮点热图未另存——JPEG 伪彩仅作可视化，`raw_range`+二值 mask 足以复现二值结果；若后续需要重选阈值再评估是否存 float16/npz。
- 边界：mask 未在 `security.py` 文件下载白名单放行（目前无 API 下发 mask，后续 UI 展示需补）；human_box 来源与确认/训练状态字段仍缺；AC-063 仍部分实现，生产验收 **0**。待同步 SYNC-10。

## 54. AC-063 human_box 来源与 region 确认/训练状态（2026-10-13，开发对话，生产未验收）

- 重要发现：region_review 中人工 confirm 的引擎框仍显示 `source=engine_box`，无法与未复核候选区分；revise 虽改 source 但查询无 reviewer/修订版本；仅 `trainable_region_boxes` 内部按 `status=="confirmed"` 过滤，没有逐 region 可查的训练状态；`status=confirmed` 但 `confirmed` 布尔不一致的脏数据也会被当作可训练。
- 实现（只读派生视图，不改存储，不加 DB 列，兼容存量 `n_tiles.region_review`）：
  - `AOI_Core/algo/decision/localization.py`：新增 `train_status(r)`（枚举 `eligible`/`excluded_candidate`/`excluded_rejected`/`excluded_disputed`/`excluded_visual`；仅 `status=="confirmed"` 且 `confirmed is True` 为 eligible，未知/不一致状态归 excluded_candidate）与 `annotate_review(review)`：每 region 增加 `origin_source`（原始来源，revise 取 revised_from.source）、`source`（人工 add/revise 几何或已确认 → `human_box`，否则保持原候选来源）、`reviewer`（最近复核人，未复核 None）、`review_revision`（最近一次涉及该框事件的修订号，按 events 下标推导）、`confirmed`、`train_status`。`trainable_region_boxes` 改为按 `train_status=="eligible"` 过滤。
  - `AOI_Core/backend/api/region_review.py`：GET 与 POST（含幂等重放）返回的 review 经 `annotate_review`；存储结构不变。
- 训练样本选择核查：框级训练唯一门控为 `trainable_region_boxes`（仅 eligible）；反馈学习的框来自操作员手画的 `Feedback.region`（`box_source="human_box"`），region_review 候选框不直接进训练；候选构建记 `candidate_guard.region_revision`，复核修订后旧候选 fail-closed（既有 test_candidate_release 覆盖）。
- 测试：NEW `AOI_Core/tests/smoke/test_region_human_box_status.py` **3 tests OK**（候选 vs 人工真值字段、disputed/不一致/visual_only 排除、API 临时 SQLite 返回契约 + 幂等 + 存储不变）。定向 new+test_localization+test_candidate_release **55 tests / 22.479s OK**；tests.ui.test_region_review_dialog+新测试 **7 tests / 0.334s OK**（offscreen）；全量 smoke **292 tests / 85.450s OK**。
- 精度/耗时：纯字典派生，O(regions+events)，仅查询路径，不进推理串行路径。
- 边界：无 `used`（已被训练消费）状态——region 确认框目前不直接喂训练，消费记录在 Feedback 级无 region 级回写；`schemas.to_dict` 透出的 region_review 仍为原始结构；UI 对话框未显示 train_status/reviewer 列；与 Detection 模型特征仅经 detection_id/n_tiles 间接关联；原始浮点热图未存。AC-063 仍部分实现，生产验收 **0**。待同步 SYNC-11。

## 55. AC-064 原图叠加框 + 绘框 add/revise；AC-063 to_dict/表格列收口（2026-10-13，开发对话，生产未验收）

- 重要发现：§51 对话框请求路径为 `/detections/{id}/regions`，缺 `/api` 前缀（后端 `include_router(prefix="/api")`，`ApiClient._request` 直接拼 base_url），接真实客户端会 404；旧 FakeClient 忽略 path 未暴露。已改为 `api_path()` 返回 `/api/detections/{id}/regions` 并加断言。另：后端 add 不强制 reason，UI 层对 add/revise 一律强制理由（几何修改即人工真值断言）；`ApiClient` 仅 409 在 conflict_ok 下回传 `_status`，422 等返回 None 走全局错误提示。
- 实现：
  - `AOI_Core/backend/api/schemas.py`：`to_dict` 中 `region_review` 为 dict 时经 `annotate_review` 透出（与 GET /regions 同一派生视图，存储不变；None 仍为 None）。
  - `AOI_Core/ui/widgets/region_review_dialog.py`：表格新增“训练状态/复核人”两列（原列索引不变）；左侧 `RegionCanvas(ImageViewer)` 显示原图并叠加各 region 框：可训=绿实线、候选=橙虚线、人工未训（如争议中的人工框）=蓝实线、拒绝=灰点线、争议=紫点划线、仅可视=浅蓝点线；选中行线宽 4 置顶，图上单击框反选表格行。左键拖拽画框 → “待提交框”，点“提交为新增框”发 add，先选中已有框再拖框点“替换选中框”发 revise；均带 operator/reason/`expected_revision`/uuid `event_id`，409 刷新修订号并提示。纯函数：`scene_rect_to_image`（显示→原图 xyxy，按 coord_space.size 缩放、反向拖拽归一、裁剪到图内）、`image_to_scene_rect`、`hit_region`、`build_box_payload`（校验 operator/reason/region_id/revision/最小边长 2px/越界）、`box_style`。
  - `AOI_Core/ui/widgets/feedback_dialog.py`：打开多框复核时传入原图 URL（非叠加图，避免框重影与尺寸不一致）。
- 设计取舍（最小可用版）：重画=选中后整框重拖替换，不做角点/边手柄拖拽微调；拖拽框需显式点按钮提交，避免误拖直接写入真值。
- 测试：NEW `AOI_Core/tests/smoke/test_region_review_draw.py` **7 tests OK**（坐标缩放/裁剪/往返、add/revise payload 与校验、颜色映射、to_dict 新字段与存储不变、对话框叠加框数/位置/高亮线宽/新列/API 路径、拖框 add→revise→乐观锁冲突刷新后成功）。定向 新测试+tests.ui.test_region_review_dialog+test_localization+test_region_human_box_status **19 tests / 3.736s OK**；全量 smoke **299 tests / 93.057s OK**（offscreen；日志中 `RuntimeError: db down` 为既有故障注入用例的预期输出）。
- 边界：离屏 smoke，实机待 TEST-04 类实测（真实鼠标拖拽、缩放后 fitInView 精度、大图性能未验）；无手柄微调、无删除/撤销框；mask 未下发 UI；`used` 消费状态仍缺。AC-063/AC-064 仍部分实现，生产验收 **0**。待同步 SYNC-12。

## 56. 稳定版第 3 轮同步验收（SYNC-08～10，2026-10-13，本对话）

- 合并同步 6 文件：SYNC-08（多框复核对话框）、SYNC-09（热图后处理版本化）、SYNC-10（二值 mask 留存）；备份于 `review\sync_backup\SYNC-08/09/10\`，复制后哈希逐一一致。
- 稳定版定向：heatmap_postproc_version+localization+display_order **11 tests / 0.861s / OK**；tests/ui **19 tests / 4.127s / OK**。
- 稳定版全量 smoke：**289 tests / 85.431s / OK**。
- 注意：新终端 `python` 9009，按既有约定改用 `py -3.11`。
- AC-063/064/065 仍部分实现，生产验收 **0**。

## 57. TEST-01 原生 2500² 端到端耗时实测（AC-066，本对话，FAIL）

### 57.1 重要发现：稳定版生产快照无法加载
- 稳定版 `storage/engine/snapshots/gold_finger/v2`（current）被当前代码拒绝加载：`algo/persist.py:_restore_slot_state` 抛 `ValueError: layout 快照缺少必要拟合统计（count_median/count_std/area_hi/spacing_median/spacing_std/sym_median/sym_std）`，要求用原训练数据重新 prepare。
- **含义**：当前稳定版直接启动后 gold_finger 检测不可用，生产可用性被阻断；本次测量模型为隔离 TEMP 内 40 张 train 域正常图（seed=123）重 prepare 的同架构模型，耗时代表性成立，精度结论不适用。
- guard_bundle 红线验证：归档路径（非 train 域）被拒绝作为 init_normal，红线工作正常。

### 57.2 实测结果（GTX 1660 SUPER，隔离 TEMP，2500×2500 原生不缩放，40 张 seed=20261013）
- 冷启动：direct 458.3ms / http 467.6ms。
- 热态空闲（n=120/入口）：direct wall P95 **295.8ms**，http wall P95 **274.7ms**（含 HTTP 传输排队），e2e P95 288.4/258.7ms，infer P95 134.7/99.6ms。
- 后台学习负载（submit_feedback×2，learning_parallel=False）：重叠请求 wall 约 1026～1847ms，前台被学习明显阻塞（n=2/入口，仅作现象记录）。
- **判定：FAIL**——G1 热态 P95 ≤200ms 不满足，最优口径（http e2e 258.7ms）仍超标约 29%。
- 产物：`review\task\test01_e2e_latency\`（result.md/env.json/cmd.txt/raw 逐图 CSV）。边界：非生产目标硬件；OS 页缓存未清。

## 58. TEST-03 显存场景采样（AC-070，本对话，PASS，OOM 子项未做）

- 整卡 6,144 MiB；场景峰值（整卡 MiB）：boot 516 / train_prepare 913 / idle 691 / infer_single 691 / 第二品类 prepare 919 / 双品类驻留 691 / 双进程并行推理 926 / replay_reload 693。
- torch 进程口径峰值：训练 276～290 MiB、推理 127 MiB、空闲 105 MiB；**特征驻留 CPU**，推理显存增量极小。
- 上限草案（待用户定值）：单进程 ≤1,024 MiB；整卡告警 3,000 MiB、拒绝 4,500 MiB；后台任务超拒绝线 deferred，前台仅降级不拒绝。
- 边界：人为 OOM 破坏性复现未做（需专用环境）；采样间隔约 100ms。
- 产物：`review\task\test03_vram\`。AC-070 仍部分实现（上限数值未冻结），生产验收 **0**。

## 59. TEST-08 业务预算与保留期草案（AC-072，本对话，待用户定值）

- 草案：G2 FPR ≤1.0%、Recall ≥99.0%、G6 旧域召回回退 ≤0.5pt/版本、灰区 ≤10%；原图热存 90 天+冷归档 12 个月、热图 30 天、反馈/事件/WAL 永久封存、快照每品类留 10 版+全部曾激活版本、标签迟到 >7 天只统计不入训。
- 依据：稳定版实测存储（归档 780 张/4.37GB、快照 27.8MiB、DB 13.5MiB），增长估算 1,000 张/天时 90 天热存约 500GB。
- 产物：`review\task\test08_business_budget\`。数值未经用户确认前不冻结，AC-072 生产验收 **0**。

## 60. 稳定版第 4 轮同步验收（SYNC-11～12，2026-10-13，本对话）

- 合并同步 7 文件（human_box/train_status 派生视图、to_dict 透出、复核对话框叠加框/绘框、2 个新测试文件）；备份于 `review\sync_backup\SYNC-11/12\`，哈希逐一一致。
- 稳定版定向：region_review_draw+region_human_box_status+localization **15 tests / 3.809s / OK**；tests/ui **19 tests / 4.117s / OK**。
- 稳定版全量 smoke：**299 tests / 83.382s / OK**，与开发版 299 一致。
- AC-063/064 仍部分实现，生产验收 **0**。未修改生产 DB/快照，未提交 git。

## 62. TEST-02 同 GPU 推理+学习并发实测（AC-069，本对话，FAIL）

### 62.1 环境与方法
- 隔离 TEMP（storage/engine/SQLite），prepare 两品类 gold_finger/gold_finger_b（40/30 张 train 域正常图，seed=777；生产快照因缺 layout 统计不可用，见 §57）。
- 前台并发 1/2/4（ThreadPoolExecutor），两品类交替请求；负载 idle 与 learning（后台持续 submit_feedback，learning_parallel=False 生产默认）。
- 逐请求记录 t_submit/t_lock_acquired/t_done/queue/wall/infer/e2e；学习事件时间线单独记录（26 事件，单事件 700~980ms）。

### 62.2 结果（GTX 1660 SUPER 6GB）
- c1 idle P95 263ms → learning P95 **1505ms**；c2 290→**820ms**；c4 idle **406ms** → learning **960ms**。
- queue≈0：阻塞在引擎内部锁/让行段，不在提交队列；与 V§23（重叠 P95 约 3.4s）同向，本次负载更轻约 0.8~1.5s。
- 取消/限流是否生效：未覆盖学习任务取消接口；限流未见生效证据。

### 62.3 判定与产物
- **FAIL**：通过条件「前台推理 P95 仍满足 G1」在所有级别不满足；学习让行机制存在但不足以免阻。
- 产物：`review\task\test02_gpu_concurrency\`（result.md/env.json/cmd.txt/raw CSV）。
- AC-069 仍部分实现，生产验收 **0**。

## 61. AC-058 UI 展示统计口径（2026-10-13，开发对话，生产未验收）

（§56～§60 已被验收对话占用，顺延 §61）

- 重要发现：工作台 KPI 只显示检测数/不良数/不良率，V§49 后端 scope_stats 新口径字段（窗口起点/分母/工单/by_source/e2e/receipt_rate）已返回但 UI 未使用；旧后端空范围时 `avg_latency_ms` 返回 0.0，若直接展示即“0 ms 冒充真实值”。统计页（stats_page）走另一接口，本次未改。
- 实现：`AOI_Core/ui/pages/dashboard_page.py`
  - 纯函数 `scope_caliber_texts(scope)`：推理耗时（均值，副文本“仅模型推理（不含采集/传输/落库）”，无记录或无 basis 时 “—”/“口径未声明”）；端到端耗时“均值 / P95”（avg 为 None → “不可用”，副文本“{n_e2e_missing} / {n} 条缺 e2e 记录未计入”）；接收→落库率（receipt_rate=None → “不可用”，“暂无相机接收计数数据源，不以 0 代替”）；口径行“窗口 近 7 天（起点 …）/全部（无起点）｜分母 已落库检测记录（非相机帧）｜工单 #id / 全局（已排除存档工单）｜来源 产线 n、上传 n…”。缺字段一律 “—”/“未声明”。
  - 工作台 KPI 下新增一行三张卡（推理耗时/端到端耗时/接收→落库率）+ 口径说明 QLabel，均挂 `CALIBER_TIP` tooltip（说明窗口非“最近 N 条”、分母不等于相机采集帧数、不可用不显示 0）。不设阈值、不着色（无业务阈值）。
- 测试：NEW `AOI_Core/tests/smoke/test_ui_stats_caliber.py` **5 tests OK**（全字段文案、缺字段不出现 “0 ms”/“0.0%”、控件与 tooltip、旧后端载荷不崩、临时 SQLite 真实 `api_stats_overview` → `_fill_kpi` 含 e2e 全缺 “不可用”与 “4 / 4”）。定向 新测试+test_stats_window_basis **7 tests / 0.630s OK**；tests/ui **19 tests / 3.915s OK**；全量 smoke **304 tests / 84.246s OK**（offscreen）。
- 边界：离屏 smoke，实机待测（真实桌面布局/字体/tooltip 未验）；接收→落库率与 backlog 仍无数据源；stats_page 未展示新口径。AC-058 仍部分实现，生产验收 **0**。待同步 SYNC-13。

## 63. FIX-SNAP 旧快照 layout 统计缺失兼容（2026-10-13，开发对话，生产未验收）

- 重要发现（生产阻断）：稳定版生产快照 gold_finger v2 保存于 layout 统计持久化修复（2026-10-11）之前，`slot_state.pkl` 中 layout 只有旧 PCA 键且全为 None；2026-10-12 新增的 `_restore_slot_state` 严格校验（防止静默用默认常数打分）因此抛 `ValueError: layout 快照缺少必要拟合统计`，导致整个快照拒载、gold_finger 检测不可用。LayoutSlot 的 7 项统计（count_median/count_std/area_hi/spacing_median/spacing_std/sym_median/sym_std）只能由 fit 从训练 tile 得到。
- 方案选择：
  - a（recomputed_on_load）不可行、未实现：快照不含训练图或特征；用标签未知的归档图重拟合会把缺陷混入正常统计，属禁止做法。
  - b（显式降级）采用：仅降级缺统计的槽位，其余槽位照常工作；不回退默认常数，原保护语义保留。
- 实现：
  - `AOI_Core/algo/persist.py`：新增 `LAYOUT_REQUIRED_STATS`、`LayoutStatsMissing(ValueError)`；`_restore_slot_state` 缺统计时抛该异常（直接调用仍拒绝）。`load_snapshot` 非 candidate 模式捕获后登记 `pipe.degraded_slots[name]`（layout_degraded=True、layout_stats_source="missing_in_snapshot"、missing、reason、action）并 `warnings.warn("快照降级加载：…")`；该槽位移出 `_score_slots`、权重置 0、其余按比例归一（记录 weights_before/weights_after）；全部槽位降级则 raise ValueError；candidate 模式仍抛错（降级 pipe 不能作为候选上线）。
  - `AOI_Core/algo/eval/offline.py`：decision_trace fusion 阶段带 degraded_slots；`_recalibrate_on_reflow` 跳过降级槽位。`_prefetch_cpu`/`_record` 原已只遍历 `_score_slots`。
  - 未改 `backend/pipeline/service.py`。
- 测试：NEW `AOI_Core/tests/smoke/test_snapshot_layout_compat.py` **4 tests / 0.63s OK**（完整快照加载后打分逐值不变；旧快照有/无 artifact_hashes 均显式降级——标注、警告、权重、无统计属性、prefetch/slot_scores 不含 layout、fused=blob、trace 带标注；直接 `_restore_slot_state` 与 candidate 加载仍拒绝；全部槽位降级拒绝）。全量 smoke **308 tests / 83.5s OK**。
- 真实快照烟测（开发代码，v2 复制到 TEMP，生产快照未动，副本已清理）：加载 1.0s 成功；slots blob/trad/layout → score_slots blob/trad；layout 7 项统计全缺；权重 0.4528/0.3906/0.1566 → 0.5369/0.4631/0。5 图推理 0.15～0.55s/张：det1_bench_00、det12_bench_00、det14_bench_02、det114_1312 fused=2.0 判 anomaly（缺陷库拦截），det118_0599_repaired fused=0.6781 判 normal；trace 均带 degraded=['layout']。
- 边界：
  - tau_high/tau_gray 是在含 layout 的融合上校准的，降级后存在阈值偏移风险，未做精度评测；正式恢复须重新 prepare。
  - router 维度不匹配时自动不参与（原本只用于追溯）。
  - 基于 v2 的反馈学习候选保存后，candidate 加载仍会严格拒绝（预期保护），须重新 prepare 后再学习。
  - 稳定版未复验，待同步 SYNC-14（生产阻断修复，优先）。生产验收 **0**。

## 64. OPT-DECODE 单图检测 decode 段降耗（2026-10-13，开发对话，生产未验收）

- 重要发现：
  - TEST-01 的测试图全部是 2500² 8 位 RGB PNG。旧 `_decode_rgb` 只让 JPEG/BMP 走 cv2，所以 PNG 全部走 PIL 的 `np.array(convert("RGB"))`，约 110ms/张。
  - 子步拆分（6 图×5 次，median/p95 ms）：PIL 全解码 109.0/118.8；cv2 imdecode BGR 92.1/98.6，加 cvtColor 5.9/8.4；`IMREAD_COLOR_RGB` 直出 89.0/94.4；np.fromfile 5.0；PIL 读头 0.08。
  - 瓶颈是 zlib inflate，换库也只能省约 20ms/张。
  - TEST-01 的 decode_ms 是 total 减 latency，除 `_decode_rgb` 外，还含 _ensure_loaded/_foreground/锁/_to_result 的开销。
  - 引擎全分辨率消费输入（_preprocess→compute_tiles→extract_tiles），因此 IMREAD_REDUCED 会改变分数，不可用。
  - decode 前面没有可重叠的阶段。
- 实现：`AOI_Core/backend/pipeline/service.py::_decode_rgb`（未改 engine 锁和学习逻辑，未改 `detect_frame`）。
  - PIL 只读头，然后用 `_cv2_decodable` 判定是否安全：
    - JPEG/BMP/PNG，模式为 RGB/L。
    - PNG 还要求单帧、无 transparency、单 tile、rawmode 为 RGB/L。这样可以排除 16 位、tRNS 和 APNG。
  - 安全时走 cv2，flag 为 `IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR_RGB`（旧 cv2 无此 flag 时用 IMREAD_COLOR 加 cvtColor），直接得到 RGB。
  - cv2 返回 None，或尺寸、通道不符时，回退原 PIL 路径。
- 测试：NEW `AOI_Core/tests/smoke/test_decode_fastpath.py`，**23 项**。用 spy 断言实际走的路径，与 PIL `convert("RGB")` 的 maxdiff 为 0，且结果 C 连续。
  - 快路径：PNG RGB8/压缩级 9/gray8/optimize/gAMA/EXIF 方向，JPEG RGB/gray/EXIF 方向，BMP，中文路径。
  - 回退：RGBA/P/RGB tRNS/gray tRNS/16 位灰度/16 位 RGB/1 位/APNG/CMYK JPEG/TIFF/cv2 返回 None/尺寸不符。
  - 与 test_g1_decode_equivalence 合跑 30 OK。全量 smoke **331 tests / 86.2s OK**（308 + 23，offscreen）。
- 端到端同进程 A/B（TEMP 隔离 config/storage/SQLite）：从稳定版归档复制 14 张正常图到 TEMP，prepare（seed=123）；6 张 TEST-01 去重图，4 轮交替，n=24，单位 ms。
  - decode p50/p95：103.9/116.5 → 94.8/105.3。
  - infer：58.8/69.4 → 58.6/68.6。
  - **e2e：162.9/205.7 → 154.7/190.8**。
  - wall p95：229.9 → 215.6。
  - 精度：6 图的 pixel/score/decision/slots/raw/boxes/heatmap 差异全为 0，旧实现自重复差异也为 0，24 组 A/B 决策和分数逐一相同。
  - 产物：`review\task\test01_e2e_latency\dev_opt\`（result.md、bench_decode.py、bench_e2e_ab.py、decode_breakdown_before/after.json、e2e_ab.json、smoke_full.log）。
- 边界：
  - A/B 用 TEMP 新 prepare 的模型，不是 TEST-01 的稳定版环境和快照（TEST-01 为 P95 direct 295.8/http 274.7ms，推理段更重）。
  - decode 只省 10～20ms/张，按此推算，TEST-01 口径下 P95 仍可能超过 200ms，需要验收对话在稳定版复测。
  - 剩余缺口要靠产线格式解决：JPEG/BMP 或原始帧经 `detect_frame` 直送。turbojpeg/pyspng 未安装，按约束不引入。
  - 待同步 SYNC-15。AC-066 仍未通过，生产验收 **0**。

## 65. 稳定版第 5 轮同步验收（SYNC-13～15）与 TEST-01 复测（2026-10-13，本对话）

### 65.1 同步与验证
- SYNC-13（工作台口径卡）：dashboard_page.py + test_ui_stats_caliber.py；定向 7 tests / 0.532s / OK，tests/ui 19 OK。
- SYNC-14（FIX-SNAP 生产阻断修复）：persist.py/offline.py + test_snapshot_layout_compat.py；定向 34 tests OK。
- SYNC-15（OPT-DECODE）：service.py `_decode_rgb` cv2 直出 RGB + test_decode_fastpath.py（23 项）；同上 34 tests OK。
- 稳定版全量 smoke（SYNC-14/15 后）：**331 tests / 89.173s / OK**，与开发版 331 一致。

### 65.2 FIX-SNAP 真实快照复验（重要）
- 稳定版生产快照 gold_finger v2 TEMP 副本：**加载成功 0.86s**（此前整体拒载，生产可用性阻断已解除），`degraded_slots` 记录正确（layout_degraded、missing 7 项、权重 0.4528/0.3906/0.1566 → 0.5369/0.4631/0），5 图推理 396～619ms。
- ⚠️ **新发现（转开发排查）**：
  1. 降级模式 slot 分数对不同图恒定——正常图 blob 0.5352/trad 0.8438 完全一致，bench 缺陷图 blob/trad 均 2.0 钳顶；可能是旧快照本身粗糙（v2 训练样本少），也可能降级路径异常，需开发定位。
  2. service 响应 `decision_trace` 为空列表（trace_len=0），SYNC-14 宣称的“fusion 带 degraded_slots”未出现在服务响应中（pipe.degraded_slots 存在）。
- 附证：`review\task\test01_e2e_latency\raw\fixsnap_real_v2.json`。

### 65.3 TEST-01 同口径复测（SYNC-15 后，同 40 图同种子 seed=20261013）
- direct：wall P95 295.8→**235.3ms**，e2e 288.4→**216.3ms**，infer 134.7→89.8ms。
- http：e2e 258.7→245.3ms；wall 274.7→283.4ms（HTTP 噪声）。
- **判定仍 FAIL**（最优口径 216.3ms > 200ms）；decode 优化有效，剩余瓶颈为 PNG zlib inflate 与 render/persist。详见 `review\task\test01_e2e_latency\result.md` 复测节。
- AC-066 仍 FAIL，生产验收 **0**。

## 66. AC-069 学习让行与锁等待可观测（2026-10-13，开发对话，生产未验收）

### 66.1 设计（`AOI_Core/backend/engine/__init__.py`，`learning_parallel` 默认仍为 False）

- 前台标记：`_foreground()` 用 Condition 维护在途前台数 `_fg_active` 和最近结束时刻 `_fg_last`。
- 合作式让行 `_learning_yield(cancel_event, max_wait_s)`：前台在途，或前台结束后 `learning_idle_ms=300` 内，学习都会等待。
  - 有 `max_wait_s` 上界。
  - `cancel_event` 可打断（检查周期 ≤0.5s）。
  - 当前线程持有品类锁或 `_write_lock` 时直接返回，避免锁序死锁（锁序为品类锁 → `_write_lock`）。
- 即学 `submit_feedback._learn` 的流程：
  1. 有上界让行（`feedback_yield_max_ms=1500`，小于 join 的 3s）。
  2. 锁外预提取 `pipe.build_item`：读图、切块、DINO 前向，不读写缓存和任何状态。
  3. 锁内校验 pipe、backbone、`_pre_cfg` 三者身份都一致时才 `adopt_item`，否则锁内现算。
  4. 锁内做打分、状态更新和 WAL/回执写入。
- `algo/eval/offline.py` 把 `_get_item` 拆成 `build_item`（无状态）和 `adopt_item`（FIFO 上限）。
- 可观测：`_timed_lock` 记录 lock_wait_ms/lock_hold_ms，并累计到 `lock_stats()`。
  - 即学回执 `info["lock"]` 中有 yield_ms/prefetch_ms/lock_wait_ms/lock_hold_ms/prefetched。
  - 推理 `extra["lock_wait_ms"]` 记录真实等品类锁时间，service.py 将其透出到响应。
- 正确性核查：
  - 推理和学习更新都在同一品类锁内完成，不会读到半更新状态（test_no_torn_read 覆盖）。
  - FrozenDINO 是无状态 no_grad 前向，可在锁外并发。
  - pipe 被替换时丢弃预提取结果（test_pipe_swapped_discards_prefetch 覆盖）。

### 66.2 本轮核查发现与修复

- **Bug**：`consolidate` 候选路径（activate=False）在锁外重装载和 replay 前，两次 `_learning_yield()` 都没有上界。consolidate 是 API/工单同步调用，持续前台负载下 `_fg_last` 不断刷新，会被无限饿死（接口挂起）。
  - 修复：两次调用都加上 `max_wait_s=feedback_yield_max_ms/1000`。
  - 补测：`tests/smoke/test_learning_yield.py::test_consolidate_yield_bounded_under_continuous_foreground`。前台常驻、上界 100ms 时，consolidate 在 2s 内返回新版本号。
- 已知取舍（未改）：候选训练 `CandidateBudget.check` 的让行无上界。但它可以取消，让行时长计入 `_paused`、从墙钟预算中扣除，所以不会超预算误杀。代价是持续前台负载下候选训练会无限推迟。

### 66.3 测量（≤20 张真实图，TEMP 快照副本，产物 `review\task\test02_gpu_concurrency\dev_opt\`）

- 口径：开发版引擎层 `predict_image_path`（无 persist/heatmap），gold_finger v1 快照的 TEMP 副本，engine_fast.yaml，cuda。
  - 前台单线程零间隔闭环，30 次请求。
  - 1 个学习线程串行提交 10 张反馈。
  - learn_before 用“yield 上界=0 + 锁外 build_item 抛错回落锁内”来模拟改动前。
- 前台 P50/P95/max（ms）：
  - idle 86.8/183.0/298.9。
  - **learn_before 101.0/1359.3/1672.5**：lock_wait P95 898.5、max 1401.1，30 次中有 10 次等锁；学习 hold P50/max 626.8/1501.2。
  - **learn_after 93.6/254.8/436.3**：lock_wait P95 为 0、max 385.5，仅 1 次等锁；yield P50/max 1281.5/1500.0，hold P50/max 413.5/441.6。
  - learn_after_cached 94.9/127.2/819.3：lock_wait max 538.9，1 次等锁；hold 461.5/538.8。
- 与 TEST-02（V§62）的口径差异：TEST-02 是稳定版 service.detect_image（含 persist/heatmap），TEMP 新 prepare 两个品类，c1/c2/c4 并发。
  - 原值：c1 learning P95 1505ms，c4 idle P95 406ms。
  - 旧 `t_lock_acquired` 实际取的是线程开始时间，因此 queue≈0 是假象，等锁时间被算进了“推理”。
  - 本测量的是引擎内真实的 lock_wait，绝对值不能直接对比。learn_before 1359ms 与 TEST-02 c1 learning 1505ms 量级一致，说明阻塞来源确实是锁内学习。

### 66.4 风险与重要发现

- **学习吞吐被限速**：前台零间隔持续负载时，学习每次都会等满 1.5s 上界。30 次请求期间只完成 2 次学习（改动前为 10 次）。上界保证学习不会饿死，但在高负载产线上，即学会明显滞后。
- **锁内残余 hold 约 0.4～0.54s**（cached 档无需提特征也是如此）：打分、状态更新和 WAL/回执写入仍持锁。前台碰上时单次 max 为 436～819ms，这是下一步压缩的方向。
- 锁外 DINO 前向与前台推理争用同一 GPU：learn_after 的 P95 比 cached 高约 130ms，这部分体现在前台推理变慢上，不体现在 lock_wait 上。
- 未覆盖：多前台请求之间的品类锁串行（TEST-02 c4 idle P95 406ms 的来源）；consolidate activate=True 整段持锁；单步前向不可中断。

### 66.5 验证

- 定向：test_learning_yield（7 项，含本轮新增 1 项）、test_candidate_isolation、test_isolated_candidate_path，共 **36 tests OK**（19.5s）。
- 全量 smoke（offscreen）：**338 tests / 89.926s / OK**（331 + AC-069 的 7 项），日志为 `dev_opt\smoke_full.log`。
- 待同步 SYNC-16。待稳定版按 TEST-02 口径（service 层、两品类、c1/c2/c4）复测。AC-069 生产验收 **0**。

## 67. AC-058 统计页接入统计口径（2026-10-13，开发对话，生产未验收）

### 67.1 重要发现（`AOI_Core/ui/pages/stats_page.py` 原实现）

- “平均延迟”卡取自 `quality_overview.avg_latency_ms`。该值只计推理耗时，却被当作总体延迟展示；空范围时后端返回 0.0，界面显示“0 ms”。
- 批次表 `avg_latency_e2e_ms` 由后端 `float(avg or 0.0)` 生成，无 e2e 时返回 0.0；UI 再 `or 0`，最终显示“0 ms”。表中也没有单独的推理耗时列。
- 覆盖率、误报率、漏报率在分母为 0 时，后端返回 0.0，界面显示“0.0%”。健康卡在 scanned=0 时同样显示 0.0%。
- 导出 CSV 用 0 填缺失值。
- 各项都没有标明分母：检测数不是相机帧数；健康卡统计的是近 7 天最新至多 3000 条记录。
- 延迟趋势图来自 timeseries，用 coalesce 把 e2e 回退成推理耗时，且按日聚合，无法按工单拆分。后端不在本次改动范围内，只在 UI 侧标注。

### 67.2 实现（只改 UI；未改 backend/service/engine）

- 直接 import 工作台的 `scope_caliber_texts`、`CALIBER_TIP`、`NA`（同一对象，未复制）。新增 `stats_overview(workorder_id, range="all")` 调用，取 `scope_stats` 填充两张卡：
  - “推理耗时”卡：副文本为仅推理口径加预算；缺失时不着色。
  - “端到端耗时”卡：显示均值 / P95，以及缺 e2e 条数。
- KPI 卡下方新增口径行：窗口、分母、工单、来源、接收→落库率（不可用）。卡片、批次表、口径行均设置 CALIBER_TIP tooltip。
- 质量卡：检测数副文本为“已落库检测记录（非相机帧）”。各比率副文本写明分子和分母（复核 x / n、fp / fp+tn、fn / fn+tp），分母为 0 时显示“—”。
- 批次表新增“推理耗时”和“端到端耗时”两列。值 ≤0 或缺失时显示“—”；无复核时比率也显示“—”。
- 健康卡：scanned=0 时显示“—”；有数据时副文本为 x / n 条，并在提示中说明上限 3000 条、非相机帧。
- 延迟图标题改为“推理耗时趋势（近 30 天日聚合）”，tooltip 说明为日聚合、无法按工单拆分。
- 导出 CSV：增加口径注释行；分母为 0 的比率和缺失的 e2e 留空，不写 0；批次列拆分为 infer 与 e2e 两列。

### 67.3 验证（offscreen）

- NEW `AOI_Core/tests/smoke/test_ui_stats_page_caliber.py`：**7 tests / 0.548s OK**。覆盖以下场景：
  - 复用对象一致性（assertIs）。
  - 初始“—”与 tooltip。
  - 旧后端或空载荷不出现 0 ms / 0.0%。
  - 完整 scope 下推理与 e2e 分列，口径行文本正确。
  - 批次表缺失值显示。
  - CSV 留空。
  - 临时 SQLite 下，真实 `api_stats_overview` 与 `api_stats_quality_overview` → 统计页的链路，含删光记录后的退化。
- tests/ui **19 tests / 3.889s OK**（含 test_missing_metric_charts）。全量 smoke **345 tests / 93.802s OK**（338+7）。

### 67.4 边界

- 离屏 smoke，实机布局未验。
- 后端 quality、batch、health 接口仍用 0.0 冒充缺失，目前只在 UI 侧用分母兜底。
- timeseries 的 e2e 用 coalesce 回退，没有 e2e 趋势，也不能按工单拆分。
- 接收→落库率和 backlog 仍无数据源。
- AC-058 仍为部分实现，生产验收 **0**。待同步 SYNC-17。

## 68. AC-063 原始浮点热图留存与离线阈值复算（2026-10-13，开发对话，生产未验收）

### 68.1 重要发现

- 在线框不是在模型输出网格上提取的，有三步：
  1. 引擎 fused_heatmap 位于工作网格（gold_finger 实测 512×512），先用 INTER_LINEAR 放大到展示尺寸（overlay_max_dim，yaml 中为 1280，代码默认 1920）。
  2. 在图内做 min-max 归一化到 uint8，用 `> int(255*rel_thresh)` 二值化。
  3. 开运算、外轮廓、min_area（展示像素）过滤，最后按 `int(v/scale)` 换回原图坐标。
  因此复算必须完整重放“resize 后量化”这条链，不能直接在 raw 网格上取阈值。
- 若在线用 float32 渲染、而留存的是 float16，min-max 和插值会有细微差异，阈值附近的像素会翻转。解决办法：开关打开时，在线先把热图量化为 float16（`quantize_raw_heatmap`），渲染和留存使用同一份数组，复算结果逐位一致。若存在非有限值或超出 float16 范围，回退 float32。
- 原下载白名单只放行 DB 已登记的 overlay/heatmap 路径。mask/raw 没有 DB 列，storage 迁移后会被拒。现改为按“同前缀 `_overlay.jpg` 已登记”放行。
- 基准中 det900_bench_02 两臂都不产出热图（_best_heatmap 为 None），与本改动无关，统计时已剔除。

### 68.2 实现（`backend/pipeline/service.py`、`backend/api/routes_data.py`、`backend/core/security.py`）

- 新设置 `pipeline.overlay_save_raw_heatmap`（默认 True）。开启时把 `{stem}_raw.npy`（float16，模型输出工作网格分辨率）提交给现有 `_io_exec` 写盘线程池，与 heat/overlay/mask 并行写入，返回前 join。未改 max_workers。
- `heatmap_postproc` 新增字段：raw_heatmap_path/raw_heatmap_file、raw_shape、raw_dtype、raw_resolution=model_output_work_grid、resize=INTER_LINEAR、display_size、image_size、scale_raw_to_display、scale_raw_to_image、box_to_image=int(v/display_scale)。
- `render_heatmap_overlay` 拆出两个纯函数 `_heatmap_u8` 和 `_mask_and_boxes`，在线与离线共用。新增：
  - `recompute_heatmap_mask(raw, display_wh, rel_thresh, min_area, morph_kernel, display_scale)`：返回 mask、boxes_display(xywh)、boxes_image(xyxy)。
  - `recompute_from_postproc(pp, base_dir, **覆盖参数)`：未给的参数取留存值。
  - `load_raw_heatmap(path, base_dir)`：resolve 后要求位于 base_dir 内且后缀为 .npy；使用 `allow_pickle=False`，越界时抛 PermissionError。
- 孤儿扫描：`_overlay.jpg` 同前缀的 `_raw.npy` 计为被引用。
- 下载白名单：`_mask.png`/`_raw.npy` 若同前缀 overlay 已在 DB 登记则放行。其余仍走原规则（storage 内或 DB 登记），`..` 和其他后缀仍拒。

### 68.3 测量（`review/task/ac063_raw_heatmap/bench_raw_ab.py` → `raw_ab.json`；TEMP 隔离，6 张 init_normal 加 4 张测试图，共 10 张，同进程 on/off 交替 6 轮）

- 文件大小：raw .npy **524,416 B**（512×512 float16）。同图 mask PNG 约 26.9 KB，heat JPEG 约 278 KB，overlay JPEG 约 358 KB。每条检测落盘增加约 0.5 MB，归档容量需按此规划。
- 写盘耗时：单独计时 `_save_npy`，3 张图各 5 次，p50 **0.63～1.01 ms**，p95 ≤1.45 ms。在线写盘在线程池中执行，不计入 latency_e2e_ms。
- 默认参数复算：3/3 张图 mask 与 heatmap_boxes **逐位一致**（det878/det27/det881 框数分别为 56/70/38）。
- e2e（剔除 det900，每臂 n=18）：on P50/P95 **229.1/266.1 ms**，off **222.2/264.3 ms**；wall 分别为 on 251.6/291.1、off 246.0/308.2；infer 两臂均约 84 ms。差值在轮间噪声内（第 3、4 轮出现双向 30～60 ms 抖动）。本机本轮基线 P95 已高于此前的 216 ms，两臂都没有达到 ≤200 ms 的目标，这是基线问题，不是本改动造成的；单图均低于 1 s。

### 68.4 验证

- NEW `tests/smoke/test_raw_heatmap_retention.py`：**5 tests OK**。覆盖：
  - 端到端默认复算与在线 mask/框一致，元数据字段完整；改阈值后框数变化。
  - 开关关闭时不写 raw。
  - 纯函数结果与 render 一致；非有限值回退 float32。
  - 路径穿越被拒。
  - 孤儿引用与 storage 迁移后的白名单放行/拒绝。
- 回归：全量 smoke **350 tests / 87.645s / OK**（345+5，offscreen；日志位于 `review/task/ac063_raw_heatmap/smoke.{out,err}.log`）。

### 68.5 边界

- 离线复算目前只有函数入口，没有 API/UI 入口。
- 未做 raw 留存天数或容量的独立清理策略，跟随 overlay 的孤儿/归档清理。
- 历史检测没有 raw，无法补算。
- region 级 used 回写仍缺。
- AC-063 仍为部分实现，生产验收 **0**。待同步 SYNC-18。

## 69. AC-066 下采样解码试验（2026-10-13，开发对话，结论：不采用，生产验收 0）

目标：热态 e2e P95 ≤ 200ms（TEST-01 直连约 216ms，PNG decode 约 90ms 为最大段），精度不得退化。本次只做试验，**未改任何生产代码**，没有新增设置项或 smoke，也不写 SYNC。产物位于 `review/task/test01_e2e_latency/dev_downsample/`（`bench_decode_ds.py`/`decode_bench.json`、`bench_ds_ab.py`/`ds_ab.json`/`ds_ab.log`、`result.md`、smoke 日志）。

### 69.1 分辨率链路（fast profile，engine_fast.yaml）

- DINO 为 single tile，用 `cv2.resize` INTER_LINEAR 缩到 448²。
- hierarchical `work_side=512`：热图、掩码、框在 ≤512 的网格上生成，再放大回原图（mask 用 NEAREST）。
- 对位 `_align_offset` 使用灰度 INTER_AREA 256²。
- defect_clf 的 `_img_features` 把输入限制在 ≤1024。
- 服务层 `_finish_detect`：`image_size` 取自解码数组，overlay 限制在 ≤1920。若下采样，框、image_size、overlay 都要映射回原图。
- 引擎内部已经是低分辨率计算，因此下采样的理论收益只在 decode 和 render 两段。
- 生产快照 gold_finger v2 为 cpu profile（blob/trad），这两个槽位吃全分辨率。U22 已证实 blob 在 2048 降采样时检出下降（3/20 vs 6/20），风险更高。

### 69.2 测量（热态，3 轮交替，ms，p50/p95）

- 解码微基准：
  - PNG 2500²：full 95.5/101.2，REDUCED_4 94.4/99.5，area1024 99.1。zlib inflate 不能跳过，因此**无收益**。
  - JPG 4096×3000：full 约 62～66，REDUCED_2 约 42，REDUCED_4 约 32。
  - JPG 5120²：full 277/396，REDUCED_4 213，PIL draft 154。
- e2e A/B（`service.detect_image`，persist+heatmap）：
  - PNG 2500²（TEST-01 口径）：full e2e 173.8/221.6，area1024 193.5/235.1，**更慢**。
  - JPG 4096×3000：full 144.1/156.9，area1024 132.1/150.8，reduced4 102.4/115.8（decode 56.5→29.6，render 27.6→13.7）。

### 69.3 精度证据（34 张 data_local test：good 10 + defect 24，带 GT 掩码；以 full 为参考，full 自重复 Δscore=0）

| 变体 | Δscore p50/p95/max | 框 IoU 均值 | 热图 corr | 判定翻转 | good 误报 | 检出/GT 命中 |
|---|---|---|---|---|---|---|
| full（参考） | 0 | 1 | 1 | 0 | 5/10 | 2/24 · 2 |
| area1024 | 0.159/0.480/0.564 | 0.211 | 0.914 | 4 | 6/10 | 5/24 · 3 |
| area512 | 0.170/0.382/0.572 | 0.225 | 0.883 | 5 | 5/10 | 7/24 · 5 |
| reduced4 | 0.103/0.472/0.592 | 0.228 | 0.918 | 6 | 6/10 | 7/24 · 5 |
| 下采样后重新 prepare（ds1024） | 0.070/0.511/0.556 | 0.142 | 0.894 | 1 | — | 1/24 · 1（丢 1 检出） |

分数、判定、框均不等价：框 IoU 约 0.2，good 出现 normal→anomaly。重新 prepare 方案丢检。基线本身较弱（域差大，样本小），只能作为“非等价”的证据，不能证明下采样“更好”。

### 69.4 结论与缺口

- **不采用**：
  - TEST-01 的 PNG 口径没有 decode 收益。
  - JPEG REDUCED_4 虽然能把 P95 降约 41ms，但违反“精度不退化”。
  - cpu profile 风险更高。
- 回归：全量 smoke **364 tests / 87.321s / OK**（零生产改动，作为环境基线）。
- 200ms 目标仍需从其他路径解决：
  - 相机直出 JPEG 或内存帧（跳过 PNG inflate）。
  - decode 与上一张 infer 流水并行。
  - render/persist 异步化。
- 生产验收 **0**。

## 70. AC-064 复核对话框角点微调、删除人工框与撤销上一步（2026-10-13，开发对话，生产未验收）

### 70.1 重要发现

- 模型框修订两次后会丢失来源。原 revise 每次写 `revised_from.source = 当前 source`，第一次 revise 后 source 已经变成 human_box，第二次再写就成了 human_box，`origin_source` 也随之变成 human_box。结果是被修订过多次的模型候选框会被当成人工框删除，绕过“模型框只能拒绝”的规则。现改为保留链首来源：`(revised_from or {}).source or source`。单次 revise 的语义不变（test_localization:74）。
- 撤销 revise 后来源判断仍是人工。`annotate_review` 原先只要看到某区域有 add/revise 事件就把 source 判为 human_box（geom_human）。撤销之后 bbox 已经恢复，但 source 仍然显示人工。现在已被 undo 的 add/revise 事件不再计入 geom_human。这是 annotate_review 中唯一的最小改动，未动 SYNC-21 的 used/stale 派生。
- `op=delete` 合法后，缺少 region_id 时原来抛 KeyError(None)，返回 404，语义不对，旧断言 test_localization:84 也随之失败（V§71 记录的唯一 error）。现在非 add 操作缺 region_id 时统一抛 ValueError，返回 422。
- 回收站式删除做不到修订链完整，因此删除与撤销都实现为追加事件：事件中记录 `before` 快照，区域保留但置为 inactive（deleted 或 retracted）。

### 70.2 op 语义（`algo/decision/localization.py`、`backend/api/region_review.py`）

- `delete`：
  - 只有原始来源为 human_box 才能删，模型候选框返回 422“请使用拒绝”。
  - 必须带 reason、expected_revision、event_id。
  - 软删除：`status=deleted, confirmed=False, deleted_from={status, operator}`，区域仍保留在 regions 中。
  - train_status 为 `excluded_deleted`，不进入 trainable。
  - 已 inactive 的区域不能再执行任何操作（422）。
- `undo`：
  - 撤销当前 operator 最近一次未被撤销的非 undo 事件。可以连续撤销逐步回退。不支持撤销 undo，也不支持重做。reason 可选。
  - 实现方式是追加反向修订：对 confirm、reject、revise、delete，把区域恢复到 `before` 快照；对 add，把区域置为 `retracted`（train_status 为 `excluded_retracted`），h* 编号不复用。
  - 被撤销的事件不会删除，只标 `undone_by`。新增的 undo 事件包含 `undo_of / undo_op / payload / before`，revision 加 1。
  - 目标事件之后若有他人的有效事件改过同一 region，抛 `ReviewConflict`，返回 409“撤销目标已被他人修改，请刷新后重试”。
  - 没有可撤销的本人操作时返回 422；历史事件缺少 before 快照时也返回 422。
  - 幂等：重复提交同一 event_id 且载荷相同，返回原结果；载荷不同则 409。

### 70.3 UI（`ui/widgets/region_review_dialog.py`）

- 纯函数：is_active、can_delete、bbox_corners、handle_tolerance（视图缩放与显示尺寸转换为原图像素容差）、hit_corner、drag_corner（对角固定、最小尺寸、限制在图内、不允许翻转）、build_delete_payload、build_undo_payload。
- 角点微调：
  - 选中框后显示 4 个白色角手柄（ItemIgnoresTransformations，6px）。
  - 拖动角点时只更新本地框。点击“替换选中框”后按 revise 提交，需填写理由，并带 expected_revision。
  - 切换选中项时放弃未提交的微调，恢复服务端坐标。
- “删除人工框”：模型框在 UI 侧先拦截，提交时须填写理由。删除后框从画布隐藏，表格中保留一行，显示“已删除(留档)/排除:删除”。
- “撤销上一步”：遇到 409 时显示 detail 并刷新。inactive 框在画布上不可点选。

### 70.4 验证（offscreen，`C:\Program Files\Python311\python.exe`）

- NEW `tests/smoke/test_region_review_edit.py`：**9 tests / 1.247s / OK**。覆盖：
  - 纯函数。
  - 后端 delete/undo：二次修订保留 engine_box 来源、不可撤销他人的操作、冲突、撤销 add 或 delete、修订链完整。
  - API：422、409、幂等；落库事件链为 `add,delete,undo,confirm,reject`；trainable 中只剩确认框。
  - 离屏 UI：手柄位置、拖拽后 bbox 数值、无理由拒绝提交、带理由 revise、删除与撤销按钮、409 后刷新重试。
- 回归：
  - test_region_review_draw、test_region_human_box_status、test_localization 与本文件合计 **24 OK**。
  - tests/ui **19 OK**。
- 全量 smoke：**364 tests / 86.771s / OK**（350 + 本批 9 + SYNC-21 的 5）。V§71 中的 test_review_ops error 已消除。日志在 `review/task/ac064_region_edit/smoke.{out,err}.log`。

### 70.5 边界

- 只支持 4 角拖动，不支持拖边或整体平移。手柄未在真实鼠标/桌面上人工验收（TEST-04 类）。
- 撤销不支持重做，也不能撤销 undo。早于本批、没有 before 快照的历史事件无法撤销。
- 与 SYNC-21 的联调只做到全量 smoke 通过。按 V§71 规则，删除或撤销已消费的框应判为 stale_reason=deleted/revised 并被阻断，但这个组合没有单独写断言，属于未验证项。
- 生产验收 **0**。待同步 SYNC-20。

## 71. AC-063/064 region 级训练消费（used）回写与消费后修订 fail-closed（2026-10-13，开发对话，生产未验收）

（§69、§70 预留给其他任务，本节按任务指定编号为 §71。）

### 71.1 重要发现

- 生产代码中没有任何地方调用 `trainable_region_boxes`，只有测试在调用。实际的框级训练样本是 `Feedback.region` 的单个人工框，有两个消费入口：
  - 候选路径 `backend/self_learning/service.py::build_feedback_candidate`，调用 `engine.learn_candidate`。
  - 即时路径 `submit_feedback`。
- `region_review` 的多框结论此前只用于整体 revision 守卫（V§54/AC-048），没有框级“被谁消费”的记录。
- 本次的口径：构建候选时，该检测 `region_review` 中全部 eligible 框都视为被这个候选消费，因为候选基于这张已复核检测的可信标注。
- 同一人重新确认会让修订号变化，这种情况也判为 revised。这是保守的 fail-closed 做法。

### 71.2 数据结构（不新增 DB 列，沿用 `Detection.n_tiles["region_review"]` JSON）

- 新增顶层键 `consumption[]`，每条记录包含：
  - `region_id`、`region_revision`（最后一次涉及该框的事件修订号，无事件为 0）、`bbox`
  - `review_revision`、`candidate_id="candidate:feedback:{id}"`、`feedback_id`
  - `version`、`parent_version`、`kind="candidate"`、`at`
- 按 (candidate_id, region_id) 去重。
- 消费记录不改 `revision`，也不写 `events`。`apply_op`/`_apply_undo` 使用 `{**review}`，会保留该键。
- 写入时用 CAS（`n_tiles == old`）最多重试 3 次，失败时不抛异常。结果写入 `learning_trace.region_consumption`，状态为 recorded、none、detection_missing 或 cas_conflict（cas_conflict 同时记 warning）。
- `candidate_guard.regions` 保存构建时的消费快照 `[{region_id, region_revision, bbox}]`。
- `algo/decision/localization.py` 新增纯函数：`consumable_regions`、`record_consumption`、`consumed_snapshot`、`stale_regions`。

### 71.3 派生与 fail-closed

- `annotate_review` 给每个 region 派生四个字段：`used`、`used_by[]`、`stale_after_use`、`stale_reason`。stale_reason 取值：
  - `deleted`：区域消失，或 status 属于 deleted/retracted。
  - `excluded`：不再 eligible，同时附带 train_status。
  - `revised`：修订号或 bbox 变化。
- `annotate_review` 顶层增加 `stale_consumption[]`。
- 未被消费的框不会算作 stale。重新消费后以最近一次记录为准，stale 解除。
- 两个输出口都经过 `annotate_review`，所以新字段会透出：`GET/POST /api/detections/{id}/regions` 和 `to_dict` 中的 `region_review`。本任务未改 API 文件。
- `CandidateRelease._validate_feedback_guard`（validate/decide/switch 都会走这里）：若 guard 带 regions 快照，且 `stale_regions` 非空，则 fail-closed，返回 `PARENT_REBUILD_REQUIRED_REGION_STALE`，violation 中附 stale 列表。这一项先于整体 `REVIEW_REVISION` 校验。没有 regions 的旧 guard 行为不变。

### 71.4 验证

- NEW `tests/smoke/test_region_used_status.py`：**5 tests / 3.654s / OK**。覆盖：
  - 快照与 used 派生。
  - stale 的三种原因，以及消失、未消费、重新消费三种情况。
  - DB 回写，以及 to_dict 和 API GET/POST 的透出（revise 后 consumption 保留）。
  - revised、excluded、deleted 均被 REGION_STALE 阻断。
  - 旧 guard 兼容；没有 eligible 框时不回写。
- 全量 smoke：**355 tests / 102.069s，errors=1**。唯一错误是 `test_localization.test_review_ops` 第 84 行，旧断言要求 `op=delete` 是未知操作并抛 ValueError。另一 agent 已在 `apply_op` 中新增 delete，缺少 region_id 时改抛 KeyError。这与本任务无关（本任务未改 apply_op），由 delete/undo 那一方更新该断言。其余 354 项 OK。

### 71.5 边界

- 即时学习路径 `submit_feedback` 未回写消费记录。
- CAS 连续冲突时只在 trace 中记 cas_conflict，没有补偿重试。
- UI 尚未展示 used/stale。
- 需要与 delete/undo 改动联调。
- 生产验收 **0**。待同步 SYNC-21。

## 72. AC-069 前台锁粒度：前台请求的预提取移出品类锁（2026-10-13，开发对话，生产未验收）

### 72.1 背景

V§66 的学习让行没有覆盖前台请求之间的品类锁串行。test02 在 stable 上测得 c4 idle P95=406ms。

在开发版上，`_pipe_lock(category)` 原本包住了 predict 的全部步骤：读图、预处理、切块、DINO 前向、特征后处理、记录与 trace。

### 72.2 三方案评估

- **读写锁：不可行。** predict 不是只读操作，会写这些状态：
  - `_item_cache`
  - `_last_hms`
  - trace_api
  - `normal_bank._observe_decision`
  - selector 统计
  
  如果改成共享读锁，就得给这些状态逐个加锁。这样风险大，结果还会依赖到达顺序，失去确定性。
- **GPU 小批合批：不采用。**
  - single tile 下每张图只有 1 个 tile，合批收益很小。
  - 合批要攒批等待，反而会推高 c1。
  - batch 组成会随到达时序变化，BN/数值路径难以保证逐位确定。
  - 要改 dino 前向接口，复杂度高。
- **缩小临界区：采用。** 把纯函数段移出锁外：读图、预处理、切块、DINO 前向。DINO 前向套全局 `_gpu_sem`（BoundedSemaphore(1)），保证单 GPU 串行，避免显存叠加。

### 72.3 实现

- `algo/eval/offline.py`：
  - 新增 `prepare_input(path|rgb, gpu_lock)`，产出 prep，内容为 rgb、tiles、特征及其指纹。
  - 新增 `_predict_prepared(prep, cache)`。
  - `predict` / `predict_frame` 新增 `prepared=` 参数。不传时行为不变。
- `backend/engine/__init__.py`：
  - `predict_image_path` / `predict_ndarray` 进入 `_foreground` 后，先在锁外调用 `_predict_prefetch`。
  - 进入品类锁后由 `_prefetch_valid` 校验指纹：id(pipe)、id(backbone)、id(_pre_cfg)、tiling。学习、替换、切换任一发生时，指纹不一致，丢弃预提取，回落到锁内重算，结果正确。
  - 以下情况跳过预提取，走原路径：缓存命中、桩 pipe（无 prepare_input）。
  - 类开关 `predict_prefetch` 用于 A/B 和回退。
- 锁序不变（品类锁 → `_write_lock`）。`_foreground`、`lock_wait_ms`、`_learning_yield` 保持原样，预提取期间也计入 `_foreground`。

### 72.4 测量

脚本和产物在 `review/task/test02_gpu_concurrency/dev_lock/`，脚本为 `bench_dev_lock.py`。before/after 分别在独立进程中运行，表中为 wall P95（ms）：

| 场景 | before(prefetch=0) | after(prefetch=1) | stable 基线 |
|---|---|---|---|
| c1_idle | 245.9 | 258.3 | 263.4 |
| c1_learning | 1410.8 | 1530.2 | 1504.8 |
| c2_idle | 294.0 | 248.9 | 290.3 |
| c2_learning | 903.6 | 899.3 | 820.2 |
| c4_idle | 302.7 | 303.1 | 406.1 |
| c4_learning | 953.0 | 963.5 | 959.6 |

- c4_idle 的 lock_wait P95 从 **77.7ms 降到 0.007ms**，前台之间的等锁基本消除。
- 精度不变：
  - 两轮内部 c4 与串行比，mismatch 均为 0。
  - 跨进程比 40 张串行结果，decision 差 0，score 最大差 0.0。
  - learning_errors=0。

### 72.5 发现（供后续改进）

- **等锁消除后 wall 没有改善。** c4 idle before 时等锁只有约 78ms，单 GPU 前向本来就是串行的。等锁去掉后，等待转到 GPU 信号量和 CPU 争用上（persist、heatmap 等锁外段）。
- stable 406ms 与开发版 before 302.7ms 之间约 100ms 的差距，来自 SYNC-15 decode 等已有优化，不是本批带来的。
- 要继续降低 c4 wall，方向是：
  - 减少单 GPU 前向耗时（半精度、更小输入）
  - 或压缩锁外的 persist/heatmap 段
  
  继续细化锁收益很小。
- learning 档的等锁为 700～1300ms，来自锁内的学习 hold，需要另外拆分 learn 的临界区。本批不涉及。
- c1 有约 ±12ms 波动，属于噪声，加上预提取多一次函数分派的开销。

### 72.6 验证

- NEW `tests/smoke/test_engine_concurrency.py`：7 项 OK。覆盖：
  - 并发结果与串行一致
  - 临界区缩小（锁内不含预处理和前向）
  - 缓存命中跳过预提取
  - pipe 替换或 backbone 变更后丢弃预提取，回落重算
  - 预提取期间计入 `_foreground`
  - `_learning_yield` 会等待预提取中的前台
- 全量 smoke：**371 tests / 105.453s / OK**，stdout/stderr 分开存放于 dev_lock。

### 72.7 边界

- 预提取后如果发生学习或替换，会回落重算，相当于白做一次前向。学习频繁时 c1_learning 可能略有上升（1410→1530，含噪声）。
- 未在真实产线多相机并发下验收。生产验收 **0**。待同步 SYNC-22。

## 74. AC-058 统计缺失值回 null + AC-063/064 即时学习消费回写、UI 消费标签与已消费框删除/撤销阻断（2026-10-13，开发对话，生产未验收）

§73 留给并行 agent，本节取 74。

### 74.1 改动

- A 统计缺失值（AC-058）：
  - `backend/api/routes_stats.py`：quality/batch/health 的比率和平均值在分母为 0 或无数据时返回 `None`，不再用 0.0 冒充。
  - overview 新增 `p95_latency_e2e_ms` 和 `n_e2e_missing`。
  - timeseries 的 e2e 改为 `func.avg(latency_e2e_ms)`，去掉 coalesce 回退到推理耗时。StatsDaily 没有 e2e 时返回 `None`。
  - export 把 `None` 写成空串。
  - `backend/api/_wo_shared.py`：`bad_rate` 和 `stats.rate` 在无数据时返回 `None`。
  - `backend/api/routes_wo_stats.py`：`_p95([])` 返回 `None`。report 的平均延迟只统计 e2e，并新增 `n_e2e_missing`。CSV 列改为“平均延迟ms(e2e)”，新增“e2e缺失”列。
  - UI 消费方：dashboard 的 rate 为 None 时显示 NA；stats_page 和 learning_page 的 `_pct` 已对 None 做容错。另外修正了 `ui/pages/stats_page.py` 的一处过时注释，逻辑不变。
- B 即时学习消费回写（AC-063）：`backend/self_learning/service.py` 的 `submit_feedback_record` 构造 instant_guard。只有 final 且无 error 时才调用 `_record_region_consumption(..., candidate_id="instant:feedback:{fb_id}", kind="instant")`，并且幂等。失败时不回写。
- C UI 消费标签（AC-064）：`ui/widgets/region_review_dialog.py` 的框列表新增“学习消费”列，由 `consumption_text()` 生成，取值为：未用于学习、已用于学习、已用后变更（删除/排除/修订）。
- D 发布阻断：只补测试，不改代码。

### 74.2 验证（offscreen，`C:\Program Files\Python311\python.exe`）

- NEW `tests/smoke/test_stats_null_and_consume.py`：7 项 OK（4.4s）。覆盖：
  - 空库比率为 None
  - timeseries e2e 不冒充推理耗时
  - 工单 bad_rate/_p95/report/CSV 输出空值
  - 即时学习成功时回写 kind=instant（3 条，幂等），失败时不回写
  - 对话框第 8 列标签
  - 删除已消费人工框后 guard 返回 `PARENT_REBUILD_REQUIRED_REGION_STALE` 和 `{"h0":"deleted"}`
  - 撤销 add 返回 deleted，撤销 revise 返回 excluded，均被阻断；恢复后放行
- 全量 smoke `-m unittest discover -s tests\smoke`：**383 tests / 104.433s / OK**，stdout/stderr 分开写在 `%TEMP%\aoi_smoke_nc_{out,err}.txt`。

### 74.3 边界

- timeseries 的 `total_latency_ms` 仍是累计和，语义不变。
- 即时路径的回写走 CAS，并发冲突时以重试为准，没有做压力测试。
- 对话框只在 offscreen 下做了断言，没有实机目检。
- 生产验收 **0**。待同步 SYNC-23。

## 75. 稳定版第 6 轮同步验收（SYNC-16～18、20～23，2026-10-13，本对话）

- 合并同步 7 批 22 文件（15 DIFF 备份于 `review\sync_backup\SYNC-16~23\` + 7 NEW），复制后哈希逐一一致。依赖关系按登记合并：SYNC-20/21 共享 localization.py 与 region_review_dialog.py，一并同步后 SYNC-21 记录的 test_review_ops 旧断言错误消除。
- 稳定版定向合并（覆盖全部 7 批复验命令的测试文件）：**147 tests / 52.560s / OK**；tests/ui **19 tests / 4.017s / OK**。
- 稳定版全量 smoke：**378 tests / 100.712s / OK**（331+47，与新增用例数精确吻合）。
- SYNC-19 编号未使用。
- 边界：TEST-02 口径复测（SYNC-16/22 的让行与锁粒度在生产口径下的效果）尚未执行，需按 dev 提示用 `lock_wait_ms` 响应字段重测；AC-058/063/064/069 仍部分实现，生产验收 **0**。未改生产 DB/快照，未提交 git。

## 76. G1 单图 e2e：异常图渲染移出关键路径（2026-10-13，开发对话，生产未验收）

> 子 agent 原登记为 §73/SYNC-23，但与并发写入冲突后丢失，SYNC-23 已被 §74 占用。现补登为 §76/SYNC-24。

- 分段（改前，n=24）：io_wait 恒为 0，等待写盘对 e2e 没有收益。只有异常图存在 render 段，耗时 32～36ms，其中 overlay 占 22～25ms。
- 改动：
  - `backend/pipeline/service.py`：overlay（resize/cvt/着色/addWeighted/画框/编码）、热图 JPEG、mask、raw npy 全部提交到 `_io_exec`。主线程只同步保留响应框所需的 `_heatmap_u8` 与 `_mask_and_boxes`。
  - 新增 `_track`/`wait_artifact`/`_safe_job`：登记在途产物，失败时写 `logger.exception`。
  - 两处逐位等价的微优化：heatmap_u8 改为原地运算，耗时 7.3→3.4ms；二值化改用 cv2.threshold。
  - `backend/api/routes_system.py` 的 `api_file` 在白名单校验后调用 `wait_artifact`：超时返回 503 并带 Retry-After，失败返回 500。原因是异步写盘后，文件可能在落盘前就被访问。
- 结果：
  - render 段 P95 在三次 A/B 中均为 34→19ms，结果确定。
  - e2e P95 空闲时为 180.6ms（≤200）。负载下三次 A/B 的新侧为 225.3、214.5、190.4ms，受噪声影响，不能稳定达标。
- 剩余瓶颈：
  - decode：PNG inflate，P95 98～107ms，约占一半。
  - infer：负载下 P95 82～96ms，偶有 >200ms 离群，属于 CPU 争用。
  - 两者都需要改输入格式（JPEG/内存帧）或做算力隔离（SS-A）。
- 精度：三次 A/B 中 decision/score/defect_boxes 以及 heatmap/overlay/mask 文件 sha 全部一致。`heatmap_postproc` 只有含时间戳的路径不同。
- 测试：新增 `tests/smoke/test_async_persist.py` 5 项。全量 smoke **383 tests / 98.3s / OK**。
- 产物：`review\task\test01_e2e_latency\dev_async\`（result.md、breakdown.json、ab*.json、service_before.py）。
- 发现：PowerShell 5 用 `1> 2>` 重定向原生进程时会截断 stderr，长测试改用 `Start-Process -RedirectStandardOutput/-RedirectStandardError`。
- 生产验收 **0**。待同步 SYNC-24。

## 77. 流/视频帧内存直检（2026-10-13，开发对话，生产未验收）

- 动机：`_rtsp_loop` 每帧先 imwrite 临时 JPEG，再走 detect_image，最后删除。这一来回多出编码、写盘、读盘、解码，还多了一次有损重编码。视频抽帧也是先落盘再从盘上读回检测。
- 改动：
  - `backend/pipeline/service.py`：
    - 新增 `DetectionService.detect_ndarray(arr_bgr, category, *, target_type, persist, with_heatmap, workorder_id=None, image_path_hint=None, image_id=None)`，流程为 BGR→RGB → `engine.predict_ndarray` → `_finish_detect`，结果标记 `zero_copy`。
    - `detect_frame` 改为解码后复用 `detect_ndarray`，行为不变：image_path 仍是 `<memory_frame>`，不归档。
    - `iter_video_detections` 保留 imwrite。在检测前同步写盘，是为了让 `_register_frame` 能登记文件路径。检测本身改为 `detect_ndarray(frame, image_path_hint=tmp)`，不再从盘上回读。
    - 无文件时的证据留存：旧 RTSP 路径中，persist 时临时 jpg 还在，异常帧会被 `_auto_archive` 复制归档。新路径没有文件，因此新增 `archive_rgb` 链路（`_finish_detect`→`_persist`→`_auto_archive`）。当源文件不存在时，直接用内存帧写 `det{id}_frame.jpg`。`detect_frame` 显式关闭这条链路，保持原行为。
  - `backend/pipeline/pipeline_service.py`：`_rtsp_loop` 直接调用 `svc.detect_ndarray(frame, persist=True, with_heatmap=False, target_type="stream", workorder_id=...)`，并删掉 tempfile/os 往返。
- 耗时（8 帧 × 3 轮交替，n=24，persist=True，无热图）：
  - 单帧墙钟 P50 **135.8→65.4ms**，P95 **166.6→123.3ms**。
  - infer P50 104.7→52.6ms。old 的 infer 含 JPEG 解码。
  - 与全量 smoke 并发的首轮结果同向：134.6→67.0 / 156.0→119.6。
- 精度：
  - old/new 逐帧对比，8 帧中 1 帧 decision 翻转（det878：old 为 anomaly，new 为 normal），score 绝对差最大 1.33。
  - 归因对照（6 张真实图）：同为 PNG 像素时，path 入口与 ndarray 入口逐帧完全相等；同为 JPEG 像素时，两者也完全相等。差异全部来自旧路径的 JPEG 有损重编码（质量 95），新路径的结果等于直接检测原图，**不是退化**。
  - 上线后，流帧分数与旧流历史记录（基于 JPEG）会有系统偏差。
- 测试：新增 `tests/smoke/test_frame_ndarray.py` 5 项，覆盖以下场景：
  - 与 detect_frame 逐字段等价；
  - 无文件入库时 image_path 为 `<memory_frame>`，且内存帧被归档；
  - detect_frame 不归档；
  - 视频抽帧在检测前已写盘且不调用 detect_image；
  - RTSP 循环不调用 mkstemp/imwrite，并核对 kwargs。

  全量 smoke **388 tests / 98.9s / OK**（383+5）。
- 产物：`review\task\test01_e2e_latency\dev_frame\`（bench_frame_ab.py、ab.json、run.log、result.md）。
- 发现（原有问题，未修）：`detect_frame`/`detect_ndarray` 在无 hint 且 with_heatmap=True 时，热图文件名 stem 为 `<memory_frame>`，含 Windows 非法字符 `<>`，热图写盘可能失败。RTSP 路径使用 with_heatmap=False，不受影响。
  - 续修（同批次）：`_finish_detect` 用 `re.sub(r'[<>:"/\\|?*]', "", stem) or "frame"` 净化热图 stem；新增 `test_memory_frame_heatmap_filename_is_valid`（detect_frame+热图，heat/overlay 文件名合法且写成）。全量 smoke **389 tests / 112.8s / OK**。
- 生产验收 **0**。待同步 SYNC-25。

## 78. 学习锁内持有缩短：RegressGate check→recenter 锚定分复用（2026-10-13，开发对话，生产未验收）

- 动机：后台即学在品类锁内持有约 600～1000ms。拆解（`review/task/test02_learn_lock/dev_split/probe_breakdown.json`）显示，主因是 `_reflow_labeled` 中的 `gate.check`：20 个锚点各跑一次完整 predict，约 450～600ms。accepted 后，`gate.recenter` 会在相同状态下再跑一整轮。
- 改动：
  - `algo/eval/offline.py`：
    - 新增 `_gate_anchor_pass`：逐张 predict，数值和副作用不变，同时留存 (fused, observed, rec)。
    - 新增 `_gate_anchor_replay`：重放 normal_bank `_observe_decision` 与 trace 落盘。
    - 新增 `_gate_state_token`：对决策阈值、融合权重、校准 edges 字节、slot/bank 身份与形状、nb/db/head 的版本与长度、router 参数字节做 blake2b 指纹。
    - `_finalize` 记录 `_last_observed`。
  - `algo/ssocl/feedback.py`：
    - check 留存 pending。recenter 在 pipeline、锚点列表、指纹三者都一致时单次消费，并重放副作用；否则重算并计 mismatch。
    - 判定非 accepted 时作废 pending。不调用 recenter 的 accepted 路径（defect_bank、router_finetune）在 accepted 后清空 pending。
    - 权重写回路径也接入复用。
    - 实例级覆盖 `_predict_final_fused`（测试桩）或伪 pipeline 时，回落到原逐张路径。
  - 统计 `gate.reuse_stats`，键为 reused/recomputed/mismatch。
  - `learning_parallel` 默认值不变。
- 实测（TEMP，详见 `review/task/test02_lock_contention/dev_learnlock/result.md`）：
  - accepted 事件 hold：908.5/1018.9 → 684.4/630.4ms，均值 -32%。全部事件 P95：783/837 → 684/725ms。
  - 前台学习中 wall：P50 18.7→17.7ms，P95 34.8→29.8ms，max 904～1018→701～743ms。
  - rolled_back 用同进程交替 A/B 测量：658.6 vs 662.7ms，无差异。`_gate_state_token` 耗时 0.1ms。
  - 精度：actions 序列与 base_mean 一致，学习后 eval fused 最大差 0.0。
- 未拆分段及原因（保持原样）：
  1. rolled_back 路径的 check 本身（占多数事件，约 600ms）未移出锁外。锁外试算要么需要 deepcopy pipe 做影子，实测 370～1290ms，比锁内 check 还贵；要么需要浅影子，而浅影子必须隔离 normal_bank、sem bank、calibrators、`_cal_normal_hist`、`_item_cache`，以及 observe/trace 副作用。
  2. check 自身会写 gate 的 history、audit、violations、frozen 状态，回到锁内后复查也无法廉价证明等价，fail-closed 风险高，因此未做。
  3. setup、unfreeze(rebase)、`_search_weights_loo` 之前没有对应的 check，仍逐张重算。
- 测试：新增 `tests/smoke/test_learning_lock_split.py`，共 7 项：
  - 复用与重算结果一致，含 observe/trace 副作用，覆盖 trace 开/关；
  - weights、edges 原地修改、sem bank、router 原地修改、nb version、tau 六类状态变化均能被检出并回退重算；
  - pending 单次消费且作用域隔离；
  - 伪 pipeline 不受影响；
  - 并发与串行结果一致，且无 torn 读；
  - accepted 时 hold 低于 0.7 倍。

  定向测试（含 candidate_learning_state、regress_gate_ac036、learning_yield、engine_concurrency）**36 tests / 24.6s / OK**。全量 smoke **395 tests / 106.8s / OK**。首轮全量发现 1 个 error：test_candidate_learning_state 在实例上覆盖了 `_predict_final_fused`，被新路径绕过，已按上面的回落规则修复。
- 生产验收 **0**。待同步 SYNC-26。

## 79. AC-072 业务预算与保留期可配置化（2026-10-13，开发对话，生产未验收）

- 用户定值（2026-10-13，取代 TEST-08 草案中的 FPR≤1%、Recall≥99%、回退≤0.5pt、灰区≤10%）：
  - G2：FPR ≤5%，缺陷召回 ≥95%。
  - G6：旧域召回回退 ≤2pt/版本，灰区 ≤15%。
  - 保留期：原图热存 90 天、冷归档 12 个月；热图/overlay/mask 30 天；反馈/事件/WAL 永久；快照每品类最近 10 版，加上所有曾激活版本；EvalRun/审计 12 个月；标签迟到容忍 7 天（超时只统计，不入训）。
- 改动：
  - `backend/core/config.py`：
    - 新增 `BUSINESS_BUDGET_SPEC`（11 项，含类型、默认值、范围）和 `_SCHEMA["business_budget"]`；
    - `update` 在类型校验后做范围校验，越界或 NaN 抛 ConfigValidationError 并附原因；
    - 新增 `business_budget()`，非法存量值回落默认；
    - 新增 `candidate_release_policy()`，把 pct/100 映射为候选门禁 policy。
  - `backend/api/routes_system.py`：
    - `_CONFIG_SCHEMA` 白名单加入 11 项；
    - GET 返回 value/default/min/max 和固定项 `fixed`（WAL=永久）；
    - POST 仍需 admin，非法值计入 `rejected`，原因写入 `errors`（原先会 500）；
    - `/system/info` 附带 business_budget。
  - `ui/pages/settings_page.py`：新增“业务预算与保留期”分组，编辑器范围取自 min/max，提供“恢复默认”（只填回编辑器，需再点保存），并展示保存时的拒绝原因。
  - 消费点：
    - `stats_page.py`：误报率、灰区率 KPI 显示上限，超限标红并标“（超限）”；UI 侧默认值兜底与后端一致。
    - `eval_page.py`：选中 EvalRun 后显示 FPR/召回与达标线对照。
    - `main_window.py`：从 system_info 注入配置。
- 测试：新增 `tests/smoke/test_business_budget.py` 10 项，覆盖默认值、校验、TEMP yaml 往返、policy 映射、API GET/POST、设置页离屏编辑→保存→恢复默认、统计页和评估页达标线。定向 **10 tests / 0.35s / OK**；全量 smoke **405 tests / 109.1s / OK**（395+10）。证据：`review/task/test08_business_budget/dev_config/`。
- 边界：
  - 保留期只登记和展示，系统不自动删除，也未做删除审批。
  - 标签迟到容忍只存配置，入训路径未接入（后端没有迟到判定）。
  - 候选门禁只提供 policy helper，`candidate_release.py` 未改，调用方需改用该 helper 才会生效。
  - 精度、service.py、生产 DB 和快照均未改动。
- 生产验收 **0**。待同步 SYNC-27。

## 80. 稳定版第 7 轮同步验收（SYNC-24～27）与 TEST-01/02 复测（2026-10-13，本对话）

### 80.1 同步与验证
- 合并同步 4 批 13 文件（9 DIFF 备份于 `review\sync_backup\SYNC-24~27\` + 4 NEW）：G1 渲染异步化、流/视频帧内存直检、学习锁拆分（RegressGate 锚定分复用）、业务预算可配置（用户已定值）。
- 稳定版定向合并（覆盖 4 批全部复验测试）：**57 tests / 26.685s / OK**；tests/ui **19 tests / 3.643s / OK**。
- 稳定版全量 smoke：**405 tests / 106.532s / OK**，与开发版 405 完全一致。

### 80.2 TEST-02 复测（验证 SYNC-16/22/26 让行+锁粒度，同脚本同种子）
- c1 learning P95 1504.8→**394.1ms（−74%）**；c2 820.2→694.7；c4 959.6→768.3；idle 各级别也降至 214.9~358.1ms。
- 学习事件被让行限速（单事件 1.0~2.3s，预期行为）。
- **判定仍 FAIL**：所有级别 P95 仍 >200ms；取消/限流接口仍未覆盖。

### 80.3 TEST-01 第三轮复测（SYNC-24/25 渲染异步化后，同 40 图同种子）
- direct：wall P95 295.8→235.3→**229.2ms**；e2e P95 288.4→216.3→**204.1ms**（距红线仅约 2%）；infer P95 86.7ms。
- 学习负载 wall P95 由约 1800ms（n=2）降至 **610.2ms**（n=12，让行使重叠样本增多且峰值大降）。
- **判定仍 FAIL**（204.1 > 200ms）；剩余瓶颈为 PNG zlib inflate 与锁外 persist。

### 80.4 结论
- SYNC-24～27 同步完成且稳定版复验通过；AC-066/069 优化方向正确且效果可测，但仍 FAIL；AC-072 业务数值已由用户定值并可配置（SYNC-27），保留期自动删除/标签迟到入训/候选门禁接线仍未做。生产验收 **0**。未改生产 DB/快照，未提交 git。

## 81. TEST-07 精度门禁小样本实测（AC-067，本对话，FAIL，重要发现）

- 数据：稳定版 DB 官方切分（gold_finger：train 14 正常 / train_anomaly 30 缺陷 / test 10+10），逐图 sha256 清单；隔离 TEMP prepare 新品类 gold_finger_eval v1；冻结阈值 FPR≤5%/Recall≥95%/灰区≤15%（用户 2026-10-13 定值）。
- 结果：**FPR 10/10=100%**（Wilson [72.2%,100%]），Recall 100%（CI 下限 72.2%），灰区 0%。全部 20 张测试图判 anomaly。
- **重要发现**：tau=0.8281 低于全部样本得分（1.49～2.0）；9/10 缺陷图与 1/10 正常图得分恰好 2.0（上限钳位饱和）。与 §65.2 降级快照分数恒定现象同向，指向融合分数钳位/tau 校准/槽位权重生效性问题，已转开发排查。训练与测试同域（data_local 官方切分），排除跨域借口。
- G6 旧域容差未测（单版本基线）；n=10/类过小，点估计通过也不能证实门禁。
- 产物：`review\task\test07_precision_gate\`（result.md/env.json/cmd.txt/raw：split_manifest、per_image_scores、eval_run JSON）。
- 门禁机制本身工作正常（明确 FAIL 而非放行）；AC-067 生产验收 **0**。

## 82. TEST-05 离线/干净缓存加载验证（AC-075，本对话，部分 FAIL）

- 方法：独立子进程 + 全新空 HOME/USERPROFILE/TORCH_HOME/HF_HOME（TEMP），HF_HUB_OFFLINE=1/TRANSFORMERS_OFFLINE=1 模拟断网；**未清用户真实缓存**。被测加载器为开发版 EXPs `backbone/dino.py`（只读引用）。
- A（离线+vendor 完整）：3.78s 加载成功（22.06M 参数），走随包权重——PASS。
- B（离线+DINOV2_LOCAL_DIR 指向空目录）：仍成功——`_vendored_dir` 有随包默认搜索路径兜底——PASS。
- C（离线+vendor 有 hubconf 缺权重文件+空缓存）：回退 `torch.hub.load` 直连 GitHub，**无超时控制，180s 挂起被杀**——**FAIL**。有网时该路径会静默联网下载，断网时卡死；违反 AC-075“缺权重明确报错、不静默联网、不卡住”。
- 建议：离线模式下禁止在线回退并立即 raise，或为 hub.load 加显式超时。产物：`review\task\test05_offline_weights\`。
- AC-075 从“已实现”角度看仍有真实缺口，生产验收 **0**。

## 83. TEST-06 旧生产 trad 协议八项等价比对（AC-062，本对话，FAIL/new_candidate）

- 前置调研：旧生产协议工件已存在且被哈希门禁冻结（`algo\vendor\traditional.py` + `algo\slots\trad.py`），parity harness（`algo\features\parity.py`）八项检查完备；唯一缺口是 AOI_feature 候选适配器，已作为测试工件写于 `review\task\test06_trad_parity\raw\aoi_feature_candidate.py`（未改任一侧源码）。
- 结果：**8 项过 2 败 6，max_abs_diff=14.665，verdict=new_candidate**。通过：dimension_order_slice（dims={200}、trace 合规）、rng（确定性、不污染全局）。
- **唯一根因**：selfsim 组子采样 RNG 口径不同——生产 `default_rng(42).choice` vs 候选 seed(0) 包裹的全局 `np.random.choice`（RandomState 与 default_rng 算法+种子均不同），块数>100 时采样集合不同，selfsim 5 维差最大 14.67，经 z-score 级联击穿 bank/kNN/scoring/report。其余 14 组全部逐位为 0。
- 次要观察：候选未实现生产的 `nan_to_num`（nan_policy=zero）；候选拟合 float64 vs 生产 float32。
- 这正是 fail-closed harness 的设计用途：AOI_feature 不能以“等价”名义接入，须先统一 selfsim RNG 口径或按新候选流程评估。
- 产物：`review\task\test06_trad_parity\`（result.md/env.json/cmd.txt/raw：适配器、运行脚本、parity_result.json、15 组逐组归因 group_diffs.json、运行日志）。
- AC-062 生产验收 **0**。

## 84. TEST-04 真实桌面 UI 联调（AC-051/056/057，本对话，PASS 覆盖场景）

- 真实桌面（非 offscreen）启动稳定版 main.py，computer-use UIA + 截图 + API JSON 逐场景配对。
- PASS 场景：Gauge 空态"未知/未判定"不补 0 不猜测；工作台口径卡（推理/e2e/接收率不可用不以 0 代替）；不完备预警横幅（open≠缺陷、处置不自动训练）；开放告警列表/详情与 API 逐字段一致（legacy_unknown→"旧证据未知"，AC-053 legacy 保留）；评估看板业务预算达标线（FPR ≤5% 超限/召回 ≥95% 达标，SYNC-27 用户定值生效）+ 延迟柱图红色超预算柱+200ms 虚线。
- 附带重要交叉证据：生产 v2（降级模式）test 集 EvalRun id=1：AUROC=0.5、FPR=100%（tn=0/fp=10），与 TEST-07 TEMP 新 prepare 结果一致——精度问题为系统性，非单模型问题；EvalRun 绑定 model_version+snapshot_meta_sha256+split（AC-067 绑定要求在真实链路生效）。
- 未覆盖：Gauge 有值态（需数据源→工单→试检链路）；alert/warn 同屏排序。
- DB 写入：稳定版 DB 新增 EvalRun id=1、task 18（正常应用记录，可在评估看板删除）。
- 产物：`review\task\test04_desktop_ui\`（result.md/env.json/cmd.txt/raw：全屏截图、api_snapshots、api_eval_pair）。
- AC-051/056/057 桌面场景联调通过；生产验收仍由用户最终确认。

## 86. PCB-ACCEPT-01/02/05 验收（2026-10-14，本对话）

- **PCB-ACCEPT-05（同步+复验）**：只读对比两树 PCB_Dual——477/471 文件，447 一致、24 DIFF、6 NEW、稳定版独有 0（无未解释冲突）。备份 `sync_backup\PCB-ACCEPT-05\` 后受控同步 30 文件，逐文件哈希一致。稳定版回归 35 tests OK（tests 无 __init__.py，须模块方式运行）；6-case 真实 paired 对照通过（SMT 3+Shift 1+TH 2，unexpected_failures=[]；SMT raw 输出图等价仍 FAIL 0/3 单列）。产物 `review\task\pcb_accept_05\`。
- **PCB-ACCEPT-01（真实桌面，稳定版 PCB_Dual）**：模板选择/标准图载入/测试图载入正常；**尺寸门禁 fail-closed**（941×620 vs 1317×545 → ERROR 已拦截，无伪 OK）；真实 NG（多锡 316,181，算法来源 solder_多锡，ROI 缺显示"未提供"、分数"不可用"不显 0）；调参预览防抖重跑 NG→OK 且**未点保存时模板参数库未被污染**（API 核对）。发现：Qt spinner 程序化赋值不提交（UI 自动化兼容性，建议复核提交时机）。未覆盖：发布链路（无已发布模板）、TH 模式、压力场景。
- **PCB-ACCEPT-02（分支/坐标）**：部分 PASS（覆盖矩阵见 `review\task\pcb_accept_02\result.md`）；BLOCKED：toe/rim、Crack 真实样本、TH 真实模式、真实配准失败样本。
- **PCB-ACCEPT-03/04 BLOCKED**：GT/阈值未冻结；性能专项未获实测授权（仅设计估算）。
- 以上均为待复核状态，生产验收不勾选。
- 续（2026-10-14 第二轮）：按协同指引完成 **TEMP 隔离发布链路**——TEMP 树（13.1MiB，排除数据目录）+ 全新模板库 + 8023 端口；checklist 门禁先正确阻断（品类模型未准备）→ 稳定 AOI_Core 启动后放行 → API 发布 → UI 选中已发布模板 → 检测 NG（多锡，与调参页一致）→ 重启后发布态保持；**稳定版 storage/模板库零写入**。产物 `review\task\pcb_accept_01\`（含 temp_tree）。TH 模式与压力场景仍缺。

## 85. final_sim 模拟验证组：UI/角色/槽位/复核/定位/视频/部署/清理（2026-10-13，开发对话，生产未验收）

- 改动：
  - `algo/backbone/dino.py`（AC-075）：回退顺序改为 vendor→hub 缓存→在线；vendor 分支 `pretrained=False`+`torch.load`，不再复制权重到 checkpoints；离线且缺权重时立即 `RuntimeError`，修正 V§82 场景 C 的 180s 挂起。
  - `tools/cleanup_data.py`（AC-079）：新增 `_protected_versions`，清理时保护在用/候选版本。
  - `algo/decision/localization.py`、`backend/api/region_review.py`、`ui/widgets/region_review_dialog.py`（AC-064）：redo、四边拖拽、整框平移。
- 测试：新增 `tests/smoke/test_final_sim_{ui,roles,slots,review,loc,video,deploy,cleanup}.py` 共 **31 tests / OK**（ui 4 / 0.67s，roles 3 / 4.96s，slots 3，review 7 / 0.88s，loc 4 / 3.0s，video 3 / 1.1s，deploy 4 / 15.5s，cleanup 3 / 0.06s）；另跑 gauge_contract 8 项、复核回归 29 项 OK；全量 smoke **485 tests / 147.314s / OK**。证据：`review/task/final_sim/`、`review/task/final_smoke/`。
- 数字：AC-065 冻结后处理 `pp-2f54c2e670`（mask_thresh 0.4 / min_area 4 / work_side 512），dev recall 0.923 / precision 0.970 / misloc 0.056，test small recall 0.833，后处理 6.35ms；AC-078 3 域×3 种子，最差域 pcb 强缺陷 0.889、弱缺陷 0.333；AC-079 dry-run 136 项、0 删除；AC-076 19 个 pkl 0 加载错误，但 defect_clf sklearn 1.9.1≠环境 1.9.0，锁文件与环境不一致（torch 2.5.1+cu121 vs 2.14.0+cu126、numpy 2.4.6 vs 2.5.2、Python 3.11 vs 3.12）；AC-073 三角色越权 401/403，发布调用 0 次。
- 建议状态：已实现 051/056/057/064/065/075；078 未实现→部分实现；部分实现 036/037/039/043/073/076/077/079/080；未实现 060、062（062 见 §89 冲突说明）。
- 边界：全部为合成数据/离屏模拟；锁文件不一致未修，只记录。
- 模拟验证，生产验收 **0**。待同步 SYNC-28。

## 86. final_perf 模拟验证组：上传延迟/候选显存/冷启动（2026-10-13，开发对话，生产未验收）

> 注：本节与上方“86. PCB-ACCEPT”重号，引用写作 V§86(final_perf)。

- 改动：
  - `backend/pipeline/service.py`：`detect_pil` 新增 `raw: bytes`，`_cv2_decodable` 时直接落盘原始 bytes；`_finish_detect` 透出 `decision_trace`。
  - `backend/api/routes_detect.py`：传 `raw=data`。
  - `algo/ssocl/candidate_budget.py`：`vram_fraction=0.85`、`max_vram_bytes`、`CandidateDeferred`。
  - `backend/engine/__init__.py`：OOM 转 deferred；`COLD_START_BUDGET_DEFAULTS`（load 30s / replay 10s / first 5s / warm 1s）、warmup、`cold_start_stats`。
  - DIFF `tests/smoke/test_async_persist.py`：EXPECTED_KEYS 同步新增字段。
- 测试：新增 `tests/smoke/test_final_perf.py` **13 tests / OK**；回归合计 88 项 OK，test_cde_report+explain 17 项 OK；全量 smoke **485 / OK**。证据：`review/task/final_perf/`（bench_*.json）。
- 数字：AC-066/007 upload p50/p95 1086.7/1971.0→513.9/894.3ms；空闲 svc_e2e p95 609.0ms，纯推理 p95 428.6ms，负载 p95 1163.6ms，流水吞吐 ×1.239；200ms 预算本机不可达。AC-070 候选峰值 127.7MiB / 上限 5222.1MiB。AC-071 gold_finger load 549.2 / first 314.0ms，smt load 1547.8 / first 216.7ms。
- 建议状态：066/007 部分实现；069 部分实现（未改）；070 已实现（候选路径）；071 已实现。
- §65 遗留：降级 slot 分数恒定不是 bug（67 张“正常图”为同一 MD5）；decision_trace 缺失已修。
- 边界：本机 CPU/GPU 下测得，非产线工控机；200ms 未达标。
- 模拟验证，生产验收 **0**。待同步 SYNC-28。

## 87. final_learning 模拟验证组：旧在线学习封禁/迟到标签/保留期/热图重算（2026-10-13，开发对话，生产未验收）

- 改动：
  - `backend/engine/__init__.py`：`LegacyOnlineLearningBlocked`、`legacy_online_guard`，`get_engine` 默认开启封禁。
  - `backend/self_learning/service.py`：透传参数；`label_lateness`/`late_label_stats`，超时标签标 `late_excluded`。
  - `backend/api/region_review.py`：`POST /detections/{id}/heatmap/recompute`。
  - `backend/api/routes_system.py`：retention preview/execute、late_labels。
  - NEW `backend/core/retention.py`。
- 测试：新增 `tests/smoke/test_final_learning.py` **13 tests / OK**；含相关回归共 74 OK，test_async_persist 5 项 OK；全量 smoke **485 / OK**。证据：`review/task/final_learning/`。
- 数字：保留期测试 TEMP 文件 mtime 设为 40/200/120 天；热图复算错误码 404/409/403/410。
- 建议状态：已实现 038、072；部分实现 063（API 有、UI 未接）、019（holdout 开关存在，默认仍 self）、059（间接）；041/042、044～050、052～054、058、068 本轮无进展，维持原状态不降级。
- 边界：保留期 execute 仅在 TEMP 库验证；pipeline/service.py 本组未改。
- 模拟验证，生产验收 **0**。待同步 SYNC-28。

## 88. final_release 模拟验证组：候选发布门禁链（2026-10-13，开发对话，生产未验收）

- 改动：
  - `backend/self_learning/candidate_release.py`：`business_policy`、Wilson CI、`looser_than_business`、`check_recovery_authorization`、release_audit 哈希链、`_check_eval_run`、`snapshot_prepare/commit`。
  - `backend/engine/__init__.py`：recovery_authorization 接线。
  - `backend/self_learning/service.py`：调用接线。
  - DIFF `tests/smoke/test_representation_rebuild.py`；NEW `tests/smoke/test_final_release_chain.py`。
- 测试：test_final_release_chain **9 tests / 8.5s / OK**；回归 97 passed、38 subtests / 42.4s；全量 smoke **485 / OK**。证据：`review/task/final_release/`。
- 数字/错误码：AC-067 `POLICY_LOOSER_THAN_BUSINESS`、`POLICY_MAX_FPR_CI`/`POLICY_MIN_RECALL_CI`、`EVAL_RUN_MISSING`/`BINDING_MISMATCH`/`PARENT_STALE`/`POLICY_CHANGED`，每类 ≥80 样本时强制 Wilson；AC-073 `STALE_AUTHORIZATION_REQUIRED`/`INVALID`/`SCOPE`/`EXPIRED`。
- 建议状态：067 已实现（模拟）；073 子缺口闭环、整体仍部分实现；074/040/055 部分实现（有增量）；061 部分实现；011～016、028～035 维持。
- 边界：V§81 真实 FPR 100% 的精度问题不因门禁而改善，门禁只保证不会放行。
- 模拟验证，生产验收 **0**。待同步 SYNC-28。

## 89. final_model 模拟验证组：parity harness 与方案比较协议（2026-10-13，开发对话，生产未验收）

- 改动（仅新增）：
  - NEW `algo/features/parity.py`、`algo/decision/scheme_compare.py`。
  - NEW `tests/smoke/test_final_model_parity.py`（5 项）、`tests/smoke/test_final_model_protocol.py`（9 项）。
- 测试：相关合计 **54 tests / ~18s / OK**；全量 smoke **485 / OK**。证据：`review/task/final_model/`。
- 数字：AC-009 三方案块数 0/24/6；AC-006 4000 张合成图 Šidák 校正。
- 建议状态：已实现 004、008（离线口径）、018、020、021（准入逻辑）、022；部分实现 001/002/003/005/006/009/010/017/023/026/061。
- AC-062 冲突说明：final_model 建议“机制已实现”，final_sim 建议“未实现”；结合 V§83 真实比对 new_candidate，取保守 **部分实现**（harness 机制具备，无真实工件通过）。
- 模拟验证，生产验收 **0**。待同步 SYNC-28。

> 全量 smoke 第二轮（含 §90/§91 新增 21 例）：**Ran 506 tests / 154.213s / OK**（`review/task/final_smoke/stderr_r2.txt`）。第一轮 485 / 147.314s / OK。

## 90. round2_learning 模拟验证组：热图复算 UI/隔离默认 auto/标签时序审计/标签身份/处置汇总（2026-10-14，开发对话，生产未验收）

- 改动：
  - `ui/widgets/region_review_dialog.py`（AC-063）：`build_recompute_payload`、`RECOMPUTE_ERRORS`；画布只读预览层 `set_preview_boxes/preview_items`；rel_thresh/min_area/morph_kernel 输入框 +「热图复算(只读预览)」「清除预览」按钮，调 `POST /api/detections/{id}/heatmap/recompute`；不改区域与修订号。
  - `algo/eval/offline.py`（AC-019）：纯函数 `resolve_isolation`；默认 `isolation` self→**auto**（fit ≥ `holdout_min_fit`=16 走 holdout，ratio 0.2、min 4、seed 固定，否则回退 self 并记原因）；决策写入 `calibration.isolation_resolution`。
  - NEW `backend/self_learning/label_audit.py`（AC-042/044）：`audit_label_order`、`LabelRecord`、`resolve_labels`。
  - `backend/self_learning/service.py`（AC-042）：只读 `label_order_audit(category)`。
  - `backend/api/open_alerts.py`（AC-052）：只读 `GET /api/review/open_alerts/summary?category=`。
- 测试：NEW `tests/smoke/test_round2_learning.py` **9 tests / OK**；定向回归 88 + 60 用例 OK；改动文件 py_compile 通过；全量 smoke 第二轮 **506 / OK**。证据：`review/task/round2_learning/`。
- 数字：AC-019 n=100→holdout 80/20，n=10 回退 self，边界 20/19；AC-063 失败码 409/403/410/404/None 均清空预览；推理链路未改（单图耗时不变为估算，未实测 GPU）。
- 建议状态：063 已实现（待真机）；019/042/044/052 部分实现；041、047～050、053、054、058、068 维持部分实现；045/046 维持未实现。
- 边界：auto 下 fit 集少约 20%，需真实数据确认不退化；已有快照需重建才生效；042 未接入候选构建过滤；044 未落 DB；052 缺 UI 面板。
- 模拟验证，生产验收 **0**。待同步 SYNC-29。

## 92. 稳定版第 8 轮同步验收（SYNC-28～29，2026-10-14，本对话）

- SYNC-28：29 文件备份+复制、哈希一致（`sync_backup\SYNC-28\`）。**批次遗漏 `tests/smoke/test_async_persist.py`**（service.py 新增 decision_trace 响应字段，SYNC-24 的 EXPECTED_KEYS 契约测试 2 项 FAIL）→ 补同步后定向 80/80 OK（44.3s）。全量 smoke **485 tests / 141.900s / OK**。
- SYNC-29：8 文件备份+复制（`sync_backup\SYNC-29\`）。**批次遗漏 `algo/feature_revision.py`**（release_ops 依赖 `production_pointer`/`lineage`，首轮定向 2 errors）→ 补同步后定向 21/21 OK。全量 smoke **506 tests / 135.809s / OK**（与开发版 506 一致）。
- 附带确认：final_sim_deploy 场景③离线缺权重 fail-fast 在稳定版通过——TEST-05（V§82）的"缺权重挂起"FAIL 子项已由 SYNC-28 的 `algo/backbone/dino.py` 修复并经模拟验证（真实断网复测未重做）。
- 流程教训（转开发）：连续两批遗漏受影响文件，建议开发对话在 SYNC 批次登记前跑"全量 smoke 通过文件清单自洽检查"（本对话两次都是全量 smoke 兜底发现的）。
- 回滚注意：test_async_persist.py 稳定版旧版（SYNC-24 时点）未留存备份；如需回滚 SYNC-28 的该文件需重建 SYNC-24 时点内容。
- 生产验收 **0** 不变。

## 91. round2_release 模拟验证组：release_ops 运维函数与 revision 绑定（2026-10-14，开发对话，生产未验收）
- 改动：
  - `algo/release_ops.py`（**未跟踪新文件**，此前无引用、无测试）：修复 `SNAPSHOT_REQUIRED` 键名 `python/torch/numpy`→`python_version/torch_version/numpy_version`（原先真实快照恒判 incomplete）；新增 `revision_binding(store, model_snapshot)`（AC-061）。
  - NEW `tests/smoke/test_round2_release.py`（12 例，`[i/12]` 进度）。
- 测试：test_round2_release **12 passed / 7.9s**；相关回归 55 passed + 36 subtests（22s）无回归；全量 smoke 第二轮 **506 / OK**。证据：`review/task/round2_release/`。
- 数字：覆盖函数 `group_selection_plan`、`stamp_profile_lineage`/`profile_completeness`、`trad_protocol_audit`、`box_attribution_audit`、`representation_change_plan`、`feature_ops_view`/`submit_toggle`、`memory_asset_inventory`、`capability_card`、`admission_registration_gate`、`snapshot_completeness`、`orchestration_table`，12/12 通过。
- 建议状态：已实现 013（文档化审计）/028/029/030/031/032；基本实现（清单记部分实现）011/012/014/015/016/033/034/035；部分实现 060（未实现→部分）、061。
- 边界：release_ops 全为离线只读/登记函数，未接 backend API 与 UI；AC-061/035 闭环需生产 trad 纳入 FeatureSetRevision 管理并在 `persist.write_snapshot_state` 调用 `revision_binding`（本轮按约束未做）。
- 模拟验证，生产验收 **0**。待同步 SYNC-29。

## 93. PCB-ACCEPT-05-r2：验收配置入口增量同步（验收对话，2026-10-14）

- 背景：开发对话在协同.md §4 登记 PCB-ACCEPT-03 配置入口交付（acceptance settings）；验收侧重跑 `raw\diff_pcbdual.py` 发现新增量。
- 增量：2 DIFF（`PCB_Dual\app\config.py`、`app\ui\settings_dialog.py`）+ 1 NEW（`tests\test_acceptance_settings.py`）。
- 发现并补漏：同步后稳定版 `test_acceptance_settings` 1 error（`rule_snapshot()` 缺 `acceptance_policy`）——根因是 diff 脚本 `EXCLUDE_DIRS` 含 `"template"`，把代码目录 `app\template\` 当数据目录排除，导致 PCB-ACCEPT-05 首轮漏同步 `app\template\model.py`。已补同步该文件（备份 `sync_backup\PCB-ACCEPT-05-r2\`）。**教训：diff 排除规则会误伤与数据目录同名的代码目录，后续比对结果须人工复核被排除的路径清单。**
- 同步后哈希核对：4 文件开发版/稳定版 SHA-256 全部一致。
- 稳定版复验：`test_acceptance_settings` + 既有回归（test_pcbdual04 / adapter_batch / shift_batch / tuner_preview / async_template_snapshot / template_delete_cascade）**39 tests / 3.211s / OK**。
- 口径确认：acceptance_policy 仅用于评估/报告，不传入传统 CV adapter，不改变产线 OK/NG 判定；99%/1%/0.5 仍为候选工程门禁，PCB-ACCEPT-03 精度 PASS 仍 BLOCKED（缺 GT 与用户冻结阈值）。
- 生产验收 **0**。

## 94. round3_speed 模拟验证组：blob 无损改写 + GPU DoG 开关 + 离线 worker 数（2026-10-14，开发对话，生产未验收）

- 改动：
  - `algo/slots/blob.py`（HARD-01/AC-066）：CPU 路径无损改写（判定与分数不变）；新增 GPU DoG 开关（env `AOI_BLOB_GPU` / cfg `gpu`，**默认关**）。
  - `algo/eval/offline.py`：`_cpu_exec` worker 数改为 `max(2, min(4, cpu//2))`。
  - NEW `tests/smoke/test_round3_speed.py`（6 项）。
- 测试：test_round3_speed 6 项 + 相关回归，合计 **66 项通过**。证据：`review/task/round3_speed/`。
- 数字（TEMP 模型，单品类小样本）：gold_finger P95 基线 397.7 → 默认无损 316.8ms（−20%，仍超 200ms）→ 开 GPU 开关 **181.8ms**（达标；26/26 判定一致，Δscore 0，raw Δ≤7.45e-8）；smt_solder P95 69.2→63.6ms。
- 口径：TEST-01 的 204.1ms 是 TEMP 新模型 2500² 图 e2e；V§86(final_perf) 的 428.6ms 是生产快照原尺寸大图纯推理，两者不可比。
- 建议状态：HARD-01/AC-066 部分实现（默认路径仍未达标，GPU 开关路径小样本达标）。
- 边界：GPU 开关默认关，原因是 U22 曾记录 AUROC 退化（0.7075→0.6112），需在全量集上复核 AUROC 后才可开启（设置页决策键 `blob_gpu_enabled`，见 §97）；原生 2500 产线条件、并发未测。
- 模拟验证，生产验收 **0**。待同步 SYNC-30。

## 95. round3_decisions 模拟验证组：人工决策前端化（governance_decisions）+ 冻结 + 依赖锁（2026-10-14，开发对话，生产未验收）

- 改动：
  - `backend/core/config.py`：`GOVERNANCE_DECISIONS_SPEC`、`FROZEN_THRESHOLD_KEYS`、`Settings.governance_decisions()`。
  - `backend/api/routes_system.py`：配置白名单新增 11 项；冻结后改 `g2_max_fpr_pct/g2_min_recall_pct/g6_max_old_recall_drop_pt/max_gray_ratio_pct/gt_dataset_ref` 返回 409 `THRESHOLD_FROZEN`；`/system/info` 新增 `governance_decisions`、`dependency_lock`。
  - `backend/api/routes_candidates.py`：`candidate_release_mode=auto_on_approve` 支持；激活前依赖锁检查（block 时 409 `DEPENDENCY_LOCK_MISMATCH`，warn 时附 `dependency_lock_warning`）。
  - `backend/engine/__init__.py`：`apply_governance_isolation`（`calibration_isolation` 注入 cfg）。
  - `ui/pages/settings_page.py`：「待决策事项」区块（admin 可改）。
  - NEW `backend/core/deplock.py`（依赖 `requirements-lock.txt`）、NEW `tests/smoke/test_round3_decisions.py`（12 项）。
  - 补充：`tests/smoke/test_candidate_audit_log.py` 断言改为只检查 `state`（warn 模式会附带 `dependency_lock_warning`）。
- 测试：**33 passed**（新 12 项 + 回归）。证据：`review/task/round3_decisions/`。
- 决策键与默认值：`gt_threshold_frozen=false`、`gt_threshold_frozen_at`（只读）、`gt_dataset_ref=""`、`ac045_l1_real_training_enabled=false`、`ac046_l1_pollution_exit_enabled=false`、`candidate_release_mode=manual`、`calibration_isolation=auto`、`dependency_lock_policy=warn`、`ac062_legacy_artifact_dir=""`、`sync28_29_accepted=false`、`production_acceptance_signed=false`。
- 建议状态：AC-067/072 维持已实现（补冻结开关）；AC-019 部分实现（auto/self/holdout 可在设置页配置）；AC-045/046 维持未实现（仅登记开关，默认关、无运行时行为）；AC-076 补依赖锁检查（部分实现）。
- 边界：AC-070 本轮未处理；默认值全部为“未决策”保守值，需用户在设置页确认；`production_acceptance_signed` 仅为登记，不替代验收框。
- 模拟验证，生产验收 **0**。待同步 SYNC-30。

## 96. round3_accuracy 模拟验证组：校准软限幅替换硬钳位（2026-10-14，开发对话，生产未验收）

- 改动：`algo/fusion/calibrate.py`（HARD-02/G2）：`_soft_bound` 替换 `np.clip(q,-1,2)`，消除 2.0 钳位饱和造成的平局；NEW `tests/smoke/test_round3_accuracy.py`（4 项）。
- 测试：**57 例 + 64 subtests 通过**。证据：`review/task/round3_accuracy/`。
- 数字（TEMP/TEST-07 口径）：baseline fused AUROC 0.935→**0.970**；固定阈值 FPR 仍 **1.00**；FPR≤5% 下最大 recall 0→**0.80**；shead cal AUROC 0.50→1.00；domain 场景 AUROC 0.784→0.703（只剩 disc 单槽排序，旧值来自平局的偶然收益）。
- 根因：train/good 与 test 正常样本来自不同采集批次（Image_2026、t_fov），阈值域覆盖不足，阈值落在全部 test 正常样本得分之下。次要发现：sem 方向反转（raw AUROC 0.30）；sanity 将 shead 权重置零。
- 建议状态：HARD-02/G2 部分实现（排序能力改善，固定 FPR 未达标，**blocked on data**）；AC-067 门禁维持已实现。
- 边界：需把新采集批次正常样本纳入 good 集并重建（见 §97 `approved_extra_good_batches`）；sem 方向与 sanity 置零未修。
- 模拟验证，生产验收 **0**。待同步 SYNC-30。

## 97. round3_coverage 模拟验证组：批次覆盖检查 + 预算工作点 + blob GPU 决策键（2026-10-14，开发对话，生产未验收）

- 改动：
  - `backend/core/config.py`：`GOVERNANCE_DECISIONS_SPEC` 新增 `blob_gpu_enabled`（默认 false）、`approved_extra_good_batches`（登记项）。
  - `algo/slots/blob.py`：`set_gpu_override`；`backend/engine/__init__.py` 注册 `governance_blob_gpu_enabled`；env `AOI_BLOB_GPU=0` 强制关闭优先。
  - NEW `algo/eval/coverage.py`：`infer_batch`、`batch_coverage`、`max_recall_at_fpr`、`budget_working_point`。
  - `backend/evaluation/service.py::run_accuracy_eval`：输出 `batch_coverage`、`budget_working_point`。
  - `ui/pages/eval_page.py`：批次覆盖告警与参考工作点显示。
  - NEW `tests/smoke/test_round3_coverage.py`（8 项）；`test_round3_decisions.py` 的 `GOV_DEFAULTS` 同步。
- 测试：**Ran 40 / OK**。证据：`review/task/round3_coverage/`。
- 建议状态：HARD-02 部分实现（评估侧能提示批次未覆盖并给出预算工作点，不改判定）；HARD-01 GPU 开关可在设置页决策。
- 边界：未在真实快照上验证；`approved_extra_good_batches` 只登记，不自动入训，需人工重建。
- 全量 smoke（round3 四组 + test_candidate_audit_log 修改后）：**536 tests / 135.936s / OK**（`review/task/final_smoke/stderr_r5.txt`，506→536，+30）。
- 模拟验证，生产验收 **0**。待同步 SYNC-30。

## 98. round4_sem_shead 根因诊断组：sem 方向反 / domain 下 shead 权重置零（2026-10-14，开发对话，生产未验收）

- 结论：**不是代码 bug，未改算法代码**。
  - sem：打分链路为距离语义（`algo/slots/sem.py`：`dist = 1 - max(cos(patch, bank))`，topk 均值，越大越异常）；`_slot_image_score` multiscale 聚合与 `CDFCalibrator` 均单调递增，不翻转方向。AUROC 0.30 来自数据与协议。
  - shead：权重置零由 `weight_mode=sanity` 的 `w=max(0, auc-0.5)` 触发，属有意逻辑（U26 已警示域差品类 sanity 不可外推）。
- 根因（数据结构）：gold_finger 每张正常图都是某张缺陷图的 `*_repaired` 孪生图，仅缺陷区域不同；init_defect（train_anomaly，0598-0621、1289-1307）的孪生图都在 train/good。
  - sem：孪生图在库中时缺陷图≈库内成员，domain 下此类缺陷 sem 均值 0.400，低于留出校准正常图 0.509；孪生图不在库中的 5 张为 0.484；sem sanity 仅 0.095。test 正常图来自 Image_2026 批，缺陷 1308-1334 与库内 1289-1307 同批，sem 实测的是批间距离。同库配对检验（缺陷原图 vs 自身 repaired）：domain 75%、baseline 60% 配对中缺陷图更高分，方向正确但信号弱（均值差 0.0125 / 0.0003）。
  - shead：domain 下 36 张正常图 holdout（7 校准 / 29 拟合）；30 张 init_defect 中 25 张孪生图在负样本训练集，DeviationLoss 不收敛（loss_n≈5.9，loss_a≈6.1）；init_defect 均值 2.57 vs 校准正常 2.56，sanity 0.38→权重置 0；而 test 上 shead AUROC 0.956。baseline 下 14 张正常图回退 self，sanity 含自匹配偏乐观（0.819）。
- 数字（seed 42，固定切分；baseline test=20，domain test=37）：
  - baseline：权重 sem/disc/shead .361/.278/.361；sanity .819/.745/.819；fused AUROC 0.970；FPR≤5% recall 0.80；等权对照 0.970 / 0.80；逐槽 test AUROC 0.300/0.970/1.000；p50/p95 167/307ms。
  - domain：权重 0/1/0；sanity .095/.643/.383；fused AUROC 0.703；FPR≤5% recall 0.00；等权对照 0.682 / 0.20；逐槽 test AUROC 0.309/0.703/0.956；p50/p95 172/293ms。
  - 评估过 domain 改等权：AUROC 0.703→0.682、recall 0→0.2，有升有降，未引入开关。
- 改动：NEW `tests/smoke/test_round4_sem_shead.py`（4 项，合成特征，CPU，不加载 DINO）；诊断脚本 `review/task/round4_sem_shead/diag.py` 与 `raw/`。
- 测试：`test_round4_sem_shead` + `test_round3_accuracy` **8 OK**。证据：`review/task/round4_sem_shead/`。
- 建议状态：HARD-02/G2 维持部分实现（blocked on data）。
- 边界：真正有效的改法是 sanity 与 shead 训练中按 stem 剔除孪生图已在正常库中的 init_defect，属数据协议改动，留作后续建议。
- 模拟验证，生产验收 **0**。待同步 SYNC-31。

## 99. round4_decisions_apply 决策推荐值落地组：governance_decisions 写入开发版 default.yaml（2026-10-14，开发对话，生产未验收）

- 运行时配置：`Settings` 读 `AOI_CONFIG`，未设置时读 `AOI_ROOT/configs/default.yaml`；当前未设 `AOI_CONFIG`，开发版实际读 `AOI_Core/configs/default.yaml`（设置页保存也写回此文件）。
- 改动：仅在 `configs/default.yaml` 末尾追加 `governance_decisions` 段；未改 config.py、生产 DB、快照、`configs/docker.yaml`。
- 开发方代定 7 项：`dependency_lock_policy=warn`、`candidate_release_mode=manual`、`calibration_isolation=auto`、`ac045_l1_real_training_enabled=false`、`ac046_l1_pollution_exit_enabled=false`、`blob_gpu_enabled=false`、`approved_extra_good_batches="Image_2026,t_fov"`（str，逗号分隔；写成 YAML 列表会类型校验失败回落为空）。
- 保留人工 6 项（未写入，保持默认）：`gt_threshold_frozen=false`、`gt_dataset_ref=""`、`gt_threshold_frozen_at=""`（只读）、`sync28_29_accepted=false`、`production_acceptance_signed=false`、`ac062_legacy_artifact_dir=""`。
- 验证：`get_settings().governance_decisions()` 7 项与写入一致、其余 6 项为默认；`parse_batch_list` 为 `['Image_2026', 't_fov']`；`test_round3_decisions` + `test_round3_coverage` **20 OK**（测试用临时 yaml，不受 default.yaml 影响）。证据：`review/task/round4_decisions_apply/`。
- 边界：`configs/default.yaml` 修改前在 git 中已有他人未提交改动，本次只追加末尾一段；该段是开发版本地决策，稳定版是否采用由验收对话决定。
- 模拟验证，生产验收 **0**。待同步 SYNC-31。

## 100. round4_batch_rebuild 模拟验证组：新批次纳入 good 重建基线（2026-10-14，开发对话，生产未验收）

- 改动：
  - `algo/eval/coverage.py`：NEW `split_extra_good(paths, approved, seed=0, fit_ratio=0.5)`（排序去重→按 `infer_batch` 分组→`default_rng(seed).permutation`→floor(n·0.5) 作 fit，其余 eval）；`batch_coverage` note 更新。
  - `backend/api/routes_models.py`：NEW `_approved_extra_fit_paths`、`_stage_approved_extra`（fit 子集复制到 `storage/images/{cat}/approved_extra/train/good/{stem}_x{sha1[:8]}{suffix}`，幂等，`infer_batch` 不变）；`_prepare_engine` DB 分支追加副本进 init_normal 并从 val 剔除 fit 原图；登记为空时行为不变。
  - `backend/evaluation/service.py::run_accuracy_eval`：登记非空时以全部 test/normal 为划分池（不受 limit 截断），剔除 fit 子集防泄漏。
  - `backend/api/routes_system.py`：`approved_extra_good_batches` 说明与 `_GOV_IMPACT` 文案（下次重建生效，不自动触发重建）。
  - NEW `tests/smoke/test_round4_batch_rebuild.py`（6 项）。
- 修复的 bug：`approved_extra_good_batches` 在重建时实际不生效——test/good 路径直接并入 init_normal 会被 `guard_bundle` 拒绝（路径须含 train 段），原 fake-engine 单测覆盖不到；现经 staging 修复。
- 划分（seed=0，未调整）：fit 8 张（6 张 Image_2026 _repaired + t_fov_26、t_fov_6），eval 9 张（6 张 Image_2026 + t_fov_3/13/14）；base_normal 14 张；E1 缺陷 10 张；E2 缺陷 9 张（与 eval 正常图同名的缺陷原图）；orig 正常 32 张。详见 `raw/split.json`。
- 数字（eval 正常仅 9 张，FPR≤5% 等价 FP=0）：
  - B 组（init_defect=[]，主结果），tau_high 0.895→0.663：E1 AUROC 0.7556→0.6222；E2 0.5432→0.5432；E1+E2 0.6550→0.5848；新批次 eval 固定阈值 FPR **0.556→0.333**；E1+E2 recall 0.79→0.26；wp_recall 0.158→0.105；原批次正常 FPR 0.9375→0.50；Image_2026 / t_fov FPR 0.333→0.167 / 1.0→0.667。
  - A 组（init_defect=30 张，生产口径），tau_high 0.828→0.749，权重 0.36/0.28/0.36→0.25/0/0.75：E1 AUROC 0.6556→1.0；E2 0.5309→0.5802；E1+E2 0.5965→0.8012；固定阈值 FPR 1.0→0.333；wp_recall 0→0.526；原批次正常 FPR 1.0→1.0；Image_2026 / t_fov FPR 1.0→0.0 / 1.0→1.0。
  - E1 AUROC=1.0 反映批次被分开（缺陷图与原批次正常图为原图/修复图配对），不是判别缺陷；**成对难例 E2 AUROC 仅 0.54～0.58，接近随机，瓶颈是模型对缺陷不敏感**，单靠补批次重建解决不了精度。
  - 两次运行指标与阈值完全一致。纯推理 p95（n=240）：B 256.2→254.6ms，A 400.7→257.3ms；n=60 的 +4～5% 属噪声；单张 max 偶发 2.0～2.6s（冷缓存/GC），与重建无关。
- 测试：新增 6/6 通过；回归 47/47 通过；改 routes_system 文案后 round4/round3_coverage/round3_decisions 重跑 26/26 通过。证据：`review/task/round4_batch_rebuild/`。
- 建议状态：HARD-02/G2 维持部分实现（登记→重建路径已接通，固定阈值 FPR 未达标，精度瓶颈在模型缺陷敏感度）。
- 边界：生产 DB 分支 `_prepare_engine` 未用真实 engine 端到端跑过（模拟脚本复现同一 bundle 组装逻辑）；`datasource_id` 分支未接 approved_extra；eval 样本小（正常 9、缺陷 19），置信区间宽。
- 全量 smoke（round4 三组后）：**549 tests / 154.231s / OK**（`review/task/final_smoke/stderr_r6.txt`，536→549，+13）。
- 模拟验证，生产验收 **0**。待同步 SYNC-31。

## 101. PCB-ACCEPT-01 第三轮：TH 交互模式真实桌面验收（验收对话，2026-10-14）

- 方法：TEMP 隔离树（`review\task\pcb_accept_01\temp_tree\PCB_Dual`，PCBDUAL_PORT=8023）真实桌面（非 offscreen，1920×1080）；新建两个 draft TH 模板（engine_mode=traditional、尺寸门禁关闭）：`TH_templ2_single`（标准图 templ_2，100×100 单焊点）/`TH_templ3_dual`（templ_3，262×124 双焊点）；真实 paired 测试图 2-1(OK)/2-8(NG)/3-1(NG)/3-2(NG)，哈希见 `th_assets\manifest.json`。产物 `raw\th_round\`。
- **场景 A（0 框自动轮廓 + OK 图）PASS**：UI OK·无缺陷·157ms；盘内四类真实执行（36–50ms）；连锡禁用，API 证实 SKIP + 原因「连锡需要在「标定」中框选至少 2 个焊点区域」。
- **场景 B（0 框 + NG 图）PASS**：NG 1 项·204ms，缺陷行「少锡 3,6 111×118 tht_少锡 ROI=未提供 NG 51ms 分数不可用」，缺陷框叠加正确。附带发现：调试页尺寸门禁只读全局 `gate_enabled` 不读模板级 gate（见问题③）。
- **场景 C（单框标定）PASS**：建模页鼠标拖框成功，标定文件 pads=1（solder_mode=roi）；重跑 NG 图盘内四类仍执行、连锡仍 SKIP。
- **场景 D（清空恢复自动模式）FAIL（真实缺陷）**：清空后 pads=[] 但 legacy `roi` 键残留，`get_th_roi` 回退逻辑（th_roi.py L52-56）把旧框复活，pad_count 仍=1、模式停在 roi，无法恢复 ellipse 自动；UI 发布检查显示「未框选」与算法实际行为不一致。
- **场景 E（双框连锡）PASS**：画 2 框后连锡启用、其余 4 项禁用（抽屉标题「插件焊点检测·连锡」）；test_3-1/3-2 均连锡 NG（桥接框 249×124 / 247×122）；API 证实其余 4 项 SKIP 带多框原因「当前标定了多个焊点，算法只会跑连锡；盘内缺陷请改用单框或自动锡面」。
- **场景 F（禁用与原因提示）PASS**：三模式禁用状态与原因串经 UI 树+探针双重核对，无伪装全绿。
- 发现的问题（转开发待修）：
  1. **场景 D 清空复活缺陷**（regions.py L115-123 清空时未删 legacy `roi` 键 + th_roi.py L52-56 回退复活）。
  2. **调试页切换模板用空 calibration 覆盖清空未保存的 legacy 标定文件**（template/store.py L364-368 sync_calibration_to_legacy），本轮双框标定曾被静默清空，需「保存模板」后才正常——数据丢失风险。
  3. 调试页尺寸门禁不随模板（只读全局 gate_enabled）。
  4. API 直跑（allow_stub=false）时未就绪项报 ERROR/NOT_READY 而非 SKIP（与 UI 口径不一致）。
  5. 建模页「关联」标签清空后不刷新。
- 未覆盖：tooltip 悬浮截图（UIA 不暴露）、REVIEW 状态触发、连锡调参联动、归档路径。
- 隔离核对：应用关停后端口 8023 释放；稳定版 `stable_AOI_tree\PCB_Dual` 的 storage/templates 前后快照（含 SHA1）逐字节一致，零修改。
- 生产验收 **0**。

## 102. 稳定版第 9 轮同步验收（SYNC-30～31，2026-10-14，本对话）

- 同步前全树比对（`review\task\final_smoke\diff_aoicore.py`，排除数据/缓存）：DIFF 21 + NEW 11。其中批次登记文件 20 个；**8 个未登记源码差异**（特征工程页/feature parity 工作：main_window、api_client、dashboard_page、offscreen、test_final_sim_roles、test_display_order_equivalence DIFF + feature_page、test_feature_parity NEW）——未纳入本批，已转开发确认归属（协同.md §4.6）；`configs\default.yaml` 按 SYNC-31 注记不同步；DB 运行时文件与 tests\results 产物排除。
- 合并同步 20 文件：12 DIFF 备份（`sync_backup\SYNC-30\` 11 个 + `sync_backup\SYNC-31\` 1 个，coverage.py 稳定版原本不存在无备份）+ 8 NEW；复制后哈希逐一一致。
- 稳定版定向：**43 tests / 2.754s / OK**（round3 四组 30 + round4 两组 10 + candidate_audit_log 3）。
- 稳定版全量 smoke：**546 tests / 143.941s / OK**（日志 `review\task\final_smoke\stable_sync30_31_smoke.log`）；与开发版 549 的 3 项差额 = 未登记 test_feature_parity，差额可解释、无未说明减少。
- 注意：本轮一次通过，无批次遗漏；SYNC-30 关键数字（P95 397.7→316.8ms、AUROC 0.935→0.970）为开发版 TEMP 小样本，稳定版未复测性能；固定阈值 FPR 仍 100%/0.333 的系统性精度问题未解。
- 结论：SYNC-30～31 同步完成且稳定版复验通过；生产验收 0。

## 103. PCB-ACCEPT-01 第四轮：压力与健壮性场景真实桌面验收（验收对话，2026-10-14）

- 方法：同第三轮 TEMP 隔离树真实桌面；产物 `review\task\pcb_accept_01\raw\stress_round\`（7 截图 + 3 模板 JSON 快照）。
- **场景 1（快速连续调参）PASS**：连续改参不等预览完成，防抖收敛后最终结果（OK 无缺陷·39ms）与最终参数一致；模板 API 快照三次 SHA256 一致——预览/整批全程未污染模板。
- **场景 2（快速切换算法/图片）PASS**：快速切换检测项与测试图后无残留缺陷框（表面异常经三重证据定位为问题①的 spinner 精度显示，非过期结果缺陷）。
- **场景 3（取消预览）PASS（变体）**：调试页无取消按钮，以「整批检测进行中切页」验证——无跨页回填，整批结果正确（直通率 50%）。取消语义本身未覆盖（无入口）。
- **场景 4（关闭运行中窗口）PASS**：检测中关窗，进程退出码 0、端口 8023 释放、关窗后 storage 写入即刻停止。
- **场景 5（窗口不卡死）PASS**：全程无 >5s 无响应。
- 新发现的问题（转开发）：
  1. **阈值 spinner 两位小数精度**：模板真实值 0.005 显示为 0.01，用滑块"调回 0.01"实为改参——显示精度导致用户误判。
  2. **「恢复默认」恢复到算法全局默认 0.15 而非模板配方 0.005**——恢复语义与用户预期不符。
  3. bridge 两张图整批报 [ERR]（未深究，另：导入文件夹不递归子目录）。
- 未覆盖：取消语义（无入口）、归档勾选路径、TH 模板压力场景。
- 隔离核对：应用与 AOI_Core 均退出、8023/8017 释放；稳定版与开发版零修改。
- 生产验收 **0**。

## 104. PCB-ACCEPT-02 补充：Crack 执行复核 + 配准错配构造（验收对话，2026-10-14）

- 方法：稳定版代码只读执行（适配器级，diagnostics=True），不写任何树；产物 `review\task\pcb_accept_02\raw\crack_mismatch_check.{py,json}`。
- **Crack pair1 执行复核 PASS**：cold_solder_pair/pair1 真实图（哈希已记录），源算法与适配器两侧 Crack 诊断均为实际执行（非"不可用：当前判定路径未执行/未导出"），code=0、0 parts 未检出——Crack 可执行分支真实覆盖；NG 正例检出仍 BLOCKED（需人工确认正例）。
- **TH 配准错配构造 FAIL（真实缺陷）**：真实图片构造错配 templ_2（单焊点板 100×100）vs 3-1（双焊点板 250×124），结果：孔洞/多锡包锡/不出脚 **OK（伪）**、少锡 NG（1 框）、连锡 SKIP。错配 align_method=`标准相位/pad_ring warp dx=0.2 dy=2.7 (resp=0.311)`、coverage=0.995、review_reason=null；匹配对（templ_2 vs 2-1）resp=0.819。
- 根因（代码级定位）：① `align.py` L310-317 相位 resp 低于 `phase_min_response` 且 NCC 不达标时仅标记"相位低置信"文案，**仍按原估计位移放行**；② `check_reliability`（pipeline.py L149-170）只门禁位移幅度与 coverage，**不门禁 resp/低置信**；③ coverage 语义是配准有效区与模板锡面 mask 的自身重合度，错配下仍 0.995，**不能识别跨板错配**；④ `phase_min_response=0.12` 相对实测分布（错配 0.311/匹配 0.819）过低。
- 结论：配准质量门禁缺失，错配输入未 fail-closed。已转开发（协同.md「PCB 缺陷转开发」9）。注意：本构造是人为跨板错配，不代表自然产线配准失败分布；自然失败样本仍 BLOCKED。
- 生产验收 **0**。

## 105. REMAIN-14：AC-001～080 + G1～G6 总验收只读审计（草稿，验收对话，2026-10-14）

- 方法：只读审计子 agent，不跑测试、不改台账/协同/验证记录原文；骨架以权威 `commit\repo\review\简明改进建议.md`（AC 编号/主题/状态以其为准），证据引用 V§1～104、台账、协同、task 产物。产物 `review\task\remain_14\`（result.md / env.json / cmd.txt / raw\ac_index.csv）。
- **80 行证据索引**（CSV 校验 80 数据行、AC-001～080 唯一齐全）：
  - 实现状态：已实现 24 / 部分实现 54 / 未实现 2（AC-045/046）——与台账「24 模拟待实机 / 54 部分 / 2 未实现」**完全一致，统计仍准确**。
  - 真实验收状态：FAIL 5（AC-007/062/066/067/069）、PASS/部分 PASS 待复核 7（AC-051/053/056/057/058/070/072）、BLOCKED 1（AC-078）、开发前置缺口 2（AC-045/046），其余模拟待实机或待复核。
  - 生产验收勾选 **0/80**（只有用户能勾）。
- **G 门禁一句话结论**：
  - **G1 时延 FAIL**——TEST-01 三轮 295.8→216.3→204.1ms 均破 200ms；GPU 开关 181.8ms 待 gold_finger 全量集 AUROC 复核前不算通过。
  - **G2 精度 FAIL**——固定阈值 FPR 100%（TEST-07 + 生产 v2 复证），SYNC-31 重建后 0.333 仍远超 ≤5%，瓶颈在模型缺陷敏感度。
  - **G3 恢复/故障 部分**——WAL/RNG/prepared 底座仅 TEMP 模拟，跨进程/崩溃/OOM 故障注入未做。
  - **G4 真实操作 部分**——TEST-04 桌面 PASS + PCB 四轮桌面（含 1 项真实缺陷 FAIL），Gauge 有值态/特征页真实 HTTP 缺。
  - **G5 定位 部分（仅合成）**——冻结后处理 dev recall 0.923 为合成标注，无真实标注定位集。
  - **G6 学习 部分**——六臂仅最小切片，AC-045/046 未实现，真实训练回放缺。
- **口径提示（非计数差异）**：①台账内并存旧口径 17/58/5（2026-10-12 段），引用须带时间戳；②AC-070 权威「已实现」但 TEST-03 仅部分验收（OOM 子项未做）；③AC-075 TEST-05 部分 FAIL 后修复仅模拟验证，真实断网复测未做；④AC-060/062 组间结论冲突已按规则取较新/保守；⑤验证记录 §86 重号（PCB-ACCEPT vs final_perf）。
- **开发前置**：AC-045/046 能力未实现；selfsim RNG 修复对应 8 个未登记差异文件待开发登记批次（协同 §4.6）；PCB 9 项缺陷转开发未修；REMAIN-03 生产特征血缘闭环实现可能不存在。
- 结论：草稿完成，状态转**待复核**（本对话复核 CSV 完整性与结构已通过）；最终汇总待 REMAIN-01～13。生产验收 **0**。

## 107. REMAIN-01：TEST-06 全链路 parity 复核（验收对话，2026-10-14）

- 方法：稳定版只读执行，沿用 TEST-06 同协议同输入（atol=1e-6），八项逐项对比（协议 A）；另以开发版做协议 B 诊断。产物 `review\task\remain_01\`（八项逐项 JSON、双协议日志、脚本副本）。
- **重大前置发现：selfsim RNG 修复不在稳定版 AOI_feature**（SYNC-30/31 全树比对只覆盖 AOI_Core）。稳定版 vs 开发版 4 文件差异：`feature_lib.py`（selfsim 仍全局 `np.random.choice`）、`feature_set.py`（extract_one 无 rng）、`scoring.py`（仍 U27 全局种子包裹）功能性差异，`presets.py` 仅 docstring。已登记协同 §4.7 转开发确认归属。
- **总判定 FAIL（8 项过 2 败 6）**，历史 TEST-06 FAIL 维持、不可改写：
  - PASS：15 组/200 维顺序与 slice（trace 0 错误）、kNN 项的 dtype 子面（float32 一致）。
  - FAIL：提取 max abs diff=14.665（逐组归因 100% 位于 selfsim，其余 14 组逐位一致）；预处理越界图同差值传导；default_rng 候选口径=全局 RandomState+seed(0) 与生产 default_rng(42) 不符；常量图候选 3 个 NaN（生产 0）；kNN 邻居精确一致率 5/6=83.33%（顺序 [5,4,0] vs [5,0,4]，元素重叠 100%）；评分 diff 0.3759；报告随传导不等价。
  - harness status=failed，max_abs_diff=14.665 与历史 TEST-06 同值复现；全局 RNG 前后 4 点快照一致（无泄漏）。
- **协议 B（开发版诊断）**：提取层修复完全有效（15 组×3 探针逐位一致、邻居 100%、0 NaN），但全链路仍 FAIL 3 项——fit 路径 dtype 口径：生产 `TradSlot.fit` 对 float32 栈求 mean/std/bank，候选 `GroupKNN.fit` 用 float64（bank 差 3.255e-03、scoring 差 6.30e-05）。**即使同步当前修复也无法改写 FAIL**，须开发对齐 dtype 后复验。
- 生产验收 **0**。

## 108. REMAIN-02：特征工程页真实 HTTP 联调（验收对话，2026-10-14）

- 方法：稳定版 AOI_Core TEMP 隔离实例（独立 DB/storage，端口 8031），requests 真实 HTTP 直连（非 TestClient/mock），engineer/admin/operator 三角色。产物 `review\task\remain_02\`（55 个请求/响应 JSON、DB/pointer 前后哈希、服务日志、UI 截图×2）。
- **总体 PASS（55 项，FAIL=0）**：
  - 端点×角色矩阵全过：health 豁免；system/info 三角色 200（无/错密钥 401、角色回显正确）；feature-set operator can_edit=false / engineer·admin=true；hard-pool/slot-health/draft/report/rebuild/transition 等 operator 403、engineer 200、admin 按权限制；参数校验面（draft 空 groups/非法 scoring 400、未知品类 rebuild 409）成立。
  - **draft 不改生产**：`changes_production=false, activated=false`，前后业务表哈希零差异、引擎 current.json 指针不变；admin 同 manifest 幂等同 rid。
  - **report/rebuild 不自动激活**：report 后台任务 16s done 指针不变；rebuild 产出未激活 v3（parent=v2，pending_rebuild_version=3），current.json 指针哈希不变。
  - **无生产绑定=unknown/unbound**：revisions 固定 `production_binding="unknown"`，生产指针文件始终不存在。
  - 关联门禁实测：THRESHOLD_FROZEN 409（冻结→拒→解冻→可改闭环）、DEPENDENCY_LOCK_MISMATCH 409（block 策略候选激活被拒且指针未变）。
- UI 对照（离屏+真实 API 响应）：operator 按钮禁用+「仅工程师」、engineer 启用+「已重建 v3，待激活门控（G2）」PASS。**缺口**：真实桌面交互链路未覆盖；transition 正向激活未走通（需合规评估证据，属 G2 流程），仅验证门禁拒绝面。
- 过程发现：首轮 draft/report/rebuild 409 根因=启动钩子 `register_engine_snapshots` 对无检测血缘品类删除 current.json（routes_models.py 204-212 预期保护行为），补 TEMP 血缘后全过。
- 清理：8031 服务停止、端口释放；生产侧 before/after 哈希 **PROD_UNTOUCHED**。
- 结论：真实 HTTP 联调 PASS，待复核；缺口两项保留。生产验收 **0**。

## 109. REMAIN-03：生产 Trad 特征血缘绑定核查（验收对话，2026-10-14）

- 方法：稳定版只读代码审计（文件+行号证据）；按规约"实现不存在则跳过 TEMP 实证、交 FAIL+开发阻塞清单，不虚构绑定"。产物 `review\task\remain_03\`（代码摘录、grep 盘点）。
- **总判定 FAIL——生产绑定闭环不存在**：
  - a. 生产 Trad 零引用 FeatureSetRevision（**FAIL/未实现**）：`algo/feature_revision.py:1` 自述不切换生产 extractor/指针；`test_feature_revision.py:155` 静态契约禁止 trad.py 引用 feature_revision；生产推理特征来源为 `features/registry.py:19-50` 常量注册表；`routes_models.py:281-282` 固定返回 `production_binding:"unknown"`。
  - b. Detection 无血缘字段（**FAIL/未实现**）：`backend/db/models.py:214-245` 无 revision/hash/seed/维度/快照血缘列；唯一写入点 `pipeline/service.py:719-745` 仅落 model_id+n_tiles；`detection_stamp()`（feature_revision.py:298-307）生产零调用。
  - c. 激活/回滚/重启无特征指针对账（**FAIL/未实现**）：激活仅切 current.json 模型指针；`routes_models.py:629-641` check_activation 门禁因 `persist.py:652` 恒写 revision_id=None 成死代码；`bind_production`/`rollback_production` 无 HTTP 入口（仅测试可达）。
  - d. 无 pointer 如实 unknown/unbound（**PASS**，唯一通过项）：feature_revision.py:300-302、persist.py:652-653 注明 `production_trad_not_revision_managed`，不从 registry 猜测。
- 已实现的是「候选特征 revision 前门门禁侧车」（注册→评估→审批→激活门禁、难例准入、parity、指针文件机制），与生产推理/Detection 落库/激活回滚主链路完全断开。
- **开发阻塞清单 6 项**（详见 result.md）：①设计决策先行（现行测试契约固化「trad 不引用 revision」）；②Detection schema 扩展+_persist 写真实血缘；③detection_stamp 接入推理路径；④bind/rollback 补 HTTP 路由；⑤persist.py:652 硬编码 unknown 替换为真实查询；⑥激活/回滚/重启快照 meta.revision_id ↔ 特征生产指针对账（含崩溃恢复）。
- 结论：FAIL，转开发；AC-073/074/076 发布链相关验收在闭环实现前保持 BLOCKED。生产验收 **0**。

## 106. round5 L3 配对模板匹配（gold_finger，2026-10-14，开发对话，生产未验收）

- 改动：`algo/slots/tpl.py` 新增 `pair_mode`（默认 false）和 `resolve_pair_template`（X→X_repaired.*）。L3 下检测图只与显式配对的修复图做配准、差分、top-k 均值，红线改为“L3 显式修复图配对合法，非 L3 仍禁止”。`algo/eval/offline.py` 的 predict/predict_frame 透传 pair_template，新增可选 `pair_auto_resolve`/`pair_dirs`/`pair_tpl_weight`（默认关闭）。`backend/engine/__init__.py` 透传。默认路径数值不变。
- 结果（同 summary_orig evalset，47 张正常 + 47 张缺陷，未挑样本）：tpl AUROC 0.4975→**1.0**，win_rate 0.489→**1.0**，FPR≤5% 时 recall 0.085→**1.0**，与 demo4 s_r 的 1.0 一致。
- 口径：修复图是对缺陷图的像素级编辑。正常图与自身配对时差分为 0；正常图换配另一张修复图（AUROC 0.0），或做重拍扰动（±3px、±4% 亮度、σ=2 噪声，AUROC 0.026）时，正常差分 1.8～2.4，高于全部缺陷差分 0.34～1.65。1.0 仅在模板与检测图像素级同源时成立，真实重拍工况需同工位样本标定后复测。固定阈值无法只用正常图正当标定（跨图 p95=4.63 时 recall 0），未写入默认配置。
- 延迟：配对模式下模板冷提取 p95 786ms（基线 558ms），默认关闭，不影响默认 p95。
- 测试：`test_round5_l3_pair` 4 OK；全量 smoke **553 tests / 138.283s / OK**（549→553，`review/task/final_smoke/stderr_r7.txt`）。证据：`review/task/round5_l3_template/`。
- 建议状态：HARD-02/AC-067 维持原状态（配对口径复现已有精度；真实重拍和固定阈值未达标）。

## 110. REMAIN-01 协议 B dtype 对齐 + REMAIN-03 血缘最小闭环（2026-10-14，开发对话，生产未验收）

> 本节接 V§107（REMAIN-01 FAIL 诊断）与 V§109（REMAIN-03 FAIL 阻塞清单）的修复交付。历史 FAIL 结论不改写，待验收对话复验。

### 110.1 REMAIN-01：协议 B fit 路径 dtype 对齐

- 根因（V§107 协议 B 诊断）：候选 `AOI_feature/scoring.py` 的 `GroupKNN.fit` 用 float64 求 mean/std/bank，生产 `TradSlot.fit` 用 float32，bank 差 3.255e-03、scoring 差 6.30e-05。
- 改动（`AOI_feature/scoring.py`）：`extract_features` 输出从 `.astype(np.float64)` 改 `.astype(np.float32)`；`GroupKNN.fit` 输入显式转 float32，mean/std 用 `dtype=np.float32`，clip/bank 全程 float32。
- 顺带：`algo/features/parity.py` 的 PROTOCOL rng 描述更新为最终协议（"explicit default_rng(42); no global RNG mutation"）。
- 新增测试（`tests/smoke/test_feature_parity.py` 第 4 项）：`test_groupknn_fit_matches_production_float32_bank`——同输入下候选 `GroupKNN().fit(train)` 的 bank/mean/std 与生产 `TradSlot.fit` 的 bank_w/norm_w **逐位完全相等**（assert_array_equal）。
- §4.7 归属确认：AOI_feature 4 文件差异（feature_lib.py/feature_set.py/scoring.py/presets.py）均为本对话交付，已登记 SYNC-33。

### 110.2 REMAIN-03：生产特征 revision 血缘最小闭环

按 V§109 阻塞清单②③④⑤实现（①设计决策未动——现行静态契约仍禁止 trad.py 引用 feature_revision，本批只记录血缘，不改生产提取路径）：

- ② Detection schema：`backend/db/models.py` 新增 `feature_revision_id`（VARCHAR(64)，索引）+ `feature_lineage`（JSON）；`backend/db/database.py` SCHEMA_VERSION 7→8，`_migrate_v8` 用 `_add_col` 轻量加列并建索引。
- ③ 落库接线：`backend/pipeline/service.py` 新增 `_feature_lineage()`——读真实生产 pointer，bound 时带 revision/state/implementation_hashes/config_hash/dimension/extract_seed/parent + lineage 链，unbound/unknown/invalid 如实标注，读取失败不阻塞检测；`_persist` 落库到两个新列。
- ④ HTTP 入口：`backend/api/routes_models.py` 新增 `GET /models/feature-revisions/production/pointer`、`POST .../production/bind`、`POST .../production/rollback`（均 admin 权限、审计日志、409 fail-closed）；feature-revisions 列表透出真实 production_pointer。
- ⑤ persist.py：`_feature_revision_binding()` 替换原恒 unknown——快照落盘时记录真实 pointer（bound/unbound/unknown 三态）。
- `algo/feature_revision.py`：`detection_stamp` 增加 lineage 链。
- 新增测试（`tests/smoke/test_feature_revision.py`）：`test_detection_stamp_lineage`——bind 后 stamp 带 revision/model_snapshot/lineage；snapshot 不匹配标 snapshot_mismatch；无绑定标 unbound。

### 110.3 验证（offscreen，Python 3.11）

- 定向 `python -m unittest tests.smoke.test_feature_parity tests.smoke.test_feature_revision -v`：**Ran 13 tests / 3.242s / OK**。
- parity.py PROTOCOL 描述变更后回归 `python -m unittest tests.smoke.test_final_model_parity -v`：**Ran 5 tests / 11.096s / OK**（激活门禁证据消费、生产自对照八项、fail-closed 拒绝均未破坏）。
- 全量 smoke 首轮复跑 459 tests / 138.378s / FAILED（errors=40）：根因为 `backend/pipeline/service.py` 新增的 `from ...algo.feature_revision import FeatureSetRevision` 相对导入越出顶层包（backend 作为顶层导入时 `...` 越界），导致 30+ 个测试模块 ImportError 级联。修复为项目惯例绝对导入 `from algo.feature_revision import FeatureSetRevision`。
- 修复后全量 smoke（offscreen，`unittest discover -s tests\smoke -v`）：**Ran 555 tests / 141.169s / OK**（549→555，`review\task\final_smoke\stderr_r7.txt`）。

### 110.4 边界

- 协议 B 全链路 FAIL 是否消除须稳定版同步 SYNC-33（含 AOI_feature 4 文件）后按 `review\task\remain_01\` 复验，不预判 PASS。
- 阻塞清单⑥（激活/回滚/重启恢复后血缘与 pointer 一致的 TEMP 实证）未在本轮实现范围，待验收对话执行。
- DB 迁移 v8 未在 TEMP 旧库做升级实证。
- bind/rollback 仅 HTTP+审计，未接 UI。
- 生产验收 **0**。已登记 SYNC-33。
