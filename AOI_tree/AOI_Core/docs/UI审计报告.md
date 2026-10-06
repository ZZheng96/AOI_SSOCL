# UI 全面审计报告

审计范围：`ui/` 全部页面（monitor / data / augment / feedback / model / eval / stats）、
`ui/main_window.py`、`ui/widgets/*`、`ui/api_client.py`，逐一对照 `backend/api/` 路由真实返回结构。
审计日期：2026-08-07。

## 总体结论

- api_client 各方法的返回结构解析（list / dict / items）与后端基本一致；此前 `list_categories`
  的字典串问题属于个例，已修复。
- 所有按钮/菜单均有处理函数；长耗时操作均走 `run_async` / 后台线程 / WebSocket，
  UI 线程无同步 requests 调用。
- 表格空态行普遍正确，且 `setRowCount` 前均有 `clearSpans`（个别遗漏已补）。
- 后端离线时各页面均能容错不崩（离屏冒烟覆盖全部 7 个页面切换）。

## 已修复问题

### 公共 / api_client
1. **`ui/api_client.py`**
   - 新增 `images_tree()`（GET /api/images/tree，静默失败）与
     `prepare_model()`（POST /api/models/prepare）两个端点方法。

### 数据管理页（data_page.py）
2. **新增树状导航**：图像选项卡改为 QSplitter 左右分栏，左侧 QTreeWidget
   （全部图像 → 品类 → 划分 → 标签，节点带数量），点击节点同步顶部筛选下拉并过滤右侧表格；
   树接口失败时静默降级为仅"全部图像"根节点；导入 / 上传 / 删除 / 任务完成后树与表格同时刷新。
3. **MVTEC_DEFAULT_ROOT** 改为 `<PROJECT_ROOT>/data_local`。
4. **上传完成弹窗** `res.get(...)` 在 res 为 None 时会抛 AttributeError，已加 `(res or {})` 容错。

### 模型管理页（model_page.py）
5. **新增【准备模型】入口**：对话框（品类可编辑下拉，候选项 = 数据树品类 ∪ 已有模型品类，
   预填当前全局品类；force / with_l3 复选框）→ 提交后轮询 /api/tasks/{id} 显示进度，
   完成后展示结果摘要（模式 / L1 状态与阈值 / L3 状态，skipped 显示原因），
   关闭后自动刷新模型表与校准集面板。

### 实时监控页（monitor_page.py）
6. **"分块统计"契约不匹配（确定 bug）**：后端 `Detection.n_tiles` / 检测返回的 `n_tiles`
   是字典 `{"l1": N, "l2": N, "l3": N}`，UI 原来把它当标量直接拼字符串（L1 显示原始 dict），
   并读取根本不存在的 `l2_tiles` / `l3_tiles` 键（恒显示 "-"）。已改为解析字典取 l1/l2/l3。
7. **ImagePickDialog 空态行前补 `clearSpans()`**，与其他表格保持一致。

### 统计报表页（stats_page.py）
8. **"累计检测" KPI 读错键（确定 bug）**：`/api/stats/overview` 的 `totals` 里是
   `n_detections`，UI 读的是不存在的 `n_inspected`，导致恒显示 "-"。已修正。
9. **"待处理反馈" KPI 名不符实（确定 bug）**：overview 无 `n_feedback_pending` 键
   （该键只在 /api/system/info 中），原代码回退到"今日反馈数"。已改为读取
   `totals.n_feedback` 并把卡片标题改为"累计反馈"。

## 遗留建议（未改动，供后续决策）

### 需要后端配合（本次约束不改 backend/）
1. **检测响应缺少阈值字段**：`/api/detect/*` 与监控帧消息均不返回融合校准阈值
   `_aoi_threshold`，监控页 ScoreGauge 只能用默认阈值 0.5 绘制刻度，
   与服务端实际判定阈值可能不一致。建议后端在检测返回中补 `threshold` 字段。
2. **监控帧消息无 detection_id**：`backend/api/ws.py` 的 `_frame_message` 未携带
   `detection_id`（视频模式下其实已持久化），导致监控过程中"标记误检/漏检/确认正确"
   按钮只能禁用。建议 `_frame_message` 补传 `detection_id`。
3. **统计 overview 无待处理反馈数**：如需恢复"待处理反馈"KPI，建议
   `/api/stats/overview` 的 totals 增加 `n_feedback_pending`（system/info 已有现成查询）。

### 纯 UI 侧（影响小，未改）
4. **augment_page 图标加载竞态**：`_IconLoader` 按行号回写图标，若加载途中列表被
   `clear()` 后重建，旧加载器可能把图标写到新行内容上（仅图标错位，无崩溃）。
   建议后续以 image id 为键校验后再回写。
5. **ImageViewer._redraw_boxes 缩放因子**有一段重复计算的冗余代码（功能正确），
   可在下次重构时顺手清理。
6. **data_page._on_task** 对任意完成的后台任务（含评估等非导入任务）都会刷新图像/
   伪异常表格，属多余的轻量刷新；如需优化可按 `task_type` 过滤。
7. **model_page._on_task** 目前只关注 self_update 类任务；"model_prepare" 任务由
   准备模型对话框自行轮询，若用户在任务进行中关闭对话框，页面进度条不会接管，
   但完成后重新进入页面 / 手动刷新即可看到新模型（对话框关闭时已强制 reload）。

---

## 附：工单「1」全流程 UI 走查（2026-08-29）

走查脚本 `tests/_wo1_fullflow_walk.py`（真实 UI 事件驱动点击，截图落
`tests/_wo1_shots/`）：顶栏选工单1 -> 工作台 -> 模型管理准备 metal_plate
（生成 v7）-> 评估看板运行落库 -> 监控页暂停/恢复产线 -> 标记漏检提交反馈
-> 学习提升固化 v8 -> 数据管理队列回队再检（检测数 313->315）。

首轮结果：**27 项通过、QMessageBox 0 次、365s**，发现 3 个问题，
当日已全部修复；修复后复跑 **28 项全过、67s、进程正常退出（exit 0）**：

1. **已修复：准备模型对话框结果区样本量显示「正常 ? 张 / 异常 ? 张」**
   （确定 bug）：`PrepareModelDialog._show_result()` 读 `counts.normal/anomaly`，
   而后端 prepare 任务结果（`routes_models._prepare_demo5` 返回值，task_manager
   原样透传）实际字段是 `n_normal` / `n_defect`。已改为读实际字段，
   复跑走查确认显示「正常 54 张 / 异常 30 张」。
2. **已修复：产线"空转"状态不可见**（UX 缺口）。工单1 的 6 个批次中，
   已准备品类全部图片检完、其余 4 个品类未准备被产线按设计跳过时，
   监控页仍只显示「产线：运行中」，用户无从得知没有可检数据
   （首轮断言「恢复后自动检测 +2」因此超时，属数据耗尽而非功能故障）。修复：
   - 后端 `GET /workorders/{id}` 详情新增 `pipeline_summary`
     （`_pipeline_summary`：n_pending 待检测图数[口径与 pipeline_service
     一致：普通批次=未检测、回队批次=最近检测早于回队时刻]、
     unprepared_categories 未准备品类、n_images_blocked 被跳过图数；
     只挂详情接口，列表接口不加，避免逐工单读引擎快照）；
   - 监控页产线框新增空转提示标签与「去准备未就绪品类」按钮
     （`goto_page` 信号 -> 主窗口跳「模型管理」；仅运行中且存在未准备
     品类时显示按钮），并加 10s 轻量轮询使提示随消费实时变化
     （页面可见时才刷）；
   - 走查步骤4改为按 `pipeline_summary` 分支断言：有积压 -> 等自动检测
     落库；无积压 -> 断言空转提示与引导按钮。复跑时走有积压分支通过；
     空转分支经离屏单测验证 4 种状态（空转+未准备 / 有积压 / 空转无未准备 /
     暂停隐藏）渲染与跳转正确。
3. **已修复：走查脚本 `finish()` 里 `sys.exit()` 在 QTimer 回调内不生效**，
   汇总打印后进程挂起需手动终止。改为 `app.quit()` 退出事件循环、
   `app.exec()` 返回后统一 `sys.exit`（复跑进程正常退出）。

---

## 附 2：学习曲线跳转品类断链修复（2026-08-29）

**现象**：从模型管理点「学习曲线」跳到学习曲线页看不到内容（三个区全空态）。

**根因（两处叠加）**：
1. 品类断链（交互缺陷）：模型管理树选中品类只是模型页局部状态 `_cur`，
   「学习曲线」按钮的 `open_learning` 信号不携带品类；学习曲线页加载用
   全局品类 `current_category()`（按前端反馈 v3 顶栏品类已移除、由工单推导），
   **单品类工单返回该品类，多品类工单返回空**。工单1 是 6 品类 -> 空 ->
   直接填空数据。
2. 「离线回放」Tab 内容有生成前提：曲线来自手动「开始回放」的记录；
   且该 Tab 原本**没有任何品类入口**，多品类工单下必空（「在线曲线」Tab
   有下拉可手选，能绕过断链）。

**修复**：
- 模型页 `open_learning` 改为 `Signal(str)`，按钮经 `_emit_open_learning`
  携带当前选中品类（`_cur.category`，回退全局品类）发射；
- 主窗口 `_on_open_learning`：先 `page_learning.show_category(cat)` 再切页；
- 学习页新增 `show_category()`：切到离线回放 Tab，离线/在线两个品类下拉
  都定位到该品类（下拉不含时追加），并触发各自数据加载；
- 离线回放 Tab 新增品类下拉（「跟随工单」= 全局推导 + 全部品类候选），
  `reload()` 改用 `_offline_cat()` 取值；离线下拉选择不再联动在线 Tab
  （两 Tab 各自独立选品类）。

**验证**：离屏 UI 冒烟（工单1 多品类、全局品类为空，树选 metal_plate ->
点「学习曲线」）12 项断言全过：跳转、Tab 定位、双下拉定位 metal_plate、
离线/在线数据请求均携带 metal_plate、在线曲线显示累计反馈 5（非空态）。
修复前同场景离线请求品类为空、页面全空态。

---

## 附 3：评估看板延迟图 / 学习曲线回放按钮修复（2026-08-29）

**问题 1：评估看板「延迟明细」看不到图**。库里全部是 accuracy 记录、
无 benchmark；原 `_on_row_selected` 只在 `run_type=="benchmark"` 时绘图，
选任何行图都是空的（accuracy 记录 latency 只有 mean/p95 两键）。
修复：不限定类型——latency dict 有键就画柱状图（按 mean/p50/p95/max/min
优先级取可用键，accuracy 显示 mean/p95 两柱），无延迟数据的记录
（如 learning_curve）明确提示。

**问题 2：学习曲线点「开始回放」出 bug**。上一轮加了离线 Tab 品类下拉，
但 `_on_run_curve` / `_on_run_contribution` 仍用全局品类 `_category()`
（多品类工单为空）→ 下拉已选品类点回放却被拦下提示"请先在顶栏选择品类"。
修复：两按钮改用 `_offline_cat()`（下拉优先、回退工单推导）。
API 复现确认后端回放链路正常（stream=20/eval=20 约 117s，
metal_plate AUROC 0.7→0.8，落库 eval_run_id=9）。

**验证**：离屏 UI 冒烟 7 项全过——选中 accuracy 记录延迟图出现柱状图
（BarGraphItem=1）、提示更新；离线下拉选 metal_plate 点回放请求品类
=metal_plate 且带 n_stream/n_eval/max_feedback。

---

## 附 4：评估看板四问修复（预算断链/自动选中/六维评价审视）（2026-08-29）

**问题 1：精度评估与延迟基准分两种有必要吗？能不能一次都做？**
结论：两口径有必要保留（benchmark=预热+大图优先的严格计时考核口径；
accuracy=带标注集的精度口径），但 accuracy 已顺带计时，此前却只存
mean/p95 两键浪费数据。修复：`run_accuracy_eval` 改为
`with_heatmap=False`（评估不需要热力图，计时更接近推理口径且更快），
latency 存完整五统计量（mean/p50/p95/max/min），metrics 增
`budget_ms`/`latency_pass`——**一次精度评估即同时产出精度与完整延迟
分布**；benchmark 保留为严格考核口径。

**问题 2：设置里把延迟预算改成 150，评估看板和顶栏都不变**。三处断链：
1. 后端 `run_latency_benchmark` 读 `evaluation.latency_budget`——yaml
   中不存在该键，永远落默认值 200 → 改读白名单键
   `pipeline.latency_budget_ms`（docker.yaml 遗留 evaluation 键不再被读）；
2. `main_window` 只在首次连接拉一次 `system_info`，设置保存后顶栏与
   `_budget_ms` 不更新 → SettingsPage 新增 `config_saved` 信号，
   保存成功（applied 非空）后发射，主窗口连接后重拉 system_info；
3. 评估页多处文案硬编码 `<200ms` → `_refresh_budget_texts()` 用当前
   预算动态化 tip/note/表头，且预算变化后对当前选中行重调
   `_on_row_selected()` 重绘柱状图与预算线（修复前 hint 会残留旧预算值）。

**问题 3：运行评估完成后「延迟明细」仍为空**。根因：评估完成 reload 后
表格无选中行，`_on_row_selected` 不会触发，且默认 hint 写
"选中一行「延迟基准」记录"误导。修复：`_fill_runs` 末尾 `selectRow(0)`
自动选中最先记录并立即画图；hint 改通用文案。

**问题 4：按 algo 六维评价框架审视看板覆盖面**（demo4 技术详解第 12 节）：
- 维度1 基础精度（accuracy）/ 维度3 计算开销（benchmark）：原有；
- 维度2 噪声鲁棒性：后端 `/eval/robustness` 现成但 UI 无入口 →
  **已补**：评估类型加第三项「噪声鲁棒性」，`_on_run` 分支调
  `eval_robustness`，`_on_type_changed` 默认样本数（鲁棒 12≈192 次前向，
  防止默认 100 样本产生 1600 次前向的超长任务；benchmark 20 / accuracy 100）；
  AUROC 列对 robustness 记录回退显示 `baseline_auroc`；
- 维度5 特征归因：在学习曲线页贡献档案；维度6 数据诊断：已由贡献档案
  替代下线；维度4 跨分布一致性：后端未实现，看板不加。

**验证**：离屏冒烟 20 项断言全过——预算=150 启动态（顶栏/评估页文案/
预算线）、accuracy 记录自动选中出柱状图、鲁棒性入口请求参数正确、
设置页改 180→顶栏"延迟预算 <180ms"+后端 system_info=180+评估页标题与
预算线全链路跟随→还原 150。

---

## 附 5：监控/标注反馈页布局重构 + 统一分页方案（2026-08-29）

**问题 1：标注反馈页「待复核」表格列错位（列表数据被遮挡的根因）**。
`QTableWidget(0, 5)` 只建 5 列却按 6 列写入（时间/品类/系统判定/图片路径/
分数/评分摘要），表头与内容整体错位：判定写进"图片路径"列、最后
"评分摘要"列被 Qt 静默丢弃，空态 `setSpan(0,0,1,6)` 同样越界。
修复：改 6 列 + 表头对齐 + 路径列宽移到第 3 列。
**教训**：QTableWidget 列数/表头/写入下标三处必须同步，Qt 对越界
setItem 静默忽略，编译与冒烟都抓不到，只能靠截图走查发现。

**问题 2：监控页垂直堆叠过多、队列表固定 120px 只剩 3 行**。
按用户意见改左右布局：数据流队列表从页顶挪入左栏（上=试检旁路面板、
下=队列，QGroupBox 可折叠保留），去掉 maxHeight 限制高度弹性伸展；
主区分栏 [340, 740, 290]。顶部只留横幅 + 告警 + 产线控制条。

**问题 3：反馈记录一次拉 200 条无分页，堆积后卡顿看不过来**。
后端 `/api/feedback` 早已支持 page/page_size/total，前端没用。
统一方案：`ui/pages/common.py` 新增 **PagerBar** 组件（上一页/下一页/
页码总数/每页条数，`page_changed` 信号 + `reset()/goto()/set_total()`），
反馈记录与待复核两个列表统一接入服务端分页（默认 20 条/页）；
筛选条件变化 `reset()` 回第 1 页；作废/复核导致当前页空时
`goto(page-1)` 自动回退。后续其它数据列表（图像、批次等）应复用该组件。

**删除语义决策（用户确认）**：反馈记录只保留「作废」软删除
（可追溯、对应检测重回复核队列），不新增物理删除——符合工业质检
审计惯例；已学习的反馈影响需模型版本回滚消除。

**布局精简（用户确认）**：反馈页顶 7 张 KPI 卡精简为 4 张核心卡
（反馈数/错检占比/待复核/已复核），反馈类型占比环形图下移
「追溯与提升」Tab 与翻案曲线并列，列表区获得垂直空间。

**验证**：`py_compile` 通过；离屏冒烟 t15 十页全过；离屏截图走查
确认监控页左栏队列、反馈页 4 卡 + Tab 区布局正常。

---

## 附 6：全流程 UI 演示走查 v2（2026-08-29，`tests/_ui_demo_flow.py`）

可见窗口全流程（建源(MVTec导入)→建工单→准备模型(预训练)→产线控制+实时流→
标记漏检→人工复核判缺陷→反馈作废→学习提升→批次回队）：**32 项全过、exit 0、
95s**；发现 2 个问题：

1. **遗留 bug（已修复）：`feedback_page._fill_summary` 引用已删除的 KPI 卡**
   `_fill_summary` 仍写 `self.kpi_fp` / `self.kpi_fn` / `self.kpi_new`
   （7 卡时代属性），当前 4 卡（kpi_total/kpi_rate/kpi_pending/kpi_reviewed）
   重构后不存在 → 每次反馈页汇总刷新抛
   `AttributeError: 'FeedbackPage' object has no attribute 'kpi_fp'`
   （本跑出现 3 次，UI 线程内被 Qt 捕获，不影响流程断言，但污染日志）。
   修复：`_fill_summary` 按新 4 卡契约改（total→kpi_total、错检占比→kpi_rate、
   已复核→kpi_reviewed，donut 数据保留）。
   **教训**：删减 QWidget 成员时必须全局搜索所有引用点（grep `kpi_fp|kpi_fn|kpi_new`），
   py_compile/冒烟只查语法与实例化，不查成员引用；本次靠全流程真实操作
   （反馈页汇总刷新）才暴露。

2. **作废反馈弹确认框（弹窗统计=1，脚本预期 0）**：「作废」按钮点击后弹出
   `QMessageBox.question` 确认框（自动应答「是」后作废成功落库）。作废属
   不可逆操作，确认框本身是合理 UX，但与全流程自动化"0 弹窗"预期冲突——
   若后续要求全流程无阻塞弹窗，需把确认框改为可旁路（如 shift 直过）
   或脚本将其计为预期项。
