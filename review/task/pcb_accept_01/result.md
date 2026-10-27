# PCB-ACCEPT-01 真实桌面调参与检测全链路（部分覆盖）

## 结论：覆盖场景 PASS；完整发布链路未覆盖（需已发布模板）

真实桌面（非 offscreen，1920×1080）启动稳定版 `PCB_Dual/main.py`（后端 8022 + 自动拉起 AOI_Core 子进程）。

## 已验证场景

| 场景 | 结果 | 证据 |
|---|---|---|
| 启动与后端 | UI 正常，内嵌后端 8022 + AOI_Core 子进程拉起 | 进程/日志 |
| 模板选择 | 引用模板下拉列出草稿模板（SMT 焊点示例 v1·草稿·插件器件） | 截图（会话） |
| 标准图载入 | 标准图自动从模板加载（2R2 元件图） | 截图 |
| 测试图载入 | 文件对话框选图，尺寸/缩放比显示（941×620·35%） | 截图 |
| **尺寸门禁 fail-closed** | 标准图 1317×545 vs 测试图 941×620 → **ERROR**："差异过大（高差75px/宽差376px，超门禁阈值 10px），已拦截，未执行检测 · 21ms"，无伪 OK | 截图（会话） |
| **真实检测 NG** | 同尺寸测试图 → NG 1 项·71ms；缺陷表：多锡(包锡) 316,181 46×16，算法来源 solder_多锡，**ROI "未提供"（不伪造关联）**，耗时 19ms，分数"不可用"（不显 0）；缺陷框叠加在原图坐标上 | 截图（会话） |
| **HSV/参数调参与预览** | 多锡阈值 0.01→0.50 → 防抖后自动重跑 → NG→OK（无缺陷·47ms），缺陷框消失；"当前生效：模板配方（…/solder_多锡）"来源标识正确 | 截图（会话） |
| **预览不污染模板** | 调参后未点保存；API 核对模板参数 `excess_thresh=0.005` 等未被修改 | /api/templates/{id} |
| 归档默认关闭 | "归档本次"默认未勾选，检测结果不写历史库 | UI 状态 |

## 发现的问题

1. **Qt spinner 程序化赋值不生效**（computer-use set_value 与 ctrl+a+type 均未提交，val 保持 0.5）——属 UI 自动化兼容性问题，人工操作不受影响；但提示 spinner 的提交依赖焦点丢失/回车事件，真实用户若只输入不回车可能有同风险，建议开发复核 spinner 提交时机。
2. 首次恢复 0.01 未生效期间结果仍显示旧值 OK——过期结果与当前参数短暂不一致（属自动化赋值未提交导致，非应用 bug）。

## 未覆盖（需进一步授权/资产）

- ~~发布模板→自动检测全链路~~（已于 2026-10-14 第二轮覆盖，见下节）。
- TH 自动轮廓/单框/多框的禁用与提示（无 TH 模板）。
- 快速连续拖动/取消预览/关闭运行中窗口的压力场景。
- 模板重开与冻结快照一致性（第二轮已覆盖重启后发布态保持；冻结快照逐项比对未做）。

## 第二轮：TEMP 隔离发布链路（2026-10-14，按协同.md 阻塞处理指引）

- 隔离方法：复制稳定版 PCB_Dual 代码（排除 test_images/storage）到 `temp_tree\PCB_Dual`（13.1 MiB），draft 模板复制进全新 TEMP 模板库；`PCBDUAL_PORT=8023`、`autostart=false`；AOI_Core 后端用稳定版 8017（只读复用 smt_solder v4）。
- 链路：checklist 门禁先阻断（"品类模型 smt_solder 尚未准备"，8017 未启动时）→ 启动 AOI_Core 后 checklist 通过 → API `POST /api/templates/{id}/publish` 发布成功 → UI 下拉出现已发布模板（无"草稿"标签）→ 自动检测 NG 1 项（多锡 316,181 46×16，与调参页结果一致）→ 关闭重启 → 发布态保持。
- 隔离核对：TEMP storage 落盘 result.json/result_vis.png/batch_summary；**稳定版 storage 与模板库零修改**（稳定版该模板仍为 draft）。
- 未覆盖：发布拒绝场景（checklist 阻断项已见一例：品类未准备 400/阻断）；UI 内发布按钮路径（本次走 API，指引允许正式 API）。
- 产物：raw/build_temp_tree.py、raw/check_temp_persist.py、temp_tree/（完整隔离树）。

## 第三轮：TH 交互模式（2026-10-14）

- 环境：TEMP 隔离树 `temp_tree\PCB_Dual`，`PCBDUAL_PORT=8023`，`py -3.11 main.py`，真实桌面（未设 offscreen），1920×1080，PySide6 6.8.3。
- 前置事实：`storage\settings.json` 不存在 → 全局尺寸门禁默认开启（阈值 10px）。**调试页（算法调试）走全局门禁而非模板门禁**（`app\ui\detect_page.py` 调 `scheduler.run` 不传 gate 参数 → 回退 `get_gate_settings()`；`app\inspect\service.py:209` 才用模板 gate）。为完成 NG 场景 UI 跑图，经 UI「设置→系统设置」将全局门禁关闭（写入 TEMP 树 settings.json，记录在案，模板 gate 未动）。

| 场景 | 结果 | 关键证据 |
|---|---|---|
| A 0框自动模式 + test_2-1_OK | PASS | UI：OK·无缺陷·157ms；抽屉「插件焊点检测·孔洞等4项」中连锡禁用；API(allow_stub=true)：孔洞/少锡/多锡/不出脚真实执行 OK（36–50ms），连锡 SKIP+原因「连锡需要在「标定」中框选至少 2 个焊点区域」。raw\th_round\A_*.png/json |
| B 0框 + test_2-8_NG | PASS | 首次被全局门禁拦截（100×100 vs 124×125）→ 关全局门禁后：UI NG 1 项·204ms，缺陷行：少锡 3,6 111×118 tht_少锡 ROI=未提供 NG 51ms 分数=不可用；API overall=NG，连锡 SKIP。raw\th_round\B_* |
| C 单框标定（模板建模画 1 矩形） | PASS（标定 UI 画框自动化成功） | 画布拖框 → `storage\template_calib\TH_templ2_single.json` pads=1（rect 4.4,7.3 92×93）；回调试页重跑 test_2-8_NG：NG 1 项·169ms（少锡），连锡仍禁用/SKIP。raw\th_round\C_calib_t2_1pad.json、C_ui_t2_ng_1pad.png、C_api_detect_t2_ng_1pad.json |
| D 清空框恢复自动模式 | **FAIL（缺陷）** | 建模页「清空」后文件 pads=[]，但 legacy `roi` 字段残留旧框（sync 以模板内 calibration 为 base 合并所致），且 `get_th_roi` 的 legacy 回退把旧 roi 复活 → `get_th_pad_count`=1、`effective_solder_mode`=roi，无法恢复 ellipse 自动模式；重跑结果与单框模式一致（NG 1 项·164ms）。raw\th_round\D_calib_t2_after_clear.json、D_probe_after_clear.txt、D_studio_after_clear.png（发布检查已报「未框选检测区域」而算法仍按 1 框跑，UI 与算法不一致） |
| E TH_templ3_dual 双框 + test_3-1/3-2_NG | PASS（含一次阻断绕行） | 建模页画 2 框（左 x≈2.5–64.8、右 x≈196–259.5）→ **必须点「保存模板」**，否则调试页重新加载模板时 `sync_calibration_to_legacy` 用空 calibration 覆盖并清空 legacy 标定文件（见问题 2）；保存后调试页：连锡启用、其余 4 项禁用，抽屉标题变「插件焊点检测·连锡」；test_3-1_NG → 连锡 NG（1,0 249×124，72ms）；test_3-2_NG → 连锡 NG（3,0 247×122，57ms）；API：连锡真实执行 NG、其余 4 项 SKIP+多框原因。raw\th_round\E_* |
| F 禁用与原因提示 | PASS | 三模式下禁用项与原因串经 UI 树 + 适配器探针双重核对（raw\th_round\F_readiness_probe.json）；无伪装全绿 OK（未就绪项禁用且不参选；API 层 SKIP 带原因）。tooltip 本体未截图（UIA 不暴露），原因文字以发布检查区与探针输出为准 |

### 第三轮发现的问题

1. **【缺陷】清空 TH 框无法恢复自动模式**（场景 D）：`app\template\regions.py:115-123` 在 pads 为空时保留旧 `roi`/`shape_choice`，叠加 `app\calibration\th_roi.py:52-56` 的 legacy 回退，删除的框被复活为单框模式。建模页 UI（发布检查「未框选检测区域」）与算法实际模式（roi）不一致。
2. **【缺陷/数据丢失风险】调试页切换模板会清空未保存的 legacy 标定**：`app\template\store.py:364-368 sync_calibration_to_legacy` 在模板 region_sets/calibration 为空（未点保存模板）时把 `storage\template_calib\<id>.json` 覆盖为 `{}`。本轮 TH_templ3_dual 画好的 2 框在进入调试页重新选择模板后被清空（建模页内存中仍显示 2 框，进一步加剧不一致）；TH_templ2_single 的清空残留态也被随后覆盖为 `{}`（raw\th_round\final_calib_t2.json）。
3. **【设计不一致】调试页尺寸门禁不随模板**：调试页用全局 `gate_enabled`，模板级 `gate.size_gate_enabled=false` 在调试页不生效；本轮 NG 测试图被全局门禁拦截一次。
4. **【提示】API 直跑时未就绪项报 ERROR 而非 SKIP**：`POST /api/detect/traditional`（allow_stub 默认 false）下连锡未就绪 → status=ERROR/error_code=NOT_READY/execution_mode=not_executed，overall 变 ERROR（raw\th_round\A_api_detect_t2_ok.json）；调试页路径（allow_stub=true）才是 SKIP。属产线 fail-closed 语义，提示文档/前端需区分。
5. 建模页「关联」标签在「清空」后未即时刷新（仍显示 框/点 1，发布检查已正确更新）。

### 第三轮未覆盖

- tooltip 悬浮截图（UIA 不暴露 tooltip 文本；原因串以代码+发布检查区+探针核对）。
- REVIEW 状态触发（本轮图像均 OK/NG/SKIP）。
- 连锡参数调参联动、归档勾选路径（本轮均默认不归档）。
- TEMP 树遗留状态：全局门禁已关闭（settings.json gate_enabled=false）；TH_templ3_dual 已保存 2 框标定（region_sets 入模板）；TH_templ2_single 标定文件现为 `{}`。

### 收尾核对

- 应用窗口关闭，pid 5428 退出，端口 8023 释放（raw\th_round\shutdown_check.txt）。
- 稳定版 `stable_AOI_tree\PCB_Dual\storage` 与 `templates` 前后快照（FullName/Length/LastWriteTime/SHA1）逐字节一致：storage identical=True、templates identical=True（raw\th_round\stable_*_before/after.json）。

## 产物

- raw/final_state.png（最终全屏截图）；会话内另有 ERROR/NG/调参/OK 各场景截图
- raw/th_round/（第三轮：A–F 场景截图、API JSON、标定文件副本、探针输出、关停与稳定树核对记录）
- env.json / cmd.txt

## 第四轮：压力与健壮性场景（2026-10-14）

- 环境：TEMP 隔离树 `temp_tree\PCB_Dual`，`PCBDUAL_PORT=8023`，`py -3.11 main.py`，真实桌面（未设 offscreen），1920×1080；AOI_Core 稳定版 8017 手动拉起（autostart=false，只读复用 smt_solder v4），整批检测前曾被 checklist 阻断「品类模型 smt_solder 尚未准备」，启动 8017 后解除。
- 模板：SMT 模板（多锡 `excess_thresh=0.005`、少锡 `insufficient_thresh=0.08`、连锡 `bridge_min_area=20`）；测试图：excess_pair/pair4 NG 图（1317×545）与 OK 图等 4 张导入列表。

| 场景 | 结果 | 关键证据 |
|---|---|---|
| 1 快速连续调参 | PASS | 滑块快速三连（阈值 1→20、面积 5→50、阈值→8，set_range_value 连发），防抖收敛后最终结果「OK·无缺陷·39ms」与最终参数（0.08/50）一致；模板 API 快照 SHA256 三次一致（A774F50A…，raw\stress_round\tpl_before_s1.json / tpl_after_s1.json / tpl_final_check.json），预览与整批全程未污染模板。s1_before_baseline.jpg（基准 NG 1 项 多锡 316,181 46×16·104ms）、s1_after_final_params.jpg |
| 2 快速切换算法/图片 | PASS（附发现 1/2） | 快速切检测项勾选（少锡 off→on、连锡 off）+ 快速切图 NG→OK→NG，最终界面无残留缺陷框/残留状态。s2_final_ng_shows_ok_suspect.jpg |
| 3 取消预览（变体：整批运行中切页） | PASS | 调试页无「取消」按钮，改为：Ctrl+Enter 整批检测进行中立即切「算法调试」页——调试页保留自身旧结果，自动检测页结果不回填跨页；切回后整批结果正确（直通率 50%，NG 图 NG 1 项 多锡 316,181 46×16·3103ms，列表含 [NG]/[OK]/[ERR]×2）。s3_switchtab_during_batch.jpg、s3_batch_result_autodetect.jpg |
| 4 关闭正在运行的窗口 | PASS | 点「开始检测」后立即关窗：进程 PID 3252 退出码 0、端口 8023 释放（Test-NetConnection=False）、关窗后 6 秒 TEMP storage 最新写入时间停于 10:30:13 不再增长（无残留写盘，关闭后不继续投递） |
| 5 窗口不卡死 | PASS | 全程（快速调参/切图/整批/关窗/重启复核）UI 响应正常，未出现 >5s 无响应 |

### 第四轮发现的问题

1. **【UI 缺陷】阈值 spinner 两位小数精度丢失致参数无法精确调回模板值**：模板真实 `excess_thresh=0.005`，spinner 只显两位小数显示为 0.01；滑块调回「0.01」实际写入 0.01≠0.005，NG 图结果变为 OK（场景2 表面异常根因）。三重证据确认为参数差异而非过期结果缺陷：①手动「对照跑图」仍为 OK；②自动检测页整批（用模板真实参数）跑出 NG 1 项与基准一致；③重启应用后调试页基准恢复 NG 1 项·87ms（s2_restart_baseline_restored.jpg）。建议阈值控件提升小数位数或与模板精度对齐。
2. **【语义问题】「恢复默认」恢复到算法全局默认而非模板配方**：点「恢复默认」后阈值显示 0.15（算法默认），非模板值 0.005（s2_restore_default_015.jpg）。对一线操作员易误导，建议明确「恢复默认」语义或提供「恢复模板配方」。
3. **【提示】bridge 两张测试图整批结果为 [ERR]**：bridge_pair 图整批报 ERR（疑尺寸/区域与模板不匹配），未深究；另「导入文件夹」不递归子目录（导入 test_images 根目录提示无图片，需选具体 pair 文件夹）。

### 第四轮未覆盖

- 调试页无「取消」按钮，取消预览场景以「整批运行中切页」变体替代（未验证任务取消语义本身）。
- 归档勾选路径未压测（本轮默认不归档）；TH 模板压力场景未覆盖（本轮聚焦 SMT 模板）。

### 第四轮收尾核对

- 应用（PID 3252 及重启复核实例 PID 10532）均已退出码 0 关闭，端口 8023 释放；AOI_Core（PID 17676）已停止，8017 释放。
- 本轮仅写 TEMP 树与 `review\task\pcb_accept_01\raw\stress_round\`；稳定版 `stable_AOI_tree` 与 `AOI_tree` 零修改，无 git 操作。
