# PCB-ACCEPT-05 稳定版同步与复验

## 结论：同步成功，稳定版独立复验通过（不等于生产验收）

## 差异清单（同步前，只读冻结双方哈希）

- 扫描范围：`AOI_tree/PCB_Dual` vs `stable_AOI_tree/PCB_Dual`，排除 data/DB/模板/参数/快照/缓存/test_images 等数据目录与 .pyc/.pkl/.npy 等二进制。
- 统计：开发版 477 文件 / 稳定版 471 文件；**一致 447、差异 24、开发版新增 6、稳定版独有 0**。
- 无稳定版独有改动 → 无未解释冲突，不需要停止确认。

### 差异 24 项（DIFF）
- 算法核心：`alg_repo/shift_smt/.../shift_smt_all_alg.py`、`alg_repo/smt/.../_solder_core.py`、`solder_smt_all_alg.py`、`alg_repo/through_hole/.../pcb_through_hole_solder_alg.py`、`pipeline.py`
- 适配器/调度/契约：`app/detect/adapters/{shift,smt,through_hole}_adapter.py`、`contract.py`、`scheduler.py`、`types.py`
- 服务/融合/存储：`app/inspect/fusion.py`、`app/inspect/service.py`、`app/result/store.py`
- UI：`app/ui/{algo_param_panel,defect_table,detect_page,history_page,inspect_page,result_hud,template_studio_page,tuner_dialog}.py`
- 文档：`docs/架构与设计.md`

### 新增 6 项（NEW，仅开发版）
- `app/ui/tuner_preview.py`
- `tests/{test_adapter_batch,test_pcbdual04,test_shift_batch,test_tuner_preview}.py`、`tests/tuner02_real_cases.py`

## 风险与排除

- 风险：24 个 DIFF 跨越算法核心+UI+存储，若稳定版有未登记改动会被覆盖——已核实稳定版独有文件为 0，且每个被覆盖文件均已备份。
- `tests/` 目录无 `__init__.py`，`unittest discover -s tests` 不可用，须用模块方式显式运行（dev/sta 两侧同构）。
- `test_images` 等数据目录按规则排除，不参与同步（两侧未动）。

## 同步执行（用户指令"能做就都做了"视为授权；备份先行）

- 备份：`review/sync_backup/PCB-ACCEPT-05/`（24 个稳定版原文件 + sync_manifest.json 含逐文件哈希）。
- 复制 30 文件（24 覆盖 + 6 新增），**复制后逐文件 SHA-256 与开发版一致（30/30）**。
- 回滚方法：将 `sync_backup/PCB-ACCEPT-05/` 下对应路径文件复制回 `stable_AOI_tree/PCB_Dual/`，并删除 6 个新增文件。

## 稳定版独立复验

- 回归（模块方式，offscreen）：tests.test_pcbdual04 + test_adapter_batch + test_shift_batch + test_tuner_preview + test_async_template_snapshot + test_template_delete_cascade，**35 tests / 3.145s / OK**。
- 6-case 真实 paired 对照（`tests.tuner02_real_cases`，固定 seed=20261013）：
  - SMT 3/3：状态与框一致，唯一差异 output_image（源渲染时序差异，与开发版结论一致）；
  - Shift 1/1：状态/框/metadata 一致；
  - TH 2/2：scalar/batch 逐项精确一致；
  - `unexpected_failures=[]`，退出码 0。raw 工件在 `raw/`。
- 明确边界：raw SMT 输出图等价仍为 FAIL（源算法工作图绘制时序差异），不以规范化 PASS 掩盖；无 GT/业务精度结论。

## 状态

同步成功 + 稳定版回归/对照通过；PCB-ACCEPT-01/02/03/04 的缺口不因本任务通过而消除。生产验收不勾选。
