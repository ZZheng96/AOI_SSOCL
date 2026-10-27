# REMAIN-02 验收结果：AOI_Core 特征工程页 API 真实 HTTP 联调

- 日期：2026-10-14 ｜ 环境：Windows / py -3.11 / torch 2.5.1+cu121 / CUDA（GTX 1660 SUPER）
- 被测服务：稳定版源码只读运行（`stable_AOI_tree\AOI_Core\server.py`），配置指向 TEMP 隔离副本
  `temp_core\config.yaml`（独立 DB、独立 storage、端口 **8031**），生产 DB/快照/配置全程未触碰
  （`raw\prod_hash_before.json` vs `raw\prod_hash_after.json` → **PROD_UNTOUCHED**）
- HTTP 客户端：requests（真实 TCP），逐请求落盘 `raw\NN_*.json`（方法/路径/身份/请求体/状态码/响应 JSON）
- 鉴权：`X-API-Key` → 角色映射；operator=1 / engineer=2 / admin=3
- **总计 55 项，FAIL=0 → 总体 PASS**

## 1. 端点 × 角色矩阵（状态码实测）

| # | 端点 | operator | engineer | admin | 判定 |
|---|------|----------|----------|-------|------|
| 1 | GET /api/health（无密钥豁免） | 200（无密钥） | – | – | PASS |
| 2 | GET /api/system/info | 200 | 200 | 200 | PASS（无/错密钥 401，角色回显正确） |
| 3 | GET /api/models/feature-set/{cat} | 200（can_edit=false） | 200（can_edit=true） | 200（can_edit=true） | PASS |
| 4 | GET /api/models/feature-revisions | 200 | 200 | 200 | PASS（含 production_binding=unknown） |
| 5 | GET .../feature-set/{cat}/hard-pool | **403** | 200 | – | PASS |
| 6 | GET /api/models/{cat}/slot-health | **403** | 200 | – | PASS |
| 7 | POST .../feature-set/{cat}/draft | **403** | 200（state=draft） | 200（幂等同 rid） | PASS |
| 8 | POST .../feature-set/{cat}/report | **403** | 200（后台任务→done） | – | PASS |
| 9 | POST .../feature-set/{cat}/rebuild | **403** | 200（activated=false） | – | PASS |
| 10 | POST /api/models/feature-revisions（注册） | **403** | 409（血缘不全拒） | – | PASS |
| 11 | POST .../feature-revisions/{rid}/admission | **403** | 200（只产证据） | – | PASS（不存在 rid → 409） |
| 12 | POST .../feature-revisions/{rid}/transition | – | **403** | 409（无证据/越级均拒） | PASS |
| 13 | POST /api/system/config | **403** | **403** | 200 | PASS |
| 14 | GET /api/candidates/recovery/prepared | **403** | 200 | – | PASS |
| 15 | POST /api/candidates/{cat}/{id}/decision | **403** | **403** | 409（不存在候选） | PASS |

补充参数校验：draft 空 groups → 400；非法 scoring → 400；未知品类 feature-set → 200 enabled=false；
未知品类 rebuild → 409「无激活版本」。

## 2. draft 不改生产（隔离证据）

- 请求：engineer POST draft（groups=["glcm"], scoring="grouped"）→ 200，
  响应 `{state:"draft", activated:false, changes_production:false}`，
  rid=`ba47d05c…ac8df8`；admin 同 manifest 重发 → 同 rid（幂等）。
- 快照对比（`raw\snap_before_draft.json` vs `snap_after_draft.json` → `isolation_draft.json`）：
  - `db_business_tables_unchanged: true`（images/models/eval_runs/detections 等业务表零差异）
  - `engine_current_pointer_unchanged: true`（引擎 current.json 哈希不变）
  - `production_pointer_missing_before/after: true`（feature-revision 生产指针
    registry.production.json 始终不存在 → 无生产绑定）
- draft 后 GET revisions：`production_binding="unknown"`、`changes_production_extractor=false`，
  manifest 含 implementation_hashes / base_signature / admission_required=true。

## 3. report / rebuild 不自动激活

- **report**：engineer POST → task_id=3，轮询 `GET /api/tasks/3` 16s 内 done；
  之后 GET feature-set `report.status="current"`；任务前后业务表与指针哈希不变
  （`snap_after_report.json`）。report 只写档案，不触碰激活。
- **rebuild**：engineer POST → `{task_id:6, parent_version:2, activated:false, scenario:"L1a", profile:"cpu"}`；
  任务 8s done，产出 **v3 未激活新版本**，`auto_report={"status":"skipped_rebuild"}`；
  重建后 GET feature-set：`version=2`（仍为旧版）+ `pending_rebuild_version=3`；
  `current.json` 指针哈希不变（`snap_before_rebuild.json` 对比）→ **不自动激活，待 G2 门控** 实测成立。
- **无生产绑定语义**：feature-revision 注册表无任何 bind_production 操作，
  列表固定返回 `production_binding="unknown"`，生产指针文件不存在（unbound）——与权威要求
  「无生产绑定保持 unknown/unbound」一致。

## 4. 关联门禁实测（G4 周边）

- **阈值冻结**：admin 冻结 → 改阈值 409 `reason_code=THRESHOLD_FROZEN` → 解冻 → 可改（闭环 4 步全 PASS）。
- **依赖锁**：环境锁不一致（mismatches=13）；policy=block 后候选激活 → 409
  `reason_code=DEPENDENCY_LOCK_MISMATCH`，指针未变；恢复 warn。
- **状态机**：transition 无独立评估证据 → 409「证据路径未知」；draft→active 越级 → 409；
  admission 只产证据不注册不激活。

## 5. UI ↔ API 对照

- 离屏（QT_QPA_PLATFORM=offscreen）用 8031 **真实响应**填充学习页「当前特征集」卡片：
  - operator：can_edit=false → 提交候选/档案/重建按钮全禁用、15/15 组复选框禁用、状态栏含「仅工程师」→ PASS
  - engineer：can_edit=true → 按钮启用、pending_rebuild_version=3 显示「已重建 v3，待激活门控（G2）」→ PASS
  - 截图：`raw\ui_feature_set_card_operator.png` / `ui_feature_set_card_engineer.png`；
    断言明细 `raw\ui_offscreen.json`
- 静态核对：`ui\api_client.py` 端点封装（feature_set/draft/report/rebuild/hard-pool/slot-health/admission）
  与后端路由逐一对应，路径/方法一致。

## 6. 缺口清单（不伪造）

1. **桌面交互对照未覆盖**：真实桌面端点击按钮→任务进度条→状态刷新链路未做人机操作验证
   （本次以离屏填充+真实 API 响应替代，按钮回显逻辑已核对）。
2. **transition 正向激活路径未走通**：draft→evaluated→approved→active 需要独立评估证据文件，
   本次只验证了门禁拒绝面（403/409），未构造合规证据完成正向激活（属 G2 门控流程范畴）。
3. 第一轮联调曾因 TEMP 库无检测血缘被启动钩子 `register_engine_snapshots` 删除 current.json
   （routes_models.py 204-212 行，设计行为）导致 draft/report/rebuild 409；已通过
   `fix_lineage.py` 插入 Detection 血缘记录修复并重跑，全部通过。此行为对「清库留 storage」
   场景是预期保护，记录在案。

## 7. 产物索引（d:\CGAIC\AOI_SSOCL\review\task\remain_02\）

- `result.md`（本文件） / `env.json` / `cmd.txt`
- `raw\`：55 个请求/响应 JSON、`summary.json`、`snap_*.json`（DB 表级+指针哈希）、
  `isolation_draft.json`、`task_*.json`、`prod_hash_before/after.json`、
  `server_8031.log`、`ui_offscreen.json`、UI 截图 ×2
- `temp_core\`：TEMP 隔离副本（保留供复核；8031 服务已停止、端口已释放）
- 脚本：`setup_temp.py` / `fix_lineage.py` / `http_accept.py` / `db_dump.py` /
  `hash_snapshot.py` / `ui_offscreen.py`
