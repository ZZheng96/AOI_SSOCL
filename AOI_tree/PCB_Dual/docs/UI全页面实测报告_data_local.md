# PCB_Dual 全页面 UI 实测报告（data_local / extra_part，第三版）

- 测试日期：2026-10-07
- 测试对象：PCB_Dual（桌面端 main.py，内嵌 FastAPI :8022 + AOI_Core 子进程 :8017）
- 测试数据：`e:\CPIPC\CGAIC\data_local\extra_part`（train/good 31；test/good 8；test/extra 39；均 4096×3000）
- 测试脚本：`tests\_ui_full_datalocal.py`（进程内 QTimer 驱动真实控件 + 服务层量化）
- 第三版在第二版基础上完成问题修复，并用同一 SEED 回归。

## 1. 结论摘要

| 层面 | 修复前（第 5 轮） | 修复后（第 11 轮） |
|---|---|---|
| 汇总 | PASS 88 / FAIL 7 / WARN 2 | PASS 94 / FAIL 1 / WARN 2 |
| 传统引擎 | 召回 3/3，误检 9/18 | 召回 3/3，误检 0/18 |
| 特征引擎 | 召回 2/3，误检 0/9 | 召回 2/3，误检 0/9（模型能力问题，未改） |
| 双引擎 dual | 召回 3/3，误检 4/9，单图 1.2~1.5s | 召回 3/3，误检 0/9，最大 0.66s |
| 数据残留 | 删模板后参数文件残留 | 级联清理，已加单测 |

SEED=1791341606，PAIRS=`11_3923_977`、`2_0_1870`、`29_2044_980`。剩余 1 个 FAIL 为 feature 召回 2/3（2_0_1870 score 0.5742 < thr 0.8444），dual 融合后召回 3/3。

## 2. 测试方法

1. **UI 驱动**：在进程内用 QTimer 链点击真实控件，patch QMessageBox/QFileDialog，处理模态框，覆盖全部 Tab、菜单和设置项。
2. **服务层量化**：直接调用 `InspectService.run_with_template` / `run_dual`，统计召回、误检和逐张耗时。
3. **随机抽样**：每轮按 SEED 随机抽 3 对（缺陷图 + 同料号良品模板），可复现、可换种子。
4. **鲁棒性 case**：每对生成 自比、jpeg95、噪声 σ2、亮度 +5、平移 2px+σ2、ROI 偏离 等良品扰动，测误检。
5. **特征链路**：导入 → 准备（fast）→ 激活 → 以 feature / dual 两种模式检测。
6. **环境隔离**：S05 前备份 `storage\recipes\global_default.json`，S05 结束后立即还原（不再拖到收尾），删除测试模板和 Core 测试数据源。

## 3. 覆盖矩阵（第 11 轮）

| 步骤 | 内容 | 结果 |
|---|---|---|
| S00 | 启动：窗口、6 Tab 标题顺序、后端 /api/health、状态栏 | PASS |
| S01 | 模板工作室：建模板、标准图（显示原始文件名）、引擎模式、dual 门控、检测项、草稿、ROI、发布 | PASS |
| S02 | 自动检测：上料、批量检测、召回/误检/耗时、直通率、归档、队列过滤、NG 定位、复判 | 召回 1/1，误检 0/5，0.83s/张 |
| S02b | 单张检测 | PASS；耗时 1.28s（含 UI 渲染）WARN |
| S02c | 服务层量化 + 各大类同图自比 | 召回 3/3，误检 0/18，均 0.23s / 最大 0.60s |
| S03 | 历史记录：列表、统计、HUD、过滤、调试开关、删除记录 | PASS |
| S04 | 算法调试：绑定模板、加载测试图、对照跑图、多件图判 NG | PASS |
| S05 | 预处理：保存全局默认（新增影响范围确认框）、算法专属保存/清除、重置 | PASS |
| S06 | AOI_Core 健康、CoreHub 7 个子页 | PASS |
| S06b~d | 数据源导入、准备模型、版本激活与门控 | PASS |
| S06e | feature / dual 检测 3 对 × 4 case | feature 召回 2/3 FAIL，其余 PASS |
| S07 | 菜单与系统设置 | PASS |

## 4. 问题清单与处理状态

| # | 级别 | 问题 | 状态 | 处理与数据 |
|---|---|---|---|---|
| 1 | 高 | body 差分对 jpeg/噪声/亮度/平移敏感 | 已修复 | `body_adapter` 增加前置门控：phaseCorrelate 求偏移，取零偏移及 floor/ceil 整数偏移候选；5×5 高斯 + 均值/增益对齐（增益限 [0.8,1.25]）；absdiff>22 最大连通域 <15 判"无变化"。误检 9/18 → 0/18，召回保持 3/3 |
| 2 | 高 | 同图自比误报、疑似非确定性 | 已定性 | 多次复跑结果一致，不存在非确定性。报 NG 的是焊锡/起皱/移位等绝对判定类算法在通用框上运行，属于域外判定，需按对象标定 ROI；已在 `wrinkle_adapter` 补充说明文案。保留为 WARN |
| 3 | 中 | 无"多件/异物"专用算法 | 未处理 | 依赖 body 差分间接检出，门控后召回仍为 3/3 |
| 4 | 高 | 特征引擎区分度不足 | 未处理 | 2_0_1870 score 0.5742 低于 thr 0.8444，属于模型能力问题；temp 多来自 train，存在训练泄漏。建议用 test/good 独立标定阈值 |
| 5 | 高 | dual 单图超 1s | 已修复 | 根因三个：①每个算法重复做整图预处理（4 算法 × std/test = 8 次）；②`_auto_contrast` 用 float32 整图计算，占 0.61s；③测试 S05 写入的全局配方直到收尾才还原，污染后续步骤。修复：`scheduler` 同配方共用预处理结果；`engine` auto contrast 改为 LUT；`service` 标准图读取缓存；测试 S05 后立即还原配方。dual 最大 1.52s → 0.66s |
| 6 | 低 | 标准图标签只显示 `standard.jpg` | 已修复 | `store` 写入 source_name，模型增加 display_name |
| 7 | 中 | 首次准备自动激活与文案矛盾 | 已修复 | `model_page.py` / `routes_models.py` 文案改为与实际行为一致 |
| 8 | 中 | 删模板不清理 `template_algorithm\{tpl_id}__*.json` | 已修复 | `TemplateStore.delete` 级联清理；新增 `tests\test_template_delete_cascade.py` |
| 9 | 中 | Core 删除数据源后模型版本残留 | 未处理 | 同品类版本持续累积 |
| 10 | 高 | 保存全局默认配方无影响范围提示 | 已修复 | 保存前弹窗列出变更项与影响范围，确认后才写入 |
| 11 | 低 | 7 个来源不明的 `CAPA_1_10_OK___body_*.json` | 未处理 | 需确认是否为历史测试残留 |

### 测试侧修正

| 项 | 处理 |
|---|---|
| ROI 偏离 用例 3/3 误检 | 原用例直接拿独立采集的 img/temp 在框外比较，两者本身整板错位，属于测试设计问题。改为把 img 的 600×600 块贴到 temp 上合成，误检 0/3 |
| S05 全局配方污染 | 拆出 `_restore_recipe()`，S05 结束即还原 |
| 诊断代码 | 已从 S06e 移除 |

## 5. 指标对比

| 指标 | 第 5 轮（修复前） | 第 11 轮（修复后） |
|---|---|---|
| UI 批量 召回 / 误检 / 耗时 | 1/1；4/5；1.04s/张 | 1/1；0/5；0.83s/张 |
| UI 单张 耗时（含渲染） | 1.63s | 1.28s |
| 服务层 traditional 召回 / 误检 | 3/3；9/18 | 3/3；0/18 |
| 服务层 traditional 耗时 | 均 0.46s，最大 0.64s | 均 0.23s，最大 0.60s |
| feature 召回 / 误检 / 最大耗时 | 2/3；0/9；0.54s | 2/3；0/9；0.48s |
| dual 召回 / 误检 / 最大耗时 | 3/3；4/9；1.52s | 3/3；0/9；0.66s |
| 单元测试 | 6 | 7，全部 OK |

单轮只抽 3 对，样本量小，准确率结论需固定多个 SEED 取平均。

## 6. 复现

```powershell
cd e:\CPIPC\CGAIC\AOI_tree\PCB_Dual
$env:PYTHONIOENCODING="utf-8"
$env:UI_TEST_SEED="1791341606"
e:\CPIPC\CGAIC\.env\Scripts\python.exe -u tests\_ui_full_datalocal.py 2>&1 | Tee-Object -FilePath $env:TEMP\ui_run.log
e:\CPIPC\CGAIC\.env\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

- 需要 AOI_Core（:8017）可用；存在 FAIL 时退出码为 1。
- 不设 SEED 时按当前时间随机抽样。

## 7. 后续建议

1. #4 特征引擎：用 test/good 做独立验证集重新标定阈值，排除训练泄漏后再评估召回。
2. #2 / #3：绝对判定类算法按对象标定 ROI；评估是否新增"多件/异物"专用算法。
3. #9 / #11：Core 删除数据源时级联删除模型版本；确认并清理来源不明的参数文件。
4. UI 单张 1.28s 超出 1s，服务层同类检测仅约 0.2~0.6s，差值主要在 UI 路径（渲染/轮询），如需达标可再拆分计时排查。
