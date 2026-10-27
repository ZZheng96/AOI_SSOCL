# AOI_Core 全流程录制脚本（MPDD · tubes）

系统：`D:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core`；数据：`D:\CGAIC\data_origin\MPDD\tubes`。

## 硬约束与真实证据

- 成片不超过60秒。保留正片开头工作台原速3秒，再去数据管理。普通操作逐项间隔1秒；系统加载、任务响应和真实结果绘制另行等待。原始素材可长于60秒。
- 最大化主窗口，仅通过 `QWidget.grab()` 获取主窗口及其所属弹窗并合成，不采集桌面、屏幕或其他软件。
- 实录写入 `TEMP/AOI_window_clips/<时间戳>`，编码、字幕和剪辑缓存也在TEMP。旧素材及成片不覆盖。
- 每次真实创建唯一MPDD tubes数据源和新工单。精确绑定返回的 `DS_ID`、`WO_ID`、`FEEDBACK_ID`；只导入CATEGORY=tubes，不选旧源或旧工单。03_imported不再提前选择旧队列。
- 包装原始client.post，保留实际业务请求，捕获本次task_id。异步导入、三维、准备、档案、评估、候选均等待对应任务done。失败、超时、无ID立即停止，失败镜头隔离TEMP，清单complete=false，剪辑拒绝合成。
- 对模态窗和数据源列表、候选反馈、反馈记录等待就绪后继续，避免盲点。模态exec前安排后续QTimer，三维与候选窗口跨镜保活。
- 三维必须真实样本/散点存在，旋转后点选真实artist，走实际pick_event回调并核验缩略图。
- tubes当前生产使用sem/disc/shead，无trad槽位；其特征页原右图无传统档案。可将本次贡献任务真实single与leave_one_out AUROC交给原BarChart显示，明确标注槽位小样本消融、非逐图传统贡献；不得编造trad_groups或逐图数值。
- 误检只能选择本次工单tubes预测异常、图片真值normal的持久检测记录。查最近20条无真实误检则停止，不修改真值、不虚构反馈。
- 评估限制12张，贡献及三维使用有界样本，不运行全量调优。用户规则单图速度红线1秒，生产原路径不增加重复前向。
- 候选精确消费本次反馈，核对父版本和图片真值，构建后显示任务返回真实版本；不批准、不激活。独立验收六项政策中max_predict_ms与valid_seconds无已确认值，只显示入口和阻碍，不能造值提交。
- 学习图使用真实翻案/候选权重/线上曲线/错检/趋势响应并校验非空，不冒充学习提升或已投产。响应保存在TEMP。

## 22镜及11页面覆盖

各镜沿用源码SHOTS名称和交互顺序，但不再使用旧固定长偏移；首镜引导2秒后工作台3秒，其他镜引导2秒后开始，普通操作间隔1秒。每次实际操作时刻写入manifest.json，异步结果校验后收镜。

| 镜头 | 页面 | 实际内容 |
|---|---|---|
| 01_new_source | 工作台→数据管理 | 工作台3秒，创建新MPDD tubes数据源 |
| 02_probe_confirm | 数据管理 | 智能识别，仅勾tubes，确认导入 |
| 03_imported | 数据管理 | 等本次导入，选本次源tubes树节点，不提前选旧队列 |
| 03b_augment | 数据增强 | 本次源正常基底，真实合成、预处理，提交三维 |
| 03c_space_done | 数据增强 | 等真实散点，旋转并点选缩略图 |
| 04_workorder | 工作台 | 创建新工单，仅挂本次DS_ID |
| 05_prepare | 模型管理 | 精确选择本次源tubes，提交准备 |
| 06_prepared | 模型管理 | 等本次准备，展示真实版本 |
| 06b_feature | 特征工程 | tubes提交贡献档案 |
| 06c_feature_done | 特征工程 | 等本次档案，展示真实右图；无trad时如实显示槽位AUROC消融，不伪造逐图 |
| 07_eval | 评估看板 | tubes真实评估limit=12 |
| 08_eval_done | 评估看板 | 等本次结果，选本次唯一评估名 |
| 09_monitor | 实时监控 | 恢复产线，动态热力图及叠加图 |
| 10_feedback | 实时监控 | 核验真实误检，填写并提交反馈 |
| 11_review | 标注反馈 | 精确选择本次FEEDBACK_ID |
| 12_learn | 实时监控 | 暂停并固化真实快照 |
| 12b_candidate | 标注反馈 | 本次反馈绑定父版本，构建未激活候选 |
| 12c_acceptance_status | 标注反馈 | 本次候选状态、独立验收政策阻碍 |
| 13_learning | 学习效果 | 真实翻案/权重/线上图，不宣称学习改善 |
| 14_stats | 存档统计 | 本次工单批次、趋势、帕累托、延迟，空值NA |
| 14b_settings | 系统设置 | 临时切换并恢复，不保存；反馈日志 |
| 15_requeue | 数据管理 | tubes真实批次回队，禁止fallback其他品类 |

## 执行与剪辑

```powershell
$env:AOI_DEMO_DATASET='mpdd'; $env:AOI_DEMO_CATEGORY='tubes'
python tests\_record_shots_btad.py
python D:\CGAIC\AOI_SSOCL\practice\BTAD\build_video.py <本次TEMP素材目录> D:\CGAIC\AOI_SSOCL\practice\MPDD\MPDD_tubes_60s_<新时间戳>.mp4
```

剪辑必须先读本次manifest.json，要求complete=true、issues为空、22镜全覆盖。裁掉各镜前2秒引导，首镜6.5秒且前3秒原速，其余21镜各2.5秒，按实际素材长度合理加速，目标59秒。统一1440×900、30fps，字幕明确真实证据边界；不复用旧镜头、不合成失败素材。ffprobe强制核验58.5–60秒；最终仍须检查关键帧、动态热力图及全部页面后才能报告成功。
