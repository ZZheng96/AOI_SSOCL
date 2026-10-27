# 第四轮：待决策事项推荐值落地

## 运行时配置文件
- `backend/core/config.py`：`Settings` 读取 `AOI_CONFIG` 环境变量，如果未设置则读取 `AOI_ROOT/configs/default.yaml`。当前环境没有设置 `AOI_CONFIG`。
- 开发版实际读取的是 `d:\CGAIC\AOI_SSOCL\AOI_tree\AOI_Core\configs\default.yaml`。设置页的“保存”也通过 `Settings.update` 写回这个文件。
- 原文件没有 `governance_decisions` 段，13 项全部使用代码默认值。本次只在 default.yaml 末尾追加这一段。未修改 config.py、生产 DB、快照，也未修改 `configs/docker.yaml`（docker 部署配置）。

## 写入值
| 键 | 值 | 说明 |
|---|---|---|
| dependency_lock_policy | warn | 锁文件尚未修好，保持现状 |
| candidate_release_mode | manual | |
| calibration_isolation | auto | |
| ac045_l1_real_training_enabled | false | |
| ac046_l1_pollution_exit_enabled | false | |
| blob_gpu_enabled | false | 全量复核前不开 |
| approved_extra_good_batches | `Image_2026,t_fov` | spec 类型为 str，逗号分隔（`parse_batch_list` 按 `,` 切分并去空白）；不能写成 YAML 列表，否则类型校验不通过，会回落为空 |

## 保持默认（未写入）
| 键 | 默认 | 原因 |
|---|---|---|
| gt_threshold_frozen | false | GT 尚未确认，不能代为冻结 |
| gt_dataset_ref | "" | GT 数据集尚未确定 |
| gt_threshold_frozen_at | "" | 冻结时由服务端自动写入，只读 |
| sync28_29_accepted | false | 属于验收对话的职责 |
| production_acceptance_signed | false | 需要真实生产验收签署 |
| ac062_legacy_artifact_dir | "" | 目前还没有旧 trad 协议工件 |

## 验证
- 在 AOI_Core 下设置 `QT_QPA_PLATFORM=offscreen`，用 Python 3.11 加载 `get_settings().governance_decisions()`：config_path 为 configs/default.yaml，7 项与上表一致，其余 6 项为默认值；`parse_batch_list` 结果为 `['Image_2026', 't_fov']`。
- `python -m unittest tests.smoke.test_round3_decisions tests.smoke.test_round3_coverage -v`：20 个测试全部通过（这些测试使用临时 yaml，不受 default.yaml 影响，默认值表仍然锁定在 config.py）。
- 未提交 git。注意：在本次修改之前，`configs/default.yaml` 在 git 中已处于修改状态（他人的未提交改动），本次只追加了末尾一段。
