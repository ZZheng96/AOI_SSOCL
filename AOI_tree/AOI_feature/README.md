# AOI_feature：特征分析与管理子系统（AOI_Core 内置）

一站式回答三个问题：

1. **为什么选择这些特征？** —— [feature_lib.py](feature_lib.py) 注册表逐组给出选择理由与目标缺陷类型；评估汇总报告附录即"特征选择依据"文档。
2. **这些特征实际起了什么作用？** —— 组级贡献评估三层证据：独立 AUROC（单独行不行）、留一消融（对整体是否不可替代）、排列重要性（打乱它整体掉多少）。
3. **如何管理特征集合？** —— GUI（`app.py`）或 CLI（`manage.py`）对 `feature_set.yaml` 增删改查，改完重跑评估即得新贡献报告。

## GUI（推荐入口）

```bash
python app.py
```

三个页签：

- **特征集管理**：组清单（启停勾选/维度/参数改动标记）、单组详情（选择理由/目标缺陷）、参数表格编辑（YAML 语法，应用即写回配置）、自定义特征下拉加入、删除、恢复主干预设、parity 校验按钮
- **单图分析**：选图即提取特征向量（按组着色条形图）；可选加载正常参考集文件夹 → 组级 mean\|z\| 对比图（>3 即显著偏离），定位"这张图哪类特征异常"
- **贡献评估**：选数据集/品类/参数 → 后台线程运行（不卡界面）→ 结果表格 + 三层证据三联图，报告同时落盘 `outputs/`

线程规范：评估走 `EvalWorker`（QThread），关窗时置中断标志并 join（有界等待 3s）。

## 与主干的关系（同源红线）

| 层 | 来源 | 说明 |
|---|---|---|
| 特征提取 | `feature_lib.py`（主干单体实现的参数化移植） | 主干预设经 `manage.py parity` **逐图校验与主干逐位一致** |
| 打分口径 | trad 槽位同款 U27 播种 / U36 z-clip ±10 / U99 跳过自身最近邻 | `scoring.py` |
| 数据隔离 | `algo/common/isolation.guard_bundle` | fit 只用 train 域 init_normal |
| 指标 | `algo/eval/metrics.image_metrics` | AUROC/AP 同一实现 |

诚实评测口径：AUROC **不做方向翻转**；负贡献如实登记；test 只验不选。

## 特征集合管理（增删改查）

`feature_set.yaml` 是特征集合的唯一事实来源（缺省时自动用主干预设初始化）：

```bash
python manage.py list                     # 清单：组名/维度/启停状态/参数改动
python manage.py show glcm                # 单组详情：选择理由/目标缺陷/参数
python manage.py disable hog              # 停用不好的组（保留配置，可恢复）
python manage.py enable hog               # 恢复
python manage.py set glcm levels=64       # 修改计算参数（值按 YAML 解析类型）
python manage.py add lap_var              # 加入自定义特征（custom_features.py 定义）
python manage.py remove lap_var           # 删除组
python manage.py reset                    # 恢复主干预设 trunk_200
python manage.py parity                   # 与主干 extractor 逐图校验一致
```

**自定义特征**：在 [custom_features.py](custom_features.py) 中用 `@feature` 装饰器定义（一个函数 + 默认参数 + 理由），`manage.py add` 即入列，评估/报告链路自动接纳。维度不做静态登记——载入时对探针图实测，维度永远与实际输出一致。

> 注意：停用/改参会改变特征向量维度与数值，评估结论对应的是**当前配置**；
> 与主干生产管线的一致性以 `manage.py parity` 为准（预设一致时结论可直接迁移回 trad 槽位）。

## 贡献评估

```bash
# 合成数据冒烟（无真实数据时验证管线端到端）
python run_eval.py --dataset synthetic

# 真实数据集（category 缺省=该数据集全品类）
python run_eval.py --dataset mvtec --root e:/CPIPC/CGAIC/data_origin/mvtec --category bottle
python run_eval.py --dataset datalocal --root <data_local根> --category solder_smt
python run_eval.py --dataset btad --root <BTAD根> --category all
```

输出到 `outputs/`：`{品类}_contribution.json` + 三张 PNG（独立 AUROC / 留一损失 / 排列重要性）+ `summary.md`（跨品类矩阵 + 当前特征集选择理由附录）。

典型工作流：`run_eval` 看贡献 → `manage.py disable/set` 调整特征集 → 重跑 `run_eval` 对比 → 满意后如需回主干生产管线，把同样改动落到 `vendor/traditional.py` 并跑 `parity` 确认。

## 历史

前身是独立仓库 `AOI_feature/`（GUI 调参 + 旧 186 维特征体系），特征实现已与主干分叉且不可运行（路径硬编码 `d:\CGAIC`）。本模块按 P12 方案 (a) 重建，特征/打分/评测全部主干同源，并补上特征集管理能力。**独立仓库已于 2026-10-02 删除**，本模块为唯一特征分析入口。
