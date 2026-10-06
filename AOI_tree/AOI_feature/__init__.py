"""AOI_feature：AOI_Core 内置特征分析与管理子系统

三个问题的一站式答案：
  1. 为什么选择这些特征？—— feature_lib.py 注册表逐组给出理由与目标缺陷
  2. 这些特征实际起了什么作用？—— contribution.py 组级贡献评估
     （独立 AUROC / 留一消融 / 排列重要性）
  3. 如何管理特征集合？—— manage.py 对 feature_set.yaml 增删改查
     （启停/改参/自定义增删），改完重跑 run_eval.py 即得新贡献报告

与主干同源红线：默认配置（presets.trunk_preset）与
algo.vendor.traditional.TraditionalFeatureExtractor 逐位一致（manage.py parity
校验）；打分口径复用 trad 槽位 U27/U36/U99；评测过 isolation.guard_bundle。
"""
import sys
from pathlib import Path

AOI_CORE = Path(__file__).resolve().parent.parent / "AOI_Core"


def ensure_core_importable():
    """把 AOI_Core 加入 sys.path（幂等），之后可 `from algo.xxx import ...`。"""
    p = str(AOI_CORE)
    if p not in sys.path:
        sys.path.insert(0, p)
    return AOI_CORE
