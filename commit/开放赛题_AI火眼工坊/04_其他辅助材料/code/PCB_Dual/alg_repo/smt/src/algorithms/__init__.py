"""PCBA 检测系统 - 算法模块

仅包含 SolderSmtAllAlg 算法。
"""

# 算法名 -> 类 映射（仅本包包含的算法）
alg_name2ext = {}

try:
    from algorithms.solder_smt.composite_smt_all.solder_smt_all_alg import SolderSmtAllAlg
    alg_name2ext["SolderSmtAllAlg"] = SolderSmtAllAlg
except ImportError:
    pass
