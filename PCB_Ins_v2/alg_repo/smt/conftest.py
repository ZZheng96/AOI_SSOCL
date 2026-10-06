"""pytest 全局配置: 自动设置 PYTHONPATH"""
import sys, os

_HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("src", "contract_reference"):
    p = os.path.join(_HERE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
