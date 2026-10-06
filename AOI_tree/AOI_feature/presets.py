"""主干预设：与 algo/vendor/traditional.py TraditionalFeatureExtractor 逐位一致

默认参数即主干行为（经 manage.py parity 逐图校验）。维度合计 200。
"""
from .feature_lib import FEATURE_FUNCS

TRUNK_ORDER = ["glcm", "selfsim", "edge", "hog", "shape", "gray", "gloss",
               "color", "fft", "wavelet", "logic", "gabor", "log", "pc", "surf"]


def trunk_preset() -> dict:
    """返回主干 200 维配置（dict，可 yaml 序列化）"""
    return {
        "name": "trunk_200",
        "version": 1,
        "preprocess": {"max_side": 256, "min_side": 32},
        "groups": [{"key": k, "enabled": True,
                    "params": dict(FEATURE_FUNCS[k]["default_params"])}
                   for k in TRUNK_ORDER],
    }
