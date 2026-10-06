"""自定义特征：在此用 @feature 装饰器定义，然后 manage.py add <key> 加入配置

函数签名：f(ctx, params) -> list[float]
  ctx = {"img": RGB, "gray": 灰度, "hsv": HSV}（已按配置 preprocess 缩放）
  params = default_params 与 feature_set.yaml 中该组 params 的合并

注册 ≠ 启用：这里的函数仅登记进函数库，须 `manage.py add <key>` 才进特征集。
以下为示例特征（默认不在特征集内）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from AOI_feature.feature_lib import feature  # noqa: E402


@feature("lap_var", "自定义·拉普拉斯方差",
         "整图清晰度/离焦代理量", "离焦、运动模糊",
         {"scale": 1.0})
def lap_var(ctx, p):
    import cv2
    return [float(cv2.Laplacian(ctx["gray"], cv2.CV_32F).var() * p["scale"])]
