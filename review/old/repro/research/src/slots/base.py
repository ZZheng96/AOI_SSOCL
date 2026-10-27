"""槽位统一契约（§3.3/§4.4）：热插拔的技术前提。

每个槽位实现：
  fit(train_feats, train_imgs, defect_feats)  仅用 train/good(+协议内缺陷) 训练
  score_image(feats, img) -> SlotOutput       tile 级打分 + block 聚合
  update(...)               在线更新接口（M0 默认 no-op）
SlotOutput: image_score, tile_scores, heatmaps(每 tile 一张), confidence
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SlotOutput:
    image_score: float
    tile_scores: np.ndarray          # (T,)
    heatmaps: list = field(default_factory=list)   # list[(G,G) float]，每 tile
    confidence: float = 1.0


class Slot:
    name = "base"
    version = "0.1.0"
    needs_dino = True          # 主干特征依赖声明（深插拔边界，§4.4 硬约束二）

    def fit(self, ctx):
        raise NotImplementedError

    def score_tiles(self, tile_feats, tile_imgs):
        """返回 (tile_scores (T,), heatmaps list[(G,G)])"""
        raise NotImplementedError

    def score_image(self, img):
        """可选整图打分（tile 级聚合信噪比不足时用，如 trad U27）。
        返回 (image_score, tile_scores, heatmaps) 或 None（回退 score_tiles 路径）。"""
        return None

    def update(self, *args, **kwargs):
        return None
