"""评测指标：图像级 AUROC / AP（无泄漏口径）"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def image_metrics(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "ap": float("nan")}
    return {"auroc": float(roc_auc_score(labels, scores)),
            "ap": float(average_precision_score(labels, scores))}
