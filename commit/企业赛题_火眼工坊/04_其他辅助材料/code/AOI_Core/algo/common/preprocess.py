"""图像预处理（训练/检测同口径，§15.29）

配置来源 cfg["augment"]["preprocess"]：
- enabled: 总开关（关 = 原图直通）
- gray: 灰度化（复制为 3 通道，保持下游 backbone 3 通道输入约定）
- clahe: >0 启用 CLAHE 限制对比度自适应直方图均衡（LAB 空间 L 通道，值=clipLimit）
- median: 0/3/5 中值滤波核（去椒盐噪点）

适用场景：光照不均（CLAHE）、彩色信息无判别价值（灰度）、传感器噪点（中值）。
所有算子为 O(像素) CPU 操作，单图几 ms~几十 ms，1s/图红线内。
训练（Pipeline._tile_and_extract）与检测（predict/predict_frame）读图后同口径调用，
确保两侧分布一致。
"""
import cv2


def preprocess_image(img, cfg):
    """img: HxWx3 uint8 RGB -> 预处理后 HxWx3 uint8 RGB；cfg 为空或未启用时原样返回。"""
    if not cfg or not cfg.get("enabled"):
        return img
    out = img
    if cfg.get("gray"):
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    clip = float(cfg.get("clahe") or 0)
    if clip > 0:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        lab[..., 0] = clahe.apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    k = int(cfg.get("median") or 0)
    if k >= 3:
        out = cv2.medianBlur(out, k if k % 2 else k + 1)
    return out
