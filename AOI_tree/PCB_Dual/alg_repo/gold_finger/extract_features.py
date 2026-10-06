# -*- coding: utf-8 -*-
"""
解析 X-AnyLabeling 标注 JSON，逐缺陷区域提取手工特征，输出特征数据集。

用法:
    python extract_features.py [数据目录]     # 默认 image_test

约定: 图片与 JSON 同名同目录 (0598.json <-> 0598.png / 0598.jpg)。
输出: features_dataset.npz  (X 特征矩阵 / y 标签 / 来源信息，供 train_classifier.py 使用)
"""
import json
import os
import sys

import cv2
import numpy as np

from defect_features import FEATURE_NAMES, crop_region, extract_region_features

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def find_image(json_path: str, image_path_hint: str) -> str:
    """优先用 JSON 里的 imagePath，找不到再按同名不同扩展名搜。"""
    folder = os.path.dirname(json_path)
    if image_path_hint:
        p = os.path.join(folder, os.path.basename(image_path_hint))
        if os.path.exists(p):
            return p
    stem = os.path.splitext(os.path.basename(json_path))[0]
    for ext in IMAGE_EXTS:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return ""


def imread_unicode(path: str):
    """cv2.imread 在 Windows 上读不了含中文路径，用 imdecode 兜底。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "image_test")
    out_path = os.path.join(os.path.dirname(__file__), "features_dataset.npz")

    json_files = sorted(
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".json")
    )
    print(f"共找到 {len(json_files)} 个标注文件: {data_dir}")

    X, y, sources = [], [], []
    n_shape_total, n_skip = 0, 0
    label_counts = {}

    for jp in json_files:
        with open(jp, "r", encoding="utf-8") as f:
            ann = json.load(f)

        img_path = find_image(jp, ann.get("imagePath", ""))
        if not img_path:
            print(f"  ⚠️ 找不到对应图片，跳过: {os.path.basename(jp)}")
            continue
        image = imread_unicode(img_path)
        if image is None:
            print(f"  ⚠️ 图片读取失败，跳过: {os.path.basename(img_path)}")
            continue

        long_side = max(image.shape[:2])
        for si, shape in enumerate(ann.get("shapes", [])):
            if shape.get("shape_type") != "polygon":
                continue
            n_shape_total += 1
            label = shape["label"]

            roi, roi_mask = crop_region(image, np.array(shape["points"]))
            if roi is None:
                n_skip += 1
                continue
            feats = extract_region_features(roi, roi_mask, long_side)
            if feats is None:
                n_skip += 1
                continue

            X.append(feats)
            y.append(label)
            sources.append(f"{os.path.basename(jp)}#{si}")
            label_counts[label] = label_counts.get(label, 0) + 1

    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    print(f"\n有效样本 {len(y)} / 标注 {n_shape_total} (跳过 {n_skip} 个退化区域)")
    print("类别分布:")
    for k in sorted(label_counts):
        print(f"  {k:10s} {label_counts[k]}")

    np.savez(
        out_path,
        X=X, y=y,
        sources=np.array(sources),
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"\n特征维度: {X.shape[1]} (与 FEATURE_NAMES 对应)")
    print(f"已保存数据集: {out_path}")


if __name__ == "__main__":
    main()
