"""SolderSmtAllAlg 单元测试。

测试覆盖：
1. 契约基础：default_config / algorithm_code 一致性
2. JSON 加载：从配置文件加载后 _defaults 正确
3. 算法行为：cold_solder 检出
4. 异常处理：空图返回 code=1

运行（项目根）:
    set PYTHONUTF8=1
    python -m pytest test_src/algorithms/solder_smt/solder_smt_all_alg_test.py -q
"""
from __future__ import annotations

import os
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import cv2

from algorithms.solder_smt.composite_smt_all.solder_smt_all_alg import SolderSmtAllAlg
from core.entities.detect import ResultType

_THIS = Path(__file__).resolve().parents[2]
_SRC_ROOT = _THIS.parent / "src"
_INSTANCE_CONFIG = _SRC_ROOT / "resources" / "config" / "alg" / "solder_smt" / "solder_smt_all_alg.json"

# 测试图片
_TEST_IMAGES = _THIS.parent / "test_images"
_COLD_PAIR1 = _TEST_IMAGES / "cold_solder_pair" / "pair1"


@contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _load_pad_frames(pad_json_path: str):
    """从 pad.json 加载焊盘框"""
    with open(pad_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pads = []
    for s in data.get("shapes", []):
        if s["label"] == "pad" and s["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = s["points"]
            pads.append((int(min(x1, x2)), int(min(y1, y2)),
                         int(abs(x2 - x1)), int(abs(y2 - y1))))
    return pads


def _get_cold_pair1_images():
    """获取 cold_solder pair1 的测试图片和配置"""
    ng_imgs = list(_COLD_PAIR1.glob("*.png"))
    ng_imgs = [p for p in ng_imgs if "_OK" not in p.name]
    ok_imgs = list(_COLD_PAIR1.glob("*_OK.png"))
    if not ng_imgs or not ok_imgs:
        pytest.skip("cold_solder pair1 test images not found")
    return ng_imgs[0], ok_imgs[0]


# ---- 契约基础 ----

def test_default_config_and_code():
    alg = SolderSmtAllAlg()
    assert alg.algorithm_code == "SolderSmtAllAlg"
    cfg = alg.default_config()
    assert "solder_blue_h_low" in cfg
    assert "insufficient_thresh" in cfg
    assert cfg["detect_excess"] is True


def test_load_from_json_config():
    if not _INSTANCE_CONFIG.is_file():
        pytest.skip(f"config not found: {_INSTANCE_CONFIG}")
    alg = SolderSmtAllAlg(str(_INSTANCE_CONFIG))
    assert alg.algorithm_code == "SolderSmtAllAlg"
    # JSON 中的 defaultParam 应该被加载到 _defaults
    assert "solder_blue_h_low" in alg._defaults


# ---- 算法行为 ----

def test_detects_cold_solder():
    """测试 cold_solder 检出"""
    ng_path, ok_path = _get_cold_pair1_images()
    img = cv2.imread(str(ng_path))
    tpl = cv2.imread(str(ok_path))
    assert img is not None, f"cannot read: {ng_path}"

    pad_json = _COLD_PAIR1 / "pad.json"
    assert pad_json.is_file(), f"pad.json not found: {pad_json}"
    pad_frames = _load_pad_frames(str(pad_json))

    alg = SolderSmtAllAlg()
    cfg = {
        "pad_frames": pad_frames,
        "pad_json_path": str(pad_json),
    }
    result = alg.run(img, cfg, original_template_image=tpl)

    assert result.code == 0, f"run failed: {result.message}"
    assert result.result_type == ResultType.PARTS
    labels = {p.label for p in result.parts}
    assert "cold_solder" in labels, f"cold_solder not detected, got: {labels}"
    # 可视化图
    vis = result.metadata.get("output_image")
    assert vis is not None and vis.dtype == np.uint8


def test_config_override_disables_bridge():
    """测试关闭 bridge 检测"""
    ng_path, ok_path = _get_cold_pair1_images()
    img = cv2.imread(str(ng_path))
    tpl = cv2.imread(str(ok_path))

    pad_json = _COLD_PAIR1 / "pad.json"
    pad_frames = _load_pad_frames(str(pad_json))

    alg = SolderSmtAllAlg()
    cfg = {
        "pad_frames": pad_frames,
        "pad_json_path": str(pad_json),
        "detect_bridge": False,
    }
    result = alg.run(img, cfg, original_template_image=tpl)
    assert result.code == 0
    assert all(p.label != "bridge" for p in result.parts)


# ---- 异常处理 ----

def test_invalid_inputs_return_code_1():
    alg = SolderSmtAllAlg()
    assert alg.run(None, {}).code == 1
    assert alg.run(np.zeros((0, 0), np.uint8), {}).code == 1


def test_no_pad_frames_returns_code_1():
    alg = SolderSmtAllAlg()
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    result = alg.run(img, {})
    assert result.code == 1
    assert "pad_frames" in result.message
