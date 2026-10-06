"""PcbThroughHoleSolderAlg 单元测试"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import pytest

from algorithms.pcb_through_hole.pcb_through_hole_solder_alg import PcbThroughHoleSolderAlg
from core.entities.detect import BoundingBox, ResultType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC = _REPO_ROOT / "src"
_INSTANCE_CONFIG = _SRC / "resources/config/alg/pcb_through_hole/pcb_through_hole_solder_alg.json"
_POOL_CONFIG = _SRC / "resources/config/alg_pool/pcb_through_hole_algs.json"
_INPUT_DIR = _REPO_ROOT / "input"


@contextmanager
def _chdir(path: Path):
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


def _std(num: str) -> np.ndarray:
    img = cv2.imread(str(_INPUT_DIR / "standard_roi" / f"templ_{num}.png"))
    assert img is not None
    return img


def _test(name: str) -> np.ndarray:
    img = cv2.imread(str(_INPUT_DIR / "roi" / f"{name}.png"))
    assert img is not None
    return img


# ---- 契约基础 ----

def test_default_config_and_code():
    alg = PcbThroughHoleSolderAlg()
    assert alg.algorithm_code == "pcb_through_hole_solder_defect"
    assert alg.default_config()["enabled_defects"] == [12, 13, 14, 15, 16]
    assert "bridge_joints" in alg.default_config()


def test_load_from_json_config():
    assert _INSTANCE_CONFIG.is_file(), _INSTANCE_CONFIG
    alg = PcbThroughHoleSolderAlg(str(_INSTANCE_CONFIG))
    assert alg._defaults == PcbThroughHoleSolderAlg.default_config()
    assert alg.algorithm_code == "pcb_through_hole_solder_defect"


def test_pool_registration_json_matches_instance_config():
    assert _POOL_CONFIG.is_file(), _POOL_CONFIG
    pool_cfg = json.loads(_POOL_CONFIG.read_text(encoding="utf-8"))
    entry = pool_cfg["cv_algorithms"][0]
    assert entry["algorithm_code"] == PcbThroughHoleSolderAlg.algorithm_code
    assert entry["class_path"] == (
        "algorithms.pcb_through_hole.pcb_through_hole_solder_alg.PcbThroughHoleSolderAlg"
    )
    instance_cfg = json.loads(_INSTANCE_CONFIG.read_text(encoding="utf-8"))
    assert instance_cfg["algorithmCode"] == entry["algorithm_code"]


# ---- 接入链路：注册 → 执行（宿主工程有 AlgorithmPool 时跑通；独立仓库则 skip）----

def test_plugs_into_algorithm_pool():
    AlgorithmPool = pytest.importorskip(
        "alg_process.alg_manager.detect_alg_pool",
        reason="独立交付包无宿主算法池；接入检测工程后应能跑通",
    ).AlgorithmPool

    assert _POOL_CONFIG.is_file(), _POOL_CONFIG
    with _chdir(_SRC):  # 池配置里的 config_path 相对 src 解析
        pool = AlgorithmPool()
        registered = pool.load_from_config(str(_POOL_CONFIG), section="cv_algorithms")
        assert "pcb_through_hole_solder_defect" in registered
        out = pool.execute_algorithm(
            algorithm_code="pcb_through_hole_solder_defect",
            image=_test("2-1"),
            config={},
            original_template_image=_std("2"),
        )
    assert out is not None, "算法池返回 None：通常是 run() 抛了异常或方法名不对"
    assert out.code == 0
    assert out.result_type == ResultType.PARTS
    assert out.metadata.get("status") in ("OK", "NG", "REVIEW")


# ---- 算法行为：真实模板/测试图 ----

def test_ok_case_returns_no_parts():
    alg = PcbThroughHoleSolderAlg()
    result = alg.run(_test("2-1"), {}, original_template_image=_std("2"))
    assert result.code == 0
    assert result.result_type == ResultType.PARTS
    assert result.metadata["status"] == "OK"
    assert result.parts == []
    vis = result.metadata["output_image"]
    assert vis.shape == _test("2-1").shape and vis.dtype == np.uint8


def test_insufficient_defect_detected_with_box_in_bounds():
    alg = PcbThroughHoleSolderAlg()
    test_img = _test("2-8")
    result = alg.run(test_img, {}, original_template_image=_std("2"))
    assert result.code == 0
    assert result.metadata["status"] == "NG"
    assert 13 in result.metadata["defect_ids"]
    assert len(result.parts) >= 1
    th, tw = test_img.shape[:2]
    for p in result.parts:
        # 坐标系契约：框相对输入 image，左上角+宽高，必须落在图像范围内
        assert 0 <= p.x <= tw and 0 <= p.y <= th
        assert p.x + p.width <= tw and p.y + p.height <= th
        assert p.box_type == "defect"
        assert p.class_id == 13


def test_void_defect_detected():
    alg = PcbThroughHoleSolderAlg()
    result = alg.run(_test("9-1"), {}, original_template_image=_std("9"))
    assert result.code == 0
    assert result.metadata["status"] == "NG"
    assert 12 in result.metadata["defect_ids"]
    assert any(p.class_id == 12 for p in result.parts)


def test_enabled_defects_override_filters_result():
    alg = PcbThroughHoleSolderAlg()
    result = alg.run(
        _test("2-8"), {"enabled_defects": [12, 14, 16]},
        original_template_image=_std("2"),
    )
    assert result.code == 0
    # 13 (少锡) 已被 enabled_defects 关闭，不应出现在结果中
    assert 13 not in result.metadata["defect_ids"]
    assert all(p.class_id != 13 for p in result.parts)


def test_enabled_defects_empty_list_disables_all_defects():
    alg = PcbThroughHoleSolderAlg()
    result = alg.run(
        _test("9-1"), {"enabled_defects": []},
        original_template_image=_std("9"),
    )
    assert result.code == 0
    # 显式传入空列表表示关闭全部缺陷类型，不应退回模板默认列表
    assert result.metadata["defect_ids"] == []
    assert result.parts == []


def test_roi_bbox_offsets_output_coordinates():
    alg = PcbThroughHoleSolderAlg()
    test_img = _test("2-8")
    th, tw = test_img.shape[:2]
    padded = np.zeros((th + 40, tw + 40, 3), dtype=np.uint8)
    padded[20:20 + th, 20:20 + tw] = test_img
    bbox = BoundingBox(box_id=1, box_type="pad", x=20, y=20, width=tw, height=th)

    baseline = alg.run(test_img, {}, original_template_image=_std("2"))
    offset = alg.run(padded, {}, roi_bbox=bbox, original_template_image=_std("2"))

    assert offset.code == 0
    assert len(offset.parts) == len(baseline.parts) >= 1
    for base_p, off_p in zip(baseline.parts, offset.parts):
        # roi_bbox 平移后坐标应整体 +offset；掩膜外接框允许小幅离散差
        assert abs(off_p.x - (base_p.x + 20)) <= 2
        assert abs(off_p.y - (base_p.y + 20)) <= 2
    assert offset.metadata["roi_bbox"] == {"x": 20, "y": 20, "width": tw, "height": th}

    # output_image 必须与 parts 用同一套坐标系：整张 padded image 尺寸，
    # 而不是只返回 roi_bbox 裁剪出的那一小块 (否则叠加 parts 框会全部错位)。
    vis = offset.metadata["output_image"]
    assert vis.shape == padded.shape
    assert offset.metadata["image_shape"] == [padded.shape[0], padded.shape[1]]


# ---- 人工框选 (solder_mode="roi") 模式 ----

_TEMPL_2_ROI = {"shape": "rect", "x": 5, "y": 5, "w": 90, "h": 90}


def _shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """把图像内容整体平移 (dx, dy) 像素，画布尺寸不变，边界用最近像素填充。"""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def test_roi_mode_requires_solder_roi():
    alg = PcbThroughHoleSolderAlg()
    result = alg.run(_test("2-1"), {"solder_mode": "roi"}, original_template_image=_std("2"))
    assert result.code == 1
    assert "solder_roi" in result.message


def test_roi_mode_invalid_roi_rejected():
    alg = PcbThroughHoleSolderAlg()
    bad_rois = [
        {"shape": "rect", "x": 5, "y": 5, "w": 0, "h": 90},       # 尺寸为 0
        {"shape": "rect", "x": 5, "y": 5},                        # 缺 w/h
        {"shape": "triangle", "x": 5, "y": 5, "w": 10, "h": 10},  # 不支持的形状
        {"shape": "circle", "cx": 50, "cy": 50, "r": -1},         # 半径非正
    ]
    for roi in bad_rois:
        result = alg.run(
            _test("2-1"), {"solder_mode": "roi", "solder_roi": roi},
            original_template_image=_std("2"),
        )
        assert result.code == 1, roi
        assert result.result_type == ResultType.ERROR, roi


def test_roi_mode_runs_end_to_end_ok_and_ng():
    alg = PcbThroughHoleSolderAlg()
    std_img = _std("2")

    ok_result = alg.run(
        _test("2-1"), {"solder_mode": "roi", "solder_roi": _TEMPL_2_ROI},
        original_template_image=std_img,
    )
    assert ok_result.code == 0
    assert ok_result.metadata["status"] in ("OK", "NG", "REVIEW")

    ng_result = alg.run(
        _test("2-8"), {"solder_mode": "roi", "solder_roi": _TEMPL_2_ROI},
        original_template_image=std_img,
    )
    assert ng_result.code == 0
    for p in ng_result.parts:
        assert p.box_type == "defect"
        assert 0 <= p.x and 0 <= p.y


def test_roi_mode_cache_rebuilds_when_roi_changes():
    alg = PcbThroughHoleSolderAlg()
    std_img = _std("2")
    roi_a = {"shape": "rect", "x": 5, "y": 5, "w": 90, "h": 90}
    roi_b = {"shape": "rect", "x": 10, "y": 10, "w": 70, "h": 70}

    r1 = alg.run(
        _test("2-1"), {"solder_mode": "roi", "solder_roi": roi_a, "template_id": "templ_2"},
        original_template_image=std_img,
    )
    assert r1.code == 0
    assert len(alg._template_cache) == 1

    # 同一 template_id，换一个人工框：不应误用框 A 生成的旧缓存，需要新增一条缓存
    r2 = alg.run(
        _test("2-1"), {"solder_mode": "roi", "solder_roi": roi_b, "template_id": "templ_2"},
        original_template_image=std_img,
    )
    assert r2.code == 0
    assert len(alg._template_cache) == 2
    assert r1.metadata["template_id"] != r2.metadata["template_id"]

    # 换回框 A：应命中框 A 当时创建的缓存条目，而不是再新增一条
    r3 = alg.run(
        _test("2-1"), {"solder_mode": "roi", "solder_roi": roi_a, "template_id": "templ_2"},
        original_template_image=std_img,
    )
    assert len(alg._template_cache) == 2
    assert r3.metadata["template_id"] == r1.metadata["template_id"]


def test_prebuild_template_generates_and_reuses_cache_without_test_image():
    """界面上"选定锡面获取方式后立即生成"对应场景：只有标准图，没有测试图。"""
    alg = PcbThroughHoleSolderAlg()
    std_img = _std("2")

    r1 = alg.prebuild_template(std_img, {"solder_mode": "contour", "template_id": "templ_2"})
    assert r1["ok"] is True
    assert r1["template_id"] == "templ_2:contour:noj"
    assert len(alg._template_cache) == 1

    # 同一模板再次预生成：应直接命中缓存，不新增条目
    r2 = alg.prebuild_template(std_img, {"solder_mode": "contour", "template_id": "templ_2"})
    assert r2["ok"] is True
    assert len(alg._template_cache) == 1

    # 后续 run() 应复用同一条缓存（template_id 相同）
    result = alg.run(
        _test("2-1"), {"solder_mode": "contour", "template_id": "templ_2"},
        original_template_image=std_img,
    )
    assert result.code == 0
    assert len(alg._template_cache) == 1


def test_prebuild_template_reports_invalid_roi():
    alg = PcbThroughHoleSolderAlg()
    result = alg.prebuild_template(_std("2"), {"solder_mode": "roi", "solder_roi": {"shape": "rect"}})
    assert result["ok"] is False
    assert "solder_roi" in result["error"] or "roi" in result["error"]
    assert len(alg._template_cache) == 0


def test_roi_mode_locates_via_real_alignment_not_raw_coords():
    """人工框选给的是标准图坐标；待检图整体平移后，应靠配准定位而非照搬框坐标。

    面积占比门槛下，弱孔洞样本在平移后可能处于检出边缘；因此以「配准预览框
    跟随平移」为主断言，并在两侧都检出时校验框中心大致跟随 dx/dy。
    """
    alg = PcbThroughHoleSolderAlg()
    std_img = _std("2")
    test_img = _test("2-1")
    cfg = {"solder_mode": "roi", "solder_roi": _TEMPL_2_ROI}
    dx, dy = 5, -4

    baseline = alg.run(test_img, cfg, original_template_image=std_img)
    assert baseline.code == 0

    shifted = _shift_image(test_img, dx=dx, dy=dy)
    shifted_result = alg.run(shifted, cfg, original_template_image=std_img)
    assert shifted_result.code == 0

    preview0 = alg.preview_roi(test_img, cfg, original_template_image=std_img)
    preview1 = alg.preview_roi(shifted, cfg, original_template_image=std_img)
    assert preview0.get("ok") and preview1.get("ok")
    r0, r1 = preview0["roi"], preview1["roi"]
    # 配准映射后的预览框应大致跟随图像平移（允许亚像素/插值误差）
    assert abs((r1["x"] - r0["x"]) - dx) <= 2.5
    assert abs((r1["y"] - r0["y"]) - dy) <= 2.5

    if baseline.parts and shifted_result.parts:
        b0 = baseline.parts[0]
        b1 = shifted_result.parts[0]
        assert b0.class_id == b1.class_id
        assert abs((b1.x - b0.x) - dx) <= 4
        assert abs((b1.y - b0.y) - dy) <= 4


def test_ellipse_mode_preview_roi_is_circle_or_ellipse_not_rect():
    """自动 ellipse 模式预览应映射锡面圆/椭圆，而不是外接矩形。"""
    alg = PcbThroughHoleSolderAlg()
    preview = alg.preview_roi(
        _test("2-1"), {"solder_mode": "ellipse", "template_id": "templ_2"},
        original_template_image=_std("2"),
    )
    assert preview.get("ok") is True
    assert preview["roi"]["shape"] in ("circle", "ellipse")
    if preview["roi"]["shape"] == "circle":
        assert preview["roi"]["r"] > 5
    else:
        assert preview["roi"]["axes_w"] > 5 and preview["roi"]["axes_h"] > 5


def test_multi_pad_preview_maps_every_bridge_joint():
    """连锡多焊点：preview_roi.rois 应对每一个 pad 框做配准映射，而非只返回一个。"""
    from algorithms.pcb_through_hole.segment import roi_from_mask
    from algorithms.pcb_through_hole.template import TemplateModel
    from algorithms.pcb_through_hole.bridge_joints import BridgeJoint

    # 两个分离焊点 mask → 并集 ROI 必须覆盖两者（非仅最大连通域）
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[20:50, 20:50] = 255
    mask[20:50, 140:180] = 255
    roi_union = roi_from_mask(mask, union_bbox=True)
    assert roi_union.x <= 20 and roi_union.x + roi_union.w >= 180
    roi_max = roi_from_mask(mask, union_bbox=False)
    assert roi_max.w < roi_union.w

    joints = [
        BridgeJoint(1, "rect", {"x": 20, "y": 20, "w": 30, "h": 30}),
        BridgeJoint(2, "rect", {"x": 140, "y": 20, "w": 40, "h": 30}),
    ]
    std = np.full((120, 200, 3), 40, dtype=np.uint8)
    std[20:50, 20:50] = (180, 180, 180)
    std[20:50, 140:180] = (180, 180, 180)
    tm = TemplateModel.build(std, roi_mask=mask, bridge_joints=joints)
    # 默认 align ROI 取最大连通域；多框位移在 pipeline 分框共识
    assert tm.roi.w <= roi_union.w + 2
    assert tm.roi.w >= roi_max.w - 2
    # 多框模板默认只启用连锡
    assert tm.enabled_defects == [15]

    alg = PcbThroughHoleSolderAlg()
    cfg = {
        "solder_mode": "roi",
        "template_id": "multi_pad_preview",
        "bridge_joints": [j.to_dict() for j in joints],
        "solder_roi": {"shape": "rect", "x": 20, "y": 20, "w": 30, "h": 30},
        "enabled_defects": [12, 13, 14, 15, 16],
    }
    preview = alg.preview_roi(std.copy(), cfg, original_template_image=std)
    assert preview.get("ok") is True
    assert isinstance(preview.get("rois"), list)
    assert len(preview["rois"]) == 2
    assert preview["roi"] == preview["rois"][0]


def test_multi_pad_run_only_emits_bridge_defect_ids():
    """多框时即使 config 列出全部缺陷，结果也只应出现连锡(15)。"""
    from algorithms.pcb_through_hole.bridge_joints import BridgeJoint

    joints = [
        BridgeJoint(1, "rect", {"x": 10, "y": 20, "w": 30, "h": 40}),
        BridgeJoint(2, "rect", {"x": 80, "y": 20, "w": 30, "h": 40}),
    ]
    h, w = 80, 120
    std = np.full((h, w, 3), 40, dtype=np.uint8)
    std[20:60, 10:40] = (180, 180, 160)
    std[20:60, 80:110] = (180, 180, 160)
    test = std.copy()
    test[35:50, 35:85] = (180, 180, 160)  # 桥

    alg = PcbThroughHoleSolderAlg()
    result = alg.run(
        test,
        {
            "solder_mode": "roi",
            "template_id": "multi_pad_bridge_only",
            "bridge_joints": [j.to_dict() for j in joints],
            "solder_roi": {"shape": "rect", "x": 10, "y": 20, "w": 30, "h": 40},
            "enabled_defects": [12, 13, 14, 15, 16],
            "enable_review": False,
        },
        original_template_image=std,
    )
    assert result.code == 0
    ids = set(result.metadata.get("defect_ids") or [])
    assert ids <= {15}
    assert all(p.class_id == 15 for p in result.parts)


def test_inspect_test_roi_follows_preview_shift():
    """inspect.test_roi 平移量应与 preview_roi 一致。"""
    from algorithms.pcb_through_hole.pipeline import inspect
    from algorithms.pcb_through_hole.template import TemplateModel
    from algorithms.pcb_through_hole.solder_extract import extract_solder_mask

    alg = PcbThroughHoleSolderAlg()
    std_img = _std("2")
    test_img = _test("2-1")
    shifted = _shift_image(test_img, dx=5, dy=-4)
    cfg = {"solder_mode": "ellipse", "template_id": "templ_2"}

    mask, _ = extract_solder_mask(std_img, {"solder_mode": "ellipse"})
    tm = TemplateModel.build(std_img, roi_mask=mask, templ_id="2")

    r0 = inspect(test_img, tm, name="0")
    r1 = inspect(shifted, tm, name="1")
    p0 = alg.preview_roi(test_img, cfg, original_template_image=std_img)
    p1 = alg.preview_roi(shifted, cfg, original_template_image=std_img)
    assert r0.test_roi is not None and r1.test_roi is not None
    assert p0.get("ok") and p1.get("ok")

    def _preview_origin(roi: dict) -> tuple[float, float]:
        if roi["shape"] in ("circle", "ellipse"):
            return float(roi["cx"]), float(roi["cy"])
        return float(roi["x"]), float(roi["y"])

    po0, po1 = _preview_origin(p0["roi"]), _preview_origin(p1["roi"])
    d_preview = (po1[0] - po0[0], po1[1] - po0[1])
    d_roi = (r1.test_roi.x - r0.test_roi.x, r1.test_roi.y - r0.test_roi.y)
    assert abs(d_roi[0] - d_preview[0]) <= 2.5
    assert abs(d_roi[1] - d_preview[1]) <= 2.5


# ---- 模板缓存 ----

def test_template_cache_reused_across_calls():
    alg = PcbThroughHoleSolderAlg()
    std_img = _std("2")
    r1 = alg.run(_test("2-1"), {"template_id": "templ_2"}, original_template_image=std_img)
    key1 = r1.metadata["template_id"]
    assert len(alg._template_cache) == 1

    r2 = alg.run(_test("2-9"), {"template_id": "templ_2"}, original_template_image=std_img)
    key2 = r2.metadata["template_id"]
    assert key1 == key2 == "templ_2:ellipse:noj"
    assert len(alg._template_cache) == 1  # 同一模板不应重复入缓存

    # 不同模板：缓存应新增一条，而不是覆盖/丢失前一条
    alg.run(_test("9-1"), {"template_id": "templ_9"}, original_template_image=_std("9"))
    assert len(alg._template_cache) == 2


def test_template_cache_key_distinguishes_solder_mode():
    """同一 template_id 切换 solder_mode 不应误用另一种方式生成的旧 Mask 缓存。"""
    alg = PcbThroughHoleSolderAlg()
    std_img = _std("2")
    r1 = alg.run(_test("2-1"), {"template_id": "templ_2", "solder_mode": "ellipse"},
                 original_template_image=std_img)
    assert len(alg._template_cache) == 1

    r2 = alg.run(_test("2-1"), {"template_id": "templ_2", "solder_mode": "contour"},
                 original_template_image=std_img)
    assert len(alg._template_cache) == 2
    assert r1.metadata["template_id"] != r2.metadata["template_id"]

    # 切回 ellipse：应命中第一次生成的缓存条目，而不是再新增一条
    r3 = alg.run(_test("2-1"), {"template_id": "templ_2", "solder_mode": "ellipse"},
                 original_template_image=std_img)
    assert len(alg._template_cache) == 2
    assert r3.metadata["template_id"] == r1.metadata["template_id"]


def test_template_cache_lru_eviction():
    alg = PcbThroughHoleSolderAlg()
    std_img = _std("2")
    for i in range(3):
        alg.run(_test("2-1"), {"template_id": f"templ_{i}", "template_cache_size": 2},
                original_template_image=std_img)
    assert len(alg._template_cache) == 2
    assert "templ_0:ellipse:noj" not in alg._template_cache  # 最早写入的应被淘汰


def test_clear_template_cache():
    alg = PcbThroughHoleSolderAlg()
    alg.run(_test("2-1"), {"template_id": "templ_2"}, original_template_image=_std("2"))
    assert len(alg._template_cache) == 1
    alg.clear_template_cache()
    assert len(alg._template_cache) == 0


# ---- 异常处理契约：run() 内部绝不抛异常，一律 code=1 返回 ----

def test_invalid_inputs_return_code_1():
    alg = PcbThroughHoleSolderAlg()

    r = alg.run(None, {}, original_template_image=_std("2"))
    assert r.code == 1 and r.result_type == ResultType.ERROR

    r = alg.run(_test("2-1"), {})  # 缺少 original_template_image
    assert r.code == 1 and r.result_type == ResultType.ERROR

    r = alg.run(np.zeros((0, 0, 3), np.uint8), {}, original_template_image=_std("2"))
    assert r.code == 1

    r = alg.run(_test("2-1"), {"solder_mode": "not_a_mode"}, original_template_image=_std("2"))
    assert r.code == 1

    r = alg.run(_test("2-1"), {"void_preset": "NOPE"}, original_template_image=_std("2"))
    assert r.code == 1


def test_run_never_raises_even_on_unexpected_exception(monkeypatch):
    """哪怕内部实现意外抛异常，run() 也必须兜底返回 code=1，而不是向上抛出。"""
    import algorithms.pcb_through_hole.pcb_through_hole_solder_alg as mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    alg = PcbThroughHoleSolderAlg()
    monkeypatch.setattr(mod, "run_inspect", _boom)
    result = alg.run(_test("2-1"), {}, original_template_image=_std("2"))
    assert result.code == 1
    assert "boom" in result.message
    assert "traceback" in result.metadata


def test_grayscale_and_bgra_inputs_are_normalized():
    """统一输入格式：允许灰度图/带 alpha 通道输入，内部自动转为 3 通道 BGR。"""
    alg = PcbThroughHoleSolderAlg()
    test_gray = cv2.cvtColor(_test("2-1"), cv2.COLOR_BGR2GRAY)
    result = alg.run(test_gray, {}, original_template_image=_std("2"))
    assert result.code == 0

    test_bgra = cv2.cvtColor(_test("2-1"), cv2.COLOR_BGR2BGRA)
    result2 = alg.run(test_bgra, {}, original_template_image=_std("2"))
    assert result2.code == 0


def test_multiple_roi_images_ignored_without_error():
    alg = PcbThroughHoleSolderAlg()
    dummy = [_test("2-1")]
    result = alg.run(
        _test("2-1"), {}, original_template_image=_std("2"),
        multiple_roi_images=dummy,
    )
    assert result.code == 0
    assert result.metadata.get("multiple_roi_images_ignored") is True


# ---- batch_run（基类默认串行实现，单项失败不影响其他） ----

def test_batch_run_default_serial_isolation():
    alg = PcbThroughHoleSolderAlg()
    images = [_test("2-1"), None, _test("2-8")]
    templates = [_std("2"), _std("2"), _std("2")]
    results = alg.batch_run(images, [{}, {}, {}], original_template_image=templates)
    assert len(results) == 3
    assert results[0].code == 0 and results[0].metadata["status"] == "OK"
    assert results[1].code == 1
    assert results[2].code == 0 and results[2].metadata["status"] == "NG"
