"""缺陷类型元数据：界面只暴露面积 / 亮度 / 覆盖率类直观参数。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PRESET_CHOICES: List[str] = ["LOW", "MED", "HIGH"]
PRESET_HINT = "LOW=宽松(少报) / MED=标准(默认) / HIGH=严格(多报)。多数情况只调挡位即可。"


@dataclass(frozen=True)
class Param:
    """界面可调参数：key 写入算法 config，label/hint/unit 仅用于展示。

    ``unit="%"`` 时，界面按百分数显示（×100），写入算法仍用 0~1 比例。
    其它单位原样作为数值后缀（如 ``×``）。
    """
    key: str
    label: str
    kind: str
    default: float = 0.0
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    hint: str = ""
    unit: str = ""


@dataclass(frozen=True)
class DefectMeta:
    defect_id: int
    code: str
    name: str
    color: str
    summary: str
    preset_key: Optional[str] = None   # 算法 config 里的挡位键，如 "void_preset"
    params: List[Param] = field(default_factory=list)

    def default_param_values(self) -> Dict[str, Any]:
        return {p.key: p.default for p in self.params}


DEFECT_METAS: List[DefectMeta] = [
    DefectMeta(
        12, "void", "孔洞", "#F5A623",
        "锡面里够大、够暗的空洞/气泡", "void_preset",
        params=[
            Param(
                "void_decide_area_frac", "面积占比", "float", 0.0035, 0.0005, 0.05, 0.0005,
                hint="孔洞相对锡面的面积。调大=只报更大的孔；调小=小孔也会报。",
                unit="%"),
            Param(
                "void_decide_min_dark", "孔洞暗度", "float", 0.90, 0.5, 1.0, 0.01,
                hint="暗斑要有多暗才算孔洞。调大=只报更黑的；调小=偏灰也会报。",
                unit="%"),
        ],
    ),
    DefectMeta(
        13, "insufficient", "少锡", "#E5484D",
        "锡没盖满：边缘发红=薄锡，硬边缺口=露铜", "insuf_preset",
        params=[
            Param(
                "insuf_area_ratio_max", "锡面覆盖率", "float", 0.85, 0.5, 1.0, 0.01,
                hint="锡盖住焊盘的比例。调大=要求盖得更满（更容易报少锡）；调小=盖少一点也放过。",
                unit="%"),
            Param(
                "insuf_exposed_hard_min", "露铜面积占比", "float", 0.06, 0.0, 0.3, 0.005,
                hint="焊盘上露铜/露底面积。调大=露得多才报；调小=露一点就报。",
                unit="%"),
        ],
    ),
    DefectMeta(
        14, "excess", "多锡(包锡)", "#8E4EC6",
        "锡堆得太多或反光异常（包锡）", None,
        params=[
            Param(
                "excess_area_ratio_min", "锡面面积比", "float", 1.15, 1.0, 2.0, 0.01,
                hint="相对标准图，锡面面积胀到几倍才算多锡。调大=胀得更多才报；调小=稍胀就报。",
                unit="×"),
            Param(
                "excess_highlight_ratio_min", "反光面积占比", "float", 0.035, 0.0, 0.3, 0.005,
                hint="锡面上异常高亮反光的面积占比。调大=反光多才报；调小=反光少也报。",
                unit="%"),
        ],
    ),
    DefectMeta(
        15, "bridge", "连锡", "#FF8000",
        "相邻焊点被锡桥连（颜色连通或缝隙收窄）", None,
        params=[
            Param(
                "bridge_min_bridge_pixels", "桥连像素门槛", "int", 6, 1, 80, 1,
                hint="两焊点之间桥区最少像素。调大=桥更粗才报；调小=细桥也报。",
                unit="px"),
            Param(
                "bridge_min_clearance_ratio", "缝隙收窄比", "float", 0.35, 0.05, 1.0, 0.01,
                hint="相对标准图缝隙宽度比。调大=更宽松；调小=缝隙稍窄就报。",
                unit="%"),
            Param(
                "bridge_min_clearance_px", "缝隙最小宽度", "int", 2, 0, 20, 1,
                hint="缝隙绝对宽度低于此值也报连锡（含完全闭合）。调大=更宽松；调小=更严。",
                unit="px"),
        ],
    ),
    DefectMeta(
        16, "no_lead", "不出脚", "#0BA5EC",
        "引脚没露出来，中心亮暗与标准图差太多", None,
        params=[
            Param(
                "nolead_center_diff_min", "中心偏离度", "float", 1.00, 0.0, 3.0, 0.05,
                hint="焊点中心相对标准图偏了多少。调大=偏得更多才报；调小=偏一点就报。",
                unit="×"),
            Param(
                "nolead_ring_v_delta_min", "引脚亮度差", "float", 30.0, 0.0, 100.0, 1.0,
                hint="引脚周围与中心的亮度差（正常出脚时通常更明显）。调大=差得更明显才报；调小=差一点就报。",
                unit=""),
        ],
    ),
]

DEFECT_BY_ID: Dict[int, DefectMeta] = {m.defect_id: m for m in DEFECT_METAS}

DEFAULT_ENABLED_DEFECTS: List[int] = [m.defect_id for m in DEFECT_METAS]

DEFECT_ID_BRIDGE = 15
PAD_INTERNAL_DEFECTS: List[int] = [12, 13, 14, 16]
BRIDGE_ONLY_DEFECTS: List[int] = [DEFECT_ID_BRIDGE]


def defects_for_pad_count(n_pads: int) -> List[int]:
    if int(n_pads) >= 2:
        return list(BRIDGE_ONLY_DEFECTS)
    return list(PAD_INTERNAL_DEFECTS)
