"""算法元数据注册表（数据驱动 UI 的唯一事实来源）。

前端所有"瑕疵清单 / 判据开关 / 参数面板"都由这里生成。新增算法时，只需：
  1. 在 ``pcb_defect_detector`` 注册算法（引擎侧）；
  2. 在此追加一条 ``DefectMeta``；
UI 会自动出现对应条目，无需改动界面代码。

字段说明：
  - code:        算法 code，与 ``PCBDefectDetector.list_algorithms`` 的 key 一致。
  - name:        中文名。
  - icon:        单字/emoji 图标（占位，UI 用彩色圆点+文字）。
  - color:       该瑕疵的主题色（十六进制）。
  - summary:     一句话原理。
  - judges:      多选判据列表（对应引擎 config 里的 enable_* 开关）；空表示该算法无判据开关。
  - params:      可调参数（滑杆/输入）列表，映射到引擎 config。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Judge:
    key: str          # 引擎 config 里的开关键，如 "enable_size"
    label: str        # UI 显示名，如 "尺寸"
    default: bool = True
    hint: str = ""


@dataclass(frozen=True)
class Param:
    key: str          # 引擎 config 键
    label: str
    kind: str         # "float" | "int" | "choice" | "bool"
    default: Any = 0.0
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    choices: Optional[List[str]] = None
    hint: str = ""
    # True 表示"高级参数"：默认界面不显示，仅在开启「高级模式」后才出现；
    # 未显示时仍以 default 值参与检测（相当于固定值）。
    advanced: bool = False
    # True 表示彻底从界面移除（无论是否开启高级模式都不出现），永远固定为 default 值。
    # 与 advanced 的区别：advanced 参数打开「高级模式」还能看到并调节，hidden 参数完全不可见。
    hidden: bool = False
    # 受「检出容忍度」联动的参数会带上此字典 {"low":..,"mid":..,"high":..}；
    # 选择 低/中/高 档位时，所有带此字典的参数会一起被设为对应值。
    # 没有该字典的参数是独立滑杆，不受档位联动。
    tolerance: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class DefectMeta:
    code: str
    name: str
    color: str
    summary: str
    judges: List[Judge] = field(default_factory=list)
    params: List[Param] = field(default_factory=list)
    icon: str = ""
    # False 时界面不显示"判据（多选）"分组，判据固定为各自的 default 值。
    show_judges: bool = True

    def default_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {}
        for j in self.judges:
            cfg[j.key] = j.default
        for p in self.params:
            cfg[p.key] = p.default
        return cfg


# ------------------------------------------------------------------ 注册表 ----
# 顺序与引擎默认执行顺序一致，便于对照。
DEFECT_METAS: List[DefectMeta] = [
    DefectMeta(
        code="component_wrong_part", name="错件", color="#E5484D", icon="错",
        summary="尺寸 + 颜色(HSV) + 轮廓，三判据投票",
        show_judges=False,  # 三项判据固定全部开启，界面不展示开关
        judges=[
            Judge("enable_size", "尺寸", True, "元件长/短边偏差；偏差超过容差判异常"),
            Judge("enable_hist", "颜色(HSV)", True, "HSV 直方图相关性；相关低于下限判异常"),
            Judge("enable_shape", "轮廓", True, "外形轮廓匹配度；差异超过上限判异常"),
        ],
        params=[
            Param("vote_fail_min", "判定严格度(异常判据数)", "int", 2, 1, 3, 1,
                  hint="至少 N 个判据判异常才判 NG（数值越小越严格）",
                  tolerance={"low": 1, "mid": 2, "high": 3}),
            Param("size_tol_ratio", "尺寸差异容忍度", "float", 0.20, 0.0, 1.0, 0.01,
                  hint="尺寸偏差超过此比例判异常（越小越严格）",
                  tolerance={"low": 0.12, "mid": 0.20, "high": 0.30}),
            Param("hist_corr_min", "颜色差异容忍度", "float", 0.80, 0.0, 1.0, 0.01,
                  hint="颜色相关低于此值判异常（数值越大越严格；设0则此判据永不触发）",
                  tolerance={"low": 0.88, "mid": 0.80, "high": 0.68}),
            Param("shape_match_max", "外形差异容忍度", "float", 0.25, 0.0, 2.0, 0.01,
                  hint="轮廓差异超过此值判异常（越小越严格）",
                  tolerance={"low": 0.16, "mid": 0.25, "high": 0.38}),
        ],
    ),
    DefectMeta(
        code="component_missing", name="缺件", color="#F76B15", icon="缺",
        summary="本体像素比 + 差分 + 边缘 + 暗通道 联合判定",
        params=[
            Param("body_ratio_min", "元件缺失灵敏度", "float", 0.30, 0.0, 1.0, 0.01,
                  hint="待检本体像素占模板比例低于此值判缺件（数值越大越严格）",
                  tolerance={"low": 0.42, "mid": 0.30, "high": 0.20}),
            Param("presence_ok_min", "元件存在判定门槛", "float", 0.60, 0.0, 1.0, 0.01,
                  hint="综合亮/暗通道的存在度高于此值判 OK（数值越大越严格，易漏检）",
                  tolerance={"low": 0.72, "mid": 0.60, "high": 0.48}),
            Param("diff_score_max", "图像差异容忍度", "float", 0.50, 0.0, 1.0, 0.01,
                  hint="差异像素占比超过此值判缺件（越小越严格）",
                  tolerance={"low": 0.35, "mid": 0.50, "high": 0.65}),
            Param("diff_thresh", "差分阈值", "int", 40, 0, 255, 1,
                  hint="灰度差超过此值才算差异像素（越小越敏感）", advanced=True),
            Param("edge_ratio_min", "边缘比下限", "float", 0.05, 0.0, 1.0, 0.01,
                  hint="待检边缘占比低于此值且差分大时判缺件（越大越严格）", advanced=True),
        ],
    ),
    DefectMeta(
        code="component_shift", name="移位", color="#3E63DD", icon="移",
        summary="ROI 位姿估计 dx/dy/θ 超公差",
        params=[
            Param("pos_tol_mm", "位置偏移容忍度(mm)", "float", 0.30, 0.0, 5.0, 0.05,
                  hint="位移超过此公差判移位（越小越严格）",
                  tolerance={"low": 0.15, "mid": 0.30, "high": 0.50}),
            Param("angle_tol_deg", "角度偏移容忍度(°)", "float", 10.0, 0.0, 45.0, 0.5,
                  hint="旋转角度超过此公差判移位（越小越严格）",
                  tolerance={"low": 5.0, "mid": 10.0, "high": 18.0}),
        ],
    ),
    DefectMeta(
        code="component_tombstone", name="立碑", color="#8E4EC6", icon="碑",
        summary="宽高比 + 焊盘不对称 + ROI 模式联合",
        params=[
            Param("aspect_dev_ratio", "立起程度容忍度", "float", 0.30, 0.0, 1.0, 0.01,
                  hint="宽高比偏差超过此值判立碑（越小越严格）",
                  tolerance={"low": 0.18, "mid": 0.30, "high": 0.45}),
            Param("pad_area_ratio_max", "焊盘不对称容忍度", "float", 0.85, 0.0, 1.0, 0.01,
                  hint="焊盘面积不对称超过此值判立碑（越小越严格）",
                  tolerance={"low": 0.65, "mid": 0.85, "high": 0.95}),
            Param("roi_diff_min", "ROI差分下限", "float", 0.28, 0.0, 1.0, 0.01,
                  hint="ROI 差分低于此值不判立碑（越大越严格）", advanced=True),
            Param("roi_ncc_max", "ROI NCC上限", "float", 0.12, 0.0, 1.0, 0.01,
                  hint="NCC 高于此值不判立碑（越小越严格）", advanced=True),
        ],
    ),
    DefectMeta(
        code="component_flipped", name="翻件", color="#0BA5EC", icon="翻",
        summary="正面图案 + 颜色，双判据投票",
        show_judges=False,  # 两项判据固定全部开启，界面不展示开关
        judges=[
            Judge("enable_ncc", "正面图案", True, "与正面模板的归一化互相关"),
            Judge("enable_hist", "颜色", True, "HSV 直方图相关性"),
        ],
        params=[
            Param("vote_fail_min", "判定严格度(异常判据数)", "int", 2, 1, 2, 1,
                  hint="至少 N 个判据判异常才判 NG（数值越小越严格）", advanced=True),
            Param("ncc_front_min", "正面图案匹配容忍度", "float", 0.55, 0.0, 1.0, 0.01,
                  hint="与正面模板相关低于此值判异常（数值越大越严格；设0则永不触发）",
                  tolerance={"low": 0.68, "mid": 0.55, "high": 0.42}),
            Param("hist_corr_min", "颜色差异容忍度", "float", 0.75, 0.0, 1.0, 0.01,
                  hint="颜色相关低于此值判异常（数值越大越严格；设0则永不触发）",
                  tolerance={"low": 0.85, "mid": 0.75, "high": 0.62}),
        ],
    ),
    DefectMeta(
        code="component_reverse_polarity", name="极反", color="#30A46C", icon="极",
        summary="丝印方向 + 灯芯形状 联合判定(无OCR)",
        show_judges=False,  # 判据固定生效，不在界面上暴露开关
        judges=[
            Judge("enable_silkscreen", "丝印", True, "丝印图案 0°/180° NCC"),
            Judge("enable_color_band", "色带", False, "较宽色带侧别"),
            Judge("enable_diode", "灯芯", True, "二极管灯芯形状/位置"),
            Judge("enable_dark_mark", "暗标记", False, "暗极性标记辅助"),
        ],
        params=[
            Param("ncc_diff_min", "丝印方向差异容忍度", "float", 0.06, 0.0, 1.0, 0.005,
                  hint="0°/180° 丝印 NCC 差异低于此值不判极反（数值越大越严格）",
                  tolerance={"low": 0.10, "mid": 0.06, "high": 0.03}),
            Param("mark_dark_ratio", "暗标记比例容忍度", "float", 0.30, 0.0, 1.0, 0.01,
                  hint="暗标记面积占比阈值（越小越敏感）；对应判据默认关闭，暂不生效", advanced=True),
            Param("band_side_tol", "色带位置容忍度", "float", 0.12, 0.0, 1.0, 0.01,
                  hint="色带侧别判定容差（越小越严格）；对应判据默认关闭，暂不生效", advanced=True),
            Param("body_ncc_max", "本体NCC上限", "float", 1.0, 0.0, 1.0, 0.01,
                  hint="本体 NCC 高于此值跳过极反（越小越严格）", advanced=True),
        ],
    ),
    DefectMeta(
        code="component_damage", name="破损", color="#E54666", icon="损",
        summary="外观差异 + 轮廓比对，先精对齐再判破损",
        params=[
            Param("ssim_diff_thresh", "外观差异容忍度", "float", 0.35, 0.0, 1.0, 0.01,
                  hint="结构相似度差异超过此值判破损（越小越严格）",
                  tolerance={"low": 0.22, "mid": 0.35, "high": 0.50}),
            Param("shape_match_max", "外形差异容忍度", "float", 0.30, 0.0, 2.0, 0.01,
                  hint="轮廓差异超过此值判异常（越小越严格）",
                  tolerance={"low": 0.20, "mid": 0.30, "high": 0.45}),
            Param("blob_area_min", "最小可判定瑕疵尺寸", "int", 30, 0, 500, 1,
                  hint="小于此面积的差异忽略（数值越小越敏感，能查出更细小的瑕疵）",
                  tolerance={"low": 15, "mid": 30, "high": 60}),
            Param("morph_ksize", "形态学核", "int", 3, 1, 15, 2,
                  hint="差异区域形态学处理核大小（越大越合并碎斑）", advanced=True),
            Param("skip_structural_hint", "避免与缺件/立碑重复判定", "bool", True,
                  hint="开启时：差异像缺件/立碑会跳过破损判定。若明显破损检不出，请关闭",
                  hidden=True),
        ],
    ),
]

DEFECT_BY_CODE: Dict[str, DefectMeta] = {m.code: m for m in DEFECT_METAS}


def get_meta(code: str) -> Optional[DefectMeta]:
    return DEFECT_BY_CODE.get(code)
