"""检测引擎组件工厂：按配置装配 backbone + 槽位。

来源：算法工程 scripts/m0_baseline.py 的 build() 函数，
导入路径改写为 algo.* 前缀，脱离脚本目录依赖。
"""
import torch


def build(cfg, device):
    torch.manual_seed(int(cfg.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.get("seed", 42)))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    bcfg = cfg["backbone"]
    backbone = None
    if bcfg.get("enabled", True):
        if bcfg.get("name") == "pdn":
            from algo.backbone.pdn import PDNBackbone
            backbone = PDNBackbone(grid=bcfg["grid"], device=device,
                                   checkpoint=bcfg.get("checkpoint"))
        else:
            from algo.backbone.dino import FrozenDINO
            backbone = FrozenDINO(output_layer=bcfg["output_layer"],
                                  grid=bcfg["grid"], device=device)
    slots = []
    s = cfg["slots"]
    if s["sem"]["enabled"] and backbone is not None:
        from algo.slots.sem import SemSlot
        slots.append(SemSlot(s["sem"], device))
    if s["disc"]["enabled"] and backbone is not None:
        from algo.slots.disc import DiscSlot
        # augment.pseudo（伪异常合成方式）与 augment.enhance（训练增强）注入 disc 槽位（§15.29）
        aug = cfg.get("augment", {}) or {}
        slots.append(DiscSlot({**s["disc"], "seed": cfg["seed"],
                               "pseudo": aug.get("pseudo", {}),
                               "enhance": aug.get("enhance", {})},
                              backbone, device))
    if s["shead"]["enabled"] and backbone is not None:
        from algo.slots.shead import SheadSlot
        slots.append(SheadSlot({**s["shead"], "seed": cfg["seed"]}, backbone, device))
    if s["blob"]["enabled"]:
        from algo.slots.blob import BlobSlot
        slots.append(BlobSlot(s["blob"], device))
    if s["trad"]["enabled"]:
        from algo.slots.trad import TradSlot
        slots.append(TradSlot(s["trad"]))
    if s["layout"]["enabled"]:
        from algo.slots.layout import LayoutSlot
        slots.append(LayoutSlot(s["layout"]))
    if s.get("color", {}).get("enabled"):
        from algo.slots.color import ColorSlot
        slots.append(ColorSlot(s["color"], device))
    if s["inp"]["enabled"] and backbone is not None:
        from algo.slots.inp import InpSlot
        slots.append(InpSlot(s["inp"], device))
    if s.get("tpl", {}).get("enabled") and backbone is not None:
        from algo.slots.tpl import TplSlot
        slots.append(TplSlot(s["tpl"], backbone, device))
    return backbone, slots
