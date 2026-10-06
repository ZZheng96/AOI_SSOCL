"""槽位注册表：name -> Slot 类（M0 各入口统一从这里取映射）"""
from .sem import SemSlot
from .disc import DiscSlot
from .blob import BlobSlot
from .trad import TradSlot
from .layout import LayoutSlot
from .inp import InpSlot
from .shead import SheadSlot

SLOT_REGISTRY = {
    "sem": SemSlot,
    "disc": DiscSlot,
    "blob": BlobSlot,
    "trad": TradSlot,
    "layout": LayoutSlot,
    "inp": InpSlot,
    "shead": SheadSlot,
}
