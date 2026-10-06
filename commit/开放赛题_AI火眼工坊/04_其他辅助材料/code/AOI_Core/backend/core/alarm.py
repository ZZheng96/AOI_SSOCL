"""连续异常告警（M6a：滑窗异常率超阈值 -> 产线批量缺陷/停线信号）。

单帧 anomaly 是噪音，近 N 帧滑窗内异常数 >= K 才构成批量缺陷事件。
按品类维护 deque(maxlen=alarm_window)，record() 为 O(window) 求和
（window=20 常数级内存/计算，远低于 1s 红线）；触发受冷却期
alarm_cooldown_s 约束防刷屏。触发后的落库/webhook 由调用方
（DetectionService）执行，本模块只管判定。

配置：configs/default.yaml notify 段
  alarm_window / alarm_k / alarm_cooldown_s
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

from .config import get_settings


class AlarmMonitor:
    """按品类的连续异常滑窗监控（进程内单例，见 get_alarm_monitor）。"""

    def __init__(self, window: int = 20, k: int = 5, cooldown_s: float = 60.0):
        self.window = max(1, int(window))
        self.k = max(1, int(k))
        self.cooldown_s = float(cooldown_s)
        self._lock = threading.Lock()
        self._windows: Dict[str, Deque[bool]] = {}
        self._last_fired: Dict[str, float] = {}

    def record(self, category: str, is_anomaly: bool) -> Tuple[bool, int]:
        """记录一帧判定结果，返回 (fired, window_hits)。

        fired=True 表示本次到达阈值且已出冷却期（应记日志/推 webhook）；
        window_hits 为当前窗口内异常帧数（响应/WS 的 alarm 字段用）。
        """
        now = time.time()
        with self._lock:
            win = self._windows.setdefault(category, deque(maxlen=self.window))
            win.append(bool(is_anomaly))
            hits = sum(win)
            if hits >= self.k:
                last = self._last_fired.get(category)
                if last is None or now - last >= self.cooldown_s:
                    self._last_fired[category] = now
                    return True, hits
            return False, hits

    def status(self, category: str) -> Dict[str, int]:
        """当前窗口状态（不改窗口）：{active, window_hits}。"""
        with self._lock:
            win = self._windows.get(category)
            hits = sum(win) if win else 0
        return {"active": int(hits >= self.k), "window_hits": hits}

    def reset(self, category: Optional[str] = None) -> None:
        """清空窗口与冷却计时（测试用；category=None 清全部）。"""
        with self._lock:
            if category is None:
                self._windows.clear()
                self._last_fired.clear()
            else:
                self._windows.pop(category, None)
                self._last_fired.pop(category, None)


_monitor: Optional[AlarmMonitor] = None


def get_alarm_monitor() -> AlarmMonitor:
    """模块级单例；首次调用读 notify 段配置，之后固定（重启进程才重读）。"""
    global _monitor
    if _monitor is None:
        cfg = get_settings().get_alarm_cfg()
        _monitor = AlarmMonitor(window=cfg["alarm_window"], k=cfg["alarm_k"],
                                cooldown_s=cfg["alarm_cooldown_s"])
    return _monitor
