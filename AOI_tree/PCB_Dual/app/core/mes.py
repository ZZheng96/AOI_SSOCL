"""MES 对接抽象（P2：借鉴 Java AOI MES架构.md 插件化三层）。

三层：业务层 → IMesGateway（统一入口）→ 插件实现（Mock/HTTP）。
同步接口：查询/校验/上报判定；事件另走 task_queue 事件总线。
"""
from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger(__name__)


@dataclass
class MesRequest:
    """标准请求对象。"""
    interface: str
    payload: dict
    trace_id: str = ""

    def to_dict(self) -> dict:
        return {"interface": self.interface, "trace_id": self.trace_id,
                "payload": self.payload}


@dataclass
class MesResponse:
    """标准响应。"""
    code: int = 0          # 0=成功
    message: str = "ok"
    data: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.code == 0


@dataclass
class MesEvent:
    """标准事件（异步通知类走事件）。"""
    event_type: str
    payload: dict
    occurred_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {"event_type": self.event_type, "occurred_at": self.occurred_at,
                "payload": self.payload}


class IMesGateway(ABC):
    """MES 统一入口（插件实现各自适配不同 MES）。"""

    @abstractmethod
    def report_detection(self, inspect_id: int) -> MesResponse:
        """检测完成上报（整板判定 + 缺陷汇总）。"""

    @abstractmethod
    def report_workorder_status(self, workorder_id: int,
                                status: str) -> MesResponse:
        """工单状态上报（open/running/closed）。"""


class MockMesGateway(IMesGateway):
    """Mock 实现：记录到 storage/mes_outbox（联调用回放）。"""

    def __init__(self, outbox: Optional[str] = None) -> None:
        from app.config import get_settings
        self.outbox = outbox or str(get_settings().storage("mes_outbox"))

    def _write(self, interface: str, data: dict) -> None:
        os.makedirs(self.outbox, exist_ok=True)
        path = os.path.join(self.outbox,
                            f"{datetime.now():%Y%m%d_%H%M%S_%f}_{interface}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"interface": interface, "ts": datetime.now().isoformat(),
                       "data": data}, f, ensure_ascii=False, indent=2)

    def report_detection(self, inspect_id: int) -> MesResponse:
        from app.db.database import session_scope
        from app.db.models import DefectRecord, Inspect
        with session_scope() as s:
            insp = s.get(Inspect, inspect_id)
            if insp is None:
                return MesResponse(code=1, message=f"inspect {inspect_id} 不存在")
            defects = (s.query(DefectRecord)
                       .filter(DefectRecord.inspect_id == inspect_id).all())
            d = {"inspect_id": inspect_id,
                 "board_barcode": insp.board_barcode,
                 "number": insp.number,
                 "pass": bool(insp.pass_),
                 "ng_count": insp.ng_count,
                 "total_count": insp.total_count,
                 "defects": [{"defect_type": x.defect_type, "engine": x.engine,
                              "board_x": x.board_x, "board_y": x.board_y,
                              "width": x.width, "height": x.height}
                             for x in defects],
                 "inspect_time": insp.inspect_time.isoformat()
                                 if insp.inspect_time else None}
        self._write("report_detection", d)
        log.info("[mes] report_detection inspect=%s pass=%s ng=%s",
                 inspect_id, d["pass"], d["ng_count"])
        return MesResponse(data=d)

    def report_workorder_status(self, workorder_id: int, status: str) -> MesResponse:
        from app.db.database import session_scope
        from app.db.models import WorkOrder
        with session_scope() as s:
            wo = s.get(WorkOrder, workorder_id)
            name = wo.name if wo else str(workorder_id)
        self._write("report_workorder_status",
                    {"workorder_id": workorder_id, "name": name, "status": status})
        log.info("[mes] workorder %s status=%s", workorder_id, status)
        return MesResponse()


class HttpMesGateway(MockMesGateway):
    """HTTP 实现：POST JSON 到 mes.url（超时 5s）。

    请求失败时记录日志并回退落盘 storage/mes_outbox（沿用 Mock 落盘机制，不丢事件）。
    """

    def __init__(self, url: str, outbox: Optional[str] = None,
                 timeout: float = 5.0) -> None:
        super().__init__(outbox)
        self.url = url
        self.timeout = timeout

    def _write(self, interface: str, data: dict) -> None:
        import requests
        body = MesRequest(interface=interface, payload=data).to_dict()
        try:
            resp = requests.post(self.url, json=body, timeout=self.timeout)
            resp.raise_for_status()
        except Exception:  # noqa: BLE001
            log.exception("[mes] HTTP 上报失败（%s），回退落盘 outbox", self.url)
            super()._write(interface, data)


_gateway: Optional[IMesGateway] = None
_gateway_lock = threading.Lock()


def get_mes_gateway() -> Optional[IMesGateway]:
    """返回启用状态的网关；config mes.enabled=false → None。"""
    from app.config import get_settings
    if not get_settings().get("mes", "enabled", False):
        return None
    global _gateway
    if _gateway is None:
        with _gateway_lock:
            if _gateway is None:
                url = str(get_settings().get("mes", "url", "") or "").strip()
                mtype = str(get_settings().get("mes", "type", "mock"))
                if url:
                    _gateway = HttpMesGateway(url)
                elif mtype == "mock":
                    _gateway = MockMesGateway()
                else:
                    raise ValueError(f"不支持的 MES 网关类型: {mtype}（支持 mock 或配置 mes.url 走 HTTP）")
    return _gateway
