"""候选专用全轮计数与墙钟上界（合作式检查点粒度）；不提供真实显存硬上界。"""
import contextlib
import contextvars
import hashlib
import json
import time


class CandidateDeferred(ValueError):
    def __init__(self, message, *, state="deferred"):
        super().__init__(message)
        self.state = state


_current = contextvars.ContextVar("candidate_budget", default=None)


def charge(key, amount=1, point="usage"):
    budget = _current.get()
    if budget is not None:
        budget.add(key, amount, point)


def charge_image(path):
    if _current.get() is not None:
        from PIL import Image
        with Image.open(path) as image:
            charge("input_pixels", image.width * image.height, "input_pixels")


@contextlib.contextmanager
def candidate_scope(budget):
    token = _current.set(budget)
    try:
        yield
    finally:
        _current.reset(token)


class CandidateBudget:
    def __init__(self, engine, *, category, event_id, cancel_event=None, clock=time.monotonic):
        self.engine, self.category, self.event_id = engine, category, event_id
        self.cancel_event = cancel_event
        cfg = engine.ssocl_cfg.get("candidate_budget", {})
        defaults = {"events": 16, "loo": 8, "search": 16, "steps": 10000,
                    "reloads": 2, "snapshot_bytes": 512 * 1024 * 1024,
                    "patch_bytes": 64 * 1024 * 1024, "input_pixels": 25_000_000,
                    # 实测 candidate p50≈6.1s；60s 只拦截失控轮次，不影响正常学习。
                    "wall_ms": 60_000}
        unknown = set(cfg) - {"max_" + k for k in defaults}
        if unknown:
            raise ValueError("未知候选预算配置：" + ",".join(sorted(unknown)))
        self.limits = {}
        for key, default in defaults.items():
            value = cfg.get("max_" + key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("候选预算必须为非负整数：max_" + key)
            self.limits[key] = value
        self.used = {k: 0 for k in self.limits}
        self.config_hash = hashlib.sha256(json.dumps(
            self.limits, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self._clock = clock
        self._t0 = clock()
        self._paused = 0.0  # 串行模式下让行前台推理的累计等待（秒），不计入墙钟预算

    def _check_wall(self, point):
        self.used["wall_ms"] = round((self._clock() - self._t0 - self._paused) * 1000)
        if self.used["wall_ms"] > self.limits["wall_ms"]:
            raise CandidateDeferred("候选墙钟超限（检查点：%s）：%sms>%sms" %
                                    (point, self.used["wall_ms"], self.limits["wall_ms"]))

    def check(self, point):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CandidateDeferred("候选已取消（合作式检查点：%s）" % point,
                                    state="cancelled")
        self.engine._candidate_checkpoint("budget:" + point)
        yield_fn = getattr(self.engine, "_learning_yield", None)
        if yield_fn is not None:
            self._paused += yield_fn(self.cancel_event)
        # 故障注入/取消请求在检查点到达后也必须生效。
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CandidateDeferred("候选已取消（合作式检查点：%s）" % point,
                                    state="cancelled")
        self._check_wall(point)

    def add(self, key, amount=1, point="usage"):
        if key == "wall_ms":
            raise ValueError("wall_ms 由墙钟计量，不可手工计费")
        self.check(point)
        if amount < 0:
            raise ValueError("候选预算用量不能为负")
        self.used[key] += int(amount)
        if self.used[key] > self.limits[key]:
            raise CandidateDeferred("候选预算超限[%s]：%s>%s" %
                                    (key, self.used[key], self.limits[key]))

    def snapshot(self):
        # 墙钟为合作式上界：超时最迟在下一个检查点生效，单步（一次前向/一次保存）不可中断。
        # used.wall_ms 取最后一次检查点的值（已≤上限），发布审计 used≤limits 不会因检查点后的提交耗时误拒；
        # 实际总耗时另报 elapsed_ms。
        return {"limits": dict(self.limits), "used": dict(self.used),
                "elapsed_ms": round((self._clock() - self._t0) * 1000),
                "yield_ms": round(self._paused * 1000),
                "config_hash": self.config_hash,
                "wall_clock_guaranteed": "checkpoint", "vram_guaranteed": False}
