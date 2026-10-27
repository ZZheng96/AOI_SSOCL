"""系统配置加载：YAML + 单例。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .yamlio import save_yaml_atomic

AOI_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = AOI_ROOT / "configs" / "default.yaml"

# A16（2026-10-03）：设置中心写回的 schema 白名单——{section: {key: 允许类型}}。
# 防 UI 写入口径漂移（如把 latency_budget_ms 写成字符串导致下游比较崩溃）。
# 仅约束 update() 写回路径；yaml 初始加载不强制（向后兼容存量配置）。
_SCHEMA: Dict[str, Dict[str, tuple]] = {
    "system": {
        "engine_max_cached_categories": (int,),
        "latency_budget_ms": (int, float),
    },
    "pipeline": {
        "tile_size": (int,), "tile_stride": (int,), "input_size": (int,),
        "latency_budget_ms": (int, float),
    },
    "security": {
        "max_upload_mb": (int, float), "api_key": (str,), "api_key_env": (str,),
    },
    "notify": {
        "timeout_s": (int, float), "alarm_window": (int,),
        "alarm_k": (int,), "alarm_cooldown_s": (int, float),
        "webhook_url": (str,),
    },
    "self_learning": {
        "trigger_min_feedback": (int,), "lr": (int, float), "epochs": (int,),
        "auto_activate": (bool,), "learning_parallel": (bool,),
    },
    "logging": {"max_mb": (int,), "backups": (int,)},
}

# AC-072（2026-10-13，用户定值“稍宽松”）：业务预算与保留期。
# key → (类型, 默认值, 下限, 上限)；百分比 0~100，天/月/版本数 ≥1。
# 保留期仅为配置与展示，系统不据此自动删除任何数据。
BUSINESS_BUDGET_SPEC: Dict[str, tuple] = {
    "g2_max_fpr_pct": (float, 5.0, 0.0, 100.0),
    "g2_min_recall_pct": (float, 95.0, 0.0, 100.0),
    "g6_max_old_recall_drop_pt": (float, 2.0, 0.0, 100.0),
    "max_gray_ratio_pct": (float, 15.0, 0.0, 100.0),
    "raw_hot_days": (int, 90, 1, 36500),
    "raw_cold_months": (int, 12, 1, 1200),
    "heatmap_overlay_days": (int, 30, 1, 36500),
    "snapshot_keep_recent": (int, 10, 1, 10000),
    "snapshot_keep_activated": (bool, True, None, None),
    "evalrun_audit_months": (int, 12, 1, 1200),
    "label_late_tolerance_days": (int, 7, 1, 3650),
}
# 反馈/事件/WAL 永久保留：固定口径，不开放修改
FEEDBACK_EVENT_WAL_RETENTION = "永久"
_SCHEMA["business_budget"] = {
    k: ((int, float) if t is float else (t,))
    for k, (t, *_r) in BUSINESS_BUDGET_SPEC.items()}


class ConfigValidationError(ValueError):
    """配置写回 schema 校验失败（A16）。"""


def _check_range(section: str, key: str, value: Any) -> None:
    """AC-072：business_budget 段取值范围校验（超界拒绝并提示）。"""
    if section != "business_budget" or key not in BUSINESS_BUDGET_SPEC:
        return
    _t, _d, lo, hi = BUSINESS_BUDGET_SPEC[key]
    if lo is None or isinstance(value, bool):
        return
    if not (lo <= value <= hi):
        raise ConfigValidationError(
            f"{section}.{key} 超出范围：允许 {lo}～{hi}，实得 {value!r}")


class Settings:
    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(
            config_path or os.environ.get("AOI_CONFIG", _DEFAULT_CONFIG))
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._cfg: Dict[str, Any] = yaml.safe_load(f)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._cfg.get(section, {}).get(key, default)

    def section(self, name: str) -> Dict[str, Any]:
        return self._cfg.get(name, {})

    # ── M16d 设置中心：白名单键写回 + 热生效 ──────────────────
    def update(self, section: str, key: str, value: Any) -> None:
        """更新内存配置并写回 yaml（注释不保留——yaml 重写固有限制，
        字段说明以用户手册/设置页 UI 为准）。

        A16：写回前 schema 校验（白名单键的类型约束）+ 原子写回
        （崩溃不留半个 yaml）。bool 与 int 区分（bool 是 int 子类，
        需先判 bool 防 1/0 混进布尔开关）。"""
        types = _SCHEMA.get(section, {}).get(key)
        if types is not None:
            if isinstance(value, bool) and bool not in types:
                raise ConfigValidationError(
                    f"{section}.{key} 不接受布尔值（期望 {types}）")
            if not isinstance(value, types):
                raise ConfigValidationError(
                    f"{section}.{key} 类型错误：期望 {types}，实得 "
                    f"{type(value).__name__}={value!r}")
        _check_range(section, key, value)
        self._cfg.setdefault(section, {})[key] = value
        save_yaml_atomic(self.config_path, self._cfg)

    # ── AC-072 业务预算与保留期 ──────────────────────────────
    def business_budget(self) -> Dict[str, Any]:
        """business_budget 段（缺项/非法存量值回落默认）+ 固定口径项。"""
        sec = self._cfg.get("business_budget") or {}
        out: Dict[str, Any] = {}
        for key, (typ, default, lo, hi) in BUSINESS_BUDGET_SPEC.items():
            v = sec.get(key, default)
            ok = (isinstance(v, bool) if typ is bool else
                  isinstance(v, (int, float)) and not isinstance(v, bool)
                  and (typ is float or isinstance(v, int))
                  and lo <= v <= hi)
            out[key] = typ(v) if ok else default
        out["feedback_event_wal_retention"] = FEEDBACK_EVENT_WAL_RETENTION
        return out

    def candidate_release_policy(self, max_predict_ms: float,
                                 valid_seconds: float) -> Dict[str, float]:
        """由业务预算构造 candidate_release.evaluate 的 policy（比例 0~1）。"""
        b = self.business_budget()
        return {"max_fpr": b["g2_max_fpr_pct"] / 100.0,
                "min_recall": b["g2_min_recall_pct"] / 100.0,
                "max_gray": b["max_gray_ratio_pct"] / 100.0,
                "old_recall_drop": b["g6_max_old_recall_drop_pt"] / 100.0,
                "max_predict_ms": float(max_predict_ms),
                "valid_seconds": float(valid_seconds)}

    # ── 常用路径 ─────────────────────────────────────────────
    @property
    def storage_dir(self) -> Path:
        d = Path(self.get("system", "storage_dir", "storage"))
        # 相对路径基于仓库根解析（打包/换机部署可移植，与 database_url 同口径）
        if not d.is_absolute():
            d = AOI_ROOT / d
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def engine_storage(self) -> Path:
        """检测引擎快照/产物根目录（默认 storage/engine，相对路径基于仓库根）。"""
        d = Path(self.get("system", "engine_storage", "storage/engine"))
        if not d.is_absolute():
            d = AOI_ROOT / d
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def engine_base_cfg(self) -> Path | None:
        """检测引擎基础配置 yaml；相对路径基于仓库根（AOI_ROOT），
        文件不存在时返回 None（由引擎工厂报错提示）。"""
        p = self.get("system", "engine_base_cfg", "configs/engine_fast.yaml")
        if not p:
            return None
        path = Path(p)
        if not path.is_absolute():
            path = AOI_ROOT / path
        return path if path.is_file() else None

    @property
    def database_url(self) -> str:
        url = self.get("system", "database_url")
        # 相对 sqlite 路径基于仓库根解析（打包/换机部署可移植，2026-08-30）
        if isinstance(url, str) and url.startswith("sqlite:///"):
            p = Path(url[len("sqlite:///"):])
            if not p.is_absolute():
                url = "sqlite:///" + (AOI_ROOT / p).as_posix()
        return url

    def get_notify(self) -> Dict[str, Any]:
        """webhook 推送配置：{webhook_url(空=禁用), events, timeout_s}。"""
        sec = self._cfg.get("notify", {}) or {}
        return {
            "webhook_url": sec.get("webhook_url") or "",
            "events": list(sec.get("events") or ["anomaly"]),
            "timeout_s": float(sec.get("timeout_s", 3)),
        }

    def get_alarm_cfg(self) -> Dict[str, Any]:
        """连续异常告警配置（M6a，notify 段）：
        {alarm_window(滑窗帧数), alarm_k(触发阈值), alarm_cooldown_s(冷却秒)}。"""
        sec = self._cfg.get("notify", {}) or {}
        return {
            "alarm_window": int(sec.get("alarm_window", 20)),
            "alarm_k": int(sec.get("alarm_k", 5)),
            "alarm_cooldown_s": float(sec.get("alarm_cooldown_s", 60)),
        }

    def storage(self, sub: str) -> Path:
        d = self.storage_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        return d


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
