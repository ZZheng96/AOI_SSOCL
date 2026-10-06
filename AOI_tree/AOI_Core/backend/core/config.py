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
        "auto_activate": (bool,),
    },
    "logging": {"max_mb": (int,), "backups": (int,)},
}


class ConfigValidationError(ValueError):
    """配置写回 schema 校验失败（A16）。"""


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
        self._cfg.setdefault(section, {})[key] = value
        save_yaml_atomic(self.config_path, self._cfg)

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
