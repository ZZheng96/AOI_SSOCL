"""系统配置加载：YAML + 单例。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

AOI_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = AOI_ROOT / "configs" / "default.yaml"


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
        字段说明以用户手册/设置页 UI 为准）。"""
        self._cfg.setdefault(section, {})[key] = value
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._cfg, f, allow_unicode=True, sort_keys=False)

    # ── 常用路径 ─────────────────────────────────────────────
    @property
    def storage_dir(self) -> Path:
        d = Path(self.get("system", "storage_dir"))
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def demo5_storage(self) -> Path:
        """Demo5 引擎快照/产物根目录（默认 storage/demo5，相对路径基于仓库根）。"""
        d = Path(self.get("system", "demo5_storage", "storage/demo5"))
        if not d.is_absolute():
            d = AOI_ROOT / d
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def demo5_base_cfg(self) -> Path | None:
        """Demo5 引擎基础配置 yaml；相对路径基于仓库根（AOI_ROOT），
        文件不存在时返回 None（由引擎工厂报错提示）。"""
        p = self.get("system", "demo5_base_cfg", "configs/demo5_fast.yaml")
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
