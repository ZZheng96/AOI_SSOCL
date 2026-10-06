"""PCB_Ins v2 全局配置。

双层设计：
1. ``Settings``：v2 服务层配置（yaml 加载，供 API/引擎使用）
2. PCB_Ins 旧接口常量/函数：兼容从 PCB_Ins 迁入的 detect/template/recipe/
   calibration/result 等模块（RECIPES_DIR、TEMPLATE_CALIB_DIR、get_gate_settings…），
   数据目录统一收敛到 storage/ 与 alg_repo/。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT_DIR / "configs" / "default.yaml"


class Settings:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self, section: str, key: str, default: Any = None) -> Any:
        sec = self._data.get(section) or {}
        if isinstance(sec, dict) and key in sec:
            return sec[key]
        return default

    def section(self, section: str) -> dict[str, Any]:
        sec = self._data.get(section)
        return dict(sec) if isinstance(sec, dict) else {}

    # ── 便捷属性 ──────────────────────────────
    @property
    def host(self) -> str:
        return os.environ.get("PCBINS_HOST") or str(self.get("system", "host", "127.0.0.1"))

    @property
    def port(self) -> int:
        return int(os.environ.get("PCBINS_PORT") or self.get("system", "port", 8021))

    def storage(self, sub: str) -> Path:
        root = Path(self.get("storage", "root", "storage"))
        p = ROOT_DIR / root / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def snapshots_dir(self) -> Path:
        return self.storage("snapshots")

    @property
    def logs_dir(self) -> Path:
        return self.storage("logs")

    @property
    def outputs_dir(self) -> Path:
        return self.storage("outputs")

    @property
    def alg_repo_root(self) -> Path:
        rel = Path(self.get("alg_repo", "root", "alg_repo"))
        return ROOT_DIR / rel

    @property
    def templates_dir(self) -> Path:
        rel = Path(self.get("templates", "dir", "templates"))
        p = ROOT_DIR / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def feature_base_cfg(self) -> Path:
        rel = Path(self.get("feature_engine", "base_cfg", "configs/demo5_fast.yaml"))
        p = ROOT_DIR / rel
        if not p.is_absolute():
            p = ROOT_DIR / rel
        return p

    @property
    def device(self) -> str:
        dev = str(self.get("feature_engine", "device", "auto"))
        if dev == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return dev


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data: dict[str, Any] = {}
    if CONFIG_FILE.is_file():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                data = loaded
    return Settings(data)


# ────────────────────────────────────────────────────────────
# PCB_Ins 旧接口兼容（迁入模块引用）
# ────────────────────────────────────────────────────────────

# 数据目录统一到 storage/ 下
OUTPUTS_DIR = ROOT_DIR / "storage" / "outputs"
RECIPES_DIR = ROOT_DIR / "storage" / "recipes"
TEMPLATE_CALIB_DIR = ROOT_DIR / "storage" / "template_calib"
TEMPLATES_DIR = ROOT_DIR / "templates"
TEMPLATE_MAP_FILE = ROOT_DIR / "template_map.json"
SETTINGS_FILE = ROOT_DIR / "storage" / "settings.json"
APP_DIR = ROOT_DIR / "app"

# 算法仓库位置（系统内预留；SMT 已从 D:\\AOI\\PCBA_detection\\test_code\\Dev_solder_smt_Alg 复制）
ALG_REPO_DIR = ROOT_DIR / "alg_repo"
TH_REPO_DIR = ALG_REPO_DIR / "through_hole"
SMT_REPO_DIR = ALG_REPO_DIR / "smt"
GOLD_REPO_DIR = ALG_REPO_DIR / "gold_finger"
WRINKLE_REPO_DIR = ALG_REPO_DIR / "wrinkle"
BODY_REPO_DIR = ALG_REPO_DIR / "body"
SHIFT_REPO_DIR = ALG_REPO_DIR / "shift_smt"

DEFAULT_SETTINGS = {
    "outputs_dir": str(OUTPUTS_DIR),
    "gate_enabled": True,
    "size_gate_max_diff_px": 10,
    "stop_on_error": False,
}

DEFAULT_CATEGORIES = ["电阻", "电容", "二极管", "LED", "IC", "插件器件", "其他"]


def _ensure_dirs() -> None:
    for p in (OUTPUTS_DIR, RECIPES_DIR, TEMPLATE_CALIB_DIR, TEMPLATES_DIR, ALG_REPO_DIR):
        p.mkdir(parents=True, exist_ok=True)
    for sub in ("through_hole", "smt", "gold_finger", "wrinkle", "body"):
        (ALG_REPO_DIR / sub).mkdir(parents=True, exist_ok=True)


# PCB_Ins UI 兼容别名
ensure_dirs = _ensure_dirs


_ensure_dirs()


def load_settings() -> dict:
    data = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            loaded = json_loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
        except Exception:
            pass
    return data


def save_settings(settings: dict) -> None:
    import json
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def json_loads(text: str) -> Any:
    import json
    return json.loads(text)


def get_outputs_dir() -> Path:
    settings = load_settings()
    path = Path(settings.get("outputs_dir") or OUTPUTS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_outputs_dir(path: str | Path) -> Path:
    settings = load_settings()
    settings["outputs_dir"] = str(path)
    save_settings(settings)
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_gate_settings() -> dict:
    settings = load_settings()
    return {
        "gate_enabled": bool(settings.get("gate_enabled", True)),
        "size_gate_max_diff_px": int(settings.get("size_gate_max_diff_px", 10)),
    }


def set_gate_settings(gate_enabled: bool, size_gate_max_diff_px: int) -> None:
    settings = load_settings()
    settings["gate_enabled"] = bool(gate_enabled)
    settings["size_gate_max_diff_px"] = int(size_gate_max_diff_px)
    save_settings(settings)


def get_stop_on_error() -> bool:
    return bool(load_settings().get("stop_on_error", False))


def set_stop_on_error(enabled: bool) -> None:
    settings = load_settings()
    settings["stop_on_error"] = bool(enabled)
    save_settings(settings)
