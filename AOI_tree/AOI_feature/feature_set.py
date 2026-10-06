"""FeatureSet：配置驱动的特征集合（增删改查 + 持久化 + 动态切片）

配置文件（YAML）即特征集合的唯一事实来源：
  name / preprocess / groups: [{key, enabled, params}]

- 提取：extract(img) 只计算 enabled 组，特征向量 = 启用组按配置顺序拼接
- 维度：不做静态登记，载入时对探针图逐组实测（维度永远与实际输出一致）
- 随机性：extract 内对每张图 seed(0) 包裹（U27 口径，主干 TradSlot._vec 同款）
- 自定义特征：custom_features.py 中的 @feature 函数，经 add_custom 加入配置
"""
import importlib.util
import os

import numpy as np
import yaml

from .feature_lib import FEATURE_FUNCS, make_ctx, preprocess
from .presets import trunk_preset

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "feature_set.yaml")
CUSTOM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "custom_features.py")


def load_custom_features(path: str = CUSTOM_FILE) -> list:
    """加载自定义特征文件（其中 @feature 装饰器会把函数注册进 FEATURE_FUNCS）"""
    if not os.path.isfile(path):
        return []
    before = set(FEATURE_FUNCS)
    spec = importlib.util.spec_from_file_location("aoi_custom_features", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return sorted(set(FEATURE_FUNCS) - before)


class FeatureSet:
    def __init__(self, config: dict):
        self.config = config
        self.preprocess_cfg = config.get("preprocess",
                                         {"max_side": 256, "min_side": 32})
        self.groups = config["groups"]
        for g in self.groups:
            if g["key"] not in FEATURE_FUNCS:
                raise KeyError(f"特征 '{g['key']}' 未注册（内置见 feature_lib，"
                               f"自定义见 custom_features.py）")
            g.setdefault("enabled", True)
            g.setdefault("params", {})

    # ── 持久化 ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str = DEFAULT_CONFIG) -> "FeatureSet":
        """配置不存在时用主干预设初始化（落盘一份，之后即唯一事实来源）"""
        load_custom_features()
        if not os.path.isfile(path):
            fs = cls(trunk_preset())
            fs.save(path)
            return fs
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def save(self, path: str = DEFAULT_CONFIG):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)

    # ── 查 ───────────────────────────────────────────────────────────

    def _find(self, key: str) -> dict:
        for g in self.groups:
            if g["key"] == key:
                return g
        raise KeyError(f"特征 '{key}' 不在当前配置中（manage.py list 查看现有组）")

    def enabled_groups(self) -> list:
        return [g for g in self.groups if g["enabled"]]

    # ── 改 ───────────────────────────────────────────────────────────

    def set_enabled(self, key: str, enabled: bool):
        self._find(key)["enabled"] = enabled

    def set_param(self, key: str, param: str, value):
        g = self._find(key)
        defaults = FEATURE_FUNCS[key]["default_params"]
        if param not in defaults:
            raise KeyError(f"参数 '{param}' 不存在，可选: {sorted(defaults)}")
        g["params"][param] = type(defaults[param])(value) \
            if not isinstance(defaults[param], list) else value

    def add(self, key: str, params: dict | None = None, enabled: bool = True):
        if any(g["key"] == key for g in self.groups):
            raise KeyError(f"特征 '{key}' 已存在（用 set 改参数或先 remove）")
        self._require_registered(key)
        self.groups.append({"key": key, "enabled": enabled,
                            "params": params or {}})

    def remove(self, key: str):
        self.groups.remove(self._find(key))

    def _require_registered(self, key: str):
        if key not in FEATURE_FUNCS:
            raise KeyError(f"特征 '{key}' 未注册——在 custom_features.py 用 "
                           f"@feature 装饰器定义后重试")

    # ── 提取（主干同口径）────────────────────────────────────────────

    def extract_one(self, img) -> np.ndarray:
        """单图提取（调用方负责 RNG 播种；批量提取请用 scoring.extract_features）"""
        img = preprocess(img, **self.preprocess_cfg)
        ctx = make_ctx(img)
        out = []
        for g in self.enabled_groups():
            f = FEATURE_FUNCS[g["key"]]["func"]
            params = {**FEATURE_FUNCS[g["key"]]["default_params"], **g["params"]}
            out.extend(f(ctx, params))
        return np.asarray(out, dtype=np.float32)

    # 与主干 extractor 同名的入口，供评估链路透明替换
    extract = extract_one

    def probe(self) -> list:
        """探针图实测各启用组维度，返回 [{key,name,dim,slice,...}]（顺序=拼接顺序）"""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        img = preprocess(img, **self.preprocess_cfg)
        ctx = make_ctx(img)
        meta, pos = [], 0
        for g in self.enabled_groups():
            reg = FEATURE_FUNCS[g["key"]]
            params = {**reg["default_params"], **g["params"]}
            dim = len(reg["func"](ctx, params))
            meta.append({"key": g["key"], "name": reg["name"], "dim": dim,
                         "slice": slice(pos, pos + dim),
                         "rationale": reg["rationale"], "targets": reg["targets"],
                         "enabled": True, "params": g["params"]})
            pos += dim
        return meta

    @property
    def dim(self) -> int:
        return sum(m["dim"] for m in self.probe())
