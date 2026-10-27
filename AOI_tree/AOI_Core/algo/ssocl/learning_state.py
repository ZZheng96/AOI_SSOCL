"""冷快照中的候选续训状态；默认不恢复进程级 RNG、锁或线程池。"""
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
import random

candidate_rng_active = ContextVar("candidate_rng_active", default=False)
import functools
import math

import numpy as np
import torch

VERSION = 2
FIELDS = (
    "difficult", "high_conf", "fuse_blocked", "_w_fb_since_update", "_w_cooldown",
    "_recent_fb", "_normal_fb_raw", "_normal_fb_cal", "_hr_cooldown",
    "_n_labeled_fb", "box_pos_feats", "normal_fb_feats", "_head_ft_counts",
    "_head_ft_cooldown", "disc_pos_patch_feats", "disc_neg_patch_feats",
    "_disc_ft_counts", "_disc_ft_cooldown", "_ws_since", "_labeled_items",
    "_lab_seen", "_preq", "normal_fused_fb", "defect_fused_fb", "_tr_last_recal")
WL_FIELDS = ("slots", "min_neg", "n", "pos", "neg", "n_updates", "height_suppress",
             "height_log", "_height_flagged", "initial_weights", "prior_fade")
GATE_FIELDS = ("tol", "anchor_size", "anchor_paths", "base_mean", "base_std")
RNG_FIELDS = ("python_rng", "numpy_rng", "torch_global_rng", "cuda_rng")


def _cuda_rng_state():
    if not torch.cuda.is_available():
        return None
    return [s.detach().cpu().clone() for s in torch.cuda.get_rng_state_all()]


def _data(value):
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: _data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_data(v) for v in value)
    return deepcopy(value)


def _plain(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("学习态包含非有限值")
        return
    if torch.is_tensor(value):
        if value.layout != torch.strided or not torch.isfinite(value).all():
            raise ValueError("学习态张量无效")
        return
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError("学习态数组无效")
        return
    if type(value) in (list, tuple, set):
        for v in value:
            _plain(v)
        return
    if type(value) is dict:
        for k, v in value.items():
            if type(k) not in (str, int):
                raise ValueError("学习态键无效")
            _plain(v)
        return
    raise ValueError("学习态禁止对象: " + type(value).__name__)


_ITEM_TRANSIENT = ("_clf_imgfeat",)  # predict 时挂的预取 Future，不可持久化


def _item(it):
    return {k: v for k, v in it.items() if k not in _ITEM_TRANSIENT} if type(it) is dict else it


def capture(pipe):
    h = pipe.handler
    handler = {k: _data([type(p)((_item(p[0]), *p[1:])) for p in h._labeled_items]
                        if k == "_labeled_items" else getattr(h, k)) for k in FIELDS}
    state = {"version": VERSION, "handler": handler,
             "config": _data(h._learning_cfg),
             "weight_learner": {k: _data(getattr(h.weight_learner, k)) for k in WL_FIELDS},
             "gate": {k: _data(getattr(h.gate, k)) for k in GATE_FIELDS},
             "lab_rng": _data(h._lab_rng.bit_generator.state),
             "python_rng": _data(random.getstate()),
             "numpy_rng": _data(np.random.get_state()),
             "torch_rng": h._candidate_torch_gen.get_state().clone(),
             "torch_global_rng": torch.get_rng_state().clone(),
             "cuda_rng": _cuda_rng_state(),
             "anchor_items": _data([_item(it) for it in (getattr(pipe, "_anchor_items", []) or [])]),
             "train_normal_paths": list(getattr(pipe, "_train_normal_paths", []) or []),
             "cal_hist": _data(pipe._cal_normal_hist),
             "head_manager_present": pipe.head_mgr is not None,
             "stream": {"recent_decisions": list(h.normal_bank.recent_decisions),
                        "defect_ratio": h.normal_bank.defect_ratio}}
    validate(state, pipe)
    return state


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("学习态字段不匹配: " + name)


def validate(state, pipe):
    _keys(state, ("version", "handler", "config", "weight_learner", "gate", "lab_rng",
                  "python_rng", "numpy_rng", "torch_rng", "torch_global_rng", "cuda_rng",
                  "anchor_items", "train_normal_paths", "cal_hist", "stream",
                  "head_manager_present"), "root")
    if type(state["version"]) is not int or state["version"] != VERSION:
        if state.get("version") == 1:
            raise ValueError("历史 v1 学习态缺少进程 RNG，不能安全恢复；请从 v2 快照重建")
        raise ValueError("未知候选学习态版本")
    _plain(state)
    if type(state["head_manager_present"]) is not bool:
        raise ValueError("孵育管理器存在性无效")
    if type(state["config"]) is not dict or type(state["anchor_items"]) is not list:
        raise ValueError("学习配置/锚定集类型无效")
    if type(state["train_normal_paths"]) is not list or any(type(p) is not str for p in state["train_normal_paths"]):
        raise ValueError("训练路径无效")
    _keys(state["handler"], FIELDS, "handler")
    _keys(state["weight_learner"], WL_FIELDS, "weight_learner")
    _keys(state["gate"], GATE_FIELDS, "gate")
    _keys(state["stream"], ("recent_decisions", "defect_ratio"), "stream")
    h = state["handler"]
    for key in ("difficult", "high_conf", "_recent_fb", "_labeled_items", "_preq", "normal_fused_fb", "defect_fused_fb"):
        if type(h[key]) is not list:
            raise ValueError("学习缓存类型无效: " + key)
    if type(h["fuse_blocked"]) is not bool or type(h["_hr_cooldown"]) is not dict:
        raise ValueError("熔断/冷却类型无效")
    if any(k not in pipe.slots or type(v) is not int for k, v in h["_hr_cooldown"].items()):
        raise ValueError("冷却槽位无效")
    if any(type(pair) not in (tuple, list) or len(pair) != 2 for pair in h["_labeled_items"]):
        raise ValueError("蓄水池条目无效")
    for key in FIELDS:
        if key in ("_n_labeled_fb", "_w_fb_since_update", "_ws_since") or key.endswith("cooldown") and key != "_hr_cooldown" or key == "_tr_last_recal":
            if type(h[key]) is not int:
                raise ValueError("学习计数器无效: " + key)
    for key in ("_lab_seen", "_head_ft_counts", "_disc_ft_counts"):
        _keys(h[key], (0, 1) if key == "_lab_seen" else ("pos", "neg"), key)
        if any(type(v) is not int or v < 0 for v in h[key].values()):
            raise ValueError("学习计数器无效: " + key)
    slots = list(pipe.slots)
    wl = state["weight_learner"]
    if wl["slots"] != slots:
        raise ValueError("学习槽位不匹配")
    for key in ("n", "pos", "neg"):
        _keys(wl[key], slots, key)
        for pair in wl[key].values():
            if type(pair) is not list or len(pair) != 2 or any(type(v) is not int or v < 0 for v in pair) or pair[1] > pair[0]:
                raise ValueError("权重计数无效")
    for key in ("_normal_fb_raw", "_normal_fb_cal"):
        _keys(h[key], slots, key)
        if any(type(v) is not list or any(type(x) not in (int, float) for x in v) for v in h[key].values()):
            raise ValueError("校准缓存无效")
    for name, fields in (("shead", ("box_pos_feats", "normal_fb_feats")),
                         ("disc", ("disc_pos_patch_feats", "disc_neg_patch_feats"))):
        slot = pipe.slots.get(name)
        dim = getattr(slot, "in_dim", None)
        if slot is not None and dim is None:
            dim = next((m.in_features for m in slot.head.modules() if isinstance(m, torch.nn.Linear)), None)
        for key in fields:
            if type(h[key]) is not list:
                raise ValueError("训练缓存类型无效")
            for tensor in h[key]:
                if not torch.is_tensor(tensor) or tensor.ndim != 2 or dim is None or tensor.shape[1] != dim or tensor.shape[0] <= 0 or not tensor.is_floating_point():
                    raise ValueError("训练缓存 shape 不匹配: " + key)
    for key in ("n_updates", "min_neg", "prior_fade"):
        if type(wl[key]) is not int or wl[key] < 0:
            raise ValueError("权重参数无效")
    if type(wl["_height_flagged"]) is not set or not wl["_height_flagged"] <= set(slots):
        raise ValueError("权重去重状态无效")
    if type(state["cal_hist"]) is not dict or not set(state["cal_hist"]) <= set(slots):
        raise ValueError("校准历史槽位无效")
    for seq in list(state["cal_hist"].values()) + [h["normal_fused_fb"], h["defect_fused_fb"]]:
        if type(seq) is not list or any(type(v) not in (int, float) for v in seq):
            raise ValueError("分数历史无效")
    stream = state["stream"]
    if type(stream["recent_decisions"]) is not list or any(type(v) is not int or v not in (0, 1) for v in stream["recent_decisions"]):
        raise ValueError("流窗口无效")
    if type(stream["defect_ratio"]) not in (int, float) or not 0 <= stream["defect_ratio"] <= 1:
        raise ValueError("流缺陷率无效")
    for item in state["anchor_items"] + [it for it, _ in h["_labeled_items"]]:
        if type(item) is not dict or "path" not in item or "feats" not in item:
            raise ValueError("重打分 item 缺状态")
        if not set(item) <= {"path", "feats", "img", "tile_imgs", "tiles", "tile_masks", "_missed"}:
            raise ValueError("重打分 item 未知字段")
        f = item["feats"]
        if f is not None and (not torch.is_tensor(f) or f.ndim != 4 or min(f.shape) <= 0):
            raise ValueError("重打分特征 shape 无效")
    if any(type(y) is not int or y not in (0, 1) for _, y in h["_labeled_items"]):
        raise ValueError("反馈标签无效")
    gate = state["gate"]
    if gate["anchor_paths"] and (type(gate["base_mean"]) not in (float, int) or type(gate["base_std"]) not in (float, int) or gate["base_std"] <= 0):
        raise ValueError("门控基准缺失")
    if [it["path"] for it in state["anchor_items"]] != gate["anchor_paths"]:
        raise ValueError("门控锚定特征缺失/顺序不一致")
    rng = np.random.default_rng()
    rng.bit_generator.state = state["lab_rng"]
    gen = torch.Generator(device="cpu")
    gen.set_state(state["torch_rng"])
    if type(state["python_rng"]) is not tuple or len(state["python_rng"]) != 3:
        raise ValueError("Python RNG 状态无效")
    if type(state["numpy_rng"]) is not tuple or len(state["numpy_rng"]) != 5:
        raise ValueError("NumPy RNG 状态无效")
    if not torch.is_tensor(state["torch_global_rng"]) or state["torch_global_rng"].dtype != torch.uint8:
        raise ValueError("Torch 全局 RNG 状态无效")
    try:
        random.Random().setstate(state["python_rng"])
        np.random.RandomState().set_state(state["numpy_rng"])
        gen.set_state(state["torch_global_rng"])
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError("进程 RNG 状态无效") from exc
    cuda_rng = state["cuda_rng"]
    if cuda_rng is not None:
        if (not torch.cuda.is_available() or type(cuda_rng) is not list or
                len(cuda_rng) != torch.cuda.device_count() or
                any(not torch.is_tensor(s) or s.dtype != torch.uint8 for s in cuda_rng)):
            raise ValueError("CUDA RNG 状态与当前环境不兼容")
        try:
            for index, rng_state in enumerate(cuda_rng):
                torch.Generator(device=f"cuda:{index}").set_state(rng_state)
        except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
            raise ValueError("CUDA RNG 状态无效") from exc


def restore(pipe, state, restore_global_rng=False):
    validate(state, pipe)
    h = pipe.handler
    for k, v in state["handler"].items():
        setattr(h, k, _data(v))
    for k, v in state["weight_learner"].items():
        setattr(h.weight_learner, k, _data(v))
    for k, v in state["gate"].items():
        setattr(h.gate, k, _data(v))
    h._lab_rng.bit_generator.state = state["lab_rng"]
    h._candidate_torch_gen.set_state(state["torch_rng"])
    if restore_global_rng:
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_global_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    pipe._anchor_items = _data(state["anchor_items"])
    pipe._train_normal_paths = list(state["train_normal_paths"])
    pipe._cal_normal_hist = _data(state["cal_hist"])
    h.normal_bank.recent_decisions = list(state["stream"]["recent_decisions"])
    h.normal_bank.defect_ratio = state["stream"]["defect_ratio"]
    h._labeled_scores = None


class _LocalDropout(torch.nn.Module):
    def __init__(self, original, generator):
        super().__init__()
        self.p, self.inplace, self.generator = original.p, original.inplace, generator
        self.train(original.training)

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        if self.p == 1:
            return x.mul_(0) if self.inplace else x * 0
        mask = (torch.rand(x.shape, generator=self.generator, device="cpu") >= self.p).to(x.device, x.dtype) / (1 - self.p)
        return x.mul_(mask) if self.inplace else x * mask


def candidate_random(method):
    @functools.wraps(method)
    def run(h, *args, **kwargs):
        if not getattr(h.p, "_candidate_mode", False):
            return method(h, *args, **kwargs)
        changed = []
        token = candidate_rng_active.set(True)
        try:
            for slot in h.p.slots.values():
                head = getattr(slot, "head", None)
                if head is None:
                    continue
                for parent in list(head.modules()):
                    for name, child in list(parent.named_children()):
                        if type(child) is torch.nn.Dropout:
                            changed.append((parent, name, child))
                            setattr(parent, name, _LocalDropout(child, h._candidate_torch_gen))
                        elif isinstance(child, (torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)):
                            raise ValueError("候选暂不支持该随机层")
            return method(h, *args, **kwargs)
        finally:
            candidate_rng_active.reset(token)
            for parent, name, child in reversed(changed):
                child.train(getattr(parent, name).training)
                setattr(parent, name, child)
    return run
