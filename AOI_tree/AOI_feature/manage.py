"""特征集合管理 CLI：对 feature_set.yaml 做增删改查 + 一致性校验

用法:
  python manage.py list                        # 特征组清单（维度/启停/参数改动）
  python manage.py show glcm                   # 单组详情（理由/目标缺陷/参数）
  python manage.py disable hog                 # 停用（保留配置，随时 enable 恢复）
  python manage.py enable hog
  python manage.py set glcm levels=64          # 修改计算参数（值按 YAML 解析类型）
  python manage.py add lap_var                 # 加入自定义特征（custom_features.py 定义）
  python manage.py remove lap_var              # 删除组
  python manage.py reset                       # 恢复主干预设（trunk_200）
  python manage.py parity                      # 与主干 extractor 逐图校验一致
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from AOI_feature.feature_set import (DEFAULT_CONFIG, FeatureSet,   # noqa: E402
                                     load_custom_features)
from AOI_feature.feature_lib import FEATURE_FUNCS                   # noqa: E402
from AOI_feature.presets import trunk_preset                        # noqa: E402


def cmd_list(fs: FeatureSet, _):
    meta = {m["key"]: m for m in fs.probe()}
    print(f"配置: {DEFAULT_CONFIG}（{fs.config.get('name')}）")
    print(f"{'key':<10} {'组名':<14} {'维':>4} {'状态':<4} 参数改动")
    for g in fs.groups:
        m = meta.get(g["key"])
        dim = m["dim"] if m else "(已停用)"
        mark = "✓" if g["enabled"] else "✗"
        diff = {k: v for k, v in g["params"].items()
                if v != FEATURE_FUNCS[g["key"]]["default_params"].get(k)}
        print(f"{g['key']:<10} {FEATURE_FUNCS[g['key']]['name']:<14} "
              f"{str(dim):>4} {mark:<4} {diff if diff else ''}")
    print(f"\n启用组合计维度: {fs.dim}")


def cmd_show(fs: FeatureSet, args):
    g = fs._find(args.key)
    reg = FEATURE_FUNCS[args.key]
    print(f"{reg['name']}（key={args.key}，{'启用' if g['enabled'] else '停用'}）")
    print(f"  选择理由: {reg['rationale']}")
    print(f"  目标缺陷: {reg['targets']}")
    params = {**reg["default_params"], **g["params"]}
    print(f"  参数（当前值 / 默认值）:")
    for k, v in params.items():
        d = reg["default_params"].get(k)
        print(f"    {k} = {v}" + ("" if v == d else f"  (默认 {d})"))


def _save_and_echo(fs: FeatureSet, msg: str):
    fs.save()
    print(f"[ok] {msg}，已写回 {DEFAULT_CONFIG}")


def cmd_parity(fs: FeatureSet, args):
    """当前配置 vs 主干 extractor：同一批探针图逐图比对（两侧都按 U27 播种包裹）"""
    from AOI_feature import ensure_core_importable
    ensure_core_importable()
    from algo.vendor.traditional import TraditionalFeatureExtractor
    trunk = TraditionalFeatureExtractor()
    rng = np.random.default_rng(0)
    max_diff, n = 0.0, args.n
    for i in range(n):
        img = rng.integers(0, 256, (128, 96, 3), dtype=np.uint8)
        state = np.random.get_state()
        np.random.seed(0)
        try:
            v_trunk = trunk.extract(img)
        finally:
            np.random.set_state(state)
        state = np.random.get_state()
        np.random.seed(0)
        try:
            v_mine = fs.extract(img)
        finally:
            np.random.set_state(state)
        if len(v_trunk) != len(v_mine):
            print(f"[FAIL] 维度不一致: 主干 {len(v_trunk)} vs 配置 {len(v_mine)}"
                  f"（有组被停用/删除或参数改维度属预期，parity 仅校验主干预设）")
            sys.exit(1)
        max_diff = max(max_diff, float(np.abs(v_trunk - v_mine).max()))
    ok = max_diff < 1e-4
    print(f"[{'ok' if ok else 'FAIL'}] {n} 图最大逐维差 = {max_diff:.3e}")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description="AOI 特征集合管理")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("show"); p.add_argument("key")
    p = sub.add_parser("enable"); p.add_argument("key")
    p = sub.add_parser("disable"); p.add_argument("key")
    p = sub.add_parser("set"); p.add_argument("key"); p.add_argument("assignments", nargs="+")
    p = sub.add_parser("add"); p.add_argument("key")
    p.add_argument("--param", action="append", default=[],
                   help="初始参数 k=v（YAML 解析）")
    p = sub.add_parser("remove"); p.add_argument("key")
    sub.add_parser("reset")
    p = sub.add_parser("parity"); p.add_argument("-n", type=int, default=5)
    args = ap.parse_args()

    load_custom_features()
    fs = FeatureSet.load()

    if args.cmd == "list":
        cmd_list(fs, args)
    elif args.cmd == "show":
        cmd_show(fs, args)
    elif args.cmd == "enable":
        fs.set_enabled(args.key, True)
        _save_and_echo(fs, f"{args.key} 已启用")
    elif args.cmd == "disable":
        fs.set_enabled(args.key, False)
        _save_and_echo(fs, f"{args.key} 已停用（特征向量维度随之变化）")
    elif args.cmd == "set":
        for a in args.assignments:
            k, _, v = a.partition("=")
            fs.set_param(args.key, k, yaml.safe_load(v))
            print(f"[ok] {args.key}.{k} = {yaml.safe_load(v)}")
        _save_and_echo(fs, f"{args.key} 参数已更新")
    elif args.cmd == "add":
        params = dict(a.split("=", 1) for a in args.param)
        params = {k: yaml.safe_load(v) for k, v in params.items()}
        fs.add(args.key, params)
        _save_and_echo(fs, f"{args.key} 已加入特征集")
    elif args.cmd == "remove":
        fs.remove(args.key)
        _save_and_echo(fs, f"{args.key} 已删除")
    elif args.cmd == "reset":
        FeatureSet(trunk_preset()).save()
        print(f"[ok] 已恢复主干预设 trunk_200 -> {DEFAULT_CONFIG}")
    elif args.cmd == "parity":
        cmd_parity(fs, args)


if __name__ == "__main__":
    main()
