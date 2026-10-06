"""测试总执行器（demo5 引擎版）。

用法（AOI_sys 根目录）：
    python -m test.run_all            # 全量（引擎冒烟 + 后端端到端 + UI 离屏）
    python -m test.run_all t15        # 仅 UI 离屏冒烟

M2 重构后说明：旧 demo1 契约测试（t01-t19 多数：L1/L3 级联、校准集、
EWC 自学习、with_l3 等）已随 model_adapter 下线失效，由以下三者替代：

  _smoke_m1.py  引擎层冒烟（fit→predict→save→load→predict 一致性）
  _smoke_m2.py  后端端到端（prepare→detect→feedback 即学→consolidate→
                A/B→学习曲线→贡献档案→复核闭环→激活门控→webhook→
                在线曲线→复核框选→连续告警→precheck）
  t15           UI 离屏冒烟（8 页面实例化+切换）

历史 demo1 测试脚本已归档至 test/_archive_demo1/（不删除，供参考）。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

AOI_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("引擎层冒烟 _smoke_m1", [sys.executable, "tests/_smoke_m1.py"], {}),
    ("后端端到端 _smoke_m2", [sys.executable, "tests/_smoke_m2.py"], {}),
    ("产线安全集成 _smoke_m10", [sys.executable, "tests/_smoke_m10.py"], {}),
    ("评审改进 _smoke_m11", [sys.executable, "tests/_smoke_m11.py"], {}),
    ("分层契约补全 _smoke_m14", [sys.executable, "tests/_smoke_m14.py"], {}),
    ("操作流对账 _smoke_m15", [sys.executable, "tests/_smoke_m15.py"], {}),
    ("UI 离屏冒烟 t15", [sys.executable, "-m", "test.t15_ui_offscreen"],
     {"QT_QPA_PLATFORM": "offscreen"}),
]


def main() -> None:
    selected = sys.argv[1:]
    failed = []
    for name, cmd, extra_env in STEPS:
        if selected and not any(s in name for s in selected):
            continue
        print(f"\n{'='*60}\n[run_all] {name}\n{'='*60}", flush=True)
        env = {**os.environ, **extra_env}
        r = subprocess.run(cmd, cwd=AOI_ROOT, env=env)
        if r.returncode != 0:
            failed.append(name)
    print(f"\n{'='*60}")
    if failed:
        print(f"✗ 失败: {', '.join(failed)}")
        sys.exit(1)
    print("✓ 全部冒烟通过")


if __name__ == "__main__":
    main()
