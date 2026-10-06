"""测试总执行器。

用法（AOI_sys 根目录）：
    python -m tests.run_all            # 全量（引擎冒烟 + 后端端到端 + UI 离屏）
    python -m tests.run_all ui         # 仅 UI 离屏冒烟
    python -m tests.run_all m01 m02    # 按关键字过滤

目录结构见 tests/__init__.py；性能/长稳实测（bench/）不在本编排内，
需后端已启动后单独运行。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

AOI_ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("引擎层冒烟 m01", [sys.executable, "-m", "tests.smoke.m01_engine"], {}),
    ("后端端到端 m02", [sys.executable, "-m", "tests.smoke.m02_e2e"], {}),
    ("产线安全集成 m10", [sys.executable, "-m", "tests.smoke.m10_line_safety"], {}),
    ("评审改进 m11", [sys.executable, "-m", "tests.smoke.m11_roles"], {}),
    ("分层契约补全 m14", [sys.executable, "-m", "tests.smoke.m14_template"], {}),
    ("操作流对账 m15", [sys.executable, "-m", "tests.smoke.m15_feedback"], {}),
    ("UI 离屏冒烟 ui", [sys.executable, "-m", "tests.ui.offscreen"],
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
