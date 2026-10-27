"""TEST-05 离线/干净缓存加载验证（AC-075）。

场景（均为独立子进程 + 全新空 HOME/USERPROFILE，绝不清用户真实缓存）：
  A 离线 + 随包 vendor 权重存在       → 应本地加载成功
  B 离线 + vendor 指向空目录 + 无缓存 → 应明确报错；记录耗时与是否疑似联网
  C 离线 + vendor 有 hubconf 缺权重 + 无缓存 → 应回退后明确报错
断网以 HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 模拟；torch.hub 直连 GitHub
不受这两个变量控制，B/C 的报错类型可证明是否尝试联网（URLError/Timeout=尝试联网）。
"""
import json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

TASK = Path(r"d:\CGAIC\AOI_SSOCL\review\task\test05_offline_weights")
VENDOR = Path(r"d:\CGAIC\AOI_SSOCL\AOI_tree\assets\dinov2")
CHILD = TASK / "raw" / "child_load.py"

scenarios = {}


def run(name, extra_env, timeout=180):
    fake_home = Path(tempfile.mkdtemp(prefix=f"aoi_test05_{name}_"))
    env = dict(os.environ)
    env.update({
        "HOME": str(fake_home), "USERPROFILE": str(fake_home),
        "HF_HOME": str(fake_home / "hf"), "TORCH_HOME": str(fake_home / "torch"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    env.update(extra_env)
    t0 = time.perf_counter()
    try:
        out = subprocess.run([sys.executable, str(CHILD)], env=env, capture_output=True,
                             text=True, timeout=timeout)
        wall = time.perf_counter() - t0
        result = {"rc": out.returncode, "wall_s": round(wall, 2),
                  "stdout_tail": out.stdout.strip().splitlines()[-3:],
                  "stderr_tail": out.stderr.strip().splitlines()[-5:]}
    except subprocess.TimeoutExpired:
        result = {"rc": None, "wall_s": timeout, "timeout": True,
                  "stdout_tail": [], "stderr_tail": ["TIMEOUT"]}
    result["fake_home"] = str(fake_home)
    result["fake_home_listing"] = sorted(os.listdir(fake_home))
    scenarios[name] = result
    print(f"[{name}] rc={result['rc']} wall={result['wall_s']}s "
          f"{result['stdout_tail'][-1] if result['stdout_tail'] else result['stderr_tail'][-1]}",
          flush=True)


# A: 离线 + vendor 完整
run("A_offline_vendor", {"DINOV2_LOCAL_DIR": str(VENDOR)})

# B: 离线 + vendor 空目录 + 空缓存
empty_vendor = Path(tempfile.mkdtemp(prefix="aoi_test05_emptyvendor_"))
run("B_offline_no_weights", {"DINOV2_LOCAL_DIR": str(empty_vendor)})

# C: 离线 + vendor 有 hubconf 缺权重 + 空缓存
partial_vendor = Path(tempfile.mkdtemp(prefix="aoi_test05_partial_"))
shutil.copy2(VENDOR / "hubconf.py", partial_vendor / "hubconf.py")
run("C_offline_missing_weight_file", {"DINOV2_LOCAL_DIR": str(partial_vendor)})

(TASK / "raw" / "scenarios.json").write_text(
    json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: {"rc": v["rc"], "wall_s": v["wall_s"],
                      "last": (v["stdout_tail"] or v["stderr_tail"])[-1]}
                  for k, v in scenarios.items()}, ensure_ascii=False, indent=2))
