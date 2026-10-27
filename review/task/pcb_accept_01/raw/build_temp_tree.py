"""PCB-ACCEPT-01 发布链路：构建 TEMP 隔离 PCB_Dual 树（代码复制、数据排除、模板库全新）。"""
import json, os, shutil
from pathlib import Path

SRC = Path(r"d:\CGAIC\AOI_SSOCL\stable_AOI_tree\PCB_Dual")
DST = Path(r"d:\CGAIC\AOI_SSOCL\review\task\pcb_accept_01\temp_tree\PCB_Dual")
EXCLUDE_DIRS = {"__pycache__", ".git", "test_images", "input", "output", "cache"}

if DST.exists():
    shutil.rmtree(DST)
DST.mkdir(parents=True)


def copy_tree(src, dst):
    for dp, dn, fn in os.walk(src):
        dn[:] = [d for d in dn if d not in EXCLUDE_DIRS]
        rel = Path(dp).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in fn:
            if f.endswith(".pyc"):
                continue
            shutil.copy2(Path(dp) / f, dst / rel / f)


for d in ["app", "configs", "tools", "alg_repo"]:
    copy_tree(SRC / d, DST / d)
(DST / "templates").mkdir()
(DST / "storage").mkdir()

# draft 模板复制进 TEMP 库（不动原库）
tpl_src = SRC / "templates" / "PixPin_2026-07-01_11-21-15_OK"
tpl_dst = DST / "templates" / tpl_src.name
shutil.copytree(tpl_src, tpl_dst)

# TEMP 配置：端口 8023、AOI_Core 绝对路径（沿用稳定版，只读使用其引擎服务）
cfg = (DST / "configs" / "default.yaml").read_text(encoding="utf-8")
cfg = cfg.replace('port: 8022', 'port: 8023').replace('url: "sqlite:///storage/aoi.db"',
                                                      'url: "sqlite:///storage/aoi.db"')
cfg = cfg.replace('aoi_core_dir: "../AOI_Core"',
                  'aoi_core_dir: "D:/CGAIC/AOI_SSOCL/stable_AOI_tree/AOI_Core"')
cfg = cfg.replace("autostart: true", "autostart: false")  # 不自动拉起 AOI_Core，避免端口/资源冲突
(DST / "configs" / "default.yaml").write_text(cfg, encoding="utf-8")

print("TEMP tree:", DST)
print("templates:", [p.name for p in (DST / "templates").iterdir()])
print("size MiB:", round(sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 2**20, 1))
