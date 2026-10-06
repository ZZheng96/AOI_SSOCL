# -*- coding: utf-8 -*-
"""开放赛题版字幕烧录：final_open_raw.mp4 -> final_open.mp4。"""
import os
import subprocess

FF = r"d:\CGAIC\.trae\tools\bin\ffmpeg.cmd"
FP = r"d:\CGAIC\.trae\tools\bin\ffprobe.cmd"
RAW = r"d:\CGAIC\AOI_sys\docs\视频录制素材\raw"
SRC = os.path.join(RAW, "final_open_raw.mp4")
OUT = os.path.join(RAW, "final_open.mp4")
SUB = os.path.join(RAW, "subs_open")
os.makedirs(SUB, exist_ok=True)

SUBS = [
    ("数据集适配：只读扫描自动判定格式/品类/分组/计数；树节点标注可支撑层级 L0→L3", 30, 65),
    ("工单 = 数据源 + 数据条件（多品类/含模板）+ 人工复判；L0-L3 按数据条件自动聚合", 65, 76),
    ("少样本冷启动：选场景层引擎按层激活槽位，冻结 DINOv2、可训练参数<0.5M；AUROC 0.98", 76, 86.5),
    ("GPU 模型段 p50 58ms/均值 102ms；CPU 全链路均值 351ms；产线统一单图 1s 红线", 86.5, 99.4),
    ("操作员三类反馈闭环，误检漏检即时回流；版本门控、回滚、人工验收", 99.4, 108),
    ("复核判缺陷即学；L1a 跨域学习 0.60→0.72，L3 模板差分 1.0 仅覆盖金样板条件", 108, 126.1),
    ("学习提升→生成候选版本，可回滚", 126.1, 136.1),
    ("批次回队复检，验证学习提升", 136.1, 143.5),
    ("五类缺陷归因全覆盖；判定可追溯到槽位证据；SPC 良率/缺陷分布", 143.5, 168.5),
    ("制造智能化双引擎：传统 CV 六大算法组 20 项检测 + 持续学习 AI 通用异常；SMT 示例双检约 326ms；SPC/CAD 已实现，MES 为 Mock 接口", 168.5, 204.5),
]

subfiles = []
for i, (txt, a, b) in enumerate(SUBS):
    p = os.path.join(SUB, f"s{i}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    subfiles.append((p, a, b))

filters = []
for p, a, b in subfiles:
    fp = p.replace("\\", "/").replace(":", "\\:")
    filters.append(
        f"drawtext=fontfile='C\\:/Windows/Fonts/msyh.ttc':textfile='{fp}':"
        f"fontsize=28:fontcolor=white:borderw=2:bordercolor=black@0.6:"
        f"x=(w-text_w)/2:y=h-95:enable='between(t,{a},{b})'")
vf = ",".join(filters)

r = subprocess.run([FF, "-y", "-i", SRC, "-vf", vf,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-an", OUT],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print("SUB FAIL" if r.returncode else "SUB OK")
if r.returncode:
    print(r.stderr[-800:])

r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration,size",
                    "-show_entries", "stream=codec_name,width,height",
                    "-of", "default=noprint_wrappers=1", OUT],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout)
