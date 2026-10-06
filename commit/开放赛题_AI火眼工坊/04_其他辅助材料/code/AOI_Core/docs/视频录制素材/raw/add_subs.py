# -*- coding: utf-8 -*-
"""给 final_raw.mp4 烧录分镜字幕 -> final.mp4；并抽查画面亮度。"""
import os
import subprocess

FF = r"d:\CGAIC\.trae\tools\bin\ffmpeg.cmd"
FP = r"d:\CGAIC\.trae\tools\bin\ffprobe.cmd"
RAW = r"d:\CGAIC\AOI_sys\docs\视频录制素材\raw"
SRC = os.path.join(RAW, "final_raw.mp4")
OUT = os.path.join(RAW, "final.mp4")
SUB = os.path.join(RAW, "subs")
os.makedirs(SUB, exist_ok=True)

# (文案, 起, 止)
SUBS = [
    ("数据集适配：自动识别 MVTec/BTAD/MPDD，100 正常 + 30 缺陷即满足赛题协议", 30, 62),
    ("工单 = 数据源 + 数据条件 + 人工复判，一站配齐", 62, 73),
    ("少样本冷启动：冻结 DINOv2 主干，可训练参数 <0.5M", 73, 83.5),
    ("GPU 模型段均值 102ms（2500x2500）；判定→定位→归因→追溯四层输出", 83.5, 96.4),
    ("操作员三类反馈闭环：判对 / 判错 / 无反馈，误检漏检即时回流", 96.4, 105),
    ("复核判缺陷即学；标错可作废撤回", 105, 123.1),
    ("学习生成新版本：跨域 AUROC 0.60→0.72，版本可回滚", 123.1, 133.1),
    ("错检批次回队重检：原错检图判对", 133.1, 140.5),
    ("五类缺陷归因全覆盖；每条判定可追溯到槽位证据", 140.5, 165.5),
    ("同一 PCB 图传统 CV + AI 双检并行：六大算法组 20 项检测融合", 165.5, 185.5),
]

# 写字幕文件（UTF-8）
subfiles = []
for i, (txt, a, b) in enumerate(SUBS):
    p = os.path.join(SUB, f"s{i}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    subfiles.append((p, a, b))

# drawtext 链
filters = []
for p, a, b in subfiles:
    fp = p.replace("\\", "/").replace(":", "\\:")
    filters.append(
        f"drawtext=fontfile='C\\:/Windows/Fonts/msyh.ttc':textfile='{fp}':"
        f"fontsize=30:fontcolor=white:borderw=2:bordercolor=black@0.6:"
        f"x=(w-text_w)/2:y=h-95:enable='between(t,{a},{b})'")
vf = ",".join(filters)

r = subprocess.run([FF, "-y", "-i", SRC, "-vf", vf,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-an", OUT],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print("SUB FAIL" if r.returncode else "SUB OK")
if r.returncode:
    print(r.stderr[-800:])

# 校验
r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration,size",
                    "-show_entries", "stream=codec_name,width,height",
                    "-of", "default=noprint_wrappers=1", OUT],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout)

# 抽查帧亮度（非黑帧）
import glob
import statistics
from PIL import Image
frames = os.path.join(RAW, "check_frames")
os.makedirs(frames, exist_ok=True)
for t in (10, 40, 80, 120, 160, 200):
    subprocess.run([FF, "-y", "-ss", str(t), "-i", OUT, "-frames:v", "1",
                    os.path.join(frames, f"f{t}.png")],
                   capture_output=True)
for f in sorted(glob.glob(os.path.join(frames, "*.png"))):
    im = Image.open(f).convert("L")
    d = list(im.getdata())
    print(os.path.basename(f), "mean=%.1f" % (sum(d) / len(d)))
