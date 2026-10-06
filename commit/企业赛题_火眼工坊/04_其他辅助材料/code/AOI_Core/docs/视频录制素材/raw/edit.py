# -*- coding: utf-8 -*-
"""项目视频剪辑流水线：seg1/stats/pcb + 静态卡 -> final_raw.mp4
统一 1440x900 / 30fps / h264 / 无音轨"""
import os
import subprocess

FF = r"d:\CGAIC\.trae\tools\bin\ffmpeg.cmd"
FP = r"d:\CGAIC\.trae\tools\bin\ffprobe.cmd"
RAW = r"d:\CGAIC\AOI_sys\docs\视频录制素材\raw"
CL = os.path.join(RAW, "clips")
os.makedirs(CL, exist_ok=True)

VF = "scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("FAIL:", os.path.basename(cmd[3]) if len(cmd) > 3 else cmd)
        print(r.stderr[-600:])
    return r


def clip(name, src, ss, dur, speed=None):
    vf = (f"setpts=PTS/{speed}," if speed else "") + VF
    out = os.path.join(CL, name)
    run([FF, "-y", "-ss", str(ss), "-t", str(dur), "-i", src, "-vf", vf, "-r", "30",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", out])


def card(name, dur, outname):
    run([FF, "-y", "-loop", "1", "-i", os.path.join(RAW, "cards", name), "-t", str(dur), "-r", "30",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
         os.path.join(CL, outname)])


# 1. 静态卡
card("title.png", 12, "00_title.mp4")
card("pain.png", 15, "01_pain.mp4")
card("metrics.png", 12, "11_metrics.mp4")
card("end.png", 8, "12_end.mp4")

# 2. 主流程片段（seg1）
clip("02_src.mp4", os.path.join(RAW, "seg1.mp4"), 6, 28, 0.8)
clip("03_wo.mp4", os.path.join(RAW, "seg1.mp4"), 36, 5.5, 0.5)
clip("04_prep.mp4", os.path.join(RAW, "seg1.mp4"), 41.5, 10.5, None)
clip("05_mon.mp4", os.path.join(RAW, "seg1.mp4"), 52, 9, 0.7)
clip("06_fb.mp4", os.path.join(RAW, "seg1.mp4"), 61, 4.3, 0.5)
clip("07_rv.mp4", os.path.join(RAW, "seg1.mp4"), 65.3, 12.7, 0.7)
clip("08_learn.mp4", os.path.join(RAW, "seg1.mp4"), 78, 5, 0.5)
clip("09_req.mp4", os.path.join(RAW, "seg1.mp4"), 83, 3.7, 0.5)

# 3. 统计页 + 双引擎
clip("10_stats.mp4", os.path.join(RAW, "stats.mp4"), 0, 25, None)
clip("10b_pcb.mp4", os.path.join(RAW, "pcb.mp4"), 6, 20, None)

# 4. 拼接（list.txt 用相对文件名，cwd=CL 目录，避免中文路径编码问题）
order = ["00_title", "01_pain", "02_src", "03_wo", "04_prep", "05_mon", "06_fb",
         "07_rv", "08_learn", "09_req", "10_stats", "10b_pcb", "11_metrics", "12_end"]
lst = os.path.join(CL, "list.txt")
with open(lst, "w", encoding="ascii") as f:
    for n in order:
        f.write(f"file '{n}.mp4'\n")
r = subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy",
                    os.path.join(RAW, "final_raw.mp4")],
                   cwd=CL, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print("CONCAT FAIL:")
    print(r.stderr[-600:])

# 5. 校验
r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration,size",
                    "-show_entries", "stream=codec_name,width,height",
                    "-of", "default=noprint_wrappers=1",
                    os.path.join(RAW, "final_raw.mp4")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout)
print(r.stderr[-300:] if r.stderr else "")
