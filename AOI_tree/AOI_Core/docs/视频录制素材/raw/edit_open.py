# -*- coding: utf-8 -*-
"""开放赛题版剪辑：复用主流程 clips + 开放卡片 + 延长 pcb 双引擎分镜。"""
import os
import subprocess

FF = r"d:\CGAIC\.trae\tools\bin\ffmpeg.cmd"
FP = r"d:\CGAIC\.trae\tools\bin\ffprobe.cmd"
RAW = r"d:\CGAIC\AOI_sys\docs\视频录制素材\raw"
CL = os.path.join(RAW, "clips")
CLO = os.path.join(RAW, "clips_open")
os.makedirs(CLO, exist_ok=True)

VF = "scale=1440:900:force_original_aspect_ratio=decrease,pad=1440:900:(ow-iw)/2:(oh-ih)/2"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("FAIL:", cmd[3] if len(cmd) > 3 else cmd)
        print(r.stderr[-500:])
    return r


def card(name, dur, outname):
    run([FF, "-y", "-loop", "1", "-i", os.path.join(RAW, "cards_open", name), "-t", str(dur), "-r", "30",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
         os.path.join(CLO, outname)])


# 1. 开放版卡片
card("title.png", 15, "00_title.mp4")
card("pain.png", 15, "01_pain.mp4")
card("metrics.png", 12, "11_metrics.mp4")
# 结束卡用开放版（AI火眼工坊）
run([FF, "-y", "-loop", "1", "-i", os.path.join(RAW, "cards_open", "end.png"), "-t", "8", "-r", "30",
     "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
     os.path.join(CLO, "12_end.mp4")])

# 2. 复用主流程片段（企业版已生成）
# 3. 统计页 + 双引擎（双引擎延长到 36s，0.75x 慢放）
def clip(name, src, ss, dur, speed=None):
    vf = (f"setpts=PTS/{speed}," if speed else "") + VF
    out = os.path.join(CLO, name)
    run([FF, "-y", "-ss", str(ss), "-t", str(dur), "-i", src, "-vf", vf, "-r", "30",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-an", out])


clip("10_stats.mp4", os.path.join(RAW, "stats.mp4"), 0, 25, None)
clip("10b_pcb.mp4", os.path.join(RAW, "pcb.mp4"), 6, 27, 0.75)

# 4. 拼接（企业版 02~09 片段 + 开放版卡片 + 新统计/双引擎）
order = ["00_title", "01_pain"]
for n in ["02_src", "03_wo", "04_prep", "05_mon", "06_fb", "07_rv", "08_learn", "09_req"]:
    src = os.path.join(CL, n + ".mp4")
    dst = os.path.join(CLO, n + ".mp4")
    if not os.path.exists(dst):
        run(["cmd", "/c", "copy", "/Y", src, dst])
order += ["02_src", "03_wo", "04_prep", "05_mon", "06_fb", "07_rv", "08_learn", "09_req",
          "10_stats", "10b_pcb", "11_metrics", "12_end"]

lst = os.path.join(CLO, "list.txt")
with open(lst, "w", encoding="ascii") as f:
    for n in order:
        f.write(f"file '{n}.mp4'\n")
r = subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy",
                    os.path.join(RAW, "final_open_raw.mp4")],
                   cwd=CLO, capture_output=True, text=True, encoding="utf-8", errors="replace")
if r.returncode != 0:
    print("CONCAT FAIL:")
    print(r.stderr[-600:])

# 5. 校验
r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration,size",
                    "-show_entries", "stream=codec_name,width,height",
                    "-of", "default=noprint_wrappers=1",
                    os.path.join(RAW, "final_open_raw.mp4")],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
print(r.stdout)
