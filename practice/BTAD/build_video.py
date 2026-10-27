# -*- coding: utf-8 -*-
"""AOI_Core 实操视频剪辑成片：裁引导(ss=2) + 逐 clip 烧各自字幕 + concat。

与 AOI_sys build_video.py 相同的原理：每个 clip 只烧它自己那一条字幕
（无全局时间轴），字幕时长 = clip 时长，拼接后天然对齐。
BTAD 版全部为实操分镜（无片头片尾静态卡）。

用法：
    python build_video.py <本次素材目录> <尚不存在的新成片路径>
输入：raw/clips/<时间戳>/<shot>.mp4（由 AOI_Core/tests/_record_shots_btad.py 录制）
输出：practice/BTAD/AOICore_BTAD_全流程_v1.mp4
"""
import os
import subprocess
import sys
import json
import tempfile

FF = (r"C:\Users\imia\AppData\Local\Microsoft\WinGet\Packages"
      r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
      r"\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe")
FP = FF.replace("ffmpeg.exe", "ffprobe.exe")

BASE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(BASE, "raw", "clips")
CACHE = tempfile.mkdtemp(prefix="AOI_MPDD_edit_")
SUBS = os.path.join(CACHE, "subs")
SEGS = os.path.join(CACHE, "segs")
os.makedirs(SUBS, exist_ok=True)
os.makedirs(SEGS, exist_ok=True)

VF_FIT = ("scale=1440:900:force_original_aspect_ratio=decrease,"
          "pad=1440:900:(ow-iw)/2:(oh-ih)/2")

OUT = (os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else
       os.path.join(BASE, "AOICore_BTAD_全流程_v1.mp4"))

DATASET_LABEL = os.environ.get("AOI_DEMO_DATASET", "btad").strip().lower()
IS_MPDD = DATASET_LABEL == "mpdd"

# ── 分镜表：(clip 名, 正片时长, 字幕) ──（与录制脚本_BTAD.md 分镜总表一致）
SHOTS = [
    ("01_new_source", 33,
     "新建数据源：声明 L1b 缺陷位置标注 + 已分品类，智能识别适配演示数据根目录"),
    ("02_probe_confirm", 20,
     "只读扫描自动识别：核对 tubes 品类与正常/缺陷样本，确认后一键导入"),
    ("03_imported", 12,
     "导入完成自动分组：展示 tubes 品类、训练/测试样本与检测批次"),
    ("03b_augment", 24,
     "数据增强：真实基底预览、预处理预览，提交有界三维分析"),
    ("03c_space_done", 20,
     "传统视觉特征空间：旋转视角、点选真实样本；非生产深度特征或效果评估"),
    ("04_workorder", 18,
     "新建工单挂接数据源：后端自动体检校正数据条件，产线默认暂停待验收"),
    ("05_prepare", 35,
     "准备模型：数据体检→特征提取→槽位拟合→校准打分→融合权重，分阶段可见"),
    ("06_prepared", 10,
     "准备任务完成：查看品类 02 的实际模型状态与版本记录"),
    ("06b_feature", 14,
     "特征工程：选择品类 02，提交小样本特征贡献档案任务"),
    ("06c_feature_done", 26,
     "特征贡献：本次test小样本槽位AUROC与留一消融；非逐图解释"),
    ("07_eval", 25,
     "评估看板运行精度评估：在 test 域图上实测检出率 / 误检率 / AUROC"),
    ("08_eval_done", 12,
     "小样本评估完成：刷新并查看真实记录，不外推全量检测效果"),
    ("09_monitor", 25,
     "恢复产线开检：综合异常分大圆表 + 热力图 + 各项评分依据实时输出"),
    ("10_feedback", 15,
     "真实误检反馈：仅选择预测异常且标签正常的记录，不虚构缺陷类型"),
    ("11_review", 12,
     "标注反馈：选择本次提交的真实反馈记录，核对留痕"),
    ("12_learn", 25,
     "暂停产线并固化现有学习快照，候选不等于已激活生产版本"),
    ("12b_candidate", 18,
     "候选构建：真实排队反馈绑定当前父版本，仅生成未激活候选"),
    ("12c_acceptance_status", 18,
     "独立验收入口与候选状态：完整政策未获确认，不提交验收、不批准激活"),
    ("13_learning", 20,
     "真实反馈重打分与权重历史：空轨迹如实显示，当前快照不等于演化"),
    ("14_stats", 24,
     "工单统计：查看真实批次、趋势、缺陷帕累托与延迟图，空值保留 NA"),
    ("14b_settings", 16,
     "系统设置：临时切换后恢复原值，不保存配置；查看反馈操作日志"),
    ("15_requeue", 10,
     "仅选择品类 02 的真实批次回队，不替代为其他品类"),
]


def sub_file(name, text):
    p = os.path.join(SUBS, f"{name}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p.replace("\\", "/").replace(":", "\\:")


def drawtext(name, text):
    fp = sub_file(name, text)
    return (f"drawtext=fontfile='C\\:/Windows/Fonts/msyh.ttc':textfile='{fp}':"
            f"fontsize=30:fontcolor=black:borderw=2:bordercolor=white@0.7:"
            f"x=(w-text_w)/2:y=h-95")


def run(cmd, tag, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", cwd=cwd)
    if r.returncode != 0:
        print(f"FAIL {tag}:")
        print(r.stderr[-600:])
        sys.exit(1)
    return r


def build():
    if os.path.exists(OUT):
        raise FileExistsError(f"成片已存在，拒绝覆盖；请传入新的输出路径：{OUT}")
    with open(os.path.join(CLIPS, "manifest.json"), encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not manifest.get("complete") or manifest.get("issues"):
        raise RuntimeError("本次22镜未全部成功，拒绝合成失败素材")
    entries = {s["name"]: s for s in manifest["shots"]}
    if set(entries) != {s[0] for s in SHOTS}:
        raise RuntimeError("22镜覆盖不完整")
    total = 59.0
    print(f"共 {len(SHOTS)} 段，目标总时长 {total}s，TEMP={CACHE}", flush=True)

    outs = []
    for name, dur, sub in SHOTS:
        out = os.path.join(SEGS, f"{name}.mp4")
        vf = drawtext(name, sub)
        srcmp4 = os.path.join(CLIPS, f"{name}.mp4")
        if not os.path.exists(srcmp4):
            print(f"缺分镜素材 {srcmp4}")
            sys.exit(1)
        entry = entries[name]
        span = float(entry["duration"]) - 2.0
        target = 6.5 if name == "01_new_source" else 2.5
        if name == "01_new_source":
            # 正片前3秒保持原速，后面的真实操作合理加速；不加速工作台停留。
            factor = (span - 3) / (target - 3)
            timeline = f"setpts='if(lt(T,3),PTS,3/TB+(PTS-3/TB)/{factor})'"
        else:
            timeline = f"setpts=PTS/{span/target}"
        subtitle = sub.replace("品类 02", "品类 tubes") if IS_MPDD else sub
        vf = drawtext(name, subtitle)
        run([FF, "-n", "-ss", "2", "-t", str(span), "-i", srcmp4,
             "-vf", f"{timeline},{VF_FIT},fps=30,{vf}", "-t", str(target),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-pix_fmt", "yuv420p", "-an", out], name)
        outs.append(out)
        print(f"  {name}: 原始{span:.2f}s -> {target}s，页面={entry['page']} ✅", flush=True)

    # 拼接（list.txt 相对文件名 + cwd=SEGS，避免中文路径编码问题）
    lst = os.path.join(SEGS, "list.txt")
    with open(lst, "w", encoding="ascii") as f:
        for o in outs:
            f.write(f"file '{os.path.basename(o)}'\n")
    run([FF, "-n", "-f", "concat", "-safe", "0", "-i",
         os.path.basename(lst), "-c", "copy", OUT], "concat", cwd=SEGS)

    # 校验
    r = subprocess.run([FP, "-v", "error", "-show_entries", "format=duration,size",
                        "-show_entries", "stream=codec_name,width,height",
                        "-of", "default=noprint_wrappers=1", OUT],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout)
    d = float([l.split("=")[1] for l in r.stdout.splitlines()
               if l.startswith("duration=")][0])
    ok = 58.5 <= d <= 60.0
    if not ok:
        raise RuntimeError(f"成片时长未满足<=60秒：{d}")
    print(f"总时长 {d:.1f}s（期望 {total}s）{'✅' if ok else '❌'}")
    print(f"成片：{OUT}")


if __name__ == "__main__":
    build()
