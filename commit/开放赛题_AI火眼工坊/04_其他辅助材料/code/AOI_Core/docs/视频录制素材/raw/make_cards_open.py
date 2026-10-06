# -*- coding: utf-8 -*-
"""生成开放赛题版静态卡（1440x900 PNG）。"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = r"d:\CGAIC\AOI_sys\docs\视频录制素材\raw\cards_open"
os.makedirs(OUT, exist_ok=True)
W, H = 1440, 900
FONT = r"C:\Windows\Fonts\msyh.ttc"


def font(size):
    return ImageFont.truetype(FONT, size)


def make_card(name, lines, dur, sub=None):
    img = Image.new("RGB", (W, H), (10, 12, 20))
    d = ImageDraw.Draw(img)
    total_h = 0
    fs = []
    for text, size, gap in lines:
        fs.append((text, font(size), gap))
        total_h += size + gap
    y = (H - total_h) // 2
    for text, f, gap in fs:
        bbox = d.textbbox((0, 0), text, font=f)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((W - tw) // 2 - bbox[0], y - bbox[1]), text, font=f, fill=(245, 246, 250))
        y += f.size + gap
    if sub:
        d.text((W // 2, H - 90), sub, font=font(28), fill=(120, 128, 150), anchor="mm")
    img.save(os.path.join(OUT, f"{name}.png"))
    print(f"{name}.png -> {dur}s")


make_card("title", [
    ("AI 火眼工坊", 88, 30),
    ("持续自学习 AOI 主干 + 传统 CV 双引擎", 44, 15),
    ("PCB 智能质检平台", 44, 60),
], 15, sub="第八届中国研究生人工智能创新大赛 · 开放赛题五")

make_card("pain", [
    ("制造业智能化升级卡在质检换型", 58, 45),
    ("新缺陷样本极少 · 换线即退化 · 算法开发周期长", 40, 70),
    ("AI 质检如何真正上产线？", 40, 0),
], 15)

make_card("metrics", [
    ("关键指标", 64, 40),
    ("公开数据集 AUROC：MVTec 0.9793 · BTAD 0.9609 · MPDD 0.9851", 38, 55),
    ("GPU 模型段均值 102ms · CPU 全链路均值 351ms", 38, 30),
    ("可训练参数 <0.5M · 双引擎可运行可复现", 38, 0),
], 12)
make_card("end", [
    ("AI 火眼工坊", 88, 40),
    ("谢谢观看", 46, 0),
], 8)
print("done")
