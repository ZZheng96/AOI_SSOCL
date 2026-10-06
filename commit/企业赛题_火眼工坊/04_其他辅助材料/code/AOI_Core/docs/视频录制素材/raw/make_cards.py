# -*- coding: utf-8 -*-
"""生成项目视频静态卡（标题/痛点/指标/结束）为 1440x900 PNG。"""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = r"d:\CGAIC\AOI_sys\docs\视频录制素材\raw\cards"
os.makedirs(OUT, exist_ok=True)
W, H = 1440, 900
FONT = r"C:\Windows\Fonts\msyh.ttc"


def font(size):
    return ImageFont.truetype(FONT, size)


def make_card(name, lines, dur):
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
    # 底部小字
    if name == "title":
        d.text((W // 2, H - 90), "第八届中国研究生人工智能创新大赛 · 华为赛题一",
               font=font(28), fill=(120, 128, 150), anchor="mm")
    img.save(os.path.join(OUT, f"{name}.png"))
    print(f"{name}.png -> {dur}s")


# (文案, 字号, 行距)
make_card("title", [
    ("火眼工坊", 96, 30),
    ("可自学习的 AOI 实时在线 AI 质检系统", 46, 60),
], 12)
make_card("pain", [
    ("工业质检三大痛点", 64, 40),
    ("新缺陷样本极少 · 换线即退化 · 算法开发周期长", 40, 80),
    ("本系统以「少样本冷启动 + 用户反馈在线学习」破局", 40, 0),
], 15)
make_card("metrics", [
    ("关键指标", 64, 40),
    ("公开数据集 AUROC：MVTec 0.9793 · BTAD 0.9609 · MPDD 0.9851", 40, 60),
    ("GPU 模型段均值 102ms · CPU 全链路均值 351ms", 40, 30),
    ("可训练参数 <0.5M · 单图双检 <400ms，满足 1s 红线", 34, 0),
], 12)
make_card("end", [
    ("火眼工坊", 96, 40),
    ("谢谢观看", 46, 0),
], 8)
print("done")
