# PCB_APP — PCB 缺陷检测算法验证台

在 `pcb_defect_detector`（7 项元件缺陷算法）之上的 **图形化验证工具**（PySide6 桌面应用）。
导入模板图（金板）+ 待检图（来料）→ 勾选瑕疵类型 → 画 ROI 重点区域 → 一键检测 → 可视化查看/复核/调参，结果自动保存。

## 特性

- **7 项算法数据驱动**：错件 / 缺件 / 移位 / 立碑 / 翻件 / 极反 / 破损，界面由算法元数据自动生成。
- **判据多选投票**（对应可配置引擎开关）：
  - 错件：尺寸 / 颜色(HSV) / 轮廓（三选多）
  - 翻件：NCC正面 / HSV颜色（二选多）
  - 极反：丝印 / 色带 / 灯芯 / 暗标记（四选多，无 OCR）
- **ROI 重点区域**：在模板图上画矩形/圆形框，**自动同位置复制到待检图**；模板与待检的 ROI 同坐标区域作为主要检测区域，降低背景干扰。
  - 例：极反时把框只圈住极性标识，算法据模板/待检该区域的标识位置判断元件方向。
- **一图多 ROI**：每个 ROI 可绑定不同瑕疵；检测编排对相同 (算法, 参数, 区域) 去重、图片仅加载一次，减少多余耗时。
- **实时参数调整**：右侧滑杆/开关透传算法阈值，改完即可重跑找边界。
- **结果自动保存**：每次检测落盘到 `sessions/<时间戳>/`（叠加结果图 + 原图 + `result.json` 参数快照），无需手动点保存。
- **精美深色 UI**：无边框自定义标题栏、卡片式面板、状态色、双图同步缩放平移、缺陷图层显隐。
- **易拓展**：新增算法只需在 `pcb_defect_detector` 注册 + 在 `pcb_app/metadata.py` 追加一条 `DefectMeta`，界面自动出现。

## 安装

```bash
pip install -r requirements.txt
```

依赖：PySide6、opencv-python、numpy、scikit-image（破损 SSIM 可选，缺失自动降级）。

## 运行

```bash
python run.py
```

算法包定位顺序：环境变量 `PCB_DETECTOR_ROOT` → 默认 `../PCB_defect/pcb_defect_detector` → 自动搜索。
如算法包不在默认位置，设置：

```bash
set PCB_DETECTOR_ROOT=D:\path\to\pcb_defect_detector   # 该目录内含 pcb_defect_detector 包
```

## 使用流程

1. 「打开模板图」「打开待检图」（或直接把图片拖到对应画布）。
2. 左侧勾选要检测的瑕疵类型；点击某类型在右侧编辑其判据与阈值。
3. （可选）工具条选「矩形框 / 圆形框」，在**模板图**上框选重点区域；框会自动出现在待检图相同位置。在 ROI 列表里为每个框绑定瑕疵类型。
4. 「运行检测」。缺陷叠加在待检图，明细见底部结果表（点击行可定位），统计与自动保存路径在结果区顶部显示。
5. 调整参数后可反复重跑。

## 目录结构

```
PCB_APP/
├── run.py                     # 启动入口
├── requirements.txt
├── pcb_app/
│   ├── config.py              # 算法包定位 / 会话目录
│   ├── metadata.py            # 算法元数据注册表（数据驱动 UI 的唯一事实来源）
│   ├── core/
│   │   ├── models.py          # ROI / DefectBox / RunSummary
│   │   └── detector_service.py# 封装 PCBDefectDetector，多 ROI 编排 + 自动保存
│   └── ui/
│       ├── theme.py           # 深色主题 QSS
│       ├── widgets.py         # 卡片 / 参数滑杆 / 状态徽标
│       ├── title_bar.py       # 无边框标题栏
│       ├── image_canvas.py    # 画布：缩放平移 + ROI 绘制 + 缺陷叠加
│       ├── defect_panel.py    # 左：瑕疵清单
│       ├── roi_panel.py       # 左下：ROI 列表
│       ├── param_panel.py     # 右：数据驱动参数面板
│       ├── result_panel.py    # 底：结果表 + 统计 + 图层
│       ├── workers.py         # 后台线程（引擎初始化 / 检测）
│       └── main_window.py     # 主窗口装配与流程串联
└── sessions/                  # 检测结果自动保存目录（运行后生成）
```

## 与算法包的对接

- 通过 `pcb_defect_detector.PCBDefectDetector.detect(algorithm, template, test, config, roi)` 调用。
- `roi` 由前端 ROI 转换而来：矩形 `{"shape":"rect","x","y","w","h"}`、圆形 `{"shape":"circle","cx","cy","r"}`；算法包内部裁剪模板/待检同一区域并把缺陷坐标换算回整图。
- 判据开关（`enable_*`）与阈值通过 `config` 透传。

> 注：本项目为配合 ROI 与判据多选，对 `pcb_defect_detector` 做了向后兼容的小增强
> （错件/翻件新增 `enable_*` 判据开关；门面 `detect/detect_all` 新增可选 `roi` 参数）。
> 旧调用行为完全不变。
