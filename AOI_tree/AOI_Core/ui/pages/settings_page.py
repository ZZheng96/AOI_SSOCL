"""系统设置中心（M16d，review R3）+ 新建品类向导（M16e，review R2）。

设置中心：GET /api/system/config 白名单项表单化展示（按 section 分组，
int/float 用数字框、bool 用勾选框、其余用文本框），保存时只回传改动项；
非白名单高危项（引擎槽位参数/安全密钥）仍走 yaml 手改。

新建品类向导：按操作流"定品类 → 定数据 → 定功能 → 冷启动 → 验收 → 投产"
给出引导清单，每一步一个直达按钮（跳转到对应导航页执行实际操作）。
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ui.api_client import ApiClient
from ui.pages.common import (
    make_banner,
    notify,
    run_async,
    warn,
)

# 配置编辑器宽度（按数据类型给"够用就好"的宽度，2026-10-14 前端反馈）
_W_INT = 120        # 整数
_W_FLOAT = 130      # 小数（如 60.000 告警冷却秒，留出 3 位小数）
_W_TEXT = 360       # 文本（目录/地址等）
_W_LIST = 420       # 列表（逗号分隔，如订阅事件）
_W_CHOICE = 180     # 下拉选择（待决策事项：候选发布/校准隔离/依赖锁等）

# section 分组中文名
_SECTION_CN = {
    "pipeline": "推理管线",
    "data": "数据目录",
    "self_learning": "在线学习",
    "evaluation": "评估",
    "notify": "告警通知",
    "archive": "自动归档",
    "plc": "PLC 触发",
    "logging": "日志",
    "business_budget": "业务预算与保留期",
    "governance_decisions": "待决策事项",
}


_SECTION_HELP = {
    "pipeline": "运行参数：延迟预算只是达标参考，不会加速推理；复判水位控制送检节奏，不改变判定阈值。",
    "data": "界面默认值：用于「数据管理」下次打开标准数据集导入框；不搬迁已导入数据，也不改变模型。",
    "self_learning": "学习运行参数：自动激活可能切换线上模型，并行开关影响资源竞争；保存不会自动发起训练。",
    "evaluation": "评估参数：当前延迟基准使用评估请求中的样本数，本组配置尚未接入该路径。",
    "notify": "外部集成：Webhook 地址/订阅对后续事件读取新配置；告警窗口、触发帧数、冷却在首次使用时缓存，修改后需重启后端。均不改变单帧检测判定。",
    "archive": "数据留存：后续命中决策的检测图复制到归档并登记，供数据管理复盘；不自动入训，也不改变当前检测结果。",
    "plc": "外部连接：后续 PLC 触发接口从此目录取最早图片。仅填目录不会启动目录监听，也不改变普通导入/送检目录。",
    "logging": "运行日志：单文件大小和轮转份数在后端启动时读取，保存后需重启后端；与下方数据库审计日志不是同一保留策略。",
    "business_budget": "业务验收线：影响统计/评估显示与候选验收，不是算法判定阈值，不能靠调宽达标线改善误报或漏检。已有评估记录不自动重算。",
    "governance_decisions": "混合决策项：包含发布门禁、下次建模参数与仅登记事项，保存不代表全部热生效；详见各项影响范围。「恢复默认」不改这些决策。",
}

_ITEM_HELP = {
    "pipeline.latency_budget_ms": "作用于顶栏预算显示、评估延迟达标判断及候选验收参考；调大只放宽预算，不会降低实际耗时或改变缺陷分数。",
    "pipeline.review_backlog_high": "仅人工复判工单：运行管线检查积压达到高水位后自动暂停；下轮检查读取新值。高水位必须大于低水位，否则不启用背压。",
    "pipeline.review_backlog_low": "仅恢复因积压而自动暂停的工单，不解除人工暂停；下轮管线检查读取。调低会等待更多复核完成，不改变检测精度。",
    "data.default_root": "下次打开「数据管理」标准数据集导入框读取；客户端环境变量 AOI_DATA_ROOT 优先，相对路径以项目根解析。仅默认路径，不改变已有图片/特征。",
    "self_learning.trigger_min_feedback": "此项虽可保存，但当前未找到运行时读取路径；不要将其当作自动巩固触发器或可靠提醒开关。",
    "self_learning.auto_activate": "后续巩固在请求未显式指定激活选项时读取；开启可切换线上版本，从而影响检测效果。与「批准后自动激活候选」是不同入口。",
    "self_learning.learning_parallel": "后续学习让行检查读取（引擎显式覆盖时除外）。关闭优先前台推理；开启学习可更快但会争用资源，不保证精度提高。",
    "evaluation.benchmark_n_images": "当前延迟基准接口使用请求 n_images，未读取此配置；改变本项不会改变评估页当前请求的测试张数，也不影响模型精度。",
    "notify.webhook_url": "后续事件推送时读取；空地址禁用推送。只向外部系统发送事件，不控制相机连接或单帧判定。",
    "notify.events": "仅过滤后续 Webhook 推送，不关闭本地检测/反馈/告警。当前空列表会回退到 anomaly 订阅，完全关闭请清空 Webhook 地址。",
    "notify.alarm_window": "按品类累计最近窗口内的异常帧，不要求连续异常；首次创建告警监视器读取，之后修改需重启后端。不改变单帧异常判定。",
    "notify.alarm_k": "窗口内异常帧数达到该值才触发批量告警；调低更容易告警，不等于提高缺陷召回。监视器已创建时需重启后端。",
    "notify.alarm_cooldown_s": "同品类重复告警的最短间隔秒数，不是检测间隔。监视器已创建时需重启后端。",
    "archive.enabled": "后续检测归档时读取；归档会占磁盘并增加文件写入，不会自动更新训练记忆或修正误报。",
    "archive.decisions": "仅匹配检测决策；当前空列表回退为 anomaly，不表示全部关闭，关闭归档请取消「缺陷自动归档」。",
    "plc.trigger_dir": "PLC 触发请求每次读取；目录必须已存在且由后端进程可访问，空值禁用该接口取图。无需重新准备模型。",
    "business_budget.raw_hot_days": "原图热存天数；与冷归档月数（每月按 30 天）相加，供独立保留期预览/审批清理接口计算过期候选。保存不自动删除；本页一键清理不使用此规则。",
    "business_budget.raw_cold_months": "原图冷存月数（每月按 30 天）；与热存天数合计供独立保留期接口使用，并不会自动搬迁到冷存储或自动删除。",
    "business_budget.heatmap_overlay_days": "热图/overlay/mask 保留期供独立保留期预览/审批清理接口使用；只有显式审批执行才删除，保存不会触发清理。",
    "governance_decisions.calibration_isolation": "只影响之后准备模型的校准阶段，已有快照不变，需重新准备模型；auto 不覆盖引擎 yaml 原值。校准样本划分可能影响分数与误报/召回，须重新评估。",
    "governance_decisions.blob_gpu_enabled": "BlobSlot 构造时读取，已有缓存引擎不变：重启后端或重新加载品类引擎后生效，不必仅为设备切换重新训练。AOI_BLOB_GPU=0/false/空值强制关闭；无可用 GPU 也不能加速。切换后需复核精度与耗时。",
    "governance_decisions.approved_extra_good_batches": "仅 DB 取样的下次准备/表示重建纳入 fit 子集，不自动重建已有模型；评估会按当前登记剔除 fit 子集，因此登记不代表模型已经学到这些批次。数据变化后需重新建模并评估。",
}


def _item_help(it: dict) -> str:
    sk = f"{it['section']}.{it['key']}"
    return _ITEM_HELP.get(sk, str(it.get("desc", "")))


def _fmt_val(v) -> str:
    if isinstance(v, bool):
        return "是" if v else "否"
    return "（空）" if v in (None, "") else str(v)


class CategoryWizardDialog(QDialog):
    """新建品类向导（M16e）：把操作流开篇步骤做成引导清单。"""

    goto_page = Signal(str)
    levels_changed = Signal()   # U-workorder：工单提交成功后通知刷新品类数据条件

    _STEPS = [
        ("① 建工单", "填工单名并挂接数据源（数据条件由数据源属性自动聚合）；"
         "数据源在「数据管理」页新建（声明标注档位）并导入，"
         "导入后系统会体检并自动校正档位。", None),
        ("② 定数据", "导入预训练/冷启动数据：正常图（train/normal）≥3 张，"
         "缺陷图（anomaly）随意；缺陷不足可用「数据增强」合成。", "数据管理"),
        ("③ 准备模型", "按工单的数据条件自动选择建模方式；"
         "有模板图的品类先在数据管理设模板图。", "模型管理"),
        ("④ 冷启动训练", "模型管理 -> 准备模型（少样本冷启动，分钟级）。", "模型管理"),
        ("⑤ 评估验收", "评估看板 -> 精度验收（test 集 AUROC），达标再上线。", "评估看板"),
        ("⑥ 投产", "模型管理激活版本 -> 实时监控开始日常检测，"
         "异常帧人工反馈即进入在线学习。", "实时监控"),
    ]

    def __init__(self, client: ApiClient, parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self.setWindowTitle("新建品类上线向导")
        self.resize(560, 560)
        lay = QVBoxLayout(self)

        lay.addWidget(make_banner(
            "按操作流走一遍：建工单 -> 定数据 -> 准备模型 -> 冷启动 -> 验收 -> 投产"))

        form = QFormLayout()
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("工单名，例如：3号线 SMT 质检 / 螺丝-A线")
        form.addRow("工单名", self.edit_name)
        self.chk_review = QCheckBox("人工复判（全部检测进待复核队列，"
                                    "复核后才计入统计与持续学习）")
        self.chk_review.setToolTip("复判开关：检测结果全部需人工复核，"
                                   "复核提交后自动生成反馈回流持续学习")
        form.addRow("复判", self.chk_review)
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("工单备注（可选）")
        form.addRow("备注", self.edit_note)
        lay.addLayout(form)

        self.lbl_wo_state = QLabel("")
        self.lbl_wo_state.setProperty("subtext", True)
        self.lbl_wo_state.setWordWrap(True)
        lay.addWidget(self.lbl_wo_state)

        row_btn = QHBoxLayout()
        self.btn_submit = QPushButton("提交工单")
        self.btn_submit.setProperty("primary", True)
        self.btn_submit.setToolTip("登记工单与数据条件（导入数据后系统自动体检校正），"
                                   "提交后可直接去导数据")
        self.btn_submit.clicked.connect(self._on_submit)
        row_btn.addWidget(self.btn_submit)
        row_btn.addStretch(1)
        lay.addLayout(row_btn)

        for title, desc, page in self._STEPS:
            box = QGroupBox(title)
            b_lay = QVBoxLayout(box)
            d = QLabel(desc)
            d.setWordWrap(True)
            d.setProperty("subtext", True)
            b_lay.addWidget(d)
            if page:
                btn = QPushButton(f"去「{page}」执行")
                btn.setProperty("flat", True)
                btn.clicked.connect(
                    lambda _=False, p=page: self._goto(p))
                b_lay.addWidget(btn, 0, Qt.AlignRight)
            lay.addWidget(box)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _on_submit(self) -> None:
        """提交工单（工单名 + 复判开关；数据条件由数据源聚合）到后端。"""
        cat = self.edit_name.text().strip()
        if not cat:
            self.lbl_wo_state.setText("请先填写工单名")
            return
        self.btn_submit.setEnabled(False)

        def _done(res) -> None:
            self.btn_submit.setEnabled(True)
            if isinstance(res, dict) and res.get("id"):
                # 响应为工单 item，体检结果在嵌套 check 字段内
                check = res.get("check") or {}
                cond_cn = str(res.get("conditions_cn") or "（未挂数据源）")
                extra = ""
                applied = check.get("applied") or {}
                if applied:  # 新建即体检：数据源声明档位按支撑自动校正
                    extra = "；体检已自动校正数据源标注档位"
                warns = [str(w) for w in (check.get("warnings") or [])]
                if warns:
                    extra += f"；提醒：{'；'.join(warns)}"
                self.lbl_wo_state.setText(
                    f"✓ 工单已登记：「{cat}」，数据条件「{cond_cn}」{extra}")
                notify(self, f"工单已登记：{cat}（{cond_cn}）")
                self.levels_changed.emit()
            else:
                self.lbl_wo_state.setText("提交失败，请重试")

        def _fail(msg) -> None:
            self.btn_submit.setEnabled(True)
            if "已存在" in str(msg):
                self.lbl_wo_state.setText(
                    f"工单「{cat}」已存在：可在「工作台」表格中双击改条件")
            else:
                self.lbl_wo_state.setText(f"提交失败：{msg}")

        run_async(self, lambda: self._client.create_workorder(
            cat, review_enabled=self.chk_review.isChecked(),
            note=self.edit_note.text().strip()), _done, _fail)

    def _goto(self, page: str) -> None:
        self.goto_page.emit(page)
        self.accept()


class SettingsPage(QWidget):
    """系统设置中心页。"""

    goto_page = Signal(str)  # 请求主窗口跳转导航页（供向导用）
    levels_changed = Signal()  # U-workorder：工单变更 -> 主窗口刷新品类数据条件
    config_saved = Signal()    # 配置保存成功 -> 主窗口重拉 system_info（预算等热更新）

    def __init__(self, client: ApiClient, parent: QWidget | None = None):
        super().__init__(parent)
        self._client = client
        self._editors: dict[str, QWidget] = {}  # "sec.key" → 编辑控件
        self._defaults: dict[str, object] = {}  # AC-072："sec.key" → 默认值（后端下发）

        # 2026-10-14（前端反馈）：改为**整页滚动**——配置项、系统自检、操作日志
        # 统一放进一个页面级 QScrollArea。此前只有配置表单套在小滚动窗口里，
        # 页面又高，导致要在小窗口内反复滚动、配置项挤在一起。
        page_host = QWidget()
        outer = QVBoxLayout(page_host)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        top = QHBoxLayout()
        # 显式换行，给右侧操作按钮留出空间。
        _set_banner = make_banner(
            "保存写回配置，不等于全部热生效；生效条件见各组说明及悬停提示。\n"
            "AOI_CONFIG 未指定时用项目配置；引擎内部参数/密钥仍需手改。")
        top.addWidget(_set_banner)
        top.addStretch(1)
        btn_wizard = QPushButton("新建品类向导")
        btn_wizard.setProperty("primary", True)
        btn_wizard.setToolTip("按操作流引导完成新品类上线（定数据/定功能/冷启动/验收）")
        btn_wizard.clicked.connect(self._open_wizard)
        top.addWidget(btn_wizard)
        btn_refresh = QPushButton("刷新")
        btn_refresh.clicked.connect(self.reload)
        top.addWidget(btn_refresh)
        self.btn_restore = QPushButton("恢复默认")
        self.btn_restore.setToolTip(
            "把「业务预算与保留期」各项填回默认值（需再点「保存修改」才写回）")
        self.btn_restore.clicked.connect(self.restore_defaults)
        top.addWidget(self.btn_restore)
        self.btn_save = QPushButton("保存修改")
        self.btn_save.setProperty("success", True)
        self.btn_save.clicked.connect(self._on_save)
        top.addWidget(self.btn_save)
        outer.addLayout(top)

        self.lbl_path = QLabel("配置文件：-")
        self.lbl_path.setProperty("subtext", True)
        self.lbl_path.setToolTip(
            "配置路径由后端决定：优先读取环境变量 AOI_CONFIG；"
            "未设置时使用项目 configs/default.yaml。修改路径后需重启后端。")
        outer.addWidget(self.lbl_path)

        # 配置项容器（_fill 只重填这里；不再自带滚动窗口）
        self._form_host = QWidget()
        self._form_lay = QVBoxLayout(self._form_host)
        self._form_lay.setContentsMargins(0, 0, 0, 0)
        self._form_lay.setSpacing(8)
        outer.addWidget(self._form_host)

        # 系统自检（2026-10-09）：数据残留/引用悬空/页面一致性体检
        outer.addWidget(self._build_selfcheck_section())

        # 操作日志（审计）：自统计报表页迁入——报表页面向生产口径，
        # 审计功能归入设置中心
        outer.addWidget(self._build_logs_section())
        outer.addStretch(1)

        page_scroll = QScrollArea()
        page_scroll.setWidgetResizable(True)
        # 2026-10-14（前端反馈）：本页**只允许纵向滚动**——顶部横幅已在标点处手工断行，
        # 内容宽度按默认窗口算得下；关掉横向滚动条可确保右上角按钮永远不需要横向滚动
        # 才能看到（内容宽度富余，不会裁掉任何控件）。
        page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page_scroll.setWidget(page_host)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addWidget(page_scroll)

        self._client.error_occurred.connect(lambda m: warn(self, m))

    # ══════════════════ 表单构建 ══════════════════
    def reload(self) -> None:
        run_async(self, self._client.get_config, self._fill)
        self.reload_logs()
        if hasattr(self, "table_check"):
            self.run_selfcheck()

    def _fill(self, data) -> None:
        if not data:
            return
        self.lbl_path.setText(f"配置文件：{data.get('config_path', '-')}")
        # 清空旧表单
        while self._form_lay.count():
            item = self._form_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._editors.clear()
        self._defaults.clear()
        fixed = data.get("fixed") or {}

        groups: dict[str, list] = {}
        for it in data.get("items") or []:
            groups.setdefault(str(it.get("section", "")), []).append(it)
        for sec, items in groups.items():
            box = QGroupBox(_SECTION_CN.get(sec, sec))
            form = QFormLayout(box)
            gov = sec == "governance_decisions"
            if sec in _SECTION_HELP:
                head = QLabel(_SECTION_HELP[sec])
                head.setProperty("subtext", True)
                head.setWordWrap(True)
                form.addRow(head)
            for it in items:
                sk = f"{it['section']}.{it['key']}"
                lb = QLabel(str(it.get("name", it["key"])))
                help_text = _item_help(it)
                lb.setToolTip(help_text)
                if it.get("readonly"):
                    # 只读项（如冻结时间）：服务端维护，不回传
                    ro = QLabel(_fmt_val(it.get("value")))
                    ro.setToolTip(help_text)
                    ro.setObjectName(f"ro:{sk}")
                    form.addRow(lb, ro)
                else:
                    editor = self._make_editor(it)
                    editor.setToolTip(help_text + (
                        "（多项用英文逗号分隔）" if it.get("type") == "list" else ""))
                    self._editors[sk] = editor
                    if "default" in it and not gov:
                        self._defaults[sk] = it["default"]
                    if it.get("locked"):
                        editor.setEnabled(False)
                        editor.setToolTip("GT 与阈值已冻结：需先在「待决策事项」"
                                          "关闭冻结并保存后才能修改")
                    form.addRow(lb, editor)
                if gov:
                    sub = QLabel(
                        f"说明：{help_text}\n"
                        f"当前值：{_fmt_val(it.get('value'))}　"
                        f"默认：{_fmt_val(it.get('default'))}\n"
                        f"影响范围：{it.get('impact', '-')}")
                    sub.setProperty("subtext", True)
                    sub.setWordWrap(True)
                    sub.setObjectName(f"sub:{sk}")
                    form.addRow(sub)
            if sec == "business_budget":
                # AC-072：固定口径项只读展示 + 保留期边界说明
                wal = fixed.get("business_budget.feedback_event_wal_retention")
                if wal:
                    form.addRow(QLabel("反馈/事件/WAL 保留"), QLabel(str(wal)))
                note = QLabel("保留期不会自动删除数据：原图/热图规则供独立的保留期预览与审批执行接口使用，"
                              "与下方一键清理不同；快照/审计保留及标签迟到容忍当前仅登记，"
                              "迟到容忍尚未接入入训路径。")
                note.setProperty("subtext", True)
                note.setWordWrap(True)
                form.addRow(note)
            self._form_lay.addWidget(box)
        self._form_lay.addStretch(1)

    def _make_editor(self, it: dict) -> QWidget:
        typ = str(it.get("type", "str"))
        val = it.get("value")
        if typ == "bool":
            w = QCheckBox("启用")
            w.setChecked(bool(val))
            w.setToolTip(str(it.get("desc", "")))
            return w
        if typ == "choice":
            w = QComboBox()
            for opt in it.get("options") or []:
                w.addItem(str(opt[1]), opt[0])
            idx = w.findData(val)
            if idx >= 0:
                w.setCurrentIndex(idx)
            w.setToolTip(str(it.get("desc", "")))
            # 2026-10-14：下拉同样按"够用就好"定宽（此前漏改，候选发布/校准隔离/
            # 依赖锁三项会一路顶到页面右侧）
            w.setFixedWidth(_W_CHOICE)
            return w
        # AC-072：后端下发 min/max 时编辑器限定范围（越界输入在控件层即被夹住）
        # 2026-10-14（前端反馈）：编辑器按数据类型给"够用就好"的宽度，
        # 不再顶到页面右侧（QFormLayout 默认会把非固定控件拉满整行）。
        lo, hi = it.get("min"), it.get("max")
        if typ == "int":
            w = QSpinBox()
            w.setRange(int(lo) if lo is not None else -1_000_000,
                       int(hi) if hi is not None else 1_000_000)
            w.setValue(int(val or 0))
            w.setFixedWidth(_W_INT)
            return w
        if typ == "float":
            w = QDoubleSpinBox()
            w.setDecimals(3)
            w.setRange(float(lo) if lo is not None else -1e9,
                       float(hi) if hi is not None else 1e9)
            w.setValue(float(val or 0.0))
            w.setFixedWidth(_W_FLOAT)
            return w
        if typ == "list":
            # list 类型：逗号分隔编辑，保存时还原为真正的列表。
            # 此前落到 QLineEdit 纯文本，把 Python list 的字符串表示写回
            # yaml（events/decisions 被污染成字符串，前端反馈 v7-6）。
            if isinstance(val, (list, tuple)):
                text = ", ".join(str(x) for x in val)
            else:
                text = "" if val is None else str(val)
            w = QLineEdit(text)
            w.setFixedWidth(_W_LIST)
            w.setProperty("ctype", "list")
            w.setToolTip(str(it.get("desc", "")) + "（多项用英文逗号分隔）")
            return w
        w = QLineEdit("" if val is None else str(val))
        w.setFixedWidth(_W_TEXT)
        return w

    def _editor_value(self, editor: QWidget):
        if isinstance(editor, QCheckBox):
            return editor.isChecked()
        if isinstance(editor, QComboBox):
            return editor.currentData()
        if isinstance(editor, (QSpinBox, QDoubleSpinBox)):
            return editor.value()
        if editor.property("ctype") == "list":
            return [x.strip() for x in editor.text().split(",") if x.strip()]
        return editor.text()

    # ══════════════════ 保存 ══════════════════
    def _on_save(self) -> None:
        updates = {k: self._editor_value(w) for k, w in self._editors.items()}
        run_async(self, lambda: self._client.save_config(updates),
                  self._on_saved)

    def _on_saved(self, res) -> None:
        if not res:
            return
        applied = res.get("applied") or []
        rejected = res.get("rejected") or []
        errors = res.get("errors") or {}
        msg = (f"已保存 {len(applied)} 项配置（不代表全部已热生效）。"
               "\n告警参数/运行日志需重启后端；校准/训练数据变更需重新准备模型；"
               "blob GPU 需重新加载品类引擎，详见对应说明。")
        if rejected:
            msg += f"\n被拒绝（非白名单/权限不足/取值非法）：{', '.join(rejected)}"
            for k in rejected:
                if k in errors:
                    msg += f"\n  {errors[k]}"
        notify(self, msg)
        if applied:
            self.config_saved.emit()
        if "governance_decisions.gt_threshold_frozen_at" in applied:
            # 冻结状态变化 → 重拉表单刷新锁定态与冻结时间
            run_async(self, self._client.get_config, self._fill)

    def restore_defaults(self) -> None:
        """AC-072：把带默认值的项（业务预算与保留期）填回默认，待用户保存。"""
        for sk, default in self._defaults.items():
            w = self._editors.get(sk)
            if w is not None and not w.isEnabled():
                continue  # 冻结锁定项不填回
            if isinstance(w, QCheckBox):
                w.setChecked(bool(default))
            elif isinstance(w, QSpinBox):
                w.setValue(int(default))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(default))
            elif isinstance(w, QLineEdit):
                w.setText(str(default))
        if self._defaults:
            notify(self, f"已填回 {len(self._defaults)} 项默认值，"
                         "点「保存修改」后生效")

    # ══════════════════ 系统自检 ══════════════════
    _STATUS_CN = {"ok": "正常", "warn": "警告", "fail": "严重"}

    def _build_selfcheck_section(self) -> QGroupBox:
        box = QGroupBox("系统自检")
        lay = QVBoxLayout(box)
        bar = QHBoxLayout()
        self.lbl_selfcheck = QLabel(
            "检查数据残留、引用悬空与页面一致性；发现问题可一键清理")
        self.lbl_selfcheck.setProperty("subtext", True)
        bar.addWidget(self.lbl_selfcheck, 1)
        self.btn_selfcheck = QPushButton("运行自检")
        self.btn_selfcheck.setProperty("primary", True)
        self.btn_selfcheck.setToolTip(
            "只读体检：孤儿检测记录、悬空反馈、图片文件缺失、"
            "认领卡死、空数据源、统计聚合偏差等")
        self.btn_selfcheck.clicked.connect(self.run_selfcheck)
        bar.addWidget(self.btn_selfcheck)
        self.btn_cleanup = QPushButton("一键清理")
        self.btn_cleanup.setProperty("success", True)
        self.btn_cleanup.setEnabled(False)
        self.btn_cleanup.setToolTip(
            "修复所有「可修复」项：级联删除孤儿/悬空检测与反馈、"
            "删除裂图登记行、释放卡死认领、重算今日统计")
        self.btn_cleanup.clicked.connect(self._on_cleanup)
        bar.addWidget(self.btn_cleanup)
        lay.addLayout(bar)
        self.table_check = QTableWidget(0, 4)
        self.table_check.setHorizontalHeaderLabels(
            ["检查项", "状态", "数量", "说明"])
        self.table_check.horizontalHeader().setStretchLastSection(True)
        self.table_check.verticalHeader().setVisible(False)
        self.table_check.setEditTriggers(QTableWidget.NoEditTriggers)
        # 整页滚动后表格用固定高度（不随页面拉伸），内容超出时表格内滚动
        self.table_check.setFixedHeight(190)
        self.table_check.setMaximumHeight(220)
        lay.addWidget(self.table_check)
        return box

    def run_selfcheck(self) -> None:
        self.btn_selfcheck.setEnabled(False)
        self.lbl_selfcheck.setText("自检中……")

        def _done(res) -> None:
            self.btn_selfcheck.setEnabled(True)
            if not res:
                self.lbl_selfcheck.setText("自检失败，请重试")
                return
            self._fill_selfcheck(res)

        def _fail(msg) -> None:
            self.btn_selfcheck.setEnabled(True)
            self.lbl_selfcheck.setText(f"自检失败：{msg}")

        run_async(self, self._client.system_selfcheck, _done, _fail)

    def _fill_selfcheck(self, res: dict) -> None:
        items = res.get("items") or []
        n_bad = int(res.get("n_issues") or 0)
        n_fixable = sum(1 for i in items if i.get("fixable"))
        self.lbl_selfcheck.setText(
            "全部正常，未发现数据残留或不一致" if n_bad == 0 else
            f"发现 {n_bad} 类问题（其中 {n_fixable} 类可一键清理）")
        self.table_check.setRowCount(len(items))
        for r, it in enumerate(items):
            status = str(it.get("status", "ok"))
            vals = [str(it.get("name", "")),
                    self._STATUS_CN.get(status, status),
                    str(it.get("count", 0)),
                    str(it.get("detail", ""))]
            for c, v in enumerate(vals):
                cell = QTableWidgetItem(v)
                if c == 1 and status != "ok":
                    cell.setForeground(Qt.red if status == "fail"
                                       else Qt.darkYellow)
                if c == 3:
                    cell.setToolTip(v)
                self.table_check.setItem(r, c, cell)
        self.btn_cleanup.setEnabled(n_fixable > 0)

    def _on_cleanup(self) -> None:
        ret = QMessageBox.question(
            self, "一键清理",
            "这不是清理缓存或默认特征：将级联删除孤儿/悬空的检测与反馈记录、"
            "删除无数据源归属的批次及其图片（仅删应用生成的文件，外部原始数据只删登记行）、"
            "删除文件已缺失的图片登记、释放卡死的图片认领，并重算今日统计。"
            "\n该操作不可恢复，确定继续？")
        if ret != QMessageBox.Yes:
            return
        self.btn_cleanup.setEnabled(False)

        def _done(res) -> None:
            if not res:
                warn(self, "清理失败，请重试")
                return
            fixed = res.get("fixed") or {}
            total = sum(int(v) for v in fixed.values())
            notify(self, f"清理完成，共处理 {total} 条记录")
            self.levels_changed.emit()
            self.run_selfcheck()  # 清理后复检，结果落表

        run_async(self, self._client.system_selfcheck_cleanup, _done)

    # ══════════════════ 操作日志（审计） ══════════════════
    _LOG_ACTIONS = [
        "feedback", "self_update", "self_update_consolidate",
        "activate_model", "model_prepare", "accuracy_eval",
        "learning_curve", "import_images", "create_workorder",
        "review_submit", "config_update", "alarm",
    ]

    def _build_logs_section(self) -> QGroupBox:
        box = QGroupBox("操作日志（审计）")
        lay = QVBoxLayout(box)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("动作过滤"))
        self.combo_log_action = QComboBox()
        self.combo_log_action.setEditable(True)
        self.combo_log_action.addItem("全部", "")
        for a in self._LOG_ACTIONS:
            self.combo_log_action.addItem(a, a)
        self.combo_log_action.currentIndexChanged.connect(
            lambda _i: self.reload_logs())
        bar.addWidget(self.combo_log_action)
        btn_refresh = QPushButton("刷新日志")
        btn_refresh.clicked.connect(self.reload_logs)
        bar.addWidget(btn_refresh)
        btn_export = QPushButton("导出 CSV")
        btn_export.setToolTip("导出审计日志（时间/用户/动作/详情，CSV 带 BOM）")
        btn_export.clicked.connect(self._export_logs)
        bar.addWidget(btn_export)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.table_logs = QTableWidget(0, 4)
        self.table_logs.setHorizontalHeaderLabels(["时间", "用户", "动作", "详情"])
        self.table_logs.horizontalHeader().setStretchLastSection(True)
        self.table_logs.verticalHeader().setVisible(False)
        self.table_logs.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_logs.setFixedHeight(230)
        self.table_logs.setMaximumHeight(260)
        lay.addWidget(self.table_logs)
        return box

    def _log_action_filter(self) -> str:
        """动作过滤：可编辑框，优先取用户手输文本，否则取下拉数据。"""
        text = self.combo_log_action.currentText().strip()
        data = str(self.combo_log_action.currentData() or "")
        return text if text and text != "全部" else data

    def reload_logs(self) -> None:
        if not hasattr(self, "table_logs"):
            return
        action = self._log_action_filter()
        run_async(self, lambda: self._client.list_logs(limit=200, action=action),
                  self._fill_logs)

    def _fill_logs(self, rows) -> None:
        if rows is None:
            return
        self.table_logs.setRowCount(len(rows))
        for r, it in enumerate(rows):
            vals = [str(it.get("created_at") or "-"),
                    str(it.get("user") or "-"),
                    str(it.get("action") or "-"),
                    str(it.get("detail") or "-")]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c == 3:
                    item.setToolTip(v)
                self.table_logs.setItem(r, c, item)

    def _export_logs(self) -> None:
        import datetime
        default = f"aoi_audit_{datetime.date.today().isoformat()}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出审计日志", default, "CSV 文件 (*.csv)")
        if not path:
            return
        action = self._log_action_filter()
        url = f"{self._client.base_url}/api/logs/export?limit=2000"
        if action:
            from urllib.parse import quote
            url += f"&action={quote(action)}"

        def _done(data) -> None:
            if not data:
                warn(self, "导出失败：未获取到数据")
                return
            with open(path, "wb") as f:
                f.write(data)
            notify(self, f"审计日志已导出：{path}")

        run_async(self, lambda: self._client.fetch_bytes(url, timeout=60), _done)

    # ══════════════════ 新建品类向导 ══════════════════
    def _open_wizard(self) -> None:
        dlg = CategoryWizardDialog(self._client, self)
        dlg.goto_page.connect(self.goto_page.emit)
        dlg.levels_changed.connect(self.levels_changed.emit)
        dlg.exec()
