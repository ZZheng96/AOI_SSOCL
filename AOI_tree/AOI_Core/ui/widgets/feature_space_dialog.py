"""Traditional-feature inspection dialog with asynchronous task polling."""
from __future__ import annotations

import base64

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QApplication, QDialog, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QSpinBox, QVBoxLayout)

from ui.pages.common import run_async


class FeatureSpaceDialog(QDialog):
    def __init__(self, client, scope, pseudo, preprocess, parent=None):
        super().__init__(parent)
        # Deliberately lazy: missing plotting dependencies are reported by the caller.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        self.setWindowTitle('真实特征空间分析 · 传统视觉')
        self.resize(1120, 780)
        self.client = client
        self.scope = dict(scope)
        self.pseudo, self.preprocess = pseudo, preprocess
        self._closed = False
        self._generation = 0
        self._task_id = None
        self._polling = False
        self._artists = {}
        self._samples = []
        root = QVBoxLayout(self)
        notice = QLabel('传统视觉特征展示，非生产模型深度特征/非效果评估\n'
                        '不是disc深度向量；三维投影距离不是概率，不提供增强收益或自动消融结论。')
        notice.setWordWrap(True)
        root.addWidget(notice)
        scope_text = f'品类：{scope["category"]} | '
        if scope.get('datasource_id') is not None:
            scope_text += f'数据源 #{scope["datasource_id"]}（同源同品类）'
        elif scope.get('dataset_id') is not None:
            scope_text += f'批次 #{scope["dataset_id"]}（无数据源，限定该批次）'
        else:
            scope_text += '该品类全部未挂批次样本及批次（未指定数据源）'
        root.addWidget(QLabel(scope_text))
        controls = QHBoxLayout()
        controls.addWidget(QLabel('每类最多（正常至少3张独立内容）'))
        self.limit = QSpinBox()
        self.limit.setRange(3, 12)
        self.limit.setValue(8)
        controls.addWidget(self.limit)
        self.start = QPushButton('开始分析')
        self.start.clicked.connect(self._start)
        controls.addWidget(self.start)
        self.cancel = QPushButton('取消任务')
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self._cancel)
        controls.addWidget(self.cancel)
        controls.addStretch()
        root.addLayout(controls)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        root.addWidget(self.progress)
        self.status = QLabel('0模型前向；512px合成/256px传统提取，预计每图<500ms。固定种子20261015。')
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        body = QHBoxLayout()
        plot = QVBoxLayout()
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111, projection='3d')
        plot.addWidget(NavigationToolbar2QT(self.canvas, self))
        plot.addWidget(self.canvas, 1)
        body.addLayout(plot, 3)
        detail = QVBoxLayout()
        self.image = QLabel('点击散点查看真实样本/内存伪异常')
        self.image.setMinimumSize(200, 180)
        self.image.setAlignment(Qt.AlignCenter)
        detail.addWidget(self.image)
        self.detail = QLabel('正常：蓝色圆点\n真实缺陷：红色三角\n伪异常：橙色方块\n\n拖动旋转；工具栏缩放/平移/复位。')
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail.addWidget(self.detail)
        detail.addStretch()
        body.addLayout(detail, 1)
        root.addLayout(body, 1)
        self.summary = QLabel('仅抽取有界候选；小样本不能代表全库或模型检测效果。')
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.summary)
        self._pick_connection = self.canvas.mpl_connect('pick_event', self._pick)
        self.timer = QTimer(self)
        self.timer.setInterval(600)
        self.timer.timeout.connect(self._poll)
        self.finished.connect(self._cleanup)

    def _start(self):
        if self._closed or not self.start.isEnabled():
            return
        self._generation += 1
        generation = self._generation
        self.start.setEnabled(False)
        self.limit.setEnabled(False)
        self.status.setText('正在提交后台分析…')
        self.progress.setValue(0)
        payload = dict(self.scope, sample_limit=self.limit.value(), pseudo=self.pseudo,
                       preprocess=self.preprocess)
        client = self.client
        # Workers belong to the application, not a potentially closed dialog.
        def submitted(result):
            task_id = (result or {}).get('task_id')
            if self._closed or generation != self._generation:
                if task_id:
                    run_async(QApplication.instance(), lambda: client.cancel_task(task_id))
                return
            if not task_id:
                self._fail('提交失败：请检查服务连接，或等待/取消已有分析任务后重试')
                return
            self._task_id = task_id
            self.cancel.setEnabled(True)
            self.timer.start()
            self._poll()  # Do not rely on websocket: short tasks can already be done.
        run_async(QApplication.instance(), lambda: client.analyze_feature_space(payload),
                  submitted, lambda err: self._guard_fail(generation, err))

    def _guard_fail(self, generation, err):
        if not self._closed and generation == self._generation:
            self._fail(str(err))

    def _fail(self, message):
        self.timer.stop()
        self._task_id = None
        self.start.setEnabled(True)
        self.limit.setEnabled(True)
        self.cancel.setEnabled(False)
        self.status.setText(message)

    def _poll(self):
        if self._closed or self._polling or self._task_id is None:
            return
        self._polling = True
        generation, task_id = self._generation, self._task_id
        def received(task):
            if self._closed or generation != self._generation:
                return
            self._polling = False
            if not task:
                self.status.setText('获取任务状态失败，正在重试；可取消或关闭。')
                return
            self.progress.setValue(round(100 * (task.get('progress') or 0)))
            self.status.setText(task.get('message') or task['status'])
            state = task['status']
            if state in ('done', 'failed', 'cancelled'):
                self._fail({'done': '分析完成', 'failed': '分析失败：' + task.get('message', ''),
                            'cancelled': '任务已取消'}[state])
                if state == 'done':
                    try:
                        self.render_result(task['result'])
                    except Exception as exc:
                        self.status.setText(f'绘图失败：{exc}')
        def failed(err):
            if not self._closed and generation == self._generation:
                self._polling = False
                self.status.setText(f'获取状态失败，正在重试：{err}')
        client = self.client
        run_async(QApplication.instance(), lambda: client.get_task(task_id), received, failed)

    def _cancel(self):
        if self._task_id is None:
            return
        task_id, client = self._task_id, self.client
        self.cancel.setEnabled(False)
        self.status.setText('已请求协作取消，等待当前图处理结束…')
        run_async(QApplication.instance(), lambda: client.cancel_task(task_id))

    def render_result(self, result):
        self.axes.clear()
        self._artists.clear()
        self._samples = result['samples']
        for kind, color, marker, label in (
                ('normal', '#2878B5', 'o', 'Normal'),
                ('defect', '#C43D3D', '^', 'Real defect'),
                ('pseudo', '#DA8A14', 's', 'Pseudo anomaly')):
            points = [p for p in self._samples if p['kind'] == kind]
            if not points:
                continue
            artist = self.axes.scatter(*zip(*(p['xyz'] for p in points)),
                                       c=color, marker=marker, label=label, s=55,
                                       picker=6, depthshade=False)
            self._artists[artist] = points
        for axis, variance, setter in zip(('PC1', 'PC2', 'PC3'), result['explained_variance'],
                                          (self.axes.set_xlabel, self.axes.set_ylabel, self.axes.set_zlabel)):
            setter(f'{axis} ({variance:.1%})')
        self.axes.legend()
        self.canvas.draw_idle()
        self.summary.setText(
            f'实际数量 {result["actual_counts"]} | 有界候选 {result["candidate_counts"]} | '
            f'维数 {result["original_dimensions"]} → {result["effective_dimensions"]} | 秩 {result["rank"]}\n'
            f'跳过 { {k: len(v) for k, v in result["skipped"].items()} }\n' + '\n'.join(result['warnings']))
        self.summary.setToolTip(str({'config': result['config'], 'skipped': result['skipped']}))

    def _pick(self, event):
        points = self._artists.get(event.artist)
        if points and len(event.ind):
            self.select_sample(points[int(event.ind[0])])

    def select_sample(self, sample):
        pixmap = QPixmap()
        if not pixmap.loadFromData(base64.b64decode(sample['thumbnail']), 'PNG'):
            raise ValueError('样本缩略图损坏')
        self.image.setPixmap(pixmap.scaled(220, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        names = {'normal': '真实正常', 'defect': '真实缺陷', 'pseudo': '内存伪异常'}
        self.detail.setText(f'{names[sample["kind"]]} | #{sample["id"]}\n品类：{sample["category"]}\n'
                            f'来源基底ID：{sample.get("base_id") or "—（真实样本）"}\n'
                            f'高维最近正常距离：{sample["distance"]:.5g}\n'
                            f'最近正常ID：{sample["nearest_normal_id"]}\n'
                            '正常排除自身/同内容；伪异常允许匹配来源正常。\n不是概率或生产异常分数。')

    def _cleanup(self, _result=None):
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        self._cancel()
        self._generation += 1
        self.canvas.mpl_disconnect(self._pick_connection)

    def closeEvent(self, event):
        self._cleanup()
        super().closeEvent(event)
