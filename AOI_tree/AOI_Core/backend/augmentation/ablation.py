"""隔离拟合两套单因素配置，在固定留出集上真实打分，不修改生产快照。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..core.tasks import task_manager, TaskCancelled
from ..db.database import session_scope
from ..db.models import EvalRun, Image

_lock = threading.Lock()
_active_task = None


def submit_ablation(payload):
    global _active_task
    with _lock:
        if _active_task is not None:
            task = task_manager.get(_active_task)
            if task and task['status'] in ('pending', 'running'):
                raise ValueError('已有增强消融任务正在运行')
        _active_task = task_manager.submit('augment_ablation', payload,
            lambda cb, cancel_event: run_ablation(payload, cb, cancel_event))
        return _active_task


def configs(base, payload):
    a = deepcopy(base)
    aug = a.setdefault('augment', {})
    for key in ('pseudo', 'preprocess', 'enhance'):
        if payload.get(key) is not None:
            aug.setdefault(key, {}).update(payload[key])
    disc = a['slots'].get('disc', {})
    count = aug.get('pseudo', {}).pop('pseudo_per_image', None)
    if count is not None:
        disc['pseudo_per_image'] = count
    factor = payload['factor']
    group, field = factor.split('.')
    if group == 'pseudo':
        if field != 'enabled':
            raise ValueError('不支持对伪异常子字段执行消融')
        if not disc.get('pseudo_per_image', 8):
            raise ValueError('基线伪异常数量已经为 0')
    elif field in ('clahe', 'median'):
        if not aug.get(group, {}).get(field) or not aug.get(group, {}).get('enabled'):
            raise ValueError('所选因素在基线中未启用')
    elif not aug.get(group, {}).get(field):
        raise ValueError('所选因素在基线中未启用')
    if group in ('pseudo', 'enhance'):
        only = a.get('fusion', {}).get('only_slots')
        if (not disc.get('enabled') or not a['backbone'].get('enabled', True)
                or (a.get('fusion', {}).get('skip_zero_weight') and only and 'disc' not in only)
                or disc.get('freeze') or (disc.get('epochs', 30) or 0) <= 0
                or (disc.get('finetune_epochs', 1) or 0) <= 0):
            raise ValueError('该因素只影响 disc 训练，但当前 disc 未参与训练（关闭/冻结/零轮数）；请先配置可训练的 disc')
        if disc.get('pseudo_per_image', 8) <= 0:
            raise ValueError('disc 训练需要正数伪异常数量')
    if group == 'preprocess' and field != 'enabled' and not aug.get('preprocess', {}).get('enabled'):
        raise ValueError('基线未启用预处理')
    b = deepcopy(a)
    if group == 'pseudo':
        b['slots']['disc']['pseudo_per_image'] = 0
    else:
        b['augment'][group][field] = False if field not in ('clahe', 'median') else 0
    return a, b


def _metrics(rows):
    y = np.asarray([r['label'] for r in rows])
    scores = np.asarray([r['score'] for r in rows])
    pred = np.asarray([r['decision'] != 'normal' for r in rows])
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    return {'auroc': float(roc_auc_score(y, scores)),
            'ap': float(average_precision_score(y, scores)),
            'f1': 2 * tp / max(2 * tp + fp + fn, 1),
            'recall': tp / (tp + fn), 'fpr': fp / (fp + tn),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def run_ablation(payload, progress_cb=None, cancel_event=None):
    from ..engine import get_engine
    from algo.factory import build
    from algo.eval.offline import Pipeline
    from algo.common.isolation import guard_bundle

    category = payload['category']
    meta = {'status': 'running', 'factor': payload['factor'], 'split': payload['split']}
    with session_scope() as s:
        run = EvalRun(name=f'augment_ablation_{category}', run_type='augment_ablation',
                      category=category, metrics=meta)
        s.add(run)
        s.flush()
        run_id = run.id

    def progress(frac, message):
        if cancel_event is not None and cancel_event.is_set():
            raise TaskCancelled('增强消融已取消')
        if progress_cb:
            progress_cb(int(frac * 1000), 1000, message)
        print(f'[augment-ablation #{run_id}] {frac:.0%} {message}', flush=True)

    try:
        progress(0, '固定配置与样本')
        engine = get_engine()
        a, b = configs(engine.base_cfg, payload)
        with session_scope() as s:
            def fetch(split, label, n):
                rows = (s.query(Image).filter(Image.category == category, Image.split == split,
                        Image.label == label, ~Image.source.in_(('pseudo', 'augmented')))
                        .order_by(Image.id).all())
                rng = np.random.default_rng(int(a.get('seed', 42)))
                indices = rng.permutation(len(rows))[:n]
                return [{'image_id': rows[i].id, 'path': rows[i].path,
                         'label': int(label == 'anomaly')} for i in indices]
            train = fetch('train', 'normal', payload['train_limit'])
            defects = fetch('train_anomaly', 'anomaly', min(payload['defect_limit'],
                             a.get('protocol', {}).get('n_init_defect', 30)))
            if not defects and payload['defect_limit']:
                defects = fetch('train', 'anomaly', min(payload['defect_limit'],
                             a.get('protocol', {}).get('n_init_defect', 30)))
            n = payload['eval_limit']
            heldout = fetch(payload['split'], 'normal', n // 2) + fetch(payload['split'], 'anomaly', n - n // 2)
        if len(train) < 2 or len({r['label'] for r in heldout}) != 2:
            raise ValueError('至少需要 2 张 train 正常图，且留出集同时包含正常与异常图')
        meta.update({'seed': a.get('seed', 42), 'baseline_config': a, 'ablated_config': b,
                     'train_images': train + defects, 'eval_images': heldout,
                     'n_images': len(heldout), 'metric_basis': 'fixed learned decision; gray counted as anomaly',
                     'limitations': '小样本单种子实验；不代表统计显著收益，不自动选型或激活模型'})
        # 存储原图指纹；训练 staging 只为兼容 guard_bundle 的 train 路径约定。
        for row in train + defects + heldout:
            row['sha256'] = hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
        meta['eval_manifest_sha256'] = hashlib.sha256(json.dumps(heldout, sort_keys=True).encode()).hexdigest()
        with session_scope() as s:
            run = s.get(EvalRun, run_id)
            run.metrics = deepcopy(meta)
            run.n_images = len(heldout)
        results = {}
        with tempfile.TemporaryDirectory(prefix='aoi_ablation_') as tmp:
            root = Path(tmp) / 'train'
            root.mkdir()
            def stage(rows):
                paths = []
                for row in rows:
                    dst = root / f"{row['image_id']}{Path(row['path']).suffix}"
                    shutil.copyfile(row['path'], dst)
                    paths.append(str(dst))
                return paths
            bundle = {'init_normal': stage(train), 'init_defect': stage(defects),
                      'val': [], 'test': [(r['path'], r['label']) for r in heldout]}
            guard_bundle(bundle, a['protocol'])
            for side, cfg, offset in (('baseline', a, 0.0), ('ablated', b, 0.5)):
                progress(offset + .01, f'{side} 初始化')
                backbone, slots = build(cfg, engine.device)
                pipe = Pipeline(cfg, backbone, slots)
                try:
                    pipe.fit(bundle, progress_cb=lambda f, m: progress(offset + .35 * f, f'{side}: {m}'))
                    rows = []
                    for i, row in enumerate(heldout):
                        progress(offset + .35 + .14 * i / len(heldout), f'{side} 留出集推理 {i + 1}/{len(heldout)}')
                        rec = pipe.predict(row['path'])
                        score = float(rec['fused'])
                        if not np.isfinite(score) or rec['decision'] not in ('normal', 'anomaly', 'gray'):
                            raise ValueError('无效预测分数或决策')
                        rows.append({**row, 'score': score, 'decision': rec['decision']})
                    results[side] = {'metrics': _metrics(rows), 'per_image': rows}
                finally:
                    pipe._cpu_exec.shutdown(wait=True)
                del pipe, slots, backbone
        meta.update(results)
        meta['delta'] = {k: results['ablated']['metrics'][k] - results['baseline']['metrics'][k]
                         for k in ('auroc', 'ap', 'f1', 'recall', 'fpr')}
        progress(1, '评估完成，保存真实指标')
        meta['status'] = 'completed'
    except Exception as exc:
        meta.update(status='cancelled' if isinstance(exc, TaskCancelled) else 'failed',
                    error=f'{type(exc).__name__}: {exc}')
        with session_scope() as s:
            s.get(EvalRun, run_id).metrics = deepcopy(meta)
        raise
    with session_scope() as s:
        s.get(EvalRun, run_id).metrics = deepcopy(meta)
    return {'eval_run_id': run_id, **meta}
