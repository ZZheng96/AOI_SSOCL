"""Bounded, model-free traditional feature display; never writes image assets."""
from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from sqlalchemy import or_

from ..core.tasks import TaskCancelled, task_manager
from ..db.database import session_scope
from ..db.models import Dataset, Image

NOTICE = "传统视觉特征展示，非生产模型深度特征/非效果评估"
SEED = 20261015
_submit_lock = threading.Lock()
_active_task = None


def submit_analysis(payload):
    global _active_task
    if not _submit_lock.acquire(blocking=False):
        raise ValueError('正在提交真实特征空间分析，请稍后重试')
    try:
        if _active_task is not None:
            current = task_manager.get(_active_task)
            if current and current['status'] in ('pending', 'running'):
                raise ValueError('已有真实特征空间分析任务，请等待或取消后重试')
        def work(progress, cancel_event):
            return analyze(payload, progress, cancel_event)
        _active_task = task_manager.submit('traditional_space', payload, work)
        return _active_task
    finally:
        _submit_lock.release()


def project_features(features, samples):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('传统特征含无效值')
    ni = [i for i, p in enumerate(samples) if p['kind'] == 'normal']
    if len({samples[i]['content_hash'] for i in ni}) < 3:
        raise ValueError('至少需要3张内容独立的有效正常参考图')
    ref = x[ni]
    mean, scale = ref.mean(axis=0), ref.std(axis=0)
    keep = scale > np.maximum(1e-8, np.abs(mean) * 1e-6)
    if not keep.any():
        raise ValueError('正常参考无有效方差维，无法标准化或投影')
    z = (x[:, keep] - mean[keep]) / scale[keep]
    for i, p in enumerate(samples):
        eligible = [j for j in ni if j != i and not (
            p['kind'] == 'normal' and samples[j]['content_hash'] == p['content_hash'])]
        distances = np.linalg.norm(z[eligible] - z[i], axis=1)
        j = eligible[int(np.argmin(distances))]
        p['nearest_normal_id'] = samples[j]['id']
        p['distance'] = float(distances.min())
    centered = z - z.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.sum(s > (s[0] * max(centered.shape) * np.finfo(float).eps))) if s.size else 0
    coords = np.zeros((len(x), 3))
    axes = min(3, rank)
    coords[:, :axes] = u[:, :axes] * s[:axes]
    variance = np.zeros(3)
    if np.sum(s ** 2) > 0:
        variance[:axes] = s[:axes] ** 2 / np.sum(s ** 2)
    for p, xyz in zip(samples, coords):
        p['xyz'] = xyz.tolist()
    return {'samples': samples, 'rank': rank, 'explained_variance': variance.tolist(),
            'original_dimensions': int(x.shape[1]), 'effective_dimensions': int(keep.sum()),
            'normal_mean': mean[keep].tolist(), 'normal_scale': scale[keep].tolist()}


def _read_rgb(path):
    # Inspect headers before decoding to bound memory, including donor mask images.
    with PILImage.open(path) as image:
        w, h = image.size
        if min(w, h) < 32 or max(w, h) > 16000 or w * h > 40_000_000 or max(w, h) / min(w, h) > 4:
            raise ValueError('尺寸超限（最短边32、最长边16000、4000万像素、宽高比≤4）')
        image = image.convert('RGB')
        image.thumbnail((512, 512))
        rgb = np.asarray(image).copy()
        # The existing synthesis operators assume square input (height-based patches).
        side = max(rgb.shape[:2])
        return cv2.resize(rgb, (side, side), interpolation=cv2.INTER_AREA)


def _mask_headers_safe(path):
    p = Path(path)
    candidates = [p.with_name(f'{p.stem}_mask{p.suffix}'),
                  p.with_name(f'{p.stem}_mask.png'),
                  p.parent / 'ground_truth' / f'{p.stem}_mask.png']
    candidates += [parent / 'ground_truth' / f'{p.stem}_mask.png'
                   for parent in list(p.parents)[1:3]]
    for candidate in candidates:
        if candidate.exists():
            with PILImage.open(candidate) as image:
                w, h = image.size
                return min(w, h) > 0 and max(w, h) <= 16000 and w * h <= 40_000_000
    return True


def analyze(payload, progress, cancel_event=None):
    from algo.vendor.traditional import TraditionalFeatureExtractor
    from algo.slots.disc import synth_pseudo, load_defect_mask
    from algo.common.preprocess import preprocess_image

    def check():
        if cancel_event is not None and cancel_event.is_set():
            raise TaskCancelled('已取消传统特征分析')

    check()
    rng = np.random.default_rng(SEED)
    limit = payload['sample_limit']
    candidates = {}
    with session_scope() as session:
        query = (session.query(Image).outerjoin(Dataset, Image.dataset_id == Dataset.id)
                 .filter(Image.category == payload['category'],
                         Image.source.notin_(['pseudo', 'augmented', 'augment']),
                         Image.split != 'pseudo',
                         or_(Dataset.id.is_(None), Dataset.source_type.notin_(['pseudo', 'augment', 'augmented']))))
        if payload.get('datasource_id') is not None:
            query = query.filter(Dataset.datasource_id == payload['datasource_id'])
        elif payload.get('dataset_id') is not None:
            query = query.filter(Image.dataset_id == payload['dataset_id'])
        for kind, label in [('normal', 'normal'), ('defect', 'anomaly')]:
            rows = query.filter(Image.label == label).order_by(Image.id).limit(120).all()
            candidates[kind] = [{'id': r.id, 'path': r.path} for r in rows]
    warnings = ['每类候选只取ID排序前120条再固定种子抽样，不代表全库分布；512px内方形归一化合成、256px提取会改变比例并丢失微小缺陷。',
                '距离为正常参考标准化后的高维欧氏距离；三维投影距离不是概率、阈值或正常范围球。']
    samples, features, normals, donors = [], [], [], []
    skipped = {'normal': [], 'defect': [], 'pseudo': []}
    extractor = TraditionalFeatureExtractor()
    total, done = sum(len(v) for v in candidates.values()) + limit, 0

    def add(kind, identifier, rgb, base_id=None):
        processed = preprocess_image(rgb, payload.get('preprocess') or {})
        small = cv2.resize(processed, (256, 256), interpolation=cv2.INTER_AREA)
        # A per-image RNG avoids extraction consuming the synthesis RNG stream.
        feature = extractor.extract(small, rng=np.random.default_rng(SEED))
        if not np.isfinite(feature).all():
            raise ValueError('特征含非有限数值')
        thumb = cv2.resize(processed, (max(1, round(processed.shape[1] * 160 / max(processed.shape[:2]))),
                                       max(1, round(processed.shape[0] * 160 / max(processed.shape[:2])))))
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError('缩略图编码失败')
        sample = {'kind': kind, 'id': identifier, 'base_id': base_id,
                  'category': payload['category'],
                  'content_hash': hashlib.sha256(rgb.tobytes()).hexdigest(),
                  'thumbnail': base64.b64encode(encoded).decode('ascii')}
        samples.append(sample)
        features.append(feature)

    for kind in ('defect', 'normal'):
        valid = 0
        for index in rng.permutation(len(candidates[kind])):
            check()
            if valid >= limit:
                break
            row = candidates[kind][int(index)]
            try:
                rgb = _read_rgb(row['path'])
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                if kind == 'normal' and any(p['kind'] == 'normal' and p['content_hash'] == digest for p in samples):
                    raise ValueError('内容重复正常图，排除以保持独立参考')
                add(kind, row['id'], rgb)
                valid += 1
                if kind == 'normal':
                    normals.append((row['id'], rgb))
                else:
                    # Only size-checked mask files may be decoded by the shared loader.
                    try:
                        safe_mask = _mask_headers_safe(row['path'])
                        mask = load_defect_mask(row['path'], rgb.shape[:2]) if safe_mask else None
                        if not safe_mask:
                            warnings.append(f'缺陷#{row["id"]}掩码尺寸超限，不用于移植。')
                        if mask is not None and np.any(mask > 127):
                            donors.append((rgb, mask))
                    except (OSError, ValueError, cv2.error) as exc:
                        warnings.append(f'缺陷#{row["id"]}掩码无效，不用于移植：{exc}')
            except (OSError, ValueError, cv2.error) as exc:
                skipped[kind].append({'id': row['id'], 'reason': str(exc)})
            done += 1
            progress(done, total, f'{kind} #{row["id"]}：有效{valid}，跳过{len(skipped[kind])}')
    if len(normals) < 3:
        raise ValueError(f'有效独立正常参考不足3张（实际{len(normals)}）；跳过：{skipped["normal"]}')
    if not any(p['kind'] == 'defect' for p in samples):
        warnings.append('当前范围无有效真实缺陷，仅展示正常与伪异常；未构造缺失类别。')
    if not donors:
        warnings.append('无有效缺陷位置掩码，真实缺陷移植按现有合成规则回退至其他选中方式/CutPaste。')
    for identifier, rgb in normals:
        check()
        pseudo, _ = synth_pseudo(rgb, rng, defect_pool=donors, pcfg=payload.get('pseudo'))
        add('pseudo', f'pseudo:{identifier}', pseudo, identifier)
        done += 1
        progress(done, total, f'伪异常 来源#{identifier}')
    check()
    result = project_features(features, samples)
    if result['rank'] < 3:
        warnings.append(f'投影秩为{result["rank"]}，不足三轴以0填充，不补造样本。')
    result.update(notice=NOTICE, warnings=warnings, config=dict(payload, seed=SEED),
                  candidate_counts={k: len(v) for k, v in candidates.items()},
                  candidate_cap=120, skipped=skipped,
                  actual_counts={k: sum(p['kind'] == k for p in samples) for k in ('normal', 'defect', 'pseudo')})
    return result
