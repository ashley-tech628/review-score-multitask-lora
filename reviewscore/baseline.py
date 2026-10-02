"""NumPy-only signed hashed bag-of-words ridge regression.

Vocabulary-free features avoid fitting text transforms on validation/test data.
The solver leaves the intercept unpenalized; each target uses its own label mask.
"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from . import TARGETS


def features(texts, dimensions=128):
    x = np.zeros((len(texts), dimensions + 1), dtype=np.float64)
    for i, text in enumerate(texts):
        for token in re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower()):
            h = hashlib.blake2b(token.encode(), digest_size=8).digest()
            x[i, int.from_bytes(h[:4], 'little') % dimensions] += 1 if h[4] % 2 else -1
        norm = np.linalg.norm(x[i, :-1])
        if norm:
            x[i, :-1] /= norm
        x[i, -1] = 1
    return x


def targets(rows):
    return np.array([[np.nan if v is None else v for v in r['labels']] for r in rows], dtype=float)


def fit(rows, alpha=1., dimensions=128):
    if not rows or alpha <= 0 or dimensions < 1:
        raise ValueError('Need training rows, positive alpha and feature dimensions')
    x = features([r['text'] for r in rows], dimensions)
    y = targets(rows)
    weights = []
    means = []
    penalty = np.eye(dimensions + 1) * alpha
    penalty[-1, -1] = 0
    for j in range(len(TARGETS)):
        mask = np.isfinite(y[:, j])
        if not mask.any():
            raise ValueError(f'No training labels for {TARGETS[j]}')
        a, b = x[mask], y[mask, j]
        weights.append(np.linalg.solve(a.T @ a + penalty, a.T @ b))
        means.append(float(b.mean()))
    return {'format_version': 1, 'kind': 'hashed_bow_ridge', 'targets': list(TARGETS),
            'dimensions': dimensions, 'alpha': alpha, 'weights': np.array(weights).T.tolist(),
            'train_means': means}


def predict(model, texts):
    if model.get('format_version') != 1 or model.get('targets') != list(TARGETS):
        raise ValueError('Unsupported model format or target order')
    return features(texts, model['dimensions']) @ np.asarray(model['weights'])


def save(model, path):
    Path(path).write_text(json.dumps(model, allow_nan=False), encoding='utf-8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics(y, predictions):
    if y.shape != predictions.shape or y.ndim != 2 or y.shape[1] != len(TARGETS):
        raise ValueError('Expected aligned N by 5 targets and predictions')
    if not np.isfinite(predictions).all():
        raise ValueError('Non-finite predictions')
    result = {}
    for j, name in enumerate(TARGETS):
        mask = np.isfinite(y[:, j])
        error = predictions[mask, j] - y[mask, j]
        result[name] = {'n': int(mask.sum()), 'mse': float(np.mean(error**2)) if len(error) else None,
                        'mae': float(np.mean(abs(error))) if len(error) else None}
    values = [v['mse'] for v in result.values() if v['mse'] is not None]
    result['macro_mse'] = float(np.mean(values)) if values else None
    return result


def bootstrap_delta(y, candidate, reference, groups, samples=300, seed=42):
    """Product-cluster bootstrap of candidate minus reference macro MSE."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if len(unique) < 2:
        return {'interval': None, 'reason': 'Fewer than two product clusters'}
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    deltas = []
    for _ in range(samples):
        idx = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        deltas.append(metrics(y[idx], candidate[idx])['macro_mse'] - metrics(y[idx], reference[idx])['macro_mse'])
    return {'interval': np.quantile(deltas, [.025, .975]).tolist(), 'samples': samples,
            'unit': 'product cluster', 'direction': 'negative favors ridge over train-mean baseline'}
