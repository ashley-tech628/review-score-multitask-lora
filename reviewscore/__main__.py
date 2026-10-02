import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
import numpy as np
from . import TARGETS
from .data import read_rows, split_rows
from .baseline import fit, predict, targets, metrics, bootstrap_delta, save, load


def main():
    p = argparse.ArgumentParser(description='Auditable review-scoring baseline and inference')
    sub = p.add_subparsers(dest='command', required=True)
    run = sub.add_parser('benchmark')
    run.add_argument('--data', required=True, type=Path)
    run.add_argument('--out', required=True, type=Path)
    run.add_argument('--limit', type=int, default=12000, help='First N input records; prefix sample, not representative')
    run.add_argument('--split', choices=['product', 'text'], default='product')
    run.add_argument('--seed', type=int, default=42)
    run.add_argument('--dimensions', type=int, default=128)
    pred = sub.add_parser('predict')
    pred.add_argument('--model', type=Path, required=True)
    pred.add_argument('--text', required=True)
    a = p.parse_args()
    if a.command == 'predict':
        if not a.text.strip():
            p.error('Text must not be empty')
        values = predict(load(a.model), [a.text])[0]
        print(json.dumps({'model': 'hashed_bow_ridge', 'scores': dict(zip(TARGETS, values.tolist())),
                          'note': 'Normalized regression estimates, not probabilities; not clipped.'}, indent=2))
        return
    if a.limit < 1 or a.dimensions < 1:
        p.error('limit and dimensions must be positive')
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Choose a new empty output directory')
    started = time.perf_counter()
    rows = list(itertools.islice(read_rows(a.data), a.limit))
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode())
    splits, cleaning = split_rows(rows, a.split, a.seed)
    if any(not v for v in splits.values()):
        p.error('Empty split: increase input limit or choose a different fixture')
    if any(not np.isfinite(targets(splits['validation'])[:, j]).any() for j in range(5)):
        p.error('Validation must contain labels for every target')
    trials = []
    models = []
    for alpha in [.1, 1., 10.]:
        model = fit(splits['train'], alpha, a.dimensions)
        score = metrics(targets(splits['validation']), predict(model, [r['text'] for r in splits['validation']]))
        trials.append({'alpha': alpha, 'validation_macro_mse': score['macro_mse']})
        models.append(model)
    best = min(range(len(trials)), key=lambda i: trials[i]['validation_macro_mse'])
    model = models[best]
    test = splits['test']
    y = targets(test)
    prediction = predict(model, [r['text'] for r in test])
    mean = np.tile(model['train_means'], (len(test), 1))
    report = {'source': a.data.name, 'sampling': 'first N records; source ordering may bias results',
              'input_records': len(rows), 'input_records_sha256': digest.hexdigest(),
              'split': a.split, 'seed': a.seed, 'cleaning': cleaning,
              'counts': {k: len(v) for k, v in splits.items()},
              'products': {k: len(set(r['group'] for r in v)) for k, v in splits.items()},
              'trials': trials, 'selected_alpha': model['alpha'], 'dimensions': a.dimensions,
              'test_ridge': metrics(y, prediction), 'test_train_mean': metrics(y, mean),
              'cluster_bootstrap': bootstrap_delta(y, prediction, mean, [r['group'] for r in test], seed=a.seed),
              'runtime_seconds': time.perf_counter() - started,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__}}
    # No review text, usernames, product IDs or individual predictions are exported.
    a.out.mkdir(parents=True, exist_ok=True)
    save(model, a.out / 'baseline.json')
    # Verify the exported serving artifact before reporting success.
    np.testing.assert_allclose(predict(load(a.out / 'baseline.json'), [r['text'] for r in test]), prediction)
    report['reload_predictions_match'] = True
    (a.out / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
