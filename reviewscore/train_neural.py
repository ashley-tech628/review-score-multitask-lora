"""Optional new training run. A supplied base path/name is an explicit download choice."""
import argparse
import copy
import itertools
import json
from pathlib import Path
import random
import numpy as np
import torch
from .data import read_rows, split_rows
from .baseline import targets, metrics
from .neural import create, masked_mse, save_bundle, load_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--base', required=True, help='Local DistilBERT directory or Hub model ID')
    p.add_argument('--revision', default=None)
    p.add_argument('--local-only', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--limit', type=int, default=20000)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    a = p.parse_args()
    if min(a.limit, a.epochs, a.batch_size) < 1:
        p.error('limit, epochs and batch-size must be positive')
    if a.out.exists() and any(a.out.iterdir()):
        p.error('Output directory must be empty')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    splits, cleaning = split_rows(itertools.islice(read_rows(a.data), a.limit), seed=a.seed)
    if any(not rows for rows in splits.values()):
        p.error('Every partition must contain rows')
    if not np.isfinite(targets(splits['validation'])).any(axis=0).all():
        p.error('Validation needs labels for every target')
    model, tokenizer = create(a.base, a.revision, a.local_only)
    model.to(a.device)
    optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=2e-5)

    def inputs(rows):
        return {k: v.to(a.device) for k, v in tokenizer([r['text'] for r in rows], padding=True,
                    truncation=True, max_length=128, return_tensors='pt').items() if k in ['input_ids', 'attention_mask']}

    def infer(rows):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(**inputs(rows[i:i+a.batch_size])).cpu().numpy()
                                   for i in range(0, len(rows), a.batch_size)])

    best, history, best_state = float('inf'), [], None
    for epoch in range(a.epochs):
        print(f'Epoch {epoch+1}/{a.epochs}: {len(splits["train"])} training reviews', flush=True)
        model.train()
        order = list(splits['train']); random.shuffle(order)
        for i in range(0, len(order), a.batch_size):
            batch = order[i:i+a.batch_size]
            labels = torch.tensor(targets(batch), dtype=torch.float32, device=a.device)
            optimizer.zero_grad()
            loss = masked_mse(model(**inputs(batch)), labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if (i // a.batch_size) % 20 == 0:
                print(f'  batch {i // a.batch_size + 1}: loss={loss.item():.6f}', flush=True)
        val = metrics(targets(splits['validation']), infer(splits['validation']))
        history.append({'epoch': epoch+1, 'validation': val})
        print(f'Validation macro MSE: {val["macro_mse"]:.6f}', flush=True)
        if val['macro_mse'] < best:
            best = val['macro_mse']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_predictions = infer(splits['test'])
    a.out.mkdir(parents=True, exist_ok=True)
    save_bundle(model, tokenizer, a.out/'bundle', a.base, a.revision)
    original = infer(splits['validation'][:2])
    model, tokenizer = load_bundle(a.out/'bundle', a.local_only)
    model.to(a.device)
    np.testing.assert_allclose(infer(splits['validation'][:2]), original, rtol=1e-4, atol=1e-5)
    report = {'new_experiment': True, 'prefix_sample_limit': a.limit, 'seed': a.seed, 'split': 'product',
              'counts': {k: len(v) for k, v in splits.items()}, 'cleaning': cleaning,
              'history': history, 'test': metrics(targets(splits['test']), test_predictions),
              'bundle_reload_matches': True}
    (a.out/'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Saved and verified serving bundle:', a.out)


if __name__ == '__main__':
    main()
