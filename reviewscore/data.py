"""Safe streaming input, strict label semantics and deterministic partitions."""
import ast
import hashlib
import json
import math
from . import TARGETS


def parse_rating(value):
    """Fractions or explicitly normalized numbers only; invalid labels are absent."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, str) and '/' in value:
            a, b = value.split('/')
            if float(b) <= 0:
                return None
            number = float(a) / float(b)
        else:
            number = float(value)
        return number if math.isfinite(number) and 0 <= number <= 1 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def read_rows(path):
    """JSONL or legacy Python-literal lines. Never execute input with eval.

    For large JSON arrays, convert to JSONL externally first. This reader rejects
    arrays instead of silently loading gigabytes into memory.
    """
    with open(path, encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = ast.literal_eval(line)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f'Invalid record on line {line_number}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Expected one object per line, at {line_number}')
            yield row


def clean_row(row):
    text = row.get('review/text')
    if not isinstance(text, str) or not text.strip():
        return None
    labels = [parse_rating(row.get('review/' + key)) for key in TARGETS]
    if all(v is None for v in labels):
        return None
    normalized = ' '.join(text.lower().split())
    group = row.get('beer/beerId')
    return {'text': text.strip(), 'labels': labels,
            'text_id': hashlib.sha256(normalized.encode()).hexdigest(),
            'group': '' if group is None else str(group).strip()}


def partition(key, seed=42):
    bucket = int(hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest()[:16], 16) / 2**64
    return 'train' if bucket < .8 else 'validation' if bucket < .9 else 'test'


def split_rows(rows, mode='product', seed=42):
    if mode not in ('product', 'text'):
        raise ValueError('mode must be product or text')
    result = {name: [] for name in ('train', 'validation', 'test')}
    seen = set()
    counts = {'invalid': 0, 'duplicates': 0, 'missing_group': 0}
    for raw in rows:
        row = clean_row(raw)
        if row is None:
            counts['invalid'] += 1
            continue
        if mode == 'product' and not row['group']:
            counts['missing_group'] += 1
            continue
        if row['text_id'] in seen:
            counts['duplicates'] += 1
            continue
        seen.add(row['text_id'])
        key = row['group'] if mode == 'product' else row['text_id']
        result[partition(key, seed)].append(row)
    return result, counts
