import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from reviewscore.data import parse_rating, read_rows, split_rows, partition
from reviewscore.baseline import fit, predict, save, load, targets, metrics, bootstrap_delta


def fixture():
    return [{'review/text': f'{word} aroma sample {i}', 'beer/beerId': str(i // 3),
             **{'review/'+k: str(value) for k in ['appearance','aroma','palate','taste','overall']}}
            for i, (word, value) in enumerate([('good', .8), ('bad', .2)]*100)]


class PipelineTests(unittest.TestCase):
    def test_fraction_and_real_zero(self):
        self.assertEqual(parse_rating('13/20'), .65)
        self.assertEqual(parse_rating(0), 0)

    def test_invalid_labels_not_zero(self):
        for value in [None, '', 'bad', '3/0', '1/-2', float('nan'), float('inf'), True, 4, '-1/5']:
            self.assertIsNone(parse_rating(value))

    def test_no_product_or_duplicate_text_leakage(self):
        rows = fixture(); rows.append({**rows[0], 'beer/beerId': 'other'})
        splits, stats = split_rows(rows)
        self.assertEqual(stats['duplicates'], 1)
        for key in ['group', 'text_id']:
            sets = [set(r[key] for r in part) for part in splits.values()]
            for i in range(3):
                for j in range(i):
                    self.assertFalse(sets[i] & sets[j])

    def test_partition_stable(self):
        forward, _ = split_rows(fixture())
        backward, _ = split_rows(reversed(fixture()))
        self.assertEqual({k: {r['text_id'] for r in v} for k,v in forward.items()},
                         {k: {r['text_id'] for r in v} for k,v in backward.items()})

    def test_reader_does_not_execute_code(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.jsonl'
            p.write_text("__import__('os').getcwd()", encoding='utf-8')
            with self.assertRaises(ValueError): list(read_rows(p))

    def test_missing_targets_excluded_from_metrics(self):
        y=np.array([[np.nan, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
        m=metrics(y,np.zeros((2,5)))
        self.assertEqual(m['appearance']['n'],1)
        self.assertEqual(m['appearance']['mse'],1)
        self.assertEqual(m['taste']['mse'],.5)

    def test_model_reload_preserves_predictions(self):
        splits,_=split_rows(fixture())
        model=fit(splits['train'],dimensions=16)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'model.json';save(model,p)
            np.testing.assert_array_equal(predict(model,['good aroma','bad aroma']),
                                          predict(load(p),['good aroma','bad aroma']))

    def test_masked_training_ignores_missing_label(self):
        splits,_=split_rows(fixture());rows=splits['train']
        rows[0]['labels'][0]=None
        model=fit(rows,dimensions=16)
        self.assertTrue(np.isfinite(predict(model,['good'])).all())
        for r in rows:r['labels'][0]=None
        with self.assertRaises(ValueError):fit(rows,dimensions=16)

    def test_bootstrap_sign_and_reproducibility(self):
        y=np.zeros((8,5));good=y.copy();bad=np.ones_like(y)
        a=bootstrap_delta(y,good,bad,['a']*4+['b']*4,samples=20)
        self.assertEqual(a['interval'],[-1.,-1.])
        self.assertEqual(a,bootstrap_delta(y,good,bad,['a']*4+['b']*4,samples=20))

    def test_invalid_predictions_rejected(self):
        with self.assertRaises(ValueError):metrics(np.zeros((2,5)),np.full((2,5),np.nan))


if __name__=='__main__':unittest.main()
