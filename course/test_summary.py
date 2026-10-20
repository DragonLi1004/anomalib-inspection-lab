"""Statistical and pairing checks without training or GPU dependencies."""
import copy
import unittest
from summarize_experiments import summarize, METHODS, SEEDS, MATCH_KEYS


def records():
    rows = []
    for seed, value in zip(SEEDS, (0.8, 0.9, 1.0)):
        for method in METHODS:
            row = {key: 'fixed' for key in MATCH_KEYS}
            row.update(dataset='bottle', seed=seed, method=method)
            row['metrics'] = [{key: value - (0.02 if method == 'pca96' else 0) for key in ('image_AUROC','image_F1Score','pixel_AUROC','pixel_F1Score')}]
            row.update(search_state_bytes=1000 if method == 'baseline' else 400, memory_bank_bytes=300,
                       projection_bytes=100, checkpoint_bytes=10000, latency_median_ms=20)
            rows.append(row)
    return rows


class SummaryTests(unittest.TestCase):
    def test_sample_sd_and_paired_changes(self):
        summary = summarize(records())
        compact = next(r for r in summary['rows'] if r['method'] == 'pca96')
        self.assertEqual(compact['n'], 3)
        self.assertAlmostEqual(compact['statistics']['image_AUROC']['mean'], 0.88)
        self.assertAlmostEqual(compact['statistics']['image_AUROC']['std'], 0.1)
        self.assertAlmostEqual(compact['paired_changes']['image_AUROC_delta_pp']['mean'], -2.0)
        self.assertAlmostEqual(compact['paired_changes']['search_state_reduction_percent']['mean'], 60.0)

    def test_mismatched_coreset_rejected(self):
        rows = records()
        rows[1]['coreset_sha256'] = 'different'
        with self.assertRaisesRegex(ValueError, 'coreset_sha256'):
            summarize(rows)

    def test_missing_method_excludes_whole_group(self):
        result = summarize(records()[1:])
        self.assertEqual(result['matched_groups'], 2)
        self.assertTrue(all(r['n'] == 2 for r in result['rows']))
        self.assertFalse(result['complete'])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            summarize(records(), require_complete=True)

    def test_one_seed_does_not_imply_zero_uncertainty(self):
        result = summarize(records()[:3])
        self.assertIsNone(result['rows'][0]['statistics']['image_AUROC']['std'])


if __name__ == '__main__':
    unittest.main()
