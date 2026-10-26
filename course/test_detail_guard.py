"""Structural guarantees for detail-mask additions, independent of model quality."""
import unittest
import torch
from anomalib.data import InferenceBatch, ImageBatch
from anomalib.post_processing import PostProcessor
from detail_guard import union_mask, install_detail_guard


class FixedCore(torch.nn.Module):
    def __init__(self, values, score=2.):
        super().__init__()
        self.register_buffer('values', torch.tensor(values, dtype=torch.float32).reshape(1, 1, 2, 2))
        self.register_buffer('score', torch.tensor([score]))
    def forward(self, image):
        return InferenceBatch(pred_score=self.score, anomaly_map=self.values)


class Container(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.model = core
        self.post_processor = PostProcessor()
        for name, value in {'_image_threshold': 1., '_pixel_threshold': 1., 'image_min': 0.,
                           'image_max': 4., 'pixel_min': 0., 'pixel_max': 4.}.items():
            getattr(self.post_processor, name).fill_(value)
    @property
    def device(self):
        return torch.device('cpu')


class DetailGuardTests(unittest.TestCase):
    def test_union_is_monotone_and_gated(self):
        original = torch.tensor([[[[True, False], [False, False]]]]).repeat(2, 1, 1, 1)
        detail = torch.ones_like(original, dtype=torch.float32) * 2
        result = union_mask(original, detail, 1., torch.tensor([True, False]))
        self.assertTrue(result[0].all())
        self.assertTrue(torch.equal(result[1], original[1]))
        self.assertFalse((original & ~result).any())

    def test_invalid_thresholds_and_shapes(self):
        mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
        for value in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                union_mask(mask, mask.float(), value, torch.tensor([True]))
        with self.assertRaises(ValueError):
            union_mask(mask, torch.zeros(1, 2, 2), 1., torch.tensor([True]))

    def test_forward_and_callback_preserve_scores_and_map(self):
        model = Container(FixedCore([.1, 2., .8, .4]))
        aux = Container(FixedCore([3., .2, .1, .1]))
        sample = torch.zeros(1, 3, 2, 2)
        reference = model.post_processor(model.model(sample))
        install_detail_guard(model, aux, 1.)
        raw = model.model(sample)
        forward = model.post_processor(raw)
        batch = ImageBatch(image=sample, pred_score=raw.pred_score, anomaly_map=raw.anomaly_map, pred_mask=raw.pred_mask)
        model.post_processor.post_process_batch(batch)
        for name in ['pred_score', 'pred_label', 'anomaly_map']:
            self.assertTrue(torch.equal(getattr(reference, name).flatten(), getattr(forward, name).flatten()))
        self.assertEqual(int(forward.pred_mask.sum()), 2)
        self.assertTrue(torch.equal(batch.pred_mask.flatten(), forward.pred_mask.flatten()))
        self.assertFalse((reference.pred_mask & ~forward.pred_mask).any())


if __name__ == '__main__':
    unittest.main()
