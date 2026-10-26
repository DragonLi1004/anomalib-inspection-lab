"""FineGuard geometry and original-score invariants with different branch resolutions."""
import unittest
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import Resize
from anomalib.data import InferenceBatch, ImageBatch
from anomalib.pre_processing import PreProcessor
from anomalib.post_processing import PostProcessor
from fine_guard import install_fine_guard


class ImageCore(torch.nn.Module):
    def __init__(self, extra=False):
        super().__init__()
        self.extra = extra
    def forward(self, image):
        anomaly = image[:, :1] + (.2 if self.extra else 0)
        return InferenceBatch(anomaly_map=anomaly, pred_score=torch.full((len(image),), 2., device=image.device))


class Branch(torch.nn.Module):
    def __init__(self, size, extra=False):
        super().__init__()
        self.model = ImageCore(extra)
        self.pre_processor = PreProcessor(Resize((size, size), antialias=True))
        self.post_processor = PostProcessor()
        for name, value in {'_image_threshold': 1., '_pixel_threshold': .5, 'image_min': 0.,
                           'image_max': 4., 'pixel_min': 0., 'pixel_max': 1.}.items():
            getattr(self.post_processor, name).fill_(value)


class FineGuardTests(unittest.TestCase):
    def test_native_input_and_common_reference_grid(self):
        torch.manual_seed(4)
        image = torch.rand(1, 3, 400, 500)
        gt = tv_tensors.Mask(torch.zeros(1, 400, 500, dtype=torch.bool))
        gt[:, 100:180, 200:260] = True
        original, auxiliary = Branch(224), Branch(320, True)
        expected_image, expected_gt = original.pre_processor.transform(image, gt)
        expected = original.post_processor(original.model(expected_image))
        install_fine_guard(original, auxiliary, .6)
        batch = ImageBatch(image=image.clone(), gt_mask=gt.clone())
        original.pre_processor.on_test_batch_start(None, None, batch, 0)
        self.assertEqual(batch.image.shape[-2:], (400, 500))
        self.assertTrue(torch.equal(batch.gt_mask, expected_gt))
        raw = original.model(batch.image)
        result = original.post_processor(raw)
        self.assertEqual(result.pred_mask.shape[-2:], (224, 224))
        for key in ['pred_score', 'pred_label', 'anomaly_map']:
            self.assertTrue(torch.equal(getattr(result, key), getattr(expected, key)))
        self.assertFalse((expected.pred_mask & ~result.pred_mask).any())
        callback = ImageBatch(image=image, pred_score=raw.pred_score, anomaly_map=raw.anomaly_map, pred_mask=raw.pred_mask)
        original.post_processor.post_process_batch(callback)
        self.assertTrue(torch.equal(result.pred_mask.flatten(), callback.pred_mask.flatten()))

    def test_predict_without_ground_truth(self):
        original = install_fine_guard(Branch(224), Branch(320, True), .6)
        batch = ImageBatch(image=torch.zeros(1, 3, 400, 500))
        original.pre_processor.on_predict_batch_start(None, None, batch, 0)
        self.assertIsNone(batch.gt_mask)
        self.assertEqual(original.model(batch.image).anomaly_map.shape[-2:], (224, 224))
        with self.assertRaises(RuntimeError):
            original.pre_processor.on_train_batch_start(None, None, batch, 0)

class FineCacheTests(unittest.TestCase):
    def test_auxiliary_checkpoint_participates_in_cache_and_integrity(self):
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import predict
        from baseline import ROOT
        with tempfile.TemporaryDirectory(dir=ROOT / 'course/outputs', prefix='fine-cache-test-') as folder:
            root = Path(folder)
            run = root / 'course/outputs/fine'
            run.mkdir(parents=True)
            (run / 'baseline.ckpt').write_bytes(b'original')
            (run / 'auxiliary.ckpt').write_bytes(b'auxiliary')
            metadata = {'method': 'fine_guard', 'fine_threshold': 1.,
                'source_checkpoint_sha256': hashlib.sha256(b'original').hexdigest(),
                'auxiliary_checkpoint_sha256': hashlib.sha256(b'auxiliary').hexdigest()}
            (run / 'metrics.json').write_text(json.dumps(metadata))
            session = predict.PredictionSession()
            with patch.object(predict, 'ROOT', root), patch.object(predict.Patchcore, 'load_from_checkpoint', side_effect=lambda *a, **k: object()) as load, patch('fine_guard.install_fine_guard'):
                first, _, hit = session.load('fine', 'cpu')
                self.assertFalse(hit)
                again, _, hit = session.load('fine', 'cpu')
                self.assertTrue(hit)
                self.assertIs(first, again)
                self.assertEqual(load.call_count, 2)
                (run / 'auxiliary.ckpt').write_bytes(b'changed auxiliary')
                with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                    session.load('fine', 'cpu')
                metadata['auxiliary_checkpoint_sha256'] = hashlib.sha256(b'changed auxiliary').hexdigest()
                (run / 'metrics.json').write_text(json.dumps(metadata))
                newer, _, hit = session.load('fine', 'cpu')
                self.assertFalse(hit)
                self.assertIsNot(newer, first)
            session.clear()


if __name__ == '__main__':
    unittest.main()
