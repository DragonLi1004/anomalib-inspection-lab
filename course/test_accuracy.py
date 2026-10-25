"""Peak-mask behavior tests: changed masks must not change ranking or classification."""
import unittest
import torch
from anomalib.data import InferenceBatch, ImageBatch
from anomalib.post_processing import PostProcessor
from peak_refinement import PeakPostProcessor, peak_mask


def configured(cls=PostProcessor,**kwargs):
    p=cls(**kwargs)
    for name,value in {'_image_threshold':1.,'_pixel_threshold':1.,'image_min':0.,'image_max':4.,'pixel_min':0.,'pixel_max':4.}.items():
        getattr(p,name).fill_(value)
    return p


class PeakMaskTests(unittest.TestCase):
    def test_peak_threshold_formula_and_subset(self):
        raw=torch.tensor([[[[0.,1.,1.1,2.]]],[[[0.,.2,.7,.9]]]])
        expected=torch.tensor([[[[False,False,False,True]]],[[[False,False,False,False]]]])
        self.assertTrue(torch.equal(peak_mask(raw,torch.tensor(1.),.2),expected))
        self.assertFalse((peak_mask(raw,torch.tensor(1.),.2)&~(raw>1)).any())

    def test_disabled_is_bit_exact_and_nonmask_outputs_unchanged(self):
        raw=InferenceBatch(pred_score=torch.tensor([.8,2.]),anomaly_map=torch.tensor([[[[0.,1.1],[1.3,2.]]],[[[.5,1.2],[3.,4.]]]]))
        before=configured()(raw)
        disabled=configured(PeakPostProcessor,alpha=0)(raw)
        after=configured(PeakPostProcessor,alpha=.2)(raw)
        for field in raw._fields:
            self.assertTrue(torch.equal(getattr(before,field),getattr(disabled,field)))
        for field in ('pred_score','pred_label','anomaly_map'):
            self.assertTrue(torch.equal(getattr(before,field),getattr(after,field)))
        self.assertTrue((before.pred_mask!=after.pred_mask).any())

    def test_callback_matches_forward_with_nondefault_sensitivity(self):
        raw=InferenceBatch(pred_score=torch.tensor([2.]),anomaly_map=torch.tensor([[[[.1,1.2],[2.1,4.]]]]))
        p=configured(PeakPostProcessor,alpha=.1,pixel_sensitivity=.4)
        expected=p(raw)
        batch=ImageBatch(image=torch.zeros(1,3,2,2),pred_score=raw.pred_score,anomaly_map=raw.anomaly_map)
        p.post_process_batch(batch)
        self.assertTrue(torch.equal(expected.pred_mask.flatten(),batch.pred_mask.flatten()))
        self.assertTrue(torch.equal(expected.anomaly_map.flatten(),batch.anomaly_map.flatten()))

    def test_thresholding_disabled_and_invalid_coefficients(self):
        raw=InferenceBatch(pred_score=torch.tensor([2.]),anomaly_map=torch.ones(1,1,2,2))
        p=configured(PeakPostProcessor,alpha=.1,enable_thresholding=False)
        self.assertIsNone(p(raw).pred_mask)
        for alpha in (-.1,1.,float('nan')):
            with self.assertRaises(ValueError):PeakPostProcessor(alpha=alpha)

    def test_no_normalization_matches_upstream_fixed_threshold(self):
        raw=InferenceBatch(pred_score=torch.tensor([.8]),anomaly_map=torch.tensor([[[[.4,.6],[.7,.9]]]]))
        p=configured(PeakPostProcessor,alpha=.1,enable_normalization=False)
        expected=peak_mask(raw.anomaly_map,torch.tensor(.5),.1)
        self.assertTrue(torch.equal(p(raw).pred_mask,expected))

if __name__=='__main__':unittest.main()
