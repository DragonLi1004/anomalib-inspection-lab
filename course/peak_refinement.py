"""Validation-calibrated peak-aware pixel threshold; image scores/maps stay upstream."""
import torch
from anomalib.post_processing import PostProcessor

ALPHAS=(0.0,0.05,0.1,0.2,0.3,0.4)


def peak_mask(raw_map, threshold, alpha):
    if not 0 <= alpha < 1:
        raise ValueError('alpha must be in [0,1)')
    if raw_map is None:
        return None
    peak=raw_map.flatten(1).amax(1).reshape(-1,*([1]*(raw_map.ndim-1)))
    return raw_map > (threshold + alpha*(peak-threshold).clamp_min(0))


class PeakPostProcessor(PostProcessor):
    def __init__(self,alpha=0.0,**kwargs):
        super().__init__(**kwargs)
        if not 0 <= alpha < 1: raise ValueError('alpha must be in [0,1)')
        self.alpha=float(alpha)

    def effective_pixel_threshold(self):
        if not self.enable_normalization:
            return self.normalized_pixel_threshold
        return self.pixel_threshold+(0.5-self.pixel_sensitivity)*(self.pixel_max-self.pixel_min)

    def post_process_batch(self,batch):
        raw_map=batch.anomaly_map
        super().post_process_batch(batch)
        if self.alpha and self.enable_thresholding:
            batch.pred_mask=peak_mask(raw_map,self.effective_pixel_threshold(),self.alpha)

    def forward(self,predictions):
        result=super().forward(predictions)
        if self.alpha and self.enable_thresholding:
            result=result._replace(pred_mask=peak_mask(predictions.anomaly_map,self.effective_pixel_threshold(),self.alpha))
        return result


def install_peak(model,alpha):
    old=model.post_processor
    new=PeakPostProcessor(alpha=alpha,enable_normalization=old.enable_normalization,
        enable_thresholding=old.enable_thresholding,enable_threshold_matching=old.enable_threshold_matching,
        image_sensitivity=old.image_sensitivity,pixel_sensitivity=old.pixel_sensitivity)
    new.load_state_dict(old.state_dict(),strict=True)
    model.post_processor=new.to(model.device)
    return model
