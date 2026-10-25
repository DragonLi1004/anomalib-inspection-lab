"""Mask-only refinement: keep anomaly scores/maps and image decisions unchanged."""
import torch
from torch.nn import functional as F
from anomalib.post_processing import PostProcessor

CANDIDATES = [{'operation':'identity','kernel':1}] + [
    {'operation':operation,'kernel':kernel}
    for operation in ('open','close') for kernel in (3,5,9)
] + [{'operation':'normal_gate','kernel':1}]


def refine_mask(mask, score, operation='identity', kernel=1):
    if operation == 'identity':
        return mask
    if operation == 'normal_gate':
        # Conservative normalized score gate, fixed before evaluating any test set.
        return mask & (score.reshape(-1,*([1]*(mask.ndim-1))) >= .25)
    if operation not in ('open','close') or kernel not in (3,5,9):
        raise ValueError('Unknown mask refinement')
    shape=mask.shape
    x=mask.float().reshape(-1,1,*shape[-2:])
    pad=kernel//2
    def dilate(t):
        return F.max_pool2d(F.pad(t,(pad,pad,pad,pad),value=0),kernel,stride=1)
    def erode(t):
        return -F.max_pool2d(-F.pad(t,(pad,pad,pad,pad),value=1),kernel,stride=1)
    result=dilate(erode(x)) if operation=='open' else erode(dilate(x))
    return result.reshape(shape).bool()


class MaskRefinement(PostProcessor):
    def __init__(self, operation='identity', kernel=1, **kwargs):
        super().__init__(**kwargs)
        if {'operation':operation,'kernel':kernel} not in CANDIDATES:
            raise ValueError('Unsupported refinement configuration')
        self.operation,self.kernel=operation,kernel

    def post_process_batch(self,batch):
        super().post_process_batch(batch)
        if batch.pred_mask is not None:
            batch.pred_mask=refine_mask(batch.pred_mask,batch.pred_score,self.operation,self.kernel)

    def forward(self,predictions):
        output=super().forward(predictions)
        if output.pred_mask is None:
            return output
        return output._replace(pred_mask=refine_mask(output.pred_mask,output.pred_score,self.operation,self.kernel))


def install_refinement(model,config):
    old=model.post_processor
    replacement=MaskRefinement(**config,enable_normalization=old.enable_normalization,
        enable_thresholding=old.enable_thresholding,enable_threshold_matching=old.enable_threshold_matching,
        image_sensitivity=old.image_sensitivity,pixel_sensitivity=old.pixel_sensitivity)
    replacement.load_state_dict(old.state_dict(),strict=True)
    model.post_processor=replacement.to(model.device)
    return model
