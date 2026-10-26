"""High-resolution auxiliary localization with original classification and mask preserved."""
import copy
import torch
from torch.nn import functional as F
from anomalib.data import InferenceBatch
from anomalib.pre_processing import PreProcessor
from anomalib.post_processing import PostProcessor
from detail_guard import union_mask
from torchvision.transforms.v2 import Resize
from anomalib.data.transforms.utils import extract_transforms_by_type


def native_branch_input(preprocessor, image):
    """Match upstream MVTec CPU resize followed by device-side normalization."""
    resizes = extract_transforms_by_type(preprocessor.transform, Resize)
    if len(resizes) != 1:
        raise ValueError("FineGuard requires exactly one reference Resize")
    resized = resizes[0](image.cpu()).to(image.device)
    return preprocessor.transform(resized)


class FineReferencePreProcessor(PreProcessor):
    """Keep original pixels for both branches; evaluate GT on the original 224px grid."""
    def __init__(self, reference):
        super().__init__(transform=None)
        self.reference = reference

    def prepare(self, batch):
        if batch.gt_mask is not None:
            _, batch.gt_mask = self.reference.transform(batch.image, batch.gt_mask)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        raise RuntimeError('FineGuard is an inference composition; train the branches separately')

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.prepare(batch)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.prepare(batch)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.prepare(batch)


class FinePostProcessor(PostProcessor):
    def forward(self, predictions):
        result = super().forward(predictions)
        if self.enable_thresholding and predictions.pred_mask is not None:
            result = result._replace(pred_mask=predictions.pred_mask)
        return result


class FineGuardCore(torch.nn.Module):
    def __init__(self, original, auxiliary, threshold):
        super().__init__()
        self.original = original.model
        self.auxiliary = auxiliary.model
        self.original_pre = original.pre_processor
        self.auxiliary_pre = auxiliary.pre_processor
        self.original_post = copy.deepcopy(original.post_processor)
        self.register_buffer('detail_threshold', torch.tensor(float(threshold)))
        if not torch.isfinite(self.detail_threshold) or self.detail_threshold < 0:
            raise ValueError('Detail threshold must be finite and nonnegative')

    def forward(self, image):
        original = self.original(native_branch_input(self.original_pre, image))
        reference = self.original_post(original)
        if reference.pred_mask is None:
            return original
        detail = self.auxiliary(native_branch_input(self.auxiliary_pre, image)).anomaly_map
        if detail.shape[-2:] != reference.pred_mask.shape[-2:]:
            detail = F.interpolate(detail, reference.pred_mask.shape[-2:], mode='bilinear', align_corners=False)
        mask = union_mask(reference.pred_mask, detail, self.detail_threshold, reference.pred_label)
        return InferenceBatch(pred_score=original.pred_score, anomaly_map=original.anomaly_map, pred_mask=mask)


def install_fine_guard(model, auxiliary, threshold):
    """Install separately trained detail branch; original buffers and weights stay untouched."""
    previous = model.post_processor
    processor = FinePostProcessor(enable_normalization=previous.enable_normalization,
        enable_thresholding=previous.enable_thresholding, enable_threshold_matching=previous.enable_threshold_matching,
        image_sensitivity=previous.image_sensitivity, pixel_sensitivity=previous.pixel_sensitivity)
    processor.load_state_dict(previous.state_dict(), strict=True)
    preprocessor = FineReferencePreProcessor(copy.deepcopy(model.pre_processor))
    core = FineGuardCore(model, auxiliary, threshold)
    model.model = core
    model.pre_processor = preprocessor
    model.post_processor = processor
    return model.eval()
