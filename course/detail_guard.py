"""Recall-preserving union of original and auxiliary defect masks."""
import torch


def union_mask(original, detail_map, threshold, image_label):
    """Add detail pixels only in images already classified anomalous.

    The original mask is always a subset of the returned mask. Scores and
    continuous anomaly maps are intentionally outside this operation.
    """
    if original.shape != detail_map.shape:
        raise ValueError('Original mask and detail map must share a shape')
    threshold = torch.as_tensor(threshold, device=detail_map.device, dtype=detail_map.dtype)
    if threshold.numel() != 1 or not torch.isfinite(threshold).all() or threshold < 0:
        raise ValueError('Detail threshold must be a finite nonnegative scalar')
    gate = image_label.bool().reshape(-1, *([1] * (original.ndim - 1)))
    return original.bool() | ((detail_map > threshold) & gate)

class DetailGuardCore(torch.nn.Module):
    """Two feature banks, original ranking, recall-preserving binary localization."""
    def __init__(self, original_core, auxiliary_core, original_processor, threshold):
        super().__init__()
        self.original = original_core
        self.auxiliary = auxiliary_core
        self.reference_processor = original_processor
        self.register_buffer('detail_threshold', torch.tensor(float(threshold)))
        if not torch.isfinite(self.detail_threshold) or self.detail_threshold < 0:
            raise ValueError('Detail threshold must be finite and nonnegative')

    def forward(self, image):
        from anomalib.data import InferenceBatch
        raw = self.original(image)
        reference = self.reference_processor(raw)
        if reference.pred_mask is None:
            return raw
        detail = self.auxiliary(image)
        mask = union_mask(reference.pred_mask, detail.anomaly_map, self.detail_threshold, reference.pred_label)
        return InferenceBatch(pred_score=raw.pred_score, anomaly_map=raw.anomaly_map, pred_mask=mask)


def install_detail_guard(model, auxiliary, threshold):
    """Runtime installation; source checkpoints are kept unchanged and separately hashed."""
    import copy
    from anomalib.post_processing import PostProcessor
    class DetailPostProcessor(PostProcessor):
        def forward(self, predictions):
            output = super().forward(predictions)
            if self.enable_thresholding and predictions.pred_mask is not None:
                output = output._replace(pred_mask=predictions.pred_mask)
            return output
    previous = model.post_processor
    processor = DetailPostProcessor(enable_normalization=previous.enable_normalization,
        enable_thresholding=previous.enable_thresholding, enable_threshold_matching=previous.enable_threshold_matching,
        image_sensitivity=previous.image_sensitivity, pixel_sensitivity=previous.pixel_sensitivity)
    processor.load_state_dict(previous.state_dict(), strict=True)
    model.model = DetailGuardCore(model.model, auxiliary.model, copy.deepcopy(previous), threshold)
    model.post_processor = processor.to(model.device)
    model.eval()
    return model
