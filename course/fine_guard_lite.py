"""FineGuard Lite: one physical ResNet prefix and conditional detail inference."""
import torch
from torch.nn import functional as F
from anomalib.data import InferenceBatch
from anomalib.models import Patchcore
from fine_guard import FineGuardCore, install_fine_guard, native_branch_input
from detail_guard import union_mask


class FineGuardLiteCore(FineGuardCore):
    def forward(self, image):
        raw = self.original(native_branch_input(self.original_pre, image))
        reference = self.original_post(raw)
        if reference.pred_mask is None:
            return raw
        mask = reference.pred_mask.clone()
        active = reference.pred_label.bool().reshape(-1)
        if active.any():
            selected = image if image.shape[0] == 1 else image[active]
            detail = self.auxiliary(native_branch_input(self.auxiliary_pre, selected)).anomaly_map
            detail = F.interpolate(detail, mask.shape[-2:], mode="bilinear", align_corners=False)
            if image.shape[0] == 1:
                mask = union_mask(mask, detail, self.detail_threshold, active)
            else:
                mask[active] |= detail > self.detail_threshold
        return InferenceBatch(pred_score=raw.pred_score, anomaly_map=raw.anomaly_map, pred_mask=mask)


def export_checkpoint(source, target, threshold):
    base = torch.load(source / "baseline.ckpt", map_location="cpu", weights_only=False)
    aux = torch.load(source / "auxiliary.ckpt", map_location="cpu", weights_only=False)
    prefix = "model.feature_extractor."
    shared = {k: v for k, v in aux["state_dict"].items() if k.startswith(prefix)}
    if not shared or not all(torch.equal(v, base["state_dict"][k]) for k, v in shared.items()):
        raise ValueError("The two backbones do not have an identical prefix")
    config = {"backbone": "resnet18.tv_in1k", "num_neighbors": 9, "pre_trained": False}
    for ckpt, layers in [(base, ["layer2", "layer3"]), (aux, ["layer2"])]:
        hp = ckpt["hyper_parameters"]
        if hp["backbone"] != config["backbone"] or list(hp["layers"]) != layers or hp["num_neighbors"] != 9:
            raise ValueError("Unsupported FineGuard architecture")
    torch.save({"config": config, "base": base["state_dict"], "threshold": threshold,
                "detail": {k: v for k, v in aux["state_dict"].items() if k not in shared}}, target)


def load_lite(checkpoint):
    packed = torch.load(checkpoint, map_location="cpu", weights_only=True)
    def branch(layers, size, neighbors):
        return Patchcore(**{**packed["config"], "num_neighbors": neighbors}, layers=layers,
                         pre_processor=Patchcore.configure_pre_processor(image_size=(size, size)),
                         visualizer=False)

    base = branch(["layer2", "layer3"], 224, 9)
    base.load_state_dict(packed["base"])
    aux = branch(["layer2"], 384, 1)
    main_net = base.model.feature_extractor.feature_extractor
    detail_net = aux.model.feature_extractor.feature_extractor
    for name, block in list(detail_net.named_children()):
        other = getattr(main_net, name)
        if type(block) is not type(other):
            raise ValueError(f"Incompatible backbone block: {name}")
        setattr(detail_net, name, other)
    shared = {k: packed["base"][k] for k in aux.state_dict() if k.startswith("model.feature_extractor.")}
    aux.load_state_dict({**packed["detail"], **shared})
    lite = FineGuardLiteCore(base, aux, packed["threshold"])
    model = install_fine_guard(base, aux, packed["threshold"])
    model.model = lite
    return model.eval()
