"""Course extension: PCA compression after upstream coreset selection.

Only normal training coreset features fit the projection. Validation/test samples
never fit PCA. The upstream backbone, coreset selection and scoring stay intact.
"""
import hashlib
import torch
from torch.nn import functional as F
from anomalib.models import Patchcore
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


def fit_projection(bank, dimension):
    if not 1 <= dimension <= bank.shape[1]:
        raise ValueError("Projection dimension outside feature dimension")
    x = bank.detach().to(device="cpu", dtype=torch.float64)
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / max(1, x.shape[0] - 1)
    eigenvalues, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -dimension:].flip(1).contiguous()
    variance = eigenvalues.clamp_min(0)
    retained = variance[-dimension:].sum() / variance.sum().clamp_min(1e-12)
    return mean.to(bank), basis.to(bank), retained.to(bank)


class CompactPatchcoreModel(PatchcoreModel):
    def __init__(self, projection_dim=96, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.register_buffer("projection_mean", torch.empty(0))
        self.register_buffer("projection_basis", torch.empty(0))
        self.register_buffer("retained_variance", torch.tensor(0.0))

    def subsample_embedding(self, sampling_ratio, embeddings=None):
        if self.projection_basis.numel():
            raise RuntimeError("Create a fresh model for each training run")
        super().subsample_embedding(sampling_ratio)
        self.coreset_sha256 = hashlib.sha256(self.memory_bank.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        mean, basis, variance = fit_projection(self.memory_bank, self.projection_dim)
        self.projection_mean, self.projection_basis = mean, basis
        self.retained_variance = variance
        self.memory_bank = ((self.memory_bank - mean) @ basis).contiguous()

    def generate_embedding(self, features):
        embedding = super().generate_embedding(features)
        if self.projection_basis.numel():
            weight = self.projection_basis.T[:, :, None, None]
            bias = -(self.projection_mean @ self.projection_basis)
            embedding = F.conv2d(embedding, weight, bias)
        return embedding


class CompactPatchcore(Patchcore):
    def __init__(self, projection_dim=96, backbone="resnet18.tv_in1k", layers=("layer2", "layer3"),
                 pre_trained=True, coreset_sampling_ratio=0.01, num_neighbors=9,
                 pre_processor=True, post_processor=True, evaluator=True, visualizer=True):
        super().__init__(backbone=backbone, layers=layers, pre_trained=pre_trained,
                         coreset_sampling_ratio=coreset_sampling_ratio, num_neighbors=num_neighbors,
                         pre_processor=pre_processor, post_processor=post_processor,
                         evaluator=evaluator, visualizer=visualizer)
        # Preserve upstream RNG and exact backbone weights, including when pre_trained=False.
        # A second random initialization must not change subsequent coreset selection.
        with torch.random.fork_rng(devices=[]):
            compact = CompactPatchcoreModel(projection_dim=projection_dim, backbone=backbone,
                                            layers=layers, pre_trained=False, num_neighbors=num_neighbors)
        compact.load_state_dict(self.model.state_dict(), strict=False)
        self.model = compact
        self.save_hyperparameters("projection_dim")
