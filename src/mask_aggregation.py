from __future__ import annotations

import torch
import torch.nn.functional as F


def assignment_affinity(
    similarity: torch.Tensor,
    influence: torch.Tensor,
    similarity_floor: float = 0.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    affinity = (similarity + influence.unsqueeze(-1)).clamp(0.0, 1.0)
    if similarity_floor > 0.0:
        affinity = affinity.clamp_min(similarity_floor)
    if valid_mask is not None:
        affinity = affinity.masked_fill(~valid_mask.unsqueeze(-2), 0.0)
    return affinity


def assignment_weights_with_influence(
    similarity: torch.Tensor,
    influence: torch.Tensor,
    alpha: float | torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    similarity_floor: float = 0.0,
) -> torch.Tensor:
    affinity = assignment_affinity(
        similarity=similarity,
        influence=influence,
        similarity_floor=similarity_floor,
        valid_mask=valid_mask,
    )
    return affinity.pow(alpha)


def normalize_assignment_weights(
    weights: torch.Tensor,
    *,
    normalize_over_queries: bool,
    eps: float = 1e-6,
) -> torch.Tensor:
    if normalize_over_queries:
        denom = weights.sum(dim=-2, keepdim=True)
    else:
        denom = weights.sum(dim=-1, keepdim=True)
    return weights / (denom + eps)


def aggregate_with_weights(
    normalized_weights: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(normalized_weights.transpose(-1, -2), values)


def project_mask_embeddings(
    mask_embeddings: torch.Tensor,
    features: torch.Tensor,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    if mask_embeddings.dim() == 3 and features.dim() == 4:
        mask_logits = torch.einsum("bmc,bchw->bmhw", mask_embeddings, features)
    elif mask_embeddings.dim() == 2 and features.dim() == 3:
        mask_logits = torch.einsum("mc,chw->mhw", mask_embeddings, features)
    else:
        raise ValueError(
            "project_mask_embeddings expects [B,M,C] x [B,C,H,W] or [M,C] x [C,H,W] tensors, "
            f"got {tuple(mask_embeddings.shape)} and {tuple(features.shape)}."
        )
    return F.interpolate(
        mask_logits,
        size=image_shape,
        mode="bilinear",
        align_corners=False,
    )
