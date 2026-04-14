from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_similarity(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    metric: str = "dot",
    clamp: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    if lhs.shape[-1] != rhs.shape[-1]:
        raise ValueError(
            "pairwise_similarity expects matching signature dimensions, "
            f"got {lhs.shape[-1]} and {rhs.shape[-1]}."
        )

    metric_name = metric.lower()
    if metric_name == "cosine" and lhs.shape[-1] > 0:
        lhs = F.normalize(lhs, p=2, dim=-1, eps=eps)
        rhs = F.normalize(rhs, p=2, dim=-1, eps=eps)

    if metric_name not in {"dot", "dot-sigmoid", "cosine"}:
        raise ValueError(f"Unsupported signature similarity metric: {metric}")

    similarity = torch.matmul(lhs, rhs.transpose(-1, -2))
    if metric_name == "dot-sigmoid":
        similarity = torch.sigmoid(similarity)
    if clamp:
        similarity = similarity.clamp(-1.0, 1.0)
    return similarity


def pairwise_distance(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    metric: str = "dot",
    clamp: bool = False,
) -> torch.Tensor:
    if lhs.shape[0] == 0 or rhs.shape[0] == 0:
        return torch.zeros((lhs.shape[0], rhs.shape[0]), dtype=torch.float32, device=lhs.device)
    return 1.0 - pairwise_similarity(
        lhs,
        rhs,
        metric=metric,
        clamp=clamp,
    )
