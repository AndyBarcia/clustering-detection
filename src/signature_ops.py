from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pairwise_similarity(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    metric: str = "dot",
    normalize: bool = False,
    clamp: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    if lhs.shape[-1] != rhs.shape[-1]:
        raise ValueError(
            "pairwise_similarity expects matching signature dimensions, "
            f"got {lhs.shape[-1]} and {rhs.shape[-1]}."
        )

    if normalize and lhs.shape[-1] > 0:
        lhs = F.normalize(lhs, p=2, dim=-1, eps=eps)
        rhs = F.normalize(rhs, p=2, dim=-1, eps=eps)

    metric_name = metric.lower()
    if metric_name != "dot":
        raise ValueError(f"Unsupported signature similarity metric: {metric}")

    similarity = torch.matmul(lhs, rhs.transpose(-1, -2))
    if clamp:
        similarity = similarity.clamp(-1.0, 1.0)
    return similarity


def pairwise_distance(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    metric: str = "dot",
    normalize: bool = False,
    clamp: bool = False,
) -> torch.Tensor:
    if lhs.shape[0] == 0 or rhs.shape[0] == 0:
        return torch.zeros((lhs.shape[0], rhs.shape[0]), dtype=torch.float32, device=lhs.device)
    return 1.0 - pairwise_similarity(
        lhs,
        rhs,
        metric=metric,
        normalize=normalize,
        clamp=clamp,
    )


def pairwise_similarity_np(
    x: np.ndarray,
    *,
    metric: str = "dot",
    normalize: bool = False,
    clamp: bool = False,
) -> np.ndarray:
    x_prepared = np.asarray(x, dtype=np.float32)
    if normalize and x_prepared.shape[-1] > 0:
        denom = np.linalg.norm(x_prepared, axis=-1, keepdims=True)
        x_prepared = x_prepared / np.clip(denom, 1e-6, None)

    metric_name = metric.lower()
    if metric_name != "dot":
        raise ValueError(f"Unsupported signature similarity metric: {metric}")

    similarity = x_prepared @ x_prepared.T
    if clamp:
        similarity = np.clip(similarity, -1.0, 1.0)
    return np.asarray(similarity, dtype=np.float32)


def pairwise_distance_np(
    x: np.ndarray,
    *,
    metric: str = "dot",
    normalize: bool = False,
    clamp: bool = False,
) -> np.ndarray:
    distance = 1.0 - pairwise_similarity_np(
        x,
        metric=metric,
        normalize=normalize,
        clamp=clamp,
    )
    return np.ascontiguousarray(distance, dtype=np.float64)
