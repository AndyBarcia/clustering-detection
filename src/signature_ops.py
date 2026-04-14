from __future__ import annotations

import torch
import torch.nn.functional as F


def _softmax_pairwise_jsd(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    lhs_probs = F.softmax(lhs, dim=-1)
    rhs_probs = F.softmax(rhs, dim=-1)

    lhs_probs = lhs_probs.unsqueeze(-2)
    rhs_probs = rhs_probs.unsqueeze(-3)
    mixture = 0.5 * (lhs_probs + rhs_probs)

    log_base_change = torch.log(torch.tensor(2.0, dtype=lhs.dtype, device=lhs.device))

    lhs_probs = lhs_probs.clamp_min(eps)
    rhs_probs = rhs_probs.clamp_min(eps)
    mixture = mixture.clamp_min(eps)

    kl_lhs = (lhs_probs * (torch.log(lhs_probs) - torch.log(mixture))).sum(dim=-1) / log_base_change
    kl_rhs = (rhs_probs * (torch.log(rhs_probs) - torch.log(mixture))).sum(dim=-1) / log_base_change
    jsd = 0.5 * (kl_lhs + kl_rhs)
    return 1.0 - jsd


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
    elif metric_name == "softmax":
        lhs = F.softmax(lhs, dim=-1)
        rhs = F.softmax(rhs, dim=-1)
    elif metric_name == "jsd":
        similarity = _softmax_pairwise_jsd(lhs, rhs, eps=eps)
        if clamp:
            similarity = similarity.clamp(-1.0, 1.0)
        return similarity

    if metric_name not in {"dot", "dot-sigmoid", "cosine", "softmax", "jsd"}:
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
