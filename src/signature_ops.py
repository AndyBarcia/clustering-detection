from __future__ import annotations

import torch
import torch.nn.functional as F


def _distance_to_similarity(
    distance: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    d_max = distance.max()
    if d_max <= eps:
        return torch.ones_like(distance)
    return 1.0 - (distance / d_max)


def _sigmoid_pairwise_intersection_stats(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lhs_probs = torch.sigmoid(lhs)
    rhs_probs = torch.sigmoid(rhs)

    intersection = torch.matmul(lhs_probs, rhs_probs.transpose(-1, -2))
    lhs_mass = lhs_probs.sum(dim=-1, keepdim=True)
    rhs_mass = rhs_probs.sum(dim=-1).unsqueeze(-2)
    return intersection, lhs_mass, rhs_mass


def _sigmoid_pairwise_jaccard(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    intersection, lhs_mass, rhs_mass = _sigmoid_pairwise_intersection_stats(lhs, rhs)
    union = (lhs_mass + rhs_mass - intersection).clamp_min(eps)
    jaccard = intersection / union

    # Scale and shift to map expected random similarity to 0.0 and perfect similarity to 1.0.
    return 1.5 * (jaccard - (1.0 / 3.0))


def _sigmoid_pairwise_dice(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    intersection, lhs_mass, rhs_mass = _sigmoid_pairwise_intersection_stats(lhs, rhs)
    denom = (lhs_mass + rhs_mass).clamp_min(eps)
    dice = (2.0 * intersection) / denom

    # Scale and shift to map expected random similarity to 0.0 and perfect similarity to 1.0.
    return 2.0 * (dice - 0.5)


def _sigmoid_pairwise_overlap(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eps: float,
    mass: str = "min",
) -> torch.Tensor:
    intersection, lhs_mass, rhs_mass = _sigmoid_pairwise_intersection_stats(lhs, rhs)
    if mass == "min":
        overlap_mass = torch.minimum(lhs_mass, rhs_mass)
    elif mass == "lhs":
        overlap_mass = lhs_mass
    elif mass == "rhs":
        overlap_mass = rhs_mass
    else:
        raise ValueError(f"Unsupported overlap mass selector: {mass}")

    overlap = intersection / overlap_mass.clamp_min(eps)

    # Scale and shift to map expected random similarity to 0.0 and perfect similarity to 1.0.
    return 2.0 * (overlap - 0.5)


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


def _pairwise_l2_similarity(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(lhs.unsqueeze(-2) - rhs.unsqueeze(-3), ord=2, dim=-1)
    return _distance_to_similarity(distance, eps=eps)


def _pairwise_mse_similarity(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    distance = (lhs.unsqueeze(-2) - rhs.unsqueeze(-3)).pow(2).mean(dim=-1)
    return _distance_to_similarity(distance, eps=eps)


def pairwise_similarity(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    metric: str = "dot",
    clamp: bool = True,
    eps: float = 1e-6,
    temp: float = 0.1,
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
        lhs = F.softmax(lhs/temp, dim=-1)
        rhs = F.softmax(rhs/temp, dim=-1)
    elif metric_name == "jsd":
        similarity = _softmax_pairwise_jsd(lhs/temp, rhs/temp, eps=eps)
        if clamp:
            similarity = similarity.clamp(0.0, 1.0)
        return similarity
    elif metric_name == "jaccard":
        similarity = _sigmoid_pairwise_jaccard(lhs/temp, rhs/temp, eps=eps)
        if clamp:
            similarity = similarity.clamp(0.0, 1.0)
        return similarity
    elif metric_name == "dice":
        similarity = _sigmoid_pairwise_dice(lhs/temp, rhs/temp, eps=eps)
        if clamp:
            similarity = similarity.clamp(0.0, 1.0)
        return similarity
    elif metric_name == "overlap":
        similarity = _sigmoid_pairwise_overlap(lhs/temp, rhs/temp, eps=eps, mass="min")
        if clamp:
            similarity = similarity.clamp(0.0, 1.0)
        return similarity
    elif metric_name == "left-overlap":
        similarity = _sigmoid_pairwise_overlap(lhs/temp, rhs/temp, eps=eps, mass="lhs")
        if clamp:
            similarity = similarity.clamp(0.0, 1.0)
        return similarity
    elif metric_name == "right-overlap":
        similarity = _sigmoid_pairwise_overlap(lhs/temp, rhs/temp, eps=eps, mass="rhs")
        if clamp:
            similarity = similarity.clamp(0.0, 1.0)
        return similarity
    elif metric_name == "l2":
        similarity = _pairwise_l2_similarity(lhs, rhs, eps=eps)
        if clamp:
            similarity = similarity.clamp(0.0, 1.0)
        return similarity
    elif metric_name == "mse":
        similarity = _pairwise_mse_similarity(lhs, rhs, eps=eps)
        if clamp:
            similarity = similarity.clamp(0.0, 1.0)
        return similarity

    if metric_name not in {
        "dot",
        "dot-sigmoid",
        "cosine",
        "softmax",
        "jsd",
        "jaccard",
        "dice",
        "overlap",
        "left-overlap",
        "right-overlap",
        "l2",
        "mse",
    }:
        raise ValueError(f"Unsupported signature similarity metric: {metric}")

    similarity = torch.matmul(lhs, rhs.transpose(-1, -2))
    if metric_name == "dot-sigmoid":
        similarity = torch.sigmoid(similarity/temp)
    if clamp:
        similarity = similarity.clamp(0.0, 1.0)
    return similarity


def pairwise_distance(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    metric: str = "dot",
    clamp: bool = True,
    temp: float = 0.1,
) -> torch.Tensor:
    if lhs.shape[0] == 0 or rhs.shape[0] == 0:
        return torch.zeros((lhs.shape[0], rhs.shape[0]), dtype=torch.float32, device=lhs.device)
    return 1.0 - pairwise_similarity(
        lhs,
        rhs,
        metric=metric,
        clamp=clamp,
        temp=temp,
    )
