from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


_LOG_2 = math.log(2.0)


def signatures_to_probabilities(signatures: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(signatures, dim=dim)


def pairwise_jsd(lhs: torch.Tensor, rhs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if lhs.shape[0] == 0 or rhs.shape[0] == 0:
        return torch.zeros((lhs.shape[0], rhs.shape[0]), dtype=torch.float32, device=lhs.device)

    lhs = lhs.clamp_min(eps)
    rhs = rhs.clamp_min(eps)

    mixture = 0.5 * (lhs[:, None, :] + rhs[None, :, :])
    lhs_log = torch.log(lhs[:, None, :])
    rhs_log = torch.log(rhs[None, :, :])
    mixture_log = torch.log(mixture)

    kl_lhs = (lhs[:, None, :] * (lhs_log - mixture_log)).sum(dim=-1)
    kl_rhs = (rhs[None, :, :] * (rhs_log - mixture_log)).sum(dim=-1)
    jsd = 0.5 * (kl_lhs + kl_rhs) / _LOG_2
    return jsd.clamp(0.0, 1.0)


def pairwise_similarity_from_jsd(lhs: torch.Tensor, rhs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 - pairwise_jsd(lhs, rhs, eps=eps)


def pairwise_jsd_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)

    probs = np.clip(x.astype(np.float64, copy=False), eps, None)
    probs = probs / probs.sum(axis=1, keepdims=True)

    mixture = 0.5 * (probs[:, None, :] + probs[None, :, :])
    kl_lhs = np.sum(probs[:, None, :] * (np.log(probs[:, None, :]) - np.log(mixture)), axis=-1)
    kl_rhs = np.sum(probs[None, :, :] * (np.log(probs[None, :, :]) - np.log(mixture)), axis=-1)
    jsd = 0.5 * (kl_lhs + kl_rhs) / _LOG_2
    return np.clip(jsd, 0.0, 1.0).astype(np.float64, copy=False)


def pairwise_affinity_from_jsd_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return np.clip(1.0 - pairwise_jsd_np(x, eps=eps), 0.0, 1.0)
