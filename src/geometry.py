import numpy as np
import torch


def lift_to_hyperboloid(u: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    sq_norm = (u * u).sum(dim=-1, keepdim=True)
    time = torch.sqrt((1.0 + sq_norm).clamp_min(eps))
    return torch.cat([time, u], dim=-1)


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -(x[..., :1] * y[..., :1]).sum(dim=-1) + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def pairwise_lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    time = -x[..., :1] @ y[..., :1].transpose(-1, -2)
    space = x[..., 1:] @ y[..., 1:].transpose(-1, -2)
    return time + space


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    z = (-lorentz_inner(x, y)).clamp_min(1.0 + eps)
    return torch.acosh(z)


def pairwise_hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    z = (-pairwise_lorentz_inner(x, y)).clamp_min(1.0 + eps)
    return torch.acosh(z)


def signed_distance_score(
    distance: torch.Tensor,
    distance_radius: float,
    distance_scale: float,
    influence: torch.Tensor | None = None,
) -> torch.Tensor:
    score = distance_radius - distance
    if influence is not None:
        score = score + influence
    return score / max(distance_scale, 1e-6)


def signed_normalize(weights: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
    denom = weights.abs().sum(dim=dim, keepdim=True).clamp_min(eps)
    return weights / denom


def hyperbolic_distance_np(x: np.ndarray, y: np.ndarray | None = None, eps: float = 1e-6) -> np.ndarray:
    if y is None:
        y = x
    x_t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
    dist = pairwise_hyperbolic_distance(x_t, y_t, eps=eps)
    return np.ascontiguousarray(dist.cpu().numpy(), dtype=np.float64)
