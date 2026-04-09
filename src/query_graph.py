import torch


def compute_query_neighbor_weights(
    q_sig: torch.Tensor,
    min_similarity: float = 0.0,
) -> torch.Tensor:
    """Builds a query-query cosine graph without self-edges."""
    similarity = torch.matmul(q_sig, q_sig.transpose(-1, -2)).clamp(0.0, 1.0)

    if min_similarity > 0.0:
        similarity = similarity.masked_fill(similarity < min_similarity, 0.0)

    num_queries = similarity.shape[-1]
    eye = torch.eye(num_queries, device=similarity.device, dtype=torch.bool)
    while eye.dim() < similarity.dim():
        eye = eye.unsqueeze(0)
    return similarity.masked_fill(eye, 0.0)


def compute_excluded_seed_values(
    q_sig: torch.Tensor,
    seed_values: torch.Tensor,
    min_similarity: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the neighbor-weighted excluded seed value and the query graph weights.

    Supports `q_sig` with shape `[N, S]` or `[B, N, S]`, and `seed_values` with
    matching leading dimensions `[N]` or `[B, N]`.
    """
    squeezed = q_sig.dim() == 2
    if squeezed:
        q_sig = q_sig.unsqueeze(0)
        seed_values = seed_values.unsqueeze(0)

    weights = compute_query_neighbor_weights(q_sig, min_similarity=min_similarity)
    denom = weights.sum(dim=-1).clamp_min(eps)
    excluded = torch.matmul(weights, seed_values.unsqueeze(-1)).squeeze(-1) / denom

    if squeezed:
        return excluded[0], weights[0]
    return excluded, weights


def compute_local_maximum_margin(
    q_sig: torch.Tensor,
    seed_values: torch.Tensor,
    min_similarity: float = 0.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    excluded, weights = compute_excluded_seed_values(
        q_sig=q_sig,
        seed_values=seed_values,
        min_similarity=min_similarity,
        eps=eps,
    )
    margin = seed_values - excluded
    return margin, excluded, weights
