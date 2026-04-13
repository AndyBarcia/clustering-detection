from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class RawOutputs:
    features: torch.Tensor          # [B, C, Hf, Wf]
    memory: torch.Tensor            # [B, HW, C]
    queries: torch.Tensor           # [L, B, Nq, C]
    intermediate_ttt_q: torch.Tensor
    mask_embs: torch.Tensor         # [L, B, Nq, C]
    cls_preds: torch.Tensor         # [L, B, Nq, K]
    img_shape: Tuple[int, int]
    sender_embs: Optional[torch.Tensor] = None     # [L, B, Nq, S]
    receiver_embs: Optional[torch.Tensor] = None   # [L, B, Nq, S]
    seed_logits: Optional[torch.Tensor] = None     # [L, B, Nq]
    seed_scores: Optional[torch.Tensor] = None     # [L, B, Nq]
    influence_preds: Optional[torch.Tensor] = None # [L, B, Nq]

    @property
    def sig_embs(self) -> Optional[torch.Tensor]:
        return self.receiver_embs


@dataclass
class FlatQueryOutputs:
    features: torch.Tensor                # [C, Hf, Wf]
    image_height: int
    image_width: int
    num_decoder_layers: int
    queries_per_layer: int
    mask_embeddings: torch.Tensor         # [Q, C]
    class_logits: torch.Tensor            # [Q, K]
    class_probabilities: torch.Tensor     # [Q, K]
    sender_embeddings: torch.Tensor       # [Q, S]
    receiver_embeddings: torch.Tensor     # [Q, S]
    seed_scores: torch.Tensor             # [Q]
    influence_scores: torch.Tensor        # [Q]
    background_confidence: torch.Tensor   # [Q]
    foreground_confidence: torch.Tensor   # [Q]
    partition_confidence: torch.Tensor    # [Q]
    predicted_labels: torch.Tensor        # [Q]

    @property
    def num_queries(self) -> int:
        return int(self.receiver_embeddings.shape[0])

    @property
    def signature_embeddings(self) -> torch.Tensor:
        return self.receiver_embeddings


@dataclass
class SeedSelection:
    indices: torch.Tensor                 # [Qs]
    scores: torch.Tensor                  # [Qs]
    effective_scores: torch.Tensor        # [Q]
    eligible_mask: torch.Tensor           # [Q]


@dataclass
class SeedClustering:
    selection: SeedSelection
    cluster_labels: torch.Tensor          # [Qs]
    cluster_members: list[torch.Tensor] = field(default_factory=list)  # list[[Qc_i]]


@dataclass
class PrototypeState:
    sender_embeddings: torch.Tensor       # [P, S]
    receiver_embeddings: torch.Tensor     # [P, S]
    class_logits: torch.Tensor            # [P, K]
    mask_embeddings: torch.Tensor         # [P, C]
    cluster_members: list[torch.Tensor]   # list[[Qc_i]]
    prototype_seed_indices: torch.Tensor  # [P]
    target_indices: Optional[torch.Tensor]  # [P]
    source_query_indices: torch.Tensor    # [Qs]
    source_cluster_labels: torch.Tensor   # [Qs]
    assignment_weights: torch.Tensor      # [Q, P]
    assignment_strength: torch.Tensor     # [P]

    @property
    def num_prototypes(self) -> int:
        return int(self.receiver_embeddings.shape[0])

    @property
    def signature_embeddings(self) -> torch.Tensor:
        return self.receiver_embeddings


@dataclass
class ResolvedPrediction:
    flat_queries: Optional[FlatQueryOutputs]
    prototypes: Optional[PrototypeState]
    kept_prototype_indices: torch.Tensor  # [Pk]
    resolved_target_indices: Optional[torch.Tensor]  # [Pr]
    sender_embeddings: torch.Tensor       # [Pk, S]
    receiver_embeddings: torch.Tensor     # [Pk, S]
    class_logits: torch.Tensor            # [Pk, K]
    class_probabilities: torch.Tensor     # [Pk, K]
    mask_embeddings: torch.Tensor         # [Pk, C]
    scores: torch.Tensor                  # [Pk]
    raw_mask_logits: torch.Tensor         # [Pk, H, W]
    raw_mask_probabilities: torch.Tensor  # [Pk, H, W]
    resolved_masks: list[torch.Tensor]    # list[[H, W]]
    resolved_labels: list[int]
    resolved_scores: list[float]

    @property
    def all_signature_embeddings(self) -> torch.Tensor:
        if self.prototypes is None:
            return self.receiver_embeddings
        return self.prototypes.receiver_embeddings

    @property
    def signature_embeddings(self) -> torch.Tensor:
        return self.receiver_embeddings


@dataclass
class GoldenQueryDiagnostics:
    matched_query_distances: list[float] = field(default_factory=list)
    unmatched_query_closest_gt_distances: list[float] = field(default_factory=list)


@dataclass
class EvaluationPredictionSet:
    clustering: ResolvedPrediction
    gt_signatures: Optional[ResolvedPrediction]
    golden_queries: Optional[ResolvedPrediction]
    golden_query_diagnostics: GoldenQueryDiagnostics = field(default_factory=GoldenQueryDiagnostics)
