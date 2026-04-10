from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

# Optional clustering backends
try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

try:
    import hdbscan as _hdbscan
except ImportError:
    _hdbscan = None

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import igraph as ig
    import leidenalg
except ImportError:
    ig = None
    leidenalg = None


from .config import PrototypeInferenceConfig
from .model import CustomMask2Former, Mask2FormerBase
from .outputs import (
    EvaluationPredictionSet,
    FlatQueryOutputs,
    GoldenQueryDiagnostics,
    PrototypeState,
    RawOutputs,
    ResolvedPrediction,
    SeedClustering,
    SeedSelection,
)


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _alpha_value(alpha_obj) -> float:
    if isinstance(alpha_obj, torch.Tensor):
        return float(alpha_obj.detach().cpu().item())
    return float(alpha_obj)


def _assignment_affinity(similarity: torch.Tensor, influence: torch.Tensor, similarity_floor: float = 0.0):
    affinity = (similarity + influence.unsqueeze(1)).clamp(0.0, 1.0)
    if similarity_floor > 0.0:
        affinity = affinity.clamp_min(similarity_floor)
    return affinity


def _cosine_affinity_np(x: np.ndarray) -> np.ndarray:
    return np.clip(x @ x.T, 0.0, 1.0)


def _cosine_distance_np(x: np.ndarray) -> np.ndarray:
    dist = 1.0 - np.clip(x @ x.T, -1.0, 1.0)
    return np.ascontiguousarray(dist, dtype=np.float64)


def _pairwise_cosine_distance(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if lhs.shape[0] == 0 or rhs.shape[0] == 0:
        return torch.zeros((lhs.shape[0], rhs.shape[0]), dtype=torch.float32, device=lhs.device)
    similarity = torch.matmul(lhs, rhs.T).clamp(-1.0, 1.0)
    return 1.0 - similarity


def _connected_components_labels(affinity: np.ndarray, threshold: float) -> np.ndarray:
    n = affinity.shape[0]
    adj = affinity >= threshold
    labels = -np.ones(n, dtype=np.int64)
    cid = 0

    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            u = stack.pop()
            nbrs = np.where(adj[u])[0]
            for v in nbrs:
                if labels[v] == -1:
                    labels[v] = cid
                    stack.append(v)
        cid += 1
    return labels


def _build_weighted_graph_edges(affinity: np.ndarray, min_edge_weight: float):
    n = affinity.shape[0]
    edges, weights = [], []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(affinity[i, j])
            if w >= min_edge_weight:
                edges.append((i, j))
                weights.append(w)
    return edges, weights


def _binary_dilate(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask.to(dtype=torch.bool)
    pad = kernel_size // 2
    pooled = F.max_pool2d(mask.float().unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=pad)
    return pooled[0, 0] > 0


def _binary_erode(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return mask.to(dtype=torch.bool)
    return ~_binary_dilate(~mask.to(dtype=torch.bool), kernel_size)


class ModularPrototypePredictor:
    def __init__(self, cfg: PrototypeInferenceConfig):
        self.cfg = cfg

    def _apply_morphology(self, mask: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg.overlap
        op = cfg.morphology_op.lower()
        kernel_size = int(cfg.morphology_kernel_size)
        iterations = max(1, int(cfg.morphology_iterations))

        if op == "none" or kernel_size <= 1:
            return mask.to(dtype=torch.bool)
        if kernel_size % 2 == 0:
            raise ValueError("overlap.morphology_kernel_size must be odd when morphology is enabled.")

        out = mask.to(dtype=torch.bool)

        for _ in range(iterations):
            if op == "opening":
                out = _binary_dilate(_binary_erode(out, kernel_size), kernel_size)
            elif op == "closing":
                out = _binary_erode(_binary_dilate(out, kernel_size), kernel_size)
            elif op == "open_close":
                out = _binary_erode(_binary_dilate(_binary_dilate(_binary_erode(out, kernel_size), kernel_size), kernel_size), kernel_size)
            elif op == "close_open":
                out = _binary_dilate(_binary_erode(_binary_erode(_binary_dilate(out, kernel_size), kernel_size), kernel_size), kernel_size)
            else:
                raise ValueError(
                    "overlap.morphology_op must be one of "
                    "{'none', 'opening', 'closing', 'open_close', 'close_open'}."
                )

        return out

    @torch.no_grad()
    def predict(self, model: CustomMask2Former, images: torch.Tensor):
        raw = model(images)
        return self.predict_from_raw(model, raw)

    @torch.no_grad()
    def predict_from_raw(self, model: CustomMask2Former, raw: RawOutputs):
        batch_predictions = [self.predict_clustered_single(model, raw, batch_index) for batch_index in range(raw.features.shape[0])]
        return batch_predictions[0] if len(batch_predictions) == 1 else batch_predictions

    @torch.no_grad()
    def predict_from_raw_with_gt_prototypes(self, model: CustomMask2Former, raw: RawOutputs, targets):
        batch_predictions = [self.predict_with_gt_signatures_single(model, raw, targets, batch_index) for batch_index in range(raw.features.shape[0])]
        return batch_predictions[0] if len(batch_predictions) == 1 else batch_predictions

    @torch.no_grad()
    def predict_evaluation_views_from_raw(self, model: CustomMask2Former, raw: RawOutputs, targets):
        batch_predictions = [self.predict_evaluation_views_single(model, raw, targets, batch_index) for batch_index in range(raw.features.shape[0])]
        return batch_predictions[0] if len(batch_predictions) == 1 else batch_predictions

    def flatten_outputs(self, raw: RawOutputs, batch_index: int) -> FlatQueryOutputs:
        features = raw.features[batch_index]
        image_height, image_width = raw.img_shape

        query_mask_embeddings = raw.mask_embs[:, batch_index]
        query_class_logits = raw.cls_preds[:, batch_index]
        query_signature_embeddings = raw.sig_embs[:, batch_index]
        query_seed_scores = raw.seed_scores[:, batch_index]
        query_influence_scores = raw.influence_preds[:, batch_index]

        num_layers, queries_per_layer, signature_dim = query_signature_embeddings.shape
        num_classes = query_class_logits.shape[-1]

        query_mask_embeddings = query_mask_embeddings.reshape(num_layers * queries_per_layer, -1)
        query_class_logits = query_class_logits.reshape(num_layers * queries_per_layer, num_classes)
        query_signature_embeddings = query_signature_embeddings.reshape(num_layers * queries_per_layer, signature_dim)
        query_seed_scores = query_seed_scores.reshape(num_layers * queries_per_layer)
        query_influence_scores = query_influence_scores.reshape(num_layers * queries_per_layer)

        query_class_probabilities = F.softmax(query_class_logits, dim=-1)
        predicted_labels = query_class_probabilities.argmax(dim=-1)
        background_confidence = query_class_probabilities[:, 0]
        foreground_confidence = 1.0 - query_class_probabilities[:, 0]
        partition_confidence = torch.where(predicted_labels == 0, background_confidence, foreground_confidence)

        return FlatQueryOutputs(
            features=features,
            image_height=image_height,
            image_width=image_width,
            num_decoder_layers=num_layers,
            queries_per_layer=queries_per_layer,
            mask_embeddings=query_mask_embeddings,
            class_logits=query_class_logits,
            class_probabilities=query_class_probabilities,
            signature_embeddings=query_signature_embeddings,
            seed_scores=query_seed_scores,
            influence_scores=query_influence_scores,
            background_confidence=background_confidence,
            foreground_confidence=foreground_confidence,
            partition_confidence=partition_confidence,
            predicted_labels=predicted_labels,
        )

    def select_seed_queries(self, flat_queries: FlatQueryOutputs) -> SeedSelection:
        cfg = self.cfg.seed
        effective_scores = flat_queries.seed_scores.clone()

        if cfg.use_foreground_in_score:
            effective_scores = effective_scores * flat_queries.partition_confidence.pow(cfg.foreground_score_power)

        eligible_mask = torch.ones_like(effective_scores, dtype=torch.bool)

        if cfg.exclude_background:
            eligible_mask &= flat_queries.predicted_labels != 0

        if cfg.min_foreground_prob > 0:
            eligible_mask &= flat_queries.partition_confidence >= cfg.min_foreground_prob

        if cfg.max_influence is not None:
            eligible_mask &= flat_queries.influence_scores <= cfg.max_influence

        selected_mask = eligible_mask.clone()
        selected_mask &= flat_queries.seed_scores >= cfg.quality_threshold
        selected_indices = torch.where(selected_mask)[0]

        if selected_indices.numel() < cfg.min_num_seeds:
            fallback_indices = torch.where(eligible_mask)[0]
            if fallback_indices.numel() > 0:
                k = min(cfg.min_num_seeds, fallback_indices.numel())
                top_local = torch.topk(effective_scores[fallback_indices], k=k).indices
                selected_indices = fallback_indices[top_local]

        if cfg.topk is not None and selected_indices.numel() > cfg.topk:
            top_local = torch.topk(effective_scores[selected_indices], k=cfg.topk).indices
            selected_indices = selected_indices[top_local]

        return SeedSelection(
            indices=selected_indices,
            scores=effective_scores[selected_indices],
            effective_scores=effective_scores,
            eligible_mask=eligible_mask,
        )

    def _cluster_local(self, seed_sigs_np: np.ndarray, seed_scores_np: np.ndarray) -> np.ndarray:
        cfg = self.cfg.cluster
        method = cfg.method.lower()

        if len(seed_sigs_np) == 0:
            return np.empty((0,), dtype=np.int64)

        if len(seed_sigs_np) == 1:
            return np.array([0], dtype=np.int64)

        if method == "dbscan":
            if DBSCAN is None:
                raise ImportError("scikit-learn is not installed. pip install scikit-learn")
            clusterer = DBSCAN(
                eps=cfg.dbscan_eps,
                min_samples=cfg.dbscan_min_samples,
                metric="cosine",
            )
            if cfg.dbscan_use_sample_weight:
                clusterer.fit(seed_sigs_np, sample_weight=seed_scores_np)
            else:
                clusterer.fit(seed_sigs_np)
            return clusterer.labels_.astype(np.int64)

        if method == "hdbscan":
            if _hdbscan is None:
                raise ImportError("hdbscan is not installed. pip install hdbscan")
            dist = _cosine_distance_np(seed_sigs_np)
            clusterer = _hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=cfg.hdbscan_min_cluster_size,
                min_samples=cfg.hdbscan_min_samples,
                cluster_selection_epsilon=cfg.hdbscan_cluster_selection_epsilon,
            )
            return clusterer.fit_predict(dist).astype(np.int64)

        affinity = _cosine_affinity_np(seed_sigs_np)

        if method == "cc":
            return _connected_components_labels(affinity, cfg.graph_affinity_threshold)

        if method == "louvain":
            if nx is None:
                raise ImportError("networkx is required for Louvain clustering")
            if not hasattr(nx.algorithms.community, "louvain_communities"):
                raise ImportError("This version of networkx does not expose louvain_communities")
            graph = nx.Graph()
            num_nodes = len(seed_sigs_np)
            graph.add_nodes_from(range(num_nodes))
            edges, weights = _build_weighted_graph_edges(affinity, cfg.graph_min_edge_weight)
            for (node_u, node_v), weight in zip(edges, weights):
                graph.add_edge(node_u, node_v, weight=weight)
            if graph.number_of_edges() == 0:
                return np.arange(num_nodes, dtype=np.int64)
            communities = nx.algorithms.community.louvain_communities(
                graph,
                weight="weight",
                resolution=cfg.louvain_resolution,
                seed=cfg.random_seed,
            )
            labels = -np.ones(num_nodes, dtype=np.int64)
            for cluster_id, community in enumerate(communities):
                for node in community:
                    labels[node] = cluster_id
            return labels

        if method == "leiden":
            if ig is None or leidenalg is None:
                raise ImportError("igraph + leidenalg are required for Leiden clustering")
            num_nodes = len(seed_sigs_np)
            edges, weights = _build_weighted_graph_edges(affinity, cfg.graph_min_edge_weight)
            if len(edges) == 0:
                return np.arange(num_nodes, dtype=np.int64)
            graph = ig.Graph(n=num_nodes, edges=edges, directed=False)
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                resolution_parameter=cfg.leiden_resolution,
                seed=cfg.random_seed,
            )
            return np.asarray(partition.membership, dtype=np.int64)

        raise ValueError(f"Unknown clustering method: {cfg.method}")

    def cluster_seed_queries(self, flat_queries: FlatQueryOutputs, selection: SeedSelection) -> SeedClustering:
        cfg = self.cfg.cluster
        device = selection.indices.device

        if selection.indices.numel() == 0:
            return SeedClustering(selection=selection, cluster_labels=torch.empty((0,), dtype=torch.long, device=device))

        cluster_labels = -torch.ones(selection.indices.numel(), dtype=torch.long, device=device)
        cluster_members: list[torch.Tensor] = []
        next_cluster_id = 0

        if cfg.cluster_per_class:
            seed_classes = flat_queries.predicted_labels[selection.indices]
            groups = [torch.where(seed_classes == class_id)[0] for class_id in seed_classes.unique().tolist()]
        else:
            groups = [torch.arange(selection.indices.numel(), device=device)]

        for position_group in groups:
            local_seed_indices = selection.indices[position_group]
            local_seed_scores = selection.scores[position_group]

            local_signatures_np = flat_queries.signature_embeddings[local_seed_indices].detach().cpu().numpy()
            local_scores_np = local_seed_scores.detach().cpu().numpy()

            local_labels_np = self._cluster_local(local_signatures_np, local_scores_np)
            local_labels = torch.as_tensor(local_labels_np, device=device)

            valid_cluster_ids = [int(value) for value in np.unique(local_labels_np) if value != -1]
            for old_cluster_id in valid_cluster_ids:
                member_positions = position_group[local_labels == old_cluster_id]
                cluster_labels[member_positions] = next_cluster_id
                cluster_members.append(selection.indices[member_positions])
                next_cluster_id += 1

            if cfg.promote_noise_to_singletons:
                noise_positions = position_group[local_labels == -1]
                for position in noise_positions:
                    cluster_labels[position] = next_cluster_id
                    cluster_members.append(selection.indices[position].unsqueeze(0))
                    next_cluster_id += 1

        return SeedClustering(
            selection=selection,
            cluster_labels=cluster_labels,
            cluster_members=cluster_members,
        )

    def _empty_prototype_state(self, flat_queries: FlatQueryOutputs, *, source_query_indices: Optional[torch.Tensor] = None, source_cluster_labels: Optional[torch.Tensor] = None) -> PrototypeState:
        device = flat_queries.signature_embeddings.device
        if source_query_indices is None:
            source_query_indices = torch.empty((0,), dtype=torch.long, device=device)
        if source_cluster_labels is None:
            source_cluster_labels = torch.empty((0,), dtype=torch.long, device=device)

        return PrototypeState(
            signature_embeddings=torch.empty((0, flat_queries.signature_embeddings.shape[-1]), device=device),
            class_logits=torch.empty((0, flat_queries.class_logits.shape[-1]), device=device),
            mask_embeddings=torch.empty((0, flat_queries.mask_embeddings.shape[-1]), device=device),
            cluster_members=[],
            prototype_seed_indices=torch.empty((0,), dtype=torch.long, device=device),
            target_indices=None,
            source_query_indices=source_query_indices,
            source_cluster_labels=source_cluster_labels,
            assignment_weights=torch.empty((flat_queries.num_queries, 0), device=device),
            assignment_strength=torch.empty((0,), device=device),
        )

    def initialize_clustered_prototypes(self, flat_queries: FlatQueryOutputs, clustering: SeedClustering) -> PrototypeState:
        device = flat_queries.signature_embeddings.device
        valid = clustering.cluster_labels >= 0

        if valid.sum() == 0:
            return self._empty_prototype_state(
                flat_queries,
                source_query_indices=clustering.selection.indices,
                source_cluster_labels=clustering.cluster_labels,
            )

        prototype_signatures = []
        prototype_logits = []
        prototype_mask_embeddings = []
        prototype_seed_indices = []

        for member_query_indices in clustering.cluster_members:
            representative_local = torch.argmax(flat_queries.seed_scores[member_query_indices])
            representative_query_index = member_query_indices[representative_local]
            prototype_seed_indices.append(representative_query_index)

            weights = flat_queries.seed_scores[member_query_indices]
            weights = weights / (weights.sum() + 1e-6)

            prototype_logits.append((flat_queries.class_logits[member_query_indices] * weights.unsqueeze(1)).sum(dim=0))
            prototype_mask_embeddings.append((flat_queries.mask_embeddings[member_query_indices] * weights.unsqueeze(1)).sum(dim=0))
            prototype_signatures.append(flat_queries.signature_embeddings[representative_query_index])

        return PrototypeState(
            signature_embeddings=torch.stack(prototype_signatures, dim=0),
            class_logits=torch.stack(prototype_logits, dim=0),
            mask_embeddings=torch.stack(prototype_mask_embeddings, dim=0),
            cluster_members=clustering.cluster_members,
            prototype_seed_indices=torch.stack(prototype_seed_indices, dim=0),
            target_indices=None,
            source_query_indices=clustering.selection.indices,
            source_cluster_labels=clustering.cluster_labels,
            assignment_weights=torch.empty((flat_queries.num_queries, 0), device=device),
            assignment_strength=torch.empty((0,), device=device),
        )

    def refine_prototypes(self, model: CustomMask2Former, flat_queries: FlatQueryOutputs, prototypes: PrototypeState) -> PrototypeState:
        cfg = self.cfg.assign
        device = flat_queries.signature_embeddings.device

        if prototypes.num_prototypes == 0:
            return prototypes

        if cfg.use_all_queries:
            source_query_indices = torch.arange(flat_queries.num_queries, device=device)
        else:
            source_query_indices = prototypes.source_query_indices

        query_signatures = flat_queries.signature_embeddings[source_query_indices]
        query_logits = flat_queries.class_logits[source_query_indices]
        query_probabilities = flat_queries.class_probabilities[source_query_indices]
        query_mask_embeddings = flat_queries.mask_embeddings[source_query_indices]

        prototype_signatures = prototypes.signature_embeddings
        prototype_logits = prototypes.class_logits

        alpha = _alpha_value(model.alpha_focal) if cfg.use_alpha_focal else 1.0
        final_raw_weights = None
        final_normalized_weights = None

        for _ in range(max(1, cfg.refinement_steps)):
            similarity = torch.matmul(query_signatures, prototype_signatures.T)
            affinity = _assignment_affinity(similarity, flat_queries.influence_scores[source_query_indices], cfg.similarity_floor)
            raw_weights = affinity.pow(alpha)

            if cfg.use_query_quality:
                raw_weights = raw_weights * flat_queries.seed_scores[source_query_indices].pow(cfg.query_quality_power).unsqueeze(1)

            prototype_probabilities = None
            if cfg.use_foreground_prob or cfg.class_compat_power > 0:
                prototype_probabilities = F.softmax(prototype_logits, dim=-1)

            if cfg.use_foreground_prob:
                query_fg = flat_queries.foreground_confidence[source_query_indices]
                query_bg = flat_queries.background_confidence[source_query_indices]
                prototype_bg = prototype_probabilities[:, 0]
                prototype_fg = 1.0 - prototype_bg
                partition_compatibility = query_fg[:, None] * prototype_fg[None, :] + query_bg[:, None] * prototype_bg[None, :]
                raw_weights = raw_weights * partition_compatibility.clamp_min(1e-6).pow(cfg.foreground_prob_power)

            if cfg.class_compat_power > 0:
                class_compatibility = torch.matmul(query_probabilities, prototype_probabilities.T).clamp_min(1e-6)
                raw_weights = raw_weights * class_compatibility.pow(cfg.class_compat_power)

            if cfg.normalize_over_queries:
                normalized_weights = raw_weights / (raw_weights.sum(dim=0, keepdim=True) + 1e-6)
            else:
                normalized_weights = raw_weights / (raw_weights.sum(dim=1, keepdim=True) + 1e-6)

            prototype_logits = torch.matmul(normalized_weights.T, query_logits)
            prototype_mask_embeddings = torch.matmul(normalized_weights.T, query_mask_embeddings)

            final_raw_weights = raw_weights
            final_normalized_weights = normalized_weights

        assignment_weights = torch.zeros(
            (flat_queries.num_queries, prototypes.num_prototypes),
            device=device,
            dtype=prototype_signatures.dtype,
        )
        assignment_weights[source_query_indices] = final_normalized_weights

        return PrototypeState(
            signature_embeddings=prototype_signatures,
            class_logits=prototype_logits,
            mask_embeddings=prototype_mask_embeddings,
            cluster_members=prototypes.cluster_members,
            prototype_seed_indices=prototypes.prototype_seed_indices,
            target_indices=prototypes.target_indices,
            source_query_indices=prototypes.source_query_indices,
            source_cluster_labels=prototypes.source_cluster_labels,
            assignment_weights=assignment_weights,
            assignment_strength=final_raw_weights.sum(dim=0),
        )

    def build_signature_prototypes(
        self,
        model: CustomMask2Former,
        flat_queries: FlatQueryOutputs,
        signature_embeddings: torch.Tensor,
        *,
        prototype_seed_indices: Optional[torch.Tensor] = None,
        target_indices: Optional[torch.Tensor] = None,
    ) -> PrototypeState:
        device = flat_queries.signature_embeddings.device
        num_prototypes = int(signature_embeddings.shape[0])
        if num_prototypes == 0:
            return self._empty_prototype_state(flat_queries)

        alpha = _alpha_value(model.alpha_focal) if self.cfg.assign.use_alpha_focal else 1.0
        similarity = torch.matmul(flat_queries.signature_embeddings, signature_embeddings.T)
        affinity = _assignment_affinity(similarity, flat_queries.influence_scores, self.cfg.assign.similarity_floor)
        raw_weights = affinity.pow(alpha)
        normalized_weights = raw_weights / (raw_weights.sum(dim=0, keepdim=True) + 1e-6)

        prototype_mask_embeddings = torch.matmul(normalized_weights.T, flat_queries.mask_embeddings)
        prototype_logits = torch.matmul(normalized_weights.T, flat_queries.class_logits)

        if prototype_seed_indices is None:
            prototype_seed_indices = torch.full((num_prototypes,), -1, dtype=torch.long, device=device)
            cluster_members: list[torch.Tensor] = []
        else:
            cluster_members = [query_index.unsqueeze(0) for query_index in prototype_seed_indices]

        return PrototypeState(
            signature_embeddings=signature_embeddings,
            class_logits=prototype_logits,
            mask_embeddings=prototype_mask_embeddings,
            cluster_members=cluster_members,
            prototype_seed_indices=prototype_seed_indices,
            target_indices=target_indices,
            source_query_indices=torch.arange(flat_queries.num_queries, device=device),
            source_cluster_labels=torch.arange(num_prototypes, device=device),
            assignment_weights=normalized_weights,
            assignment_strength=raw_weights.sum(dim=0),
        )

    def encode_gt_signatures(
        self,
        model: CustomMask2Former,
        raw: RawOutputs,
        targets,
        batch_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        labels = targets[batch_index]["labels"].to(device)
        masks = targets[batch_index]["masks"].to(device).float()

        if labels.numel() == 0:
            return torch.empty((0, raw.sig_embs.shape[-1]), device=device)

        gt_masks = masks.unsqueeze(0)
        gt_labels = labels.unsqueeze(0)
        gt_pad_mask = torch.ones((1, labels.shape[0]), dtype=torch.bool, device=device)

        return model.encode_gts(
            raw.memory[batch_index:batch_index + 1],
            raw.features[batch_index:batch_index + 1],
            gt_masks,
            gt_labels,
            gt_pad_mask,
            ttt_steps_override=self.cfg.ttt_steps,
        )[0]

    def build_gt_signature_prototypes(
        self,
        model: CustomMask2Former,
        raw: RawOutputs,
        flat_queries: FlatQueryOutputs,
        targets,
        batch_index: int,
    ) -> PrototypeState:
        device = flat_queries.signature_embeddings.device
        labels = targets[batch_index]["labels"].to(device)

        if labels.numel() == 0:
            return self._empty_prototype_state(flat_queries)

        gt_signatures = self.encode_gt_signatures(model, raw, targets, batch_index, device)

        return self.build_signature_prototypes(
            model,
            flat_queries,
            gt_signatures,
            target_indices=torch.arange(labels.shape[0], device=device, dtype=torch.long),
        )

    def resolve_prediction(self, flat_queries: FlatQueryOutputs, prototypes: PrototypeState) -> ResolvedPrediction:
        cfg = self.cfg.overlap
        features = flat_queries.features
        image_height, image_width = flat_queries.image_height, flat_queries.image_width

        if prototypes.num_prototypes == 0:
            empty_mask_grid = torch.empty((0, image_height, image_width), device=features.device)
            return ResolvedPrediction(
                flat_queries=flat_queries,
                prototypes=prototypes,
                kept_prototype_indices=torch.empty((0,), dtype=torch.long, device=features.device),
                resolved_target_indices=None,
                signature_embeddings=torch.empty((0, flat_queries.signature_embeddings.shape[-1]), device=features.device),
                class_logits=torch.empty((0, flat_queries.class_logits.shape[-1]), device=features.device),
                class_probabilities=torch.empty((0, flat_queries.class_logits.shape[-1]), device=features.device),
                mask_embeddings=torch.empty((0, flat_queries.mask_embeddings.shape[-1]), device=features.device),
                scores=torch.empty((0,), device=features.device),
                raw_mask_logits=empty_mask_grid,
                raw_mask_probabilities=empty_mask_grid,
                resolved_masks=[],
                resolved_labels=[],
                resolved_scores=[],
            )

        mask_logits = torch.einsum("pc,chw->phw", prototypes.mask_embeddings, features)
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0),
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        mask_probabilities = F.softmax(mask_logits, dim=0)

        class_probabilities = F.softmax(prototypes.class_logits, dim=-1)
        predicted_labels = class_probabilities.argmax(dim=-1)
        class_confidence = class_probabilities.max(dim=-1).values
        foreground_confidence = 1.0 - class_probabilities[:, 0]

        prototype_scores = torch.ones_like(class_confidence)
        if cfg.use_class_confidence:
            prototype_scores = prototype_scores * class_confidence
        if cfg.use_foreground_confidence:
            prototype_scores = prototype_scores * foreground_confidence
        if cfg.use_assignment_strength and prototypes.assignment_strength.numel() > 0:
            normalized_strength = prototypes.assignment_strength / (prototypes.assignment_strength.max() + 1e-6)
            prototype_scores = prototype_scores * normalized_strength.pow(cfg.assignment_strength_power)

        keep_mask = torch.ones_like(prototype_scores, dtype=torch.bool)
        if cfg.remove_background:
            keep_mask &= predicted_labels != 0
        keep_mask &= prototype_scores >= cfg.min_prototype_score

        if keep_mask.sum() == 0:
            empty_mask_grid = torch.empty((0, image_height, image_width), device=features.device)
            return ResolvedPrediction(
                flat_queries=flat_queries,
                prototypes=prototypes,
                kept_prototype_indices=torch.empty((0,), dtype=torch.long, device=features.device),
                resolved_target_indices=None,
                signature_embeddings=torch.empty((0, flat_queries.signature_embeddings.shape[-1]), device=features.device),
                class_logits=torch.empty((0, flat_queries.class_logits.shape[-1]), device=features.device),
                class_probabilities=torch.empty((0, flat_queries.class_logits.shape[-1]), device=features.device),
                mask_embeddings=torch.empty((0, flat_queries.mask_embeddings.shape[-1]), device=features.device),
                scores=torch.empty((0,), device=features.device),
                raw_mask_logits=empty_mask_grid,
                raw_mask_probabilities=empty_mask_grid,
                resolved_masks=[],
                resolved_labels=[],
                resolved_scores=[],
            )

        kept_prototype_indices = torch.where(keep_mask)[0]
        kept_mask_probabilities = mask_probabilities[keep_mask]
        kept_mask_logits = mask_logits[keep_mask]
        kept_class_probabilities = class_probabilities[keep_mask]
        kept_predicted_labels = predicted_labels[keep_mask]
        kept_scores = prototype_scores[keep_mask]
        kept_signatures = prototypes.signature_embeddings[keep_mask]
        kept_logits = prototypes.class_logits[keep_mask]
        kept_mask_embeddings = prototypes.mask_embeddings[keep_mask]
        kept_target_indices = None if prototypes.target_indices is None else prototypes.target_indices[keep_mask]

        pixel_scores = mask_probabilities * prototype_scores[:, None, None]
        max_pixel_score, winners = pixel_scores.max(dim=0)

        resolved_kept_positions = []
        resolved_masks = []
        resolved_labels = []
        resolved_scores = []
        resolved_target_indices = []

        for kept_position, prototype_index in enumerate(kept_prototype_indices.tolist()):
            mask = (winners == prototype_index) & (max_pixel_score >= cfg.pixel_score_threshold)
            mask &= mask_probabilities[prototype_index] >= cfg.mask_threshold
            mask = self._apply_morphology(mask)

            if mask.sum().item() < cfg.min_area:
                continue

            resolved_kept_positions.append(kept_position)
            resolved_masks.append(mask)
            resolved_labels.append(int(kept_predicted_labels[kept_position].item()))
            resolved_scores.append(float(kept_scores[kept_position].item()))
            if kept_target_indices is not None:
                resolved_target_indices.append(int(kept_target_indices[kept_position].item()))

        if resolved_kept_positions:
            resolved_index_tensor = torch.as_tensor(
                resolved_kept_positions,
                device=features.device,
                dtype=torch.long,
            )
            kept_prototype_indices = kept_prototype_indices[resolved_index_tensor]
            kept_signatures = kept_signatures[resolved_index_tensor]
            kept_logits = kept_logits[resolved_index_tensor]
            kept_class_probabilities = kept_class_probabilities[resolved_index_tensor]
            kept_mask_embeddings = kept_mask_embeddings[resolved_index_tensor]
            kept_scores = kept_scores[resolved_index_tensor]
            kept_mask_logits = kept_mask_logits[resolved_index_tensor]
            kept_mask_probabilities = kept_mask_probabilities[resolved_index_tensor]
        else:
            kept_prototype_indices = kept_prototype_indices[:0]
            kept_signatures = kept_signatures[:0]
            kept_logits = kept_logits[:0]
            kept_class_probabilities = kept_class_probabilities[:0]
            kept_mask_embeddings = kept_mask_embeddings[:0]
            kept_scores = kept_scores[:0]
            kept_mask_logits = kept_mask_logits[:0]
            kept_mask_probabilities = kept_mask_probabilities[:0]

        return ResolvedPrediction(
            flat_queries=flat_queries,
            prototypes=prototypes,
            kept_prototype_indices=kept_prototype_indices,
            resolved_target_indices=None if kept_target_indices is None else torch.as_tensor(
                resolved_target_indices,
                device=features.device,
                dtype=torch.long,
            ),
            signature_embeddings=kept_signatures,
            class_logits=kept_logits,
            class_probabilities=kept_class_probabilities,
            mask_embeddings=kept_mask_embeddings,
            scores=kept_scores,
            raw_mask_logits=kept_mask_logits,
            raw_mask_probabilities=kept_mask_probabilities,
            resolved_masks=resolved_masks,
            resolved_labels=resolved_labels,
            resolved_scores=resolved_scores,
        )

    def predict_clustered_single(self, model: CustomMask2Former, raw: RawOutputs, batch_index: int) -> ResolvedPrediction:
        flat_queries = self.flatten_outputs(raw, batch_index)
        seed_selection = self.select_seed_queries(flat_queries)
        seed_clustering = self.cluster_seed_queries(flat_queries, seed_selection)
        prototypes = self.initialize_clustered_prototypes(flat_queries, seed_clustering)
        prototypes = self.refine_prototypes(model, flat_queries, prototypes)
        return self.resolve_prediction(flat_queries, prototypes)

    def predict_with_gt_signatures_single(self, model: CustomMask2Former, raw: RawOutputs, targets, batch_index: int) -> ResolvedPrediction:
        flat_queries = self.flatten_outputs(raw, batch_index)
        prototypes = self.build_gt_signature_prototypes(model, raw, flat_queries, targets, batch_index)
        return self.resolve_prediction(flat_queries, prototypes)

    def predict_with_golden_queries_single(
        self,
        model: CustomMask2Former,
        raw: RawOutputs,
        targets,
        batch_index: int,
    ) -> tuple[ResolvedPrediction, GoldenQueryDiagnostics]:
        flat_queries = self.flatten_outputs(raw, batch_index)
        gt_signatures = self.encode_gt_signatures(
            model,
            raw,
            targets,
            batch_index,
            flat_queries.signature_embeddings.device,
        )

        candidate_indices = torch.arange(flat_queries.num_queries, device=flat_queries.signature_embeddings.device)
        if gt_signatures.shape[0] == 0 or candidate_indices.numel() == 0:
            empty_prediction = self.resolve_prediction(flat_queries, self._empty_prototype_state(flat_queries))
            return empty_prediction, GoldenQueryDiagnostics()

        query_signatures = flat_queries.signature_embeddings[candidate_indices]
        distances = _pairwise_cosine_distance(query_signatures, gt_signatures)

        num_queries, num_gt = distances.shape
        size = max(num_queries, num_gt)
        cost = np.full((size, size), 1.0, dtype=np.float64)
        cost[:num_queries, :num_gt] = distances.detach().cpu().numpy()

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_local_query_indices = []
        matched_target_indices = []
        matched_distances = []
        for query_index, gt_index in zip(row_ind.tolist(), col_ind.tolist()):
            if query_index >= num_queries or gt_index >= num_gt:
                continue
            matched_local_query_indices.append(query_index)
            matched_target_indices.append(gt_index)
            matched_distances.append(float(distances[query_index, gt_index].item()))

        matched_mask = torch.zeros(candidate_indices.numel(), dtype=torch.bool, device=candidate_indices.device)
        if matched_local_query_indices:
            matched_mask[torch.as_tensor(matched_local_query_indices, device=candidate_indices.device)] = True

        matched_query_indices = candidate_indices[matched_mask]
        matched_gt_indices = (
            torch.as_tensor(matched_target_indices, device=candidate_indices.device, dtype=torch.long)
            if matched_target_indices
            else torch.empty((0,), device=candidate_indices.device, dtype=torch.long)
        )
        unmatched_query_distances = []
        if (~matched_mask).any():
            unmatched_query_distances = distances[~matched_mask].min(dim=1).values.detach().cpu().tolist()

        matched_query_signatures = flat_queries.signature_embeddings[matched_query_indices]
        prototypes = self.build_signature_prototypes(
            model,
            flat_queries,
            matched_query_signatures,
            prototype_seed_indices=matched_query_indices,
            target_indices=matched_gt_indices,
        )
        prediction = self.resolve_prediction(flat_queries, prototypes)
        diagnostics = GoldenQueryDiagnostics(
            matched_query_distances=matched_distances,
            unmatched_query_closest_gt_distances=unmatched_query_distances,
        )
        return prediction, diagnostics

    def predict_evaluation_views_single(self, model: CustomMask2Former, raw: RawOutputs, targets, batch_index: int) -> EvaluationPredictionSet:
        clustering_prediction = self.predict_clustered_single(model, raw, batch_index)
        gt_signature_prediction = self.predict_with_gt_signatures_single(model, raw, targets, batch_index)
        golden_query_prediction, golden_query_diagnostics = self.predict_with_golden_queries_single(model, raw, targets, batch_index)
        return EvaluationPredictionSet(
            clustering=clustering_prediction,
            gt_signatures=gt_signature_prediction,
            golden_queries=golden_query_prediction,
            golden_query_diagnostics=golden_query_diagnostics,
        )


class StandardMask2FormerPredictor:
    def __init__(self, cfg: PrototypeInferenceConfig):
        self.cfg = cfg

    @torch.no_grad()
    def predict(self, model: Mask2FormerBase, images: torch.Tensor):
        raw = model(images)
        return self.predict_from_raw(model, raw)

    @torch.no_grad()
    def predict_from_raw(self, model: Mask2FormerBase, raw: RawOutputs):
        batch_predictions = [self._predict_single(model, raw, batch_index) for batch_index in range(raw.features.shape[0])]
        return batch_predictions[0] if len(batch_predictions) == 1 else batch_predictions

    @torch.no_grad()
    def predict_from_raw_with_gt_prototypes(self, model: Mask2FormerBase, raw: RawOutputs, targets):
        del targets
        return self.predict_from_raw(model, raw)

    @torch.no_grad()
    def predict_evaluation_views_from_raw(self, model: Mask2FormerBase, raw: RawOutputs, targets):
        del targets
        batch_predictions = [
            EvaluationPredictionSet(
                clustering=self._predict_single(model, raw, batch_index),
                gt_signatures=None,
                golden_queries=None,
            )
            for batch_index in range(raw.features.shape[0])
        ]
        return batch_predictions[0] if len(batch_predictions) == 1 else batch_predictions

    def _flatten_outputs(self, raw: RawOutputs, batch_index: int) -> FlatQueryOutputs:
        query_mask_embeddings = raw.mask_embs[-1, batch_index]
        query_class_logits = raw.cls_preds[-1, batch_index]
        query_class_probabilities = F.softmax(query_class_logits, dim=-1)
        predicted_labels = query_class_probabilities.argmax(dim=-1)

        return FlatQueryOutputs(
            features=raw.features[batch_index],
            image_height=raw.img_shape[0],
            image_width=raw.img_shape[1],
            num_decoder_layers=1,
            queries_per_layer=query_mask_embeddings.shape[0],
            mask_embeddings=query_mask_embeddings,
            class_logits=query_class_logits,
            class_probabilities=query_class_probabilities,
            signature_embeddings=torch.empty((query_mask_embeddings.shape[0], 0), device=query_mask_embeddings.device),
            seed_scores=query_class_probabilities.max(dim=-1).values,
            influence_scores=torch.zeros((query_mask_embeddings.shape[0],), device=query_mask_embeddings.device),
            background_confidence=query_class_probabilities[:, 0],
            foreground_confidence=1.0 - query_class_probabilities[:, 0],
            partition_confidence=torch.where(
                predicted_labels == 0,
                query_class_probabilities[:, 0],
                1.0 - query_class_probabilities[:, 0],
            ),
            predicted_labels=predicted_labels,
        )

    def _predict_single(self, model: Mask2FormerBase, raw: RawOutputs, batch_index: int) -> ResolvedPrediction:
        del model

        flat_queries = self._flatten_outputs(raw, batch_index)
        cfg = self.cfg.overlap

        mask_logits = torch.einsum("qc,chw->qhw", flat_queries.mask_embeddings, flat_queries.features)
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0),
            size=(flat_queries.image_height, flat_queries.image_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        mask_probabilities = torch.sigmoid(mask_logits)
        binary_masks = mask_probabilities >= cfg.mask_threshold

        scores = torch.ones_like(flat_queries.seed_scores)
        if cfg.use_class_confidence:
            scores = scores * flat_queries.class_probabilities.max(dim=-1).values
        if cfg.use_foreground_confidence:
            scores = scores * flat_queries.foreground_confidence

        keep_mask = torch.ones_like(scores, dtype=torch.bool)
        if cfg.remove_background:
            keep_mask &= flat_queries.predicted_labels != 0
        keep_mask &= scores >= cfg.min_prototype_score

        resolved_masks = []
        resolved_labels = []
        resolved_scores = []
        kept_mask_logits = []
        kept_mask_probabilities = []
        kept_query_indices = []

        for query_index in torch.where(keep_mask)[0].tolist():
            mask = binary_masks[query_index]
            if mask.sum().item() < cfg.min_area:
                continue
            kept_query_indices.append(query_index)
            kept_mask_logits.append(mask_logits[query_index])
            kept_mask_probabilities.append(mask_probabilities[query_index])
            resolved_masks.append(mask)
            resolved_labels.append(int(flat_queries.predicted_labels[query_index].item()))
            resolved_scores.append(float(scores[query_index].item()))

        kept_query_indices = torch.as_tensor(kept_query_indices, device=flat_queries.features.device, dtype=torch.long)
        empty_mask_grid = torch.empty((0, flat_queries.image_height, flat_queries.image_width), device=flat_queries.features.device)

        return ResolvedPrediction(
            flat_queries=flat_queries,
            prototypes=None,
            kept_prototype_indices=kept_query_indices,
            resolved_target_indices=None,
            signature_embeddings=torch.empty((kept_query_indices.numel(), 0), device=flat_queries.features.device),
            class_logits=flat_queries.class_logits[kept_query_indices],
            class_probabilities=flat_queries.class_probabilities[kept_query_indices],
            mask_embeddings=flat_queries.mask_embeddings[kept_query_indices],
            scores=scores[kept_query_indices],
            raw_mask_logits=torch.stack(kept_mask_logits, dim=0) if kept_mask_logits else empty_mask_grid,
            raw_mask_probabilities=torch.stack(kept_mask_probabilities, dim=0) if kept_mask_probabilities else empty_mask_grid,
            resolved_masks=resolved_masks,
            resolved_labels=resolved_labels,
            resolved_scores=resolved_scores,
        )


def build_predictor(cfg: PrototypeInferenceConfig, model_variant: str):
    variant = model_variant.lower()
    if variant == "standard_mask2former":
        return StandardMask2FormerPredictor(cfg)
    if variant == "clustered":
        return ModularPrototypePredictor(cfg)
    raise ValueError(f"Unknown model variant: {model_variant}")
