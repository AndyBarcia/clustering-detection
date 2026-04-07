from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

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


from .model import CustomMask2Former
from .config import PrototypeInferenceConfig
from .outputs import RawOutputs


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
        B = raw.features.shape[0]
        preds = [self._predict_single(model, raw, b) for b in range(B)]
        return preds[0] if B == 1 else preds

    @torch.no_grad()
    def predict_from_raw_with_gt_prototypes(self, model: CustomMask2Former, raw: RawOutputs, targets):
        B = raw.features.shape[0]
        preds = [self._predict_single_with_gt_prototypes(model, raw, targets, b) for b in range(B)]
        return preds[0] if B == 1 else preds

    def _flatten_outputs(self, raw: RawOutputs, b: int) -> Dict[str, torch.Tensor]:
        features = raw.features[b]
        H_img, W_img = raw.img_shape

        q_mask_emb = raw.mask_embs[:, b]
        q_cls = raw.cls_preds[:, b]
        q_sig = raw.sig_embs[:, b]
        q_seed = raw.seed_scores[:, b]
        q_influence = raw.influence_preds[:, b]

        L, N_q, S = q_sig.shape
        num_classes = q_cls.shape[-1]

        q_mask_emb = q_mask_emb.reshape(L * N_q, -1)
        q_cls = q_cls.reshape(L * N_q, num_classes)
        q_sig = q_sig.reshape(L * N_q, S)
        q_seed = q_seed.reshape(L * N_q)
        q_influence = q_influence.reshape(L * N_q)

        q_cls_prob = F.softmax(q_cls, dim=-1)
        pred_cls = q_cls_prob.argmax(dim=-1)
        bg_conf = q_cls_prob[:, 0]
        fg_conf = 1.0 - q_cls_prob[:, 0]
        partition_conf = torch.where(pred_cls == 0, bg_conf, fg_conf)

        return {
            "features": features,
            "H_img": H_img,
            "W_img": W_img,
            "L": L,
            "N_q": N_q,
            "q_mask_emb": q_mask_emb,
            "q_cls": q_cls,
            "q_cls_prob": q_cls_prob,
            "q_sig": q_sig,
            "q_seed": q_seed,
            "q_influence": q_influence,
            "bg_conf": bg_conf,
            "fg_conf": fg_conf,
            "partition_conf": partition_conf,
            "pred_cls": pred_cls,
        }

    def _select_seeds(self, flat: Dict[str, torch.Tensor]):
        cfg = self.cfg.seed
        score = flat["q_seed"].clone()

        if cfg.use_foreground_in_score:
            score = score * flat["partition_conf"].pow(cfg.foreground_score_power)

        eligible = torch.ones_like(score, dtype=torch.bool)

        if cfg.exclude_background:
            eligible &= (flat["pred_cls"] != 0)

        if cfg.min_foreground_prob > 0:
            eligible &= (flat["partition_conf"] >= cfg.min_foreground_prob)

        if cfg.max_influence is not None:
            eligible &= (flat["q_influence"] <= cfg.max_influence)

        keep = eligible.clone()
        keep &= (flat["q_seed"] >= cfg.quality_threshold)

        seed_idx = torch.where(keep)[0]

        if seed_idx.numel() < cfg.min_num_seeds:
            fallback_idx = torch.where(eligible)[0]

            if fallback_idx.numel() > 0:
                k = min(cfg.min_num_seeds, fallback_idx.numel())
                top_local = torch.topk(score[fallback_idx], k=k).indices
                seed_idx = fallback_idx[top_local]

        if cfg.topk is not None and seed_idx.numel() > cfg.topk:
            top_local = torch.topk(score[seed_idx], k=cfg.topk).indices
            seed_idx = seed_idx[top_local]

        seed_scores = score[seed_idx]
        return seed_idx, seed_scores

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
            G = nx.Graph()
            n = len(seed_sigs_np)
            G.add_nodes_from(range(n))
            edges, weights = _build_weighted_graph_edges(affinity, cfg.graph_min_edge_weight)
            for (u, v), w in zip(edges, weights):
                G.add_edge(u, v, weight=w)
            if G.number_of_edges() == 0:
                return np.arange(n, dtype=np.int64)
            comms = nx.algorithms.community.louvain_communities(
                G,
                weight="weight",
                resolution=cfg.louvain_resolution,
                seed=cfg.random_seed,
            )
            labels = -np.ones(n, dtype=np.int64)
            for cid, comm in enumerate(comms):
                for node in comm:
                    labels[node] = cid
            return labels

        if method == "leiden":
            if ig is None or leidenalg is None:
                raise ImportError("igraph + leidenalg are required for Leiden clustering")
            n = len(seed_sigs_np)
            edges, weights = _build_weighted_graph_edges(affinity, cfg.graph_min_edge_weight)
            if len(edges) == 0:
                return np.arange(n, dtype=np.int64)
            g = ig.Graph(n=n, edges=edges, directed=False)
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                resolution_parameter=cfg.leiden_resolution,
                seed=cfg.random_seed,
            )
            return np.asarray(partition.membership, dtype=np.int64)

        raise ValueError(f"Unknown clustering method: {cfg.method}")

    def _cluster_seeds(self, flat: Dict[str, torch.Tensor], seed_idx: torch.Tensor, seed_scores: torch.Tensor):
        cfg = self.cfg.cluster
        device = seed_idx.device

        if seed_idx.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=device)

        seed_labels = -torch.ones(seed_idx.numel(), dtype=torch.long, device=device)
        next_cluster_id = 0

        if cfg.cluster_per_class:
            seed_classes = flat["pred_cls"][seed_idx]
            unique_classes = seed_classes.unique().tolist()
            groups = [torch.where(seed_classes == c)[0] for c in unique_classes]
        else:
            groups = [torch.arange(seed_idx.numel(), device=device)]

        for pos_group in groups:
            local_seed_idx = seed_idx[pos_group]
            local_scores = seed_scores[pos_group]

            local_sigs_np = flat["q_sig"][local_seed_idx].detach().cpu().numpy()
            local_scores_np = local_scores.detach().cpu().numpy()

            local_labels_np = self._cluster_local(local_sigs_np, local_scores_np)
            local_labels = torch.as_tensor(local_labels_np, device=device)

            good_ids = [int(x) for x in np.unique(local_labels_np) if x != -1]
            for old_id in good_ids:
                m = (local_labels == old_id)
                seed_labels[pos_group[m]] = next_cluster_id
                next_cluster_id += 1

            if cfg.promote_noise_to_singletons:
                noise_pos = pos_group[local_labels == -1]
                for p in noise_pos:
                    seed_labels[p] = next_cluster_id
                    next_cluster_id += 1

        return seed_labels

    def _initialize_prototypes(self, model: CustomMask2Former, flat: Dict[str, torch.Tensor], seed_idx: torch.Tensor, cluster_labels: torch.Tensor):
        device = flat["q_sig"].device
        valid = (cluster_labels >= 0)

        if valid.sum() == 0:
            return {
                "num_prototypes": 0,
                "proto_sig": torch.empty((0, flat["q_sig"].shape[-1]), device=device),
                "proto_cls": torch.empty((0, flat["q_cls"].shape[-1]), device=device),
                "proto_mask_emb": torch.empty((0, flat["q_mask_emb"].shape[-1]), device=device),
                "cluster_members": [],
                "proto_seed_idx": torch.empty((0,), dtype=torch.long, device=device),
                "seed_idx": seed_idx,
                "seed_cluster_labels": cluster_labels,
            }

        cluster_ids = cluster_labels[valid].unique().tolist()

        proto_sig = []
        proto_cls = []
        proto_mask_emb = []
        cluster_members = []
        proto_seed_idx = []

        for cid in cluster_ids:
            member_seed_idx = seed_idx[cluster_labels == cid]
            cluster_members.append(member_seed_idx)

            rep_local = torch.argmax(flat["q_seed"][member_seed_idx])
            rep_seed_idx = member_seed_idx[rep_local]
            proto_seed_idx.append(rep_seed_idx)

            w = flat["q_seed"][member_seed_idx]
            w = w / (w.sum() + 1e-6)

            p_cls = model.aggregate_cls_logits(w.unsqueeze(0), flat["q_cls"][member_seed_idx])[0]
            p_mask = model.aggregate_mask_embeddings(w.unsqueeze(0), flat["q_mask_emb"][member_seed_idx])[0]

            proto_sig.append(flat["q_sig"][rep_seed_idx])
            proto_cls.append(p_cls)
            proto_mask_emb.append(p_mask)

        return {
            "num_prototypes": len(cluster_ids),
            "proto_sig": torch.stack(proto_sig, dim=0),
            "proto_cls": torch.stack(proto_cls, dim=0),
            "proto_mask_emb": torch.stack(proto_mask_emb, dim=0),
            "cluster_members": cluster_members,
            "proto_seed_idx": torch.stack(proto_seed_idx, dim=0),
            "seed_idx": seed_idx,
            "seed_cluster_labels": cluster_labels,
        }

    def _soft_refine_prototypes(self, model: CustomMask2Former, flat: Dict[str, torch.Tensor], proto_state: Dict[str, Any]):
        cfg = self.cfg.assign
        device = flat["q_sig"].device

        if proto_state["num_prototypes"] == 0:
            return proto_state

        if cfg.use_all_queries:
            q_idx = torch.arange(flat["q_sig"].shape[0], device=device)
        else:
            q_idx = proto_state["seed_idx"]

        q_sig = flat["q_sig"][q_idx]
        q_cls = flat["q_cls"][q_idx]
        q_cls_prob = flat["q_cls_prob"][q_idx]
        q_mask_emb = flat["q_mask_emb"][q_idx]

        proto_sig = proto_state["proto_sig"]
        proto_cls = proto_state["proto_cls"]

        alpha = _alpha_value(model.alpha_focal) if cfg.use_alpha_focal else 1.0

        final_norm_w = None
        final_raw_w = None

        n_steps = max(1, cfg.refinement_steps)
        for _ in range(n_steps):
            sim = torch.matmul(q_sig, proto_sig.T)
            affinity = _assignment_affinity(sim, flat["q_influence"][q_idx], cfg.similarity_floor)
            raw_w = affinity.pow(alpha)

            if cfg.use_query_quality:
                raw_w = raw_w * flat["q_seed"][q_idx].pow(cfg.query_quality_power).unsqueeze(1)

            proto_cls_prob = None
            if cfg.use_foreground_prob or cfg.class_compat_power > 0:
                proto_cls_prob = F.softmax(proto_cls, dim=-1)

            if cfg.use_foreground_prob:
                q_fg_conf = flat["fg_conf"][q_idx]
                q_bg_conf = flat["bg_conf"][q_idx]
                proto_bg_conf = proto_cls_prob[:, 0]
                proto_fg_conf = 1.0 - proto_bg_conf
                partition_compat = (
                    q_fg_conf[:, None] * proto_fg_conf[None, :]
                    + q_bg_conf[:, None] * proto_bg_conf[None, :]
                )
                raw_w = raw_w * partition_compat.clamp_min(1e-6).pow(cfg.foreground_prob_power)

            if cfg.class_compat_power > 0:
                class_compat = torch.matmul(q_cls_prob, proto_cls_prob.T).clamp_min(1e-6)
                raw_w = raw_w * class_compat.pow(cfg.class_compat_power)

            if cfg.normalize_over_queries:
                norm_w = raw_w / (raw_w.sum(dim=0, keepdim=True) + 1e-6)
            else:
                norm_w = raw_w / (raw_w.sum(dim=1, keepdim=True) + 1e-6)

            proto_mask_emb = model.aggregate_mask_embeddings(norm_w.T, q_mask_emb)
            proto_cls = model.aggregate_cls_logits(norm_w.T, q_cls)

            final_raw_w = raw_w
            final_norm_w = norm_w

        full_norm_w = torch.zeros(
            flat["q_sig"].shape[0],
            proto_sig.shape[0],
            device=device,
            dtype=proto_sig.dtype,
        )
        full_norm_w[q_idx] = final_norm_w

        proto_state = dict(proto_state)
        proto_state.update({
            "proto_sig": proto_sig,
            "proto_cls": proto_cls,
            "proto_mask_emb": proto_mask_emb,
            "assignment_weights": full_norm_w,
            "assignment_strength": final_raw_w.sum(dim=0),
        })
        return proto_state

    def _build_gt_proto_state(
        self,
        model: CustomMask2Former,
        raw: RawOutputs,
        flat: Dict[str, torch.Tensor],
        targets,
        b: int,
    ):
        device = flat["q_sig"].device
        labels = targets[b]["labels"].to(device)
        masks = targets[b]["masks"].to(device).float()

        if labels.numel() == 0:
            return {
                "num_prototypes": 0,
                "proto_sig": torch.empty((0, flat["q_sig"].shape[-1]), device=device),
                "proto_cls": torch.empty((0, flat["q_cls"].shape[-1]), device=device),
                "proto_mask_emb": torch.empty((0, flat["q_mask_emb"].shape[-1]), device=device),
                "cluster_members": [],
                "proto_seed_idx": torch.empty((0,), dtype=torch.long, device=device),
                "seed_idx": torch.empty((0,), dtype=torch.long, device=device),
                "seed_cluster_labels": torch.empty((0,), dtype=torch.long, device=device),
                "assignment_weights": torch.empty((flat["q_sig"].shape[0], 0), device=device),
                "assignment_strength": torch.empty((0,), device=device),
            }

        gt_masks = masks.unsqueeze(0)
        gt_labels = labels.unsqueeze(0)
        gt_pad_mask = torch.ones((1, labels.shape[0]), dtype=torch.bool, device=device)

        gt_sig = model.encode_gts(
            raw.memory[b:b + 1],
            raw.features[b:b + 1],
            gt_masks,
            gt_labels,
            gt_pad_mask,
            ttt_steps_override=self.cfg.ttt_steps,
        )[0]

        q_sig = flat["q_sig"]
        q_cls = flat["q_cls"]
        q_mask_emb = flat["q_mask_emb"]

        alpha = _alpha_value(model.alpha_focal) if self.cfg.assign.use_alpha_focal else 1.0
        sim = torch.matmul(q_sig, gt_sig.T)
        affinity = _assignment_affinity(sim, flat["q_influence"], self.cfg.assign.similarity_floor)
        raw_w = affinity.pow(alpha)

        # Keep GT-oracle assignment aligned with training:
        # training uses similarity + influence only before query-normalization.
        norm_w = raw_w / (raw_w.sum(dim=0, keepdim=True) + 1e-6)

        proto_mask_emb = model.aggregate_mask_embeddings(norm_w.T, q_mask_emb)
        proto_cls = model.aggregate_cls_logits(norm_w.T, q_cls)

        return {
            "num_prototypes": labels.shape[0],
            "proto_sig": gt_sig,
            "proto_cls": proto_cls,
            "proto_mask_emb": proto_mask_emb,
            "cluster_members": [],
            "proto_seed_idx": torch.full((labels.shape[0],), -1, dtype=torch.long, device=device),
            "seed_idx": torch.arange(labels.shape[0], device=device),
            "seed_cluster_labels": torch.arange(labels.shape[0], device=device),
            "assignment_weights": norm_w,
            "assignment_strength": raw_w.sum(dim=0),
        }

    def _decode_and_resolve(self, flat: Dict[str, torch.Tensor], proto_state: Dict[str, Any]):
        cfg = self.cfg.overlap
        features = flat["features"]
        H_img, W_img = flat["H_img"], flat["W_img"]

        proto_mask_emb = proto_state["proto_mask_emb"]
        proto_cls = proto_state["proto_cls"]
        proto_sig = proto_state["proto_sig"]

        if proto_mask_emb.shape[0] == 0:
            return {
                "seed_idx": proto_state["seed_idx"],
                "seed_cluster_labels": proto_state["seed_cluster_labels"],
                "cluster_members": proto_state["cluster_members"],
                "proto_seed_idx": proto_state["proto_seed_idx"],
                "assignment_weights": torch.empty((flat["q_sig"].shape[0], 0), device=flat["q_sig"].device),
                "all_proto_sig": proto_state["proto_sig"],
                "proto_sig": torch.empty((0, flat["q_sig"].shape[-1]), device=flat["q_sig"].device),
                "proto_cls": torch.empty((0, flat["q_cls"].shape[-1]), device=flat["q_sig"].device),
                "proto_cls_prob": torch.empty((0, flat["q_cls"].shape[-1]), device=flat["q_sig"].device),
                "proto_mask_emb": torch.empty((0, flat["q_mask_emb"].shape[-1]), device=flat["q_sig"].device),
                "proto_score": torch.empty((0,), device=flat["q_sig"].device),
                "raw_mask_logits": torch.empty((0, H_img, W_img), device=flat["q_sig"].device),
                "raw_mask_probs": torch.empty((0, H_img, W_img), device=flat["q_sig"].device),
                "resolved_masks": [],
                "resolved_labels": [],
                "resolved_scores": [],
            }

        mask_logits = torch.einsum("pc,chw->phw", proto_mask_emb, features)
        mask_logits = F.interpolate(
            mask_logits.unsqueeze(0),
            size=(H_img, W_img),
            mode="bilinear",
            align_corners=False,
        )[0]
        mask_probs = F.softmax(mask_logits, dim=0)

        cls_prob = F.softmax(proto_cls, dim=-1)
        pred_cls = cls_prob.argmax(dim=-1)
        cls_conf = cls_prob.max(dim=-1).values
        fg_conf = 1.0 - cls_prob[:, 0]

        proto_score = torch.ones_like(cls_conf)

        if cfg.use_class_confidence:
            proto_score = proto_score * cls_conf
        if cfg.use_foreground_confidence:
            proto_score = proto_score * fg_conf
        if cfg.use_assignment_strength:
            assign_strength = proto_state["assignment_strength"]
            assign_strength = assign_strength / (assign_strength.max() + 1e-6)
            proto_score = proto_score * assign_strength.pow(cfg.assignment_strength_power)

        keep = torch.ones_like(proto_score, dtype=torch.bool)
        if cfg.remove_background:
            keep &= (pred_cls != 0)
        keep &= (proto_score >= cfg.min_prototype_score)

        if keep.sum() == 0:
            return {
                "seed_idx": proto_state["seed_idx"],
                "seed_cluster_labels": proto_state["seed_cluster_labels"],
                "cluster_members": proto_state["cluster_members"],
                "proto_seed_idx": proto_state["proto_seed_idx"],
                "assignment_weights": proto_state["assignment_weights"],
                "all_proto_sig": proto_state["proto_sig"],
                "proto_sig": torch.empty((0, flat["q_sig"].shape[-1]), device=flat["q_sig"].device),
                "proto_cls": torch.empty((0, flat["q_cls"].shape[-1]), device=flat["q_sig"].device),
                "proto_cls_prob": torch.empty((0, flat["q_cls"].shape[-1]), device=flat["q_sig"].device),
                "proto_mask_emb": torch.empty((0, flat["q_mask_emb"].shape[-1]), device=flat["q_sig"].device),
                "proto_score": torch.empty((0,), device=flat["q_sig"].device),
                "raw_mask_logits": torch.empty((0, H_img, W_img), device=flat["q_sig"].device),
                "raw_mask_probs": torch.empty((0, H_img, W_img), device=flat["q_sig"].device),
                "resolved_masks": [],
                "resolved_labels": [],
                "resolved_scores": [],
            }

        keep_idx = torch.where(keep)[0]
        mask_probs_kept = mask_probs[keep]
        mask_logits_kept = mask_logits[keep]
        cls_prob_kept = cls_prob[keep]
        pred_cls_kept = pred_cls[keep]
        proto_score_kept = proto_score[keep]
        proto_sig_kept = proto_sig[keep]
        proto_cls_kept = proto_cls[keep]
        proto_mask_emb_kept = proto_mask_emb[keep]

        pixel_scores = mask_probs * proto_score[:, None, None]
        max_pixel_score, winners = pixel_scores.max(dim=0)

        resolved_masks = []
        resolved_labels = []
        resolved_scores = []

        for kept_pos, proto_idx in enumerate(keep_idx.tolist()):
            m = (winners == proto_idx) & (max_pixel_score >= cfg.pixel_score_threshold)
            m &= (mask_probs[proto_idx] >= cfg.mask_threshold)
            m = self._apply_morphology(m)

            if m.sum().item() < cfg.min_area:
                continue

            resolved_masks.append(m)
            resolved_labels.append(int(pred_cls_kept[kept_pos].item()))
            resolved_scores.append(float(proto_score_kept[kept_pos].item()))

        return {
            "seed_idx": proto_state["seed_idx"],
            "seed_cluster_labels": proto_state["seed_cluster_labels"],
            "cluster_members": proto_state["cluster_members"],
            "proto_seed_idx": proto_state["proto_seed_idx"][keep] if proto_state["proto_seed_idx"].numel() > 0 else proto_state["proto_seed_idx"],
            "assignment_weights": proto_state["assignment_weights"],

            "all_proto_sig": proto_state["proto_sig"],
            "proto_sig": proto_sig_kept,
            "proto_cls": proto_cls_kept,
            "proto_cls_prob": cls_prob_kept,
            "proto_mask_emb": proto_mask_emb_kept,
            "proto_score": proto_score_kept,

            "raw_mask_logits": mask_logits_kept,
            "raw_mask_probs": mask_probs_kept,

            "resolved_masks": resolved_masks,
            "resolved_labels": resolved_labels,
            "resolved_scores": resolved_scores,
        }

    def _predict_single(self, model: CustomMask2Former, raw: RawOutputs, b: int):
        flat = self._flatten_outputs(raw, b)
        seed_idx, seed_scores = self._select_seeds(flat)
        cluster_labels = self._cluster_seeds(flat, seed_idx, seed_scores)
        proto_state = self._initialize_prototypes(model, flat, seed_idx, cluster_labels)
        proto_state = self._soft_refine_prototypes(model, flat, proto_state)
        pred = self._decode_and_resolve(flat, proto_state)
        pred["flat"] = flat
        return pred

    def _predict_single_with_gt_prototypes(self, model: CustomMask2Former, raw: RawOutputs, targets, b: int):
        flat = self._flatten_outputs(raw, b)
        proto_state = self._build_gt_proto_state(model, raw, flat, targets, b)
        pred = self._decode_and_resolve(flat, proto_state)
        pred["flat"] = flat
        return pred
