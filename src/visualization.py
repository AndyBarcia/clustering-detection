from __future__ import annotations

import copy
import hashlib
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

try:
    import umap
except ImportError:
    umap = None

from .config import PrototypeInferenceConfig
from .dataset import SyntheticPanopticBatchGenerator
from .predictor import ModularPrototypePredictor
from .panoptic import PanopticSystem


DEFAULT_CLASS_NAMES = ["Background", "Square", "Triangle"]
_SIGNATURE_PROJECTION_CACHE: dict[tuple, np.ndarray] = {}


def sample_synthetic_examples(
    *,
    num_samples: int,
    dataset_length: int,
    height: int,
    width: int,
    max_objects: int,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[List[torch.Tensor], List[dict]]:
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        generator = SyntheticPanopticBatchGenerator(
            height=height,
            width=width,
            max_objects=max_objects,
            device=device or "cpu",
        )
        batch_size = min(num_samples, max(dataset_length, num_samples))
        batch_images, batch_targets = generator.generate_batch(
            batch_size=batch_size,
            start_idx=0,
        )
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

    images = [image for image in batch_images]
    targets = batch_targets
    return images, targets


def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image_np, 0.0, 1.0)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())
    return xmin, ymin, xmax, ymax


def _project_signatures_2d(signatures: np.ndarray) -> np.ndarray:
    if signatures.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    if signatures.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)

    signatures = signatures.astype(np.float32, copy=False)
    signatures_key = (
        signatures.shape,
        hashlib.blake2b(np.ascontiguousarray(signatures).view(np.uint8), digest_size=16).hexdigest(),
    )
    cached = _SIGNATURE_PROJECTION_CACHE.get(signatures_key)
    if cached is not None:
        return cached.copy()

    if umap is not None:
        n_neighbors = max(2, min(15, signatures.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.15,
            metric="cosine",
            random_state=0,
        )
        embedding = reducer.fit_transform(signatures).astype(np.float32, copy=False)
    else:
        centered = signatures - signatures.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        basis = vh[:2].T
        if basis.shape[1] < 2:
            basis = np.pad(basis, ((0, 0), (0, 2 - basis.shape[1])))
        embedding = (centered @ basis[:, :2]).astype(np.float32, copy=False)

    _SIGNATURE_PROJECTION_CACHE[signatures_key] = embedding.copy()
    return embedding


def _instance_color(image_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    if not mask_bool.any():
        return np.array([0.85, 0.85, 0.85], dtype=np.float32)

    pixels = image_np[mask_bool]
    if pixels.size == 0:
        return np.array([0.85, 0.85, 0.85], dtype=np.float32)

    color = pixels.mean(axis=0)
    color = 0.2 + 0.8 * color
    return np.clip(color, 0.0, 1.0).astype(np.float32, copy=False)


def _gt_marker(label: int) -> str:
    markers = ["o", "s", "^", "D", "P", "X", "v", "*", "<", ">"]
    return markers[int(label) % len(markers)]


def _mix_with_gray(color: np.ndarray, gray: float, amount: float) -> np.ndarray:
    base = np.full((3,), float(gray), dtype=np.float32)
    return np.clip((1.0 - amount) * np.asarray(color, dtype=np.float32) + amount * base, 0.0, 1.0)


def _cluster_colors(num_clusters: int) -> np.ndarray:
    if num_clusters <= 0:
        return np.empty((0, 3), dtype=np.float32)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(idx % 20)[:3] for idx in range(num_clusters)]
    return np.asarray(colors, dtype=np.float32)


def _edge_threshold_for_cfg(cfg: PrototypeInferenceConfig) -> Optional[float]:
    method = cfg.cluster.method.lower()
    if method == "cc":
        return float(cfg.cluster.graph_affinity_threshold)
    if method == "dbscan":
        return float(np.clip(1.0 - cfg.cluster.dbscan_eps, 0.0, 1.0))
    if method in {"louvain", "leiden"}:
        return float(cfg.cluster.graph_min_edge_weight)
    return None


def _draw_signature_umap(
    ax,
    image_np: np.ndarray,
    target: dict,
    prediction: dict,
    *,
    class_names: Optional[Sequence[str]] = None,
    title: str,
):
    flat = prediction.get("flat")
    q_sig = None if flat is None else flat.get("q_sig")
    q_sim = None if flat is None else flat.get("q_sim")
    gt_sig = prediction.get("all_proto_sig", prediction.get("proto_sig"))

    if q_sig is None or gt_sig is None:
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, "No signatures", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return

    q_sig_np = q_sig.detach().cpu().numpy()
    gt_sig_np = gt_sig.detach().cpu().numpy()

    all_sig = np.concatenate([q_sig_np, gt_sig_np], axis=0) if gt_sig_np.shape[0] > 0 else q_sig_np
    embedding = _project_signatures_2d(all_sig)
    q_pts = embedding[: q_sig_np.shape[0]]
    gt_pts = embedding[q_sig_np.shape[0] :]

    gt_masks = [mask.detach().cpu().numpy() for mask in target["masks"]]
    gt_labels = [int(label) for label in target["labels"].detach().cpu().tolist()]
    gt_colors = [_instance_color(image_np, mask) for mask in gt_masks]
    gt_marker_size = 150.0
    min_query_marker_size = 18.0

    ax.set_title(title)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    if q_sig_np.shape[0] > 1:
        q_sig_norm = q_sig_np / np.clip(np.linalg.norm(q_sig_np, axis=1, keepdims=True), 1e-6, None)
        cosine_dist = 1.0 - np.clip(q_sig_norm @ q_sig_norm.T, -1.0, 1.0)
        np.fill_diagonal(cosine_dist, np.inf)
        neighbor_order = np.argsort(cosine_dist, axis=1)
        arrow_specs = [
            (0, "red", 0.22, 0.45),
            (1, "orange", 0.14, 0.35),
        ]
        for neighbor_rank, color, alpha, linewidth in arrow_specs:
            if neighbor_rank >= neighbor_order.shape[1]:
                continue
            neighbor_idx = neighbor_order[:, neighbor_rank]
            for idx, nbr_idx in enumerate(neighbor_idx):
                dx = q_pts[nbr_idx, 0] - q_pts[idx, 0]
                dy = q_pts[nbr_idx, 1] - q_pts[idx, 1]
                ax.arrow(
                    q_pts[idx, 0],
                    q_pts[idx, 1],
                    dx,
                    dy,
                    color=color,
                    alpha=alpha,
                    linewidth=linewidth,
                    length_includes_head=True,
                    head_width=0.0,
                    head_length=0.0,
                    zorder=1,
                )

    if q_sim is not None:
        q_sim_np = q_sim.detach().cpu().numpy()
        query_sizes = min_query_marker_size + (gt_marker_size - min_query_marker_size) * np.clip(q_sim_np, 0.0, 1.0)
    else:
        query_sizes = np.full((q_pts.shape[0],), min_query_marker_size, dtype=np.float32)

    if gt_sig_np.shape[0] > 0 and prediction.get("assignment_weights") is not None:
        assignment = prediction["assignment_weights"].detach().cpu().numpy()
        if assignment.shape[1] > 0:
            query_owner = assignment.argmax(axis=1)
            query_strength = assignment.max(axis=1)
            query_colors = np.asarray([gt_colors[idx] for idx in query_owner], dtype=np.float32)
            query_alpha = 0.55 + 0.35 * np.clip(query_strength, 0.0, 1.0)
            for idx in range(q_pts.shape[0]):
                ax.scatter(
                    q_pts[idx, 0],
                    q_pts[idx, 1],
                    s=float(query_sizes[idx]),
                    c=[query_colors[idx]],
                    alpha=float(query_alpha[idx]),
                    marker=".",
                    linewidths=0,
                    zorder=2,
                )
        else:
            ax.scatter(q_pts[:, 0], q_pts[:, 1], s=query_sizes, c="0.7", alpha=0.72, marker=".", linewidths=0, zorder=2)
    else:
        ax.scatter(q_pts[:, 0], q_pts[:, 1], s=query_sizes, c="0.7", alpha=0.72, marker=".", linewidths=0, zorder=2)

    for idx, pt in enumerate(gt_pts):
        label = gt_labels[idx] if idx < len(gt_labels) else idx
        marker = _gt_marker(label)
        color = gt_colors[idx] if idx < len(gt_colors) else np.array([0.2, 0.2, 0.2], dtype=np.float32)
        class_name = str(label)
        if class_names is not None and 0 <= label < len(class_names):
            class_name = class_names[label]

        ax.scatter(
            pt[0],
            pt[1],
            s=gt_marker_size,
            c=[color],
            marker=marker,
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
        )
        ax.text(
            pt[0],
            pt[1],
            f" {class_name}",
            fontsize=8,
            color="black",
            va="center",
            ha="left",
            zorder=4,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def _draw_inference_umap(
    ax,
    target: dict,
    prediction: dict,
    *,
    inference_cfg: PrototypeInferenceConfig,
    class_names: Optional[Sequence[str]] = None,
    gt_reference: Optional[dict] = None,
    title: str,
):
    flat = prediction.get("flat")
    q_sig = None if flat is None else flat.get("q_sig")
    all_proto_sig = prediction.get("all_proto_sig")

    if q_sig is None or all_proto_sig is None:
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, "No inference signatures", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return

    q_sig_np = q_sig.detach().cpu().numpy()
    all_proto_sig_np = all_proto_sig.detach().cpu().numpy()
    gt_sig = None if gt_reference is None else gt_reference.get("all_proto_sig")
    gt_sig_np = np.empty((0, q_sig_np.shape[1]), dtype=np.float32)
    if gt_sig is not None:
        gt_sig_np = gt_sig.detach().cpu().numpy()

    all_sig_parts = [q_sig_np]
    if all_proto_sig_np.shape[0] > 0:
        all_sig_parts.append(all_proto_sig_np)
    if gt_sig_np.shape[0] > 0:
        all_sig_parts.append(gt_sig_np)
    embedding = _project_signatures_2d(np.concatenate(all_sig_parts, axis=0))

    q_end = q_sig_np.shape[0]
    proto_end = q_end + all_proto_sig_np.shape[0]
    q_pts = embedding[:q_end]
    proto_pts = embedding[q_end:proto_end]
    gt_pts = embedding[proto_end:]

    diagnostics = prediction.get("diagnostics", {})
    selected_seed_mask = diagnostics.get("selected_seed_mask")
    pre_topk_seed_mask = diagnostics.get("pre_topk_seed_mask")
    seed_score = diagnostics.get("seed_score")
    passes_background = diagnostics.get("passes_background")
    passes_foreground = diagnostics.get("passes_foreground")
    passes_quality = diagnostics.get("passes_quality")
    seed_cluster_labels = diagnostics.get("seed_cluster_labels_full")
    assignment_weights = prediction.get("assignment_weights")

    q_quality_np = flat["q_quality"].detach().cpu().numpy()
    q_quality_norm = q_quality_np / max(float(q_quality_np.max()), 1e-6)
    query_sizes = 16.0 + 52.0 * np.sqrt(np.clip(q_quality_norm, 0.0, 1.0))

    selected_seed_mask_np = np.zeros((q_pts.shape[0],), dtype=bool) if selected_seed_mask is None else selected_seed_mask.detach().cpu().numpy().astype(bool)
    pre_topk_seed_mask_np = np.zeros((q_pts.shape[0],), dtype=bool) if pre_topk_seed_mask is None else pre_topk_seed_mask.detach().cpu().numpy().astype(bool)
    passes_background_np = np.ones((q_pts.shape[0],), dtype=bool) if passes_background is None else passes_background.detach().cpu().numpy().astype(bool)
    passes_foreground_np = np.ones((q_pts.shape[0],), dtype=bool) if passes_foreground is None else passes_foreground.detach().cpu().numpy().astype(bool)
    passes_quality_np = np.ones((q_pts.shape[0],), dtype=bool) if passes_quality is None else passes_quality.detach().cpu().numpy().astype(bool)
    seed_cluster_labels_np = -np.ones((q_pts.shape[0],), dtype=np.int64) if seed_cluster_labels is None else seed_cluster_labels.detach().cpu().numpy().astype(np.int64)
    seed_score_np = np.zeros((q_pts.shape[0],), dtype=np.float32) if seed_score is None else seed_score.detach().cpu().numpy().astype(np.float32)

    num_proto = all_proto_sig_np.shape[0]
    cluster_colors = _cluster_colors(num_proto)
    proto_keep_mask = prediction.get("all_proto_keep_mask")
    proto_drop_background_mask = prediction.get("all_proto_drop_background_mask")
    proto_drop_score_mask = prediction.get("all_proto_drop_score_mask")
    proto_pred_cls = prediction.get("all_proto_pred_cls")
    proto_score = prediction.get("all_proto_score")

    proto_keep_mask_np = np.ones((num_proto,), dtype=bool) if proto_keep_mask is None else proto_keep_mask.detach().cpu().numpy().astype(bool)
    proto_drop_background_mask_np = np.zeros((num_proto,), dtype=bool) if proto_drop_background_mask is None else proto_drop_background_mask.detach().cpu().numpy().astype(bool)
    proto_drop_score_mask_np = np.zeros((num_proto,), dtype=bool) if proto_drop_score_mask is None else proto_drop_score_mask.detach().cpu().numpy().astype(bool)
    proto_pred_cls_np = np.zeros((num_proto,), dtype=np.int64) if proto_pred_cls is None else proto_pred_cls.detach().cpu().numpy().astype(np.int64)
    proto_score_np = np.zeros((num_proto,), dtype=np.float32) if proto_score is None else proto_score.detach().cpu().numpy().astype(np.float32)

    query_owner = -np.ones((q_pts.shape[0],), dtype=np.int64)
    query_strength = np.zeros((q_pts.shape[0],), dtype=np.float32)
    query_used_mask = np.zeros((q_pts.shape[0],), dtype=bool)
    if assignment_weights is not None and num_proto > 0:
        assignment_np = assignment_weights.detach().cpu().numpy()
        if assignment_np.shape[1] > 0:
            query_owner = assignment_np.argmax(axis=1).astype(np.int64)
            query_strength = assignment_np.max(axis=1).astype(np.float32)
            query_used_mask = query_strength > 1e-8

    cluster_owner = query_owner.copy()
    unresolved_seed_mask = selected_seed_mask_np & (cluster_owner < 0) & (seed_cluster_labels_np >= 0)
    cluster_owner[unresolved_seed_mask] = seed_cluster_labels_np[unresolved_seed_mask]

    active_kept_mask = query_used_mask & (cluster_owner >= 0)
    if num_proto > 0:
        active_kept_mask &= proto_keep_mask_np[cluster_owner]
    active_dropped_mask = query_used_mask & (cluster_owner >= 0) & (~active_kept_mask)

    background_filtered_mask = (~selected_seed_mask_np) & (~passes_background_np)
    low_fg_filtered_mask = (~selected_seed_mask_np) & passes_background_np & (~passes_foreground_np)
    quality_filtered_mask = (~selected_seed_mask_np) & passes_background_np & passes_foreground_np & (~passes_quality_np)
    topk_filtered_mask = pre_topk_seed_mask_np & (~selected_seed_mask_np)
    inactive_mask = (~selected_seed_mask_np) & (~query_used_mask) & passes_background_np & passes_foreground_np & passes_quality_np & (~topk_filtered_mask)

    ax.set_title(title)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    edge_threshold = _edge_threshold_for_cfg(inference_cfg)
    if edge_threshold is not None and selected_seed_mask_np.sum() > 1:
        seed_indices = np.where(selected_seed_mask_np)[0]
        seed_sigs = q_sig_np[seed_indices].astype(np.float32, copy=False)
        seed_sigs = seed_sigs / np.clip(np.linalg.norm(seed_sigs, axis=1, keepdims=True), 1e-6, None)
        affinity = np.clip(seed_sigs @ seed_sigs.T, 0.0, 1.0)
        for src_pos in range(seed_indices.shape[0]):
            for dst_pos in range(src_pos + 1, seed_indices.shape[0]):
                strength = float(affinity[src_pos, dst_pos])
                if strength < edge_threshold:
                    continue
                src_idx = seed_indices[src_pos]
                dst_idx = seed_indices[dst_pos]
                src_cluster = seed_cluster_labels_np[src_idx]
                dst_cluster = seed_cluster_labels_np[dst_idx]
                if src_cluster >= 0 and src_cluster == dst_cluster and src_cluster < cluster_colors.shape[0]:
                    edge_color = cluster_colors[src_cluster]
                    edge_alpha = 0.18 + 0.32 * strength
                else:
                    edge_color = np.array([0.55, 0.55, 0.55], dtype=np.float32)
                    edge_alpha = 0.05 + 0.12 * strength
                ax.plot(
                    [q_pts[src_idx, 0], q_pts[dst_idx, 0]],
                    [q_pts[src_idx, 1], q_pts[dst_idx, 1]],
                    color=edge_color,
                    alpha=float(np.clip(edge_alpha, 0.0, 0.45)),
                    linewidth=0.4 + 0.9 * strength,
                    zorder=1,
                )

    if background_filtered_mask.any():
        ax.scatter(
            q_pts[background_filtered_mask, 0],
            q_pts[background_filtered_mask, 1],
            s=query_sizes[background_filtered_mask],
            c=[[0.87, 0.87, 0.87]],
            alpha=0.22,
            marker="o",
            linewidths=0,
            zorder=2,
        )

    muted_filtered_mask = low_fg_filtered_mask | quality_filtered_mask | topk_filtered_mask | inactive_mask
    if muted_filtered_mask.any():
        ax.scatter(
            q_pts[muted_filtered_mask, 0],
            q_pts[muted_filtered_mask, 1],
            s=query_sizes[muted_filtered_mask],
            c=[[0.62, 0.62, 0.62]],
            alpha=0.34,
            marker="o",
            linewidths=0,
            zorder=2,
        )

    if active_dropped_mask.any():
        dropped_colors = []
        for owner in cluster_owner[active_dropped_mask]:
            color = cluster_colors[owner] if 0 <= owner < cluster_colors.shape[0] else np.array([0.55, 0.55, 0.55], dtype=np.float32)
            dropped_colors.append(_mix_with_gray(color, gray=0.72, amount=0.55))
        ax.scatter(
            q_pts[active_dropped_mask, 0],
            q_pts[active_dropped_mask, 1],
            s=query_sizes[active_dropped_mask],
            c=np.asarray(dropped_colors, dtype=np.float32),
            alpha=0.45,
            marker="o",
            linewidths=0,
            zorder=3,
        )

    active_nonseed_mask = active_kept_mask & (~selected_seed_mask_np)
    if active_nonseed_mask.any():
        active_indices = np.where(active_nonseed_mask)[0]
        active_colors = cluster_colors[cluster_owner[active_nonseed_mask]]
        active_alpha = 0.38 + 0.32 * np.clip(query_strength[active_nonseed_mask], 0.0, 1.0)
        for local_idx, (idx, alpha) in enumerate(zip(active_indices.tolist(), active_alpha.tolist())):
            ax.scatter(
                q_pts[idx, 0],
                q_pts[idx, 1],
                s=float(query_sizes[idx]),
                c=[active_colors[local_idx]],
                alpha=float(alpha),
                marker="o",
                linewidths=0,
                zorder=4,
            )

    seed_mask = selected_seed_mask_np & (cluster_owner >= 0)
    if seed_mask.any():
        seed_colors = []
        seed_edge_colors = []
        for idx in np.where(seed_mask)[0]:
            owner = int(cluster_owner[idx])
            base_color = cluster_colors[owner] if 0 <= owner < cluster_colors.shape[0] else np.array([0.25, 0.25, 0.25], dtype=np.float32)
            if owner < num_proto and not proto_keep_mask_np[owner]:
                seed_colors.append(_mix_with_gray(base_color, gray=0.72, amount=0.45))
                seed_edge_colors.append((0.35, 0.35, 0.35))
            else:
                seed_colors.append(base_color)
                seed_edge_colors.append((0.05, 0.05, 0.05))

        seed_indices = np.where(seed_mask)[0]
        for local_idx, query_idx in enumerate(seed_indices.tolist()):
            size_boost = 26.0 * np.clip(seed_score_np[query_idx] / max(float(seed_score_np.max()), 1e-6), 0.0, 1.0)
            ax.scatter(
                q_pts[query_idx, 0],
                q_pts[query_idx, 1],
                s=float(query_sizes[query_idx] + size_boost),
                c=[seed_colors[local_idx]],
                alpha=0.96,
                marker="o",
                edgecolors=[seed_edge_colors[local_idx]],
                linewidths=0.8,
                zorder=5,
            )

    for proto_idx in range(num_proto):
        color = cluster_colors[proto_idx] if proto_idx < cluster_colors.shape[0] else np.array([0.2, 0.2, 0.2], dtype=np.float32)
        if proto_keep_mask_np[proto_idx]:
            marker = "*"
            face_color = color
            edge_color = "black"
            alpha = 0.98
            size = 270.0
        else:
            marker = "X" if proto_drop_background_mask_np[proto_idx] else "D"
            face_color = _mix_with_gray(color, gray=0.74, amount=0.58)
            edge_color = "0.35"
            alpha = 0.72
            size = 180.0

        ax.scatter(
            proto_pts[proto_idx, 0],
            proto_pts[proto_idx, 1],
            s=size,
            c=[face_color],
            marker=marker,
            edgecolors=edge_color,
            linewidths=1.0,
            alpha=alpha,
            zorder=6,
        )

        if proto_keep_mask_np[proto_idx]:
            label = int(proto_pred_cls_np[proto_idx]) if proto_idx < proto_pred_cls_np.shape[0] else proto_idx
            class_name = str(label)
            if class_names is not None and 0 <= label < len(class_names):
                class_name = class_names[label]
            ax.text(
                proto_pts[proto_idx, 0],
                proto_pts[proto_idx, 1],
                f" {class_name} {proto_score_np[proto_idx]:.2f}",
                fontsize=8,
                color="black",
                va="center",
                ha="left",
                zorder=7,
            )

    if gt_pts.shape[0] > 0:
        gt_labels = [int(label) for label in target["labels"].detach().cpu().tolist()]
        for idx, pt in enumerate(gt_pts):
            gt_label = gt_labels[idx] if idx < len(gt_labels) else idx
            ax.scatter(
                pt[0],
                pt[1],
                s=95.0,
                facecolors="none",
                edgecolors="black",
                linewidths=1.0,
                marker=_gt_marker(gt_label),
                alpha=0.55,
                zorder=6,
            )

    ax.text(
        0.01,
        0.01,
        "color=active cluster | gray=filtered | *=kept proto | X=dropped proto",
        transform=ax.transAxes,
        fontsize=8,
        color="0.28",
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
    )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_alpha(0.3)


def _draw_instances(
    ax,
    image_np: np.ndarray,
    masks: Sequence[np.ndarray],
    labels: Sequence[int],
    *,
    scores: Optional[Sequence[float]] = None,
    class_names: Optional[Sequence[str]] = None,
    title: str,
):
    ax.imshow(image_np)
    ax.set_title(title)
    ax.axis("off")

    if len(masks) == 0:
        ax.text(
            0.5,
            0.5,
            "No instances",
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            transform=ax.transAxes,
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 4},
        )
        return

    cmap = plt.get_cmap("tab20")
    for idx, mask in enumerate(masks):
        mask_bool = np.asarray(mask).astype(bool)
        if not mask_bool.any():
            continue

        color = np.array(cmap(idx % 20)[:3])
        overlay = np.zeros((*mask_bool.shape, 4), dtype=np.float32)
        overlay[mask_bool, :3] = color
        overlay[mask_bool, 3] = 0.35
        ax.imshow(overlay)

        bbox = _mask_bbox(mask_bool)
        if bbox is None:
            continue

        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle(
            (xmin, ymin),
            max(xmax - xmin, 1),
            max(ymax - ymin, 1),
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        class_name = str(labels[idx])
        if class_names is not None and 0 <= int(labels[idx]) < len(class_names):
            class_name = class_names[int(labels[idx])]

        label_text = class_name
        if scores is not None:
            label_text = f"{class_name} {float(scores[idx]):.2f}"

        ax.text(
            xmin,
            max(ymin - 4, 0),
            label_text,
            fontsize=9,
            color="white",
            bbox={"facecolor": color, "alpha": 0.85, "pad": 2},
        )


@torch.no_grad()
def run_raw_outputs(
    system: PanopticSystem,
    images: Sequence[torch.Tensor],
    *,
    inference_cfg: Optional[PrototypeInferenceConfig] = None,
    device: Optional[torch.device] = None,
):
    if len(images) == 0:
        return None

    model_device = device
    if model_device is None:
        model_device = next(system.parameters()).device

    batch = torch.stack(list(images)).to(model_device)
    was_training = system.training
    system.eval()
    try:
        ttt_steps = system._resolve_ttt_steps(inference_cfg)
        raw = system.model(batch, ttt_steps_override=ttt_steps)
    finally:
        system.train(was_training)

    return raw


@torch.no_grad()
def run_predictions(
    system: PanopticSystem,
    images: Sequence[torch.Tensor],
    *,
    raw=None,
    inference_cfg: Optional[PrototypeInferenceConfig] = None,
    device: Optional[torch.device] = None,
):
    if len(images) == 0:
        return []

    predictor = system.predictor if inference_cfg is None else ModularPrototypePredictor(inference_cfg)
    if raw is None:
        raw = run_raw_outputs(system, images, inference_cfg=inference_cfg, device=device)

    was_training = system.training
    system.eval()
    try:
        predictions = predictor.predict_from_raw(system.model, raw)
    finally:
        system.train(was_training)

    if isinstance(predictions, list):
        return predictions
    return [predictions]


@torch.no_grad()
def run_predictions_with_gt_prototypes(
    system: PanopticSystem,
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    *,
    raw=None,
    inference_cfg: Optional[PrototypeInferenceConfig] = None,
    device: Optional[torch.device] = None,
):
    if len(images) == 0:
        return []

    predictor = system.predictor if inference_cfg is None else ModularPrototypePredictor(inference_cfg)
    if raw is None:
        raw = run_raw_outputs(system, images, inference_cfg=inference_cfg, device=device)

    was_training = system.training
    system.eval()
    try:
        predictions = predictor.predict_from_raw_with_gt_prototypes(system.model, raw, targets)
    finally:
        system.train(was_training)

    if isinstance(predictions, list):
        return predictions
    return [predictions]


def run_gt_signature_reference(
    system: PanopticSystem,
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    *,
    raw=None,
    inference_cfg: Optional[PrototypeInferenceConfig] = None,
    device: Optional[torch.device] = None,
):
    if len(images) == 0:
        return []

    predictor = system.predictor if inference_cfg is None else ModularPrototypePredictor(inference_cfg)
    if raw is None:
        raw = run_raw_outputs(system, images, inference_cfg=inference_cfg, device=device)

    refs = []
    for idx, target in enumerate(targets):
        flat = predictor._flatten_outputs(raw, idx)
        # GT encoding may run the decoder's TTT adaptation loop, which needs gradients
        # even during interactive evaluation.
        with torch.enable_grad():
            proto_state = predictor._build_gt_proto_state(system.model, raw, flat, targets, idx)
        refs.append(
            {
                "flat": flat,
                "all_proto_sig": proto_state["proto_sig"],
                "assignment_weights": proto_state["assignment_weights"],
                "resolved_masks": [],
                "resolved_labels": [],
                "resolved_scores": [],
            }
        )
    return refs


def _build_prediction_columns(
    predictions: Sequence[dict],
    gt_proto_predictions: Optional[Sequence[dict]] = None,
):
    prediction_columns = [("Clustered Prediction", predictions)]
    if gt_proto_predictions is not None:
        prediction_columns.append(("GT Prototype Prediction", gt_proto_predictions))
    return prediction_columns


def _draw_prediction_grid_axes(
    axes,
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    prediction_columns: Sequence[Tuple[str, Sequence[dict]]],
    *,
    class_names: Optional[Sequence[str]] = None,
):
    axes = np.atleast_2d(axes)
    add_signature_column = any("GT Prototype" in title for title, _ in prediction_columns)

    for ax in axes.flat:
        ax.clear()

    for row_idx, (image, target) in enumerate(zip(images, targets)):
        image_np = _to_numpy_image(image)
        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_title("Input")
        axes[row_idx, 0].axis("off")

        gt_masks = [mask.detach().cpu().numpy() for mask in target["masks"]]
        gt_labels = [int(label) for label in target["labels"].detach().cpu().tolist()]
        _draw_instances(
            axes[row_idx, 1],
            image_np,
            gt_masks,
            gt_labels,
            class_names=class_names,
            title="Ground Truth",
        )

        next_col_idx = 2
        for column_title, predictions in prediction_columns:
            prediction = predictions[row_idx]
            pred_masks = [mask.detach().cpu().numpy() for mask in prediction["resolved_masks"]]
            pred_labels = [int(label) for label in prediction["resolved_labels"]]
            pred_scores = [float(score) for score in prediction["resolved_scores"]]
            _draw_instances(
                axes[row_idx, next_col_idx],
                image_np,
                pred_masks,
                pred_labels,
                scores=pred_scores,
                class_names=class_names,
                title=column_title,
            )
            next_col_idx += 1

            if add_signature_column and "GT Prototype" in column_title:
                _draw_signature_umap(
                    axes[row_idx, next_col_idx],
                    image_np,
                    target,
                    prediction,
                    class_names=class_names,
                    title="GT Signature UMAP",
                )
                next_col_idx += 1


def _draw_interactive_sample_axes(
    axes,
    image: torch.Tensor,
    target: dict,
    prediction: dict,
    *,
    inference_cfg: PrototypeInferenceConfig,
    signature_reference: Optional[dict] = None,
    class_names: Optional[Sequence[str]] = None,
):
    image_np = _to_numpy_image(image)
    axes = list(np.ravel(axes))
    for ax in axes:
        ax.clear()

    gt_masks = [mask.detach().cpu().numpy() for mask in target["masks"]]
    gt_labels = [int(label) for label in target["labels"].detach().cpu().tolist()]
    _draw_instances(
        axes[0],
        image_np,
        gt_masks,
        gt_labels,
        class_names=class_names,
        title="Ground Truth",
    )

    pred_masks = [mask.detach().cpu().numpy() for mask in prediction["resolved_masks"]]
    pred_labels = [int(label) for label in prediction["resolved_labels"]]
    pred_scores = [float(score) for score in prediction["resolved_scores"]]
    _draw_instances(
        axes[1],
        image_np,
        pred_masks,
        pred_labels,
        scores=pred_scores,
        class_names=class_names,
        title="Clustered Prediction",
    )

    if len(axes) > 2:
        _draw_inference_umap(
            axes[2],
            target,
            prediction,
            inference_cfg=inference_cfg,
            gt_reference=signature_reference,
            class_names=class_names,
            title="Inference UMAP",
        )


def render_prediction_grid(
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    prediction_columns: Sequence[Tuple[str, Sequence[dict]]],
    *,
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
):
    num_samples = len(images)
    add_signature_column = any("GT Prototype" in title for title, _ in prediction_columns)
    num_cols = 2 + len(prediction_columns) + int(add_signature_column)
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(5 * num_cols, 5 * max(num_samples, 1)), squeeze=False)

    if figure_title:
        fig.suptitle(figure_title, fontsize=14)

    _draw_prediction_grid_axes(
        axes,
        images,
        targets,
        prediction_columns,
        class_names=class_names,
    )

    if figure_title:
        plt.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        plt.tight_layout()
    return fig


def save_prediction_grid(
    path: Path,
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    predictions: Sequence[dict],
    *,
    gt_proto_predictions: Optional[Sequence[dict]] = None,
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_columns = _build_prediction_columns(predictions, gt_proto_predictions)
    fig = render_prediction_grid(
        images,
        targets,
        prediction_columns,
        class_names=class_names,
        figure_title=figure_title,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def show_prediction_grid(
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    predictions: Sequence[dict],
    *,
    gt_proto_predictions: Optional[Sequence[dict]] = None,
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
    window_title: Optional[str] = None,
):
    prediction_columns = _build_prediction_columns(predictions, gt_proto_predictions)
    fig = render_prediction_grid(
        images,
        targets,
        prediction_columns,
        class_names=class_names,
        figure_title=figure_title,
    )
    if window_title and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(window_title)
    plt.show()
    plt.close(fig)


def render_prediction_grid_to_image(
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    prediction_columns: Sequence[Tuple[str, Sequence[dict]]],
    *,
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
    dpi: float = 110.0,
) -> np.ndarray:
    fig = render_prediction_grid(
        images,
        targets,
        prediction_columns,
        class_names=class_names,
        figure_title=figure_title,
    )
    fig.set_dpi(dpi)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return image


def _available_cluster_methods() -> List[str]:
    from . import predictor as predictor_module

    methods = ["cc"]
    if predictor_module.DBSCAN is not None:
        methods.append("dbscan")
    if predictor_module._hdbscan is not None:
        methods.append("hdbscan")
    if predictor_module.nx is not None and hasattr(predictor_module.nx.algorithms.community, "louvain_communities"):
        methods.append("louvain")
    if predictor_module.ig is not None and predictor_module.leidenalg is not None:
        methods.append("leiden")
    return methods


class InteractivePredictionGrid:
    def __init__(
        self,
        system: PanopticSystem,
        images: Sequence[torch.Tensor],
        targets: Sequence[dict],
        *,
        class_names: Optional[Sequence[str]] = None,
        figure_title: Optional[str] = None,
        window_title: Optional[str] = None,
        include_gt_proto_predictions: bool = True,
        device: Optional[torch.device] = None,
        sample_callback: Optional[Callable[[], Tuple[List[torch.Tensor], List[dict]]]] = None,
    ):
        self.system = system
        self.images = list(images)
        self.targets = list(targets)
        self.class_names = class_names
        self.figure_title = figure_title
        self.include_gt_proto_predictions = include_gt_proto_predictions
        self.device = device
        self.sample_callback = sample_callback

        self.base_cfg = copy.deepcopy(system.cfg.inference)
        self.available_cluster_methods = _available_cluster_methods()
        if self.base_cfg.cluster.method not in self.available_cluster_methods:
            self.available_cluster_methods.append(self.base_cfg.cluster.method)

        num_cols = 2 + int(include_gt_proto_predictions)
        self.current_index = 0

        self.fig = plt.figure(figsize=(16, 6.8))
        grid_spec = self.fig.add_gridspec(
            1,
            num_cols,
            left=0.04,
            right=0.62,
            top=0.86,
            bottom=0.08,
            wspace=0.08,
        )
        self.display_axes = np.empty((1, num_cols), dtype=object)
        for col_idx in range(num_cols):
            self.display_axes[0, col_idx] = self.fig.add_subplot(grid_spec[0, col_idx])
        self._dirty = False
        self._raw_cache: dict[tuple[int, Optional[int]], object] = {}
        self._gt_signature_cache: dict[tuple, dict] = {}

        if figure_title:
            self.fig.suptitle(figure_title, fontsize=14)
        if window_title and self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title(window_title)

        self._suspend_callbacks = False
        self._build_controls()
        self.refresh_predictions()

    def _build_controls(self):
        x0 = 0.66
        width = 0.30
        button_w = 0.072
        gap = 0.007

        self.prev_button = Button(self.fig.add_axes([0.04, 0.90, 0.05, 0.04]), "Prev")
        self.prev_button.on_clicked(self._on_prev_sample)
        self.next_button = Button(self.fig.add_axes([0.095, 0.90, 0.05, 0.04]), "Next")
        self.next_button.on_clicked(self._on_next_sample)
        self.sample_text = self.fig.text(0.155, 0.912, "", fontsize=10, ha="left", va="center")

        self.apply_button = Button(self.fig.add_axes([x0, 0.92, button_w, 0.04]), "Apply")
        self.apply_button.on_clicked(self._on_apply_clicked)

        self.reset_button = Button(self.fig.add_axes([x0 + button_w + gap, 0.92, button_w, 0.04]), "Reset")
        self.reset_button.on_clicked(self._on_reset_clicked)

        self.resample_button = None
        if self.sample_callback is not None:
            self.resample_button = Button(self.fig.add_axes([x0 + 2 * (button_w + gap), 0.92, button_w, 0.04]), "Resample")
            self.resample_button.on_clicked(self._on_resample_clicked)

        title_ax = self.fig.add_axes([x0, 0.88, width, 0.03])
        title_ax.axis("off")
        title_ax.text(0.0, 0.5, "Inference Controls", fontsize=11, fontweight="bold", va="center")

        radio_ax = self.fig.add_axes([x0, 0.72, width, 0.14])
        self.cluster_method_radio = RadioButtons(
            radio_ax,
            self.available_cluster_methods,
            active=self.available_cluster_methods.index(self.base_cfg.cluster.method),
        )
        self.cluster_method_radio.on_clicked(self._on_widget_change)

        checks_ax = self.fig.add_axes([x0, 0.49, width, 0.2])
        self.toggle_labels = [
            "Cluster per class",
            "Use all queries",
            "Use query quality",
            "FG confidence",
            "Assign strength",
        ]
        self.toggle_checks = CheckButtons(
            checks_ax,
            self.toggle_labels,
            [
                self.base_cfg.cluster.cluster_per_class,
                self.base_cfg.assign.use_all_queries,
                self.base_cfg.assign.use_query_quality,
                self.base_cfg.overlap.use_foreground_confidence,
                self.base_cfg.overlap.use_assignment_strength,
            ],
        )
        self.toggle_checks.on_clicked(self._on_widget_change)

        slider_h = 0.025
        slider_gap = 0.042
        slider_y = 0.45

        max_topk = self.system.cfg.model.decoder.num_layers * self.system.cfg.model.decoder.num_queries
        max_ttt = max(12, int(self.system.cfg.model.decoder_layer.ttt_steps) + 6)
        max_refinement = max(4, int(self.base_cfg.assign.refinement_steps) + 3)

        self.ttt_steps_slider = self._create_slider(
            x0, slider_y, width, slider_h, "TTT steps (0=ckpt)", 0, max_ttt, self._optional_int_to_slider(self.base_cfg.ttt_steps), 1
        )
        slider_y -= slider_gap
        self.quality_threshold_slider = self._create_slider(
            x0, slider_y, width, slider_h, "Seed quality", 0.0, 1.0, self.base_cfg.seed.quality_threshold, 0.01
        )
        slider_y -= slider_gap
        self.min_fg_slider = self._create_slider(
            x0, slider_y, width, slider_h, "Min FG prob", 0.0, 1.0, self.base_cfg.seed.min_foreground_prob, 0.01
        )
        slider_y -= slider_gap
        self.topk_slider = self._create_slider(
            x0, slider_y, width, slider_h, "Seed top-k (0=all)", 0, max_topk, self._optional_int_to_slider(self.base_cfg.seed.topk), 1
        )
        slider_y -= slider_gap
        self.refinement_slider = self._create_slider(
            x0, slider_y, width, slider_h, "Refine steps", 1, max_refinement, self.base_cfg.assign.refinement_steps, 1
        )
        slider_y -= slider_gap
        self.graph_affinity_slider = self._create_slider(
            x0, slider_y, width, slider_h, "Graph affinity", 0.0, 1.0, self.base_cfg.cluster.graph_affinity_threshold, 0.01
        )
        slider_y -= slider_gap
        self.dbscan_eps_slider = self._create_slider(
            x0, slider_y, width, slider_h, "DBSCAN eps", 0.01, 1.0, self.base_cfg.cluster.dbscan_eps, 0.01
        )
        slider_y -= slider_gap
        self.min_proto_slider = self._create_slider(
            x0, slider_y, width, slider_h, "Min proto score", 0.0, 1.0, self.base_cfg.overlap.min_prototype_score, 0.01
        )
        slider_y -= slider_gap
        self.mask_threshold_slider = self._create_slider(
            x0, slider_y, width, slider_h, "Mask threshold", 0.0, 1.0, self.base_cfg.overlap.mask_threshold, 0.01
        )
        slider_y -= slider_gap
        self.pixel_threshold_slider = self._create_slider(
            x0, slider_y, width, slider_h, "Pixel threshold", 0.0, 1.0, self.base_cfg.overlap.pixel_score_threshold, 0.01
        )

        self.status_text = self.fig.text(
            x0,
            0.02,
            "",
            fontsize=9,
            color="0.25",
            ha="left",
            va="bottom",
        )

    def _create_slider(
        self,
        x0: float,
        y0: float,
        width: float,
        height: float,
        label: str,
        vmin: float,
        vmax: float,
        valinit: float,
        step,
    ):
        label_width = 0.11
        label_gap = 0.008
        label_ax = self.fig.add_axes([x0, y0, label_width, height])
        label_ax.axis("off")
        label_ax.text(1.0, 0.5, label, ha="right", va="center", fontsize=9)

        slider_ax = self.fig.add_axes([x0 + label_width + label_gap, y0, width - label_width - label_gap, height])
        slider = Slider(slider_ax, "", vmin, vmax, valinit=valinit, valstep=step, dragging=False)
        slider.on_changed(self._on_widget_change)
        return slider

    def _optional_int_to_slider(self, value: Optional[int]) -> int:
        return 0 if value is None else int(value)

    def _slider_to_optional_int(self, value: float) -> Optional[int]:
        value = int(round(value))
        return None if value <= 0 else value

    def _toggle_state(self, label: str) -> bool:
        return bool(dict(zip(self.toggle_labels, self.toggle_checks.get_status()))[label])

    def _current_inference_cfg(self) -> PrototypeInferenceConfig:
        cfg = copy.deepcopy(self.base_cfg)
        cfg.ttt_steps = self._slider_to_optional_int(self.ttt_steps_slider.val)
        cfg.seed.quality_threshold = float(self.quality_threshold_slider.val)
        cfg.seed.min_foreground_prob = float(self.min_fg_slider.val)
        cfg.seed.topk = self._slider_to_optional_int(self.topk_slider.val)
        cfg.cluster.method = str(self.cluster_method_radio.value_selected)
        cfg.cluster.cluster_per_class = self._toggle_state("Cluster per class")
        cfg.cluster.graph_affinity_threshold = float(self.graph_affinity_slider.val)
        cfg.cluster.dbscan_eps = float(self.dbscan_eps_slider.val)
        cfg.assign.use_all_queries = self._toggle_state("Use all queries")
        cfg.assign.use_query_quality = self._toggle_state("Use query quality")
        cfg.assign.refinement_steps = int(round(self.refinement_slider.val))
        cfg.overlap.min_prototype_score = float(self.min_proto_slider.val)
        cfg.overlap.mask_threshold = float(self.mask_threshold_slider.val)
        cfg.overlap.pixel_score_threshold = float(self.pixel_threshold_slider.val)
        cfg.overlap.use_foreground_confidence = self._toggle_state("FG confidence")
        cfg.overlap.use_assignment_strength = self._toggle_state("Assign strength")
        return cfg

    def _status_summary(self, cfg: PrototypeInferenceConfig) -> str:
        topk = "all" if cfg.seed.topk is None else str(cfg.seed.topk)
        ttt = "ckpt" if cfg.ttt_steps is None else str(cfg.ttt_steps)
        return (
            f"method={cfg.cluster.method} | ttt={ttt} | quality>={cfg.seed.quality_threshold:.2f} | "
            f"topk={topk} | refine={cfg.assign.refinement_steps} | proto>={cfg.overlap.min_prototype_score:.2f}"
        )

    def _set_status(self, message: str, *, error: bool = False):
        self.status_text.set_text(message)
        self.status_text.set_color("tab:red" if error else "0.25")

    def _update_sample_text(self):
        total = max(len(self.images), 1)
        self.sample_text.set_text(f"Sample {self.current_index + 1}/{total}")

    def _mark_dirty(self):
        cfg = self._current_inference_cfg()
        self._dirty = True
        self._set_status(f"Pending changes. Click Apply. {self._status_summary(cfg)}")

    def _gt_signature_cache_key(self, cfg: PrototypeInferenceConfig) -> tuple:
        return (
            self.current_index,
            cfg.ttt_steps,
            repr(asdict(cfg.assign)),
        )

    def _get_cached_raw_outputs(self, cfg: PrototypeInferenceConfig):
        cache_key = (self.current_index, cfg.ttt_steps)
        raw = self._raw_cache.get(cache_key)
        if raw is None:
            raw = run_raw_outputs(
                self.system,
                [self.images[self.current_index]],
                inference_cfg=cfg,
                device=self.device,
            )
            self._raw_cache[cache_key] = raw
        return raw

    def refresh_predictions(self, _event=None):
        if self._suspend_callbacks:
            return

        cfg = self._current_inference_cfg()
        current_images = [self.images[self.current_index]]
        current_targets = [self.targets[self.current_index]]
        try:
            raw = self._get_cached_raw_outputs(cfg)
            predictions = run_predictions(
                self.system,
                current_images,
                raw=raw,
                inference_cfg=cfg,
                device=self.device,
            )
            signature_reference = None
            if self.include_gt_proto_predictions:
                gt_signature_cache_key = self._gt_signature_cache_key(cfg)
                signature_reference = self._gt_signature_cache.get(gt_signature_cache_key)
                if signature_reference is None:
                    signature_reference = run_gt_signature_reference(
                        self.system,
                        current_images,
                        current_targets,
                        raw=raw,
                        inference_cfg=cfg,
                        device=self.device,
                    )[0]
                    self._gt_signature_cache[gt_signature_cache_key] = signature_reference
        except Exception as exc:
            self._set_status(f"Prediction refresh failed: {exc}", error=True)
            self.fig.canvas.draw_idle()
            return

        _draw_interactive_sample_axes(
            self.display_axes,
            current_images[0],
            current_targets[0],
            predictions[0],
            inference_cfg=cfg,
            signature_reference=signature_reference,
            class_names=self.class_names,
        )
        self._update_sample_text()
        self._dirty = False
        self._set_status(self._status_summary(cfg))
        self.fig.canvas.draw_idle()

    def _on_widget_change(self, _value):
        if self._suspend_callbacks:
            return
        self._mark_dirty()

    def _on_apply_clicked(self, _event):
        self.refresh_predictions()

    def _step_sample(self, delta: int):
        if len(self.images) == 0:
            return
        self.current_index = (self.current_index + delta) % len(self.images)
        self.refresh_predictions()

    def _on_prev_sample(self, _event):
        self._step_sample(-1)

    def _on_next_sample(self, _event):
        self._step_sample(1)

    def _on_resample_clicked(self, _event):
        if self.sample_callback is None:
            return
        try:
            images, targets = self.sample_callback()
        except Exception as exc:
            self._set_status(f"Resample failed: {exc}", error=True)
            self.fig.canvas.draw_idle()
            return

        self.images = list(images)
        self.targets = list(targets)
        self.current_index = 0
        self._raw_cache.clear()
        self._gt_signature_cache.clear()
        self.refresh_predictions()

    def _on_reset_clicked(self, _event):
        self._suspend_callbacks = True
        try:
            self.ttt_steps_slider.set_val(self._optional_int_to_slider(self.base_cfg.ttt_steps))
            self.quality_threshold_slider.set_val(self.base_cfg.seed.quality_threshold)
            self.min_fg_slider.set_val(self.base_cfg.seed.min_foreground_prob)
            self.topk_slider.set_val(self._optional_int_to_slider(self.base_cfg.seed.topk))
            self.refinement_slider.set_val(self.base_cfg.assign.refinement_steps)
            self.graph_affinity_slider.set_val(self.base_cfg.cluster.graph_affinity_threshold)
            self.dbscan_eps_slider.set_val(self.base_cfg.cluster.dbscan_eps)
            self.min_proto_slider.set_val(self.base_cfg.overlap.min_prototype_score)
            self.mask_threshold_slider.set_val(self.base_cfg.overlap.mask_threshold)
            self.pixel_threshold_slider.set_val(self.base_cfg.overlap.pixel_score_threshold)

            if self.cluster_method_radio.value_selected != self.base_cfg.cluster.method:
                self.cluster_method_radio.set_active(self.available_cluster_methods.index(self.base_cfg.cluster.method))

            desired_checks = [
                self.base_cfg.cluster.cluster_per_class,
                self.base_cfg.assign.use_all_queries,
                self.base_cfg.assign.use_query_quality,
                self.base_cfg.overlap.use_foreground_confidence,
                self.base_cfg.overlap.use_assignment_strength,
            ]
            for idx, (current_state, desired_state) in enumerate(zip(self.toggle_checks.get_status(), desired_checks)):
                if bool(current_state) != bool(desired_state):
                    self.toggle_checks.set_active(idx)
        finally:
            self._suspend_callbacks = False

        self._mark_dirty()
        self.refresh_predictions()


def show_interactive_prediction_grid(
    system: PanopticSystem,
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    *,
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
    window_title: Optional[str] = None,
    include_gt_proto_predictions: bool = True,
    device: Optional[torch.device] = None,
    sample_callback: Optional[Callable[[], Tuple[List[torch.Tensor], List[dict]]]] = None,
    block: bool = True,
):
    viewer = InteractivePredictionGrid(
        system,
        images,
        targets,
        class_names=class_names,
        figure_title=figure_title,
        window_title=window_title,
        include_gt_proto_predictions=include_gt_proto_predictions,
        device=device,
        sample_callback=sample_callback,
    )
    if block:
        plt.show()
    return viewer
