from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

try:
    import umap
except ImportError:
    umap = None

from .dataset import SyntheticPanopticBatchGenerator
from .mask_aggregation import project_mask_embeddings
from .outputs import FlatQueryOutputs, ResolvedPrediction
from .panoptic import PanopticSystem
from .signature_ops import pairwise_distance


DEFAULT_CLASS_NAMES = ["Background", "Square", "Triangle"]


@dataclass
class _InstanceArtistGroup:
    overlays: List[object]
    rect: Optional[patches.Rectangle]
    text: object


@dataclass
class _SelectableInstance:
    kind: str
    index: int
    mask: np.ndarray
    gt_indices: List[int]
    artists: _InstanceArtistGroup


@dataclass
class _UmapArtistGroup:
    scatter: object
    text: Optional[object]
    base_size: Optional[float] = None


@dataclass
class _PredictionPanelState:
    axis: object
    image_np: np.ndarray
    prediction: ResolvedPrediction
    gt_masks: List[np.ndarray]
    class_names: Optional[Sequence[str]]
    title: str


@dataclass
class _RowInteractionState:
    gt_masks: List[np.ndarray]
    axis_instances: dict = field(default_factory=dict)
    umap_gt_artists: List[_UmapArtistGroup] = field(default_factory=list)
    umap_query_artists: List[_UmapArtistGroup] = field(default_factory=list)
    umap_query_points: Optional[np.ndarray] = None
    umap_query_pick_radius: float = 0.0
    umap_query_axis: Optional[object] = None
    prediction_panel: Optional[_PredictionPanelState] = None
    selected_axis: Optional[object] = None
    selected_index: Optional[int] = None
    selected_query_index: Optional[int] = None


def _filter_background_instances(
    masks: Sequence[np.ndarray],
    labels: Sequence[int],
    scores: Optional[Sequence[float]] = None,
):
    keep_indices = [idx for idx, label in enumerate(labels) if int(label) != 0]
    filtered_masks = [masks[idx] for idx in keep_indices]
    filtered_labels = [labels[idx] for idx in keep_indices]

    if scores is None:
        return filtered_masks, filtered_labels, None

    filtered_scores = [scores[idx] for idx in keep_indices]
    return filtered_masks, filtered_labels, filtered_scores


def _foreground_instance_indices(labels: Sequence[int]) -> List[int]:
    return [idx for idx, label in enumerate(labels) if int(label) != 0]


def sample_synthetic_examples(
    *,
    num_samples: int,
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
        batch_images, batch_targets = generator.generate_batch(
            batch_size=num_samples,
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

    if umap is not None:
        n_neighbors = max(2, min(15, signatures.shape[0] - 1))
        signatures_t = torch.from_numpy(signatures)
        distances = pairwise_distance(signatures_t, signatures_t, clamp=True).detach().cpu().numpy()
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.15,
            metric="precomputed",
            random_state=0,
        )
        return reducer.fit_transform(distances).astype(np.float32, copy=False)

    centered = signatures - signatures.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    if basis.shape[1] < 2:
        basis = np.pad(basis, ((0, 0), (0, 2 - basis.shape[1])))
    return (centered @ basis[:, :2]).astype(np.float32, copy=False)


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


def _matching_gt_indices(mask: np.ndarray, gt_masks: Sequence[np.ndarray]) -> List[int]:
    mask_bool = np.asarray(mask).astype(bool)
    matches = []
    for idx, gt_mask in enumerate(gt_masks):
        gt_mask_bool = np.asarray(gt_mask).astype(bool)
        if np.logical_and(mask_bool, gt_mask_bool).any():
            matches.append(idx)
    return matches


def _set_instance_highlight(instance: _SelectableInstance, highlighted: bool):
    overlay_alpha = 0.6 if highlighted else 0.08
    rect_alpha = 1.0 if highlighted else 0.18
    text_alpha = 0.95 if highlighted else 0.0
    line_width = 2.6 if highlighted else 1.0

    for overlay in instance.artists.overlays:
        overlay.set_visible(True)
        alpha_data = overlay.get_array()
        if alpha_data is not None and alpha_data.ndim == 3 and alpha_data.shape[-1] == 4:
            alpha_data[..., 3] = np.where(alpha_data[..., 3] > 0, overlay_alpha, 0.0)
            overlay.set_data(alpha_data)

    if instance.artists.rect is not None:
        instance.artists.rect.set_visible(True)
        instance.artists.rect.set_alpha(rect_alpha)
        instance.artists.rect.set_linewidth(line_width)

    if instance.artists.text is not None:
        instance.artists.text.set_visible(True)
        instance.artists.text.set_visible(highlighted)
        instance.artists.text.set_alpha(text_alpha)


def _hide_instance(instance: _SelectableInstance):
    for overlay in instance.artists.overlays:
        overlay.set_visible(False)
    if instance.artists.rect is not None:
        instance.artists.rect.set_visible(False)
    if instance.artists.text is not None:
        instance.artists.text.set_visible(False)


def _mask_overlap_score(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a).astype(bool)
    b_bool = np.asarray(b).astype(bool)
    intersection = float(np.logical_and(a_bool, b_bool).sum())
    if intersection <= 0.0:
        return 0.0
    union = float(np.logical_or(a_bool, b_bool).sum())
    if union <= 0.0:
        return 0.0
    return intersection / union


def _select_single_detection_for_gt(gt_mask: np.ndarray, instances: Sequence[_SelectableInstance]) -> Optional[int]:
    if len(instances) == 0:
        return None

    costs = np.array([[1.0 - _mask_overlap_score(gt_mask, instance.mask) for instance in instances]], dtype=np.float32)
    row_ind, col_ind = linear_sum_assignment(costs)
    if len(col_ind) == 0:
        return None
    return int(col_ind[0])


def _select_single_gt_for_detection(det_mask: np.ndarray, gt_masks: Sequence[np.ndarray]) -> Optional[int]:
    if len(gt_masks) == 0:
        return None

    costs = np.array([[1.0 - _mask_overlap_score(det_mask, gt_mask) for gt_mask in gt_masks]], dtype=np.float32)
    row_ind, col_ind = linear_sum_assignment(costs)
    if len(col_ind) == 0:
        return None
    return int(col_ind[0])


def _set_umap_gt_highlight(artist_group: _UmapArtistGroup, highlighted: bool):
    artist_group.scatter.set_alpha(1.0 if highlighted else 0.15)
    artist_group.scatter.set_linewidths(2.2 if highlighted else 0.8)
    if artist_group.text is not None:
        artist_group.text.set_alpha(1.0 if highlighted else 0.2)


def _set_umap_query_highlight(artist_group: _UmapArtistGroup, highlighted: bool):
    if artist_group.base_size is not None:
        base_size = float(artist_group.base_size)
    else:
        sizes = np.asarray(artist_group.scatter.get_sizes(), dtype=np.float32)
        base_size = float(sizes[0]) if sizes.size > 0 else 18.0
    artist_group.scatter.set_alpha(1.0 if highlighted else 0.15)
    artist_group.scatter.set_sizes([base_size * 1.8 if highlighted else base_size])
    artist_group.scatter.set_linewidths(2.2 if highlighted else 0.8)


def _hit_test_umap_queries(row_state: _RowInteractionState, x: float, y: float) -> Optional[int]:
    if row_state.umap_query_points is None or x is None or y is None:
        return None

    points = np.asarray(row_state.umap_query_points, dtype=np.float32)
    if points.shape[0] == 0:
        return None

    deltas = points - np.array([x, y], dtype=np.float32)
    dist2 = np.sum(deltas * deltas, axis=1)
    hit_idx = int(np.argmin(dist2))
    if row_state.umap_query_pick_radius > 0.0 and dist2[hit_idx] > row_state.umap_query_pick_radius ** 2:
        return None
    return hit_idx


def _single_query_preview(
    prediction: ResolvedPrediction,
    query_index: int,
):
    flat = prediction.flat_queries
    if flat is None:
        return [], [], [], "Query preview"

    features = flat.features
    q_mask_emb = flat.mask_embeddings[query_index]
    q_cls_prob = flat.class_probabilities[query_index]
    q_seed = flat.seed_scores

    mask_logits = project_mask_embeddings(
        q_mask_emb.view(1, 1, -1),
        features.unsqueeze(0),
        (flat.image_height, flat.image_width),
    )[0, 0]

    mask = mask_logits > 0.0
    if not bool(mask.any()):
        flat_logits = mask_logits.flatten()
        top_k = max(1, int(0.05 * flat_logits.numel()))
        top_idx = torch.topk(flat_logits, k=top_k).indices
        mask = torch.zeros_like(mask_logits, dtype=torch.bool).flatten()
        mask[top_idx] = True
        mask = mask.view_as(mask_logits)

    label = int(q_cls_prob.argmax().item())
    score = float(q_seed[query_index].item()) if q_seed.numel() > 0 else float(q_cls_prob.max().item())

    return [mask.detach().cpu().numpy()], [label], [score], f"Query {query_index} by itself"


def _render_prediction_panel(
    row_state: _RowInteractionState,
    *,
    query_index: Optional[int] = None,
):
    panel = row_state.prediction_panel
    if panel is None:
        return

    if query_index is None:
        masks = [mask.detach().cpu().numpy() for mask in panel.prediction.resolved_masks]
        labels = [int(label) for label in panel.prediction.resolved_labels]
        scores = [float(score) for score in panel.prediction.resolved_scores]
        title = panel.title
    else:
        masks, labels, scores, title = _single_query_preview(panel.prediction, query_index)

    panel.axis.clear()
    instances = _draw_instances(
        panel.axis,
        panel.image_np,
        masks,
        labels,
        scores=scores,
        class_names=panel.class_names,
        title=title,
        gt_masks=panel.gt_masks,
        kind="prediction",
    )
    row_state.axis_instances[panel.axis] = instances


def _select_instance(row_state: _RowInteractionState, axis, instance_index: int):
    panel_axis = row_state.prediction_panel.axis if row_state.prediction_panel is not None else None
    selected = row_state.axis_instances.get(axis, [])[instance_index]

    if row_state.selected_query_index is not None and panel_axis is not None and axis is not panel_axis:
        _render_prediction_panel(row_state, query_index=None)
        row_state.selected_query_index = None
        for artist_group in row_state.umap_query_artists:
            _set_umap_query_highlight(artist_group, False)
        selected = row_state.axis_instances.get(axis, [])[instance_index]

    row_state.selected_axis = axis
    row_state.selected_index = instance_index

    selected_gt_indices = set(selected.gt_indices)

    if selected.kind == "gt":
        gt_mask = selected.mask
        matched_detection_indices = {}
        for current_axis, instances in row_state.axis_instances.items():
            if len(instances) == 0 or instances[0].kind != "prediction":
                continue
            match_idx = _select_single_detection_for_gt(gt_mask, instances)
            matched_detection_indices[current_axis] = match_idx

        for current_axis, instances in row_state.axis_instances.items():
            for idx, instance in enumerate(instances):
                if current_axis is axis and idx == instance_index:
                    _set_instance_highlight(instance, True)
                elif instance.kind == "gt":
                    _set_instance_highlight(instance, instance.index in selected_gt_indices)
                elif current_axis in matched_detection_indices and matched_detection_indices[current_axis] == idx:
                    _set_instance_highlight(instance, True)
                else:
                    _hide_instance(instance)

        for idx, artist_group in enumerate(row_state.umap_gt_artists):
            _set_umap_gt_highlight(artist_group, idx in selected_gt_indices)
        return

    matched_gt_idx = _select_single_gt_for_detection(selected.mask, row_state.gt_masks)
    matched_gt_indices = {matched_gt_idx} if matched_gt_idx is not None else set()

    for current_axis, instances in row_state.axis_instances.items():
        for idx, instance in enumerate(instances):
            if current_axis is axis and idx == instance_index:
                is_highlighted = True
            elif instance.kind == "gt":
                is_highlighted = instance.index in matched_gt_indices
            else:
                is_highlighted = False
            _set_instance_highlight(instance, is_highlighted)

    for idx, artist_group in enumerate(row_state.umap_gt_artists):
        _set_umap_gt_highlight(artist_group, idx in matched_gt_indices)


def _clear_selection(row_state: _RowInteractionState):
    row_state.selected_axis = None
    row_state.selected_index = None
    row_state.selected_query_index = None

    _render_prediction_panel(row_state, query_index=None)

    for instances in row_state.axis_instances.values():
        for instance in instances:
            _set_instance_highlight(instance, True)

    for artist_group in row_state.umap_gt_artists:
        _set_umap_gt_highlight(artist_group, True)

    for artist_group in row_state.umap_query_artists:
        _set_umap_query_highlight(artist_group, False)


def _hit_test_masks(masks: Sequence[np.ndarray], x: float, y: float) -> Optional[int]:
    if x is None or y is None:
        return None

    px = int(round(x))
    py = int(round(y))
    for idx in reversed(range(len(masks))):
        mask = np.asarray(masks[idx]).astype(bool)
        if py < 0 or px < 0 or py >= mask.shape[0] or px >= mask.shape[1]:
            continue
        if mask[py, px]:
            return idx
    return None


def _draw_signature_umap(
    ax,
    image_np: np.ndarray,
    target: dict,
    prediction: ResolvedPrediction,
    *,
    class_names: Optional[Sequence[str]] = None,
    title: str,
    row_state: Optional[_RowInteractionState] = None,
):
    flat: Optional[FlatQueryOutputs] = prediction.flat_queries
    q_sig = None if flat is None else flat.signature_embeddings
    q_seed = None if flat is None else flat.seed_scores
    q_influence = None if flat is None else flat.influence_scores
    gt_sig = prediction.all_signature_embeddings

    if q_sig is None or gt_sig is None:
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, "No signatures", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return

    q_sig_np = q_sig.detach().cpu().numpy()
    gt_sig_np = gt_sig.detach().cpu().numpy()

    gt_masks_all = [mask.detach().cpu().numpy() for mask in target["masks"]]
    gt_labels_all = [int(label) for label in target["labels"].detach().cpu().tolist()]
    fg_indices = _foreground_instance_indices(gt_labels_all)
    gt_masks = [gt_masks_all[idx] for idx in fg_indices]
    gt_labels = [gt_labels_all[idx] for idx in fg_indices]
    if gt_sig_np.shape[0] == len(gt_labels_all):
        gt_sig_np = gt_sig_np[fg_indices]

    all_sig = np.concatenate([q_sig_np, gt_sig_np], axis=0) if gt_sig_np.shape[0] > 0 else q_sig_np
    embedding = _project_signatures_2d(all_sig)
    q_pts = embedding[: q_sig_np.shape[0]]
    gt_pts = embedding[q_sig_np.shape[0] :]

    gt_colors = [_instance_color(image_np, mask) for mask in gt_masks]
    gt_marker_size = 150.0
    min_query_marker_size = 18.0
    max_query_marker_size = 300.0

    ax.set_title(title)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    if q_sig_np.shape[0] > 1:
        q_sig_t = torch.from_numpy(q_sig_np)
        pairwise_distances = pairwise_distance(q_sig_t, q_sig_t, clamp=True).detach().cpu().numpy()
        np.fill_diagonal(pairwise_distances, np.inf)
        neighbor_order = np.argsort(pairwise_distances, axis=1)
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
    
    if q_seed is not None:
        q_seed_np = q_seed.detach().cpu().numpy()
        query_sizes = min_query_marker_size + (max_query_marker_size - min_query_marker_size) * np.clip(q_seed_np, 0.0, 1.0)
    elif q_influence is not None:
        q_influence_np = q_influence.detach().cpu().numpy()
        influence_min = float(np.min(q_influence_np))
        influence_max = float(np.max(q_influence_np))
        if influence_max > influence_min:
            influence_norm = (q_influence_np - influence_min) / (influence_max - influence_min)
        else:
            influence_norm = np.ones_like(q_influence_np, dtype=np.float32)
        query_sizes = min_query_marker_size + (max_query_marker_size - min_query_marker_size) * np.clip(influence_norm, 0.0, 1.0)
    else:
        query_sizes = np.full((q_pts.shape[0],), min_query_marker_size, dtype=np.float32)

    query_colors = np.full((q_pts.shape[0], 3), 0.7, dtype=np.float32)
    query_alpha = np.full((q_pts.shape[0],), 0.72, dtype=np.float32)

    if row_state is not None:
        row_state.umap_query_points = q_pts
        row_state.umap_query_pick_radius = 0.03 * max(
            float(np.ptp(q_pts[:, 0])) if q_pts.shape[0] > 0 else 1.0,
            float(np.ptp(q_pts[:, 1])) if q_pts.shape[0] > 0 else 1.0,
            1.0,
        )
        row_state.umap_query_axis = ax
        row_state.umap_query_artists = []

    for idx in range(q_pts.shape[0]):
        scatter = ax.scatter(
            q_pts[idx, 0],
            q_pts[idx, 1],
            s=float(query_sizes[idx]),
            c=[query_colors[idx]],
            alpha=float(query_alpha[idx]),
            marker=".",
            linewidths=0,
            zorder=2,
        )
        if row_state is not None:
            row_state.umap_query_artists.append(
                _UmapArtistGroup(scatter=scatter, text=None, base_size=float(query_sizes[idx]))
            )

    for idx, pt in enumerate(gt_pts):
        label = gt_labels[idx] if idx < len(gt_labels) else idx
        marker = _gt_marker(label)
        color = gt_colors[idx] if idx < len(gt_colors) else np.array([0.2, 0.2, 0.2], dtype=np.float32)
        class_name = str(label)
        if class_names is not None and 0 <= label < len(class_names):
            class_name = class_names[label]

        scatter = ax.scatter(
            pt[0],
            pt[1],
            s=gt_marker_size,
            c=[color],
            marker=marker,
            edgecolors="black",
            linewidths=1.2,
            zorder=3,
        )
        text = ax.text(
            pt[0],
            pt[1],
            f" {class_name}",
            fontsize=8,
            color="black",
            va="center",
            ha="left",
            zorder=4,
        )
        if row_state is not None:
            row_state.umap_gt_artists.append(_UmapArtistGroup(scatter=scatter, text=text))

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
    gt_masks: Optional[Sequence[np.ndarray]] = None,
    kind: str = "prediction",
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
        return []

    cmap = plt.get_cmap("tab20")
    instances = []
    for idx, mask in enumerate(masks):
        mask_bool = np.asarray(mask).astype(bool)
        if not mask_bool.any():
            continue

        color = np.array(cmap(idx % 20)[:3])
        overlay = np.zeros((*mask_bool.shape, 4), dtype=np.float32)
        overlay[mask_bool, :3] = color
        overlay[mask_bool, 3] = 0.6
        overlay_artist = ax.imshow(overlay)

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

        text = ax.text(
            xmin,
            max(ymin - 4, 0),
            label_text,
            fontsize=9,
            color="white",
            bbox={"facecolor": color, "alpha": 0.85, "pad": 2},
        )
        instances.append(
            _SelectableInstance(
                kind=kind,
                index=idx,
                mask=mask_bool,
                gt_indices=[idx] if kind == "gt" else _matching_gt_indices(mask_bool, gt_masks or []),
                artists=_InstanceArtistGroup(overlays=[overlay_artist], rect=rect, text=text),
            )
        )

    return instances


@torch.no_grad()
def run_predictions(
    system: PanopticSystem,
    images: Sequence[torch.Tensor],
    *,
    device: Optional[torch.device] = None,
):
    if len(images) == 0:
        return []

    model_device = device
    if model_device is None:
        model_device = next(system.parameters()).device

    batch = torch.stack(list(images)).to(model_device)
    was_training = system.training
    system.eval()
    try:
        predictions = system.predict(batch)
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
    device: Optional[torch.device] = None,
):
    if len(images) == 0:
        return []

    model_device = device
    if model_device is None:
        model_device = next(system.parameters()).device

    batch = torch.stack(list(images)).to(model_device)
    was_training = system.training
    system.eval()
    try:
        predictions = system.predict_with_gt_prototypes(batch, targets)
    finally:
        system.train(was_training)

    if isinstance(predictions, list):
        return predictions
    return [predictions]


def render_prediction_grid(
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    prediction_columns: Sequence[Tuple[str, Sequence[dict]]],
    *,
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
    interactive: bool = False,
):
    num_samples = len(images)
    add_signature_column = any("GT Prototype" in title for title, _ in prediction_columns)
    num_cols = 2 + len(prediction_columns) + int(add_signature_column)
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(5 * num_cols, 5 * max(num_samples, 1)), squeeze=False)
    row_states: List[_RowInteractionState] = []

    if figure_title:
        fig.suptitle(figure_title, fontsize=14)

    for row_idx, (image, target) in enumerate(zip(images, targets)):
        image_np = _to_numpy_image(image)
        row_state = _RowInteractionState(gt_masks=[])
        row_states.append(row_state)
        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_title("Input")
        axes[row_idx, 0].axis("off")

        gt_masks = [mask.detach().cpu().numpy() for mask in target["masks"]]
        gt_labels = [int(label) for label in target["labels"].detach().cpu().tolist()]
        gt_masks, gt_labels, _ = _filter_background_instances(gt_masks, gt_labels)
        row_state.gt_masks = gt_masks
        gt_instances = _draw_instances(
            axes[row_idx, 1],
            image_np,
            gt_masks,
            gt_labels,
            class_names=class_names,
            title="Ground Truth",
            gt_masks=gt_masks,
            kind="gt",
        )
        row_state.axis_instances[axes[row_idx, 1]] = gt_instances

        next_col_idx = 2
        for column_title, predictions in prediction_columns:
            prediction = predictions[row_idx]
            pred_masks = [mask.detach().cpu().numpy() for mask in prediction.resolved_masks]
            pred_labels = [int(label) for label in prediction.resolved_labels]
            pred_scores = [float(score) for score in prediction.resolved_scores]
            pred_masks, pred_labels, pred_scores = _filter_background_instances(
                pred_masks,
                pred_labels,
                pred_scores,
            )
            pred_axis = axes[row_idx, next_col_idx]
            pred_instances = _draw_instances(
                pred_axis,
                image_np,
                pred_masks,
                pred_labels,
                scores=pred_scores,
                class_names=class_names,
                title=column_title,
                gt_masks=gt_masks,
                kind="prediction",
            )
            row_state.axis_instances[pred_axis] = pred_instances
            if add_signature_column and "GT Prototype" in column_title:
                row_state.prediction_panel = _PredictionPanelState(
                    axis=pred_axis,
                    image_np=image_np,
                    prediction=prediction,
                    gt_masks=gt_masks,
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
                    row_state=row_state,
                )
                next_col_idx += 1

        if interactive:
            _clear_selection(row_state)

    if interactive:
        axis_to_row_state = {
            axis: row_state
            for row_state in row_states
            for axis in row_state.axis_instances
        }
        for row_state in row_states:
            if row_state.umap_query_axis is not None:
                axis_to_row_state[row_state.umap_query_axis] = row_state

        def _on_click(event):
            row_state = axis_to_row_state.get(event.inaxes)
            if row_state is None:
                return

            if event.inaxes is row_state.umap_query_axis:
                query_idx = _hit_test_umap_queries(row_state, event.xdata, event.ydata)
                if query_idx is None:
                    _clear_selection(row_state)
                else:
                    row_state.selected_axis = None
                    row_state.selected_index = None
                    row_state.selected_query_index = query_idx
                    _render_prediction_panel(row_state, query_index=query_idx)
                    for idx, artist_group in enumerate(row_state.umap_query_artists):
                        _set_umap_query_highlight(artist_group, idx == query_idx)

                    preview = row_state.prediction_panel.prediction if row_state.prediction_panel is not None else None
                    if preview is not None:
                        preview_masks, _, _, _ = _single_query_preview(preview, query_idx)
                        matched_gt_idx = None
                        if len(preview_masks) > 0:
                            matched_gt_idx = _select_single_gt_for_detection(preview_masks[0], row_state.gt_masks)
                        matched_gt_indices = {matched_gt_idx} if matched_gt_idx is not None else set()
                        for idx, artist_group in enumerate(row_state.umap_gt_artists):
                            _set_umap_gt_highlight(artist_group, idx in matched_gt_indices)
                fig.canvas.draw_idle()
                return

            instance_idx = _hit_test_masks(
                [instance.mask for instance in row_state.axis_instances[event.inaxes]],
                event.xdata,
                event.ydata,
            )
            if instance_idx is None:
                _clear_selection(row_state)
            else:
                _select_instance(row_state, event.inaxes, instance_idx)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_press_event", _on_click)

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
    prediction_columns = [("Clustered Prediction", predictions)]
    if gt_proto_predictions is not None:
        prediction_columns.append(("GT Prototype Prediction", gt_proto_predictions))
    fig = render_prediction_grid(
        images,
        targets,
        prediction_columns,
        class_names=class_names,
        figure_title=figure_title,
        interactive=False,
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
    prediction_columns = [("Clustered Prediction", predictions)]
    if gt_proto_predictions is not None:
        prediction_columns.append(("GT Prototype Prediction", gt_proto_predictions))
    fig = render_prediction_grid(
        images,
        targets,
        prediction_columns,
        class_names=class_names,
        figure_title=figure_title,
        interactive=True,
    )
    if window_title and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(window_title)
    plt.show()
    plt.close(fig)
