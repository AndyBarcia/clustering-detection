from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
from .outputs import EvaluationPredictionSet, FlatQueryOutputs, ResolvedPrediction
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
    base_color: Optional[np.ndarray] = None
    base_alpha: float = 0.72


@dataclass
class _PredictionPanelState:
    axis: object
    image_np: np.ndarray
    prediction: ResolvedPrediction
    gt_masks: List[np.ndarray]
    class_names: Optional[Sequence[str]]
    title: str
    mode: str = "instances"
    info_title: Optional[str] = None
    info_text: Optional[str] = None
    info_payload: Optional[dict] = None


@dataclass
class _SelectedPixelState:
    axis: object
    px: int
    py: int


@dataclass
class _RowInteractionState:
    gt_masks: List[np.ndarray]
    axis_instances: dict = field(default_factory=dict)
    pixel_click_sources: dict = field(default_factory=dict)
    umap_gt_artists: List[_UmapArtistGroup] = field(default_factory=list)
    umap_query_artists: List[_UmapArtistGroup] = field(default_factory=list)
    umap_query_points: Optional[np.ndarray] = None
    umap_query_pick_radius: float = 0.0
    umap_query_axis: Optional[object] = None
    prediction: Optional[ResolvedPrediction] = None
    class_names: Optional[Sequence[str]] = None
    prediction_panels: List[_PredictionPanelState] = field(default_factory=list)
    query_heatmap_panels: List[_PredictionPanelState] = field(default_factory=list)
    query_info_panel: Optional[_PredictionPanelState] = None
    pixel_distribution_panel: Optional[_PredictionPanelState] = None
    selected_axis: Optional[object] = None
    selected_index: Optional[int] = None
    selected_query_index: Optional[int] = None
    selected_pixel: Optional[_SelectedPixelState] = None
    selected_pixel_artists: dict = field(default_factory=dict)
    query_contributions: Optional[np.ndarray] = None


@dataclass
class PredictionBundle:
    clustered: ResolvedPrediction
    gt_signatures: Optional[ResolvedPrediction] = None
    golden_queries: Optional[ResolvedPrediction] = None


@dataclass
class DetailedPredictionBundle:
    image: torch.Tensor
    target: dict
    raw_feature_maps: Dict[str, torch.Tensor]
    predictions: PredictionBundle
    identity_similarity_metric: str = "cosine"


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


def _project_signatures_2d(signatures: np.ndarray, *, metric: str = "cosine") -> np.ndarray:
    if signatures.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    if signatures.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)

    signatures = signatures.astype(np.float32, copy=False)

    if umap is not None:
        n_neighbors = max(2, min(15, signatures.shape[0] - 1))
        signatures_t = torch.from_numpy(signatures)
        distances = pairwise_distance(
            signatures_t,
            signatures_t,
            metric=metric,
            clamp=True,
        ).detach().cpu().numpy()
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
    if a_bool.shape != b_bool.shape:
        target_shape = b_bool.shape
        a_tensor = torch.as_tensor(a_bool, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        a_resized = torch.nn.functional.interpolate(
            a_tensor,
            size=target_shape,
            mode="nearest",
        )[0, 0].numpy() > 0.5
        a_bool = a_resized
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


def _refresh_umap_query_artists(row_state: _RowInteractionState):
    if len(row_state.umap_query_artists) == 0:
        return

    contributions = row_state.query_contributions
    abs_contributions = None
    max_abs = 0.0
    if contributions is not None:
        abs_contributions = np.abs(contributions).astype(np.float32, copy=False)
        max_abs = float(abs_contributions.max()) if abs_contributions.size > 0 else 0.0

    for idx, artist_group in enumerate(row_state.umap_query_artists):
        base_size = float(artist_group.base_size) if artist_group.base_size is not None else 18.0
        size = base_size
        alpha = float(artist_group.base_alpha)
        color = artist_group.base_color if artist_group.base_color is not None else np.array([0.7, 0.7, 0.7], dtype=np.float32)
        linewidth = 0.0
        edgecolor = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if contributions is not None and max_abs > 0.0:
            contribution = float(contributions[idx]) if idx < len(contributions) else 0.0
            magnitude = abs(contribution) / max_abs
            size = base_size + 220.0 * magnitude
            color = (
                np.array([0.15, 0.7, 0.2], dtype=np.float32)
                if contribution >= 0.0
                else np.array([0.85, 0.2, 0.2], dtype=np.float32)
            )
            alpha = 0.15 + 0.85 * magnitude

        if row_state.selected_query_index == idx:
            size *= 1.25
            linewidth = 2.2

        artist_group.scatter.set_sizes([size])
        artist_group.scatter.set_color([color])
        artist_group.scatter.set_alpha(alpha)
        artist_group.scatter.set_linewidths(linewidth)
        artist_group.scatter.set_edgecolors([edgecolor])


def _clear_query_contribution_overlay(row_state: _RowInteractionState):
    row_state.query_contributions = None
    _refresh_umap_query_artists(row_state)


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
    *,
    feature_map: Optional[torch.Tensor] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
):
    flat = prediction.flat_queries
    if flat is None:
        return None, "Query preview", None

    features = flat.features if feature_map is None else feature_map
    if image_shape is None:
        if feature_map is None:
            image_shape = (flat.image_height, flat.image_width)
        else:
            image_shape = tuple(int(v) for v in feature_map.shape[-2:])
    q_mask_emb = flat.mask_embeddings[query_index]
    q_cls_prob = flat.class_probabilities[query_index]
    q_seed_score = float(flat.seed_scores[query_index].item()) if flat.seed_scores.numel() > 0 else None
    q_influence = float(flat.influence_scores[query_index].item()) if flat.influence_scores.numel() > 0 else None
    q_foreground = float(flat.foreground_confidence[query_index].item())
    q_background = float(flat.background_confidence[query_index].item())
    q_partition = float(flat.partition_confidence[query_index].item())

    mask_logits = project_mask_embeddings(
        q_mask_emb.view(1, 1, -1),
        features.unsqueeze(0),
        image_shape,
    )[0, 0]

    label = int(q_cls_prob.argmax().item())
    class_confidence = float(q_cls_prob[label].item())
    layer_idx = query_index // flat.queries_per_layer
    slot_idx = query_index % flat.queries_per_layer

    info_lines = [
        f"query={query_index} layer={layer_idx} slot={slot_idx}",
        f"class={label} p={class_confidence:.3f}",
        f"seed={q_seed_score:.3f}" if q_seed_score is not None else "seed=n/a",
        f"influence={q_influence:.3f}" if q_influence is not None else "influence=n/a",
        f"fg={q_foreground:.3f} bg={q_background:.3f}",
        f"partition={q_partition:.3f}",
    ]

    if prediction.prototypes is not None:
        source_query_indices = prediction.prototypes.source_query_indices
        if source_query_indices.numel() > 0:
            selected_positions = torch.where(source_query_indices == query_index)[0]
            if selected_positions.numel() > 0:
                cluster_label = int(prediction.prototypes.source_cluster_labels[selected_positions[0]].item())
                info_lines.append(f"selected_seed=yes cluster={cluster_label}")
            else:
                info_lines.append("selected_seed=no")

        if prediction.prototypes.assignment_weights.numel() > 0:
            query_weights = prediction.prototypes.assignment_weights[query_index]
            best_proto = int(torch.argmax(query_weights).item())
            best_weight = float(query_weights[best_proto].item())
            info_lines.append(f"best_proto={best_proto} weight={best_weight:.3f}")

    title = title or "Selected query logits"
    info_text = "\n".join(info_lines)

    return mask_logits.detach().cpu().numpy(), title, info_text


def _draw_query_logit_heatmap(
    ax,
    logit_map: Optional[np.ndarray],
    *,
    title: str,
):
    ax.clear()
    ax.set_title(title)
    ax.axis("off")
    if logit_map is None:
        ax.text(0.5, 0.5, "No query preview", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return

    vmax = float(np.max(np.abs(logit_map)))
    vmax = max(vmax, 1e-6)
    ax.imshow(logit_map, cmap="coolwarm", interpolation="nearest", vmin=-vmax, vmax=vmax)


def _draw_query_info_panel(ax, *, title: str, info_text: Optional[str]):
    ax.clear()
    ax.set_title(title)
    ax.axis("off")
    if not info_text:
        ax.text(0.5, 0.5, "No query selected", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return

    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        color="black",
        bbox={"facecolor": "white", "alpha": 0.95, "pad": 4},
    )


def _draw_pixel_distribution_panel(ax, *, title: str, payload: Optional[dict]):
    ax.clear()
    ax.set_title(title)
    if not payload or payload.get("probabilities") is None or len(payload["probabilities"]) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No pixel selected", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return

    probabilities = np.asarray(payload["probabilities"], dtype=np.float32)
    labels = payload["labels"]
    colors = payload["colors"]
    y = np.arange(len(probabilities))
    ax.barh(y, probabilities, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.0, 1.0)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2)
    xlabel = payload.get("xlabel", "Probability")
    ax.set_xlabel(xlabel)
    for idx, prob in enumerate(probabilities):
        ax.text(min(float(prob) + 0.02, 0.98), idx, f"{prob:.2f}", va="center", ha="left", fontsize=9)


def _prediction_masks_for_feature_map(
    prediction: ResolvedPrediction,
    feature_map: torch.Tensor,
    *,
    image_height: Optional[int] = None,
    image_width: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    if prediction.mask_embeddings.shape[0] == 0:
        return [], [], []

    if image_height is None or image_width is None:
        image_height, image_width = feature_map.shape[-2:]

    mask_logits = project_mask_embeddings(
        prediction.mask_embeddings.unsqueeze(0),
        feature_map.unsqueeze(0),
        (image_height, image_width),
    )[0]
    mask_probabilities = torch.softmax(mask_logits, dim=0)
    pixel_scores = mask_probabilities * prediction.scores[:, None, None]
    max_pixel_score, winners = pixel_scores.max(dim=0)

    masks = []
    labels = []
    scores = []
    for prototype_idx in range(prediction.mask_embeddings.shape[0]):
        mask = winners == prototype_idx
        if not bool(mask.any()):
            continue
        masks.append(mask.detach().cpu().numpy())
        labels.append(int(prediction.resolved_labels[prototype_idx]))
        scores.append(float(prediction.resolved_scores[prototype_idx]))
    return masks, labels, scores


def _compute_query_logits(
    prediction: ResolvedPrediction,
    *,
    feature_map: Optional[torch.Tensor] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Optional[torch.Tensor]:
    flat = prediction.flat_queries
    if flat is None:
        return None
    if feature_map is None:
        feature_map = flat.features
    if image_shape is None:
        image_shape = tuple(int(v) for v in feature_map.shape[-2:])
    return project_mask_embeddings(
        flat.mask_embeddings.unsqueeze(0),
        feature_map.unsqueeze(0),
        image_shape,
    )[0]


def _compute_pixel_analysis(
    prediction: ResolvedPrediction,
    *,
    px: int,
    py: int,
    class_names: Optional[Sequence[str]] = None,
    feature_map: Optional[torch.Tensor] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Optional[dict]:
    flat = prediction.flat_queries
    prototypes = prediction.prototypes
    if flat is None or prototypes is None:
        return None

    if feature_map is None:
        prototype_mask_logits = prediction.raw_mask_logits
        prototype_mask_probabilities = prediction.raw_mask_probabilities
        analysis_shape = (prediction.raw_mask_probabilities.shape[1], prediction.raw_mask_probabilities.shape[2])
    else:
        if image_shape is None:
            image_shape = tuple(int(v) for v in feature_map.shape[-2:])
        prototype_mask_logits = project_mask_embeddings(
            prediction.mask_embeddings.unsqueeze(0),
            feature_map.unsqueeze(0),
            image_shape,
        )[0]
        prototype_mask_probabilities = torch.softmax(prototype_mask_logits, dim=0)
        analysis_shape = image_shape

    if prototype_mask_probabilities.shape[0] == 0:
        return None
    if py < 0 or px < 0 or py >= analysis_shape[0] or px >= analysis_shape[1]:
        return None

    probabilities_tensor = prototype_mask_probabilities[:, py, px]
    probability_sum = float(probabilities_tensor.sum().item())
    if probability_sum <= 0.0:
        return None

    probabilities = probabilities_tensor.detach().cpu().numpy()
    argmax_kept_idx = int(torch.argmax(probabilities_tensor).item())
    global_proto_idx = int(prediction.kept_prototype_indices[argmax_kept_idx].item())

    query_logits = _compute_query_logits(
        prediction,
        feature_map=feature_map,
        image_shape=analysis_shape,
    )
    if query_logits is None:
        return None
    query_pixel_logits = query_logits[:, py, px]
    assignment_weights = prototypes.assignment_weights[:, global_proto_idx]
    query_contributions = assignment_weights * query_pixel_logits

    cmap = plt.get_cmap("tab20")
    labels = []
    colors = []
    for kept_idx, (label, score) in enumerate(zip(prediction.resolved_labels, probabilities.tolist())):
        class_name = str(label)
        if class_names is not None and 0 <= int(label) < len(class_names):
            class_name = class_names[int(label)]
        labels.append(f"{kept_idx}: {class_name}")
        colors.append(cmap(kept_idx % 20))

    argmax_label = int(prediction.resolved_labels[argmax_kept_idx])
    argmax_class_name = str(argmax_label)
    if class_names is not None and 0 <= argmax_label < len(class_names):
        argmax_class_name = class_names[argmax_label]

    return {
        "pixel": (px, py),
        "probabilities": probabilities,
        "labels": labels,
        "colors": colors,
        "xlabel": "Raw mask softmax probability",
        "argmax_kept_idx": argmax_kept_idx,
        "argmax_global_proto_idx": global_proto_idx,
        "argmax_class_name": argmax_class_name,
        "query_contributions": query_contributions.detach().cpu().numpy(),
    }


def _apply_query_contribution_overlay(row_state: _RowInteractionState, contributions: np.ndarray):
    row_state.query_contributions = np.asarray(contributions, dtype=np.float32)
    _refresh_umap_query_artists(row_state)


def _clear_selected_pixel_marker(row_state: _RowInteractionState):
    for artists in row_state.selected_pixel_artists.values():
        for artist in artists:
            try:
                artist.remove()
            except ValueError:
                pass
    row_state.selected_pixel_artists.clear()


def _draw_selected_pixel_marker(row_state: _RowInteractionState):
    _clear_selected_pixel_marker(row_state)
    if row_state.selected_pixel is None:
        return

    axis = row_state.selected_pixel.axis
    px = row_state.selected_pixel.px
    py = row_state.selected_pixel.py

    outer = axis.scatter(
        [px],
        [py],
        s=90,
        facecolors="none",
        edgecolors="white",
        linewidths=1.8,
        zorder=8,
    )
    inner = axis.scatter(
        [px],
        [py],
        s=28,
        c=["black"],
        marker="x",
        linewidths=1.2,
        zorder=9,
    )
    row_state.selected_pixel_artists[axis] = [outer, inner]


def _render_panel(
    row_state: _RowInteractionState,
    panel: Optional[_PredictionPanelState],
    *,
    query_index: Optional[int] = None,
):
    if panel is None:
        return

    panel.axis.clear()
    if panel.mode == "query_heatmap":
        if query_index is None:
            panel.info_title = panel.title
            panel.info_text = None
            _draw_query_logit_heatmap(panel.axis, None, title=panel.title)
        else:
            preview_payload = panel.info_payload or {}
            logit_map, title, info_text = _single_query_preview(
                panel.prediction,
                query_index,
                feature_map=preview_payload.get("feature_map"),
                image_shape=preview_payload.get("image_shape"),
                title=panel.title,
            )
            panel.info_title = title
            panel.info_text = info_text
            _draw_query_logit_heatmap(panel.axis, logit_map, title=title)
        row_state.axis_instances.pop(panel.axis, None)
        return

    if panel.mode == "query_info":
        _draw_query_info_panel(
            panel.axis,
            title=panel.info_title or panel.title,
            info_text=panel.info_text,
        )
        row_state.axis_instances.pop(panel.axis, None)
        return

    if panel.mode == "pixel_distribution":
        _draw_pixel_distribution_panel(
            panel.axis,
            title=panel.info_title or panel.title,
            payload=panel.info_payload,
        )
        row_state.axis_instances.pop(panel.axis, None)
        return

    if panel.mode == "native_instances":
        payload = panel.info_payload or {}
        feature_map = payload.get("feature_map")
        if feature_map is None:
            row_state.axis_instances[panel.axis] = []
            panel.axis.clear()
            panel.axis.set_title(panel.title)
            panel.axis.axis("off")
            return
        masks, labels, scores = _prediction_masks_for_feature_map(
            panel.prediction,
            feature_map,
        )
        instances = _draw_native_resolution_instances(
            panel.axis,
            masks,
            labels,
            scores=scores,
            class_names=panel.class_names,
            title=panel.title,
        )
        row_state.axis_instances[panel.axis] = instances
        return

    masks = [mask.detach().cpu().numpy() for mask in panel.prediction.resolved_masks]
    labels = [int(label) for label in panel.prediction.resolved_labels]
    scores = [float(score) for score in panel.prediction.resolved_scores]
    title = panel.title
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


def _render_prediction_panels(row_state: _RowInteractionState):
    for panel in row_state.prediction_panels:
        _render_panel(row_state, panel, query_index=None)
    _draw_selected_pixel_marker(row_state)


def _select_instance(row_state: _RowInteractionState, axis, instance_index: int):
    selected = row_state.axis_instances.get(axis, [])[instance_index]

    if row_state.selected_query_index is not None:
        _clear_query_panels(row_state)
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
    row_state.selected_pixel = None

    _render_prediction_panels(row_state)
    for panel in row_state.query_heatmap_panels:
        _render_panel(row_state, panel, query_index=None)
    if row_state.query_info_panel is not None:
        row_state.query_info_panel.info_title = row_state.query_info_panel.title
        row_state.query_info_panel.info_text = None
        _render_panel(row_state, row_state.query_info_panel, query_index=None)
    if row_state.pixel_distribution_panel is not None:
        row_state.pixel_distribution_panel.info_title = row_state.pixel_distribution_panel.title
        row_state.pixel_distribution_panel.info_payload = None
        _render_panel(row_state, row_state.pixel_distribution_panel, query_index=None)
    _clear_selected_pixel_marker(row_state)

    for instances in row_state.axis_instances.values():
        for instance in instances:
            _set_instance_highlight(instance, True)

    for artist_group in row_state.umap_gt_artists:
        _set_umap_gt_highlight(artist_group, True)

    _clear_query_contribution_overlay(row_state)


def _clear_instance_highlights(row_state: _RowInteractionState):
    row_state.selected_axis = None
    row_state.selected_index = None
    for instances in row_state.axis_instances.values():
        for instance in instances:
            _set_instance_highlight(instance, True)
    for artist_group in row_state.umap_gt_artists:
        _set_umap_gt_highlight(artist_group, True)


def _clear_query_panels(row_state: _RowInteractionState):
    row_state.selected_query_index = None
    for panel in row_state.query_heatmap_panels:
        _render_panel(row_state, panel, query_index=None)
    if row_state.query_info_panel is not None:
        row_state.query_info_panel.info_title = row_state.query_info_panel.title
        row_state.query_info_panel.info_text = None
        _render_panel(row_state, row_state.query_info_panel, query_index=None)
    _refresh_umap_query_artists(row_state)


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
    identity_similarity_metric: str = "cosine",
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
    gt_target_indices: list[int]
    if prediction.prototypes is not None and prediction.prototypes.target_indices is not None:
        target_indices_all = [
            int(idx)
            for idx in prediction.prototypes.target_indices.detach().cpu().tolist()
            if 0 <= int(idx) < len(gt_labels_all)
        ]
        gt_target_indices = [idx for idx in target_indices_all if int(gt_labels_all[idx]) != 0]
        keep_signature_indices = [
            sig_idx
            for sig_idx, target_idx in enumerate(target_indices_all)
            if int(gt_labels_all[target_idx]) != 0
        ]
        gt_sig_np = gt_sig_np[keep_signature_indices]
    else:
        gt_target_indices = _foreground_instance_indices(gt_labels_all)
        if gt_sig_np.shape[0] == len(gt_labels_all):
            gt_sig_np = gt_sig_np[gt_target_indices]
        elif gt_sig_np.shape[0] != len(gt_target_indices):
            gt_target_indices = gt_target_indices[: gt_sig_np.shape[0]]

    gt_masks = [gt_masks_all[idx] for idx in gt_target_indices]
    gt_labels = [gt_labels_all[idx] for idx in gt_target_indices]

    all_sig = np.concatenate([q_sig_np, gt_sig_np], axis=0) if gt_sig_np.shape[0] > 0 else q_sig_np
    embedding = _project_signatures_2d(all_sig, metric=identity_similarity_metric)
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
        pairwise_distances = pairwise_distance(
            q_sig_t,
            q_sig_t,
            metric=identity_similarity_metric,
            clamp=True,
        ).detach().cpu().numpy()
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
                _UmapArtistGroup(
                    scatter=scatter,
                    text=None,
                    base_size=float(query_sizes[idx]),
                    base_color=np.array(query_colors[idx], dtype=np.float32),
                    base_alpha=float(query_alpha[idx]),
                )
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
    ax.imshow(image_np, interpolation="nearest")
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
        overlay_artist = ax.imshow(overlay, interpolation="nearest")

        bbox = _mask_bbox(mask_bool)
        if bbox is None:
            continue

        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle(
            (xmin - 0.5, ymin - 0.5),
            max(xmax - xmin + 1, 1),
            max(ymax - ymin + 1, 1),
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


def _draw_native_resolution_instances(
    ax,
    masks: Sequence[np.ndarray],
    labels: Sequence[int],
    *,
    scores: Optional[Sequence[float]] = None,
    class_names: Optional[Sequence[str]] = None,
    title: str,
):
    if len(masks) == 0:
        ax.clear()
        ax.set_title(title)
        ax.axis("off")
        ax.text(0.5, 0.5, "No instances", ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return []

    height, width = np.asarray(masks[0]).shape
    background = np.zeros((height, width, 3), dtype=np.float32)
    instances = _draw_instances(
        ax,
        background,
        masks,
        labels,
        scores=scores,
        class_names=class_names,
        title=title,
        gt_masks=[],
        kind="prediction",
    )
    return instances


def _prediction_columns(
    predictions: Sequence[ResolvedPrediction],
    *,
    gt_proto_predictions: Optional[Sequence[ResolvedPrediction]] = None,
    golden_predictions: Optional[Sequence[ResolvedPrediction]] = None,
):
    prediction_columns = [("Clustered Prediction", predictions)]
    if gt_proto_predictions is not None:
        prediction_columns.append(("GT Prototype Prediction", gt_proto_predictions))
    if golden_predictions is not None:
        prediction_columns.append(("Golden Prediction", golden_predictions))
    return prediction_columns


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


@torch.no_grad()
def run_evaluation_view_predictions(
    system: PanopticSystem,
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    *,
    device: Optional[torch.device] = None,
) -> List[EvaluationPredictionSet]:
    if len(images) == 0:
        return []

    model_device = device
    if model_device is None:
        model_device = next(system.parameters()).device

    batch = torch.stack(list(images)).to(model_device)
    was_training = system.training
    system.eval()
    try:
        raw = system.model(batch)
        predictions = system.predictor.predict_evaluation_views_from_raw(system.model, raw, targets)
    finally:
        system.train(was_training)

    if isinstance(predictions, list):
        return predictions
    return [predictions]


@torch.no_grad()
def collect_prediction_bundles(
    system: PanopticSystem,
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    *,
    device: Optional[torch.device] = None,
) -> List[PredictionBundle]:
    if system.supports_gt_prototypes:
        evaluation_predictions = run_evaluation_view_predictions(
            system,
            images,
            targets,
            device=device,
        )
        return [
            PredictionBundle(
                clustered=prediction_set.clustering,
                gt_signatures=prediction_set.gt_signatures,
                golden_queries=prediction_set.golden_queries,
            )
            for prediction_set in evaluation_predictions
        ]

    predictions = run_predictions(system, images, device=device)
    return [PredictionBundle(clustered=prediction) for prediction in predictions]


@torch.no_grad()
def collect_detailed_prediction_bundle(
    system: PanopticSystem,
    image: torch.Tensor,
    target: dict,
    *,
    device: Optional[torch.device] = None,
) -> DetailedPredictionBundle:
    model_device = device
    if model_device is None:
        model_device = next(system.parameters()).device

    batch = image.unsqueeze(0).to(model_device)
    was_training = system.training
    system.eval()
    try:
        raw = system.model(batch)
        if system.supports_gt_prototypes:
            prediction_set = system.predictor.predict_evaluation_views_from_raw(system.model, raw, [target])
            predictions = PredictionBundle(
                clustered=prediction_set.clustering,
                gt_signatures=prediction_set.gt_signatures,
                golden_queries=prediction_set.golden_queries,
            )
        else:
            prediction = system.predictor.predict_from_raw(system.model, raw)
            predictions = PredictionBundle(
                clustered=prediction,
            )
    finally:
        system.train(was_training)

    raw_feature_maps = {
        level_name: level_features[0].detach()
        for level_name, level_features in raw.feature_maps.items()
    }
    identity_similarity_metric = getattr(system.model, "identity_similarity_metric", "cosine")
    return DetailedPredictionBundle(
        image=image.detach().cpu(),
        target=target,
        raw_feature_maps=raw_feature_maps,
        predictions=predictions,
        identity_similarity_metric=identity_similarity_metric,
    )


def render_prediction_grid(
    images: Sequence[torch.Tensor],
    targets: Sequence[dict],
    prediction_columns: Sequence[Tuple[str, Sequence[dict]]],
    *,
    identity_similarity_metric: str = "cosine",
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
    interactive: bool = False,
):
    num_samples = len(images)
    gt_signature_prediction_column_idx = next(
        (idx for idx, (title, _) in enumerate(prediction_columns) if "GT Prototype" in title),
        None,
    )
    add_signature_column = gt_signature_prediction_column_idx is not None
    num_cols = 2 + len(prediction_columns) + int(add_signature_column)
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(5 * num_cols, 5 * max(num_samples, 1)), squeeze=False)

    if figure_title:
        fig.suptitle(figure_title, fontsize=14)

    for row_idx, (image, target) in enumerate(zip(images, targets)):
        image_np = _to_numpy_image(image)
        axes[row_idx, 0].imshow(image_np)
        axes[row_idx, 0].set_title("Input")
        axes[row_idx, 0].axis("off")

        gt_masks = [mask.detach().cpu().numpy() for mask in target["masks"]]
        gt_labels = [int(label) for label in target["labels"].detach().cpu().tolist()]
        gt_masks, gt_labels, _ = _filter_background_instances(gt_masks, gt_labels)
        _draw_instances(
            axes[row_idx, 1],
            image_np,
            gt_masks,
            gt_labels,
            class_names=class_names,
            title="Ground Truth",
            gt_masks=gt_masks,
            kind="gt",
        )

        next_col_idx = 2
        gt_signature_prediction = None
        for column_idx, (column_title, predictions) in enumerate(prediction_columns):
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
            _draw_instances(
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
            if column_idx == gt_signature_prediction_column_idx:
                gt_signature_prediction = prediction
            next_col_idx += 1

        if add_signature_column and gt_signature_prediction is not None:
            _draw_signature_umap(
                axes[row_idx, next_col_idx],
                image_np,
                target,
                gt_signature_prediction,
                identity_similarity_metric=identity_similarity_metric,
                class_names=class_names,
                title="GT Signature UMAP",
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
    golden_predictions: Optional[Sequence[dict]] = None,
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_columns = [("Clustered Prediction", predictions)]
    if gt_proto_predictions is not None:
        prediction_columns.append(("GT Prototype Prediction", gt_proto_predictions))
    if golden_predictions is not None:
        prediction_columns.append(("Golden Prediction", golden_predictions))
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
    golden_predictions: Optional[Sequence[dict]] = None,
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
    window_title: Optional[str] = None,
):
    prediction_columns = [("Clustered Prediction", predictions)]
    if gt_proto_predictions is not None:
        prediction_columns.append(("GT Prototype Prediction", gt_proto_predictions))
    if golden_predictions is not None:
        prediction_columns.append(("Golden Prediction", golden_predictions))
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


def show_detailed_prediction_view(
    bundle: DetailedPredictionBundle,
    *,
    class_names: Optional[Sequence[str]] = None,
    figure_title: Optional[str] = None,
    window_title: Optional[str] = None,
):
    image_np = _to_numpy_image(bundle.image)
    target = bundle.target
    gt_masks = [mask.detach().cpu().numpy() for mask in target["masks"]]
    gt_labels = [int(label) for label in target["labels"].detach().cpu().tolist()]
    gt_masks, gt_labels, _ = _filter_background_instances(gt_masks, gt_labels)

    level_names = tuple(level_name for level_name in ("p2", "p3", "p4", "p5") if level_name in bundle.raw_feature_maps)
    num_cols = max(len(level_names) + 2, 3)
    fig, axes = plt.subplots(2, num_cols, figsize=(5 * num_cols, 10), squeeze=False)

    if figure_title:
        fig.suptitle(figure_title, fontsize=14)

    row_state = _RowInteractionState(gt_masks=gt_masks)
    primary_prediction = bundle.predictions.clustered
    row_state.prediction = primary_prediction
    row_state.class_names = class_names

    prediction_axes = axes[0]
    heatmap_axes = axes[1]

    gt_instances = _draw_instances(
        prediction_axes[0],
        image_np,
        gt_masks,
        gt_labels,
        class_names=class_names,
        title="Ground Truth",
        gt_masks=gt_masks,
        kind="gt",
    )
    row_state.axis_instances[prediction_axes[0]] = gt_instances

    pixel_distribution_axis = prediction_axes[-1]
    info_axis = heatmap_axes[-1]
    row_state.query_info_panel = _PredictionPanelState(
        axis=info_axis,
        image_np=image_np,
        prediction=primary_prediction,
        gt_masks=gt_masks,
        class_names=class_names,
        title="Selected query info",
        mode="query_info",
    )
    row_state.pixel_distribution_panel = _PredictionPanelState(
        axis=pixel_distribution_axis,
        image_np=image_np,
        prediction=primary_prediction,
        gt_masks=gt_masks,
        class_names=class_names,
        title="Pixel object distribution",
        mode="pixel_distribution",
    )

    umap_prediction = bundle.predictions.gt_signatures or primary_prediction
    umap_title = (
        "Query / GT Signatures"
        if bundle.predictions.gt_signatures is not None
        else "Query / Prototype Signatures"
    )

    _draw_signature_umap(
        heatmap_axes[0],
        image_np,
        target,
        umap_prediction,
        identity_similarity_metric=bundle.identity_similarity_metric,
        class_names=class_names,
        title=umap_title,
        row_state=row_state,
    )

    for col_idx, level_name in enumerate(level_names, start=1):
        feature_map = bundle.raw_feature_maps[level_name]
        masks, labels, scores = _prediction_masks_for_feature_map(
            primary_prediction,
            feature_map,
        )
        prediction_axis = prediction_axes[col_idx]
        instances = _draw_native_resolution_instances(
            prediction_axis,
            masks,
            labels,
            scores=scores,
            class_names=class_names,
            title=f"{level_name} prediction",
        )
        row_state.axis_instances[prediction_axis] = instances
        row_state.pixel_click_sources[prediction_axis] = {
            "feature_map": feature_map,
            "title": f"{level_name} prediction",
        }
        row_state.prediction_panels.append(
            _PredictionPanelState(
                axis=prediction_axis,
                image_np=image_np,
                prediction=primary_prediction,
                gt_masks=gt_masks,
                class_names=class_names,
                title=f"{level_name} prediction",
                mode="native_instances",
                info_payload={"feature_map": feature_map},
            )
        )

        heatmap_panel = _PredictionPanelState(
            axis=heatmap_axes[col_idx],
            image_np=image_np,
            prediction=primary_prediction,
            gt_masks=gt_masks,
            class_names=class_names,
            title=f"{level_name} heatmap",
            mode="query_heatmap",
            info_payload={
                "feature_map": feature_map,
                "image_shape": tuple(int(v) for v in feature_map.shape[-2:]),
            },
        )
        row_state.query_heatmap_panels.append(heatmap_panel)
        _render_panel(row_state, heatmap_panel, query_index=None)

    _render_panel(row_state, row_state.query_info_panel, query_index=None)
    _render_panel(row_state, row_state.pixel_distribution_panel, query_index=None)

    for unused_axis in prediction_axes[len(level_names) + 1 : -1]:
        unused_axis.axis("off")
    for unused_axis in heatmap_axes[len(level_names) + 1 : -1]:
        unused_axis.axis("off")

    _clear_selection(row_state)

    axis_to_row_state = {
        axis: row_state
        for axis in row_state.axis_instances
    }
    for axis in row_state.pixel_click_sources:
        axis_to_row_state[axis] = row_state
    if row_state.umap_query_axis is not None:
        axis_to_row_state[row_state.umap_query_axis] = row_state

    def _on_click(event):
        local_row_state = axis_to_row_state.get(event.inaxes)
        if local_row_state is None:
            return

        if event.inaxes is local_row_state.umap_query_axis:
            query_idx = _hit_test_umap_queries(local_row_state, event.xdata, event.ydata)
            if query_idx is None:
                _clear_selection(local_row_state)
            else:
                local_row_state.selected_axis = None
                local_row_state.selected_index = None
                local_row_state.selected_query_index = query_idx
                for panel in local_row_state.query_heatmap_panels:
                    _render_panel(local_row_state, panel, query_index=query_idx)
                if local_row_state.query_info_panel is not None and local_row_state.prediction is not None:
                    _, _, info_text = _single_query_preview(local_row_state.prediction, query_idx)
                    local_row_state.query_info_panel.info_title = local_row_state.query_info_panel.title
                    local_row_state.query_info_panel.info_text = info_text
                    _render_panel(local_row_state, local_row_state.query_info_panel, query_index=query_idx)
                _refresh_umap_query_artists(local_row_state)

                preview = local_row_state.prediction
                if preview is not None:
                    preview_logits, _, _ = _single_query_preview(preview, query_idx)
                    matched_gt_idx = None
                    if preview_logits is not None:
                        matched_gt_idx = _select_single_gt_for_detection(preview_logits > 0.0, local_row_state.gt_masks)
                    matched_gt_indices = {matched_gt_idx} if matched_gt_idx is not None else set()
                    for idx, artist_group in enumerate(local_row_state.umap_gt_artists):
                        _set_umap_gt_highlight(artist_group, idx in matched_gt_indices)
            fig.canvas.draw_idle()
            return

        pixel_source = local_row_state.pixel_click_sources.get(event.inaxes)
        if pixel_source is not None:
            px = None if event.xdata is None else int(round(event.xdata))
            py = None if event.ydata is None else int(round(event.ydata))
            if px is None or py is None:
                _clear_selection(local_row_state)
                fig.canvas.draw_idle()
                return

            pixel_payload = _compute_pixel_analysis(
                local_row_state.prediction,
                px=px,
                py=py,
                class_names=local_row_state.class_names,
                feature_map=pixel_source["feature_map"],
            )
            if pixel_payload is not None:
                _clear_instance_highlights(local_row_state)
                local_row_state.selected_pixel = _SelectedPixelState(
                    axis=event.inaxes,
                    px=px,
                    py=py,
                )
                _draw_selected_pixel_marker(local_row_state)
                local_row_state.pixel_distribution_panel.info_title = (
                    f"{pixel_source['title']} pixel distribution @ ({px}, {py})"
                )
                local_row_state.pixel_distribution_panel.info_payload = pixel_payload
                _render_panel(local_row_state, local_row_state.pixel_distribution_panel, query_index=None)
                _apply_query_contribution_overlay(local_row_state, pixel_payload["query_contributions"])
            else:
                _clear_selection(local_row_state)
            fig.canvas.draw_idle()
            return

        instance_idx = _hit_test_masks(
            [instance.mask for instance in local_row_state.axis_instances[event.inaxes]],
            event.xdata,
            event.ydata,
        )
        if instance_idx is None:
            _clear_selection(local_row_state)
        else:
            _select_instance(local_row_state, event.inaxes, instance_idx)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)

    if window_title and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(window_title)
    if figure_title:
        plt.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        plt.tight_layout()
    fig.subplots_adjust(hspace=0.38)
    plt.show()
    plt.close(fig)
