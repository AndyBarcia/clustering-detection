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
        reference = prediction if signature_reference is None else signature_reference
        _draw_signature_umap(
            axes[2],
            image_np,
            target,
            reference,
            class_names=class_names,
            title="GT Signature UMAP",
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
