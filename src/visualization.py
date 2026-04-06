from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

try:
    import umap
except ImportError:
    umap = None

from .dataset import SyntheticPanopticBatchGenerator
from .panoptic import PanopticSystem


DEFAULT_CLASS_NAMES = ["Background", "Square", "Triangle"]


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

    if umap is not None:
        n_neighbors = max(2, min(15, signatures.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.15,
            metric="cosine",
            random_state=0,
        )
        return reducer.fit_transform(signatures).astype(np.float32, copy=False)

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
    q_seed = None if flat is None else flat.get("q_seed")
    q_influence = None if flat is None else flat.get("q_influence")
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
    max_query_marker_size = 300.0

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
):
    num_samples = len(images)
    add_signature_column = any("GT Prototype" in title for title, _ in prediction_columns)
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
        )

        next_col_idx = 2
        for column_title, predictions in prediction_columns:
            prediction = predictions[row_idx]
            pred_masks = [mask.detach().cpu().numpy() for mask in prediction["resolved_masks"]]
            pred_labels = [int(label) for label in prediction["resolved_labels"]]
            pred_scores = [float(score) for score in prediction["resolved_scores"]]
            pred_masks, pred_labels, pred_scores = _filter_background_instances(
                pred_masks,
                pred_labels,
                pred_scores,
            )
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
    )
    if window_title and fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(window_title)
    plt.show()
    plt.close(fig)
