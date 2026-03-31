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

from .dataset import SyntheticPanopticBatchGenerator
from .panoptic import PanopticSystem


DEFAULT_CLASS_NAMES = ["Background", "Square", "Triangle"]


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
    num_cols = 2 + len(prediction_columns)
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
        _draw_instances(
            axes[row_idx, 1],
            image_np,
            gt_masks,
            gt_labels,
            class_names=class_names,
            title="Ground Truth",
        )

        for col_idx, (column_title, predictions) in enumerate(prediction_columns, start=2):
            prediction = predictions[row_idx]
            pred_masks = [mask.detach().cpu().numpy() for mask in prediction["resolved_masks"]]
            pred_labels = [int(label) for label in prediction["resolved_labels"]]
            pred_scores = [float(score) for score in prediction["resolved_scores"]]
            _draw_instances(
                axes[row_idx, col_idx],
                image_np,
                pred_masks,
                pred_labels,
                scores=pred_scores,
                class_names=class_names,
                title=column_title,
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
