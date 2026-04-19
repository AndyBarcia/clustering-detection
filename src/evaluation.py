from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .dataset import SyntheticPanopticBatchGenerator, BatchedSyntheticIterableDataset
from .outputs import EvaluationPredictionSet, ResolvedPrediction
from .predictor import ModularPrototypePredictor
from .signature_ops import pairwise_distance


@dataclass
class ImageEvaluation:
    num_gt: int
    num_pred: int
    matched_iou_sum: float
    matched_box_iou_sum: float
    num_tp: int
    prediction_records: List[Tuple[float, int]]
    matched_query_distances: List[float] = field(default_factory=list)
    unmatched_query_closest_gt_distances: List[float] = field(default_factory=list)
    signature_count_pred: int | None = None
    signature_count_gt: int | None = None
    signature_chamfer_distance: float | None = None
    signature_hausdorff_distance: float | None = None


class _ProbeMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _stack_masks(masks: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(masks) == 0:
        return torch.zeros((0, 1, 1), dtype=torch.bool)
    return torch.stack([mask.detach().to(dtype=torch.bool, device="cpu") for mask in masks], dim=0)


def _to_prediction_tensors(prediction: ResolvedPrediction) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_masks = _stack_masks(prediction.resolved_masks)
    pred_labels = torch.as_tensor(prediction.resolved_labels, dtype=torch.long)
    pred_scores = torch.as_tensor(prediction.resolved_scores, dtype=torch.float32)
    keep = pred_labels != 0
    return pred_masks[keep], pred_labels[keep], pred_scores[keep]


def _to_target_tensors(target: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    gt_masks = target["masks"].detach().to(dtype=torch.bool, device="cpu")
    gt_labels = target["labels"].detach().to(dtype=torch.long, device="cpu")
    keep = gt_labels != 0
    return gt_masks[keep], gt_labels[keep]


def _masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    if masks.shape[0] == 0:
        return torch.zeros((0, 4), dtype=torch.float32)

    height, width = masks.shape[-2:]
    rows = masks.any(dim=2)
    cols = masks.any(dim=1)

    y_min = rows.float().argmax(dim=1)
    y_max = height - 1 - rows.flip(1).float().argmax(dim=1)
    x_min = cols.float().argmax(dim=1)
    x_max = width - 1 - cols.flip(1).float().argmax(dim=1)
    return torch.stack([x_min, y_min, x_max, y_max], dim=1).to(torch.float32)


def _pairwise_mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
        return torch.zeros((pred_masks.shape[0], gt_masks.shape[0]), dtype=torch.float32)

    pred_flat = pred_masks.reshape(pred_masks.shape[0], -1).float()
    gt_flat = gt_masks.reshape(gt_masks.shape[0], -1).float()

    intersection = pred_flat @ gt_flat.T
    pred_area = pred_flat.sum(dim=1, keepdim=True)
    gt_area = gt_flat.sum(dim=1).unsqueeze(0)
    union = pred_area + gt_area - intersection
    return intersection / union.clamp_min(1e-6)


def _pairwise_box_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    if pred_masks.shape[0] == 0 or gt_masks.shape[0] == 0:
        return torch.zeros((pred_masks.shape[0], gt_masks.shape[0]), dtype=torch.float32)

    pred_boxes = _masks_to_boxes(pred_masks)
    gt_boxes = _masks_to_boxes(gt_masks)

    top_left = torch.maximum(pred_boxes[:, None, :2], gt_boxes[None, :, :2])
    bottom_right = torch.minimum(pred_boxes[:, None, 2:], gt_boxes[None, :, 2:])
    wh = (bottom_right - top_left + 1).clamp_min(0)
    intersection = wh[..., 0] * wh[..., 1]

    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1)
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return intersection / union.clamp_min(1e-6)


def _compute_signature_set_distance_metrics(
    pred_sig: torch.Tensor,
    gt_sig: torch.Tensor,
    *,
    similarity_metric: str = "dot",
    empty_distance: float = 1.0,
) -> Tuple[float, float]:
    if pred_sig.shape[0] == 0 and gt_sig.shape[0] == 0:
        return 0.0, 0.0
    if pred_sig.shape[0] == 0 or gt_sig.shape[0] == 0:
        return empty_distance, empty_distance

    distances = pairwise_distance(
        pred_sig,
        gt_sig,
        metric=similarity_metric,
        clamp=True,
    )
    pred_to_gt = distances.min(dim=1).values
    gt_to_pred = distances.min(dim=0).values

    chamfer = 0.5 * (pred_to_gt.mean() + gt_to_pred.mean())
    hausdorff = torch.maximum(pred_to_gt.max(), gt_to_pred.max())
    return float(chamfer.item()), float(hausdorff.item())


def _foreground_resolved_signature_embeddings(prediction: ResolvedPrediction) -> torch.Tensor:
    signature_embeddings = prediction.signature_embeddings
    if signature_embeddings.shape[0] == 0:
        return signature_embeddings

    if len(prediction.resolved_labels) != signature_embeddings.shape[0]:
        raise ValueError(
            "Resolved prediction signatures and labels must stay aligned after filtering."
        )

    keep = torch.as_tensor(prediction.resolved_labels, dtype=torch.long, device=signature_embeddings.device) != 0
    return signature_embeddings[keep]


def hungarian_match_instances(
    pred_masks: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_labels: torch.Tensor,
    *,
    unmatched_cost: float = 1.0,
    invalid_cost: float = 2.0,
) -> Tuple[List[Tuple[int, int, float]], torch.Tensor]:
    iou = _pairwise_mask_iou(pred_masks, gt_masks)
    num_pred, num_gt = iou.shape

    if num_pred == 0 or num_gt == 0:
        return [], iou

    size = max(num_pred, num_gt)
    cost = np.full((size, size), unmatched_cost, dtype=np.float64)

    same_class = pred_labels[:, None].eq(gt_labels[None, :])
    valid_match = same_class & (iou > 0)

    valid_cost = 1.0 - iou.numpy()
    cost[:num_pred, :num_gt] = np.where(valid_match.numpy(), valid_cost, invalid_cost)

    row_ind, col_ind = linear_sum_assignment(cost)

    matches: List[Tuple[int, int, float]] = []
    for pred_idx, gt_idx in zip(row_ind.tolist(), col_ind.tolist()):
        if pred_idx >= num_pred or gt_idx >= num_gt:
            continue
        if not valid_match[pred_idx, gt_idx]:
            continue
        matches.append((pred_idx, gt_idx, float(iou[pred_idx, gt_idx].item())))

    return matches, iou


def evaluate_image(
    prediction: ResolvedPrediction,
    target: Dict,
    *,
    ap_iou_threshold: float = 0.5,
) -> ImageEvaluation:
    if prediction.resolved_target_indices is not None:
        return evaluate_aligned_image(
            prediction,
            target,
            ap_iou_threshold=ap_iou_threshold,
        )

    pred_masks, pred_labels, pred_scores = _to_prediction_tensors(prediction)
    gt_masks, gt_labels = _to_target_tensors(target)
    matches, _ = hungarian_match_instances(pred_masks, pred_labels, gt_masks, gt_labels)
    box_iou = _pairwise_box_iou(pred_masks, gt_masks)

    pred_is_tp = torch.zeros(pred_masks.shape[0], dtype=torch.int64)
    matched_iou_sum = 0.0
    matched_box_iou_sum = 0.0

    for pred_idx, gt_idx, match_iou in matches:
        matched_iou_sum += match_iou
        matched_box_iou_sum += float(box_iou[pred_idx, gt_idx].item())
        if match_iou >= ap_iou_threshold:
            pred_is_tp[pred_idx] = 1

    prediction_records = [
        (float(score), int(is_tp))
        for score, is_tp in zip(pred_scores.tolist(), pred_is_tp.tolist())
    ]

    return ImageEvaluation(
        num_gt=int(gt_masks.shape[0]),
        num_pred=int(pred_masks.shape[0]),
        matched_iou_sum=matched_iou_sum,
        matched_box_iou_sum=matched_box_iou_sum,
        num_tp=int(pred_is_tp.sum().item()),
        prediction_records=prediction_records,
    )


def evaluate_aligned_image(
    prediction: ResolvedPrediction,
    target: Dict,
    *,
    ap_iou_threshold: float = 0.5,
) -> ImageEvaluation:
    pred_masks = _stack_masks(prediction.resolved_masks)
    pred_labels = torch.as_tensor(prediction.resolved_labels, dtype=torch.long)
    pred_scores = torch.as_tensor(prediction.resolved_scores, dtype=torch.float32)
    target_indices = prediction.resolved_target_indices.detach().to(dtype=torch.long, device="cpu")

    target_masks_all = target["masks"].detach().to(dtype=torch.bool, device="cpu")
    target_labels_all = target["labels"].detach().to(dtype=torch.long, device="cpu")
    num_gt = int((target_labels_all != 0).sum().item())

    if target_indices.numel() == 0 or pred_masks.shape[0] == 0:
        return ImageEvaluation(
            num_gt=num_gt,
            num_pred=0,
            matched_iou_sum=0.0,
            matched_box_iou_sum=0.0,
            num_tp=0,
            prediction_records=[],
        )

    aligned_gt_masks = target_masks_all[target_indices]
    aligned_gt_labels = target_labels_all[target_indices]

    keep = (pred_labels != 0) & (aligned_gt_labels != 0)
    pred_masks = pred_masks[keep]
    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    aligned_gt_masks = aligned_gt_masks[keep]
    aligned_gt_labels = aligned_gt_labels[keep]

    if pred_masks.shape[0] == 0:
        return ImageEvaluation(
            num_gt=num_gt,
            num_pred=0,
            matched_iou_sum=0.0,
            matched_box_iou_sum=0.0,
            num_tp=0,
            prediction_records=[],
        )

    iou_matrix = _pairwise_mask_iou(pred_masks, aligned_gt_masks)
    box_iou_matrix = _pairwise_box_iou(pred_masks, aligned_gt_masks)
    diag_indices = torch.arange(pred_masks.shape[0], dtype=torch.long)
    aligned_iou = iou_matrix[diag_indices, diag_indices]
    aligned_box_iou = box_iou_matrix[diag_indices, diag_indices]
    class_match = pred_labels.eq(aligned_gt_labels)
    pred_is_tp = class_match & (aligned_iou >= ap_iou_threshold)

    prediction_records = [
        (float(score), int(is_tp))
        for score, is_tp in zip(pred_scores.tolist(), pred_is_tp.to(dtype=torch.int64).tolist())
    ]

    return ImageEvaluation(
        num_gt=num_gt,
        num_pred=int(pred_masks.shape[0]),
        matched_iou_sum=float((aligned_iou * class_match.float()).sum().item()),
        matched_box_iou_sum=float((aligned_box_iou * class_match.float()).sum().item()),
        num_tp=int(pred_is_tp.sum().item()),
        prediction_records=prediction_records,
    )


def _ensure_prediction_list(predictions):
    return predictions if isinstance(predictions, list) else [predictions]


def _compute_average_precision(prediction_records: Iterable[Tuple[float, int]], num_gt: int) -> float:
    if num_gt == 0:
        return 0.0

    ranked = sorted(prediction_records, key=lambda item: item[0], reverse=True)
    if not ranked:
        return 0.0

    tp = np.asarray([item[1] for item in ranked], dtype=np.float64)
    fp = 1.0 - tp

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / max(num_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))

    for idx in range(precision.shape[0] - 1, 0, -1):
        precision[idx - 1] = max(precision[idx - 1], precision[idx])

    change_points = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[change_points + 1] - recall[change_points]) * precision[change_points + 1]))


def _append_value_summary(summary: Dict[str, float], prefix: str, values: Sequence[float]):
    if not values:
        return
    summary[f"{prefix}_count"] = int(len(values))
    summary[f"{prefix}_mean"] = float(sum(values) / len(values))
    if len(values) > 1:
        summary[f"{prefix}_std"] = float(np.std(np.asarray(values, dtype=np.float64), ddof=0))
    else:
        summary[f"{prefix}_std"] = 0.0
    summary[f"{prefix}_min"] = float(min(values))
    summary[f"{prefix}_max"] = float(max(values))


def summarize_evaluations(image_evaluations: Sequence[ImageEvaluation]) -> Dict[str, float]:
    num_images = len(image_evaluations)
    total_gt = sum(item.num_gt for item in image_evaluations)
    total_pred = sum(item.num_pred for item in image_evaluations)
    total_matched_iou = sum(item.matched_iou_sum for item in image_evaluations)
    total_matched_box_iou = sum(item.matched_box_iou_sum for item in image_evaluations)

    prediction_records: List[Tuple[float, int]] = []
    matched_query_distances: List[float] = []
    unmatched_query_distances: List[float] = []
    count_pred: List[int] = []
    count_gt: List[int] = []
    chamfer_values: List[float] = []
    hausdorff_values: List[float] = []
    per_image_mean_iou: List[float] = []
    per_image_mean_iou_box: List[float] = []
    per_image_ap: List[float] = []

    for item in image_evaluations:
        prediction_records.extend(item.prediction_records)
        matched_query_distances.extend(item.matched_query_distances)
        unmatched_query_distances.extend(item.unmatched_query_closest_gt_distances)
        count_pred.append(int(item.num_pred))
        count_gt.append(int(item.num_gt))
        per_image_mean_iou.append(item.matched_iou_sum / item.num_gt if item.num_gt > 0 else 0.0)
        per_image_mean_iou_box.append(item.matched_box_iou_sum / item.num_gt if item.num_gt > 0 else 0.0)
        per_image_ap.append(_compute_average_precision(item.prediction_records, item.num_gt))
        if item.signature_chamfer_distance is not None:
            chamfer_values.append(float(item.signature_chamfer_distance))
        if item.signature_hausdorff_distance is not None:
            hausdorff_values.append(float(item.signature_hausdorff_distance))

    mean_iou = total_matched_iou / total_gt if total_gt > 0 else 0.0
    mean_iou_box = total_matched_box_iou / total_gt if total_gt > 0 else 0.0
    ap = _compute_average_precision(prediction_records, total_gt)

    summary: Dict[str, float] = {
        "num_images": num_images,
        "num_gt_instances": total_gt,
        "num_predictions": total_pred,
        "mean_iou": mean_iou,
        "mean_iou_box": mean_iou_box,
        "ap": ap,
    }
    _append_value_summary(summary, "mean_iou", per_image_mean_iou)
    _append_value_summary(summary, "mean_iou_box", per_image_mean_iou_box)
    _append_value_summary(summary, "ap", per_image_ap)

    _append_value_summary(summary, "matched_query_cosine_distance", matched_query_distances)
    _append_value_summary(summary, "unmatched_query_closest_gt_cosine_distance", unmatched_query_distances)

    count_errors = [pred - gt for pred, gt in zip(count_pred, count_gt)]
    abs_count_errors = [abs(error) for error in count_errors]
    summary.update({
        "mean_count_error": float(sum(count_errors) / len(count_errors)),
        "mean_abs_count_error": float(sum(abs_count_errors) / len(abs_count_errors)),
        "exact_count_accuracy": float(sum(error == 0 for error in count_errors) / len(count_errors)),
        "overpredict_rate": float(sum(error > 0 for error in count_errors) / len(count_errors)),
        "underpredict_rate": float(sum(error < 0 for error in count_errors) / len(count_errors)),
    })
    _append_value_summary(summary, "mean_count_error", count_errors)
    _append_value_summary(summary, "mean_abs_count_error", abs_count_errors)
    _append_value_summary(
        summary,
        "exact_count_accuracy",
        [1.0 if error == 0 else 0.0 for error in count_errors],
    )
    _append_value_summary(
        summary,
        "overpredict_rate",
        [1.0 if error > 0 else 0.0 for error in count_errors],
    )
    _append_value_summary(
        summary,
        "underpredict_rate",
        [1.0 if error < 0 else 0.0 for error in count_errors],
    )

    if chamfer_values:
        summary["signature_chamfer_distance_mean"] = float(sum(chamfer_values) / len(chamfer_values))
        summary["signature_chamfer_distance_std"] = float(np.std(np.asarray(chamfer_values, dtype=np.float64), ddof=0))
    if hausdorff_values:
        summary["signature_hausdorff_distance_mean"] = float(sum(hausdorff_values) / len(hausdorff_values))
        summary["signature_hausdorff_distance_std"] = float(np.std(np.asarray(hausdorff_values, dtype=np.float64), ddof=0))

    return summary


def _build_summary_payload(
    clustering_summary: Dict[str, float],
    gt_signature_summary: Dict[str, float] | None,
    golden_query_summary: Dict[str, float] | None,
    signature_probe_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "clustering": clustering_summary,
        "gt_signatures": gt_signature_summary,
        "golden_queries": golden_query_summary,
        "signature_probes": signature_probe_summary,
        "available_evaluations": [
            name
            for name, metrics in (
                ("signature_probes", signature_probe_summary),
                ("gt_signatures", gt_signature_summary),
                ("golden_queries", golden_query_summary),
                ("clustering", clustering_summary),
            )
            if metrics is not None
        ],
    }


def _evaluate_prediction_set(
    prediction_set: EvaluationPredictionSet,
    target: Dict[str, Any],
    *,
    ap_iou_threshold: float,
    signature_metric: str = "dot",
) -> tuple[ImageEvaluation, ImageEvaluation | None, ImageEvaluation | None]:
    gt_signature_evaluation = None
    golden_query_evaluation = None

    gt_prediction = prediction_set.gt_signatures
    gt_signatures = None if gt_prediction is None else gt_prediction.all_signature_embeddings

    clustering_evaluation = evaluate_image(
        prediction_set.clustering,
        target,
        ap_iou_threshold=ap_iou_threshold,
    )
    if gt_signatures is not None:
        pred_signatures = _foreground_resolved_signature_embeddings(prediction_set.clustering)
        chamfer, hausdorff = _compute_signature_set_distance_metrics(
            pred_signatures,
            gt_signatures,
            similarity_metric=signature_metric,
        )
        clustering_evaluation.signature_count_pred = int(clustering_evaluation.num_pred)
        clustering_evaluation.signature_count_gt = int(clustering_evaluation.num_gt)
        clustering_evaluation.signature_chamfer_distance = chamfer
        clustering_evaluation.signature_hausdorff_distance = hausdorff

    if gt_prediction is not None:
        gt_signature_evaluation = evaluate_image(
            gt_prediction,
            target,
            ap_iou_threshold=ap_iou_threshold,
        )

    if prediction_set.golden_queries is not None:
        golden_query_evaluation = evaluate_image(
            prediction_set.golden_queries,
            target,
            ap_iou_threshold=ap_iou_threshold,
        )
        golden_query_evaluation.matched_query_distances.extend(
            float(value) for value in prediction_set.golden_query_diagnostics.matched_query_distances
        )
        golden_query_evaluation.unmatched_query_closest_gt_distances.extend(
            float(value) for value in prediction_set.golden_query_diagnostics.unmatched_query_closest_gt_distances
        )

    return clustering_evaluation, gt_signature_evaluation, golden_query_evaluation


def _summarize_grouped_evaluations(
    clustering_evaluations: Sequence[ImageEvaluation],
    gt_signature_evaluations: Sequence[ImageEvaluation] | None,
    golden_query_evaluations: Sequence[ImageEvaluation] | None,
    object_counts: Sequence[int],
) -> Dict[int, Dict[str, Any]]:
    grouped_clustering = defaultdict(list)
    grouped_gt = defaultdict(list) if gt_signature_evaluations is not None else None
    grouped_golden = defaultdict(list) if golden_query_evaluations is not None else None

    for idx, object_count in enumerate(object_counts):
        key = int(object_count)
        grouped_clustering[key].append(clustering_evaluations[idx])
        if grouped_gt is not None:
            grouped_gt[key].append(gt_signature_evaluations[idx])
        if grouped_golden is not None:
            grouped_golden[key].append(golden_query_evaluations[idx])

    return {
        count: _build_summary_payload(
            summarize_evaluations(grouped_clustering[count]),
            None if grouped_gt is None else summarize_evaluations(grouped_gt[count]),
            None if grouped_golden is None else summarize_evaluations(grouped_golden[count]),
        )
        for count in sorted(grouped_clustering)
    }


@torch.no_grad()
def evaluate_system(
    system,
    *,
    dataset_length: int,
    height: int,
    width: int,
    max_objects: int,
    batch_size: int,
    device,
    seed: int = 0,
    ap_iou_threshold: float = 0.5,
):
    random_state = random.getstate()
    py_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    generator = SyntheticPanopticBatchGenerator(
        height=height,
        width=width,
        max_objects=max_objects,
        device=device,
    )
    dataset = BatchedSyntheticIterableDataset(
        generator=generator,
        total_samples=dataset_length,
        batch_size=batch_size,
    )

    was_training = system.training
    system.eval()

    clustering_evaluations: List[ImageEvaluation] = []
    gt_signature_evaluations: List[ImageEvaluation] | None = [] if system.supports_gt_prototypes else None
    golden_query_evaluations: List[ImageEvaluation] | None = [] if system.supports_gt_prototypes else None
    object_counts: List[int] = []
    signature_probe_records: list[dict[str, Any]] = []

    try:
        for batch, targets in dataset:
            batch = batch.to(device)
            raw = system.model(batch, ttt_steps_override=system.cfg.inference.ttt_steps)
            prediction_sets = _ensure_prediction_list(
                system.predictor.predict_evaluation_views_from_raw(system.model, raw, targets)
            )

            for batch_index, (prediction_set, target) in enumerate(zip(prediction_sets, targets)):
                signature_probe_records.extend(
                    _extract_signature_probe_records(
                        prediction_set,
                        batch[batch_index],
                        target,
                        batch_device=batch.device,
                    )
                )
                clustering_eval, gt_eval, golden_eval = _evaluate_prediction_set(
                    prediction_set,
                    target,
                    ap_iou_threshold=ap_iou_threshold,
                    signature_metric=getattr(system.model, "identity_similarity_metric", "dot"),
                )
                clustering_evaluations.append(clustering_eval)
                object_counts.append(int((target["labels"] != 0).sum().item()))
                if gt_signature_evaluations is not None and gt_eval is not None:
                    gt_signature_evaluations.append(gt_eval)
                if golden_query_evaluations is not None and golden_eval is not None:
                    golden_query_evaluations.append(golden_eval)
    finally:
        system.train(was_training)
        random.setstate(random_state)
        np.random.set_state(py_state)
        torch.random.set_rng_state(torch_state)

    overall = _build_summary_payload(
        summarize_evaluations(clustering_evaluations),
        None if gt_signature_evaluations is None else summarize_evaluations(gt_signature_evaluations),
        None if golden_query_evaluations is None else summarize_evaluations(golden_query_evaluations),
        _evaluate_signature_probe_summary(signature_probe_records, device=torch.device(device)),
    )
    by_count = _summarize_grouped_evaluations(
        clustering_evaluations,
        gt_signature_evaluations,
        golden_query_evaluations,
        object_counts,
    )
    return overall, by_count


@torch.no_grad()
def evaluate_system_many_configs(
    system,
    inference_cfgs: Dict[str, object],
    *,
    dataset_length: int,
    height: int,
    width: int,
    max_objects: int,
    batch_size: int,
    device,
    seed: int = 0,
    ap_iou_threshold: float = 0.5,
):
    random_state = random.getstate()
    py_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    generator = SyntheticPanopticBatchGenerator(
        height=height,
        width=width,
        max_objects=max_objects,
        device=device,
    )
    dataset = BatchedSyntheticIterableDataset(
        generator=generator,
        total_samples=dataset_length,
        batch_size=batch_size,
    )

    was_training = system.training
    system.eval()

    predictors = {
        key: ModularPrototypePredictor(cfg)
        for key, cfg in inference_cfgs.items()
    }
    clustering_evaluations = {key: [] for key in inference_cfgs}
    gt_signature_evaluations = {key: [] for key in inference_cfgs} if system.supports_gt_prototypes else None
    golden_query_evaluations = {key: [] for key in inference_cfgs} if system.supports_gt_prototypes else None
    object_counts: List[int] = []

    try:
        for batch, targets in dataset:
            batch = batch.to(device)
            configs_by_ttt_steps = defaultdict(list)
            for key, cfg in inference_cfgs.items():
                configs_by_ttt_steps[cfg.ttt_steps].append((key, predictors[key]))

            batch_object_counts = [int((target["labels"] != 0).sum().item()) for target in targets]
            object_counts.extend(batch_object_counts)

            for ttt_steps, group in configs_by_ttt_steps.items():
                raw = system.model(batch, ttt_steps_override=ttt_steps)
                for key, predictor in group:
                    prediction_sets = _ensure_prediction_list(
                        predictor.predict_evaluation_views_from_raw(system.model, raw, targets)
                    )
                    for prediction_set, target in zip(prediction_sets, targets):
                        clustering_eval, gt_eval, golden_eval = _evaluate_prediction_set(
                            prediction_set,
                            target,
                            ap_iou_threshold=ap_iou_threshold,
                            signature_metric=getattr(system.model, "identity_similarity_metric", "dot"),
                        )
                        clustering_evaluations[key].append(clustering_eval)
                        if gt_signature_evaluations is not None and gt_eval is not None:
                            gt_signature_evaluations[key].append(gt_eval)
                        if golden_query_evaluations is not None and golden_eval is not None:
                            golden_query_evaluations[key].append(golden_eval)
    finally:
        system.train(was_training)
        random.setstate(random_state)
        np.random.set_state(py_state)
        torch.random.set_rng_state(torch_state)

    results = {}
    for key, evaluations in clustering_evaluations.items():
        gt_evals = None if gt_signature_evaluations is None else gt_signature_evaluations[key]
        golden_evals = None if golden_query_evaluations is None else golden_query_evaluations[key]
        overall = _build_summary_payload(
            summarize_evaluations(evaluations),
            None if gt_evals is None else summarize_evaluations(gt_evals),
            None if golden_evals is None else summarize_evaluations(golden_evals),
        )
        by_count = _summarize_grouped_evaluations(evaluations, gt_evals, golden_evals, object_counts)
        results[key] = (overall, by_count)
    return results


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "-"
    return f"{value:.{digits}f}"


def _format_metric(metrics: Dict[str, Any], metric_key: str | None) -> str:
    if metric_key is None:
        return "-"

    value = metrics.get(metric_key)
    if metric_key.endswith("_mean"):
        std_key = metric_key[:-5] + "_std"
    else:
        std_key = f"{metric_key}_std"
    deviation = metrics.get(std_key)

    formatted_value = _format_float(value)
    if deviation is None:
        return formatted_value
    return f"{formatted_value}±{_format_float(deviation)}"


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(header)), max(len(str(row[idx])) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def _fmt(row):
        return " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_fmt(headers), separator]
    lines.extend(_fmt(row) for row in rows)
    return "\n".join(lines)


def _extract_signature_probe_records(
    prediction_set: EvaluationPredictionSet,
    image: torch.Tensor,
    target: Dict[str, Any],
    *,
    batch_device: torch.device,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if prediction_set.gt_signatures is None:
        return records

    height = image.shape[-2]
    width = image.shape[-1]
    labels = target["labels"].to(batch_device)
    masks = target["masks"].to(batch_device).float()
    boxes = target["boxes"].to(batch_device).float()
    signatures = prediction_set.gt_signatures.all_signature_embeddings.to(batch_device)

    if signatures.shape[0] <= 1:
        return records

    fg_signatures = signatures[1:]
    fg_labels = labels[1:]
    fg_masks = masks[1:]
    fg_boxes = boxes[1:]
    fg_count = int(fg_labels.shape[0])

    areas = fg_masks.flatten(1).sum(dim=1)
    centers_x = (fg_boxes[:, 0] + fg_boxes[:, 2]) * 0.5
    centers_y = (fg_boxes[:, 1] + fg_boxes[:, 3]) * 0.5
    size_scalar = torch.sqrt(areas / float(height * width)).unsqueeze(-1)
    position_xy = torch.stack(
        [centers_x / float(width - 1), centers_y / float(height - 1)],
        dim=-1,
    )
    color = target.get("color").to(batch_device).float()[1:] / 255.0

    for obj_idx in range(fg_signatures.shape[0]):
        records.append(
            {
                "signature": fg_signatures[obj_idx].detach().cpu(),
                "size": size_scalar[obj_idx].detach().cpu(),
                "position": position_xy[obj_idx].detach().cpu(),
                "color": color[obj_idx].detach().cpu(),
                "class": fg_labels[obj_idx].detach().cpu(),
                "object_count": fg_count,
            }
        )

    return records


def _train_probe_regressor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 256,
) -> tuple[_ProbeMLP, torch.Tensor, torch.Tensor]:
    mean = train_y.mean(dim=0, keepdim=True)
    std = train_y.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_y_norm = (train_y - mean) / std
    val_y_norm = (val_y - mean) / std

    model = _ProbeMLP(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = math.inf

    for _ in range(epochs):
        perm = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            batch_x = train_x[idx].to(device)
            batch_y = train_y_norm[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = model(val_x.to(device))
            val_loss = float(F.mse_loss(pred, val_y_norm.to(device)).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, mean, std


def _train_probe_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    num_classes: int,
    device: torch.device,
    epochs: int = 40,
    batch_size: int = 256,
) -> _ProbeMLP:
    model = _ProbeMLP(train_x.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_val = math.inf

    for _ in range(epochs):
        perm = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            batch_x = train_x[idx].to(device)
            batch_y = train_y[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            val_loss = float(F.cross_entropy(model(val_x.to(device)), val_y.to(device)).item())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def _regression_r2(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    target_mean = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target_mean) ** 2).sum(dim=0).clamp_min(1e-12)
    return (1.0 - (ss_res / ss_tot)).tolist()


def _evaluate_signature_probe_summary(
    records: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any] | None:
    if len(records) < 32:
        return None

    signatures = torch.stack([record["signature"] for record in records], dim=0)
    size = torch.stack([record["size"] for record in records], dim=0)
    position = torch.stack([record["position"] for record in records], dim=0)
    color = torch.stack([record["color"] for record in records], dim=0)
    labels = torch.stack([record["class"] for record in records], dim=0).long()
    object_counts = torch.tensor([int(record["object_count"]) for record in records], dtype=torch.long)

    perm = torch.randperm(signatures.shape[0])
    train_end = int(0.7 * signatures.shape[0])
    val_end = int(0.85 * signatures.shape[0])
    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]
    if train_idx.numel() == 0 or val_idx.numel() == 0 or test_idx.numel() == 0:
        return None

    train_x = signatures[train_idx]
    val_x = signatures[val_idx]
    test_x = signatures[test_idx]

    with torch.enable_grad():
        size_model, size_mean, size_std = _train_probe_regressor(
            train_x, size[train_idx], val_x, size[val_idx], device=device
        )
        pos_model, pos_mean, pos_std = _train_probe_regressor(
            train_x, position[train_idx], val_x, position[val_idx], device=device
        )
        color_model, color_mean, color_std = _train_probe_regressor(
            train_x, color[train_idx], val_x, color[val_idx], device=device
        )
        cls_model = _train_probe_classifier(
            train_x,
            labels[train_idx],
            val_x,
            labels[val_idx],
            num_classes=int(labels.max().item()) + 1,
            device=device,
        )

    with torch.no_grad():
        size_pred = size_model(test_x.to(device)).cpu() * size_std + size_mean
        pos_pred = pos_model(test_x.to(device)).cpu() * pos_std + pos_mean
        color_pred = color_model(test_x.to(device)).cpu() * color_std + color_mean
        cls_pred = cls_model(test_x.to(device)).cpu().argmax(dim=-1)

    per_count: dict[int, dict[str, float]] = {}
    test_counts = object_counts[test_idx]
    for count in sorted(int(v) for v in test_counts.unique().tolist()):
        idx = torch.nonzero(test_counts == count, as_tuple=False).squeeze(1)
        per_count[count] = {
            "num_samples": int(idx.numel()),
            "size_r2": _regression_r2(size_pred[idx], size[test_idx][idx])[0],
            "pos_r2_x": _regression_r2(pos_pred[idx], position[test_idx][idx])[0],
            "pos_r2_y": _regression_r2(pos_pred[idx], position[test_idx][idx])[1],
            "color_r2_r": _regression_r2(color_pred[idx], color[test_idx][idx])[0],
            "color_r2_g": _regression_r2(color_pred[idx], color[test_idx][idx])[1],
            "color_r2_b": _regression_r2(color_pred[idx], color[test_idx][idx])[2],
            "class_acc": float((cls_pred[idx] == labels[test_idx][idx]).float().mean().item()),
        }

    return {
        "num_train": int(train_idx.numel()),
        "num_val": int(val_idx.numel()),
        "num_test": int(test_idx.numel()),
        "size_r2": _regression_r2(size_pred, size[test_idx])[0],
        "pos_r2_x": _regression_r2(pos_pred, position[test_idx])[0],
        "pos_r2_y": _regression_r2(pos_pred, position[test_idx])[1],
        "color_r2_r": _regression_r2(color_pred, color[test_idx])[0],
        "color_r2_g": _regression_r2(color_pred, color[test_idx])[1],
        "color_r2_b": _regression_r2(color_pred, color[test_idx])[2],
        "class_acc": float((cls_pred == labels[test_idx]).float().mean().item()),
        "by_count": per_count,
    }


def _append_overall_row(
    rows: List[List[str]],
    metrics: Dict[str, Any],
    *,
    value_columns: Sequence[tuple[str, str | None]],
):
    row = ["all"]
    for column_type, metric_key in value_columns:
        if column_type == "images":
            row.append(str(metrics["num_images"]))
        elif column_type == "gt":
            row.append(str(metrics["num_gt_instances"]))
        elif column_type == "pred":
            row.append(str(metrics["num_predictions"]))
        elif column_type == "metric":
            row.append(_format_metric(metrics, metric_key))
        else:
            raise ValueError(f"Unsupported overall row column type: {column_type}")
    rows.append(row)


def format_metrics_table(
    overall: Dict[str, Any],
    by_count: Dict[int, Dict[str, Any]],
    *,
    ap_threshold: float,
) -> str:
    sections: List[str] = []

    table_specs = [
        (
            "gt_signatures",
            "GT signatures by object count",
            [
                "obj.",
                "img.",
                "gt",
                "pred",
                "IoU_m",
                "Iou_b",
                f"AP@{ap_threshold:.2f}",
            ],
            [
                ("images", None),
                ("gt", None),
                ("pred", None),
                ("metric", "mean_iou"),
                ("metric", "mean_iou_box"),
                ("metric", "ap"),
            ],
        ),
        (
            "golden_queries",
            "Golden queries by object count",
            [
                "obj.",
                "img.",
                "gt",
                "pred",
                "IoU_m",
                "Iou_b",
                f"AP@{ap_threshold:.2f}",
                "match_d",
                "unmatch_d",
            ],
            [
                ("images", None),
                ("gt", None),
                ("pred", None),
                ("metric", "mean_iou"),
                ("metric", "mean_iou_box"),
                ("metric", "ap"),
                ("metric", "matched_query_cosine_distance_mean"),
                ("metric", "unmatched_query_closest_gt_cosine_distance_mean"),
            ],
        ),
        (
            "clustering",
            "Clustering by object count",
            [
                "obj.",
                "img.",
                "gt",
                "pred",
                "IoU_m",
                "IoU_b",
                f"AP@{ap_threshold:.2f}",
                "count_MAE",
                "count_acc",
                "chamfer",
                "hausdorff",
            ],
            [
                ("images", None),
                ("gt", None),
                ("pred", None),
                ("metric", "mean_iou"),
                ("metric", "mean_iou_box"),
                ("metric", "ap"),
                ("metric", "mean_abs_count_error"),
                ("metric", "exact_count_accuracy"),
                ("metric", "signature_chamfer_distance_mean"),
                ("metric", "signature_hausdorff_distance_mean"),
            ],
        ),
    ]

    for key, title, headers, value_columns in table_specs:
        overall_metrics = overall.get(key)
        if overall_metrics is None:
            continue

        rows: List[List[str]] = []
        for object_count, metrics in by_count.items():
            eval_metrics = metrics.get(key)
            if eval_metrics is None:
                continue

            row = [str(object_count)]
            for column_type, metric_key in value_columns:
                if column_type == "images":
                    row.append(str(eval_metrics["num_images"]))
                elif column_type == "gt":
                    row.append(str(eval_metrics["num_gt_instances"]))
                elif column_type == "pred":
                    row.append(str(eval_metrics["num_predictions"]))
                elif column_type == "metric":
                    row.append(_format_metric(eval_metrics, metric_key))
                else:
                    raise ValueError(f"Unsupported table column type: {column_type}")
            rows.append(row)

        _append_overall_row(rows, overall_metrics, value_columns=value_columns)
        sections.append(title + "\n" + _format_table(headers, rows))

    probe_metrics = overall.get("signature_probes")
    if probe_metrics is not None:
        headers = [
            "obj.",
            "n",
            "size_r2",
            "pos_r2_x",
            "pos_r2_y",
            "color_r2_r",
            "color_r2_g",
            "color_r2_b",
            "class_acc",
        ]
        rows: List[List[str]] = []
        for object_count, metrics in sorted(probe_metrics.get("by_count", {}).items()):
            rows.append(
                [
                    str(object_count),
                    str(metrics["num_samples"]),
                    _format_float(metrics.get("size_r2")),
                    _format_float(metrics.get("pos_r2_x")),
                    _format_float(metrics.get("pos_r2_y")),
                    _format_float(metrics.get("color_r2_r")),
                    _format_float(metrics.get("color_r2_g")),
                    _format_float(metrics.get("color_r2_b")),
                    _format_float(metrics.get("class_acc")),
                ]
            )
        rows.append(
            [
                "all",
                str(probe_metrics["num_test"]),
                _format_float(probe_metrics.get("size_r2")),
                _format_float(probe_metrics.get("pos_r2_x")),
                _format_float(probe_metrics.get("pos_r2_y")),
                _format_float(probe_metrics.get("color_r2_r")),
                _format_float(probe_metrics.get("color_r2_g")),
                _format_float(probe_metrics.get("color_r2_b")),
                _format_float(probe_metrics.get("class_acc")),
            ]
        )
        sections.append("Signature probes by object count\n" + _format_table(headers, rows))

    return "\n\n".join(sections)
