from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .dataset import SyntheticPanopticBatchGenerator, BatchedSyntheticIterableDataset
from .predictor import ModularPrototypePredictor


@dataclass
class ImageEvaluation:
    num_gt: int
    num_pred: int
    matched_iou_sum: float
    num_tp: int
    prediction_records: List[Tuple[float, int]]


def _stack_masks(masks: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(masks) == 0:
        return torch.zeros((0, 1, 1), dtype=torch.bool)
    return torch.stack([mask.detach().to(dtype=torch.bool, device="cpu") for mask in masks], dim=0)


def _to_prediction_tensors(prediction: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_masks = _stack_masks(prediction.get("resolved_masks", []))
    pred_labels = torch.as_tensor(prediction.get("resolved_labels", []), dtype=torch.long)
    pred_scores = torch.as_tensor(prediction.get("resolved_scores", []), dtype=torch.float32)
    keep = pred_labels != 0
    return pred_masks[keep], pred_labels[keep], pred_scores[keep]


def _to_target_tensors(target: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    gt_masks = target["masks"].detach().to(dtype=torch.bool, device="cpu")
    gt_labels = target["labels"].detach().to(dtype=torch.long, device="cpu")
    keep = gt_labels != 0
    return gt_masks[keep], gt_labels[keep]


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
    prediction: Dict,
    target: Dict,
    *,
    ap_iou_threshold: float = 0.5,
) -> ImageEvaluation:
    pred_masks, pred_labels, pred_scores = _to_prediction_tensors(prediction)
    gt_masks, gt_labels = _to_target_tensors(target)
    matches, _ = hungarian_match_instances(pred_masks, pred_labels, gt_masks, gt_labels)

    pred_is_tp = torch.zeros(pred_masks.shape[0], dtype=torch.int64)
    matched_iou_sum = 0.0

    for pred_idx, _, match_iou in matches:
        matched_iou_sum += match_iou
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
        num_tp=int(pred_is_tp.sum().item()),
        prediction_records=prediction_records,
    )


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


def summarize_evaluations(image_evaluations: Sequence[ImageEvaluation]) -> Dict[str, float]:
    num_images = len(image_evaluations)
    total_gt = sum(item.num_gt for item in image_evaluations)
    total_pred = sum(item.num_pred for item in image_evaluations)
    total_tp = sum(item.num_tp for item in image_evaluations)
    total_matched_iou = sum(item.matched_iou_sum for item in image_evaluations)

    prediction_records: List[Tuple[float, int]] = []
    for item in image_evaluations:
        prediction_records.extend(item.prediction_records)

    mean_iou = total_matched_iou / total_gt if total_gt > 0 else 0.0
    ap = _compute_average_precision(prediction_records, total_gt)
    precision = total_tp / total_pred if total_pred > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0

    return {
        "num_images": num_images,
        "num_gt_instances": total_gt,
        "num_predictions": total_pred,
        "num_true_positives": total_tp,
        "mean_iou": mean_iou,
        "ap": ap,
        "precision": precision,
        "recall": recall,
    }


def summarize_by_object_count(
    image_evaluations: Sequence[ImageEvaluation],
    object_counts: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    grouped = defaultdict(list)
    for image_eval, object_count in zip(image_evaluations, object_counts):
        grouped[int(object_count)].append(image_eval)

    return {
        count: summarize_evaluations(grouped[count])
        for count in sorted(grouped)
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
    use_gt_prototypes: bool = False,
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

    image_evaluations = []
    object_counts = []

    try:
        for batch, targets in dataset:
            batch = batch.to(device)
            if use_gt_prototypes:
                predictions = system.predict_with_gt_prototypes(batch, targets)
            else:
                predictions = system.predict(batch)

            if not isinstance(predictions, list):
                predictions = [predictions]

            for prediction, target in zip(predictions, targets):
                image_evaluations.append(
                    evaluate_image(
                        prediction,
                        target,
                        ap_iou_threshold=ap_iou_threshold,
                    )
                )
                object_counts.append(int((target["labels"] != 0).sum().item()))
    finally:
        system.train(was_training)
        random.setstate(random_state)
        np.random.set_state(py_state)
        torch.random.set_rng_state(torch_state)

    overall = summarize_evaluations(image_evaluations)
    by_count = summarize_by_object_count(image_evaluations, object_counts)
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
    use_gt_prototypes: bool = False,
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
    image_evaluations = {key: [] for key in inference_cfgs}
    object_counts = []

    try:
        for batch, targets in dataset:
            batch = batch.to(device)
            configs_by_ttt_steps = defaultdict(list)
            for key, cfg in inference_cfgs.items():
                configs_by_ttt_steps[cfg.ttt_steps].append((key, predictors[key]))

            for ttt_steps, group in configs_by_ttt_steps.items():
                raw = system.model(batch, ttt_steps_override=ttt_steps)
                if use_gt_prototypes:
                    predictions_by_key = {
                        key: predictor.predict_from_raw_with_gt_prototypes(system.model, raw, targets)
                        for key, predictor in group
                    }
                else:
                    predictions_by_key = {
                        key: predictor.predict_from_raw(system.model, raw)
                        for key, predictor in group
                    }

                for key, predictions in predictions_by_key.items():
                    if not isinstance(predictions, list):
                        predictions = [predictions]

                    for prediction, target in zip(predictions, targets):
                        image_evaluations[key].append(
                            evaluate_image(
                                prediction,
                                target,
                                ap_iou_threshold=ap_iou_threshold,
                            )
                        )

            object_counts.extend(int((target["labels"] != 0).sum().item()) for target in targets)
    finally:
        system.train(was_training)
        random.setstate(random_state)
        np.random.set_state(py_state)
        torch.random.set_rng_state(torch_state)

    results = {}
    for key, evaluations in image_evaluations.items():
        overall = summarize_evaluations(evaluations)
        by_count = summarize_by_object_count(evaluations, object_counts)
        results[key] = (overall, by_count)
    return results


def format_metrics_table(
    overall: Dict[str, float],
    by_count: Dict[int, Dict[str, float]],
    *,
    ap_threshold: float,
) -> str:
    headers = ["split", "images", "gt", "pred", "mIoU", f"AP@{ap_threshold:.2f}", "P", "R"]
    rows = [[
        "overall",
        str(overall["num_images"]),
        str(overall["num_gt_instances"]),
        str(overall["num_predictions"]),
        f"{overall['mean_iou']:.4f}",
        f"{overall['ap']:.4f}",
        f"{overall['precision']:.4f}",
        f"{overall['recall']:.4f}",
    ]]

    for object_count, metrics in by_count.items():
        rows.append([
            f"{object_count} obj",
            str(metrics["num_images"]),
            str(metrics["num_gt_instances"]),
            str(metrics["num_predictions"]),
            f"{metrics['mean_iou']:.4f}",
            f"{metrics['ap']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
        ])

    widths = [
        max(len(header), max(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]

    def _fmt(row):
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    table_lines = [_fmt(headers), separator]
    table_lines.extend(_fmt(row) for row in rows)
    return "\n".join(table_lines)
