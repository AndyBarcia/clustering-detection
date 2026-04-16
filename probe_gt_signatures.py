import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.dataset import SyntheticPanopticBatchGenerator
from src.panoptic import load_system_checkpoint


CLASS_NAMES = {
    0: "Background",
    1: "Square",
    2: "Triangle",
}


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int, depth: int, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe what GT signatures encode by training simple MLPs on top of them."
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/test-influence-head-no-per-layer-weights-BCE-seed-head-CE-mask-attn-residuals-seed-hung-mask-bias-3/checkpoint.pt",
        help="Checkpoint to analyze.",
    )
    parser.add_argument("--device", default=None, help="Execution device. Defaults to cuda if available.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num-images", type=int, default=1024, help="Number of synthetic images to sample.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size used to extract GT signatures.")
    parser.add_argument("--probe-batch-size", type=int, default=256, help="Batch size for probe training.")
    parser.add_argument("--epochs", type=int, default=80, help="Probe training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Probe optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Probe optimizer weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Probe MLP hidden size.")
    parser.add_argument("--depth", type=int, default=3, help="Number of linear layers in each probe.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Probe dropout.")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Train split fraction.")
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Validation split fraction.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save a JSON report. Defaults next to the checkpoint.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_probe_dataset(system, num_images: int, batch_size: int, device: torch.device):
    model = system.model
    model.eval()

    generator = SyntheticPanopticBatchGenerator(
        height=256,
        width=256,
        max_objects=10,
        device=device,
    )

    all_signatures = []
    all_size = []
    all_position = []
    all_class = []
    all_object_pixels = []
    all_box_wh = []
    all_color = []
    all_image_object_count = []
    all_image_ids = []

    produced = 0
    progress = tqdm(total=num_images, desc="Extracting GT signatures", unit="image")
    while produced < num_images:
        current_bs = min(batch_size, num_images - produced)
        images, targets = generator.generate_batch(current_bs, start_idx=produced)
        raw = model(images)

        for sample_index in range(current_bs):
            labels = targets[sample_index]["labels"].to(device)
            masks = targets[sample_index]["masks"].to(device).float()
            boxes = targets[sample_index]["boxes"].to(device).float()
            image = images[sample_index]

            gt_masks = masks.unsqueeze(0)
            gt_labels = labels.unsqueeze(0)
            gt_pad_mask = torch.ones((1, labels.shape[0]), dtype=torch.bool, device=device)

            signatures = model.encode_gts(
                raw.memory[sample_index:sample_index + 1],
                raw.features[sample_index:sample_index + 1],
                gt_masks,
                gt_labels,
                gt_pad_mask,
                ttt_steps_override=system.cfg.inference.ttt_steps,
            )[0]

            if signatures.shape[0] <= 1:
                continue

            fg_signatures = signatures[1:]
            fg_labels = labels[1:]
            fg_masks = masks[1:]
            fg_boxes = boxes[1:]

            areas = fg_masks.flatten(1).sum(dim=1)
            box_w = (fg_boxes[:, 2] - fg_boxes[:, 0] + 1.0).clamp_min(1.0)
            box_h = (fg_boxes[:, 3] - fg_boxes[:, 1] + 1.0).clamp_min(1.0)
            centers_x = (fg_boxes[:, 0] + fg_boxes[:, 2]) * 0.5
            centers_y = (fg_boxes[:, 1] + fg_boxes[:, 3]) * 0.5
            color_mean = (
                torch.einsum("nhw,chw->nc", fg_masks, image)
                / areas.unsqueeze(-1).clamp_min(1e-6)
            )

            size_scalar = torch.sqrt(areas / float(generator.height * generator.width)).unsqueeze(-1)
            position_xy = torch.stack(
                [centers_x / float(generator.width - 1), centers_y / float(generator.height - 1)],
                dim=-1,
            )
            box_wh = torch.stack(
                [box_w / float(generator.width), box_h / float(generator.height)],
                dim=-1,
            )

            all_signatures.append(fg_signatures.detach().cpu())
            all_size.append(size_scalar.detach().cpu())
            all_position.append(position_xy.detach().cpu())
            all_class.append(fg_labels.detach().cpu())
            all_object_pixels.append((areas / float(generator.height * generator.width)).unsqueeze(-1).detach().cpu())
            all_box_wh.append(box_wh.detach().cpu())
            all_color.append(color_mean.detach().cpu())
            all_image_object_count.append(torch.full((fg_signatures.shape[0],), fg_signatures.shape[0], dtype=torch.long))
            all_image_ids.append(torch.full((fg_signatures.shape[0],), produced + sample_index, dtype=torch.long))

        produced += current_bs
        progress.update(current_bs)
    progress.close()

    signatures = torch.cat(all_signatures, dim=0)
    size = torch.cat(all_size, dim=0)
    position = torch.cat(all_position, dim=0)
    cls = torch.cat(all_class, dim=0)
    area_fraction = torch.cat(all_object_pixels, dim=0)
    box_wh = torch.cat(all_box_wh, dim=0)
    color = torch.cat(all_color, dim=0)
    image_object_count = torch.cat(all_image_object_count, dim=0)
    image_ids = torch.cat(all_image_ids, dim=0)

    return {
        "signatures": signatures,
        "size": size,
        "position": position,
        "class": cls,
        "area_fraction": area_fraction,
        "box_wh": box_wh,
        "color": color,
        "image_object_count": image_object_count,
        "image_ids": image_ids,
    }


def split_indices(num_items: int, train_fraction: float, val_fraction: float):
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be between 0 and 1.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.")

    perm = torch.randperm(num_items)
    train_end = int(num_items * train_fraction)
    val_end = train_end + int(num_items * val_fraction)
    return perm[:train_end], perm[train_end:val_end], perm[val_end:]


def build_loader(features: torch.Tensor, targets: torch.Tensor, batch_size: int, shuffle: bool):
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def standardize_regression_targets(train_targets: torch.Tensor, targets: torch.Tensor):
    mean = train_targets.mean(dim=0, keepdim=True)
    std = train_targets.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (targets - mean) / std, mean, std


def regression_metrics(pred: torch.Tensor, target: torch.Tensor):
    abs_err = (pred - target).abs()
    mse = ((pred - target) ** 2).mean(dim=0)
    rmse = torch.sqrt(mse)
    mae = abs_err.mean(dim=0)

    target_mean = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target_mean) ** 2).sum(dim=0).clamp_min(1e-12)
    r2 = 1.0 - (ss_res / ss_tot)

    return {
        "rmse": rmse.tolist(),
        "mae": mae.tolist(),
        "r2": r2.tolist(),
    }


@torch.no_grad()
def evaluate_regressor(model, features, target, mean, std, batch_size: int, device: torch.device):
    model.eval()
    preds = []
    loader = build_loader(features, target, batch_size=batch_size, shuffle=False)
    for batch_features, _ in loader:
        batch_features = batch_features.to(device)
        pred_norm = model(batch_features)
        pred = pred_norm.cpu() * std + mean
        preds.append(pred)
    pred = torch.cat(preds, dim=0)
    return regression_metrics(pred, target)


@torch.no_grad()
def evaluate_classifier(model, features, target, batch_size: int, device: torch.device):
    model.eval()
    logits = []
    loader = build_loader(features, target, batch_size=batch_size, shuffle=False)
    for batch_features, _ in loader:
        logits.append(model(batch_features.to(device)).cpu())
    logits = torch.cat(logits, dim=0)
    pred = logits.argmax(dim=-1)
    accuracy = (pred == target).float().mean().item()

    return {
        "accuracy": accuracy,
    }


def subset_regression_metrics(model, features, target, mean, std, indices, batch_size: int, device: torch.device):
    if indices.numel() == 0:
        return None
    return evaluate_regressor(model, features[indices], target[indices], mean, std, batch_size, device)


def subset_classification_metrics(model, features, target, indices, batch_size: int, device: torch.device):
    if indices.numel() == 0:
        return None
    return evaluate_classifier(model, features[indices], target[indices], batch_size, device)


def evaluate_by_object_count(
    object_counts: torch.Tensor,
    size_model,
    size_mean,
    size_std,
    pos_model,
    pos_mean,
    pos_std,
    color_model,
    color_mean,
    color_std,
    cls_model,
    features,
    size_target,
    position_target,
    color_target,
    class_target,
    batch_size: int,
    device: torch.device,
):
    per_count = {}
    unique_counts = sorted(int(count) for count in object_counts.unique().tolist())
    for count in unique_counts:
        indices = torch.nonzero(object_counts == count, as_tuple=False).squeeze(1)
        per_count[str(count)] = {
            "num_objects_in_image": count,
            "num_test_samples": int(indices.numel()),
            "size_probe": subset_regression_metrics(
                size_model, features, size_target, size_mean, size_std, indices, batch_size, device
            ),
            "position_probe": subset_regression_metrics(
                pos_model, features, position_target, pos_mean, pos_std, indices, batch_size, device
            ),
            "color_probe": subset_regression_metrics(
                color_model, features, color_target, color_mean, color_std, indices, batch_size, device
            ),
            "class_probe": subset_classification_metrics(
                cls_model, features, class_target, indices, batch_size, device
            ),
        }
    return per_count


def build_two_object_relative_dataset(dataset):
    signatures = dataset["signatures"]
    sizes = dataset["size"]
    positions = dataset["position"]
    image_counts = dataset["image_object_count"]
    image_ids = dataset["image_ids"]

    keep = image_counts == 2
    if int(keep.sum().item()) == 0:
        return None

    kept_signatures = signatures[keep]
    kept_sizes = sizes[keep]
    kept_positions = positions[keep]
    kept_image_ids = image_ids[keep]

    unique_ids = kept_image_ids.unique(sorted=True)
    rel_features = []
    rel_size_targets = []
    rel_position_targets = []

    for image_id in unique_ids.tolist():
        indices = torch.nonzero(kept_image_ids == image_id, as_tuple=False).squeeze(1)
        if indices.numel() != 2:
            continue

        a_idx = int(indices[0].item())
        b_idx = int(indices[1].item())

        for self_idx, other_idx in ((a_idx, b_idx), (b_idx, a_idx)):
            self_signature = kept_signatures[self_idx]
            self_size = kept_sizes[self_idx]
            other_size = kept_sizes[other_idx]
            self_position = kept_positions[self_idx]
            other_position = kept_positions[other_idx]

            relative_log_size = torch.log(self_size.clamp_min(1e-6) / other_size.clamp_min(1e-6))
            relative_position = other_position - self_position

            rel_features.append(self_signature.unsqueeze(0))
            rel_size_targets.append(relative_log_size.unsqueeze(0))
            rel_position_targets.append(relative_position.unsqueeze(0))

    if not rel_features:
        return None

    return {
        "features": torch.cat(rel_features, dim=0),
        "relative_size": torch.cat(rel_size_targets, dim=0),
        "relative_position": torch.cat(rel_position_targets, dim=0),
        "num_source_images": int(unique_ids.numel()),
    }


def train_regressor(
    train_x,
    train_y,
    val_x,
    val_y,
    input_dim: int,
    output_dim: int,
    args,
    device: torch.device,
    probe_name: str,
):
    train_y_norm, mean, std = standardize_regression_targets(train_y, train_y)
    val_y_norm = (val_y - mean) / std

    model = MLPRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = math.inf

    train_loader = build_loader(train_x, train_y_norm, batch_size=args.probe_batch_size, shuffle=True)
    val_loader = build_loader(val_x, val_y_norm, batch_size=args.probe_batch_size, shuffle=False)

    epoch_bar = tqdm(range(args.epochs), desc=f"Training {probe_name} probe", unit="epoch")
    for _ in epoch_bar:
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = F.mse_loss(model(batch_x), batch_y, reduction="sum")
                total += float(loss.item())
                count += int(batch_y.numel())
        val_loss = total / max(count, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epoch_bar.set_postfix(val_mse=f"{val_loss:.6f}", best=f"{best_val:.6f}")
    epoch_bar.close()

    model.load_state_dict(best_state)
    return model, mean, std, best_val


def train_classifier(train_x, train_y, val_x, val_y, input_dim: int, num_classes: int, args, device: torch.device, probe_name: str):
    model = MLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val = math.inf

    train_loader = build_loader(train_x, train_y, batch_size=args.probe_batch_size, shuffle=True)
    val_loader = build_loader(val_x, val_y, batch_size=args.probe_batch_size, shuffle=False)

    epoch_bar = tqdm(range(args.epochs), desc=f"Training {probe_name} probe", unit="epoch")
    for _ in epoch_bar:
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                loss = F.cross_entropy(model(batch_x), batch_y, reduction="sum")
                total += float(loss.item())
                count += int(batch_y.shape[0])
        val_loss = total / max(count, 1)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epoch_bar.set_postfix(val_ce=f"{val_loss:.6f}", best=f"{best_val:.6f}")
    epoch_bar.close()

    model.load_state_dict(best_state)
    return model, best_val


def summarize_class_distribution(labels: torch.Tensor):
    values, counts = labels.unique(return_counts=True)
    return {
        CLASS_NAMES.get(int(value), str(int(value))): int(count)
        for value, count in zip(values.tolist(), counts.tolist())
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else checkpoint_path.with_name(f"{checkpoint_path.stem}_gt_signature_probe_report.json")
    )

    device = resolve_device(args.device)
    system, _ = load_system_checkpoint(str(checkpoint_path), map_location=device)
    system = system.to(device)
    system.eval()

    dataset = extract_probe_dataset(
        system=system,
        num_images=args.num_images,
        batch_size=args.batch_size,
        device=device,
    )
    signatures = dataset["signatures"]
    size_target = dataset["size"]
    position_target = dataset["position"]
    class_target = dataset["class"]
    color_target = dataset["color"]
    image_object_count = dataset["image_object_count"]
    relative_two_object_dataset = build_two_object_relative_dataset(dataset)

    train_idx, val_idx, test_idx = split_indices(
        num_items=signatures.shape[0],
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )

    train_x = signatures[train_idx]
    val_x = signatures[val_idx]
    test_x = signatures[test_idx]

    size_model, size_mean, size_std, size_best_val = train_regressor(
        train_x, size_target[train_idx], val_x, size_target[val_idx],
        input_dim=signatures.shape[1], output_dim=size_target.shape[1], args=args, device=device, probe_name="size",
    )
    pos_model, pos_mean, pos_std, pos_best_val = train_regressor(
        train_x, position_target[train_idx], val_x, position_target[val_idx],
        input_dim=signatures.shape[1], output_dim=position_target.shape[1], args=args, device=device, probe_name="position",
    )
    color_model, color_mean, color_std, color_best_val = train_regressor(
        train_x, color_target[train_idx], val_x, color_target[val_idx],
        input_dim=signatures.shape[1], output_dim=color_target.shape[1], args=args, device=device, probe_name="color",
    )
    cls_model, cls_best_val = train_classifier(
        train_x, class_target[train_idx], val_x, class_target[val_idx],
        input_dim=signatures.shape[1], num_classes=int(class_target.max().item()) + 1, args=args, device=device, probe_name="class",
    )

    size_metrics = evaluate_regressor(
        size_model, test_x, size_target[test_idx], size_mean, size_std, args.probe_batch_size, device
    )
    position_metrics = evaluate_regressor(
        pos_model, test_x, position_target[test_idx], pos_mean, pos_std, args.probe_batch_size, device
    )
    color_metrics = evaluate_regressor(
        color_model, test_x, color_target[test_idx], color_mean, color_std, args.probe_batch_size, device
    )
    class_metrics = evaluate_classifier(
        cls_model, test_x, class_target[test_idx], args.probe_batch_size, device
    )

    relative_two_object_results = None
    if relative_two_object_dataset is not None:
        rel_features = relative_two_object_dataset["features"]
        rel_size_target = relative_two_object_dataset["relative_size"]
        rel_position_target = relative_two_object_dataset["relative_position"]

        rel_train_idx, rel_val_idx, rel_test_idx = split_indices(
            num_items=rel_features.shape[0],
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )

        rel_train_x = rel_features[rel_train_idx]
        rel_val_x = rel_features[rel_val_idx]
        rel_test_x = rel_features[rel_test_idx]

        rel_size_model, rel_size_mean, rel_size_std, rel_size_best_val = train_regressor(
            rel_train_x,
            rel_size_target[rel_train_idx],
            rel_val_x,
            rel_size_target[rel_val_idx],
            input_dim=rel_features.shape[1],
            output_dim=rel_size_target.shape[1],
            args=args,
            device=device,
            probe_name="relative-size-2obj",
        )
        rel_pos_model, rel_pos_mean, rel_pos_std, rel_pos_best_val = train_regressor(
            rel_train_x,
            rel_position_target[rel_train_idx],
            rel_val_x,
            rel_position_target[rel_val_idx],
            input_dim=rel_features.shape[1],
            output_dim=rel_position_target.shape[1],
            args=args,
            device=device,
            probe_name="relative-position-2obj",
        )

        relative_two_object_results = {
            "dataset_summary": {
                "num_source_images": relative_two_object_dataset["num_source_images"],
                "num_samples_total": int(rel_features.shape[0]),
                "num_train": int(rel_train_idx.numel()),
                "num_val": int(rel_val_idx.numel()),
                "num_test": int(rel_test_idx.numel()),
                "relative_size_target": "log(self_size / other_size)",
                "relative_position_target": "other_center - self_center, normalized to [-1, 1]-like image coordinates",
            },
            "relative_size_probe": {
                "best_val_mse": rel_size_best_val,
                "test_metrics": evaluate_regressor(
                    rel_size_model,
                    rel_test_x,
                    rel_size_target[rel_test_idx],
                    rel_size_mean,
                    rel_size_std,
                    args.probe_batch_size,
                    device,
                ),
            },
            "relative_position_probe": {
                "best_val_mse": rel_pos_best_val,
                "test_metrics": evaluate_regressor(
                    rel_pos_model,
                    rel_test_x,
                    rel_position_target[rel_test_idx],
                    rel_pos_mean,
                    rel_pos_std,
                    args.probe_batch_size,
                    device,
                ),
            },
        }

    report = {
        "checkpoint": str(checkpoint_path),
        "probe_config": {
            "seed": args.seed,
            "num_images": args.num_images,
            "feature_dim": int(signatures.shape[1]),
            "probe_batch_size": args.probe_batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "dropout": args.dropout,
        },
        "dataset_summary": {
            "num_objects_total": int(signatures.shape[0]),
            "num_train": int(train_idx.numel()),
            "num_val": int(val_idx.numel()),
            "num_test": int(test_idx.numel()),
            "class_distribution": summarize_class_distribution(class_target),
            "object_count_distribution": {
                str(int(value)): int(count)
                for value, count in zip(*image_object_count.unique(return_counts=True))
            },
            "size_target": "sqrt(mask_area / image_area)",
            "position_target": "bounding-box center (x, y), normalized to [0, 1]",
        },
        "results": {
            "size_probe": {
                "best_val_mse": size_best_val,
                "test_metrics": size_metrics,
            },
            "position_probe": {
                "best_val_mse": pos_best_val,
                "test_metrics": position_metrics,
            },
            "color_probe": {
                "best_val_mse": color_best_val,
                "test_metrics": color_metrics,
            },
            "class_probe": {
                "best_val_cross_entropy": cls_best_val,
                "test_metrics": class_metrics,
            },
            "relative_two_object_probes": relative_two_object_results,
            "by_num_objects_in_image": evaluate_by_object_count(
                image_object_count[test_idx],
                size_model,
                size_mean,
                size_std,
                pos_model,
                pos_mean,
                pos_std,
                color_model,
                color_mean,
                color_std,
                cls_model,
                test_x,
                size_target[test_idx],
                position_target[test_idx],
                color_target[test_idx],
                class_target[test_idx],
                args.probe_batch_size,
                device,
            ),
        },
    }

    output_json.write_text(json.dumps(report, indent=2))

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved JSON report to: {output_json}")
    print(
        "Dataset: "
        f"{report['dataset_summary']['num_objects_total']} objects "
        f"(train={report['dataset_summary']['num_train']}, "
        f"val={report['dataset_summary']['num_val']}, "
        f"test={report['dataset_summary']['num_test']})"
    )
    print(f"Class distribution: {report['dataset_summary']['class_distribution']}")
    print("")
    print("Probe results on held-out GT signatures")
    print(
        "Size probe: "
        f"R2={report['results']['size_probe']['test_metrics']['r2'][0]:.4f}, "
        f"MAE={report['results']['size_probe']['test_metrics']['mae'][0]:.4f}, "
        f"RMSE={report['results']['size_probe']['test_metrics']['rmse'][0]:.4f}"
    )
    print(
        "Position probe: "
        f"R2_x={report['results']['position_probe']['test_metrics']['r2'][0]:.4f}, "
        f"R2_y={report['results']['position_probe']['test_metrics']['r2'][1]:.4f}, "
        f"MAE_x={report['results']['position_probe']['test_metrics']['mae'][0]:.4f}, "
        f"MAE_y={report['results']['position_probe']['test_metrics']['mae'][1]:.4f}"
    )
    print(
        "Class probe: "
        f"accuracy={report['results']['class_probe']['test_metrics']['accuracy']:.4f}"
    )
    print(
        "Color probe: "
        f"R2_r={report['results']['color_probe']['test_metrics']['r2'][0]:.4f}, "
        f"R2_g={report['results']['color_probe']['test_metrics']['r2'][1]:.4f}, "
        f"R2_b={report['results']['color_probe']['test_metrics']['r2'][2]:.4f}, "
        f"MAE_r={report['results']['color_probe']['test_metrics']['mae'][0]:.4f}, "
        f"MAE_g={report['results']['color_probe']['test_metrics']['mae'][1]:.4f}, "
        f"MAE_b={report['results']['color_probe']['test_metrics']['mae'][2]:.4f}"
    )
    if report["results"]["relative_two_object_probes"] is not None:
        relative_results = report["results"]["relative_two_object_probes"]
        print("")
        print("Two-object relative probes")
        print(
            "Relative size probe: "
            f"R2={relative_results['relative_size_probe']['test_metrics']['r2'][0]:.4f}, "
            f"MAE={relative_results['relative_size_probe']['test_metrics']['mae'][0]:.4f}, "
            f"RMSE={relative_results['relative_size_probe']['test_metrics']['rmse'][0]:.4f}"
        )
        print(
            "Relative position probe: "
            f"R2_dx={relative_results['relative_position_probe']['test_metrics']['r2'][0]:.4f}, "
            f"R2_dy={relative_results['relative_position_probe']['test_metrics']['r2'][1]:.4f}, "
            f"MAE_dx={relative_results['relative_position_probe']['test_metrics']['mae'][0]:.4f}, "
            f"MAE_dy={relative_results['relative_position_probe']['test_metrics']['mae'][1]:.4f}"
        )
    print("")
    print("Results by number of objects in the image")
    for count_key, metrics in report["results"]["by_num_objects_in_image"].items():
        size_metrics = metrics["size_probe"]
        position_metrics = metrics["position_probe"]
        color_metrics = metrics["color_probe"]
        class_metrics = metrics["class_probe"]
        print(
            f"{count_key} objects | n={metrics['num_test_samples']} | "
            f"size_r2={size_metrics['r2'][0]:.4f} | "
            f"pos_r2_x={position_metrics['r2'][0]:.4f} | "
            f"pos_r2_y={position_metrics['r2'][1]:.4f} | "
            f"color_r2_r={color_metrics['r2'][0]:.4f} | "
            f"color_r2_g={color_metrics['r2'][1]:.4f} | "
            f"color_r2_b={color_metrics['r2'][2]:.4f} | "
            f"class_acc={class_metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
