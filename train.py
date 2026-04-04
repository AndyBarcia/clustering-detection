import argparse
import json
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import torch
from torch.utils.data import DataLoader

from src.config import PanopticSystemConfig
from src.dataset import SyntheticPanopticBatchGenerator, BatchedSyntheticIterableDataset
from src.evaluation import evaluate_system, format_metrics_table
from src.panoptic import PanopticSystem, load_system_checkpoint, save_system_checkpoint
from src.visualization import (
    DEFAULT_CLASS_NAMES,
    run_predictions,
    run_predictions_with_gt_prototypes,
    sample_synthetic_examples,
    save_prediction_grid,
)


CHECKPOINT_NAME = "checkpoint.pt"
METRICS_NAME = "training_losses.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the panoptic system and report disaggregated losses.")
    parser.add_argument("--epochs", type=int, default=10, help="Total number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--dataset-length", type=int, default=100, help="Number of synthetic samples.")
    parser.add_argument("--height", type=int, default=256, help="Synthetic image height.")
    parser.add_argument("--width", type=int, default=256, help="Synthetic image width.")
    parser.add_argument("--max-objects", type=int, default=10, help="Maximum objects per image.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Report running mean losses every N training iterations. Set to 0 to disable intra-epoch logging.",
    )
    parser.add_argument("--device", default=None, help="Training device, e.g. cpu or cuda.")
    parser.add_argument("--vis-samples", type=int, default=4, help="Number of synthetic samples rendered after each epoch.")
    parser.add_argument("--vis-seed", type=int, default=0, help="Random seed for the fixed visualization batch.")
    parser.add_argument("--eval-dataset-length", type=int, default=32, help="Number of synthetic samples used for epoch evaluation.")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Batch size used for epoch evaluation. Defaults to training batch size.")
    parser.add_argument("--eval-seed", type=int, default=123, help="Base random seed used for synthetic evaluation.")
    parser.add_argument("--eval-ap-threshold", type=float, default=0.5, help="IoU threshold used for AP during evaluation.")
    parser.add_argument("--use-gt-prototypes-for-eval", action="store_true", help="Evaluate the GT-prototype decoding path instead of clustered predictions.")
    parser.add_argument(
        "--skip-epoch-vis",
        action="store_true",
        help="Disable prediction PNG export during training.",
    )
    parser.add_argument(
        "--skip-epoch-eval",
        action="store_true",
        help="Disable the synthetic evaluation pass after each epoch.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where checkpoints and JSON metrics will be written.",
    )
    return parser.parse_args()


def resolve_device(requested_device: Optional[str]) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader(args, device):
    generator = SyntheticPanopticBatchGenerator(
        height=args.height,
        width=args.width,
        max_objects=args.max_objects,
        device=device,
    )
    dataset = BatchedSyntheticIterableDataset(
        generator=generator,
        total_samples=args.dataset_length,
        batch_size=args.batch_size,
        drop_last=getattr(args, "drop_last", False),
    )

    def identity_collate(sample):
        # With batch_size=None, collate_fn receives a single yielded item.
        return sample

    loader = DataLoader(
        dataset,
        batch_size=None,      # dataset already yields full batches
        num_workers=0,        # important for CUDA-generated batches
        pin_memory=False,     # data is already on GPU
        collate_fn=identity_collate,
    )
    return loader


def format_epoch_metrics(epoch_metrics):
    ordered_keys = [
        "loss_total",
        "loss_proto_sig",
        "loss_proto_ttt",
        "loss_cls",
        "loss_mask_bce",
        "loss_mask_iou",
        "loss_mask_total",
        "loss_sim",
        "loss_margin",
        "loss_inter",
        "loss_seed_cover",
        "loss_seed_sparse",
        "loss_graph_pos",
        "loss_graph_neg",
    ]
    return " | ".join(f"{key}={epoch_metrics[key]:.4f}" for key in ordered_keys if key in epoch_metrics)


def format_iteration_metrics(metrics):
    return format_epoch_metrics(metrics)


def load_training_state(output_dir: Path, device: torch.device, lr: float, weight_decay: float):
    checkpoint_path = output_dir / CHECKPOINT_NAME
    if checkpoint_path.exists():
        system, ckpt = load_system_checkpoint(checkpoint_path, map_location=device)
        system = system.to(device)
        optimizer = torch.optim.Adam(system.parameters(), lr=lr, weight_decay=weight_decay)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        extra = ckpt.get("extra", {})
        history = extra.get("history", [])
        start_epoch = int(extra.get("epoch", 0))
        print(f"Resuming from {checkpoint_path.resolve()} at epoch {start_epoch}.", flush=True)
        return system, optimizer, history, start_epoch

    cfg = PanopticSystemConfig()
    system = PanopticSystem(cfg).to(device)
    optimizer = torch.optim.Adam(system.parameters(), lr=lr, weight_decay=weight_decay)
    return system, optimizer, [], 0


def save_training_state(output_dir: Path, system: PanopticSystem, optimizer, history, epoch: int):
    checkpoint_path = output_dir / CHECKPOINT_NAME
    metrics_path = output_dir / METRICS_NAME

    save_system_checkpoint(
        system,
        str(checkpoint_path),
        optimizer=optimizer,
        extra={
            "epoch": epoch,
            "history": history,
        },
    )
    metrics_path.write_text(json.dumps(history, indent=2))


def build_visualization_batch(args):
    if args.vis_samples <= 0:
        return [], []

    return sample_synthetic_examples(
        num_samples=args.vis_samples,
        dataset_length=args.dataset_length,
        height=args.height,
        width=args.width,
        max_objects=args.max_objects,
        seed=args.vis_seed,
        device="cpu",
    )


def save_epoch_visualization(output_dir: Path, system: PanopticSystem, images, targets, epoch: int):
    if len(images) == 0:
        return None

    predictions = run_predictions(system, images)
    gt_proto_predictions = run_predictions_with_gt_prototypes(system, images, targets)
    path = output_dir / f"predictions_epoch_{epoch:03d}.png"
    save_prediction_grid(
        path,
        images,
        targets,
        predictions,
        gt_proto_predictions=gt_proto_predictions,
        class_names=DEFAULT_CLASS_NAMES,
        figure_title=f"Epoch {epoch} predictions",
    )
    return path


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    system, optimizer, history, start_epoch = load_training_state(
        output_dir=output_dir,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    data_loader = build_dataloader(args, device)
    vis_images, vis_targets = build_visualization_batch(args)

    for epoch_idx in range(start_epoch, args.epochs):
        system.train()
        epoch_sums = {}
        interval_sums = {}
        num_batches = 0
        for images, targets in data_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, components = system.training_step(images, targets)
            loss.backward()
            optimizer.step()

            num_batches += 1
            for name, value in components.items():
                scalar = float(value.detach().item())
                epoch_sums[name] = epoch_sums.get(name, 0.0) + scalar
                interval_sums[name] = interval_sums.get(name, 0.0) + scalar

            if args.log_every > 0 and num_batches % args.log_every == 0:
                interval_metrics = {
                    name: value / args.log_every for name, value in interval_sums.items()
                }
                print(
                    f"Epoch {epoch_idx + 1}/{args.epochs} | Iteration {num_batches}: "
                    f"{format_iteration_metrics(interval_metrics)}",
                    flush=True,
                )
                interval_sums = {}

        epoch_metrics = {
            "epoch": epoch_idx + 1,
            "num_batches": num_batches,
        }
        for name, value in epoch_sums.items():
            epoch_metrics[name] = value / max(num_batches, 1)

        history.append(epoch_metrics)
        print(f"Epoch {epoch_idx + 1}/{args.epochs}: {format_epoch_metrics(epoch_metrics)}", flush=True)

        if not args.skip_epoch_eval:
            eval_batch_size = args.eval_batch_size or args.batch_size
            overall_eval, per_count_eval = evaluate_system(
                system,
                dataset_length=args.eval_dataset_length,
                height=args.height,
                width=args.width,
                max_objects=args.max_objects,
                batch_size=eval_batch_size,
                device=device,
                seed=args.eval_seed + epoch_idx,
                ap_iou_threshold=args.eval_ap_threshold,
                use_gt_prototypes=args.use_gt_prototypes_for_eval,
            )
            epoch_metrics["evaluation"] = {
                "overall": overall_eval,
                "by_object_count": per_count_eval,
                "ap_threshold": args.eval_ap_threshold,
                "dataset_length": args.eval_dataset_length,
                "seed": args.eval_seed + epoch_idx,
                "use_gt_prototypes": args.use_gt_prototypes_for_eval,
            }
            print("Evaluation", flush=True)
            print(format_metrics_table(overall_eval, per_count_eval, ap_threshold=args.eval_ap_threshold), flush=True)

        save_training_state(output_dir, system, optimizer, history, epoch_idx + 1)
        if not args.skip_epoch_vis:
            vis_path = save_epoch_visualization(output_dir, system, vis_images, vis_targets, epoch_idx + 1)
            if vis_path is not None:
                print(f"Saved prediction snapshot to {vis_path.resolve()}", flush=True)

    print(f"Saved checkpoint to {(output_dir / CHECKPOINT_NAME).resolve()}", flush=True)
    print(f"Saved epoch metrics to {(output_dir / METRICS_NAME).resolve()}", flush=True)


if __name__ == "__main__":
    main()
