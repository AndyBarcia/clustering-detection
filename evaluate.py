import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import torch

from src.config import dataclass_from_dict, PrototypeInferenceConfig
from src.evaluation import evaluate_system, format_metrics_table
from src.panoptic import load_system_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a synthetic panoptic dataset using Hungarian matching."
    )
    parser.add_argument("checkpoint", help="Path to a saved checkpoint.")
    parser.add_argument("--device", default="cpu", help="Evaluation device, e.g. cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size used for inference.")
    parser.add_argument("--dataset-length", type=int, default=100, help="Number of synthetic samples to evaluate.")
    parser.add_argument("--height", type=int, default=256, help="Synthetic image height.")
    parser.add_argument("--width", type=int, default=256, help="Synthetic image width.")
    parser.add_argument("--max-objects", type=int, default=10, help="Maximum objects per synthetic image.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset generation.")
    parser.add_argument("--ap-threshold", type=float, default=0.5, help="IoU threshold used for AP.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path where the aggregate metrics JSON will be written.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    set_seed(args.seed)

    system, _ = load_system_checkpoint(args.checkpoint, map_location=args.device)
    system = system.to(args.device)
    overall, by_count = evaluate_system(
        system,
        dataset_length=args.dataset_length,
        height=args.height,
        width=args.width,
        max_objects=args.max_objects,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        ap_iou_threshold=args.ap_threshold,
    )

    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Evaluation samples: {args.dataset_length}")
    print(f"AP IoU threshold: {args.ap_threshold:.2f}")
    print()
    print(format_metrics_table(overall, by_count, ap_threshold=args.ap_threshold))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "dataset_length": args.dataset_length,
            "height": args.height,
            "width": args.width,
            "max_objects": args.max_objects,
            "seed": args.seed,
            "ap_threshold": args.ap_threshold,
            "overall": overall,
            "by_object_count": by_count,
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print()
        print(f"Saved metrics JSON to {output_path.resolve()}")


if __name__ == "__main__":
    main()
