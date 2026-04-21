import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from src.panoptic import load_system_checkpoint
from src.visualization import (
    DEFAULT_CLASS_NAMES,
    collect_detailed_prediction_bundle,
    sample_synthetic_examples,
    show_detailed_prediction_view,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect one checkpoint prediction in detail.")
    parser.add_argument("checkpoint", help="Path to a saved checkpoint.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index of the synthetic sample to inspect within the generated batch.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of synthetic samples to generate before choosing --sample-index.",
    )
    parser.add_argument("--height", type=int, default=256, help="Synthetic image height.")
    parser.add_argument("--width", type=int, default=256, help="Synthetic image width.")
    parser.add_argument("--max-objects", type=int, default=10, help="Maximum objects per synthetic image.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic sample selection.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sample_index < 0 or args.sample_index >= args.num_samples:
        raise ValueError("--sample-index must be in [0, --num-samples).")

    system, _ = load_system_checkpoint(args.checkpoint, map_location=args.device)
    system = system.to(args.device)

    images, targets = sample_synthetic_examples(
        num_samples=args.num_samples,
        height=args.height,
        width=args.width,
        max_objects=args.max_objects,
        seed=args.seed,
    )
    image = images[args.sample_index]
    target = targets[args.sample_index]

    bundle = collect_detailed_prediction_bundle(system, image, target)
    show_detailed_prediction_view(
        bundle,
        class_names=DEFAULT_CLASS_NAMES,
        figure_title=f"Detailed prediction view: {args.checkpoint} [sample {args.sample_index}]",
        window_title="Detailed Prediction Viewer",
    )


if __name__ == "__main__":
    main()
