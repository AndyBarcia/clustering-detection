import argparse

from src.panoptic import load_system_checkpoint
from src.visualization import (
    DEFAULT_CLASS_NAMES,
    run_predictions,
    run_predictions_with_gt_prototypes,
    sample_synthetic_examples,
    save_prediction_grid,
    show_interactive_prediction_grid,
    show_prediction_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize checkpoint predictions on synthetic samples.")
    parser.add_argument("checkpoint", help="Path to a saved checkpoint.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda.")
    parser.add_argument("--num-samples", type=int, default=4, help="Number of synthetic samples to visualize.")
    parser.add_argument("--dataset-length", type=int, default=100, help="Synthetic dataset length used for random sampling.")
    parser.add_argument("--height", type=int, default=256, help="Synthetic image height.")
    parser.add_argument("--width", type=int, default=256, help="Synthetic image width.")
    parser.add_argument("--max-objects", type=int, default=10, help="Maximum objects per synthetic image.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for synthetic sample selection.")
    parser.add_argument("--save-path", default=None, help="Optional PNG path to save the rendered figure.")
    parser.add_argument("--static", action="store_true", help="Open the original static prediction grid instead of the interactive viewer.")
    parser.add_argument("--no-show", action="store_true", help="Save/render without opening a window.")
    return parser.parse_args()


def main():
    args = parse_args()
    system, _ = load_system_checkpoint(args.checkpoint, map_location=args.device)
    system = system.to(args.device)

    images, targets = sample_synthetic_examples(
        num_samples=args.num_samples,
        dataset_length=args.dataset_length,
        height=args.height,
        width=args.width,
        max_objects=args.max_objects,
        seed=args.seed,
    )

    figure_title = f"Checkpoint preview: {args.checkpoint}"
    if args.save_path or args.no_show or args.static:
        predictions = run_predictions(system, images)
        gt_proto_predictions = run_predictions_with_gt_prototypes(system, images, targets)

    if args.save_path:
        save_prediction_grid(
            args.save_path,
            images,
            targets,
            predictions,
            gt_proto_predictions=gt_proto_predictions,
            class_names=DEFAULT_CLASS_NAMES,
            figure_title=figure_title,
        )

    if not args.no_show:
        if args.static:
            show_prediction_grid(
                images,
                targets,
                predictions,
                gt_proto_predictions=gt_proto_predictions,
                class_names=DEFAULT_CLASS_NAMES,
                figure_title=figure_title,
                window_title="Synthetic Prediction Viewer",
            )
        else:
            sample_state = {"seed": args.seed}

            def sample_callback():
                sample_state["seed"] += 1
                return sample_synthetic_examples(
                    num_samples=args.num_samples,
                    dataset_length=args.dataset_length,
                    height=args.height,
                    width=args.width,
                    max_objects=args.max_objects,
                    seed=sample_state["seed"],
                )

            show_interactive_prediction_grid(
                system,
                images,
                targets,
                class_names=DEFAULT_CLASS_NAMES,
                figure_title=figure_title,
                window_title="Interactive Synthetic Prediction Viewer",
                device=args.device,
                sample_callback=sample_callback,
            )


if __name__ == "__main__":
    main()
