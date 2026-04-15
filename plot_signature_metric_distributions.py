import argparse
import os
from pathlib import Path
from typing import Dict, List
import math

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import torch

from src.signature_ops import pairwise_similarity


METRICS = ["dot", "dot-sigmoid", "cosine", "centered-cosine", "softmax", "jsd", "jaccard", "dice", "overlap"]
TEMPERATURE_METRICS = {"dot-sigmoid", "softmax", "jsd", "jaccard", "dice", "overlap"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the distribution of signature similarity metrics at random initialization."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="signature_metric_distributions",
        help="Directory where the metric distribution plots will be saved.",
    )
    parser.add_argument(
        "--num-lhs",
        type=int,
        default=512,
        help="Number of random left-hand signatures to sample.",
    )
    parser.add_argument(
        "--num-rhs",
        type=int,
        default=512,
        help="Number of random right-hand signatures to sample.",
    )
    parser.add_argument(
        "--sig-dim",
        type=int,
        default=64,
        help="Signature dimensionality.",
    )
    parser.add_argument(
        "--temps",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="Temperatures to compare for temperature-aware metrics.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=120,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to sample the signatures.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used for the computation, for example cpu or cuda.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the combined figure without opening a visualization window.",
    )
    return parser.parse_args()


def sample_random_signatures(
    num_lhs: int,
    num_rhs: int,
    sig_dim: int,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    lhs = torch.randn(num_lhs, sig_dim, generator=generator, device=device)
    rhs = torch.randn(num_rhs, sig_dim, generator=generator, device=device)
    return lhs, rhs


def flatten_similarity(values: torch.Tensor) -> torch.Tensor:
    return values.detach().reshape(-1).to(torch.float32).cpu()


def collect_metric_distributions(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    temps: List[float],
) -> Dict[str, Dict[str, torch.Tensor]]:
    distributions: Dict[str, Dict[str, torch.Tensor]] = {}

    for metric in METRICS:
        metric_runs: Dict[str, torch.Tensor] = {}
        if metric in TEMPERATURE_METRICS:
            for temp in temps:
                values = pairwise_similarity(lhs, rhs, metric=metric, temp=temp)
                metric_runs[f"T={temp:g}"] = flatten_similarity(values)
        else:
            values = pairwise_similarity(lhs, rhs, metric=metric)
            metric_runs["default"] = flatten_similarity(values)
        distributions[metric] = metric_runs

    return distributions


def plot_metric_distribution(
    ax: plt.Axes,
    metric: str,
    runs: Dict[str, torch.Tensor],
    bins: int,
    sig_dim: int,
):
    for label, values in runs.items():
        ax.hist(
            values.numpy(),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            alpha=0.95,
            label=label,
        )

    ax.set_title(metric)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 1.0)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()

    stats_lines = []
    for label, values in runs.items():
        mean = values.mean().item()
        std = values.std(unbiased=False).item()
        prefix = label if label != "default" else "default"
        stats_lines.append(f"{prefix}: mean={mean:.4f}, std={std:.4f}")

    stats_text = f"sig_dim={sig_dim}\n" + "\n".join(stats_lines)
    ax.text(
        0.99,
        0.99,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )


def save_combined_metric_figure(
    distributions: Dict[str, Dict[str, torch.Tensor]],
    output_dir: Path,
    bins: int,
    sig_dim: int,
    show: bool,
):
    num_metrics = len(distributions)
    cols = 3
    rows = math.ceil(num_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.8 * rows))
    axes = axes.reshape(rows, cols) if hasattr(axes, "reshape") else [[axes]]

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, (metric, runs) in zip(axes.flat, distributions.items()):
        ax.set_visible(True)
        plot_metric_distribution(
            ax=ax,
            metric=metric,
            runs=runs,
            bins=bins,
            sig_dim=sig_dim,
        )

    fig.suptitle("Signature similarity metric distributions at random initialization", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path = output_dir / "signature_metric_distributions.png"
    fig.savefig(output_path, dpi=180)

    if show:
        plt.show()

    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lhs, rhs = sample_random_signatures(
        num_lhs=args.num_lhs,
        num_rhs=args.num_rhs,
        sig_dim=args.sig_dim,
        device=args.device,
        seed=args.seed,
    )
    distributions = collect_metric_distributions(lhs, rhs, args.temps)

    output_path = save_combined_metric_figure(
        distributions=distributions,
        output_dir=output_dir,
        bins=args.bins,
        sig_dim=args.sig_dim,
        show=not args.no_show,
    )

    print(f"Saved combined plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
