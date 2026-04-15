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


METRICS = [
    "dot",
    "dot-sigmoid",
    "cosine",
    "centered-cosine",
    "softmax",
    "jsd",
    "jaccard",
    "dice",
    "overlap",
    "left-overlap",
    "right-overlap",
    "l2",
    "mse"
]
TEMPERATURE_METRICS = {
    "dot-sigmoid",
    "softmax",
    "jsd",
    "jaccard",
    "dice",
    "overlap",
    "left-overlap",
    "right-overlap",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot similarity metric distributions at random initialization, either directly "
            "in signature space or in identity space after inducing aggregation patterns."
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
        default=[0.01, 0.1, 1.0],
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
        "--plot-space",
        choices=["aggregation", "identity"],
        default="aggregation",
        help=(
            "Which initialization quantity to plot. "
            "'aggregation' plots direct signature-space aggregation similarities. "
            "'identity' first computes aggregation patterns using --aggregation-metric "
            "and then plots identity similarities between those patterns."
        ),
    )
    parser.add_argument(
        "--aggregation-metric",
        choices=METRICS,
        default="cosine",
        help=(
            "Aggregation metric used to induce initial aggregation patterns when "
            "--plot-space=identity."
        ),
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


def _pairwise_metric(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    metric: str,
    temp: float | None = None,
) -> torch.Tensor:
    kwargs = {"metric": metric}
    if temp is not None:
        kwargs["temp"] = temp
    return pairwise_similarity(lhs, rhs, **kwargs)


def compute_initial_aggregation_patterns(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    aggregation_metric: str,
    temp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lhs_patterns = _pairwise_metric(lhs, rhs, metric=aggregation_metric, temp=temp)
    rhs_patterns = _pairwise_metric(rhs, rhs, metric=aggregation_metric, temp=temp)
    return lhs_patterns, rhs_patterns


def collect_metric_distributions(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    temps: List[float],
    plot_space: str,
    aggregation_metric: str,
) -> Dict[str, Dict[str, torch.Tensor]]:
    distributions: Dict[str, Dict[str, torch.Tensor]] = {}

    for metric in METRICS:
        metric_runs: Dict[str, torch.Tensor] = {}
        if metric in TEMPERATURE_METRICS:
            for temp in temps:
                if plot_space == "aggregation":
                    values = _pairwise_metric(lhs, rhs, metric=metric, temp=temp)
                else:
                    lhs_patterns, rhs_patterns = compute_initial_aggregation_patterns(
                        lhs,
                        rhs,
                        aggregation_metric=aggregation_metric,
                        temp=temp if aggregation_metric in TEMPERATURE_METRICS else None,
                    )
                    values = _pairwise_metric(lhs_patterns, rhs_patterns, metric=metric, temp=temp)
                metric_runs[f"T={temp:g}"] = flatten_similarity(values)
        else:
            if plot_space == "aggregation":
                values = _pairwise_metric(lhs, rhs, metric=metric)
            else:
                lhs_patterns, rhs_patterns = compute_initial_aggregation_patterns(
                    lhs,
                    rhs,
                    aggregation_metric=aggregation_metric,
                )
                values = _pairwise_metric(lhs_patterns, rhs_patterns, metric=metric)
            metric_runs["default"] = flatten_similarity(values)
        distributions[metric] = metric_runs

    return distributions


def plot_metric_distribution(
    ax: plt.Axes,
    metric: str,
    runs: Dict[str, torch.Tensor],
    bins: int,
    sig_dim: int,
    plot_space: str,
    aggregation_metric: str,
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
    plot_space: str,
    aggregation_metric: str,
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
            plot_space=plot_space,
            aggregation_metric=aggregation_metric,
        )

    if plot_space == "identity":
        title = (
            "Identity similarity distributions at random initialization "
            f"(aggregation patterns induced with {aggregation_metric})"
        )
        filename = f"identity_metric_distributions_from_{aggregation_metric}.png"
    else:
        title = "Aggregation similarity metric distributions at random initialization"
        filename = "signature_metric_distributions.png"

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    output_path = output_dir / filename
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
    distributions = collect_metric_distributions(
        lhs,
        rhs,
        args.temps,
        plot_space=args.plot_space,
        aggregation_metric=args.aggregation_metric,
    )

    output_path = save_combined_metric_figure(
        distributions=distributions,
        output_dir=output_dir,
        bins=args.bins,
        sig_dim=args.sig_dim,
        show=not args.no_show,
        plot_space=args.plot_space,
        aggregation_metric=args.aggregation_metric,
    )

    print(f"Saved combined plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
