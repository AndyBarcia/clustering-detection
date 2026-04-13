import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["dbscan", "hdbscan", "cc", "louvain", "leiden"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create summary plots from an Optuna tuning JSON export."
    )
    parser.add_argument("tuning_json", help="Path to the tuning-optuna JSON file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where plots will be saved. Defaults to '<json_stem>_plots'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of top numeric parameters to plot against the objective.",
    )
    return parser.parse_args()


def load_trials(path: Path) -> Tuple[Dict, List[Dict]]:
    payload = json.loads(path.read_text())
    trials = [
        trial for trial in payload.get("trials", [])
        if trial.get("state") == "COMPLETE" and trial.get("value") is not None
    ]
    if not trials:
        raise ValueError(f"No completed trials found in {path}.")
    return payload, trials


def group_trials_by_method(trials: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for trial in trials:
        method = trial.get("params", {}).get("cluster.method", "unknown")
        grouped.setdefault(method, []).append(trial)
    return grouped


def collect_numeric_param_names(trials: List[Dict]) -> List[str]:
    names = set()
    for trial in trials:
        params = trial.get("params", {})
        for name, value in params.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                names.add(name)
    return sorted(names)


def best_so_far(values: List[float]) -> List[float]:
    best = []
    current = -float("inf")
    for value in values:
        current = max(current, value)
        best.append(current)
    return best


def save_optimization_history(trials: List[Dict], metric_name: str, output_path: Path):
    values = [float(trial["value"]) for trial in trials]
    numbers = [int(trial["number"]) for trial in trials]
    running_best = best_so_far(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(numbers, values, marker="o", linewidth=1.2, alpha=0.75, label=f"trial {metric_name}")
    ax.plot(numbers, running_best, linewidth=2.5, color="tab:red", label=f"best-so-far {metric_name}")
    ax.set_title("Optimization History")
    ax.set_xlabel("Trial")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_method_summary(trials: List[Dict], metric_name: str, output_path: Path):
    grouped: Dict[str, List[float]] = {}
    for trial in trials:
        method = trial.get("params", {}).get("cluster.method", "unknown")
        grouped.setdefault(method, []).append(float(trial["value"]))

    methods = [name for name in METHOD_ORDER if name in grouped] + sorted(
        name for name in grouped if name not in METHOD_ORDER
    )
    means = [float(np.mean(grouped[name])) for name in methods]
    bests = [float(np.max(grouped[name])) for name in methods]
    counts = [len(grouped[name]) for name in methods]

    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, means, width=width, label=f"mean {metric_name}", color="#5B8FF9")
    ax.bar(x + width / 2, bests, width=width, label=f"best {metric_name}", color="#61DDAA")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20)
    ax.set_title("Clustering Method Performance")
    ax.set_xlabel("Method")
    ax.set_ylabel(metric_name)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    for idx, count in enumerate(counts):
        ax.text(x[idx], max(means[idx], bests[idx]), f"n={count}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_param_scores(trials: List[Dict], numeric_names: List[str]) -> List[Tuple[str, float]]:
    y = np.asarray([float(trial["value"]) for trial in trials], dtype=np.float64)
    scores: List[Tuple[str, float]] = []

    for name in numeric_names:
        xs = []
        ys = []
        for trial in trials:
            value = trial.get("params", {}).get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                xs.append(float(value))
                ys.append(float(trial["value"]))

        if len(xs) < 3:
            continue

        x = np.asarray(xs, dtype=np.float64)
        y_local = np.asarray(ys, dtype=np.float64)
        if np.allclose(x, x[0]):
            continue
        corr = np.corrcoef(x, y_local)[0, 1]
        if np.isnan(corr):
            continue
        scores.append((name, abs(float(corr))))

    return sorted(scores, key=lambda item: item[1], reverse=True)


def save_param_scatter_grid(
    trials: List[Dict],
    metric_name: str,
    numeric_names: List[str],
    top_k: int,
    output_path: Path,
):
    top_params = [name for name, _ in compute_param_scores(trials, numeric_names)]
    top_params = top_params[: max(1, min(len(top_params), top_k))]
    if not top_params:
        return

    cols = 2
    rows = math.ceil(len(top_params) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, name in enumerate(top_params):
        ax = axes.flat[idx]
        ax.set_visible(True)

        xs = []
        ys = []
        for trial in trials:
            value = trial.get("params", {}).get(name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            xs.append(float(value))
            ys.append(float(trial["value"]))
        ax.scatter(xs, ys, alpha=0.75, s=28, color="#5B8FF9")
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Most Correlated Numeric Parameters")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_correlation_heatmap(
    trials: List[Dict],
    metric_name: str,
    numeric_names: List[str],
    output_path: Path,
):
    usable = []
    series = []
    objective = []

    for name in numeric_names:
        values = []
        missing = 0
        for trial in trials:
            value = trial.get("params", {}).get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
            else:
                values.append(np.nan)
                missing += 1
        if missing > 0.4 * len(trials):
            continue
        usable.append(name)
        series.append(values)

    if not usable:
        return

    objective = [float(trial["value"]) for trial in trials]
    matrix_names = usable + [metric_name]
    matrix_data = series + [objective]
    arr = np.asarray(matrix_data, dtype=np.float64)
    corr = np.corrcoef(np.nan_to_num(arr, nan=np.nanmean(arr, axis=1, keepdims=True)))

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(matrix_names)), max(5, 0.5 * len(matrix_names))))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(matrix_names)))
    ax.set_xticklabels(matrix_names, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(matrix_names)))
    ax.set_yticklabels(matrix_names)
    ax.set_title("Parameter Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_top_trials_table(trials: List[Dict], metric_name: str, output_path: Path, top_n: int = 15):
    top_trials = sorted(trials, key=lambda trial: float(trial["value"]), reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(14, 0.55 * (top_n + 2)))
    ax.axis("off")

    rows = []
    for trial in top_trials:
        params = trial.get("params", {})
        rows.append([
            str(trial["number"]),
            f"{float(trial['value']):.4f}",
            str(params.get("cluster.method", "")),
            f"{params.get('seed.max_distance', '')}",
            f"{params.get('overlap.pixel_score_threshold', '')}",
            f"{params.get('overlap.min_prototype_score', '')}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=["trial", metric_name, "method", "seed.max_d", "pixel.th", "proto.score"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Top Trials", pad=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_method_specific_plots(
    grouped_trials: Dict[str, List[Dict]],
    metric_name: str,
    top_k: int,
    output_dir: Path,
):
    methods = [name for name in METHOD_ORDER if name in grouped_trials] + sorted(
        name for name in grouped_trials if name not in METHOD_ORDER
    )

    for method in methods:
        trials = grouped_trials[method]
        method_dir = output_dir / f"method_{method}"
        method_dir.mkdir(parents=True, exist_ok=True)
        numeric_names = collect_numeric_param_names(trials)

        save_optimization_history(trials, metric_name, method_dir / "optimization_history.png")
        save_top_trials_table(trials, metric_name, method_dir / "top_trials_table.png")
        save_param_scatter_grid(
            trials,
            metric_name,
            numeric_names,
            top_k,
            method_dir / "parameter_scatter_grid.png",
        )
        save_correlation_heatmap(
            trials,
            metric_name,
            numeric_names,
            method_dir / "parameter_correlation_heatmap.png",
        )


def main():
    args = parse_args()
    tuning_path = Path(args.tuning_json)
    payload, trials = load_trials(tuning_path)
    metric_name = payload.get("metric", "objective")
    output_dir = Path(args.output_dir) if args.output_dir else tuning_path.with_name(f"{tuning_path.stem}_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_names = collect_numeric_param_names(trials)
    grouped_trials = group_trials_by_method(trials)

    save_optimization_history(trials, metric_name, output_dir / "optimization_history.png")
    save_method_summary(trials, metric_name, output_dir / "method_summary.png")
    save_param_scatter_grid(
        trials,
        metric_name,
        numeric_names[:],
        args.top_k,
        output_dir / "parameter_scatter_grid.png",
    )
    save_correlation_heatmap(trials, metric_name, numeric_names[:], output_dir / "parameter_correlation_heatmap.png")
    save_top_trials_table(trials, metric_name, output_dir / "top_trials_table.png")
    save_method_specific_plots(grouped_trials, metric_name, args.top_k, output_dir)

    print(f"Saved plots to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
