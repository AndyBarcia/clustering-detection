import argparse
import copy
import hashlib
import itertools
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from src.config import PrototypeInferenceConfig, dataclass_from_dict
from src.evaluation import evaluate_system
from src.panoptic import load_system_checkpoint


DEFAULT_SEARCH_SPACE = {
    "seed.quality_threshold": [0.05, 0.1, 0.15],
    "cluster.method": ["dbscan", "hdbscan", "cc", "louvain", "leiden"],
    "cluster.dbscan_eps": [0.1, 0.15, 0.2],
    "cluster.graph_affinity_threshold": [0.65, 0.75, 0.85],
    "assign.refinement_steps": [1, 2],
    "assign.class_compat_power": [0.0, 0.5],
    "overlap.min_prototype_score": [0.03, 0.05],
    "overlap.pixel_score_threshold": [0.2, 0.25],
}

METRIC_ALIASES = {
    "ap": ("overall", "ap"),
    "mean_iou": ("overall", "mean_iou"),
    "precision": ("overall", "precision"),
    "recall": ("overall", "recall"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search inference hyperparameters for clustered panoptic predictions, "
            "persisting incremental results so the search can be resumed."
        )
    )
    parser.add_argument("checkpoint", help="Path to a saved checkpoint.")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path where tuning results will be written as JSON.",
    )
    parser.add_argument("--device", default="cpu", help="Evaluation device, e.g. cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size used for inference.")
    parser.add_argument("--dataset-length", type=int, default=100, help="Number of synthetic samples to evaluate.")
    parser.add_argument("--height", type=int, default=256, help="Synthetic image height.")
    parser.add_argument("--width", type=int, default=256, help="Synthetic image width.")
    parser.add_argument("--max-objects", type=int, default=10, help="Maximum objects per synthetic image.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dataset generation.")
    parser.add_argument("--ap-threshold", type=float, default=0.5, help="IoU threshold used for AP.")
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_ALIASES),
        default="ap",
        help="Metric used to rank hyperparameter combinations.",
    )
    parser.add_argument(
        "--search-space-json",
        default=None,
        help=(
            "Optional JSON file describing a search space. "
            "Format: {\"cluster.method\": [\"dbscan\", \"cc\"], \"overlap.mask_threshold\": [0.4, 0.5]}."
        ),
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional limit on how many trial combinations to execute from the search space.",
    )
    parser.add_argument(
        "--use-gt-prototypes",
        action="store_true",
        help="Evaluate the GT-prototype decoding path instead of clustered predictions.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore any existing output JSON and start a fresh search.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed trials and continue running the remaining combinations.",
    )
    return parser.parse_args()


def load_search_space(path: str | None) -> Dict[str, List[Any]]:
    if path is None:
        return copy.deepcopy(DEFAULT_SEARCH_SPACE)

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Search-space JSON must be an object mapping dot-paths to candidate lists.")

    normalized: Dict[str, List[Any]] = {}
    for key, values in payload.items():
        if not isinstance(key, str):
            raise ValueError("Search-space keys must be strings.")
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Search-space entry '{key}' must be a non-empty list.")
        normalized[key] = values
    return normalized


def normalize_search_space(search_space: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    return {key: list(values) for key, values in sorted(search_space.items())}


def iter_trial_params(search_space: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(search_space)
    value_lists = [search_space[key] for key in keys]
    for combo in itertools.product(*value_lists):
        yield {key: value for key, value in zip(keys, combo)}


def make_trial_id(params: Dict[str, Any]) -> str:
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]


def _split_config_path(path: str) -> List[str]:
    parts = path.split(".")
    if parts and parts[0] == "inference":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Invalid config path '{path}'.")
    return parts


def set_nested_value(obj: Any, path: str, value: Any):
    parts = _split_config_path(path)
    cursor = obj
    for part in parts[:-1]:
        if not hasattr(cursor, part):
            raise AttributeError(f"Unknown config path '{path}'.")
        cursor = getattr(cursor, part)
    leaf = parts[-1]
    if not hasattr(cursor, leaf):
        raise AttributeError(f"Unknown config path '{path}'.")
    setattr(cursor, leaf, value)


def build_inference_config(base_cfg: PrototypeInferenceConfig, params: Dict[str, Any]) -> PrototypeInferenceConfig:
    cfg = dataclass_from_dict(PrototypeInferenceConfig, asdict(base_cfg))
    for path, value in params.items():
        set_nested_value(cfg, path, value)
    return cfg


def read_metric(payload: Dict[str, Any], metric_name: str) -> float:
    scope, key = METRIC_ALIASES[metric_name]
    return float(payload[scope][key])


def load_existing_results(
    output_path: Path,
    *,
    restart: bool,
    metadata_signature: Dict[str, Any],
    search_space: Dict[str, List[Any]],
) -> Dict[str, Any]:
    if restart or not output_path.exists():
        return {
            "metadata": metadata_signature,
            "search_space": search_space,
            "completed_trials": [],
            "failed_trials": [],
            "best_trial_id": None,
            "best_metric": None,
        }

    payload = json.loads(output_path.read_text())
    if payload.get("metadata") != metadata_signature:
        raise ValueError(
            "Existing results JSON does not match the current search setup. "
            "Use --restart or choose a different --output-json."
        )
    if payload.get("search_space") != search_space:
        raise ValueError(
            "Existing results JSON was created with a different search space. "
            "Use --restart or choose a different --output-json."
        )

    payload.setdefault("completed_trials", [])
    payload.setdefault("failed_trials", [])
    payload.setdefault("best_trial_id", None)
    payload.setdefault("best_metric", None)
    return payload


def save_results(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)


def update_best_trial(results: Dict[str, Any], metric_name: str):
    best_trial_id = None
    best_metric = None

    for trial in results["completed_trials"]:
        value = read_metric(trial["metrics"], metric_name)
        if best_metric is None or value > best_metric:
            best_metric = value
            best_trial_id = trial["trial_id"]

    results["best_trial_id"] = best_trial_id
    results["best_metric"] = best_metric


def main():
    args = parse_args()
    output_path = Path(args.output_json)
    search_space = normalize_search_space(load_search_space(args.search_space_json))
    checkpoint_path = Path(args.checkpoint).resolve()

    metadata_signature = {
        "checkpoint": str(checkpoint_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "dataset_length": args.dataset_length,
        "height": args.height,
        "width": args.width,
        "max_objects": args.max_objects,
        "seed": args.seed,
        "ap_threshold": args.ap_threshold,
        "metric": args.metric,
        "use_gt_prototypes": args.use_gt_prototypes,
    }
    results = load_existing_results(
        output_path,
        restart=args.restart,
        metadata_signature=metadata_signature,
        search_space=search_space,
    )

    completed_ids = {trial["trial_id"] for trial in results["completed_trials"]}
    total_trials = 1
    for values in search_space.values():
        total_trials *= len(values)
    if args.max_trials is not None:
        total_trials = min(total_trials, args.max_trials)

    system, ckpt = load_system_checkpoint(str(checkpoint_path), map_location=args.device)
    system = system.to(args.device)
    base_inference_cfg = dataclass_from_dict(PrototypeInferenceConfig, ckpt["inference_config"])

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Search metric: {args.metric}")
    print(f"Output JSON: {output_path.resolve()}")
    print(f"Total planned trials: {total_trials}")
    print(f"Already completed: {len(completed_ids)}")

    for trial_idx, params in enumerate(iter_trial_params(search_space), start=1):
        if args.max_trials is not None and trial_idx > args.max_trials:
            break

        trial_id = make_trial_id(params)
        if trial_id in completed_ids:
            print(f"[{trial_idx}/{total_trials}] Skipping completed trial {trial_id}")
            continue

        trial_cfg = build_inference_config(base_inference_cfg, params)
        system.set_inference_config(trial_cfg)

        print(f"[{trial_idx}/{total_trials}] Running trial {trial_id}: {json.dumps(params, sort_keys=True)}", flush=True)
        started_at = time.time()

        try:
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
                use_gt_prototypes=args.use_gt_prototypes,
            )
        except Exception as exc:
            failure = {
                "trial_id": trial_id,
                "params": params,
                "error": repr(exc),
                "elapsed_sec": time.time() - started_at,
            }
            results["failed_trials"].append(failure)
            save_results(output_path, results)
            print(f"[{trial_idx}/{total_trials}] Trial {trial_id} failed: {exc}", flush=True)
            if not args.continue_on_error:
                raise
            continue

        trial_result = {
            "trial_id": trial_id,
            "params": params,
            "elapsed_sec": time.time() - started_at,
            "metrics": {
                "overall": overall,
                "by_object_count": by_count,
            },
        }
        results["completed_trials"].append(trial_result)
        completed_ids.add(trial_id)
        update_best_trial(results, args.metric)
        save_results(output_path, results)

        metric_value = read_metric(trial_result["metrics"], args.metric)
        print(
            f"[{trial_idx}/{total_trials}] Finished trial {trial_id} | "
            f"{args.metric}={metric_value:.4f} | "
            f"best={results['best_metric']:.4f}",
            flush=True,
        )

    update_best_trial(results, args.metric)
    save_results(output_path, results)

    if results["best_trial_id"] is None:
        print("No successful trials were completed.")
        return

    best_trial = next(trial for trial in results["completed_trials"] if trial["trial_id"] == results["best_trial_id"])
    print()
    print(f"Best trial: {results['best_trial_id']}")
    print(f"Best {args.metric}: {results['best_metric']:.4f}")
    print(f"Best params: {json.dumps(best_trial['params'], sort_keys=True)}")


if __name__ == "__main__":
    main()
