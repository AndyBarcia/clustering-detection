import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import optuna
import torch

from src.config import PrototypeInferenceConfig, dataclass_from_dict
from src.evaluation import evaluate_system_many_configs
from src.panoptic import load_system_checkpoint


DEFAULT_SEARCH_SPACE = {
    "seed.quality_threshold": {"type": "float", "low": 0.03, "high": 0.2},
    "seed.min_foreground_prob": {"type": "float", "low": 0.0, "high": 0.25},
    "cluster.method": {"type": "categorical", "choices": ["dbscan", "hdbscan", "cc", "louvain", "leiden"]},
    "cluster.dbscan_eps": {"type": "float", "low": 0.05, "high": 0.3},
    "cluster.hdbscan_min_cluster_size": {"type": "int", "low": 2, "high": 8},
    "cluster.hdbscan_min_samples": {"type": "int", "low": 1, "high": 8},
    "cluster.hdbscan_cluster_selection_epsilon": {"type": "float", "low": 0.0, "high": 0.2},
    "cluster.graph_affinity_threshold": {"type": "float", "low": 0.5, "high": 0.9},
    "cluster.graph_min_edge_weight": {"type": "float", "low": 0.001, "high": 0.1, "log": True},
    "cluster.louvain_resolution": {"type": "float", "low": 0.25, "high": 2.0},
    "cluster.leiden_resolution": {"type": "float", "low": 0.25, "high": 2.0},
    "assign.refinement_steps": {"type": "int", "low": 1, "high": 3},
    "assign.class_compat_power": {"type": "float", "low": 0.0, "high": 1.0},
    "assign.query_quality_power": {"type": "float", "low": 0.5, "high": 2.0},
    "overlap.min_prototype_score": {"type": "float", "low": 0.01, "high": 0.1},
    "overlap.pixel_score_threshold": {"type": "float", "low": 0.15, "high": 0.35},
    "overlap.mask_threshold": {"type": "float", "low": 0.3, "high": 0.7},
}

BEST_KNOWN_PARAMS = {
    "assign.class_compat_power": 0.0,
    "assign.refinement_steps": 1,
    "assign.query_quality_power": 0.6,
    "cluster.graph_affinity_threshold": 0.76,
    "cluster.method": "cc",
    "overlap.mask_threshold": 0.42,
    "overlap.min_prototype_score": 0.04,
    "overlap.pixel_score_threshold": 0.28,
    "seed.min_foreground_prob": 0.22,
    "seed.quality_threshold": 0.17,
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
            "Optimize clustered panoptic inference hyperparameters with Optuna. "
            "The study can be resumed via persistent storage."
        )
    )
    parser.add_argument("checkpoint", help="Path to a saved checkpoint.")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path where a JSON summary of the Optuna study will be written.",
    )
    parser.add_argument(
        "--storage",
        default=None,
        help=(
            "Optuna storage URL. Defaults to a SQLite DB next to --output-json, "
            "for example sqlite:///outputs/tuning.db."
        ),
    )
    parser.add_argument(
        "--study-name",
        default="cluster_hparam_tuning",
        help="Optuna study name used for resume/load.",
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
        help="Metric used as the Optuna objective.",
    )
    parser.add_argument(
        "--search-space-json",
        default=None,
        help=(
            "Optional JSON file describing the search space. "
            "Each value must be a distribution spec like "
            "{\"type\": \"float\", \"low\": 0.05, \"high\": 0.3} or "
            "{\"type\": \"categorical\", \"choices\": [\"dbscan\", \"cc\"]}."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials to run in this invocation.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional time budget in seconds for this invocation.",
    )
    parser.add_argument(
        "--startup-trials",
        type=int,
        default=10,
        help="Number of startup trials for the TPE sampler.",
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        help=(
            "Number of Optuna trials to evaluate together per round. "
            "Trials in the same round share each batch's model forward pass."
        ),
    )
    parser.add_argument(
        "--use-gt-prototypes",
        action="store_true",
        help="Evaluate the GT-prototype decoding path instead of clustered predictions.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Mark failed trials and continue instead of aborting the study run.",
    )
    return parser.parse_args()


def load_search_space(path: str | None) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return json.loads(json.dumps(DEFAULT_SEARCH_SPACE))

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Search-space JSON must be an object mapping dot-paths to distribution specs.")

    normalized: Dict[str, Dict[str, Any]] = {}
    for key, spec in payload.items():
        if not isinstance(key, str):
            raise ValueError("Search-space keys must be strings.")
        if not isinstance(spec, dict):
            raise ValueError(f"Search-space entry '{key}' must be an object.")
        normalized[key] = spec
    return {key: normalized[key] for key in sorted(normalized)}


def _split_config_path(path: str):
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


def read_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    scope, key = METRIC_ALIASES[metric_name]
    return float(metrics[scope][key])


def suggest_value(trial: optuna.Trial, name: str, spec: Dict[str, Any]):
    spec_type = spec.get("type")
    if spec_type == "float":
        kwargs = {
            "low": spec["low"],
            "high": spec["high"],
            "log": bool(spec.get("log", False)),
        }
        if "step" in spec:
            kwargs["step"] = spec["step"]
        return trial.suggest_float(name, **kwargs)
    if spec_type == "int":
        kwargs = {
            "low": spec["low"],
            "high": spec["high"],
            "log": bool(spec.get("log", False)),
        }
        if "step" in spec:
            kwargs["step"] = spec["step"]
        return trial.suggest_int(name, **kwargs)
    if spec_type == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unsupported distribution type '{spec_type}' for '{name}'.")


def sample_params(trial: optuna.Trial, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    if "cluster.method" in search_space:
        params["cluster.method"] = suggest_value(trial, "cluster.method", search_space["cluster.method"])

    for path, spec in search_space.items():
        if path == "cluster.method":
            continue
        method = params.get("cluster.method")
        if path.startswith("cluster.dbscan_") and method != "dbscan":
            continue
        if path.startswith("cluster.hdbscan_") and method != "hdbscan":
            continue
        if path == "cluster.graph_affinity_threshold" and method not in {"cc"}:
            continue
        if path == "cluster.graph_min_edge_weight" and method not in {"louvain", "leiden"}:
            continue
        if path.startswith("cluster.louvain_") and method != "louvain":
            continue
        if path.startswith("cluster.leiden_") and method != "leiden":
            continue
        params[path] = suggest_value(trial, path, spec)

    return params


def sanitize_for_json(value: Any):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(item) for item in value]
    return value


def canonicalize_params(params: Dict[str, Any]) -> str:
    return json.dumps(sanitize_for_json(params), sort_keys=True, separators=(",", ":"))


def get_best_trial(study: optuna.Study):
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return None
    return study.best_trial


def maybe_enqueue_trial(study: optuna.Study, params: Dict[str, Any]):
    target = canonicalize_params(params)
    for trial in study.trials:
        previous_params = trial.user_attrs.get("resolved_params", trial.params)
        if canonicalize_params(previous_params) == target:
            return
    study.enqueue_trial(params)


def find_completed_duplicate(study: optuna.Study, params: Dict[str, Any]):
    target = canonicalize_params(params)
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        previous_params = trial.user_attrs.get("resolved_params", trial.params)
        if canonicalize_params(previous_params) == target:
            return trial
    return None


def evaluate_trial_batch(system, batch_records: List[Dict[str, Any]], *, args):
    if not batch_records:
        return {}

    inference_cfgs = {
        str(record["trial"].number): record["cfg"]
        for record in batch_records
    }
    return evaluate_system_many_configs(
        system,
        inference_cfgs,
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


def export_study_summary(
    study: optuna.Study,
    output_path: Path,
    *,
    args,
    storage: str,
    checkpoint_path: Path,
    search_space: Dict[str, Dict[str, Any]],
):
    best_trial = get_best_trial(study)
    payload = {
        "study_name": study.study_name,
        "storage": storage,
        "checkpoint": str(checkpoint_path),
        "metric": args.metric,
        "direction": "maximize",
        "search_space": search_space,
        "evaluation": {
            "device": args.device,
            "batch_size": args.batch_size,
            "dataset_length": args.dataset_length,
            "height": args.height,
            "width": args.width,
            "max_objects": args.max_objects,
            "seed": args.seed,
            "ap_threshold": args.ap_threshold,
            "use_gt_prototypes": args.use_gt_prototypes,
        },
        "best_trial": None if best_trial is None else {
            "number": best_trial.number,
            "value": best_trial.value,
            "params": best_trial.params,
            "user_attrs": sanitize_for_json(best_trial.user_attrs),
            "state": best_trial.state.name,
        },
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": sanitize_for_json(trial.user_attrs),
                "state": trial.state.name,
            }
            for trial in study.trials
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def main():
    args = parse_args()
    output_path = Path(args.output_json)
    checkpoint_path = Path(args.checkpoint).resolve()
    search_space = load_search_space(args.search_space_json)

    if args.storage is None:
        storage_path = output_path.with_suffix(".db").resolve()
        storage = f"sqlite:///{storage_path}"
    else:
        storage = args.storage

    system, ckpt = load_system_checkpoint(str(checkpoint_path), map_location=args.device)
    system = system.to(args.device)
    base_inference_cfg = dataclass_from_dict(PrototypeInferenceConfig, ckpt["inference_config"])

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup_trials)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    study.set_user_attr("checkpoint", str(checkpoint_path))
    study.set_user_attr("metric", args.metric)
    study.set_user_attr("search_space", search_space)
    study.set_user_attr(
        "evaluation",
        {
            "device": args.device,
            "batch_size": args.batch_size,
            "dataset_length": args.dataset_length,
            "height": args.height,
            "width": args.width,
            "max_objects": args.max_objects,
            "seed": args.seed,
            "ap_threshold": args.ap_threshold,
            "use_gt_prototypes": args.use_gt_prototypes,
        },
    )

    maybe_enqueue_trial(study, BEST_KNOWN_PARAMS)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Study name: {study.study_name}")
    print(f"Storage: {storage}")
    print(f"Search metric: {args.metric}")
    print(f"Existing trials: {len(study.trials)}")
    print(f"Parallel trials per round: {args.parallel_trials}")
    print(f"Output JSON: {output_path.resolve()}")
    start_time = time.time()
    scheduled_trials = 0

    while scheduled_trials < args.n_trials:
        if args.timeout is not None and (time.time() - start_time) >= args.timeout:
            break

        round_records: List[Dict[str, Any]] = []
        pending_by_params: Dict[str, Dict[str, Any]] = {}

        while len(round_records) < args.parallel_trials and scheduled_trials < args.n_trials:
            if args.timeout is not None and (time.time() - start_time) >= args.timeout:
                break

            trial = study.ask()
            scheduled_trials += 1
            params = sample_params(trial, search_space)
            canonical_params = canonicalize_params(params)

            duplicate_trial = find_completed_duplicate(study, params)
            if duplicate_trial is not None:
                metrics = duplicate_trial.user_attrs.get("metrics")
                if metrics is not None:
                    objective_value = read_metric(metrics, args.metric)
                    trial.set_user_attr("metrics", metrics)
                    trial.set_user_attr("resolved_params", params)
                    trial.set_user_attr("duplicate_of", duplicate_trial.number)
                    study.tell(trial, objective_value)
                    print(
                        f"Trial {trial.number} reused trial {duplicate_trial.number} | "
                        f"{args.metric}={objective_value:.4f} | "
                        f"params={json.dumps(params, sort_keys=True)}",
                        flush=True,
                    )
                    continue

            if canonical_params in pending_by_params:
                pending_by_params[canonical_params]["aliases"].append(trial)
                continue

            record = {
                "trial": trial,
                "params": params,
                "cfg": build_inference_config(base_inference_cfg, params),
                "aliases": [],
            }
            pending_by_params[canonical_params] = record
            round_records.append(record)

        if not round_records:
            continue

        try:
            batch_results = evaluate_trial_batch(system, round_records, args=args)
        except Exception as exc:
            for record in round_records:
                study.tell(record["trial"], state=optuna.trial.TrialState.FAIL)
                for alias_trial in record["aliases"]:
                    study.tell(alias_trial, state=optuna.trial.TrialState.FAIL)
            export_study_summary(
                study,
                output_path,
                args=args,
                storage=storage,
                checkpoint_path=checkpoint_path,
                search_space=search_space,
            )
            if not args.continue_on_error:
                raise
            print(f"Batch failed: {exc}", flush=True)
            continue

        for record in round_records:
            trial = record["trial"]
            params = record["params"]
            overall, by_count = batch_results[str(trial.number)]
            metrics = {
                "overall": overall,
                "by_object_count": by_count,
            }
            objective_value = read_metric(metrics, args.metric)

            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("resolved_params", params)
            study.tell(trial, objective_value)
            print(
                f"Trial {trial.number} finished | {args.metric}={objective_value:.4f} | "
                f"params={json.dumps(params, sort_keys=True)}",
                flush=True,
            )

            for alias_trial in record["aliases"]:
                alias_trial.set_user_attr("metrics", metrics)
                alias_trial.set_user_attr("resolved_params", params)
                alias_trial.set_user_attr("duplicate_of", trial.number)
                study.tell(alias_trial, objective_value)
                print(
                    f"Trial {alias_trial.number} reused trial {trial.number} in-batch | "
                    f"{args.metric}={objective_value:.4f} | "
                    f"params={json.dumps(params, sort_keys=True)}",
                    flush=True,
                )

        export_study_summary(
            study,
            output_path,
            args=args,
            storage=storage,
            checkpoint_path=checkpoint_path,
            search_space=search_space,
        )

    export_study_summary(
        study,
        output_path,
        args=args,
        storage=storage,
        checkpoint_path=checkpoint_path,
        search_space=search_space,
    )

    best_trial = get_best_trial(study)
    if best_trial is not None:
        print()
        print(f"Best trial: {best_trial.number}")
        print(f"Best {args.metric}: {best_trial.value:.4f}")
        print(f"Best params: {json.dumps(best_trial.params, sort_keys=True)}")
        print(f"Saved study summary to {output_path.resolve()}")


if __name__ == "__main__":
    main()
