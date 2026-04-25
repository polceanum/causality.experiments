from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment


def _all_trials() -> list[dict[str, object]]:
    return _baseline_trials() + _quick_counterfactual_trials()


def _baseline_trials() -> list[dict[str, object]]:
    return [
        {
            "method": {"kind": "erm", "hidden_dim": 64},
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 64, "weight_decay": 0.0},
        },
        {
            "method": {"kind": "erm", "hidden_dim": 64},
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 128, "weight_decay": 1e-4},
        },
        {
            "method": {"kind": "group_balanced_erm", "hidden_dim": 64},
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 64, "weight_decay": 0.0},
        },
        {
            "method": {"kind": "group_balanced_erm", "hidden_dim": 64},
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 128, "weight_decay": 1e-4},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.01},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.03},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.1},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.3},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.03},
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 64},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.03},
            "training": {"epochs": 160, "lr": 3e-4, "batch_size": 64, "weight_decay": 0.0},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.03},
            "training": {"epochs": 160, "lr": 3e-4, "batch_size": 128, "weight_decay": 1e-4},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.05},
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 64, "weight_decay": 0.0},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.05},
            "training": {"epochs": 160, "lr": 3e-4, "batch_size": 128, "weight_decay": 1e-4},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 128, "dro_eta": 0.03},
            "training": {"epochs": 160, "lr": 3e-4, "batch_size": 64, "weight_decay": 1e-4},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 64, "dro_eta": 0.03},
            "training": {"epochs": 120, "lr": 3e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "group_dro", "hidden_dim": 128, "dro_eta": 0.03},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "jtt", "hidden_dim": 64, "upweight": 10.0, "stage1_epochs": 10},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "jtt", "hidden_dim": 64, "upweight": 20.0, "stage1_epochs": 10},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "jtt", "hidden_dim": 64, "upweight": 50.0, "stage1_epochs": 10},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "jtt", "hidden_dim": 64, "upweight": 20.0, "stage1_epochs": 20},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "jtt", "hidden_dim": 64, "upweight": 50.0, "stage1_epochs": 20},
            "training": {"epochs": 120, "lr": 1e-3, "batch_size": 64},
        },
        {
            "method": {"kind": "jtt", "hidden_dim": 64, "upweight": 20.0, "stage1_epochs": 10},
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 64},
        },
    ]


def _quick_counterfactual_trials() -> list[dict[str, object]]:
    return [
        {
            "dataset": {"causal_mask_min_margin": 0.0, "causal_mask_top_k": 256},
            "method": {
                "kind": "counterfactual_adversarial",
                "hidden_dim": 64,
                "adv_weight": 0.05,
                "adv_schedule": "linear",
                "adv_warmup_frac": 0.3,
                "consistency_weight": 0.1,
            },
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 64, "weight_decay": 0.0},
        },
        {
            "dataset": {"causal_mask_min_margin": 0.01, "causal_mask_top_k": 512},
            "method": {
                "kind": "counterfactual_adversarial",
                "hidden_dim": 64,
                "adv_weight": 0.05,
                "adv_schedule": "linear",
                "adv_warmup_frac": 0.3,
                "consistency_weight": 0.1,
            },
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 64, "weight_decay": 0.0},
        },
        {
            "dataset": {"causal_mask_min_margin": 0.01, "causal_mask_top_k": 1024},
            "method": {
                "kind": "counterfactual_adversarial",
                "hidden_dim": 64,
                "adv_weight": 0.1,
                "adv_schedule": "linear",
                "adv_warmup_frac": 0.3,
                "consistency_weight": 0.2,
            },
            "training": {"epochs": 120, "lr": 3e-4, "batch_size": 64, "weight_decay": 0.0},
        },
    ]


def _trial_name(method: str, params: dict[str, float | int], training: dict[str, float | int]) -> str:
    parts = ["waterbirds_tune", method]
    for key, value in params.items():
        text = str(value).replace(".", "p").replace("-", "m")
        parts.append(f"{key}{text}")
    for key, value in training.items():
        text = str(value).replace(".", "p").replace("-", "m")
        parts.append(f"{key}{text}")
    return "_".join(parts)


def _trials_for_profile(profile: str) -> list[dict[str, object]]:
    if profile == "all":
        return _all_trials()
    if profile == "quick-counterfactual":
        return _quick_counterfactual_trials()
    raise ValueError(f"Unknown profile {profile!r}.")


def _write_results_csv(results: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "run_dir",
        "method",
        "val_wga",
        "test_wga",
        "val_acc",
        "test_acc",
        "params",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    **row,
                    "params": json.dumps(row.get("params", {}), sort_keys=True),
                }
            )


def run_trials(
    config_path: Path,
    top_k: int,
    allowed_methods: set[str] | None = None,
    *,
    profile: str = "all",
    out_path: Path | None = None,
) -> list[dict[str, object]]:
    base = load_config(config_path)
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for trial in _trials_for_profile(profile):
            method = dict(trial["method"])
            if allowed_methods is not None and str(method.get("kind", "")) not in allowed_methods:
                continue
            training = dict(trial["training"])
            dataset = dict(trial.get("dataset", {}))
            config = deepcopy(base)
            config["name"] = _trial_name(
                str(method["kind"]),
                {
                    **dataset,
                    **{key: value for key, value in method.items() if key != "kind"},
                },
                training,
            )
            config["method"] = method
            config.setdefault("training", {})
            config["training"] = {**config["training"], **training}
            config.setdefault("dataset", {})
            config["dataset"] = {**config["dataset"], **dataset}
            temp_config = tmp_path / f"{config['name']}.yaml"
            temp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_dir = Path(run_experiment(temp_config))
            payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            metrics = payload["metrics"]
            result = {
                "name": config["name"],
                "run_dir": run_dir.name,
                "method": method["kind"],
                "val_wga": metrics.get("val/worst_group_accuracy"),
                "test_wga": metrics.get("test/worst_group_accuracy"),
                "val_acc": metrics.get("val/accuracy"),
                "test_acc": metrics.get("test/accuracy"),
                "params": {**method, **training},
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True))
    if out_path is not None:
        _write_results_csv(results, out_path)
    print("TOP_BY_VAL_WGA")
    ranked = sorted(results, key=lambda row: (float(row["val_wga"]), float(row["test_wga"])), reverse=True)
    for row in ranked[:top_k]:
        print(json.dumps(row, sort_keys=True))
    print("BEST_BY_METHOD")
    best_by_method: dict[str, dict[str, object]] = {}
    for row in ranked:
        method = str(row["method"])
        if method not in best_by_method:
            best_by_method[method] = row
    for method in sorted(best_by_method):
        print(json.dumps(best_by_method[method], sort_keys=True))
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/benchmarks/waterbirds_features.yaml",
        help="Benchmark config to tune.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top trials to print after ranking by validation WGA.",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Only run tuning trials for the specified method kinds. Can be passed multiple times.",
    )
    parser.add_argument(
        "--profile",
        default="all",
        choices=("all", "quick-counterfactual"),
        help="Named tuning profile to run.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Optional CSV path for persisted tuning results.",
    )
    args = parser.parse_args()
    allowed_methods = set(args.method) if args.method else None
    out_path = Path(args.out) if args.out else None
    run_trials(Path(args.config), args.top_k, allowed_methods, profile=args.profile, out_path=out_path)


if __name__ == "__main__":
    main()