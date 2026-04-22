from __future__ import annotations

from copy import deepcopy
import argparse
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment


METHODS = (
    {"kind": "erm"},
    {"kind": "group_balanced_erm"},
    {"kind": "group_dro", "dro_eta": 0.1},
    {"kind": "irm", "penalty_weight": 1.0, "anneal_epochs": 5},
    {"kind": "jtt", "upweight": 5.0},
    {"kind": "adversarial_probe", "adv_weight": 0.05},
    {"kind": "counterfactual_adversarial", "adv_weight": 0.05, "consistency_weight": 0.2},
    {"kind": "counterfactual_augmentation", "consistency_weight": 0.2},
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Specific config path to sweep. Can be passed multiple times.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/experiments",
        help="Directory of YAML configs to sweep when --config is not supplied.",
    )
    parser.add_argument(
        "--match",
        default="",
        help="Only run configs whose filename or config name contains this text.",
    )
    parser.add_argument(
        "--skip-incompatible",
        action="store_true",
        help="Skip methods that require a causal mask when the config cannot provide one.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned run names without loading data or fitting models.",
    )
    args = parser.parse_args()
    configs = (
        [Path(path) for path in args.config]
        if args.config
        else sorted(Path(args.config_dir).glob("*.yaml"))
    )
    if args.match:
        configs = [
            config
            for config in configs
            if args.match in config.name or args.match in load_config(config).get("name", "")
        ]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for config_path in configs:
            base = load_config(config_path)
            hidden_dim = int(base.get("method", {}).get("hidden_dim", 64))
            for method in METHODS:
                if (
                    args.skip_incompatible
                    and _requires_causal_mask(method)
                    and not _config_has_causal_mask(base)
                ):
                    print(f"skipping {base['name']}_{method['kind']} (requires causal mask)")
                    continue
                config = deepcopy(base)
                config["name"] = f"{base['name']}_{method['kind']}"
                config["method"] = {**method, "hidden_dim": hidden_dim}
                config.setdefault("training", {})
                if method["kind"] in {
                    "irm",
                    "group_dro",
                    "jtt",
                    "adversarial_probe",
                    "counterfactual_adversarial",
                }:
                    config["training"]["epochs"] = max(int(config["training"].get("epochs", 15)), 60)
                if args.dry_run:
                    print(f"would run {config['name']}")
                    continue
                out = tmp_path / f"{config['name']}.yaml"
                out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                print(run_experiment(out))


def _requires_causal_mask(method: dict[str, object]) -> bool:
    return str(method.get("kind")) in {"counterfactual_adversarial", "counterfactual_augmentation"}


def _config_has_causal_mask(config: dict[str, object]) -> bool:
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict):
        return False
    fixture_kinds = {
        "synthetic_linear",
        "synthetic_nonlinear",
        "dsprites_tiny",
        "causal3d_tiny",
        "waterbirds_tiny",
        "shapes_spurious_tiny",
        "text_toy",
        "fewshot_ner_tiny",
    }
    if dataset.get("kind") in fixture_kinds:
        return True
    return bool(dataset.get("causal_feature_columns") or dataset.get("causal_feature_prefixes"))


if __name__ == "__main__":
    main()
