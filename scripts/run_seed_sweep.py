from __future__ import annotations

import argparse
from copy import deepcopy
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
    {"kind": "irm", "penalty_weight": 1.0, "anneal_epochs": 5},
    {"kind": "counterfactual_augmentation", "consistency_weight": 0.2},
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default="", help="Filter configs by filename or config name.")
    parser.add_argument("--seeds", default="11,12,13", help="Comma-separated integer seeds.")
    args = parser.parse_args()
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    configs = sorted(Path("configs/experiments").glob("*.yaml"))
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
            embedding_dim = base.get("method", {}).get("embedding_dim")
            for seed in seeds:
                for method in METHODS:
                    config = deepcopy(base)
                    config["seed"] = seed
                    config["dataset"] = {**config.get("dataset", {}), "seed": seed}
                    config["name"] = f"{base['name']}_{method['kind']}_seed{seed}"
                    config["method"] = {**method, "hidden_dim": hidden_dim}
                    if embedding_dim is not None:
                        config["method"]["embedding_dim"] = embedding_dim
                    config.setdefault("training", {})
                    if method["kind"] == "irm":
                        config["training"]["epochs"] = max(
                            int(config["training"].get("epochs", 15)),
                            60,
                        )
                    out = tmp_path / f"{config['name']}.yaml"
                    out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                    print(run_experiment(out))


if __name__ == "__main__":
    main()
