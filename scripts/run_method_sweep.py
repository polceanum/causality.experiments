from __future__ import annotations

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
    configs = sorted(Path("configs/experiments").glob("*.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for config_path in configs:
            base = load_config(config_path)
            hidden_dim = int(base.get("method", {}).get("hidden_dim", 64))
            for method in METHODS:
                config = deepcopy(base)
                config["name"] = f"{base['name']}_{method['kind']}"
                config["method"] = {**method, "hidden_dim": hidden_dim}
                config.setdefault("training", {})
                if method["kind"] == "irm":
                    config["training"]["epochs"] = max(int(config["training"].get("epochs", 15)), 60)
                out = tmp_path / f"{config['name']}.yaml"
                out.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                print(run_experiment(out))


if __name__ == "__main__":
    main()
