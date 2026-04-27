from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.run import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True, help="Config path to run. Can be passed multiple times.")
    args = parser.parse_args()

    for config in args.config:
        config_path = Path(config)
        run_dir = Path(run_experiment(config_path))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        print(
            json.dumps(
                {
                    "config": config_path.stem,
                    "run": run_dir.name,
                    "val_wga": metrics.get("val/worst_group_accuracy"),
                    "test_wga": metrics.get("test/worst_group_accuracy"),
                    "val_acc": metrics.get("val/accuracy"),
                    "test_acc": metrics.get("test/accuracy"),
                    "causal_probe": metrics.get("probe/causal_accuracy"),
                    "nuisance_probe": metrics.get("probe/nuisance_accuracy"),
                    "selectivity": metrics.get("probe/selectivity"),
                    "nuisance_to_causal_importance": metrics.get("feature_importance/nuisance_to_causal"),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()