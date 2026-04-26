from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment
from scripts.tune_discovery_mask import _build_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out",
        default="outputs/runs/waterbirds-scored-gate-weight-sweep.csv",
    )
    args = parser.parse_args()

    base = load_config(
        Path("configs/benchmarks/waterbirds_features_counterfactual_adversarial_schedule_learned_grouped_scored_gate_nuisance0p9.yaml")
    )
    rows: list[dict[str, str | float]] = []
    for weight in args.weight:
        config = deepcopy(base)
        config.setdefault("training", {})["epochs"] = args.epochs
        config.setdefault("training", {})["device"] = args.device
        config.setdefault("method", {})["input_gate_score_weight"] = float(weight)
        config = _build_candidate(
            config,
            top_k=args.top_k,
            variant="discovery_full",
            score_path="outputs/runs/waterbirds-feature-discovery-scores.csv",
        )
        config.setdefault("training", {})["epochs"] = args.epochs
        config.setdefault("training", {})["device"] = args.device
        config.setdefault("method", {})["input_gate_score_weight"] = float(weight)
        config["name"] = config["name"] + f"_scoreweight{str(weight).replace('.', 'p')}"
        tmp_config = Path(tempfile.mkdtemp(prefix="waterbirds-scored-gate-sweep-")) / "config.yaml"
        tmp_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run_dir = run_experiment(tmp_config)
        payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            {
                "score_weight": float(weight),
                "run": run_dir.name,
                "config": config["name"],
                "test_wga": float(metrics["test/worst_group_accuracy"]),
                "val_wga": float(metrics["val/worst_group_accuracy"]),
                "test_acc": float(metrics["test/accuracy"]),
                "val_acc": float(metrics["val/accuracy"]),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)
    for row in rows:
        print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()