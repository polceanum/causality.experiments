from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.component_representation import (
    feature_columns_from_frame,
    infer_feature_components,
    load_feature_components_from_manifest,
    train_component_adapter,
    write_component_manifest,
)


def _read_clue_rows(path: Path | None) -> list[dict[str, object]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def train_adapter_artifact(
    *,
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
    clue_csv: Path | None = None,
    manifest_path: Path | None = None,
    epochs: int = 200,
    lr: float = 0.05,
    env_penalty_weight: float = 0.2,
    env_adversary_weight: float = 0.0,
    clue_prior_weight: float = 1.0,
    seed: int = 0,
) -> dict[str, object]:
    frame = pd.read_csv(input_csv)
    feature_columns = feature_columns_from_frame(frame)
    clue_rows = _read_clue_rows(clue_csv)
    result = train_component_adapter(
        frame,
        feature_columns=feature_columns,
        clue_rows=clue_rows,
        epochs=epochs,
        lr=lr,
        env_penalty_weight=env_penalty_weight,
        env_adversary_weight=env_adversary_weight,
        clue_prior_weight=clue_prior_weight,
        seed=seed,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.feature_frame.to_csv(output_csv, index=False)
    source_manifest = manifest_path or input_csv.with_suffix(input_csv.suffix + ".manifest.json")
    manifest_components = load_feature_components_from_manifest(source_manifest)
    input_components = infer_feature_components(feature_columns, manifest_components)
    adapted_components = {
        "adapted": [str(column) for column in result.report["adapted_feature_columns"]],
    }
    write_component_manifest(
        output_csv=output_csv,
        feature_columns=result.report["adapted_feature_columns"],
        feature_components=adapted_components,
        source_path=input_csv,
        extra={
            "adapter_report": str(output_json),
            "clue_csv": str(clue_csv or ""),
            "input_components": {key: len(value) for key, value in input_components.items()},
        },
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result.report, indent=2, sort_keys=True), encoding="utf-8")
    return result.report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--clue-csv", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--env-penalty-weight", type=float, default=0.2)
    parser.add_argument("--env-adversary-weight", type=float, default=0.0)
    parser.add_argument("--clue-prior-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    report = train_adapter_artifact(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        output_json=Path(args.output_json),
        clue_csv=Path(args.clue_csv) if args.clue_csv else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        epochs=args.epochs,
        lr=args.lr,
        env_penalty_weight=args.env_penalty_weight,
        env_adversary_weight=args.env_adversary_weight,
        clue_prior_weight=args.clue_prior_weight,
        seed=args.seed,
    )
    print(json.dumps({"output_feature_count": report["output_feature_count"], "train_accuracy": report["train_accuracy"]}, sort_keys=True))


if __name__ == "__main__":
    main()
