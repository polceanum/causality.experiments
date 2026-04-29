from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment
from scripts.prepare_waterbirds_features import prepare_waterbirds_features_artifact


def _prep_config(path: Path, features_csv: Path) -> None:
    payload: dict[str, Any] = {
        "name": path.stem,
        "benchmark": {
            "kind": "real",
            "id": "waterbirds",
            "comparable_to_literature": True,
        },
        "dataset": {
            "kind": "waterbirds_features",
            "path": str(features_csv),
        },
        "method": {"kind": "official_dfr_val_tr"},
        "training": {"device": "auto"},
        "metrics": ["accuracy", "worst_group_accuracy"],
        "output_dir": "outputs/runs",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _method_config(
    base: dict[str, Any],
    *,
    name: str,
    features_csv: Path,
    provenance: dict[str, Any],
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    config = deepcopy(base)
    config["name"] = name
    config.setdefault("dataset", {})
    config["dataset"]["path"] = str(features_csv)
    config.setdefault("benchmark", {})
    config["benchmark"].setdefault("provenance", {})
    config["benchmark"]["provenance"] = dict(provenance)
    config.setdefault("training", {})
    config["training"]["device"] = device
    config["output_dir"] = str(output_root)
    return config


def _compact_metrics(run_dir: Path) -> dict[str, Any]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    config = payload["config"]
    return {
        "config": config.get("name", ""),
        "run": run_dir.name,
        "method": config.get("method", {}).get("kind", ""),
        "feature_extractor": payload.get("benchmark", {}).get("provenance", {}).get("feature_extractor", ""),
        "val_wga": metrics.get("val/worst_group_accuracy"),
        "test_wga": metrics.get("test/worst_group_accuracy"),
        "val_acc": metrics.get("val/accuracy"),
        "test_acc": metrics.get("test/accuracy"),
        "nuisance_to_causal_importance": metrics.get("feature_importance/nuisance_to_causal"),
        "probe_selectivity": metrics.get("probe/selectivity"),
    }


def _comparison_payload(base_metrics: dict[str, float], rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((row for row in rows if row["method"] == "official_dfr_val_tr"), None)
    output: dict[str, Any] = {
        "base_erm": {
            "val_wga": base_metrics.get("val/worst_group_accuracy"),
            "test_wga": base_metrics.get("test/worst_group_accuracy"),
            "val_acc": base_metrics.get("val/accuracy"),
            "test_acc": base_metrics.get("test/accuracy"),
        },
        "methods": rows,
    }
    if baseline is not None:
        baseline_test_wga = baseline.get("test_wga")
        comparisons: list[dict[str, Any]] = []
        for row in rows:
            comparisons.append(
                {
                    "method": row["method"],
                    "run": row["run"],
                    "test_wga": row.get("test_wga"),
                    "delta_to_official_dfr_test_wga": (
                        None
                        if row.get("test_wga") is None or baseline_test_wga is None
                        else float(row["test_wga"]) - float(baseline_test_wga)
                    ),
                    "test_acc": row.get("test_acc"),
                    "probe_selectivity": row.get("probe_selectivity"),
                }
            )
        output["comparisons"] = comparisons
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="", help="Optional suffix for feature/run names.")
    parser.add_argument("--download-dir", default="data/downloads")
    parser.add_argument("--raw-dir", default="data/waterbirds/raw")
    parser.add_argument("--features-dir", default="data/waterbirds")
    parser.add_argument("--features-csv", default="", help="Explicit official feature CSV to reuse.")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--official-dfr-config",
        default="configs/benchmarks/waterbirds_features_official_dfr_val_tr.yaml",
    )
    parser.add_argument(
        "--official-causal-dfr-config",
        default="configs/benchmarks/waterbirds_features_official_causal_dfr.yaml",
    )
    parser.add_argument(
        "--official-counterfactual-config",
        default="configs/benchmarks/waterbirds_features_official_counterfactual_adversarial.yaml",
    )
    parser.add_argument(
        "--official-representation-config",
        default="configs/benchmarks/waterbirds_features_official_adv_representation_dfr.yaml",
    )
    parser.add_argument(
        "--official-representation-configs",
        nargs="*",
        default=[],
    )
    args = parser.parse_args()

    dfr_base = load_config(args.official_dfr_config)
    causal_base = load_config(args.official_causal_dfr_config)
    counterfactual_base = load_config(args.official_counterfactual_config)
    representation_paths = list(args.official_representation_configs) or [args.official_representation_config]
    representation_bases = [(Path(path).stem, load_config(path)) for path in representation_paths]
    tag = f"_{args.tag.strip()}" if args.tag.strip() else ""
    features_csv = (
        Path(args.features_csv)
        if args.features_csv
        else Path(args.features_dir) / f"features_official_erm{tag}.csv"
    )
    output_root = Path(args.output_root)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prep_config = tmp_path / f"waterbirds_features_official_transfer{tag}_prep.yaml"
        _prep_config(prep_config, features_csv)
        artifact = prepare_waterbirds_features_artifact(
            download_dir=Path(args.download_dir),
            raw_dir=Path(args.raw_dir),
            features_csv=features_csv,
            config_path=prep_config,
            device_name=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            force_download=args.force_download,
            force_extract=args.force_extract,
            overwrite_features=args.overwrite_features,
            erm_finetune_preset="official",
        )
        provenance = {
            "feature_extractor": artifact.feature_extractor,
            "feature_source": artifact.feature_source,
            "split_definition": artifact.split_definition,
        }
        configs = [
            _method_config(
                dfr_base,
                name=f"waterbirds_features_official_dfr_val_tr{tag}",
                features_csv=artifact.features_csv,
                provenance=provenance,
                output_root=output_root,
                device=args.device,
            ),
            _method_config(
                causal_base,
                name=f"waterbirds_features_official_causal_dfr{tag}",
                features_csv=artifact.features_csv,
                provenance=provenance,
                output_root=output_root,
                device=args.device,
            ),
            _method_config(
                counterfactual_base,
                name=f"waterbirds_features_official_counterfactual_adversarial{tag}",
                features_csv=artifact.features_csv,
                provenance=provenance,
                output_root=output_root,
                device=args.device,
            ),
        ]
        for stem, representation_base in representation_bases:
            configs.append(
                _method_config(
                    representation_base,
                    name=f"{stem}{tag}",
                    features_csv=artifact.features_csv,
                    provenance=provenance,
                    output_root=output_root,
                    device=args.device,
                )
            )
        rows: list[dict[str, Any]] = []
        for config in configs:
            config_path = tmp_path / f"{config['name']}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(
                json.dumps(
                    {"stage": "run_official_transfer", "config": config["name"], "method": config["method"]["kind"]},
                    sort_keys=True,
                ),
                flush=True,
            )
            run_dir = run_experiment(config_path, output_root)
            row = _compact_metrics(run_dir)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        comparison = _comparison_payload(artifact.base_metrics, rows)
        summary_path = output_root / f"waterbirds_official_transfer{tag}_comparison.json"
        summary_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps({"comparison_path": str(summary_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
