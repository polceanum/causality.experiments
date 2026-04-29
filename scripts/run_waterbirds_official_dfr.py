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
) -> dict[str, Any]:
    config = deepcopy(base)
    config["name"] = name
    config.setdefault("dataset", {})
    config["dataset"]["path"] = str(features_csv)
    config.setdefault("benchmark", {})
    config["benchmark"].setdefault("provenance", {})
    config["benchmark"]["provenance"] = dict(provenance)
    config["output_dir"] = str(output_root)
    return config


def _compact_metrics(run_dir: Path, base_metrics: dict[str, float]) -> dict[str, Any]:
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    config = payload["config"]
    return {
        "config": config.get("name", ""),
        "run": run_dir.name,
        "method": config.get("method", {}).get("kind", ""),
        "feature_extractor": payload.get("benchmark", {}).get("provenance", {}).get("feature_extractor", ""),
        "base_erm_val_wga": base_metrics.get("val/worst_group_accuracy"),
        "base_erm_test_wga": base_metrics.get("test/worst_group_accuracy"),
        "base_erm_val_acc": base_metrics.get("val/accuracy"),
        "base_erm_test_acc": base_metrics.get("test/accuracy"),
        "official_dfr_val_wga": metrics.get("val/worst_group_accuracy"),
        "official_dfr_test_wga": metrics.get("test/worst_group_accuracy"),
        "official_dfr_val_acc": metrics.get("val/accuracy"),
        "official_dfr_test_acc": metrics.get("test/accuracy"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tag", default="", help="Optional suffix for feature/run names.")
    parser.add_argument("--download-dir", default="data/downloads")
    parser.add_argument("--raw-dir", default="data/waterbirds/raw")
    parser.add_argument("--features-dir", default="data/waterbirds")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument(
        "--official-dfr-config",
        default="configs/benchmarks/waterbirds_features_official_dfr_val_tr.yaml",
    )
    args = parser.parse_args()

    base_config = load_config(args.official_dfr_config)
    tag = f"_{args.tag.strip()}" if args.tag.strip() else ""
    stem = f"waterbirds_features_official_dfr_val_tr{tag}"
    features_csv = Path(args.features_dir) / f"features_official_erm{tag}.csv"
    output_root = Path(args.output_root)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        prep_config = tmp_path / f"{stem}_prep.yaml"
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
        config = _method_config(
            base_config,
            name=stem,
            features_csv=artifact.features_csv,
            provenance={
                "feature_extractor": artifact.feature_extractor,
                "feature_source": artifact.feature_source,
                "split_definition": artifact.split_definition,
            },
            output_root=output_root,
        )
        config_path = tmp_path / f"{stem}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(json.dumps({"stage": "run_official_dfr", "config": config["name"]}, sort_keys=True), flush=True)
        run_dir = run_experiment(config_path, output_root)
        compact = _compact_metrics(run_dir, artifact.base_metrics)
        (run_dir / "official_waterbirds_comparison.json").write_text(
            json.dumps(compact, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(compact, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
