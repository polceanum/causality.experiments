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
from scripts.prepare_waterbirds_features import prepare_waterbirds_features


def _epochs(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


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
        "method": {"kind": "dfr"},
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
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default="1,3,5", help="Comma-separated ERM fine-tuning epochs.")
    parser.add_argument("--mode", choices=("head", "layer4", "all"), default="layer4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--balance-groups", action="store_true")
    parser.add_argument("--tag", default="", help="Optional suffix for feature/run names, e.g. sgd_aug.")
    parser.add_argument("--features-dir", default="data/waterbirds")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--dfr-config", default="configs/benchmarks/waterbirds_features_dfr.yaml")
    parser.add_argument("--causal-dfr-config", default="configs/benchmarks/waterbirds_features_causal_dfr.yaml")
    args = parser.parse_args()

    dfr_base = load_config(args.dfr_config)
    causal_base = load_config(args.causal_dfr_config)
    features_dir = Path(args.features_dir)
    output_root = Path(args.output_root)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for epoch in _epochs(args.epochs):
            tag = f"_{args.tag.strip()}" if args.tag.strip() else ""
            stem = f"waterbirds_features_erm_{args.mode}{tag}_e{epoch}"
            features_csv = features_dir / f"features_erm_{args.mode}{tag}_e{epoch}.csv"
            prep_config = tmp_path / f"{stem}.yaml"
            _prep_config(prep_config, features_csv)
            print(
                json.dumps(
                    {
                        "stage": "prepare_features",
                        "epoch": epoch,
                        "mode": args.mode,
                        "optimizer": args.optimizer,
                        "augment": args.augment,
                        "balance_groups": args.balance_groups,
                        "features_csv": str(features_csv),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            prepare_waterbirds_features(
                download_dir=Path("data/downloads"),
                raw_dir=Path("data/waterbirds/raw"),
                features_csv=features_csv,
                config_path=prep_config,
                device_name=args.device,
                batch_size=args.batch_size,
                limit=None,
                force_download=False,
                force_extract=False,
                overwrite_features=args.overwrite_features,
                erm_finetune_epochs=epoch,
                erm_finetune_lr=args.lr,
                erm_finetune_weight_decay=args.weight_decay,
                erm_finetune_mode=args.mode,
                erm_finetune_optimizer=args.optimizer,
                erm_finetune_momentum=args.momentum,
                erm_finetune_augment=args.augment,
                erm_finetune_balance_groups=args.balance_groups,
            )
            provenance = load_config(prep_config).get("benchmark", {}).get("provenance", {})
            configs = [
                _method_config(
                    dfr_base,
                    name=f"{stem}_dfr",
                    features_csv=features_csv,
                    provenance=provenance,
                    output_root=output_root,
                ),
                _method_config(
                    causal_base,
                    name=f"{stem}_causal_dfr",
                    features_csv=features_csv,
                    provenance=provenance,
                    output_root=output_root,
                ),
            ]
            for config in configs:
                config_path = tmp_path / f"{config['name']}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                print(
                    json.dumps(
                        {"stage": "run_dfr", "config": config["name"]},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                run_dir = run_experiment(config_path, output_root)
                print(json.dumps(_compact_metrics(run_dir), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
