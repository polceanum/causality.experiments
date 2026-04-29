from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import itertools
import json
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.config import load_config
from causality_experiments.run import run_experiment
from scripts.prepare_waterbirds_features import _resolve_erm_settings, prepare_waterbirds_features_artifact


def _settings_match_status(expected: dict[str, Any], actual: dict[str, Any]) -> str:
    if not actual:
        return "missing_manifest_settings"
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        return "settings_mismatch:" + ",".join(sorted(mismatches))
    return "ok"


def _base_metric_status(base_metrics: dict[str, float]) -> str:
    required = (
        "train/worst_group_accuracy",
        "val/worst_group_accuracy",
        "test/worst_group_accuracy",
    )
    missing = [key for key in required if key not in base_metrics]
    if missing:
        return "missing_base_metrics:" + ",".join(missing)
    zero_or_negative = [key for key in required if float(base_metrics.get(key, 0.0)) <= 0.0]
    if zero_or_negative:
        return "blocked_base_erm:" + ",".join(zero_or_negative)
    return "ok"


def _row(
    tag: str,
    seed: int,
    metrics: dict[str, float],
    artifact: Any,
    *,
    settings_status: str,
    base_metric_status: str,
) -> dict[str, Any]:
    return {
        "tag": tag,
        "seed": seed,
        "feature_extractor": artifact.feature_extractor,
        "manifest_settings_status": settings_status,
        "base_metric_status": base_metric_status,
        "base_val_wga": float(artifact.base_metrics.get("val/worst_group_accuracy", 0.0)),
        "base_test_wga": float(artifact.base_metrics.get("test/worst_group_accuracy", 0.0)),
        "base_val_acc": float(artifact.base_metrics.get("val/accuracy", 0.0)),
        "base_test_acc": float(artifact.base_metrics.get("test/accuracy", 0.0)),
        "official_dfr_val_wga": float(metrics.get("val/worst_group_accuracy", 0.0)),
        "official_dfr_test_wga": float(metrics.get("test/worst_group_accuracy", 0.0)),
        "official_dfr_val_acc": float(metrics.get("val/accuracy", 0.0)),
        "official_dfr_test_acc": float(metrics.get("test/accuracy", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", default="data/downloads")
    parser.add_argument("--raw-dir", default="data/waterbirds/raw")
    parser.add_argument("--features-dir", default="data/waterbirds")
    parser.add_argument("--output-root", default="outputs/runs")
    parser.add_argument("--output-csv", default="outputs/dfr_sweeps/official-backbone-sweep.csv")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/official-backbone-sweep.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Use a stratified metadata slice for quick diagnostics only.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[101])
    parser.add_argument("--epochs", nargs="*", type=int, default=[50, 100])
    parser.add_argument("--lrs", nargs="*", type=float, default=[0.001, 0.0003])
    parser.add_argument("--env-adv-weights", nargs="*", type=float, default=[0.0, 0.05])
    parser.add_argument("--balance-groups", action="store_true")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-mode", choices=("head", "layer4", "all"), default="head")
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="Audit existing feature CSV/manifest artifacts instead of regenerating them.",
    )
    parser.add_argument(
        "--official-dfr-config",
        default="configs/benchmarks/waterbirds_features_official_dfr_val_tr.yaml",
    )
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_root = Path(args.output_root)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    dfr_base = load_config(args.official_dfr_config)
    rows: list[dict[str, Any]] = []

    for seed, epochs, lr, env_adv_weight in itertools.product(args.seeds, args.epochs, args.lrs, args.env_adv_weights):
        limit_tag = f"_limit{args.limit}" if args.limit is not None else ""
        balance_tag = "_gb" if args.balance_groups else ""
        tag = f"official_e{epochs}_lr{lr:g}_envadv{env_adv_weight:g}{balance_tag}{limit_tag}_seed{seed}"
        features_csv = Path(args.features_dir) / f"features_{tag}.csv"
        feature_extractor_suffix = f"waterbirds_official_backbone_e{epochs}_lr{lr:g}_envadv{env_adv_weight:g}{balance_tag}{limit_tag}_seed{seed}_penultimate"
        expected_settings = _resolve_erm_settings(
            batch_size=args.batch_size,
            erm_finetune_epochs=epochs,
            erm_finetune_lr=lr,
            erm_finetune_weight_decay=1e-3,
            erm_finetune_mode="all",
            erm_finetune_optimizer="sgd",
            erm_finetune_momentum=0.9,
            erm_finetune_augment=True,
            erm_finetune_balance_groups=args.balance_groups,
            erm_env_adv_weight=env_adv_weight,
            erm_env_adv_hidden_dim=128 if env_adv_weight > 0.0 else 0,
            erm_env_adv_loss_weight=1.0,
            erm_finetune_warmup_epochs=args.warmup_epochs,
            erm_finetune_warmup_mode=args.warmup_mode,
            weights_variant="legacy_pretrained",
            eval_transform_style="official",
            feature_extractor_suffix=feature_extractor_suffix,
            erm_finetune_preset=None,
        )
        expected_settings["erm_finetune_preset"] = ""
        prep_config = Path(tempfile.mkdtemp()) / f"{tag}_prep.yaml"
        prep_config.write_text(
            yaml.safe_dump(
                {
                    "name": prep_config.stem,
                    "benchmark": {"kind": "real", "id": "waterbirds", "comparable_to_literature": True},
                    "dataset": {"kind": "waterbirds_features", "path": str(features_csv)},
                    "method": {"kind": "official_dfr_val_tr"},
                    "training": {"device": args.device},
                    "metrics": ["accuracy", "worst_group_accuracy"],
                    "output_dir": str(output_root),
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "stage": "prepare_backbone_features",
                    "tag": tag,
                    "reuse_features": bool(args.reuse_features),
                    "features_csv": str(features_csv),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        artifact = prepare_waterbirds_features_artifact(
            download_dir=Path(args.download_dir),
            raw_dir=Path(args.raw_dir),
            features_csv=features_csv,
            config_path=prep_config,
            device_name=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            force_download=False,
            force_extract=False,
            overwrite_features=not args.reuse_features,
            erm_finetune_epochs=epochs,
            erm_finetune_lr=lr,
            erm_finetune_weight_decay=1e-3,
            erm_finetune_mode="all",
            erm_finetune_optimizer="sgd",
            erm_finetune_momentum=0.9,
            erm_finetune_augment=True,
            erm_finetune_balance_groups=args.balance_groups,
            erm_env_adv_weight=env_adv_weight,
            erm_env_adv_hidden_dim=128 if env_adv_weight > 0.0 else 0,
            erm_env_adv_loss_weight=1.0,
            erm_finetune_warmup_epochs=args.warmup_epochs,
            erm_finetune_warmup_mode=args.warmup_mode,
            weights_variant="legacy_pretrained",
            eval_transform_style="official",
            feature_extractor_suffix=feature_extractor_suffix,
            erm_finetune_preset=None,
        )
        settings_status = _settings_match_status(expected_settings, dict(getattr(artifact, "resolved_settings", {})))
        base_metric_status = _base_metric_status(dict(artifact.base_metrics))
        metrics: dict[str, float] = {}
        if base_metric_status == "ok":
            config = deepcopy(dfr_base)
            config["name"] = f"waterbirds_features_official_dfr_val_tr_{tag}"
            config["seed"] = seed
            config["dataset"]["path"] = str(artifact.features_csv)
            config["output_dir"] = str(output_root)
            with tempfile.TemporaryDirectory() as tmp:
                config_path = Path(tmp) / f"{tag}.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                print(
                    json.dumps(
                        {"stage": "run_official_dfr", "tag": tag, "config": config["name"]},
                        sort_keys=True,
                    ),
                    flush=True,
                )
                run_dir = run_experiment(config_path, output_root)
            payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            metrics = payload["metrics"]
        row = _row(
            tag,
            seed,
            metrics,
            artifact,
            settings_status=settings_status,
            base_metric_status=base_metric_status,
        )
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["base_metric_status"] != "ok" or row["manifest_settings_status"] != "ok":
            continue
        grouped.setdefault(row["tag"].rsplit("_seed", 1)[0], []).append(float(row["official_dfr_test_wga"]))
    summary = {
        "blocked_rows": [
            row
            for row in rows
            if row["base_metric_status"] != "ok" or row["manifest_settings_status"] != "ok"
        ],
        "candidates": [
            {
                "tag": tag,
                "mean_test_wga": statistics.mean(values),
                "std_test_wga": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "min_test_wga": min(values),
                "max_test_wga": max(values),
            }
            for tag, values in sorted(grouped.items())
        ]
    }
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "output_json": str(output_json)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
