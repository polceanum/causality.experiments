from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

from causality_experiments.config import load_config
from scripts.prepare_waterbirds_features import PreparedWaterbirdsFeatures
from scripts import run_waterbirds_official_dfr


def test_run_waterbirds_official_dfr_smoke(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "official_dfr.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "waterbirds_features_official_dfr_val_tr",
                "benchmark": {
                    "kind": "real",
                    "id": "waterbirds",
                    "comparable_to_literature": True,
                },
                "dataset": {
                    "kind": "waterbirds_features",
                    "path": "data/waterbirds/features_official_erm.csv",
                },
                "method": {
                    "kind": "official_dfr_val_tr",
                    "official_dfr_c_grid": [1.0],
                    "official_dfr_num_retrains": 1,
                    "official_dfr_balance_val": True,
                    "official_dfr_add_train": False,
                },
                "training": {"device": "cpu"},
                "metrics": ["accuracy", "worst_group_accuracy"],
                "output_dir": str(tmp_path / "runs"),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def fake_prepare(**_: object) -> PreparedWaterbirdsFeatures:
        features_csv = tmp_path / "features_official_erm_smoke.csv"
        features_csv.write_text("split,y,place,group,feature_0\nval,0,0,0,0.0\n", encoding="utf-8")
        manifest_path = features_csv.with_suffix(features_csv.suffix + ".manifest.json")
        return PreparedWaterbirdsFeatures(
            features_csv=features_csv,
            manifest_path=manifest_path,
            feature_extractor="torchvision_resnet50_imagenet1k_v1_waterbirds_official_erm_sgd_aug_e100_penultimate",
            feature_source="local smoke fixture",
            split_definition="official split",
            base_metrics={
                "val/accuracy": 0.91,
                "val/worst_group_accuracy": 0.81,
                "test/accuracy": 0.92,
                "test/worst_group_accuracy": 0.82,
            },
        )

    def fake_run_experiment(config_path_arg: str | Path, output_root: str | Path | None = None) -> Path:
        config = load_config(config_path_arg)
        run_dir = Path(output_root or config["output_dir"]) / f"{config['name']}-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": config,
            "benchmark": {"provenance": dict(config["benchmark"]["provenance"])},
            "metrics": {
                "val/accuracy": 0.95,
                "val/worst_group_accuracy": 0.85,
                "test/accuracy": 0.96,
                "test/worst_group_accuracy": 0.86,
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return run_dir

    monkeypatch.setattr(run_waterbirds_official_dfr, "prepare_waterbirds_features_artifact", fake_prepare)
    monkeypatch.setattr(run_waterbirds_official_dfr, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_official_dfr.py",
            "--official-dfr-config",
            str(config_path),
            "--features-dir",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--tag",
            "smoke",
        ],
    )

    run_waterbirds_official_dfr.main()

    comparison_path = tmp_path / "runs" / "waterbirds_features_official_dfr_val_tr_smoke-fake" / "official_waterbirds_comparison.json"
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["base_erm_test_wga"] == 0.82
    assert payload["official_dfr_test_wga"] == 0.86
    assert payload["feature_extractor"].endswith("penultimate")
