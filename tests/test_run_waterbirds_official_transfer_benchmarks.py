from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

from causality_experiments.config import load_config
from scripts import run_waterbirds_official_transfer_benchmarks
from scripts.prepare_waterbirds_features import PreparedWaterbirdsFeatures


def _write_config(path: Path, *, name: str, method_kind: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "benchmark": {
                    "kind": "real",
                    "id": "waterbirds",
                    "comparable_to_literature": True,
                },
                "dataset": {
                    "kind": "waterbirds_features",
                    "path": "data/waterbirds/features_official_erm.csv",
                },
                "method": {"kind": method_kind},
                "training": {"device": "cpu"},
                "metrics": ["accuracy", "worst_group_accuracy"],
                "output_dir": str(path.parent / "runs"),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_run_waterbirds_official_transfer_benchmarks_smoke(tmp_path: Path, monkeypatch) -> None:
    dfr_config = tmp_path / "official_dfr.yaml"
    causal_config = tmp_path / "official_causal.yaml"
    counterfactual_config = tmp_path / "official_counterfactual.yaml"
    representation_config = tmp_path / "official_representation.yaml"
    _write_config(dfr_config, name="waterbirds_features_official_dfr_val_tr", method_kind="official_dfr_val_tr")
    _write_config(causal_config, name="waterbirds_features_official_causal_dfr", method_kind="causal_dfr")
    _write_config(
        counterfactual_config,
        name="waterbirds_features_official_counterfactual_adversarial",
        method_kind="counterfactual_adversarial",
    )
    _write_config(
        representation_config,
        name="waterbirds_features_official_adv_representation_dfr",
        method_kind="official_representation_dfr",
    )

    def fake_prepare(**_: object) -> PreparedWaterbirdsFeatures:
        features_csv = tmp_path / "features_official_erm_smoke.csv"
        features_csv.write_text("split,y,place,group,feature_0\nval,0,0,0,0.0\n", encoding="utf-8")
        return PreparedWaterbirdsFeatures(
            features_csv=features_csv,
            manifest_path=features_csv.with_suffix(features_csv.suffix + ".manifest.json"),
            feature_extractor="torch_hub_resnet50_pretrained_waterbirds_official_erm_sgd_aug_e100_penultimate",
            feature_source="local smoke fixture",
            split_definition="official split",
            base_metrics={
                "val/accuracy": 0.91,
                "val/worst_group_accuracy": 0.81,
                "test/accuracy": 0.92,
                "test/worst_group_accuracy": 0.82,
            },
        )

    method_scores = {
        "official_dfr_val_tr": (0.95, 0.85, 0.96, 0.86),
        "causal_dfr": (0.96, 0.86, 0.965, 0.865),
        "counterfactual_adversarial": (0.97, 0.87, 0.975, 0.875),
        "official_representation_dfr": (0.955, 0.855, 0.968, 0.872),
    }

    def fake_run_experiment(config_path_arg: str | Path, output_root: str | Path | None = None) -> Path:
        config = load_config(config_path_arg)
        run_dir = Path(output_root or config["output_dir"]) / f"{config['name']}-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        val_acc, val_wga, test_acc, test_wga = method_scores[config["method"]["kind"]]
        payload = {
            "config": config,
            "benchmark": {"provenance": dict(config["benchmark"]["provenance"])},
            "metrics": {
                "val/accuracy": val_acc,
                "val/worst_group_accuracy": val_wga,
                "test/accuracy": test_acc,
                "test/worst_group_accuracy": test_wga,
                "feature_importance/nuisance_to_causal": 0.5,
                "probe/selectivity": 0.1,
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return run_dir

    monkeypatch.setattr(run_waterbirds_official_transfer_benchmarks, "prepare_waterbirds_features_artifact", fake_prepare)
    monkeypatch.setattr(run_waterbirds_official_transfer_benchmarks, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_official_transfer_benchmarks.py",
            "--official-dfr-config",
            str(dfr_config),
            "--official-causal-dfr-config",
            str(causal_config),
            "--official-counterfactual-config",
            str(counterfactual_config),
            "--official-representation-config",
            str(representation_config),
            "--features-dir",
            str(tmp_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--tag",
            "smoke",
        ],
    )

    run_waterbirds_official_transfer_benchmarks.main()

    comparison_path = tmp_path / "runs" / "waterbirds_official_transfer_smoke_comparison.json"
    payload = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert payload["base_erm"]["test_wga"] == 0.82
    assert [row["method"] for row in payload["methods"]] == [
        "official_dfr_val_tr",
        "causal_dfr",
        "counterfactual_adversarial",
        "official_representation_dfr",
    ]
    deltas = {row["method"]: row["delta_to_official_dfr_test_wga"] for row in payload["comparisons"]}
    assert deltas["official_dfr_val_tr"] == 0.0
    assert abs(deltas["causal_dfr"] - 0.005) < 1e-9
    assert abs(deltas["counterfactual_adversarial"] - 0.015) < 1e-9
    assert abs(deltas["official_representation_dfr"] - 0.012) < 1e-9
