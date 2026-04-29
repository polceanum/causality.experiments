from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

from causality_experiments.config import load_config
from scripts import run_waterbirds_official_backbone_sweep, run_waterbirds_official_representation_sweep
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


def test_run_waterbirds_official_representation_sweep_smoke(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.yaml"
    candidate = tmp_path / "candidate.yaml"
    _write_config(baseline, name="waterbirds_features_official_dfr_val_tr", method_kind="official_dfr_val_tr")
    _write_config(candidate, name="waterbirds_features_official_adv_representation_dfr_score_gate", method_kind="official_representation_dfr")
    features_csv = tmp_path / "features.csv"
    features_csv.write_text("split,y,place,group,feature_0\ntrain,0,0,0,0.0\nval,0,0,0,0.0\ntest,0,0,0,0.0\n", encoding="utf-8")

    def fake_run_experiment(config_path_arg: str | Path, output_root: str | Path | None = None) -> Path:
        config = load_config(config_path_arg)
        run_dir = Path(output_root or config["output_dir"]) / f"{config['name']}-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed = int(config["seed"])
        if config["method"]["kind"] == "official_dfr_val_tr":
            test_wga = 0.90 + seed * 0.0001
        else:
            test_wga = 0.91 + seed * 0.0001
        payload = {
            "config": config,
            "metrics": {
                "val/accuracy": 0.95,
                "val/worst_group_accuracy": 0.85,
                "test/accuracy": 0.96,
                "test/worst_group_accuracy": test_wga,
                "feature_importance/nuisance_to_causal": 0.3,
                "probe/selectivity": 0.2,
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        return run_dir

    monkeypatch.setattr(run_waterbirds_official_representation_sweep, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_official_representation_sweep.py",
            "--features-csv",
            str(features_csv),
            "--baseline-config",
            str(baseline),
            "--candidate-configs",
            str(candidate),
            "--seeds",
            "101",
            "102",
            "--output-root",
            str(tmp_path / "runs"),
            "--output-csv",
            str(tmp_path / "rows.csv"),
            "--output-json",
            str(tmp_path / "summary.json"),
        ],
    )

    run_waterbirds_official_representation_sweep.main()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_label"] == "baseline"
    assert {candidate["label"] for candidate in summary["candidates"]} == {"baseline", "candidate"}


def test_run_waterbirds_official_backbone_sweep_smoke(tmp_path: Path, monkeypatch) -> None:
    dfr_config = tmp_path / "official_dfr.yaml"
    _write_config(dfr_config, name="waterbirds_features_official_dfr_val_tr", method_kind="official_dfr_val_tr")

    def fake_prepare(**kwargs: object) -> PreparedWaterbirdsFeatures:
        features_csv = tmp_path / "features_backbone.csv"
        features_csv.write_text("split,y,place,group,feature_0\ntrain,0,0,0,0.0\nval,0,0,0,0.0\ntest,0,0,0,0.0\n", encoding="utf-8")
        env_adv_weight = float(kwargs["erm_env_adv_weight"])
        epochs = int(kwargs["erm_finetune_epochs"])
        lr = float(kwargs["erm_finetune_lr"])
        seed = 101
        return PreparedWaterbirdsFeatures(
            features_csv=features_csv,
            manifest_path=features_csv.with_suffix(".json"),
            feature_extractor="official_backbone_smoke",
            feature_source="smoke",
            split_definition="official split",
            base_metrics={
                "train/accuracy": 0.93,
                "train/worst_group_accuracy": 0.83,
                "val/accuracy": 0.91,
                "val/worst_group_accuracy": 0.81,
                "test/accuracy": 0.92,
                "test/worst_group_accuracy": 0.82,
            },
            resolved_settings={
                "batch_size": 32,
                "erm_finetune_epochs": epochs,
                "erm_finetune_lr": lr,
                "erm_finetune_weight_decay": 1e-3,
                "erm_finetune_mode": "all",
                "erm_finetune_optimizer": "sgd",
                "erm_finetune_momentum": 0.9,
                "erm_finetune_augment": True,
                "erm_finetune_balance_groups": False,
                "erm_env_adv_weight": env_adv_weight,
                "erm_env_adv_hidden_dim": 128 if env_adv_weight > 0.0 else 0,
                "erm_env_adv_loss_weight": 1.0,
                "erm_finetune_warmup_epochs": 0,
                "erm_finetune_warmup_mode": "head",
                "weights_variant": "legacy_pretrained",
                "eval_transform_style": "official",
                "feature_extractor_suffix": f"waterbirds_official_backbone_e{epochs}_lr{lr:g}_envadv{env_adv_weight:g}_seed{seed}_penultimate",
                "erm_finetune_preset": "",
            },
        )

    def fake_run_experiment(config_path_arg: str | Path, output_root: str | Path | None = None) -> Path:
        config = load_config(config_path_arg)
        run_dir = Path(output_root or config["output_dir"]) / f"{config['name']}-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": config,
            "metrics": {
                "val/accuracy": 0.95,
                "val/worst_group_accuracy": 0.85,
                "test/accuracy": 0.96,
                "test/worst_group_accuracy": 0.87,
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        return run_dir

    monkeypatch.setattr(run_waterbirds_official_backbone_sweep, "prepare_waterbirds_features_artifact", fake_prepare)
    monkeypatch.setattr(run_waterbirds_official_backbone_sweep, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_official_backbone_sweep.py",
            "--official-dfr-config",
            str(dfr_config),
            "--seeds",
            "101",
            "--epochs",
            "50",
            "--lrs",
            "0.001",
            "--env-adv-weights",
            "0.0",
            "0.05",
            "--output-root",
            str(tmp_path / "runs"),
            "--output-csv",
            str(tmp_path / "backbone_rows.csv"),
            "--output-json",
            str(tmp_path / "backbone_summary.json"),
        ],
    )

    run_waterbirds_official_backbone_sweep.main()
    summary = json.loads((tmp_path / "backbone_summary.json").read_text(encoding="utf-8"))
    assert len(summary["candidates"]) == 2


def test_run_waterbirds_official_backbone_sweep_blocks_broken_cached_artifact(tmp_path: Path, monkeypatch) -> None:
    dfr_config = tmp_path / "official_dfr.yaml"
    _write_config(dfr_config, name="waterbirds_features_official_dfr_val_tr", method_kind="official_dfr_val_tr")

    def fake_prepare(**_: object) -> PreparedWaterbirdsFeatures:
        features_csv = tmp_path / "features_backbone.csv"
        features_csv.write_text("split,y,place,group,feature_0\ntrain,0,0,0,0.0\n", encoding="utf-8")
        return PreparedWaterbirdsFeatures(
            features_csv=features_csv,
            manifest_path=features_csv.with_suffix(".json"),
            feature_extractor="official_backbone_cached",
            feature_source="smoke",
            split_definition="official split",
            base_metrics={
                "train/accuracy": 0.77,
                "train/worst_group_accuracy": 0.0,
                "val/accuracy": 0.78,
                "val/worst_group_accuracy": 0.0,
                "test/accuracy": 0.78,
                "test/worst_group_accuracy": 0.0,
            },
            resolved_settings={},
        )

    def fail_run_experiment(*_: object, **__: object) -> Path:
        raise AssertionError("broken base ERM artifacts must not run downstream DFR")

    monkeypatch.setattr(run_waterbirds_official_backbone_sweep, "prepare_waterbirds_features_artifact", fake_prepare)
    monkeypatch.setattr(run_waterbirds_official_backbone_sweep, "run_experiment", fail_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_official_backbone_sweep.py",
            "--reuse-features",
            "--official-dfr-config",
            str(dfr_config),
            "--seeds",
            "101",
            "--epochs",
            "50",
            "--lrs",
            "0.001",
            "--env-adv-weights",
            "0.0",
            "--output-root",
            str(tmp_path / "runs"),
            "--output-csv",
            str(tmp_path / "blocked_rows.csv"),
            "--output-json",
            str(tmp_path / "blocked_summary.json"),
        ],
    )

    run_waterbirds_official_backbone_sweep.main()

    summary = json.loads((tmp_path / "blocked_summary.json").read_text(encoding="utf-8"))
    assert summary["candidates"] == []
    assert summary["blocked_rows"][0]["manifest_settings_status"] == "missing_manifest_settings"
    assert summary["blocked_rows"][0]["base_metric_status"].startswith("blocked_base_erm")


def test_run_waterbirds_official_backbone_sweep_passes_limit_to_feature_prep(tmp_path: Path, monkeypatch) -> None:
    dfr_config = tmp_path / "official_dfr.yaml"
    _write_config(dfr_config, name="waterbirds_features_official_dfr_val_tr", method_kind="official_dfr_val_tr")
    seen: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> PreparedWaterbirdsFeatures:
        seen.update(kwargs)
        features_csv = Path(kwargs["features_csv"])
        features_csv.write_text("split,y,place,group,feature_0\ntrain,0,0,0,0.0\n", encoding="utf-8")
        return PreparedWaterbirdsFeatures(
            features_csv=features_csv,
            manifest_path=features_csv.with_suffix(".json"),
            feature_extractor="official_backbone_limit_smoke",
            feature_source="smoke",
            split_definition="official split",
            base_metrics={
                "train/accuracy": 0.93,
                "train/worst_group_accuracy": 0.83,
                "val/accuracy": 0.91,
                "val/worst_group_accuracy": 0.81,
                "test/accuracy": 0.92,
                "test/worst_group_accuracy": 0.82,
            },
            resolved_settings={
                "batch_size": 32,
                "erm_finetune_epochs": 1,
                "erm_finetune_lr": 0.001,
                "erm_finetune_weight_decay": 1e-3,
                "erm_finetune_mode": "all",
                "erm_finetune_optimizer": "sgd",
                "erm_finetune_momentum": 0.9,
                "erm_finetune_augment": True,
                "erm_finetune_balance_groups": False,
                "erm_env_adv_weight": 0.0,
                "erm_env_adv_hidden_dim": 0,
                "erm_env_adv_loss_weight": 1.0,
                "erm_finetune_warmup_epochs": 0,
                "erm_finetune_warmup_mode": "head",
                "weights_variant": "legacy_pretrained",
                "eval_transform_style": "official",
                "feature_extractor_suffix": "waterbirds_official_backbone_e1_lr0.001_envadv0_limit48_seed101_penultimate",
                "erm_finetune_preset": "",
            },
        )

    def fake_run_experiment(config_path_arg: str | Path, output_root: str | Path | None = None) -> Path:
        config = load_config(config_path_arg)
        run_dir = Path(output_root or config["output_dir"]) / f"{config['name']}-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": config,
            "metrics": {
                "val/accuracy": 0.95,
                "val/worst_group_accuracy": 0.85,
                "test/accuracy": 0.96,
                "test/worst_group_accuracy": 0.87,
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        return run_dir

    monkeypatch.setattr(run_waterbirds_official_backbone_sweep, "prepare_waterbirds_features_artifact", fake_prepare)
    monkeypatch.setattr(run_waterbirds_official_backbone_sweep, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_official_backbone_sweep.py",
            "--official-dfr-config",
            str(dfr_config),
            "--seeds",
            "101",
            "--epochs",
            "1",
            "--lrs",
            "0.001",
            "--env-adv-weights",
            "0.0",
            "--limit",
            "48",
            "--output-root",
            str(tmp_path / "runs"),
            "--output-csv",
            str(tmp_path / "limit_rows.csv"),
            "--output-json",
            str(tmp_path / "limit_summary.json"),
        ],
    )

    run_waterbirds_official_backbone_sweep.main()

    summary = json.loads((tmp_path / "limit_summary.json").read_text(encoding="utf-8"))
    assert seen["limit"] == 48
    assert str(seen["features_csv"]).endswith("features_official_e1_lr0.001_envadv0_limit48_seed101.csv")
    assert summary["candidates"][0]["tag"] == "official_e1_lr0.001_envadv0_limit48"


def test_run_waterbirds_official_backbone_sweep_tags_group_balanced_features(tmp_path: Path, monkeypatch) -> None:
    dfr_config = tmp_path / "official_dfr.yaml"
    _write_config(dfr_config, name="waterbirds_features_official_dfr_val_tr", method_kind="official_dfr_val_tr")
    seen: dict[str, object] = {}

    def fake_prepare(**kwargs: object) -> PreparedWaterbirdsFeatures:
        seen.update(kwargs)
        features_csv = Path(kwargs["features_csv"])
        features_csv.write_text("split,y,place,group,feature_0\ntrain,0,0,0,0.0\n", encoding="utf-8")
        return PreparedWaterbirdsFeatures(
            features_csv=features_csv,
            manifest_path=features_csv.with_suffix(".json"),
            feature_extractor="official_backbone_gb_limit_smoke",
            feature_source="smoke",
            split_definition="official split",
            base_metrics={
                "train/accuracy": 0.93,
                "train/worst_group_accuracy": 0.83,
                "val/accuracy": 0.91,
                "val/worst_group_accuracy": 0.81,
                "test/accuracy": 0.92,
                "test/worst_group_accuracy": 0.82,
            },
            resolved_settings={
                "batch_size": 32,
                "erm_finetune_epochs": 1,
                "erm_finetune_lr": 0.001,
                "erm_finetune_weight_decay": 1e-3,
                "erm_finetune_mode": "all",
                "erm_finetune_optimizer": "sgd",
                "erm_finetune_momentum": 0.9,
                "erm_finetune_augment": True,
                "erm_finetune_balance_groups": True,
                "erm_env_adv_weight": 0.0,
                "erm_env_adv_hidden_dim": 0,
                "erm_env_adv_loss_weight": 1.0,
                "erm_finetune_warmup_epochs": 0,
                "erm_finetune_warmup_mode": "head",
                "weights_variant": "legacy_pretrained",
                "eval_transform_style": "official",
                "feature_extractor_suffix": "waterbirds_official_backbone_e1_lr0.001_envadv0_gb_limit48_seed101_penultimate",
                "erm_finetune_preset": "",
            },
        )

    def fake_run_experiment(config_path_arg: str | Path, output_root: str | Path | None = None) -> Path:
        config = load_config(config_path_arg)
        run_dir = Path(output_root or config["output_dir"]) / f"{config['name']}-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": config,
            "metrics": {
                "val/accuracy": 0.95,
                "val/worst_group_accuracy": 0.85,
                "test/accuracy": 0.96,
                "test/worst_group_accuracy": 0.87,
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        return run_dir

    monkeypatch.setattr(run_waterbirds_official_backbone_sweep, "prepare_waterbirds_features_artifact", fake_prepare)
    monkeypatch.setattr(run_waterbirds_official_backbone_sweep, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_official_backbone_sweep.py",
            "--official-dfr-config",
            str(dfr_config),
            "--seeds",
            "101",
            "--epochs",
            "1",
            "--lrs",
            "0.001",
            "--env-adv-weights",
            "0.0",
            "--balance-groups",
            "--limit",
            "48",
            "--output-root",
            str(tmp_path / "runs"),
            "--output-csv",
            str(tmp_path / "gb_limit_rows.csv"),
            "--output-json",
            str(tmp_path / "gb_limit_summary.json"),
        ],
    )

    run_waterbirds_official_backbone_sweep.main()

    summary = json.loads((tmp_path / "gb_limit_summary.json").read_text(encoding="utf-8"))
    assert seen["erm_finetune_balance_groups"] is True
    assert str(seen["features_csv"]).endswith("features_official_e1_lr0.001_envadv0_gb_limit48_seed101.csv")
    assert summary["candidates"][0]["tag"] == "official_e1_lr0.001_envadv0_gb_limit48"
