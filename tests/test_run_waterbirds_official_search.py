from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

from causality_experiments.config import load_config
from scripts import (
    run_waterbirds_official_backbone_sweep,
    run_waterbirds_official_causal_dfr_sweep,
    run_waterbirds_official_representation_sweep,
    run_waterbirds_official_shrink_sweep,
)
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


def test_run_waterbirds_official_causal_dfr_sweep_reports_paired_deltas(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.yaml"
    candidate = tmp_path / "candidate.yaml"
    _write_config(baseline, name="waterbirds_features_official_dfr_val_tr", method_kind="official_dfr_val_tr")
    _write_config(candidate, name="waterbirds_features_official_causal_dfr", method_kind="causal_dfr")
    features_csv = tmp_path / "features.csv"
    features_csv.write_text("split,y,place,group,feature_0\ntrain,0,0,0,0.0\nval,0,0,0,0.0\ntest,0,0,0,0.0\n", encoding="utf-8")

    def fake_load_dataset(config: dict[str, object]) -> dict[str, object]:
        return config

    def fake_fit_method(bundle: dict[str, object], config: dict[str, object]) -> dict[str, object]:
        return config

    def fake_evaluate(model: dict[str, object], bundle: dict[str, object], config: dict[str, object]) -> dict[str, float]:
        seed = int(config["seed"])
        method = dict(config["method"])
        if method["kind"] == "official_dfr_val_tr":
            test_wga = 0.90 + seed * 0.0001
        else:
            nuisance_weight = float(method["causal_dfr_nuisance_weight"])
            test_wga = 0.905 + seed * 0.0001 + nuisance_weight * 0.001
        return {
            "val/accuracy": 0.95,
            "val/worst_group_accuracy": 0.85,
            "test/accuracy": 0.96,
            "test/worst_group_accuracy": test_wga,
            "feature_importance/nuisance_to_causal": 0.3,
            "probe/selectivity": 0.2,
        }

    monkeypatch.setattr(run_waterbirds_official_causal_dfr_sweep, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(run_waterbirds_official_causal_dfr_sweep, "fit_method", fake_fit_method)
    monkeypatch.setattr(run_waterbirds_official_causal_dfr_sweep, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_official_causal_dfr_sweep.py",
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--dataset-path",
            str(features_csv),
            "--seeds",
            "101",
            "102",
            "--causal-mask-top-ks",
            "128",
            "--causal-mask-min-margins",
            "0.01",
            "--nuisance-weights",
            "2.0",
            "3.0",
            "--nuisance-priors",
            "mask",
            "--max-candidates",
            "1",
            "--output-csv",
            str(tmp_path / "paired.csv"),
            "--output-json",
            str(tmp_path / "paired.json"),
        ],
    )

    run_waterbirds_official_causal_dfr_sweep.main()

    rows = (tmp_path / "paired.csv").read_text(encoding="utf-8").splitlines()
    summary = json.loads((tmp_path / "paired.json").read_text(encoding="utf-8"))
    assert len(rows) == 5
    assert summary["candidates"][0]["count"] == 2
    assert summary["candidates"][0]["mean_delta_to_baseline"] > 0.006
    assert summary["candidates"][0]["passes_promotion_gate"] is False


def test_run_waterbirds_official_shrink_sweep_reports_paired_deltas(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.yaml"
    candidate = tmp_path / "candidate.yaml"
    _write_config(baseline, name="waterbirds_features_official_dfr_val_tr", method_kind="official_dfr_val_tr")
    _write_config(candidate, name="waterbirds_features_official_causal_shrink_dfr_val_tr", method_kind="official_causal_shrink_dfr_val_tr")
    features_csv = tmp_path / "features.csv"
    features_csv.write_text("split,y,place,group,feature_0\ntrain,0,0,0,0.0\nval,0,0,0,0.0\ntest,0,0,0,0.0\n", encoding="utf-8")

    def fake_load_dataset(config: dict[str, object]) -> dict[str, object]:
        return config

    def fake_fit_method(bundle: dict[str, object], config: dict[str, object]) -> dict[str, object]:
        return config

    def fake_evaluate(model: dict[str, object], bundle: dict[str, object], config: dict[str, object]) -> dict[str, float]:
        seed = int(config["seed"])
        method = dict(config["method"])
        if method["kind"] == "official_dfr_val_tr":
            test_wga = 0.90 + seed * 0.0001
            selected_shrink = 1.0
        else:
            selected_shrink = min(float(value) for value in method["official_causal_shrink_grid"])
            test_wga = 0.905 + seed * 0.0001 + (1.0 - selected_shrink) * 0.01
        return {
            "val/accuracy": 0.95,
            "val/worst_group_accuracy": 0.85,
            "test/accuracy": 0.96,
            "test/worst_group_accuracy": test_wga,
            "model/official_dfr_best_c": 0.1,
            "model/official_dfr_best_feature_scale": selected_shrink,
        }

    monkeypatch.setattr(run_waterbirds_official_shrink_sweep, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(run_waterbirds_official_shrink_sweep, "fit_method", fake_fit_method)
    monkeypatch.setattr(run_waterbirds_official_shrink_sweep, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_official_shrink_sweep.py",
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--dataset-path",
            str(features_csv),
            "--seeds",
            "101",
            "102",
            "--causal-mask-top-ks",
            "128",
            "--causal-mask-min-margins",
            "0.01",
            "--shrink-priors",
            "mask",
            "--shrink-grid",
            "1.0",
            "0.9",
            "--output-csv",
            str(tmp_path / "shrink.csv"),
            "--output-json",
            str(tmp_path / "shrink.json"),
        ],
    )

    run_waterbirds_official_shrink_sweep.main()

    rows = (tmp_path / "shrink.csv").read_text(encoding="utf-8").splitlines()
    summary = json.loads((tmp_path / "shrink.json").read_text(encoding="utf-8"))
    assert len(rows) == 5
    assert summary["candidates"][0]["count"] == 2
    assert summary["candidates"][0]["mean_delta_to_baseline"] > 0.005
    assert summary["candidates"][0]["passes_promotion_gate"] is False


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
                "erm_finetune_seed": seed,
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
        assert features_csv.parent.exists()
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
                "erm_finetune_seed": 101,
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
            "--features-dir",
            str(tmp_path / "missing" / "waterbirds"),
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
                "erm_finetune_seed": 101,
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


def test_run_waterbirds_official_backbone_sweep_tags_staged_conflict_sample_mode(tmp_path: Path, monkeypatch) -> None:
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
            feature_extractor="official_backbone_conflict_smoke",
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
                "erm_finetune_sample_mode": "conflict_upweight",
                "erm_finetune_minority_weight": 3.0,
                "erm_finetune_sample_warmup_epochs": 1,
                "erm_finetune_contrastive_weight": 0.2,
                "erm_finetune_contrastive_temperature": 0.15,
                "erm_finetune_contrastive_hard_negative_weight": 2.0,
                "erm_env_adv_weight": 0.0,
                "erm_env_adv_hidden_dim": 0,
                "erm_env_adv_loss_weight": 1.0,
                "erm_finetune_warmup_epochs": 0,
                "erm_finetune_warmup_mode": "head",
                "erm_finetune_seed": 101,
                "feature_decomposition": "center_background",
                "weights_variant": "legacy_pretrained",
                "eval_transform_style": "official",
                "feature_extractor_suffix": "waterbirds_official_backbone_e1_lr0.001_envadv0_conflictw3_samplewarm1_supconw0p2_t0p15_hn2_decompcenterbg_limit48_seed101_penultimate",
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
            "--sample-modes",
            "conflict_upweight",
            "--minority-weight",
            "3.0",
            "--sample-warmup-epochs",
            "1",
            "--contrastive-weight",
            "0.2",
            "--contrastive-temperature",
            "0.15",
            "--contrastive-hard-negative-weight",
            "2.0",
            "--feature-decompositions",
            "center_background",
            "--limit",
            "48",
            "--output-root",
            str(tmp_path / "runs"),
            "--output-csv",
            str(tmp_path / "conflict_rows.csv"),
            "--output-json",
            str(tmp_path / "conflict_summary.json"),
        ],
    )

    run_waterbirds_official_backbone_sweep.main()

    summary = json.loads((tmp_path / "conflict_summary.json").read_text(encoding="utf-8"))
    assert seen["erm_finetune_sample_mode"] == "conflict_upweight"
    assert seen["erm_finetune_minority_weight"] == 3.0
    assert seen["erm_finetune_sample_warmup_epochs"] == 1
    assert seen["erm_finetune_contrastive_weight"] == 0.2
    assert seen["erm_finetune_contrastive_temperature"] == 0.15
    assert seen["erm_finetune_contrastive_hard_negative_weight"] == 2.0
    assert seen["erm_finetune_seed"] == 101
    assert seen["feature_decomposition"] == "center_background"
    assert seen["erm_finetune_balance_groups"] is False
    assert str(seen["features_csv"]).endswith("features_official_e1_lr0.001_envadv0_conflictw3_samplewarm1_supconw0p2_t0p15_hn2_decompcenterbg_limit48_seed101.csv")
    assert summary["candidates"][0]["tag"] == "official_e1_lr0.001_envadv0_conflictw3_samplewarm1_supconw0p2_t0p15_hn2_decompcenterbg_limit48"


def test_run_waterbirds_official_backbone_sweep_tags_alternate_representation_source(tmp_path: Path, monkeypatch) -> None:
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
            feature_extractor="alternate_source_smoke",
            feature_source="smoke",
            split_definition="official split",
            base_metrics={},
            resolved_settings={
                "batch_size": 32,
                "erm_finetune_epochs": 0,
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
                "weights_variant": "imagenet1k_v2",
                "eval_transform_style": "weights",
                "feature_extractor_suffix": "waterbirds_official_backbone_e0_lr0.001_envadv0_bconvnextptiny_wimagenet1kpv2_evalweights_limit48_seed101_penultimate",
                "erm_finetune_preset": "",
                "backbone_name": "convnext_tiny",
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
            "0",
            "--lrs",
            "0.001",
            "--env-adv-weights",
            "0.0",
            "--backbones",
            "convnext_tiny",
            "--weights-variants",
            "imagenet1k_v2",
            "--eval-transform-styles",
            "weights",
            "--limit",
            "48",
            "--output-root",
            str(tmp_path / "runs"),
            "--output-csv",
            str(tmp_path / "alternate_rows.csv"),
            "--output-json",
            str(tmp_path / "alternate_summary.json"),
        ],
    )

    run_waterbirds_official_backbone_sweep.main()

    summary = json.loads((tmp_path / "alternate_summary.json").read_text(encoding="utf-8"))
    assert seen["weights_variant"] == "imagenet1k_v2"
    assert seen["backbone_name"] == "convnext_tiny"
    assert seen["eval_transform_style"] == "weights"
    assert str(seen["features_csv"]).endswith(
        "features_official_e0_lr0.001_envadv0_bconvnextptiny_wimagenet1kpv2_evalweights_limit48_seed101.csv"
    )
    assert summary["blocked_rows"] == []
    assert summary["candidates"][0]["tag"] == "official_e0_lr0.001_envadv0_bconvnextptiny_wimagenet1kpv2_evalweights_limit48"
