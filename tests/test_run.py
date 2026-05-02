from pathlib import Path
import csv
import pytest

from causality_experiments.run import run_experiment, summarize_runs
from causality_experiments.reporting import is_ad_hoc_config, write_csv_rows
from scripts.report_benchmark_alignment import _experiment_name, build_alignment_rows
from scripts.run_method_sweep import _config_has_causal_mask


def test_run_experiment_writes_outputs(tmp_path: Path) -> None:
    config = tmp_path / "exp.yaml"
    config.write_text(
        """
name: smoke
benchmark:
  kind: real
  id: waterbirds
  comparable_to_literature: true
  provenance:
    feature_extractor: resnet50
    feature_source: local-export
    split_definition: official split
seed: 2
dataset:
  kind: synthetic_linear
  n: 120
method:
  kind: erm
  hidden_dim: 8
training:
  device: cpu
  epochs: 1
  batch_size: 32
metrics:
  - accuracy
""",
        encoding="utf-8",
    )
    run_dir = run_experiment(config, tmp_path / "runs")
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "metrics.csv").exists()
    summary = summarize_runs(tmp_path / "runs")
    assert summary.exists()
    rows = list(csv.DictReader(summary.open()))
    assert rows[0]["benchmark_provenance_complete"] == "True"
    assert rows[0]["benchmark_feature_extractor"] == "resnet50"


def test_alignment_rows_mark_fixture_as_non_comparable(tmp_path: Path) -> None:
    config = tmp_path / "fixture.yaml"
    config.write_text(
        """
name: 05_waterbirds_erm
benchmark:
  kind: fixture
  id: waterbirds
  comparable_to_literature: false
seed: 2
dataset:
  kind: waterbirds_tiny
  n: 120
method:
  kind: erm
  hidden_dim: 8
training:
  device: cpu
  epochs: 1
  batch_size: 32
metrics:
  - accuracy
  - worst_group_accuracy
""",
        encoding="utf-8",
    )
    run_experiment(config, tmp_path / "runs")
    rows = build_alignment_rows(tmp_path / "runs", config_dirs=(str(tmp_path),))
    waterbirds_rows = [row for row in rows if row["benchmark_id"] == "waterbirds" and row["method"] == "erm"]
    assert waterbirds_rows
    assert waterbirds_rows[0]["comparison_status"] == "fixture_only"
    assert waterbirds_rows[0]["literature_sota_wga"] == "0.929"


def test_alignment_rows_emit_blocked_real_benchmark_without_runs(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir()
    config = benchmark_dir / "waterbirds_features.yaml"
    config.write_text(
        """
name: waterbirds_features
benchmark:
  kind: real
  id: waterbirds
  comparable_to_literature: true
dataset:
  kind: waterbirds_features
  path: missing/waterbirds/features.csv
method:
  kind: erm
  hidden_dim: 8
training:
  device: cpu
  epochs: 1
metrics:
  - accuracy
  - worst_group_accuracy
""",
        encoding="utf-8",
    )
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text(
        """
name: smoke
seed: 1
dataset:
  kind: synthetic_linear
  n: 120
method:
  kind: erm
  hidden_dim: 8
training:
  device: cpu
  epochs: 1
metrics:
  - accuracy
""",
        encoding="utf-8",
    )
    run_experiment(smoke, tmp_path / "runs")
    rows = build_alignment_rows(tmp_path / "runs", config_dirs=(str(benchmark_dir),))
    blocked = [row for row in rows if row["config"] == "waterbirds_features"]
    assert blocked
    assert blocked[0]["comparison_status"] == "blocked_missing_local_data"
    assert blocked[0]["literature_sota_wga"] == "0.929"


def test_alignment_rows_emit_blocked_missing_provenance_when_data_exists(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir()
    data_dir = tmp_path / "data" / "waterbirds"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "features.csv"
    csv_path.write_text(
        "split,y,place,feature_0\ntrain,0,0,0.1\nval,0,0,0.1\ntest,0,0,0.1\n",
        encoding="utf-8",
    )
    config = benchmark_dir / "waterbirds_features.yaml"
    config.write_text(
        f"""
name: waterbirds_features
benchmark:
  kind: real
  id: waterbirds
  comparable_to_literature: true
  provenance:
    feature_extractor: ""
    feature_source: local-export
    split_definition: official split
dataset:
  kind: waterbirds_features
  path: {csv_path}
method:
  kind: erm
  hidden_dim: 8
training:
  device: cpu
  epochs: 1
metrics:
  - accuracy
  - worst_group_accuracy
""",
        encoding="utf-8",
    )
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text(
        """
name: smoke
seed: 1
dataset:
  kind: synthetic_linear
  n: 120
method:
  kind: erm
  hidden_dim: 8
training:
  device: cpu
  epochs: 1
metrics:
  - accuracy
""",
        encoding="utf-8",
    )
    run_experiment(smoke, tmp_path / "runs")
    rows = build_alignment_rows(tmp_path / "runs", config_dirs=(str(benchmark_dir),))
    blocked = [row for row in rows if row["config"] == "waterbirds_features"]
    assert blocked
    assert blocked[0]["comparison_status"] == "blocked_missing_provenance"


def test_method_sweep_accepts_derived_causal_mask_strategy() -> None:
    config = {
        "dataset": {
            "kind": "waterbirds_features",
            "path": "data/waterbirds/features.csv",
            "causal_mask_strategy": "label_minus_env_correlation",
        }
    }
    assert _config_has_causal_mask(config) is True


def test_alignment_suffix_parsing_keeps_causal_dfr_distinct() -> None:
    assert _experiment_name("waterbirds_features_causal_dfr") == "waterbirds_features"
    assert _experiment_name("waterbirds_features_dfr") == "waterbirds_features"
    assert _experiment_name("waterbirds_features_official_dfr_val_tr") == "waterbirds_features"
    assert _experiment_name("waterbirds_features_official_causal_shrink_dfr_val_tr") == "waterbirds_features"
    assert _experiment_name("waterbirds_features_official_dfr_val_tr_retrains50") == "waterbirds_features"
    assert _experiment_name("waterbirds_features_official_causal_shrink_dfr_val_tr_gentle") == "waterbirds_features"
    assert _experiment_name("waterbirds_features_counterfactual_causal_dfr") == "waterbirds_features"
    assert _experiment_name("waterbirds_features_loss_weighted_dfr") == "waterbirds_features"
    assert _experiment_name("waterbirds_features_representation_dfr") == "waterbirds_features"
    assert _experiment_name("waterbirds_features_causal_dfr") != "waterbirds_features_causal"


def test_report_ad_hoc_filter_can_preserve_script_scope() -> None:
    assert is_ad_hoc_config("tmp_causal_recheck") is True
    assert is_ad_hoc_config("waterbirds_features_dfr_seed101") is True
    assert is_ad_hoc_config("sweep_waterbirds_dfr") is True
    assert is_ad_hoc_config("waterbirds_tune_dfr") is True
    assert is_ad_hoc_config("sweep_waterbirds_dfr", include_sweep_prefix=False) is False
    assert is_ad_hoc_config("waterbirds_tune_dfr", include_waterbirds_tune=False) is False


def test_write_csv_rows_rejects_empty_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No rows to write"):
        write_csv_rows(tmp_path / "empty.csv", [])


def test_write_csv_rows_writes_header_and_rows(tmp_path: Path) -> None:
    out = tmp_path / "rows.csv"
    write_csv_rows(out, [{"name": "a", "value": "1"}])
    assert out.read_text(encoding="utf-8").splitlines() == ["name,value", "a,1"]


def test_alignment_rows_mark_dfr_validation_usage(tmp_path: Path) -> None:
    config = tmp_path / "waterbirds_features_dfr.yaml"
    config.write_text(
        """
name: waterbirds_features_dfr
benchmark:
  kind: real
  id: waterbirds
  comparable_to_literature: true
  provenance:
    feature_extractor: resnet50
    feature_source: local-export
    split_definition: official split
seed: 2
dataset:
  kind: synthetic_linear
  n: 120
method:
  kind: dfr
  dfr_epochs: 2
training:
  device: cpu
  batch_size: 32
metrics:
  - accuracy
  - worst_group_accuracy
""",
        encoding="utf-8",
    )
    run_experiment(config, tmp_path / "runs")
    rows = build_alignment_rows(tmp_path / "runs", config_dirs=(str(tmp_path),))
    dfr_rows = [row for row in rows if row["method"] == "dfr"]
    assert dfr_rows
    assert dfr_rows[0]["config"] == "waterbirds_features"
    assert dfr_rows[0]["validation_usage"] == "trains_on_validation_groups"
    assert not [row for row in rows if row["config"] == "waterbirds_features_dfr" and row["method"] == ""]
