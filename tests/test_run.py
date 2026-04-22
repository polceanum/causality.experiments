from pathlib import Path

from causality_experiments.run import run_experiment, summarize_runs


def test_run_experiment_writes_outputs(tmp_path: Path) -> None:
    config = tmp_path / "exp.yaml"
    config.write_text(
        """
name: smoke
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
