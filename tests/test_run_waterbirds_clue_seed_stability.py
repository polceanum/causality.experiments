from __future__ import annotations

import json
from pathlib import Path
import sys

from causality_experiments.config import load_config
from scripts import run_waterbirds_clue_seed_stability


def test_summary_marks_paired_candidate_promotion() -> None:
    rows = [
        {"label": "baseline", "seed": 1, "test_wga": 0.90, "val_wga": 0.91},
        {"label": "baseline", "seed": 2, "test_wga": 0.91, "val_wga": 0.92},
        {"label": "candidate", "seed": 1, "test_wga": 0.94, "val_wga": 0.93},
        {"label": "candidate", "seed": 2, "test_wga": 0.95, "val_wga": 0.94},
    ]

    summary = run_waterbirds_clue_seed_stability._summary(
        rows,
        baseline_label="baseline",
        min_mean_delta=0.003,
        min_mean_wga=0.935,
    )

    candidate = next(row for row in summary["candidates"] if row["label"] == "candidate")
    assert candidate["passes_promotion_gate"] is True
    assert candidate["non_negative_seed_count"] == 2
    assert candidate["mean_delta_to_baseline"] > 0.03


def test_clue_seed_stability_main_smoke(tmp_path: Path, monkeypatch) -> None:
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text(
        """
name: baseline
seed: 101
dataset:
  kind: waterbirds_features
  path: features.csv
method:
  kind: official_dfr_val_tr
training:
  device: cpu
metrics:
- accuracy
- worst_group_accuracy
output_dir: outputs/runs
""",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        """
name: candidate
seed: 101
dataset:
  kind: waterbirds_features
  path: features.csv
method:
  kind: causal_dfr
  causal_dfr_nuisance_prior: soft_scores
training:
  device: cpu
metrics:
- accuracy
- worst_group_accuracy
output_dir: outputs/runs
""",
        encoding="utf-8",
    )

    def fake_write_clue_artifacts(config, *, split, card_top_k, sources, top_k_values, out_dir):
        paths = {}
        for source in sources:
            path = out_dir / f"scores_{source}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "dataset,feature_index,feature_name,score\nwaterbirds_features,0,feature_0,0.9\n",
                encoding="utf-8",
            )
            paths[f"scores_{source}"] = path
        ablation = out_dir / "source_ablation.csv"
        ablation.write_text("label,top_k\nstats,1\n", encoding="utf-8")
        paths["source_ablation"] = ablation
        return paths

    def fake_run_experiment(config_path_arg: str | Path, output_root: str | Path | None = None) -> Path:
        config = load_config(config_path_arg)
        run_dir = Path(output_root or config["output_dir"]) / f"{config['name']}-fake"
        run_dir.mkdir(parents=True, exist_ok=True)
        seed = int(config["seed"])
        is_baseline = config["method"]["kind"] == "official_dfr_val_tr"
        test_wga = (0.90 if is_baseline else 0.94) + seed * 0.0001
        payload = {
            "config": config,
            "metrics": {
                "val/accuracy": 0.95,
                "val/worst_group_accuracy": 0.93,
                "test/accuracy": 0.96,
                "test/worst_group_accuracy": test_wga,
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        return run_dir

    monkeypatch.setattr(run_waterbirds_clue_seed_stability, "write_clue_artifacts", fake_write_clue_artifacts)
    monkeypatch.setattr(run_waterbirds_clue_seed_stability, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_waterbirds_clue_seed_stability.py",
            "--baseline-config",
            str(baseline),
            "--candidate-config",
            str(candidate),
            "--dataset-path",
            str(tmp_path / "features.csv"),
            "--seeds",
            "101,102",
            "--top-k",
            "1",
            "--sources",
            "stats",
            "--out-dir",
            str(tmp_path / "stability"),
            "--output-root",
            str(tmp_path / "runs"),
        ],
    )

    run_waterbirds_clue_seed_stability.main()

    summary = json.loads((tmp_path / "stability" / "seed_stability_summary.json").read_text(encoding="utf-8"))
    assert summary["baseline_label"] == "baseline"
    candidate_rows = [row for row in summary["candidates"] if row["label"] != "baseline"]
    assert candidate_rows[0]["passes_promotion_gate"] is True
