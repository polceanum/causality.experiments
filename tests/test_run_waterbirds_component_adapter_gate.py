from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts import run_waterbirds_component_adapter_gate as gate


def test_component_adapter_gate_runs_candidate_and_random_controls(tmp_path: Path, monkeypatch) -> None:
    features = tmp_path / "features.csv"
    features.write_text(
        "split,y,place,group,feature_foreground_0000,feature_background_0000\n"
        "train,0,0,0,0.0,1.0\n"
        "train,1,1,3,1.0,0.0\n"
        "val,0,0,0,0.0,1.0\n"
        "val,1,1,3,1.0,0.0\n"
        "test,0,1,2,0.0,0.8\n"
        "test,1,0,1,1.0,0.2\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "base",
                "seed": 101,
                "dataset": {"kind": "waterbirds_features", "path": str(features)},
                "method": {"kind": "official_dfr_val_tr", "official_dfr_num_retrains": 1},
                "training": {"device": "cpu"},
                "metrics": ["accuracy", "worst_group_accuracy"],
                "output_dir": str(tmp_path / "runs"),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    def fake_train_adapter_artifact(*, input_csv: Path, output_csv: Path, output_json: Path, **kwargs):
        output_csv.write_text(Path(input_csv).read_text(encoding="utf-8"), encoding="utf-8")
        output_json.write_text(json.dumps({"output_feature_count": 2}), encoding="utf-8")
        output_csv.with_suffix(output_csv.suffix + ".manifest.json").write_text(
            json.dumps({"feature_components": {"adapted": ["feature_foreground_0000", "feature_background_0000"]}}),
            encoding="utf-8",
        )
        return {"output_feature_count": 2}

    def fake_run_experiment(config_path: Path, output_root: Path) -> Path:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        name = str(config["name"])
        if "clue" in name:
            wga = 0.9
        elif "random" in name:
            wga = 0.85
        else:
            wga = 0.8
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "test/worst_group_accuracy": wga,
                        "test/accuracy": wga,
                        "val/worst_group_accuracy": wga,
                        "val/accuracy": wga,
                    }
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr(gate, "train_adapter_artifact", fake_train_adapter_artifact)
    monkeypatch.setattr(gate, "run_experiment", fake_run_experiment)

    summary = gate.run_component_adapter_gate(
        input_csv=features,
        base_config_path=config_path,
        out_dir=tmp_path / "gate",
        output_csv=tmp_path / "rows.csv",
        output_json=tmp_path / "summary.json",
        output_root=tmp_path / "runs",
        seed=101,
        num_retrains=1,
        device="cpu",
        adapter_epochs=1,
        adapter_lr=0.01,
        env_penalty_weight=0.0,
        env_adversary_weight=0.0,
        clue_prior_weight=1.0,
        random_control_count=2,
    )

    assert summary["best"]["label"] == "clue_adapter"
    assert len(summary["rows"]) == 4
    candidate = next(row for row in summary["rows"] if row["label"] == "clue_adapter")
    assert candidate["delta_to_raw"] == 0.09999999999999998
    assert candidate["delta_to_best_random"] == 0.050000000000000044
