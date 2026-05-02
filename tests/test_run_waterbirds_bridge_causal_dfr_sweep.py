from pathlib import Path
import json

import yaml

from scripts import run_waterbirds_bridge_causal_dfr_sweep as sweep


def _write_features(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "split,y,place,feature_good,feature_bad",
                "train,0,0,0.0,1.0",
                "train,0,1,0.0,0.8",
                "train,1,0,1.0,0.2",
                "train,1,1,1.0,0.0",
                "val,0,0,0.0,1.0",
                "val,1,1,1.0,0.0",
                "test,0,1,0.0,0.8",
                "test,1,0,1.0,0.2",
            ]
        ),
        encoding="utf-8",
    )


def _write_config(path: Path, *, name: str, method: str) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "name": name,
                "seed": 101,
                "dataset": {"kind": "waterbirds_features", "path": "features.csv"},
                "method": {"kind": method},
                "metrics": ["accuracy", "worst_group_accuracy"],
                "output_dir": "outputs/runs",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_bridge_trace(root: Path) -> None:
    run_dir = root / "fixture"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text('{"config": "configs/experiments/synthetic.yaml"}', encoding="utf-8")
    (run_dir / "training_traces.jsonl").write_text(
        "\n".join(
            [
                '{"feature_name":"feature_good","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"test_value":1.0,"score_delta":0.5,"hypothesis_correct":true,"passed_control":true}',
                '{"feature_name":"feature_bad","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"test_value":0.0,"score_delta":0.0,"hypothesis_correct":false,"passed_control":false}',
            ]
        ),
        encoding="utf-8",
    )


def test_bridge_causal_dfr_sweep_reports_paired_deltas(tmp_path: Path, monkeypatch) -> None:
    features = tmp_path / "features.csv"
    baseline = tmp_path / "baseline.yaml"
    candidate = tmp_path / "candidate.yaml"
    traces = tmp_path / "traces"
    _write_features(features)
    _write_config(baseline, name="official_dfr", method="official_dfr_val_tr")
    _write_config(candidate, name="causal_dfr_soft", method="causal_dfr")
    _write_bridge_trace(traces)
    seen_configs: list[dict] = []

    def fake_run_experiment(config_path: Path, output_root: Path) -> Path:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        seen_configs.append(config)
        name = str(config["name"])
        if "bridge_fused" in name:
            test_wga = 0.94
        elif "stats" in name:
            test_wga = 0.92
        else:
            test_wga = 0.93
        run_dir = output_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "val/worst_group_accuracy": test_wga,
                        "test/worst_group_accuracy": test_wga,
                        "val/accuracy": test_wga,
                        "test/accuracy": test_wga,
                    }
                }
            ),
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr(sweep, "run_experiment", fake_run_experiment)

    summary = sweep.run_bridge_causal_dfr_sweep(
        baseline_config_path=baseline,
        candidate_config_path=candidate,
        dataset_path=str(features),
        bridge_input_dir=traces,
        out_dir=tmp_path / "scores",
        output_csv=tmp_path / "rows.csv",
        output_json=tmp_path / "summary.json",
        seeds=[101],
        top_k_values=[1],
        bridge_fused_weight=0.2,
        nuisance_weights=[30.0],
        bridge_alpha=10.0,
        bridge_exclude_datasets=[],
        card_top_k=2,
        dfr_num_retrains=2,
        training_device="cpu",
        output_root=tmp_path / "runs",
    )

    candidate_configs = [config for config in seen_configs if "bridge_fused" in str(config["name"])]
    assert candidate_configs[0]["method"]["causal_dfr_nuisance_prior"] == "soft_scores"
    assert candidate_configs[0]["method"]["dfr_num_retrains"] == 2
    assert candidate_configs[0]["dataset"]["discovery_score_soft_selection"] == "selected"
    assert (tmp_path / "rows.csv").exists()
    candidate_summary = summary["candidates"][0]
    assert candidate_summary["mean_delta_to_baseline"] == 0.009999999999999898
    assert candidate_summary["mean_delta_to_stats"] == 0.019999999999999907
