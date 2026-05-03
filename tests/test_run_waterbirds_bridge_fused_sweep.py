from pathlib import Path
import json

import numpy as np
import torch
import yaml

from causality_experiments.data import DatasetBundle
from scripts import run_waterbirds_bridge_fused_sweep as sweep


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
                "method": {"kind": method, "official_dfr_num_retrains": 1},
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


def test_bridge_fused_sweep_reports_paired_deltas(tmp_path: Path, monkeypatch) -> None:
    features = tmp_path / "features.csv"
    baseline = tmp_path / "baseline.yaml"
    candidate = tmp_path / "candidate.yaml"
    traces = tmp_path / "traces"
    _write_features(features)
    _write_config(baseline, name="official_dfr", method="official_dfr_val_tr")
    _write_config(candidate, name="official_shrink", method="official_causal_shrink_dfr_val_tr")
    _write_bridge_trace(traces)

    def fake_run_experiment(config_path: Path, output_root: Path) -> Path:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        name = str(config["name"])
        if "bridge_fused" in name or "policy_fused" in name:
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

    summary = sweep.run_bridge_fused_sweep(
        baseline_config_path=baseline,
        candidate_config_path=candidate,
        dataset_path=str(features),
        bridge_input_dir=traces,
        out_dir=tmp_path / "scores",
        output_csv=tmp_path / "rows.csv",
        output_json=tmp_path / "summary.json",
        seeds=[101],
        top_k_values=[1],
        bridge_fused_weights=[0.2],
        support_variants=[
            "env_filter",
            "soft_env_penalty",
            "score_square",
            "constrained_support",
            "constrained_support_bridge",
            "artifact_risk_boundary",
            "active_boundary",
            "active_boundary_model_effect",
            "active_boundary_model_effect_ensemble",
            "active_boundary_model_effect_env_guard",
            "active_boundary_paired_replacement",
        ],
        bridge_score_source="bridge_fused",
        bridge_alpha=10.0,
        bridge_exclude_datasets=[],
        policy_input_dir=traces,
        policy_alpha=10.0,
        policy_exclude_datasets=[],
        card_top_k=2,
        random_control_count=1,
        num_retrains=1,
        training_device="cpu",
        output_root=tmp_path / "runs",
    )

    assert (tmp_path / "rows.csv").exists()
    candidate_summary = summary["candidates"][0]
    assert candidate_summary["mean_delta_to_baseline"] == 0.009999999999999898
    assert candidate_summary["mean_delta_to_stats"] == 0.019999999999999907
    assert candidate_summary["non_negative_best_random_seeds"] == 1
    labels = {candidate["label"] for candidate in summary["candidates"]}
    assert "bridge_fused_w0p2_env_filter_top1" in labels
    assert "bridge_fused_w0p2_soft_env_penalty_top1" in labels
    assert "bridge_fused_w0p2_score_square_top1" in labels
    assert "bridge_fused_w0p2_constrained_support_top1" in labels
    assert "bridge_fused_w0p2_constrained_support_bridge_top1" in labels
    assert "bridge_fused_w0p2_artifact_risk_boundary_top1" in labels
    assert "bridge_fused_w0p2_active_boundary_top1" in labels
    assert "bridge_fused_w0p2_active_boundary_model_effect_top1" in labels
    assert "bridge_fused_w0p2_active_boundary_model_effect_ensemble_top1" in labels
    assert "bridge_fused_w0p2_active_boundary_model_effect_env_guard_top1" in labels
    assert "bridge_fused_w0p2_active_boundary_paired_replacement_top1" in labels
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_env_filter.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_soft_env_penalty.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_score_square.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_constrained_support_top1.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_constrained_support_bridge_top1.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_artifact_risk_boundary_top1.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_active_boundary_top1.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_active_boundary_model_effect_top1.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_active_boundary_model_effect_ensemble_top1.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_active_boundary_model_effect_env_guard_top1.csv").exists()
    assert (tmp_path / "scores" / "scores_bridge_fused_w0p2_active_boundary_paired_replacement_top1.csv").exists()
    random_summary = summary["random_controls"][0]
    assert random_summary["label"] == "random_score_0_top1"


def test_bridge_fused_sweep_can_use_policy_fused_source(tmp_path: Path, monkeypatch) -> None:
    features = tmp_path / "features.csv"
    baseline = tmp_path / "baseline.yaml"
    candidate = tmp_path / "candidate.yaml"
    traces = tmp_path / "traces"
    _write_features(features)
    _write_config(baseline, name="official_dfr", method="official_dfr_val_tr")
    _write_config(candidate, name="official_shrink", method="official_causal_shrink_dfr_val_tr")
    _write_bridge_trace(traces)
    run_dir = traces / "fixture"
    (run_dir / "latent_clue_packets.jsonl").write_text(
        "\n".join(
            [
                '{"candidate_id":"good","feature_name":"feature_good","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"causal_target":1.0}',
                '{"candidate_id":"bad","feature_name":"feature_bad","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"causal_target":0.0}',
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "feature_clues.csv").write_text(
        "feature_name,causal_target\nfeature_good,1.0\nfeature_bad,0.0\n",
        encoding="utf-8",
    )

    def fake_run_experiment(config_path: Path, output_root: Path) -> Path:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        name = str(config["name"])
        test_wga = 0.94 if "policy_fused" in name else 0.93
        output = output_root / name
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
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
        return output

    monkeypatch.setattr(sweep, "run_experiment", fake_run_experiment)

    summary = sweep.run_bridge_fused_sweep(
        baseline_config_path=baseline,
        candidate_config_path=candidate,
        dataset_path=str(features),
        bridge_input_dir=traces,
        out_dir=tmp_path / "scores",
        output_csv=tmp_path / "rows.csv",
        output_json=tmp_path / "summary.json",
        seeds=[101],
        top_k_values=[1],
        bridge_fused_weights=[0.5],
        support_variants=[],
        bridge_score_source="policy_fused",
        bridge_alpha=10.0,
        bridge_exclude_datasets=[],
        policy_input_dir=traces,
        policy_alpha=10.0,
        policy_exclude_datasets=[],
        card_top_k=2,
        random_control_count=0,
        num_retrains=1,
        training_device="cpu",
        output_root=tmp_path / "runs",
    )

    labels = {candidate["label"] for candidate in summary["candidates"]}
    assert "policy_fused_w0p5_top1" in labels
    assert (tmp_path / "scores" / "scores_policy_fused_w0p5.csv").exists()


def test_bridge_fused_sweep_can_use_pairwise_fused_source(tmp_path: Path, monkeypatch) -> None:
    features = tmp_path / "features.csv"
    baseline = tmp_path / "baseline.yaml"
    candidate = tmp_path / "candidate.yaml"
    traces = tmp_path / "traces"
    _write_features(features)
    _write_config(baseline, name="official_dfr", method="official_dfr_val_tr")
    _write_config(candidate, name="official_shrink", method="official_causal_shrink_dfr_val_tr")
    _write_bridge_trace(traces)

    def fake_run_experiment(config_path: Path, output_root: Path) -> Path:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        name = str(config["name"])
        test_wga = 0.94 if "pairwise_bridge_fused" in name else 0.93
        output = output_root / name
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
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
        return output

    monkeypatch.setattr(sweep, "run_experiment", fake_run_experiment)

    summary = sweep.run_bridge_fused_sweep(
        baseline_config_path=baseline,
        candidate_config_path=candidate,
        dataset_path=str(features),
        bridge_input_dir=traces,
        out_dir=tmp_path / "scores",
        output_csv=tmp_path / "rows.csv",
        output_json=tmp_path / "summary.json",
        seeds=[101],
        top_k_values=[1],
        bridge_fused_weights=[0.3],
        support_variants=[],
        bridge_score_source="pairwise_bridge_fused",
        bridge_alpha=10.0,
        bridge_exclude_datasets=[],
        policy_input_dir=traces,
        policy_alpha=10.0,
        policy_exclude_datasets=[],
        card_top_k=2,
        random_control_count=0,
        num_retrains=1,
        training_device="cpu",
        output_root=tmp_path / "runs",
    )

    labels = {candidate["label"] for candidate in summary["candidates"]}
    assert "pairwise_bridge_fused_w0p3_top1" in labels
    assert (tmp_path / "scores" / "scores_pairwise_bridge_fused_w0p3.csv").exists()


def test_constrained_support_preserves_stats_core_and_caps_env_risk() -> None:
    clue_rows = [
        {"feature_name": "stats_good", "label_corr": "0.9", "env_corr": "0.1"},
        {"feature_name": "bridge_good", "label_corr": "0.8", "env_corr": "0.2"},
        {"feature_name": "env_risk", "label_corr": "0.1", "env_corr": "0.9"},
        {"feature_name": "safe_fill", "label_corr": "0.7", "env_corr": "0.1"},
    ]
    stats_rows = [
        {"feature_name": "stats_good", "score": "1.0"},
        {"feature_name": "env_risk", "score": "0.8"},
        {"feature_name": "safe_fill", "score": "0.7"},
        {"feature_name": "bridge_good", "score": "0.1"},
    ]
    bridge_rows = [
        {"feature_name": "env_risk", "score": "1.0"},
        {"feature_name": "bridge_good", "score": "0.9"},
        {"feature_name": "safe_fill", "score": "0.4"},
        {"feature_name": "stats_good", "score": "0.3"},
    ]

    rows = sweep.build_constrained_support_score_rows(
        clue_rows=clue_rows,
        stats_rows=stats_rows,
        candidate_rows=bridge_rows,
        top_k=2,
        stats_core_fraction=0.5,
        env_dominant_cap=0,
    )

    selected = [row["feature_name"] for row in sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:2]]
    assert selected == ["stats_good", "bridge_good"]
    assert "env_risk" not in selected
    assert {row["score_source"] for row in rows} == {"constrained_support"}


def test_artifact_risk_boundary_penalizes_near_cutoff_only() -> None:
    clue_rows = [
        {"feature_name": "core", "label_corr": "0.9", "env_corr": "0.1"},
        {"feature_name": "env_boundary", "label_corr": "0.1", "env_corr": "0.9"},
        {"feature_name": "safe_boundary", "label_corr": "0.8", "env_corr": "0.1"},
        {"feature_name": "tail", "label_corr": "0.2", "env_corr": "0.1"},
    ]
    bridge_rows = [
        {"feature_name": "core", "score": "1.0"},
        {"feature_name": "env_boundary", "score": "0.9"},
        {"feature_name": "safe_boundary", "score": "0.88"},
        {"feature_name": "tail", "score": "0.1"},
    ]
    weights = np.zeros(10, dtype=np.float64)
    weights[8] = 1.0
    risk_head = sweep.ArtifactRiskHead(
        weights=weights,
        mean=np.zeros(10, dtype=np.float64),
        scale=np.ones(10, dtype=np.float64),
        train_trace_count=4,
    )

    rows = sweep.build_artifact_risk_score_rows(
        clue_rows=clue_rows,
        candidate_rows=bridge_rows,
        risk_head=risk_head,
        top_k=2,
        risk_weight=0.25,
        boundary_fraction=0.5,
    )

    by_feature = {row["feature_name"]: row for row in rows}
    selected = [row["feature_name"] for row in sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:2]]
    assert selected == ["core", "safe_boundary"]
    assert float(by_feature["core"]["score"]) == 1.0
    assert float(by_feature["env_boundary"]["artifact_risk"]) > float(by_feature["safe_boundary"]["artifact_risk"])
    assert {row["score_source"] for row in rows} == {"artifact_risk_boundary"}


def test_artifact_risk_head_learns_nonzero_baseline(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    _write_bridge_trace(traces)
    risk_head = sweep.fit_artifact_risk_head(traces, alpha=10.0, exclude_datasets=[])
    rows = sweep.build_artifact_risk_score_rows(
        clue_rows=[
            {"feature_name": "feature_good", "label_corr": "0.9", "env_corr": "0.1", "corr_margin": "0.8"},
            {"feature_name": "feature_bad", "label_corr": "0.1", "env_corr": "0.8", "corr_margin": "-0.7"},
        ],
        candidate_rows=[
            {"feature_name": "feature_good", "score": "1.0"},
            {"feature_name": "feature_bad", "score": "0.9"},
        ],
        risk_head=risk_head,
        top_k=1,
        risk_weight=0.25,
        boundary_fraction=None,
    )

    by_feature = {row["feature_name"]: row for row in rows}
    assert risk_head.train_trace_count == 2
    assert float(by_feature["feature_bad"]["artifact_risk"]) > 0.0
    assert float(by_feature["feature_bad"]["artifact_risk"]) > float(by_feature["feature_good"]["artifact_risk"])


def test_active_boundary_retests_near_cutoff_only(monkeypatch) -> None:
    candidate_rows = [
        {"feature_name": "core", "score": "1.0"},
        {"feature_name": "weak_boundary", "score": "0.90"},
        {"feature_name": "strong_boundary", "score": "0.89"},
        {"feature_name": "tail", "score": "0.1"},
    ]
    clue_rows = [{"feature_name": row["feature_name"]} for row in candidate_rows]
    calls: list[str] = []

    def fake_execute_clue_test(bundle, spec, *, packet=None, model=None, split_name="train"):
        calls.append(spec.feature_name)
        if spec.feature_name == "strong_boundary":
            label_delta, env_delta, random_delta, selectivity = 1.0, 0.0, 0.0, 0.2
        elif spec.feature_name == "weak_boundary":
            label_delta, env_delta, random_delta, selectivity = 0.0, 0.6, 0.2, -0.1
        else:
            label_delta, env_delta, random_delta, selectivity = 0.0, 0.0, 0.0, 0.0
        return {
            "test_effect_label_delta": label_delta,
            "test_effect_env_delta": env_delta,
            "test_random_control_delta": random_delta,
            "test_effect_selectivity": selectivity,
        }

    monkeypatch.setattr(sweep, "execute_clue_test", fake_execute_clue_test)

    rows = sweep.build_active_boundary_score_rows(
        bundle=object(),
        clue_rows=clue_rows,
        candidate_rows=candidate_rows,
        top_k=2,
        boundary_fraction=0.5,
        evidence_weight=0.25,
    )

    selected = [row["feature_name"] for row in sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:2]]
    assert set(selected) == {"core", "strong_boundary"}
    assert calls == ["strong_boundary", "weak_boundary"]
    assert {row["score_source"] for row in rows} == {"active_boundary"}


def test_active_boundary_model_effect_promotes_helpful_boundary_feature() -> None:
    rng = np.random.default_rng(7)
    y = np.tile(np.array([0, 1], dtype=np.int64), 80)
    env = np.repeat(np.array([0, 1], dtype=np.int64), 80)
    rng.shuffle(env)
    group = env * 2 + y
    x = np.stack(
        [
            y + rng.normal(scale=0.35, size=len(y)),
            env + rng.normal(scale=0.05, size=len(y)),
            y + rng.normal(scale=0.05, size=len(y)),
            rng.normal(size=len(y)),
        ],
        axis=1,
    ).astype(np.float32)
    split = {
        "x": torch.tensor(x, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.long),
        "env": torch.tensor(env, dtype=torch.long),
        "group": torch.tensor(group, dtype=torch.long),
    }
    bundle = DatasetBundle(
        name="probe_fixture",
        task="classification",
        splits={"train": split},
        input_dim=4,
        output_dim=2,
        metadata={"feature_columns": ["core", "env_boundary", "strong_boundary", "tail"]},
    )
    clue_rows = [
        {"feature_name": "core", "feature_index": "0", "label_corr": "0.9", "env_corr": "0.1"},
        {"feature_name": "env_boundary", "feature_index": "1", "label_corr": "0.1", "env_corr": "0.9"},
        {"feature_name": "strong_boundary", "feature_index": "2", "label_corr": "0.9", "env_corr": "0.1"},
        {"feature_name": "tail", "feature_index": "3", "label_corr": "0.1", "env_corr": "0.1"},
    ]
    candidate_rows = [
        {"feature_name": "core", "score": "1.0"},
        {"feature_name": "env_boundary", "score": "0.90"},
        {"feature_name": "strong_boundary", "score": "0.89"},
        {"feature_name": "tail", "score": "0.1"},
    ]

    rows = sweep.build_active_boundary_model_effect_score_rows(
        bundle=bundle,
        clue_rows=clue_rows,
        candidate_rows=candidate_rows,
        top_k=2,
        boundary_fraction=0.5,
        evidence_weight=0.35,
        probe_seed=5,
    )

    by_feature = {row["feature_name"]: row for row in rows}
    selected = [row["feature_name"] for row in sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:2]]
    assert set(selected) == {"core", "strong_boundary"}
    assert float(by_feature["strong_boundary"]["active_boundary_model_effect"]) > float(
        by_feature["env_boundary"]["active_boundary_model_effect"]
    )
    assert {row["score_source"] for row in rows} == {"active_boundary_model_effect"}

    ensemble_rows = sweep.build_active_boundary_model_effect_score_rows(
        bundle=bundle,
        clue_rows=clue_rows,
        candidate_rows=candidate_rows,
        top_k=2,
        boundary_fraction=0.5,
        evidence_weight=0.30,
        probe_seeds=(5, 11, 17),
        score_source="active_boundary_model_effect_ensemble",
    )
    ensemble_selected = [
        row["feature_name"]
        for row in sorted(ensemble_rows, key=lambda row: float(row["score"]), reverse=True)[:2]
    ]
    assert set(ensemble_selected) == {"core", "strong_boundary"}
    assert {row["score_source"] for row in ensemble_rows} == {"active_boundary_model_effect_ensemble"}

    guarded_rows = sweep.build_active_boundary_model_effect_score_rows(
        bundle=bundle,
        clue_rows=clue_rows,
        candidate_rows=candidate_rows,
        top_k=2,
        boundary_fraction=0.5,
        evidence_weight=0.35,
        probe_seed=5,
        env_risk_weight=0.50,
        score_source="active_boundary_model_effect_env_guard",
    )
    guarded_by_feature = {row["feature_name"]: row for row in guarded_rows}
    guarded_selected = [
        row["feature_name"]
        for row in sorted(guarded_rows, key=lambda row: float(row["score"]), reverse=True)[:2]
    ]
    assert set(guarded_selected) == {"core", "strong_boundary"}
    assert float(guarded_by_feature["env_boundary"]["active_boundary_env_risk"]) > 0.0
    assert {row["score_source"] for row in guarded_rows} == {"active_boundary_model_effect_env_guard"}

    paired_rows = sweep.build_active_boundary_paired_replacement_score_rows(
        bundle=bundle,
        clue_rows=clue_rows,
        candidate_rows=candidate_rows,
        top_k=2,
        boundary_fraction=0.5,
        probe_seeds=(5, 11),
        env_risk_weight=0.0,
    )
    paired_by_feature = {row["feature_name"]: row for row in paired_rows}
    paired_selected = [
        row["feature_name"]
        for row in sorted(paired_rows, key=lambda row: float(row["score"]), reverse=True)[:2]
    ]
    assert set(paired_selected) == {"core", "strong_boundary"}
    assert paired_by_feature["strong_boundary"]["active_boundary_pair_role"] == "accepted"
    assert paired_by_feature["env_boundary"]["active_boundary_pair_role"] == "evicted"
    assert float(paired_by_feature["strong_boundary"]["active_boundary_pair_delta"]) > 0.0
    assert {row["score_source"] for row in paired_rows} == {"active_boundary_paired_replacement"}
