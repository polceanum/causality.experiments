from pathlib import Path

from scripts.run_waterbirds_clue_fusion_sweep import (
    build_downstream_candidate,
    build_bridge_score_rows,
    build_policy_score_rows,
    build_source_score_rows,
    resolve_sources,
    source_score,
    with_dataset_path,
    with_runtime_overrides,
)


def test_source_score_uses_language_confidence_for_fused_score() -> None:
    row = {
        "corr_margin": "0.0",
        "soft_causal_target": "0.5",
        "language_causal_score": "1.0",
        "language_spurious_score": "0.0",
        "language_confidence": "1.0",
    }
    assert source_score(row, "stats") == 0.5
    assert source_score(row, "language") > 0.9
    assert 0.5 < source_score(row, "fused") < source_score(row, "language")


def test_source_score_uses_image_confidence_for_fused_score() -> None:
    row = {
        "corr_margin": "0.0",
        "soft_causal_target": "0.5",
        "image_label_score": "1.0",
        "image_background_score": "0.0",
        "image_confidence": "1.0",
    }
    assert source_score(row, "stats") == 0.5
    assert source_score(row, "image") > 0.9
    assert 0.5 < source_score(row, "fused") < source_score(row, "image")


def test_build_source_score_rows_emits_discovery_score_schema() -> None:
    rows = build_source_score_rows(
        [
            {
                "dataset": "waterbirds_features",
                "feature_index": 0,
                "feature_name": "feature_0",
                "soft_causal_target": 0.7,
            }
        ],
        "stats",
    )
    assert rows[0]["dataset"] == "waterbirds_features"
    assert rows[0]["support_score"] == rows[0]["score"]
    assert rows[0]["score_source"] == "stats"


def test_resolve_sources_accepts_comma_separated_unique_values() -> None:
    assert resolve_sources(["fused,stats", "image", "fused"]) == ["fused", "stats", "image"]


def test_resolve_sources_keeps_bridge_opt_in() -> None:
    assert resolve_sources([]) == ["stats", "language", "image", "fused"]
    assert resolve_sources(["bridge"]) == ["bridge"]
    assert resolve_sources(["bridge_fused"]) == ["bridge_fused"]
    assert resolve_sources(["bridge_gated"]) == ["bridge_gated"]
    assert resolve_sources(["policy_safe"]) == ["policy_safe"]


def test_build_downstream_candidate_uses_source_score_path() -> None:
    candidate = build_downstream_candidate(
        {
            "name": "waterbirds_base",
            "dataset": {
                "kind": "waterbirds_features",
                "path": "features.csv",
                "causal_mask_strategy": "label_minus_env_correlation",
            },
        },
        label="fused",
        top_k=128,
        score_path=Path("scores_fused.csv"),
    )
    dataset = candidate["dataset"]
    assert candidate["name"] == "waterbirds_base_clue_fused_top128"
    assert dataset["causal_mask_strategy"] == "discovery_scores"
    assert dataset["discovery_scores_path"] == "scores_fused.csv"
    assert dataset["discovery_score_top_k"] == 128
    assert dataset["discovery_score_threshold"] > 1.0
    assert "discovery_score_soft_selection" not in dataset


def test_build_downstream_candidate_accepts_bridge_score_path() -> None:
    candidate = build_downstream_candidate(
        {
            "name": "waterbirds_base",
            "dataset": {
                "kind": "waterbirds_features",
                "path": "features.csv",
            },
        },
        label="bridge",
        top_k=32,
        score_path=Path("scores_bridge.csv"),
    )
    assert candidate["dataset"]["causal_mask_strategy"] == "discovery_scores"
    assert candidate["dataset"]["discovery_scores_path"] == "scores_bridge.csv"


def test_build_downstream_candidate_accepts_bridge_gated_score_path() -> None:
    candidate = build_downstream_candidate(
        {
            "name": "waterbirds_base",
            "dataset": {
                "kind": "waterbirds_features",
                "path": "features.csv",
            },
        },
        label="bridge_gated",
        top_k=32,
        score_path=Path("scores_bridge_gated.csv"),
    )
    assert candidate["dataset"]["causal_mask_strategy"] == "discovery_scores"
    assert candidate["dataset"]["discovery_scores_path"] == "scores_bridge_gated.csv"


def test_build_downstream_candidate_accepts_policy_safe_score_path() -> None:
    candidate = build_downstream_candidate(
        {
            "name": "waterbirds_base",
            "dataset": {
                "kind": "waterbirds_features",
                "path": "features.csv",
            },
        },
        label="policy_safe",
        top_k=32,
        score_path=Path("scores_policy_safe.csv"),
    )
    assert candidate["dataset"]["causal_mask_strategy"] == "discovery_scores"
    assert candidate["dataset"]["discovery_scores_path"] == "scores_policy_safe.csv"


def test_build_downstream_candidate_can_prune_soft_scores() -> None:
    candidate = build_downstream_candidate(
        {
            "name": "waterbirds_base",
            "dataset": {
                "kind": "waterbirds_features",
                "path": "features.csv",
                "causal_mask_strategy": "label_minus_env_correlation",
            },
        },
        label="image",
        top_k=64,
        score_path=Path("scores_image.csv"),
        prune_soft_scores=True,
    )
    assert candidate["dataset"]["discovery_score_soft_selection"] == "selected"


def test_build_downstream_candidate_keeps_heuristic_control() -> None:
    candidate = build_downstream_candidate(
        {
            "name": "waterbirds_base",
            "dataset": {
                "kind": "waterbirds_features",
                "path": "features.csv",
                "causal_mask_strategy": "label_minus_env_correlation",
                "discovery_scores_path": "old.csv",
            },
        },
        label="heuristic",
        top_k=64,
    )
    dataset = candidate["dataset"]
    assert dataset["causal_mask_strategy"] == "label_minus_env_correlation"
    assert dataset["causal_mask_top_k"] == 64
    assert "discovery_scores_path" not in dataset
    assert "discovery_score_soft_selection" not in dataset


def test_with_dataset_path_overrides_config_without_mutating_original() -> None:
    config = {"dataset": {"kind": "waterbirds_features", "path": "old.csv"}}
    updated = with_dataset_path(config, "new.csv")
    assert updated["dataset"]["path"] == "new.csv"
    assert config["dataset"]["path"] == "old.csv"


def test_with_runtime_overrides_sets_compact_screen_knobs() -> None:
    config = {"method": {"kind": "official_dfr_val_tr"}, "training": {"device": "auto"}}
    updated = with_runtime_overrides(config, official_dfr_num_retrains=2, training_device="cpu")
    assert updated["method"]["official_dfr_num_retrains"] == 2
    assert updated["training"]["device"] == "cpu"
    assert "official_dfr_num_retrains" not in config["method"]


def test_build_bridge_score_rows_scores_target_packets(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    features.write_text(
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
    bundle = __import__("causality_experiments.data", fromlist=["load_dataset"]).load_dataset(
        {
            "dataset": {
                "kind": "waterbirds_features",
                "path": str(features),
            }
        }
    )
    traces = tmp_path / "bridge_runs" / "fixture"
    traces.mkdir(parents=True)
    (traces / "manifest.json").write_text('{"config": "configs/experiments/synthetic.yaml"}', encoding="utf-8")
    (traces / "training_traces.jsonl").write_text(
        "\n".join(
            [
                '{"feature_name":"feature_good","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"test_value":1.0,"score_delta":0.5,"hypothesis_correct":true,"passed_control":true}',
                '{"feature_name":"feature_bad","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"test_value":0.0,"score_delta":0.0,"hypothesis_correct":false,"passed_control":false}',
            ]
        ),
        encoding="utf-8",
    )

    rows = build_bridge_score_rows(bundle, bridge_input_dir=tmp_path / "bridge_runs", exclude_datasets=[])

    by_feature = {row["feature_name"]: row for row in rows}
    assert set(by_feature) == {"feature_good", "feature_bad"}
    assert by_feature["feature_good"]["score_source"] == "bridge"


def test_build_bridge_score_rows_can_blend_with_stats(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    features.write_text(
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
    bundle = __import__("causality_experiments.data", fromlist=["load_dataset"]).load_dataset(
        {"dataset": {"kind": "waterbirds_features", "path": str(features)}}
    )
    traces = tmp_path / "bridge_runs" / "fixture"
    traces.mkdir(parents=True)
    (traces / "manifest.json").write_text('{"config": "configs/experiments/synthetic.yaml"}', encoding="utf-8")
    (traces / "training_traces.jsonl").write_text(
        "\n".join(
            [
                '{"feature_name":"feature_good","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"test_value":1.0,"score_delta":0.5,"hypothesis_correct":true,"passed_control":true}',
                '{"feature_name":"feature_bad","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"test_value":0.0,"score_delta":0.0,"hypothesis_correct":false,"passed_control":false}',
            ]
        ),
        encoding="utf-8",
    )

    rows = build_bridge_score_rows(
        bundle,
        bridge_input_dir=tmp_path / "bridge_runs",
        exclude_datasets=[],
        blend_with_stats_weight=0.2,
    )

    assert {row["score_source"] for row in rows} == {"bridge_fused"}
    assert all(0.0 <= float(row["score"]) <= 1.0 for row in rows)


def test_build_bridge_score_rows_can_gate_stats_with_bridge(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    features.write_text(
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
    bundle = __import__("causality_experiments.data", fromlist=["load_dataset"]).load_dataset(
        {"dataset": {"kind": "waterbirds_features", "path": str(features)}}
    )
    traces = tmp_path / "bridge_runs" / "fixture"
    traces.mkdir(parents=True)
    (traces / "manifest.json").write_text('{"config": "configs/experiments/synthetic.yaml"}', encoding="utf-8")
    (traces / "training_traces.jsonl").write_text(
        "\n".join(
            [
                '{"feature_name":"feature_good","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"test_value":1.0,"score_delta":0.5,"hypothesis_correct":true,"passed_control":true}',
                '{"feature_name":"feature_bad","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"test_value":0.0,"score_delta":0.0,"hypothesis_correct":false,"passed_control":false}',
            ]
        ),
        encoding="utf-8",
    )

    rows = build_bridge_score_rows(
        bundle,
        bridge_input_dir=tmp_path / "bridge_runs",
        exclude_datasets=[],
        blend_with_stats_weight=0.2,
        blend_mode="gated",
    )

    assert {row["score_source"] for row in rows} == {"bridge_gated"}
    assert all(float(row["score"]) >= 0.0 for row in rows)


def test_build_policy_score_rows_can_emit_safe_residual_scores(tmp_path: Path) -> None:
    features = tmp_path / "features.csv"
    features.write_text(
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
    bundle = __import__("causality_experiments.data", fromlist=["load_dataset"]).load_dataset(
        {"dataset": {"kind": "waterbirds_features", "path": str(features)}}
    )
    traces = tmp_path / "policy_runs" / "fixture"
    traces.mkdir(parents=True)
    (traces / "manifest.json").write_text('{"config": "configs/experiments/synthetic.yaml"}', encoding="utf-8")
    (traces / "latent_clue_packets.jsonl").write_text(
        "\n".join(
            [
                '{"candidate_id":"good","feature_name":"feature_good","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"causal_target":1.0}',
                '{"candidate_id":"bad","feature_name":"feature_bad","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"causal_target":0.0}',
            ]
        ),
        encoding="utf-8",
    )
    (traces / "training_traces.jsonl").write_text(
        "\n".join(
            [
                '{"candidate_id":"good","feature_name":"feature_good","action":"feature_mean_ablation","label_corr":0.9,"env_corr":0.1,"corr_margin":0.8,"abs_corr_margin":0.8,"uncertainty":0.1,"top_group_entropy":0.2,"label_env_disentanglement":0.8,"test_value":1.0,"score_delta":0.5,"hypothesis_correct":true,"passed_control":true}',
                '{"candidate_id":"bad","feature_name":"feature_bad","action":"donor_swap_same_label_diff_env","label_corr":0.1,"env_corr":0.8,"corr_margin":-0.7,"abs_corr_margin":0.7,"uncertainty":0.9,"top_group_entropy":0.8,"label_env_disentanglement":0.1,"test_value":0.0,"score_delta":0.0,"hypothesis_correct":false,"passed_control":false}',
            ]
        ),
        encoding="utf-8",
    )
    (traces / "feature_clues.csv").write_text(
        "feature_name,causal_target\nfeature_good,1.0\nfeature_bad,0.0\n",
        encoding="utf-8",
    )

    rows = build_policy_score_rows(
        bundle,
        policy_input_dir=tmp_path / "policy_runs",
        exclude_datasets=[],
        blend_with_stats_weight=0.5,
        blend_mode="safe_residual",
    )

    by_feature = {row["feature_name"]: row for row in rows}
    assert {row["score_source"] for row in rows} == {"policy_safe"}
    assert float(by_feature["feature_good"]["score"]) > float(by_feature["feature_bad"]["score"])
