from pathlib import Path

from scripts.run_waterbirds_clue_fusion_sweep import (
    build_downstream_candidate,
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
