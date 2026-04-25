from scripts.tune_discovery_mask import _build_candidate


def test_build_candidate_uses_top_k_without_zero_threshold() -> None:
    candidate = _build_candidate(
        {
            "name": "base_experiment",
            "seed": 17,
            "dataset": {
                "kind": "waterbirds_features",
                "path": "data/waterbirds/features.csv",
            },
        },
        score_path="outputs/runs/waterbirds-feature-discovery-scores.csv",
        top_k=128,
        variant="discovery_full",
    )
    dataset = candidate["dataset"]
    assert dataset["causal_mask_strategy"] == "discovery_scores"
    assert dataset["discovery_scores_path"] == "outputs/runs/waterbirds-feature-discovery-scores.csv"
    assert dataset["discovery_score_top_k"] == 128
    assert dataset["discovery_score_threshold"] > 1.0


def test_build_candidate_can_emit_heuristic_control() -> None:
    candidate = _build_candidate(
        {
            "name": "base_experiment",
            "dataset": {
                "kind": "waterbirds_features",
                "path": "data/waterbirds/features.csv",
                "causal_mask_strategy": "label_minus_env_correlation",
                "causal_mask_min_margin": 0.01,
                "causal_mask_top_k": 512,
            },
        },
        top_k=64,
        variant="heuristic",
    )
    dataset = candidate["dataset"]
    assert dataset["causal_mask_strategy"] == "label_minus_env_correlation"
    assert dataset["causal_mask_top_k"] == 64
    assert "discovery_scores_path" not in dataset


def test_build_candidate_can_emit_random_control() -> None:
    candidate = _build_candidate(
        {
            "name": "base_experiment",
            "seed": 17,
            "dataset": {
                "kind": "waterbirds_features",
                "path": "data/waterbirds/features.csv",
                "causal_mask_strategy": "label_minus_env_correlation",
            },
        },
        top_k=64,
        variant="random",
    )
    dataset = candidate["dataset"]
    assert dataset["causal_mask_strategy"] == "random_top_k"
    assert dataset["causal_mask_top_k"] == 64
    assert dataset["causal_mask_random_seed"] == 17