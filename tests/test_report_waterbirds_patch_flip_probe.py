import torch

from causality_experiments.methods import OfficialDFRClassifier
from scripts.report_waterbirds_patch_flip_probe import (
    HiddenBundle,
    adapted_probe_logits,
    build_component_feature_rows,
    build_component_bundle,
    build_excess_feature_score_rows,
    build_intervention_feature_score_rows,
    compact_official_details,
    component_feature_names,
    evaluate_intervention_strategy,
    normalized_patch_prior_scores,
)
from causality_experiments.patch_interventions import PatchFlipProbe


def test_build_component_bundle_pools_hidden_splits() -> None:
    hidden = {
        split: torch.randn(4, 5, 2)
        for split in ("train", "val", "test")
    }
    labels = {split: torch.tensor([0, 1, 0, 1]) for split in hidden}
    groups = {split: torch.tensor([0, 1, 2, 3]) for split in hidden}

    bundle = build_component_bundle(HiddenBundle(hidden=hidden, labels=labels, groups=groups), pooling="cls_similarity")

    assert bundle.input_dim == 8
    assert bundle.output_dim == 2
    assert bundle.split("test")["x"].shape == (4, 8)
    assert bundle.split("test")["env"].tolist() == [0, 0, 1, 1]
    assert bundle.metadata["feature_columns"] == component_feature_names(8, pooling="cls_similarity")


def test_build_component_feature_rows_uses_discovery_feature_names() -> None:
    hidden = {split: torch.ones(2, 5, 2) for split in ("train", "val", "test")}
    labels = {split: torch.tensor([0, 1]) for split in hidden}
    groups = {split: torch.tensor([0, 3]) for split in hidden}

    rows, feature_names = build_component_feature_rows(
        HiddenBundle(hidden=hidden, labels=labels, groups=groups),
        pooling="cls_similarity",
    )

    assert feature_names == [
        "feature_cls_0000",
        "feature_cls_0001",
        "feature_foreground_0000",
        "feature_foreground_0001",
        "feature_background_0000",
        "feature_background_0001",
        "feature_foreground_minus_background_0000",
        "feature_foreground_minus_background_0001",
    ]
    assert rows[0]["split"] == "train"
    assert rows[0]["place"] == 0
    assert rows[-1]["split"] == "test"
    assert rows[-1]["place"] == 1


def test_adapted_probe_logits_can_use_patch_prior() -> None:
    hidden = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    probe = PatchFlipProbe(token_dim=2, initial_mask_probability=0.25)

    prior_scores = normalized_patch_prior_scores(hidden, "cls_similarity")
    logits = adapted_probe_logits(
        probe,
        hidden,
        prior_selector="cls_similarity",
        prior_weight=1.0,
        budget=0.25,
    )

    assert prior_scores.argmax(dim=1).tolist() == [0]
    assert logits.argmax(dim=1).tolist() == [0]


def test_evaluate_intervention_strategy_reports_flat_group_metrics() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    classifier = OfficialDFRClassifier(
        weight=torch.tensor([[0.0] * 8, [1.0, 0.0, 0.5, 0.0, -0.5, 0.0, 0.25, 0.0]]),
        bias=torch.zeros(2),
        output_dim=2,
        details={"official_dfr_best_c": 0.7, "official_dfr_retrains": ["large"]},
    )

    row = evaluate_intervention_strategy(
        strategy="cls_similarity_top",
        hidden=hidden,
        labels=torch.tensor([1, 0]),
        groups=torch.tensor([0, 3]),
        classifier=classifier,
        pooling="cls_similarity",
        top_k=1,
        replacement="zero",
    )

    assert row["strategy"] == "cls_similarity_top"
    assert row["mean_mask_fraction"] == 0.25
    assert "label_group_mean_target_logit_delta_group_0" in row
    assert "decision_group_mean_target_logit_delta_group_3" in row
    assert compact_official_details(classifier) == {"official_dfr_best_c": 0.7}


def test_build_intervention_feature_score_rows_emits_discovery_schema() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    classifier = OfficialDFRClassifier(
        weight=torch.tensor([[0.0] * 8, [1.0, 0.0, 0.5, 0.0, -0.5, 0.0, 0.25, 0.0]]),
        bias=torch.zeros(2),
        output_dim=2,
    )

    rows = build_intervention_feature_score_rows(
        strategy="cls_similarity_top",
        hidden=hidden,
        classifier=classifier,
        pooling="cls_similarity",
        top_k=1,
        replacement="zero",
    )

    assert len(rows) == 8
    assert rows[0]["feature_name"] == "feature_cls_0000"
    assert rows[0]["score"]
    assert rows[0]["strategy"] == "cls_similarity_top"


def test_build_excess_feature_score_rows_subtracts_control_raw_scores() -> None:
    primary = [
        {"feature_name": "feature_0", "raw_score": "0.900000000", "score": "1.000000", "strategy": "learned"},
        {"feature_name": "feature_1", "raw_score": "0.100000000", "score": "0.111111", "strategy": "learned"},
    ]
    control = [
        {"feature_name": "feature_0", "raw_score": "0.400000000"},
        {"feature_name": "feature_1", "raw_score": "0.200000000"},
    ]

    rows = build_excess_feature_score_rows(primary, control, strategy="learned_excess_random")

    assert rows[0]["raw_score"] == "0.500000000"
    assert rows[0]["score"] == "1.000000"
    assert rows[1]["raw_score"] == "0.000000000"
    assert rows[1]["score"] == "0.000000"
    assert rows[0]["strategy"] == "learned_excess_random"