import torch

from causality_experiments.methods import OfficialDFRClassifier
from scripts.report_waterbirds_patch_flip_probe import (
    HiddenBundle,
    PatchFlipMixtureProbe,
    adapted_probe_logits,
    build_component_feature_rows,
    build_component_bundle,
    build_excess_feature_score_rows,
    build_intervention_feature_variants,
    build_intervention_feature_score_rows,
    compact_official_details,
    component_feature_names,
    evaluate_effect_weighted_feature_bundles,
    evaluate_intervention_strategy,
    mixture_component_diversity_loss,
    mixture_mask_scores,
    normalized_patch_prior_scores,
    _mixture_probe_loss,
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


def test_mixture_probe_emits_posterior_mask_scores() -> None:
    hidden = torch.randn(3, 5, 2)
    probe = PatchFlipMixtureProbe(token_dim=2, component_count=3, hidden_dim=4, initial_mask_probability=0.2)

    mask_logits, component_logits = probe(hidden)
    marginal_scores = mixture_mask_scores(
        probe,
        hidden,
        mode="marginal",
        prior_selector="none",
        prior_weight=0.0,
        budget=0.2,
    )

    assert mask_logits.shape == (3, 3, 4)
    assert component_logits.shape == (3, 3)
    assert marginal_scores.shape == (3, 4)
    assert torch.all((marginal_scores >= 0.0) & (marginal_scores <= 1.0))


def test_mixture_component_diversity_penalizes_overlap() -> None:
    identical = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    different = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    assert mixture_component_diversity_loss(identical).item() > mixture_component_diversity_loss(different).item()


def test_mixture_probe_loss_can_optimize_effect_best_component() -> None:
    baseline_logits = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
    edited_logits_by_component = torch.tensor(
        [
            [[0.0, 1.0], [0.0, -1.0]],
            [[2.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask_weights = torch.full((2, 2, 3), 0.25)
    component_logits = torch.tensor([[0.0, 2.0], [0.0, 2.0]])

    loss, parts = _mixture_probe_loss(
        edited_logits_by_component,
        torch.tensor([0, 1]),
        mask_weights,
        component_logits,
        baseline_logits=baseline_logits,
        mixture_objective="effect_best",
        mixture_effect_weight=1.0,
        mixture_routing_weight=0.1,
        mixture_best_of_k_temperature=1.0,
        sparsity_weight=0.0,
        budget=0.25,
        budget_weight=0.0,
        entropy_weight=0.0,
        mixture_entropy_weight=0.0,
        mixture_diversity_weight=0.0,
    )

    assert parts["best_effect_drop"] == 3.0
    assert parts["routing_loss"] < 0.2
    assert loss.item() < parts["selected_flip_loss"]


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


def test_evaluate_intervention_strategy_accepts_mixture_component() -> None:
    hidden = torch.randn(2, 5, 2)
    classifier = OfficialDFRClassifier(
        weight=torch.tensor([[0.0] * 8, [1.0, 0.0, 0.5, 0.0, -0.5, 0.0, 0.25, 0.0]]),
        bias=torch.zeros(2),
        output_dim=2,
    )
    probe = PatchFlipMixtureProbe(token_dim=2, component_count=2, hidden_dim=4, initial_mask_probability=0.25)

    row = evaluate_intervention_strategy(
        strategy="mixture_val_best_component",
        hidden=hidden,
        labels=torch.tensor([1, 0]),
        groups=torch.tensor([0, 3]),
        classifier=classifier,
        pooling="cls_similarity",
        top_k=1,
        replacement="zero",
        probe=probe,
        mixture_component_index=1,
    )

    assert row["strategy"] == "mixture_val_best_component"
    assert row["mixture_component_index"] == 1
    assert row["mean_mask_fraction"] == 0.25


def test_evaluate_intervention_strategy_can_select_best_effect_component() -> None:
    hidden = torch.randn(2, 5, 2)
    classifier = OfficialDFRClassifier(
        weight=torch.tensor([[0.0] * 8, [1.0, 0.0, 0.5, 0.0, -0.5, 0.0, 0.25, 0.0]]),
        bias=torch.zeros(2),
        output_dim=2,
    )
    probe = PatchFlipMixtureProbe(token_dim=2, component_count=2, hidden_dim=4, initial_mask_probability=0.25)

    row = evaluate_intervention_strategy(
        strategy="mixture_effect_best_component",
        hidden=hidden,
        labels=torch.tensor([1, 0]),
        groups=torch.tensor([0, 3]),
        classifier=classifier,
        pooling="cls_similarity",
        top_k=1,
        replacement="zero",
        probe=probe,
    )

    assert row["strategy"] == "mixture_effect_best_component"
    assert "mean_mixture_component_index" in row
    assert row["mean_mask_fraction"] == 0.25


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


def test_build_intervention_feature_variants_emits_counterfactual_tables() -> None:
    hidden = {split: torch.randn(2, 5, 2) for split in ("train", "val", "test")}
    labels = {split: torch.tensor([1, 0]) for split in hidden}
    groups = {split: torch.tensor([0, 3]) for split in hidden}
    classifier = OfficialDFRClassifier(
        weight=torch.tensor([[0.0] * 8, [1.0, 0.0, 0.5, 0.0, -0.5, 0.0, 0.25, 0.0]]),
        bias=torch.zeros(2),
        output_dim=2,
    )
    probe = PatchFlipMixtureProbe(token_dim=2, component_count=2, hidden_dim=4, initial_mask_probability=0.25)

    bundles, rows_by_variant, feature_names_by_variant = build_intervention_feature_variants(
        hidden_bundle=HiddenBundle(hidden=hidden, labels=labels, groups=groups),
        classifier=classifier,
        pooling="cls_similarity",
        top_k=1,
        replacement="zero",
        strategy="mixture_effect_best_component",
        probe=probe,
        budget=0.25,
    )

    assert bundles["original"].input_dim == 8
    assert bundles["edited"].input_dim == 8
    assert bundles["delta"].input_dim == 8
    assert bundles["original_plus_delta"].input_dim == 16
    assert bundles["original_plus_edited"].input_dim == 16
    assert bundles["all_views"].input_dim == 24
    assert feature_names_by_variant["delta"][0] == "feature_delta_cls_0000"
    assert rows_by_variant["edited"][0]["split"] == "train"
    assert "intervention_effect_drop" in rows_by_variant["edited"][0]
    assert "selected_component_index" in rows_by_variant["edited"][0]
    assert "intervention_effect_drop" in bundles["original"].split("val")


def test_evaluate_effect_weighted_feature_bundles_reports_controls() -> None:
    hidden = {split: torch.randn(8, 5, 2) for split in ("train", "val", "test")}
    labels = {split: torch.tensor([0, 0, 1, 1, 0, 1, 0, 1]) for split in hidden}
    groups = {split: torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]) for split in hidden}
    classifier = OfficialDFRClassifier(
        weight=torch.tensor([[0.0] * 8, [1.0, 0.0, 0.5, 0.0, -0.5, 0.0, 0.25, 0.0]]),
        bias=torch.zeros(2),
        output_dim=2,
    )
    probe = PatchFlipMixtureProbe(token_dim=2, component_count=2, hidden_dim=4, initial_mask_probability=0.25)
    bundles, _, _ = build_intervention_feature_variants(
        hidden_bundle=HiddenBundle(hidden=hidden, labels=labels, groups=groups),
        classifier=classifier,
        pooling="cls_similarity",
        top_k=1,
        replacement="zero",
        strategy="mixture_effect_best_component",
        probe=probe,
        budget=0.25,
    )
    config = {
        "seed": 7,
        "method": {
            "kind": "official_dfr_val_tr",
            "official_dfr_c_grid": [1.0],
            "official_dfr_num_retrains": 1,
            "official_dfr_balance_val": False,
        },
    }

    rows = evaluate_effect_weighted_feature_bundles(
        bundles=bundles,
        config=config,
        variants=["original"],
        scales=[2.0],
        seed=7,
    )

    assert {row["weight_mode"] for row in rows} == {"effect", "random", "inverse_effect"}
    assert all(row["weight_scale"] == 2.0 for row in rows)
    assert all(row["variant"] == "original" for row in rows)


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