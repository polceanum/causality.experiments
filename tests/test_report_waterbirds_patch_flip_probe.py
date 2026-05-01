import torch

from causality_experiments.methods import OfficialDFRClassifier
from scripts.report_waterbirds_patch_flip_probe import (
    HiddenBundle,
    build_component_bundle,
    compact_official_details,
    evaluate_intervention_strategy,
)


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