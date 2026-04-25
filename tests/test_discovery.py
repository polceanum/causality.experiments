from pathlib import Path

from causality_experiments.data import load_dataset
from causality_experiments.discovery import (
    aggregate_soft_causal_target,
    build_feature_clue_rows,
    combine_discovery_scores,
)
from scripts.score_discovery_model import _apply_support_restriction
from scripts.train_discovery_model import _pairwise_ranking_loss


def test_build_feature_clue_rows_uses_ground_truth_mask() -> None:
    bundle = load_dataset({"seed": 3, "dataset": {"kind": "synthetic_linear", "n": 120}})
    rows = build_feature_clue_rows(bundle)
    assert len(rows) == 2
    assert rows[0]["has_ground_truth_mask"] is True
    assert rows[0]["has_explicit_supervision"] is True
    assert rows[0]["supervision_source"] == "explicit_mask"
    assert rows[0]["causal_target"] == 1.0
    assert rows[1]["causal_target"] == 0.0
    assert rows[0]["corr_margin"] > rows[1]["corr_margin"]


def test_build_feature_clue_rows_uses_feature_names_from_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,y,place,feature_a,feature_b",
                "train,0,0,0.0,0.0",
                "train,1,1,1.0,1.0",
                "train,1,0,0.9,0.0",
                "train,0,1,0.1,1.0",
                "val,0,0,0.0,0.0",
                "test,1,1,1.0,1.0",
            ]
        ),
        encoding="utf-8",
    )
    bundle = load_dataset(
        {
            "dataset": {
                "kind": "waterbirds_features",
                "path": str(csv_path),
                "causal_feature_columns": ["feature_a"],
            }
        }
    )
    rows = build_feature_clue_rows(bundle)
    assert [row["feature_name"] for row in rows] == ["feature_a", "feature_b"]
    assert rows[0]["causal_target"] == 1.0
    assert rows[1]["causal_target"] == 0.0
    assert rows[0]["modality_features"] == 1.0
    assert rows[0]["feature_index_frac"] == 0.0
    assert rows[1]["feature_index_frac"] == 1.0


def test_soft_target_uses_margin_without_ground_truth() -> None:
    score = aggregate_soft_causal_target(
        {
            "has_ground_truth_mask": False,
            "has_cause_position": False,
            "corr_margin": 0.3,
        }
    )
    assert 0.5 < score < 1.0


def test_waterbirds_derived_mask_is_not_treated_as_explicit_supervision(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,y,place,feature_0,feature_1",
                "train,0,0,0.0,1.0",
                "train,1,1,1.0,0.0",
                "train,1,0,0.9,0.1",
                "train,0,1,0.1,0.9",
                "val,0,0,0.0,1.0",
                "test,1,1,1.0,0.0",
            ]
        ),
        encoding="utf-8",
    )
    bundle = load_dataset(
        {
            "dataset": {
                "kind": "waterbirds_features",
                "path": str(csv_path),
                "causal_mask_strategy": "label_minus_env_correlation",
                "causal_mask_top_k": 1,
            }
        }
    )
    rows = build_feature_clue_rows(bundle)
    assert rows[0]["has_ground_truth_mask"] is True
    assert rows[0]["has_explicit_supervision"] is False
    assert rows[0]["supervision_source"] == "derived_mask"
    assert rows[0]["supervision_derived"] == 1.0
    assert rows[0]["supervision_explicit"] == 0.0


def test_pairwise_ranking_loss_prefers_higher_positive_logits() -> None:
    good_logits = __import__("torch").tensor([[1.0], [-1.0]], dtype=__import__("torch").float32)
    bad_logits = __import__("torch").tensor([[-1.0], [1.0]], dtype=__import__("torch").float32)
    labels = __import__("torch").tensor([[1.0], [0.0]], dtype=__import__("torch").float32)
    datasets = ["synthetic_linear", "synthetic_linear"]
    assert _pairwise_ranking_loss(good_logits, labels, datasets, margin=0.2).item() == 0.0
    assert _pairwise_ranking_loss(bad_logits, labels, datasets, margin=0.2).item() > 0.0


def test_support_restriction_zeroes_scores_outside_allowed_features() -> None:
    rows = [
        {"feature_name": "feature_0", "score": "0.700000"},
        {"feature_name": "feature_1", "score": "0.400000"},
    ]
    restricted = _apply_support_restriction(rows, {"feature_1"}, outside_score=-1.0)
    assert restricted[0]["score"] == "-1.000000"
    assert restricted[1]["score"] == "0.400000"


def test_combined_discovery_score_uses_support_gate() -> None:
    import torch

    rank_logits = torch.tensor([[4.0], [4.0]], dtype=torch.float32)
    support_logits = torch.tensor([[4.0], [-4.0]], dtype=torch.float32)
    scores = combine_discovery_scores(rank_logits, support_logits).squeeze(1)
    assert scores[0].item() > 0.9
    assert scores[1].item() < 0.1