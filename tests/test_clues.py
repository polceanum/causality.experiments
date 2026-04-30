from pathlib import Path

from causality_experiments.clues import build_feature_cards, build_language_clue_rows
from causality_experiments.data import load_dataset
from causality_experiments.discovery import (
    DISCOVERY_FEATURE_COLUMNS_V2,
    build_feature_clue_rows,
    clue_feature_vector,
)


def _write_waterbirds_features(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "split,y,place,bird_shape_0,background_0",
                "train,0,0,0.0,0.0",
                "train,0,1,0.0,1.0",
                "train,1,0,1.0,0.0",
                "train,1,1,1.0,1.0",
                "val,0,0,0.0,0.0",
                "val,1,1,1.0,1.0",
                "test,0,1,0.0,1.0",
                "test,1,0,1.0,0.0",
            ]
        ),
        encoding="utf-8",
    )


def _load_waterbirds_bundle(path: Path):
    return load_dataset(
        {
            "dataset": {
                "kind": "waterbirds_features",
                "path": str(path),
                "causal_feature_columns": ["bird_shape_0"],
            }
        }
    )


def test_build_feature_cards_summarizes_label_and_environment_alignment(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    bundle = _load_waterbirds_bundle(csv_path)

    cards = build_feature_cards(bundle, top_k=2)

    by_name = {str(card["feature_name"]): card for card in cards}
    assert by_name["bird_shape_0"]["activation_alignment"] == "label"
    assert by_name["bird_shape_0"]["top_label_rate"] == 1.0
    assert by_name["background_0"]["activation_alignment"] == "environment"
    assert by_name["background_0"]["top_env_rate"] == 1.0


def test_language_clues_score_domain_terms_and_card_statements(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    bundle = _load_waterbirds_bundle(csv_path)
    cards = build_feature_cards(bundle, top_k=2)

    clues = build_language_clue_rows(cards, domain="waterbirds")

    by_name = {str(row["feature_name"]): row for row in clues}
    bird = by_name["bird_shape_0"]
    background = by_name["background_0"]
    assert float(bird["language_causal_score"]) > float(bird["language_spurious_score"])
    assert float(background["language_spurious_score"]) > float(background["language_causal_score"])
    assert float(bird["language_confidence"]) > 0.0
    assert bird["language_prior_source"] == "template:waterbirds"


def test_language_clues_use_activation_alignment_for_opaque_features() -> None:
    clues = build_language_clue_rows(
        [
            {
                "dataset": "waterbirds_features",
                "split": "train",
                "feature_index": 0,
                "feature_name": "feature_0",
                "activation_alignment": "label",
                "activation_label_gap": 0.7,
                "activation_env_gap": 0.1,
                "top_group_entropy": 0.5,
                "label_env_disentanglement": 0.2,
            },
            {
                "dataset": "waterbirds_features",
                "split": "train",
                "feature_index": 1,
                "feature_name": "feature_1",
                "activation_alignment": "environment",
                "activation_label_gap": 0.1,
                "activation_env_gap": 0.7,
                "top_group_entropy": 0.2,
                "label_env_disentanglement": 0.1,
            },
        ],
        domain="waterbirds",
    )
    by_name = {str(row["feature_name"]): row for row in clues}
    assert float(by_name["feature_0"]["language_causal_score"]) > 0.9
    assert float(by_name["feature_0"]["language_confidence"]) > 0.9
    assert float(by_name["feature_1"]["language_spurious_score"]) > 0.9


def test_feature_clue_rows_merge_language_clues_into_v2_vector(tmp_path: Path) -> None:
    csv_path = tmp_path / "features.csv"
    _write_waterbirds_features(csv_path)
    bundle = _load_waterbirds_bundle(csv_path)
    clues = build_language_clue_rows(build_feature_cards(bundle, top_k=2), domain="waterbirds")

    rows = build_feature_clue_rows(bundle, external_clues=clues)
    by_name = {str(row["feature_name"]): row for row in rows}
    bird = by_name["bird_shape_0"]
    vector = clue_feature_vector(bird, DISCOVERY_FEATURE_COLUMNS_V2)

    assert float(bird["language_causal_score"]) > 0.0
    assert len(vector) == len(DISCOVERY_FEATURE_COLUMNS_V2)
    assert vector[DISCOVERY_FEATURE_COLUMNS_V2.index("language_confidence")] > 0.0
