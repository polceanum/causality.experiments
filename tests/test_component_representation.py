from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from causality_experiments.component_representation import (
    build_component_clue_rows,
    columns_for_components,
    component_summary_rows,
    component_test_rows,
    feature_columns_from_frame,
    filter_feature_components,
    infer_feature_components,
    ordered_feature_components,
    train_component_adapter,
)
from causality_experiments.data import load_dataset
from scripts.build_waterbirds_component_features import build_component_feature_artifacts
from scripts.train_waterbirds_component_adapter import train_adapter_artifact


def _toy_component_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        for label, place in ((0, 0), (1, 1), (1, 0), (0, 1)):
            for replica in range(4):
                rows.append(
                    {
                        "split": split,
                        "y": label,
                        "place": place,
                        "group": place * 2 + label,
                        "feature_foreground_0000": float(label) + 0.01 * replica,
                        "feature_foreground_0001": float(label) - 0.01 * replica,
                        "feature_background_0000": float(place) + 0.01 * replica,
                        "feature_background_0001": float(place) - 0.01 * replica,
                    }
                )
    return pd.DataFrame(rows)


def test_infer_feature_components_uses_manifest_and_prefixes() -> None:
    columns = ["feature_foreground_0000", "feature_background_0000", "feature_0"]
    manifest = {"foreground": ["feature_foreground_0000"], "missing": ["feature_missing_0000"]}

    assert infer_feature_components(columns, manifest) == {
        "foreground": ["feature_foreground_0000"],
        "global": ["feature_background_0000", "feature_0"],
    }
    assert infer_feature_components(columns) == {
        "foreground": ["feature_foreground_0000"],
        "background": ["feature_background_0000"],
        "global": ["feature_0"],
    }


def test_filter_feature_components_keeps_requested_columns() -> None:
    components = {
        "foreground": ["feature_foreground_0000"],
        "background": ["feature_background_0000"],
    }

    filtered = filter_feature_components(components, include_components=("foreground",))

    assert filtered == {"foreground": ["feature_foreground_0000"]}
    assert columns_for_components(filtered) == ["feature_foreground_0000"]


def test_ordered_feature_components_recovers_generic_component_columns() -> None:
    components = ordered_feature_components(
        ["feature_0", "feature_1", "feature_2", "feature_3"],
        ["cls", "center"],
    )

    assert components == {
        "cls": ["feature_0", "feature_1"],
        "center": ["feature_2", "feature_3"],
    }


def test_component_clues_rank_foreground_over_background(tmp_path: Path) -> None:
    frame = _toy_component_frame()
    csv_path = tmp_path / "features.csv"
    frame.to_csv(csv_path, index=False)
    bundle = load_dataset({"dataset": {"kind": "waterbirds_features", "path": str(csv_path)}})
    components = infer_feature_components(feature_columns_from_frame(frame))

    summaries = component_summary_rows(bundle, feature_components=components)
    clues = build_component_clue_rows(bundle, feature_components=components)
    tests = component_test_rows(bundle, feature_components=components)

    by_component = {row["component_group"]: row for row in summaries}
    assert by_component["foreground"]["component_causal_score"] > by_component["background"]["component_causal_score"]
    foreground_clues = [row for row in clues if row["component_group"] == "foreground"]
    background_clues = [row for row in clues if row["component_group"] == "background"]
    assert min(float(row["adapter_prior"]) for row in foreground_clues) > max(
        float(row["adapter_prior"]) for row in background_clues
    )
    assert any(row["component_group"] == "foreground" and row["test_passed_control"] for row in tests)


def test_build_component_feature_artifacts_writes_manifest_and_clues(tmp_path: Path) -> None:
    input_csv = tmp_path / "features.csv"
    output_csv = tmp_path / "component_features.csv"
    clues_csv = tmp_path / "component_clues.csv"
    summary_csv = tmp_path / "component_summary.csv"
    _toy_component_frame().to_csv(input_csv, index=False)

    result = build_component_feature_artifacts(
        input_csv=input_csv,
        output_csv=output_csv,
        output_clues_csv=clues_csv,
        output_summary_csv=summary_csv,
        include_components=("foreground",),
    )

    manifest = json.loads(output_csv.with_suffix(output_csv.suffix + ".manifest.json").read_text(encoding="utf-8"))
    assert result["component_count"] == 1
    assert manifest["feature_components"]["foreground"] == ["feature_foreground_0000", "feature_foreground_0001"]
    clues = pd.read_csv(clues_csv)
    assert set(clues["component_group"]) == {"foreground"}
    assert summary_csv.exists()


def test_component_adapter_suppresses_shortcut_feature(tmp_path: Path) -> None:
    frame = _toy_component_frame()
    clues = [
        {
            "feature_name": "feature_foreground_0000",
            "adapter_prior": "0.95",
        },
        {
            "feature_name": "feature_foreground_0001",
            "adapter_prior": "0.95",
        },
        {
            "feature_name": "feature_background_0000",
            "adapter_prior": "0.05",
        },
        {
            "feature_name": "feature_background_0001",
            "adapter_prior": "0.05",
        },
    ]

    result = train_component_adapter(
        frame,
        clue_rows=clues,
        epochs=80,
        lr=0.03,
        env_penalty_weight=0.5,
        env_adversary_weight=0.05,
        clue_prior_weight=4.0,
        seed=4,
    )

    weights = result.report["feature_weights"]
    assert weights["feature_foreground_0000"] > weights["feature_background_0000"]
    assert weights["feature_foreground_0001"] > weights["feature_background_0001"]
    assert result.report["train_accuracy"] >= 0.9
    assert "train_env_accuracy" in result.report
    assert all(column.startswith("feature_adapted_") for column in result.report["adapted_feature_columns"])


def test_train_adapter_artifact_writes_dfr_compatible_feature_table(tmp_path: Path) -> None:
    input_csv = tmp_path / "features.csv"
    clue_csv = tmp_path / "clues.csv"
    output_csv = tmp_path / "adapted.csv"
    output_json = tmp_path / "adapted.json"
    frame = _toy_component_frame()
    frame.to_csv(input_csv, index=False)
    pd.DataFrame(
        [
            {"feature_name": "feature_foreground_0000", "adapter_prior": "0.95"},
            {"feature_name": "feature_foreground_0001", "adapter_prior": "0.95"},
            {"feature_name": "feature_background_0000", "adapter_prior": "0.05"},
            {"feature_name": "feature_background_0001", "adapter_prior": "0.05"},
        ]
    ).to_csv(clue_csv, index=False)

    report = train_adapter_artifact(
        input_csv=input_csv,
        output_csv=output_csv,
        output_json=output_json,
        clue_csv=clue_csv,
        epochs=20,
        clue_prior_weight=4.0,
    )

    adapted = pd.read_csv(output_csv)
    adapted_bundle = load_dataset({"dataset": {"kind": "waterbirds_features", "path": str(output_csv)}})
    manifest = json.loads(output_csv.with_suffix(output_csv.suffix + ".manifest.json").read_text(encoding="utf-8"))
    assert {"split", "y", "place", "group"}.issubset(adapted.columns)
    assert any(column.startswith("feature_adapted_") for column in adapted.columns)
    assert adapted_bundle.input_dim == 4
    assert report["output_feature_count"] == 4
    assert manifest["feature_components"]["adapted"] == report["adapted_feature_columns"]
