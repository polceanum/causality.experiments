from pathlib import Path

import pandas as pd
import torch
import yaml

from scripts.prepare_waterbirds_features import build_feature_frame, update_benchmark_config


def test_build_feature_frame_maps_metadata_and_group() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["train", "test"],
            "y": [0, 1],
            "place": [1, 0],
            "group": [2, 1],
            "img_filename": ["a.jpg", "b.jpg"],
            "place_filename": ["/o/ocean.jpg", "/b/forest.jpg"],
        }
    )
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    frame = build_feature_frame(metadata, features)
    assert list(frame.columns[:6]) == [
        "split",
        "y",
        "place",
        "group",
        "img_filename",
        "place_filename",
    ]
    assert frame["feature_0"].tolist() == [1.0, 3.0]
    assert frame["feature_1"].tolist() == [2.0, 4.0]


def test_update_benchmark_config_writes_provenance(tmp_path: Path) -> None:
    config_path = tmp_path / "waterbirds_features.yaml"
    config_path.write_text(
        """
name: waterbirds_features
benchmark:
  kind: real
  id: waterbirds
  comparable_to_literature: true
dataset:
  kind: waterbirds_features
  path: data/waterbirds/features.csv
""",
        encoding="utf-8",
    )
    update_benchmark_config(
        config_path,
        features_path=Path("data/waterbirds/features.csv"),
        feature_extractor="torchvision_resnet50_imagenet1k_v2_penultimate",
        feature_source="local featurization",
        split_definition="official Stanford Waterbirds split",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["dataset"]["path"] == "data/waterbirds/features.csv"
    assert config["benchmark"]["provenance"]["feature_extractor"] == "torchvision_resnet50_imagenet1k_v2_penultimate"
    assert config["benchmark"]["provenance"]["feature_source"] == "local featurization"