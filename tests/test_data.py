from causality_experiments.data import load_dataset


def test_synthetic_linear_is_deterministic() -> None:
    config = {"seed": 7, "dataset": {"kind": "synthetic_linear", "n": 120}}
    a = load_dataset(config)
    b = load_dataset(config)
    assert a.input_dim == 2
    assert a.split("train")["x"].equal(b.split("train")["x"])
    assert a.causal_mask is not None
    assert a.causal_mask.tolist() == [1.0, 0.0]


def test_all_fixture_datasets_load() -> None:
    kinds = [
        "synthetic_linear",
        "synthetic_nonlinear",
        "dsprites_tiny",
        "causal3d_tiny",
        "waterbirds_tiny",
        "shapes_spurious_tiny",
        "text_toy",
        "fewshot_ner_tiny",
    ]
    for kind in kinds:
        bundle = load_dataset({"seed": 1, "dataset": {"kind": kind, "n": 120}})
        assert set(bundle.splits) == {"train", "val", "test"}
        assert bundle.input_dim > 0
        assert bundle.output_dim >= 2


def test_sequence_fixture_preserves_integer_tokens() -> None:
    bundle = load_dataset({"seed": 1, "dataset": {"kind": "text_toy", "n": 120}})
    x = bundle.split("train")["x"]
    assert bundle.metadata is not None
    assert bundle.metadata["modality"] == "sequence"
    assert float(x.max()) > 1.0
    assert bool((x == x.long().float()).all())


def test_waterbirds_feature_adapter_loads_local_csv(tmp_path) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,y,place,feature_0,feature_1",
                "train,0,0,0.0,1.0",
                "train,1,1,1.0,0.0",
                "val,0,1,0.2,0.8",
                "val,1,0,0.8,0.2",
                "test,0,0,0.1,0.9",
                "test,1,1,0.9,0.1",
            ]
        ),
        encoding="utf-8",
    )
    bundle = load_dataset({"dataset": {"kind": "waterbirds_features", "path": str(csv_path)}})
    assert bundle.name == "waterbirds_features"
    assert bundle.input_dim == 2
    assert bundle.metadata is not None
    assert bundle.metadata["fixture"] is False
    assert bundle.split("train")["group"].tolist() == [0, 3]


def test_waterbirds_feature_adapter_accepts_optional_causal_mask(tmp_path) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,y,place,bird_shape_0,bird_shape_1,background_0",
                "train,0,0,0.0,1.0,0.1",
                "train,1,1,1.0,0.0,0.9",
                "val,0,1,0.2,0.8,0.8",
                "val,1,0,0.8,0.2,0.2",
                "test,0,0,0.1,0.9,0.3",
                "test,1,1,0.9,0.1,0.7",
            ]
        ),
        encoding="utf-8",
    )
    bundle = load_dataset(
        {
            "dataset": {
                "kind": "waterbirds_features",
                "path": str(csv_path),
                "causal_feature_prefixes": ["bird_"],
            }
        }
    )
    assert bundle.causal_mask is not None
    assert bundle.causal_mask.tolist() == [1.0, 1.0, 0.0]
    assert bundle.metadata is not None
    assert bundle.metadata["causal_feature_columns"] == ["bird_shape_0", "bird_shape_1"]


def test_waterbirds_feature_adapter_can_derive_causal_mask_from_train_split(tmp_path) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,y,place,feature_0,feature_1,feature_2",
                "train,0,0,0.0,0.0,0.1",
                "train,0,1,0.1,1.0,0.8",
                "train,1,0,1.0,0.0,0.2",
                "train,1,1,0.9,1.0,0.9",
                "val,0,0,0.2,0.0,0.2",
                "val,1,1,0.8,1.0,0.8",
                "test,0,1,0.1,1.0,0.7",
                "test,1,0,0.9,0.0,0.3",
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
                "causal_mask_min_margin": 0.05,
                "causal_mask_top_k": 1,
            }
        }
    )
    assert bundle.causal_mask is not None
    assert bundle.causal_mask.tolist() == [1.0, 0.0, 0.0]
    assert bundle.metadata is not None
    assert bundle.metadata["causal_mask_strategy"] == "label_minus_env_correlation"
    assert bundle.metadata["causal_feature_columns"] == ["feature_0"]


def test_waterbirds_feature_adapter_accepts_discovery_scores_mask(tmp_path) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,y,place,feature_0,feature_1",
                "train,0,0,0.0,1.0",
                "train,1,1,1.0,0.0",
                "val,0,1,0.2,0.8",
                "val,1,0,0.8,0.2",
                "test,0,0,0.1,0.9",
                "test,1,1,0.9,0.1",
            ]
        ),
        encoding="utf-8",
    )
    score_path = tmp_path / "scores.csv"
    score_path.write_text(
        "dataset,feature_index,feature_name,score\nwaterbirds_features,0,feature_0,0.9\nwaterbirds_features,1,feature_1,0.1\n",
        encoding="utf-8",
    )
    bundle = load_dataset(
        {
            "dataset": {
                "kind": "waterbirds_features",
                "path": str(csv_path),
                "causal_mask_strategy": "discovery_scores",
                "discovery_scores_path": str(score_path),
                "discovery_score_threshold": 0.5,
                "discovery_score_top_k": 1,
            }
        }
    )
    assert bundle.causal_mask is not None
    assert bundle.causal_mask.tolist() == [1.0, 0.0]
    assert bundle.metadata is not None
    assert bundle.metadata["causal_feature_scores"] == [0.9, 0.1]


def test_waterbirds_feature_adapter_discovery_top_k_overrides_threshold(tmp_path) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,y,place,feature_0,feature_1",
                "train,0,0,0.0,1.0",
                "train,1,1,1.0,0.0",
                "val,0,1,0.2,0.8",
                "val,1,0,0.8,0.2",
                "test,0,0,0.1,0.9",
                "test,1,1,0.9,0.1",
            ]
        ),
        encoding="utf-8",
    )
    score_path = tmp_path / "scores.csv"
    score_path.write_text(
        "dataset,feature_index,feature_name,score\nwaterbirds_features,0,feature_0,0.9\nwaterbirds_features,1,feature_1,0.1\n",
        encoding="utf-8",
    )
    bundle = load_dataset(
        {
            "dataset": {
                "kind": "waterbirds_features",
                "path": str(csv_path),
                "causal_mask_strategy": "discovery_scores",
                "discovery_scores_path": str(score_path),
                "discovery_score_threshold": 0.95,
                "discovery_score_top_k": 1,
            }
        }
    )
    assert bundle.causal_mask is not None
    assert bundle.causal_mask.tolist() == [1.0, 0.0]


def test_waterbirds_feature_adapter_accepts_random_top_k_mask(tmp_path) -> None:
    csv_path = tmp_path / "features.csv"
    csv_path.write_text(
        "\n".join(
            [
                "split,y,place,feature_0,feature_1,feature_2",
                "train,0,0,0.0,1.0,0.5",
                "train,1,1,1.0,0.0,0.4",
                "val,0,1,0.2,0.8,0.6",
                "val,1,0,0.8,0.2,0.3",
                "test,0,0,0.1,0.9,0.7",
                "test,1,1,0.9,0.1,0.2",
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "dataset": {
            "kind": "waterbirds_features",
            "path": str(csv_path),
            "causal_mask_strategy": "random_top_k",
            "causal_mask_top_k": 2,
            "causal_mask_random_seed": 11,
        }
    }
    bundle_a = load_dataset(config)
    bundle_b = load_dataset(config)
    assert bundle_a.causal_mask is not None
    assert bundle_b.causal_mask is not None
    assert bundle_a.causal_mask.equal(bundle_b.causal_mask)
    assert int(bundle_a.causal_mask.sum().item()) == 2
