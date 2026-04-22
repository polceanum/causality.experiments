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
