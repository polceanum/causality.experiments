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
