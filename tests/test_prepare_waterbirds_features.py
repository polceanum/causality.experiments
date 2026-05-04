from pathlib import Path

import pandas as pd
from PIL import Image
import torch
import yaml

from scripts.prepare_waterbirds_features import (
    _active_erm_sample_mode,
    _balanced_group_sample_weights,
    _canonical_erm_sample_mode,
    _cross_env_supervised_contrastive_loss,
    _erm_finetune_sample_weights,
    _feature_columns_for_components,
    _feature_component_names,
    _hf_patch_component_features,
    _hf_patch_center_background_features,
    _object_proxy_views,
    _load_prepared_artifact,
    _store_prepared_artifact,
    _set_trainable_layers,
    _stratified_metadata_limit,
    _waterbirds_contrastive_pair_masks,
    build_feature_frame,
    build_train_transform,
    evaluate_waterbirds_model,
    extract_feature_matrix_from_model,
    PreparedWaterbirdsFeatures,
    update_benchmark_config,
)
from scripts import prepare_waterbirds_features


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


def test_build_feature_frame_can_use_component_feature_names() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["train"],
            "y": [1],
            "place": [0],
            "group": [1],
            "img_filename": ["a.jpg"],
            "place_filename": ["p.jpg"],
        }
    )
    features = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    columns, groups = _feature_columns_for_components(4, ["cls", "foreground"])

    frame = build_feature_frame(metadata, features, feature_columns=columns)

    assert columns == ["feature_cls_0000", "feature_cls_0001", "feature_foreground_0000", "feature_foreground_0001"]
    assert groups == {
        "cls": ["feature_cls_0000", "feature_cls_0001"],
        "foreground": ["feature_foreground_0000", "feature_foreground_0001"],
    }
    assert frame["feature_foreground_0001"].tolist() == [4.0]


def test_feature_component_names_reflect_patch_pooling_modes() -> None:
    assert _feature_component_names(
        resolved_settings={"feature_decomposition": "center_background", "backbone_name": "resnet50"},
        extractor_name="resnet_decompcenterbg",
    ) == ["full", "center", "background", "center_minus_background"]
    assert _feature_component_names(
        resolved_settings={"feature_decomposition": "object_proxy", "backbone_name": "resnet50"},
        extractor_name="resnet_decompobjectproxy",
    ) == ["full", "foreground", "background", "foreground_minus_background"]
    assert _feature_component_names(
        resolved_settings={"feature_decomposition": "none", "backbone_name": "hf_patch_cls_components"},
        extractor_name="hf_patch_cls_similarity_penultimate",
    ) == ["cls", "foreground", "background", "foreground_minus_background"]
    assert _feature_component_names(
        resolved_settings={"feature_decomposition": "none", "backbone_name": "hf_patch_center_norm_components"},
        extractor_name="hf_patch_center_norm_penultimate",
    ) == ["cls", "foreground", "background", "foreground_minus_background"]


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


def test_set_trainable_layers_supports_head_and_layer4_modes() -> None:
    model = torch.nn.Module()
    model.layer1 = torch.nn.Linear(2, 2)
    model.layer4 = torch.nn.Linear(2, 2)
    model.fc = torch.nn.Linear(2, 2)

    _set_trainable_layers(model, "head")
    assert not any(parameter.requires_grad for parameter in model.layer1.parameters())
    assert not any(parameter.requires_grad for parameter in model.layer4.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())

    _set_trainable_layers(model, "layer4")
    assert not any(parameter.requires_grad for parameter in model.layer1.parameters())
    assert all(parameter.requires_grad for parameter in model.layer4.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())


def test_build_train_transform_returns_image_tensor() -> None:
    transform = build_train_transform(augment=True)
    image = Image.new("RGB", (256, 256), color=(128, 128, 128))
    tensor = transform(image)
    assert tuple(tensor.shape) == (3, 224, 224)


def test_balanced_group_sample_weights_equalize_group_mass() -> None:
    metadata = pd.DataFrame({"group": [0, 0, 0, 1]})
    weights = _balanced_group_sample_weights(metadata)
    group0_mass = float(weights[:3].sum())
    group1_mass = float(weights[3:].sum())
    assert group0_mass == group1_mass
    assert float(weights.mean()) == 1.0


def test_conflict_sample_weights_upweight_minority_waterbirds_groups() -> None:
    metadata = pd.DataFrame(
        {
            "y": [0, 0, 1, 1],
            "place": [0, 1, 0, 1],
            "group": [0, 2, 1, 3],
        }
    )

    weights = _erm_finetune_sample_weights(
        metadata,
        sample_mode="conflict_upweight",
        minority_weight=4.0,
    )

    assert weights is not None
    assert _canonical_erm_sample_mode("minority") == "conflict_upweight"
    assert weights[1].item() / weights[0].item() == 4.0
    assert weights[2].item() / weights[3].item() == 4.0
    assert float(weights.mean()) == 1.0


def test_active_sample_mode_uses_natural_warmup_before_target() -> None:
    assert _active_erm_sample_mode("conflict_upweight", epoch_index=0, sample_warmup_epochs=2) == "natural"
    assert _active_erm_sample_mode("conflict_upweight", epoch_index=1, sample_warmup_epochs=2) == "natural"
    assert _active_erm_sample_mode("conflict_upweight", epoch_index=2, sample_warmup_epochs=2) == "conflict_upweight"


def test_contrastive_pair_masks_cross_background_positives_and_same_background_negatives() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    envs = torch.tensor([0, 1, 0, 1])

    positives, hard_negatives = _waterbirds_contrastive_pair_masks(labels, envs)

    assert positives.tolist() == [
        [False, True, False, False],
        [True, False, False, False],
        [False, False, False, True],
        [False, False, True, False],
    ]
    assert hard_negatives.tolist() == [
        [False, False, True, False],
        [False, False, False, True],
        [True, False, False, False],
        [False, True, False, False],
    ]


def test_cross_env_supervised_contrastive_loss_rewards_aligned_positives() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    envs = torch.tensor([0, 1, 0, 1])
    aligned = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    misaligned = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.9, 0.1], [-0.9, -0.1]])

    aligned_loss = _cross_env_supervised_contrastive_loss(aligned, labels, envs, temperature=0.2)
    misaligned_loss = _cross_env_supervised_contrastive_loss(misaligned, labels, envs, temperature=0.2)

    assert aligned_loss.item() < misaligned_loss.item()


def test_extract_feature_matrix_center_background_decomposition(tmp_path: Path) -> None:
    Image.new("RGB", (10, 8), color=(128, 128, 128)).save(tmp_path / "uniform.jpg")
    metadata = pd.DataFrame(
        {
            "split": ["test"],
            "y": [1],
            "place": [0],
            "group": [1],
            "img_filename": ["uniform.jpg"],
        }
    )

    class MeanPixelModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)

    def transform(image: Image.Image) -> torch.Tensor:
        value = image.getpixel((0, 0))[0] / 255.0
        return torch.tensor([[[value]]], dtype=torch.float32)

    features = extract_feature_matrix_from_model(
        MeanPixelModel(),
        tmp_path,
        metadata,
        transform,
        device=torch.device("cpu"),
        batch_size=1,
        feature_decomposition="center_background",
    )

    expected_value = 128.0 / 255.0
    assert features.shape == (1, 4)
    assert torch.allclose(features[0, :3], torch.full((3,), expected_value))
    assert features[0, 3].item() == 0.0


def test_object_proxy_views_separate_center_object_from_border() -> None:
    image = Image.new("RGB", (32, 32), color=(0, 80, 180))
    for x in range(10, 22):
        for y in range(10, 22):
            image.putpixel((x, y), (220, 40, 20))

    foreground, background = _object_proxy_views(image)

    assert foreground.getpixel((16, 16))[0] > foreground.getpixel((1, 1))[0]
    assert background.getpixel((16, 16))[2] > background.getpixel((16, 16))[0]
    assert background.getpixel((1, 1)) == image.getpixel((1, 1))


def test_extract_feature_matrix_object_proxy_decomposition(tmp_path: Path) -> None:
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(tmp_path / "uniform.jpg")
    metadata = pd.DataFrame(
        {
            "split": ["test"],
            "y": [1],
            "place": [0],
            "group": [1],
            "img_filename": ["uniform.jpg"],
        }
    )

    class MeanPixelModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)

    def transform(image: Image.Image) -> torch.Tensor:
        return torch.tensor([[[image.getpixel((0, 0))[0] / 255.0]]], dtype=torch.float32)

    features = extract_feature_matrix_from_model(
        MeanPixelModel(),
        tmp_path,
        metadata,
        transform,
        device=torch.device("cpu"),
        batch_size=1,
        feature_decomposition="object_proxy",
    )

    assert features.shape == (1, 4)


def test_hf_patch_center_background_features_pool_square_grid() -> None:
    cls = torch.tensor([[[100.0, 101.0]]])
    patches = torch.arange(16 * 2, dtype=torch.float32).reshape(1, 16, 2)
    hidden = torch.cat([cls, patches], dim=1)

    features = _hf_patch_center_background_features(hidden)

    grid = patches.reshape(1, 4, 4, 2)
    center = grid[:, 1:3, 1:3, :].reshape(1, -1, 2).mean(dim=1)
    corners = grid[:, [0, 0, 3, 3], [0, 3, 0, 3], :].mean(dim=1)

    assert features.shape == (1, 8)
    assert torch.equal(features[:, :2], cls[:, 0, :])
    assert torch.equal(features[:, 2:4], center)
    assert torch.equal(features[:, 4:6], corners)
    assert torch.equal(features[:, 6:], center - corners)


def test_hf_patch_component_features_can_pool_by_cls_similarity() -> None:
    hidden = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.0, -1.0],
            ]
        ],
        dtype=torch.float32,
    )

    features = _hf_patch_component_features(hidden, pooling="cls_similarity")

    assert features.shape == (1, 8)
    assert torch.equal(features[:, :2], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(features[:, 2:4], torch.tensor([[1.0, 0.0]]))
    assert torch.equal(features[:, 4:6], torch.tensor([[-1.0, 0.0]]))
    assert torch.equal(features[:, 6:], torch.tensor([[2.0, 0.0]]))


def test_hf_patch_component_features_can_pool_by_token_norm() -> None:
    hidden = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
                [0.0, 0.5],
                [0.0, 0.2],
            ]
        ],
        dtype=torch.float32,
    )

    features = _hf_patch_component_features(hidden, pooling="token_norm")

    assert features.shape == (1, 8)
    assert torch.equal(features[:, 2:4], torch.tensor([[3.0, 0.0]]))
    assert torch.equal(features[:, 4:6], torch.tensor([[0.0, 0.2]]))


def test_hf_patch_component_features_can_pool_by_new_selector_recipes() -> None:
    hidden = torch.tensor(
        [
            [
                [1.0, 0.0],
                [4.0, 0.0],
                [0.0, 3.0],
                [-1.0, 0.0],
                [0.0, 0.1],
            ]
        ],
        dtype=torch.float32,
    )

    for pooling in ("cls_norm", "residual_norm", "center_norm"):
        features = _hf_patch_component_features(hidden, pooling=pooling)
        assert features.shape == (1, 8)
        assert torch.equal(features[:, :2], torch.tensor([[1.0, 0.0]]))
        assert torch.equal(features[:, 6:], features[:, 2:4] - features[:, 4:6])


def test_stratified_metadata_limit_keeps_split_group_coverage() -> None:
    rows = []
    for split in ("train", "val", "test"):
        for group in range(4):
            for index in range(3):
                rows.append({"split": split, "group": group, "row": f"{split}-{group}-{index}"})
    metadata = pd.DataFrame(rows)

    limited = _stratified_metadata_limit(metadata, 12)

    counts = limited.groupby(["split", "group"]).size().to_dict()
    assert set(counts.values()) == {1}
    assert len(counts) == 12


def test_prepared_artifact_manifest_round_trips_resolved_settings(tmp_path: Path) -> None:
    features_csv = tmp_path / "features.csv"
    features_csv.write_text("split,y,place,group,feature_0\ntrain,0,0,0,0.0\n", encoding="utf-8")
    artifact = PreparedWaterbirdsFeatures(
        features_csv=features_csv,
        manifest_path=features_csv.with_suffix(features_csv.suffix + ".manifest.json"),
        feature_extractor="resnet50_custom",
        feature_source="local test",
        split_definition="official split",
        base_metrics={"test/worst_group_accuracy": 0.5},
        resolved_settings={
            "erm_finetune_epochs": 50,
            "erm_finetune_lr": 0.001,
            "erm_finetune_preset": "",
        },
        feature_columns=["feature_cls_0000", "feature_background_0000"],
        feature_components={"cls": ["feature_cls_0000"], "background": ["feature_background_0000"]},
    )
    _store_prepared_artifact(artifact)

    loaded = _load_prepared_artifact(features_csv)

    assert loaded is not None
    assert loaded.resolved_settings["erm_finetune_epochs"] == 50
    assert loaded.resolved_settings["erm_finetune_lr"] == 0.001
    assert loaded.resolved_settings["erm_finetune_preset"] == ""
    assert loaded.feature_components["background"] == ["feature_background_0000"]


def test_protocol_feature_matrix_uses_official_train_transform_for_official_style(monkeypatch, tmp_path: Path) -> None:
    calls: list[bool] = []
    seen_train_kwargs: dict[str, object] = {}

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(1, 2)

    def fake_build_train_transform(augment: bool, *, official: bool = False):
        calls.append(official)
        return lambda image: torch.zeros(3, 1, 1)

    def fake_build_resnet50_model(device: torch.device, **_: object):
        return TinyModel().to(device), (lambda image: torch.zeros(3, 1, 1)), "tiny_resnet"

    def fake_train_erm_featurizer(model: torch.nn.Module, *args: object, **kwargs: object) -> None:
        seen_train_kwargs.update(kwargs)
        model.fc = torch.nn.Identity()

    def fake_evaluate_waterbirds_model(*_: object, **__: object) -> dict[str, float]:
        return {"test/worst_group_accuracy": 1.0}

    def fake_extract_feature_matrix_from_model(*_: object, **__: object) -> torch.Tensor:
        return torch.zeros(1, 1)

    monkeypatch.setattr(prepare_waterbirds_features, "build_train_transform", fake_build_train_transform)
    monkeypatch.setattr(prepare_waterbirds_features, "build_resnet50_model", fake_build_resnet50_model)
    monkeypatch.setattr(prepare_waterbirds_features, "train_erm_featurizer", fake_train_erm_featurizer)
    monkeypatch.setattr(prepare_waterbirds_features, "evaluate_waterbirds_model", fake_evaluate_waterbirds_model)
    monkeypatch.setattr(prepare_waterbirds_features, "extract_feature_matrix_from_model", fake_extract_feature_matrix_from_model)
    metadata = pd.DataFrame(
        {
            "split": ["train"],
            "y": [0],
            "place": [0],
            "group": [0],
            "img_filename": ["a.jpg"],
            "place_filename": ["p.jpg"],
        }
    )

    prepare_waterbirds_features.extract_protocol_feature_matrix(
        tmp_path,
        metadata,
        device=torch.device("cpu"),
        batch_size=1,
        erm_finetune_epochs=1,
        erm_finetune_lr=0.001,
        erm_finetune_weight_decay=0.001,
        erm_finetune_mode="all",
        erm_finetune_optimizer="sgd",
        erm_finetune_momentum=0.9,
        erm_finetune_augment=True,
        erm_finetune_balance_groups=False,
        erm_finetune_sample_mode="conflict_upweight",
        erm_finetune_minority_weight=3.0,
        erm_finetune_sample_warmup_epochs=1,
        erm_finetune_contrastive_weight=0.2,
        erm_finetune_contrastive_temperature=0.15,
        erm_finetune_contrastive_hard_negative_weight=2.0,
        erm_finetune_seed=101,
        eval_transform_style="official",
        erm_finetune_preset=None,
        training_checkpoint_path=tmp_path / "train.pt",
    )

    assert calls == [True]
    assert seen_train_kwargs["checkpoint_path"] == tmp_path / "train.pt"
    assert seen_train_kwargs["sample_mode"] == "conflict_upweight"
    assert seen_train_kwargs["minority_weight"] == 3.0
    assert seen_train_kwargs["sample_warmup_epochs"] == 1
    assert seen_train_kwargs["contrastive_weight"] == 0.2
    assert seen_train_kwargs["contrastive_temperature"] == 0.15
    assert seen_train_kwargs["contrastive_hard_negative_weight"] == 2.0
    assert seen_train_kwargs["seed"] == 101


def test_protocol_feature_matrix_can_use_frozen_huggingface_backbone(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class TinyHFBackbone(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones((x.shape[0], 3), dtype=torch.float32)

    def fake_build_frozen_hf_vision_backbone(*, model_id: str, local_files_only: bool = False, pooling: str = "cls"):
        seen["model_id"] = model_id
        seen["local_files_only"] = local_files_only
        seen["pooling"] = pooling
        return TinyHFBackbone(), (lambda image: torch.zeros(3, 2, 2)), "hf_fake_vision"

    monkeypatch.setattr(prepare_waterbirds_features, "build_frozen_hf_vision_backbone", fake_build_frozen_hf_vision_backbone)
    metadata = pd.DataFrame(
        {
            "split": ["train"],
            "y": [0],
            "place": [0],
            "group": [0],
            "img_filename": ["a.jpg"],
            "place_filename": ["p.jpg"],
        }
    )
    Image.new("RGB", (4, 4), color=(128, 128, 128)).save(tmp_path / "a.jpg")

    features, extractor_name, base_metrics, resolved_settings = prepare_waterbirds_features.extract_protocol_feature_matrix(
        tmp_path,
        metadata,
        device=torch.device("cpu"),
        batch_size=1,
        erm_finetune_epochs=0,
        erm_finetune_lr=0.001,
        erm_finetune_weight_decay=0.001,
        erm_finetune_mode="all",
        erm_finetune_optimizer="sgd",
        erm_finetune_momentum=0.9,
        erm_finetune_augment=True,
        erm_finetune_balance_groups=False,
        weights_variant="openai/clip-vit-base-patch32",
        eval_transform_style="local",
        backbone_name="hf_auto",
    )

    assert seen == {"model_id": "openai/clip-vit-base-patch32", "local_files_only": True, "pooling": "cls"}
    assert features.shape == (1, 3)
    assert extractor_name == "hf_fake_vision_penultimate"
    assert base_metrics == {}
    assert resolved_settings["backbone_name"] == "hf_auto"


def test_protocol_feature_matrix_can_use_huggingface_cls_patch_pooling(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class TinyHFBackbone(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones((x.shape[0], 4), dtype=torch.float32)

    def fake_build_frozen_hf_vision_backbone(*, model_id: str, local_files_only: bool = False, pooling: str = "cls"):
        seen["pooling"] = pooling
        return TinyHFBackbone(), (lambda image: torch.zeros(3, 2, 2)), "hf_fake_patch_cls_similarity"

    monkeypatch.setattr(prepare_waterbirds_features, "build_frozen_hf_vision_backbone", fake_build_frozen_hf_vision_backbone)
    metadata = pd.DataFrame(
        {
            "split": ["train"],
            "y": [0],
            "place": [0],
            "group": [0],
            "img_filename": ["a.jpg"],
            "place_filename": ["p.jpg"],
        }
    )
    Image.new("RGB", (4, 4), color=(128, 128, 128)).save(tmp_path / "a.jpg")

    features, extractor_name, _, resolved_settings = prepare_waterbirds_features.extract_protocol_feature_matrix(
        tmp_path,
        metadata,
        device=torch.device("cpu"),
        batch_size=1,
        erm_finetune_epochs=0,
        erm_finetune_lr=0.001,
        erm_finetune_weight_decay=0.001,
        erm_finetune_mode="all",
        erm_finetune_optimizer="sgd",
        erm_finetune_momentum=0.9,
        erm_finetune_augment=True,
        erm_finetune_balance_groups=False,
        weights_variant="facebook/dinov2-small",
        eval_transform_style="local",
        backbone_name="hf_patch_cls_components",
    )

    assert seen["pooling"] == "patch_cls_similarity"
    assert features.shape == (1, 4)
    assert extractor_name == "hf_fake_patch_cls_similarity_penultimate"
    assert resolved_settings["backbone_name"] == "hf_patch_cls_components"


def test_protocol_feature_matrix_can_use_huggingface_center_norm_patch_pooling(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class TinyHFBackbone(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones((x.shape[0], 4), dtype=torch.float32)

    def fake_build_frozen_hf_vision_backbone(*, model_id: str, local_files_only: bool = False, pooling: str = "cls"):
        seen["pooling"] = pooling
        return TinyHFBackbone(), (lambda image: torch.zeros(3, 2, 2)), "hf_fake_patch_center_norm"

    monkeypatch.setattr(prepare_waterbirds_features, "build_frozen_hf_vision_backbone", fake_build_frozen_hf_vision_backbone)
    metadata = pd.DataFrame(
        {
            "split": ["train"],
            "y": [0],
            "place": [0],
            "group": [0],
            "img_filename": ["a.jpg"],
            "place_filename": ["p.jpg"],
        }
    )
    Image.new("RGB", (4, 4), color=(128, 128, 128)).save(tmp_path / "a.jpg")

    features, extractor_name, _, resolved_settings = prepare_waterbirds_features.extract_protocol_feature_matrix(
        tmp_path,
        metadata,
        device=torch.device("cpu"),
        batch_size=1,
        erm_finetune_epochs=0,
        erm_finetune_lr=0.001,
        erm_finetune_weight_decay=0.001,
        erm_finetune_mode="all",
        erm_finetune_optimizer="sgd",
        erm_finetune_momentum=0.9,
        erm_finetune_augment=True,
        erm_finetune_balance_groups=False,
        weights_variant="facebook/dinov2-small",
        eval_transform_style="local",
        backbone_name="hf_patch_center_norm_components",
    )

    assert seen["pooling"] == "patch_center_norm"
    assert features.shape == (1, 4)
    assert extractor_name == "hf_fake_patch_center_norm_penultimate"
    assert resolved_settings["backbone_name"] == "hf_patch_center_norm_components"


def test_evaluate_waterbirds_model_records_group_and_prediction_counts(tmp_path: Path) -> None:
    for filename in ("train0.jpg", "train1.jpg", "val0.jpg", "val1.jpg", "test0.jpg", "test1.jpg"):
        Image.new("RGB", (8, 8), color=(128, 128, 128)).save(tmp_path / filename)
    metadata = pd.DataFrame(
        {
            "split": ["train", "train", "val", "val", "test", "test"],
            "y": [0, 1, 0, 1, 0, 1],
            "place": [0, 1, 0, 1, 0, 1],
            "group": [0, 3, 0, 3, 0, 3],
            "img_filename": ["train0.jpg", "train1.jpg", "val0.jpg", "val1.jpg", "test0.jpg", "test1.jpg"],
        }
    )

    class AlwaysZero(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32).repeat(x.shape[0], 1)

    metrics = evaluate_waterbirds_model(
        AlwaysZero(),
        tmp_path,
        metadata,
        lambda image: torch.zeros(3, 8, 8),
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert metrics["test/accuracy"] == 0.5
    assert metrics["test/group_0_accuracy"] == 1.0
    assert metrics["test/group_3_accuracy"] == 0.0
    assert metrics["test/predicted_label_0_count"] == 2.0
    assert metrics["test/label_1_count"] == 1.0