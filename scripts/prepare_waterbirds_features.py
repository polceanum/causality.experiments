from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
import tarfile
from typing import Any
import urllib.request

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import yaml


WATERBIRDS_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
ARCHIVE_NAME = "waterbird_complete95_forest2water2.tar.gz"
DATASET_DIRNAME = "waterbird_complete95_forest2water2"
DEFAULT_DOWNLOAD_DIR = Path("data/downloads")
DEFAULT_DATA_DIR = Path("data/waterbirds")
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_FEATURES_CSV = DEFAULT_DATA_DIR / "features.csv"
DEFAULT_CONFIG_PATH = Path("configs/benchmarks/waterbirds_features.yaml")
SPLIT_MAP = {0: "train", 1: "val", 2: "test"}
TORCHVISION_STUB_LIB: torch.library.Library | None = None
ERM_SAMPLE_MODES = {
    "natural",
    "group_balanced",
    "conflict_upweight",
    "group_balanced_conflict_upweight",
}
FEATURE_DECOMPOSITION_MODES = {"none", "center_background"}


@dataclass
class PreparedWaterbirdsFeatures:
    features_csv: Path
    manifest_path: Path
    feature_extractor: str
    feature_source: str
    split_definition: str
    base_metrics: dict[str, float]
    resolved_settings: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    feature_components: dict[str, list[str]] = field(default_factory=dict)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.weight * grad_output, None


def _grad_reverse(x: torch.Tensor, weight: float) -> torch.Tensor:
    return _GradientReverse.apply(x, weight)


class WaterbirdsImageDataset(Dataset):
    def __init__(self, root_dir: Path, metadata: pd.DataFrame, transform: Any) -> None:
        self.root_dir = root_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[idx]
        image_path = self.root_dir / str(row["img_filename"])
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        return self.transform(image), idx


class WaterbirdsImageViewDataset(Dataset):
    def __init__(self, root_dir: Path, metadata: pd.DataFrame, transform: Any, feature_decomposition: str) -> None:
        self.root_dir = root_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform
        self.feature_decomposition = _canonical_feature_decomposition(feature_decomposition)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.metadata.iloc[idx]
        image_path = self.root_dir / str(row["img_filename"])
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        views = _waterbirds_decomposition_views(image, self.feature_decomposition)
        return torch.stack([self.transform(view) for view in views]), idx


def _canonical_feature_decomposition(feature_decomposition: str | None) -> str:
    mode = (feature_decomposition or "none").strip().lower().replace("-", "_")
    aliases = {
        "": "none",
        "center_bg": "center_background",
        "center_background_corners": "center_background",
    }
    mode = aliases.get(mode, mode)
    if mode not in FEATURE_DECOMPOSITION_MODES:
        known = ", ".join(sorted(FEATURE_DECOMPOSITION_MODES))
        raise ValueError(f"Feature decomposition must be one of: {known}.")
    return mode


def _center_crop_box(width: int, height: int, fraction: float) -> tuple[int, int, int, int]:
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return left, top, min(width, left + crop_width), min(height, top + crop_height)


def _corner_crop_boxes(width: int, height: int, fraction: float) -> list[tuple[int, int, int, int]]:
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    return [
        (0, 0, crop_width, crop_height),
        (width - crop_width, 0, width, crop_height),
        (0, height - crop_height, crop_width, height),
        (width - crop_width, height - crop_height, width, height),
    ]


def _waterbirds_decomposition_views(image: Image.Image, feature_decomposition: str) -> list[Image.Image]:
    mode = _canonical_feature_decomposition(feature_decomposition)
    if mode == "none":
        return [image]
    width, height = image.size
    center = image.crop(_center_crop_box(width, height, 0.65))
    corners = [image.crop(box) for box in _corner_crop_boxes(width, height, 0.45)]
    return [image, center, *corners]


def _compose_decomposed_view_features(view_features: torch.Tensor, feature_decomposition: str) -> torch.Tensor:
    mode = _canonical_feature_decomposition(feature_decomposition)
    if mode == "none":
        return view_features[:, 0, :]
    full_features = view_features[:, 0, :]
    center_features = view_features[:, 1, :]
    background_features = view_features[:, 2:, :].mean(dim=1)
    return torch.cat(
        [
            full_features,
            center_features,
            background_features,
            center_features - background_features,
        ],
        dim=1,
    )


def _feature_decomposition_tag(feature_decomposition: str) -> str:
    mode = _canonical_feature_decomposition(feature_decomposition)
    if mode == "none":
        return ""
    if mode == "center_background":
        return "_decompcenterbg"
    return f"_decomp{mode}"


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_resnet50_model(
    device: torch.device,
    *,
    weights_variant: str = "imagenet1k_v2",
    eval_transform_style: str = "weights",
) -> tuple[torch.nn.Module, Any, str]:
    try:
        from torchvision.models import ResNet50_Weights, resnet50

        variant = weights_variant.strip().lower()
        if variant in {"legacy_pretrained", "pretrained", "imagenet1k_v1"}:
            weights = ResNet50_Weights.IMAGENET1K_V1
            model_name = "torchvision_resnet50_imagenet1k_v1"
        else:
            weights = ResNet50_Weights.IMAGENET1K_V2
            model_name = "torchvision_resnet50_imagenet1k_v2"
        model = resnet50(weights=weights)
        transform = imagenet_transform if eval_transform_style == "official" else weights.transforms()
    except ModuleNotFoundError:
        model, transform, model_name = build_hub_resnet50(
            weights_variant=weights_variant,
            eval_transform_style=eval_transform_style,
        )
    except Exception:
        model, transform, model_name = build_hub_resnet50(
            weights_variant=weights_variant,
            eval_transform_style=eval_transform_style,
            prefer_pretrained=True,
        )
    model.to(device)
    return model, transform, model_name


def build_featurizer(device: torch.device) -> tuple[torch.nn.Module, Any, str]:
    model, transform, model_name = build_resnet50_model(device)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, transform, f"{model_name}_penultimate"


def build_hub_resnet50(
    *,
    weights_variant: str = "imagenet1k_v2",
    eval_transform_style: str = "weights",
    prefer_pretrained: bool = False,
) -> tuple[torch.nn.Module, Any, str]:
    ensure_torchvision_nms_stub()
    try:
        variant = weights_variant.strip().lower()
        if variant in {"legacy_pretrained", "pretrained", "imagenet1k_v1"}:
            model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
            extractor_name = "torch_hub_resnet50_pretrained"
        else:
            model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
            extractor_name = "torch_hub_resnet50_imagenet1k_v2"
    except Exception:
        model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
        extractor_name = "torch_hub_resnet50_pretrained"
    transform = imagenet_transform if eval_transform_style == "official" else imagenet_transform
    return model, transform, extractor_name


def build_frozen_hub_backbone(
    *,
    backbone_name: str,
    weights_variant: str = "imagenet1k_v1",
    eval_transform_style: str = "weights",
) -> tuple[torch.nn.Module, Any, str]:
    ensure_torchvision_nms_stub()
    backbone_key = backbone_name.strip().lower()
    if backbone_key == "resnet50":
        model, transform, extractor_name = build_hub_resnet50(
            weights_variant=weights_variant,
            eval_transform_style=eval_transform_style,
        )
        model.fc = torch.nn.Identity()
        model.eval()
        return model, transform, f"{extractor_name}_penultimate"
    if backbone_key not in {"convnext_tiny", "efficientnet_b0"}:
        raise ValueError("Frozen hub backbone must be one of: resnet50, convnext_tiny, efficientnet_b0.")
    variant = weights_variant.strip().lower()
    weights = None if variant in {"none", "random"} else "IMAGENET1K_V1"
    model = torch.hub.load("pytorch/vision", backbone_key, weights=weights)
    if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Sequential):
        modules = list(model.classifier.children())
        if modules and isinstance(modules[-1], torch.nn.Linear):
            model.classifier = torch.nn.Sequential(*modules[:-1], torch.nn.Identity())
        else:
            raise ValueError(f"Unsupported classifier shape for frozen backbone {backbone_name!r}.")
    else:
        raise ValueError(f"Frozen backbone {backbone_name!r} does not expose a supported classifier.")
    model.eval()
    transform = imagenet_transform if eval_transform_style in {"official", "weights"} else imagenet_transform
    suffix = "random" if weights is None else "imagenet1k_v1"
    return model, transform, f"torch_hub_{backbone_key}_{suffix}_penultimate"


class FrozenHuggingFaceVisionBackbone(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=x)
        pooled = getattr(output, "pooler_output", None)
        if pooled is not None:
            return pooled
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("Hugging Face vision model output has no pooler_output or last_hidden_state.")
        return hidden[:, 0]


def _topk_patch_mask(scores: torch.Tensor, *, count: int, largest: bool) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("Patch selection scores must have shape [batch, patches].")
    resolved_count = min(max(1, int(count)), int(scores.shape[1]))
    indices = torch.topk(scores, k=resolved_count, dim=1, largest=largest).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(1, indices, True)


def _masked_patch_mean(patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != patches.shape[:2]:
        raise ValueError("Patch mask must match the batch and patch dimensions.")
    weights = mask.to(dtype=patches.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (patches * weights).sum(dim=1) / denominator


def _hf_patch_component_features(hidden: torch.Tensor, *, pooling: str = "center_background") -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[1] < 2:
        raise ValueError("Patch component pooling requires a sequence of CLS plus patch tokens.")
    cls_features = hidden[:, 0]
    patches = hidden[:, 1:]
    patch_count = int(patches.shape[1])
    pooling_key = pooling.strip().lower().replace("-", "_")
    if pooling_key in {"center_background", "patch_center_background", "patch_components"}:
        grid_size = int(math.sqrt(patch_count))
        if grid_size * grid_size != patch_count:
            raise ValueError("Patch component pooling requires a square patch grid.")
        grid = patches.reshape(patches.shape[0], grid_size, grid_size, patches.shape[2])
        indices = torch.arange(grid_size, device=hidden.device)
        row = indices[:, None]
        col = indices[None, :]
        center_start = max(0, grid_size // 4)
        center_end = min(grid_size, grid_size - center_start)
        center_mask = (row >= center_start) & (row < center_end) & (col >= center_start) & (col < center_end)
        corner_width = max(1, grid_size // 4)
        corner_rows = (row < corner_width) | (row >= grid_size - corner_width)
        corner_cols = (col < corner_width) | (col >= grid_size - corner_width)
        background_mask = corner_rows & corner_cols
        center_features = grid[:, center_mask, :].mean(dim=1)
        background_features = grid[:, background_mask, :].mean(dim=1)
    elif pooling_key in {"cls_similarity", "patch_cls_similarity", "patch_cls_components"}:
        scores = torch.nn.functional.cosine_similarity(patches.float(), cls_features[:, None, :].float(), dim=2)
        count = max(1, patch_count // 4)
        foreground_mask = _topk_patch_mask(scores, count=count, largest=True)
        background_mask = _topk_patch_mask(scores, count=count, largest=False)
        center_features = _masked_patch_mean(patches, foreground_mask)
        background_features = _masked_patch_mean(patches, background_mask)
    elif pooling_key in {"token_norm", "patch_token_norm", "patch_norm_components"}:
        scores = torch.linalg.vector_norm(patches.float(), dim=2)
        count = max(1, patch_count // 4)
        foreground_mask = _topk_patch_mask(scores, count=count, largest=True)
        background_mask = _topk_patch_mask(scores, count=count, largest=False)
        center_features = _masked_patch_mean(patches, foreground_mask)
        background_features = _masked_patch_mean(patches, background_mask)
    else:
        raise ValueError("Patch component pooling must be one of: center_background, cls_similarity, token_norm.")
    return torch.cat(
        [
            cls_features,
            center_features,
            background_features,
            center_features - background_features,
        ],
        dim=1,
    )


def _hf_patch_center_background_features(hidden: torch.Tensor) -> torch.Tensor:
    return _hf_patch_component_features(hidden, pooling="center_background")


class FrozenHuggingFacePatchComponentBackbone(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, *, pooling: str = "center_background") -> None:
        super().__init__()
        self.model = model
        self.pooling = pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=x)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            raise ValueError("Hugging Face vision model output has no last_hidden_state for patch components.")
        return _hf_patch_component_features(hidden, pooling=self.pooling)


def build_frozen_hf_vision_backbone(
    *,
    model_id: str,
    local_files_only: bool = False,
    pooling: str = "cls",
) -> tuple[torch.nn.Module, Any, str]:
    try:
        from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel
    except Exception as exc:
        raise RuntimeError("Frozen Hugging Face backbones require transformers to be installed.") from exc

    processor = AutoImageProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    if "clip" in model_id.lower():
        model = CLIPVisionModel.from_pretrained(model_id, local_files_only=local_files_only)
    else:
        model = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)
    model.eval()

    def transform(image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        encoded = processor(images=image, return_tensors="pt")
        return encoded["pixel_values"][0]

    safe_model_id = model_id.replace("/", "_").replace("-", "_").replace(".", "_")
    pooling_key = pooling.strip().lower()
    pooling_aliases = {
        "patch_center_background": ("center_background", "patch_center_background"),
        "patch_components": ("center_background", "patch_center_background"),
        "patch_cls_similarity": ("cls_similarity", "patch_cls_similarity"),
        "patch_cls_components": ("cls_similarity", "patch_cls_similarity"),
        "patch_token_norm": ("token_norm", "patch_token_norm"),
        "patch_norm_components": ("token_norm", "patch_token_norm"),
    }
    if pooling_key in pooling_aliases:
        resolved_pooling, name_suffix = pooling_aliases[pooling_key]
        return FrozenHuggingFacePatchComponentBackbone(model, pooling=resolved_pooling), transform, f"hf_{safe_model_id}_{name_suffix}"
    if pooling_key != "cls":
        raise ValueError("Hugging Face vision pooling must be one of: cls, patch_center_background, patch_cls_similarity, patch_token_norm.")
    return FrozenHuggingFaceVisionBackbone(model), transform, f"hf_{safe_model_id}_vision"


def build_train_transform(augment: bool, *, official: bool = False) -> Any:
    if not augment:
        return imagenet_transform
    try:
        from torchvision import transforms

        if official:
            return transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        224,
                        scale=(0.7, 1.0),
                        ratio=(0.75, 4.0 / 3.0),
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    except Exception:
        return imagenet_transform


def ensure_torchvision_nms_stub() -> None:
    global TORCHVISION_STUB_LIB
    if TORCHVISION_STUB_LIB is not None:
        return
    try:
        TORCHVISION_STUB_LIB = torch.library.Library("torchvision", "DEF")
        TORCHVISION_STUB_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass


def imagenet_transform(image: Image.Image) -> torch.Tensor:
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    scale = 256.0 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
    left = max((resized.size[0] - 224) // 2, 0)
    top = max((resized.size[1] - 224) // 2, 0)
    cropped = resized.crop((left, top, left + 224, top + 224))
    array = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def ensure_downloaded(download_dir: Path, force: bool = False) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    archive_path = download_dir / ARCHIVE_NAME
    if archive_path.exists() and not force:
        return archive_path
    urllib.request.urlretrieve(WATERBIRDS_URL, archive_path)
    return archive_path


def ensure_extracted(archive_path: Path, raw_dir: Path, force: bool = False) -> Path:
    dataset_dir = raw_dir / DATASET_DIRNAME
    metadata_path = dataset_dir / "metadata.csv"
    if metadata_path.exists() and not force:
        return dataset_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as handle:
        handle.extractall(raw_dir)
    return dataset_dir


def _stratified_metadata_limit(metadata: pd.DataFrame, limit: int) -> pd.DataFrame:
    if limit <= 0 or limit >= len(metadata):
        return metadata.copy()
    split_order = {name: index for index, name in enumerate(SPLIT_MAP.values())}
    buckets = [
        group.index.to_list()
        for _, group in sorted(
            metadata.groupby(["split", "group"], sort=True),
            key=lambda item: (split_order.get(str(item[0][0]), 99), int(item[0][1])),
        )
    ]
    selected: list[int] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for bucket in buckets:
            if depth < len(bucket):
                selected.append(int(bucket[depth]))
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        depth += 1
    return metadata.loc[selected].copy()


def load_metadata(dataset_dir: Path, limit: int | None = None) -> pd.DataFrame:
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    metadata["split"] = metadata["split"].map(SPLIT_MAP)
    metadata["group"] = metadata["place"].astype(int) * 2 + metadata["y"].astype(int)
    if limit is not None:
        metadata = _stratified_metadata_limit(metadata, limit)
    return metadata


def _feature_columns_for_components(feature_dim: int, component_names: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    if feature_dim <= 0:
        raise ValueError("Feature matrix must have at least one column.")
    if not component_names:
        component_names = ["global"]
    if feature_dim % len(component_names) != 0:
        columns = [f"feature_{index}" for index in range(feature_dim)]
        return columns, {"global": columns}
    component_dim = feature_dim // len(component_names)
    columns: list[str] = []
    groups: dict[str, list[str]] = {}
    for component in component_names:
        safe_component = component.strip().lower().replace("-", "_").replace(" ", "_")
        component_columns = [f"feature_{safe_component}_{index:04d}" for index in range(component_dim)]
        columns.extend(component_columns)
        groups[safe_component] = component_columns
    return columns, groups


def _feature_component_names(*, resolved_settings: dict[str, Any], extractor_name: str) -> list[str]:
    feature_decomposition = _canonical_feature_decomposition(str(resolved_settings.get("feature_decomposition", "none")))
    if feature_decomposition != "none":
        return ["full", "center", "background", "center_minus_background"]
    backbone_name = str(resolved_settings.get("backbone_name", "")).strip().lower()
    extractor_key = extractor_name.strip().lower()
    if backbone_name in {"hf_patch_components", "hf_patch"} or "patch_center_background" in extractor_key:
        return ["cls", "center", "background", "center_minus_background"]
    if backbone_name in {"hf_patch_cls_components", "hf_patch_cls"} or "patch_cls_similarity" in extractor_key:
        return ["cls", "foreground", "background", "foreground_minus_background"]
    if backbone_name in {"hf_patch_norm_components", "hf_patch_norm"} or "patch_token_norm" in extractor_key:
        return ["cls", "foreground", "background", "foreground_minus_background"]
    return ["global"]


def build_feature_frame(
    metadata: pd.DataFrame,
    feature_matrix: torch.Tensor,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    frame = metadata.loc[:, ["split", "y", "place", "group", "img_filename", "place_filename"]].copy()
    features = feature_matrix.detach().cpu().numpy()
    columns = feature_columns or [f"feature_{index}" for index in range(features.shape[1])]
    if len(columns) != features.shape[1]:
        raise ValueError("feature_columns length must match feature_matrix width.")
    feature_frame = pd.DataFrame(
        features,
        columns=columns,
    )
    return pd.concat([frame.reset_index(drop=True), feature_frame], axis=1)


def extract_feature_matrix(
    dataset_dir: Path,
    metadata: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, str]:
    model, transform, extractor_name = build_featurizer(device)
    dataset = WaterbirdsImageDataset(dataset_dir, metadata, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch, _ in loader:
            features = model(batch.to(device))
            outputs.append(features.detach().cpu())
    return torch.cat(outputs, dim=0), extractor_name


def _set_trainable_layers(model: torch.nn.Module, mode: str) -> None:
    mode = mode.strip().lower()
    if mode not in {"head", "layer4", "all"}:
        raise ValueError("ERM fine-tune mode must be one of: head, layer4, all.")
    for parameter in model.parameters():
        parameter.requires_grad = mode == "all"
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    if mode == "layer4" and hasattr(model, "layer4"):
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True


def _balanced_group_sample_weights(metadata: pd.DataFrame) -> torch.Tensor:
    group_counts = metadata["group"].value_counts().to_dict()
    if not group_counts:
        raise ValueError("Cannot balance empty Waterbirds metadata.")
    weights = [1.0 / float(group_counts[group]) for group in metadata["group"]]
    tensor = torch.tensor(weights, dtype=torch.double)
    return tensor / tensor.mean()


def _canonical_erm_sample_mode(sample_mode: str | None, *, balance_groups: bool = False) -> str:
    key = (sample_mode or "").strip().lower().replace("-", "_")
    if key in {"", "auto"}:
        return "group_balanced" if balance_groups else "natural"
    aliases = {
        "none": "natural",
        "erm": "natural",
        "natural": "natural",
        "balanced": "group_balanced",
        "group": "group_balanced",
        "group_balanced": "group_balanced",
        "minority": "conflict_upweight",
        "minority_upweight": "conflict_upweight",
        "conflict": "conflict_upweight",
        "conflict_upweight": "conflict_upweight",
        "group_conflict": "group_balanced_conflict_upweight",
        "balanced_conflict": "group_balanced_conflict_upweight",
        "group_balanced_conflict": "group_balanced_conflict_upweight",
        "group_balanced_conflict_upweight": "group_balanced_conflict_upweight",
    }
    try:
        mode = aliases[key]
    except KeyError as exc:
        known = ", ".join(sorted(ERM_SAMPLE_MODES))
        raise ValueError(f"ERM fine-tune sample mode must be one of: {known}.") from exc
    if balance_groups and mode == "natural":
        return "group_balanced"
    return mode


def _waterbirds_conflict_mask(metadata: pd.DataFrame) -> np.ndarray:
    return metadata["y"].astype(int).to_numpy() != metadata["place"].astype(int).to_numpy()


def _erm_finetune_sample_weights(
    metadata: pd.DataFrame,
    *,
    sample_mode: str,
    minority_weight: float,
) -> torch.Tensor | None:
    mode = _canonical_erm_sample_mode(sample_mode)
    if mode == "natural":
        return None
    if minority_weight <= 0.0:
        raise ValueError("erm_finetune_minority_weight must be positive.")
    if mode == "group_balanced":
        return _balanced_group_sample_weights(metadata)
    conflict_weight = torch.tensor(
        np.where(_waterbirds_conflict_mask(metadata), float(minority_weight), 1.0),
        dtype=torch.double,
    )
    if mode == "conflict_upweight":
        return conflict_weight / conflict_weight.mean()
    if mode == "group_balanced_conflict_upweight":
        weights = _balanced_group_sample_weights(metadata) * conflict_weight
        return weights / weights.mean()
    known = ", ".join(sorted(ERM_SAMPLE_MODES))
    raise ValueError(f"ERM fine-tune sample mode must be one of: {known}.")


def _active_erm_sample_mode(target_sample_mode: str, *, epoch_index: int, sample_warmup_epochs: int) -> str:
    if sample_warmup_epochs < 0:
        raise ValueError("erm_finetune_sample_warmup_epochs must be non-negative.")
    if epoch_index < sample_warmup_epochs:
        return "natural"
    return _canonical_erm_sample_mode(target_sample_mode)


def _waterbirds_contrastive_pair_masks(labels: torch.Tensor, envs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    labels = labels.view(-1)
    envs = envs.view(-1)
    if labels.shape != envs.shape:
        raise ValueError("labels and envs must have the same shape.")
    not_self = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    same_label = labels[:, None] == labels[None, :]
    same_env = envs[:, None] == envs[None, :]
    positives = same_label & ~same_env & not_self
    hard_negatives = ~same_label & same_env & not_self
    return positives, hard_negatives


def _cross_env_supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    envs: torch.Tensor,
    *,
    temperature: float,
    hard_negative_weight: float = 1.0,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("erm_finetune_contrastive_temperature must be positive.")
    if hard_negative_weight <= 0.0:
        raise ValueError("erm_finetune_contrastive_hard_negative_weight must be positive.")
    positives, hard_negatives = _waterbirds_contrastive_pair_masks(labels, envs)
    valid_anchors = positives.any(dim=1)
    if not bool(valid_anchors.any().item()):
        return features.sum() * 0.0

    embeddings = torch.nn.functional.normalize(features.float(), dim=1)
    logits = embeddings @ embeddings.t() / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    not_self = ~torch.eye(len(labels), dtype=torch.bool, device=features.device)
    denom_weights = torch.ones_like(logits)
    if hard_negative_weight != 1.0:
        denom_weights = torch.where(
            hard_negatives,
            torch.full_like(denom_weights, float(hard_negative_weight)),
            denom_weights,
        )
    denom_weights = torch.where(not_self, denom_weights, torch.zeros_like(denom_weights))
    exp_logits = torch.exp(logits)
    denominator = (exp_logits * denom_weights).sum(dim=1).clamp_min(1e-12)
    numerator = (exp_logits * positives.float()).sum(dim=1).clamp_min(1e-12)
    return (-torch.log(numerator[valid_anchors] / denominator[valid_anchors])).mean()


def _set_erm_finetune_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _make_optimizer(
    parameters: list[torch.nn.Parameter],
    *,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    optimizer_key = optimizer_name.strip().lower()
    if optimizer_key == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_key == "sgd":
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError("ERM fine-tune optimizer must be one of: adam, sgd.")


def _resnet_penultimate(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return torch.flatten(x, 1)


def train_erm_featurizer(
    model: torch.nn.Module,
    dataset_dir: Path,
    metadata: pd.DataFrame,
    transform: Any,
    *,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    mode: str,
    optimizer_name: str,
    momentum: float,
    balance_groups: bool,
    sample_mode: str = "natural",
    minority_weight: float = 1.0,
    sample_warmup_epochs: int = 0,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
    contrastive_hard_negative_weight: float = 1.0,
    seed: int | None = None,
    env_adv_weight: float = 0.0,
    env_adv_hidden_dim: int = 0,
    env_adv_loss_weight: float = 1.0,
    warmup_epochs: int = 0,
    warmup_mode: str = "head",
    drop_head: bool = True,
    checkpoint_path: Path | None = None,
) -> None:
    train_metadata = metadata[metadata["split"] == "train"]
    if train_metadata.empty:
        raise ValueError("Waterbirds ERM fine-tuning requires a non-empty train split.")
    resolved_seed = None if seed is None else int(seed)
    if resolved_seed is not None:
        _set_erm_finetune_seed(resolved_seed, device)
    output_dim = int(metadata["y"].max()) + 1
    in_features = int(model.fc.in_features)
    model.fc = torch.nn.Linear(in_features, output_dim).to(device)
    initial_mode = warmup_mode if warmup_epochs > 0 else mode
    _set_trainable_layers(model, initial_mode)
    dataset = WaterbirdsImageDataset(dataset_dir, train_metadata, transform)
    resolved_sample_mode = _canonical_erm_sample_mode(sample_mode, balance_groups=balance_groups)
    sample_warmup_epochs = int(sample_warmup_epochs)
    contrastive_weight = float(contrastive_weight)
    contrastive_temperature = float(contrastive_temperature)
    contrastive_hard_negative_weight = float(contrastive_hard_negative_weight)
    if contrastive_weight < 0.0:
        raise ValueError("erm_finetune_contrastive_weight must be non-negative.")
    if contrastive_weight > 0.0 and contrastive_temperature <= 0.0:
        raise ValueError("erm_finetune_contrastive_temperature must be positive.")
    if contrastive_weight > 0.0 and contrastive_hard_negative_weight <= 0.0:
        raise ValueError("erm_finetune_contrastive_hard_negative_weight must be positive.")

    def build_loader(active_sample_mode: str, *, epoch_index: int) -> DataLoader:
        generator = None
        if resolved_seed is not None:
            generator = torch.Generator().manual_seed(resolved_seed + epoch_index)
        sample_weights = _erm_finetune_sample_weights(
            train_metadata,
            sample_mode=active_sample_mode,
            minority_weight=minority_weight,
        )
        sampler = None
        shuffle = True
        if sample_weights is not None:
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(train_metadata),
                replacement=True,
                generator=generator,
            )
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            generator=generator,
        )

    nuisance_head: torch.nn.Module | None = None
    if env_adv_weight > 0.0:
        if env_adv_hidden_dim > 0:
            nuisance_head = torch.nn.Sequential(
                torch.nn.Linear(in_features, env_adv_hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(env_adv_hidden_dim, int(train_metadata["place"].max()) + 1),
            ).to(device)
        else:
            nuisance_head = torch.nn.Linear(in_features, int(train_metadata["place"].max()) + 1).to(device)

    def build_optimizer() -> torch.optim.Optimizer:
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if nuisance_head is not None:
            params.extend(parameter for parameter in nuisance_head.parameters() if parameter.requires_grad)
        return _make_optimizer(
            params,
            optimizer_name=optimizer_name,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    optimizer = build_optimizer()
    start_epoch = 0
    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = int(checkpoint.get("completed_epochs", 0))
        if warmup_epochs > 0 and start_epoch > warmup_epochs:
            _set_trainable_layers(model, mode)
            optimizer = build_optimizer()
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if nuisance_head is not None and checkpoint.get("nuisance_head_state") is not None:
            nuisance_head.load_state_dict(checkpoint["nuisance_head_state"])
        print(
            json.dumps(
                {
                    "stage": "train_erm_featurizer",
                    "event": "resume_checkpoint",
                    "completed_epochs": start_epoch,
                    "epochs": epochs,
                    "checkpoint_path": str(checkpoint_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    for epoch_idx in range(start_epoch, epochs):
        if warmup_epochs > 0 and epoch_idx == warmup_epochs:
            _set_trainable_layers(model, mode)
            optimizer = build_optimizer()
        if resolved_seed is not None:
            _set_erm_finetune_seed(resolved_seed + epoch_idx, device)
        active_sample_mode = _active_erm_sample_mode(
            resolved_sample_mode,
            epoch_index=epoch_idx,
            sample_warmup_epochs=sample_warmup_epochs,
        )
        loader = build_loader(active_sample_mode, epoch_index=epoch_idx)
        progress_interval = max(1, len(loader) // 6)
        model.train()
        if nuisance_head is not None:
            nuisance_head.train()
        epoch_loss = 0.0
        epoch_label_loss = 0.0
        epoch_contrastive_loss = 0.0
        seen_examples = 0
        print(
            json.dumps(
                {
                    "stage": "train_erm_featurizer",
                    "event": "epoch_start",
                    "epoch": epoch_idx + 1,
                    "epochs": epochs,
                    "batches": len(loader),
                    "train_examples": len(train_metadata),
                    "sample_mode": active_sample_mode,
                    "target_sample_mode": resolved_sample_mode,
                    "minority_weight": float(minority_weight),
                    "sample_warmup_epochs": sample_warmup_epochs,
                    "contrastive_weight": contrastive_weight,
                    "contrastive_temperature": contrastive_temperature,
                    "contrastive_hard_negative_weight": contrastive_hard_negative_weight,
                    "seed": resolved_seed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        for batch_idx, (batch, row_idx) in enumerate(loader, start=1):
            labels = torch.tensor(train_metadata.iloc[row_idx.numpy()]["y"].to_numpy(), dtype=torch.long, device=device)
            envs = torch.tensor(train_metadata.iloc[row_idx.numpy()]["place"].to_numpy(), dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(device)
            features = _resnet_penultimate(model, batch)
            logits = model.fc(features)
            label_loss = torch.nn.functional.cross_entropy(logits, labels)
            loss = label_loss
            contrastive_loss = features.sum() * 0.0
            if contrastive_weight > 0.0:
                contrastive_loss = _cross_env_supervised_contrastive_loss(
                    features,
                    labels,
                    envs,
                    temperature=contrastive_temperature,
                    hard_negative_weight=contrastive_hard_negative_weight,
                )
                loss = loss + contrastive_weight * contrastive_loss
            if nuisance_head is not None:
                nuisance_logits = nuisance_head(_grad_reverse(features, env_adv_weight))
                loss = loss + env_adv_loss_weight * torch.nn.functional.cross_entropy(nuisance_logits, envs)
            loss.backward()
            optimizer.step()
            batch_size_seen = int(batch.shape[0])
            epoch_loss += float(loss.detach().cpu().item()) * batch_size_seen
            epoch_label_loss += float(label_loss.detach().cpu().item()) * batch_size_seen
            epoch_contrastive_loss += float(contrastive_loss.detach().cpu().item()) * batch_size_seen
            seen_examples += batch_size_seen
            if batch_idx == len(loader) or batch_idx % progress_interval == 0:
                print(
                    json.dumps(
                        {
                            "stage": "train_erm_featurizer",
                            "event": "batch_progress",
                            "epoch": epoch_idx + 1,
                            "epochs": epochs,
                            "batch": batch_idx,
                            "batches": len(loader),
                            "mean_loss": epoch_loss / max(seen_examples, 1),
                            "mean_label_loss": epoch_label_loss / max(seen_examples, 1),
                            "mean_contrastive_loss": epoch_contrastive_loss / max(seen_examples, 1),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "stage": "train_erm_featurizer",
                    "event": "epoch_end",
                    "epoch": epoch_idx + 1,
                    "epochs": epochs,
                    "mean_loss": epoch_loss / max(seen_examples, 1),
                    "mean_label_loss": epoch_label_loss / max(seen_examples, 1),
                    "mean_contrastive_loss": epoch_contrastive_loss / max(seen_examples, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "completed_epochs": epoch_idx + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "nuisance_head_state": None if nuisance_head is None else nuisance_head.state_dict(),
                },
                checkpoint_path,
            )
            print(
                json.dumps(
                    {
                        "stage": "train_erm_featurizer",
                        "event": "checkpoint_saved",
                        "completed_epochs": epoch_idx + 1,
                        "epochs": epochs,
                        "checkpoint_path": str(checkpoint_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if drop_head:
        model.fc = torch.nn.Identity()
    model.eval()


def _official_erm_settings() -> dict[str, Any]:
    return {
        "batch_size": 32,
        "erm_finetune_epochs": 100,
        "erm_finetune_lr": 1e-3,
        "erm_finetune_weight_decay": 1e-3,
        "erm_finetune_mode": "all",
        "erm_finetune_optimizer": "sgd",
        "erm_finetune_momentum": 0.9,
        "erm_finetune_augment": True,
        "erm_finetune_balance_groups": False,
        "erm_finetune_sample_mode": "natural",
        "erm_finetune_minority_weight": 1.0,
        "erm_finetune_sample_warmup_epochs": 0,
        "erm_finetune_contrastive_weight": 0.0,
        "erm_finetune_contrastive_temperature": 0.1,
        "erm_finetune_contrastive_hard_negative_weight": 1.0,
        "erm_finetune_seed": None,
        "feature_decomposition": "none",
        "erm_env_adv_weight": 0.0,
        "erm_env_adv_hidden_dim": 0,
        "erm_env_adv_loss_weight": 1.0,
        "erm_finetune_warmup_epochs": 0,
        "erm_finetune_warmup_mode": "head",
        "weights_variant": "legacy_pretrained",
        "eval_transform_style": "official",
        "feature_extractor_suffix": "waterbirds_official_erm_sgd_aug_e100_penultimate",
    }


def _resolve_erm_settings(
    *,
    batch_size: int,
    erm_finetune_epochs: int,
    erm_finetune_lr: float,
    erm_finetune_weight_decay: float,
    erm_finetune_mode: str,
    erm_finetune_optimizer: str,
    erm_finetune_momentum: float,
    erm_finetune_augment: bool,
    erm_finetune_balance_groups: bool,
    erm_env_adv_weight: float,
    erm_env_adv_hidden_dim: int,
    erm_env_adv_loss_weight: float,
    erm_finetune_warmup_epochs: int,
    erm_finetune_warmup_mode: str,
    weights_variant: str,
    eval_transform_style: str,
    feature_extractor_suffix: str,
    erm_finetune_preset: str | None,
    erm_finetune_sample_mode: str = "natural",
    erm_finetune_minority_weight: float = 1.0,
    erm_finetune_sample_warmup_epochs: int = 0,
    erm_finetune_contrastive_weight: float = 0.0,
    erm_finetune_contrastive_temperature: float = 0.1,
    erm_finetune_contrastive_hard_negative_weight: float = 1.0,
    erm_finetune_seed: int | None = None,
    feature_decomposition: str = "none",
) -> dict[str, Any]:
    if int(erm_finetune_sample_warmup_epochs) < 0:
        raise ValueError("erm_finetune_sample_warmup_epochs must be non-negative.")
    if float(erm_finetune_contrastive_weight) < 0.0:
        raise ValueError("erm_finetune_contrastive_weight must be non-negative.")
    if float(erm_finetune_contrastive_temperature) <= 0.0:
        raise ValueError("erm_finetune_contrastive_temperature must be positive.")
    if float(erm_finetune_contrastive_hard_negative_weight) <= 0.0:
        raise ValueError("erm_finetune_contrastive_hard_negative_weight must be positive.")
    settings = {
        "batch_size": int(batch_size),
        "erm_finetune_epochs": int(erm_finetune_epochs),
        "erm_finetune_lr": float(erm_finetune_lr),
        "erm_finetune_weight_decay": float(erm_finetune_weight_decay),
        "erm_finetune_mode": erm_finetune_mode,
        "erm_finetune_optimizer": erm_finetune_optimizer,
        "erm_finetune_momentum": float(erm_finetune_momentum),
        "erm_finetune_augment": bool(erm_finetune_augment),
        "erm_finetune_balance_groups": bool(erm_finetune_balance_groups),
        "erm_finetune_sample_mode": _canonical_erm_sample_mode(
            erm_finetune_sample_mode,
            balance_groups=bool(erm_finetune_balance_groups),
        ),
        "erm_finetune_minority_weight": float(erm_finetune_minority_weight),
        "erm_finetune_sample_warmup_epochs": int(erm_finetune_sample_warmup_epochs),
        "erm_finetune_contrastive_weight": float(erm_finetune_contrastive_weight),
        "erm_finetune_contrastive_temperature": float(erm_finetune_contrastive_temperature),
        "erm_finetune_contrastive_hard_negative_weight": float(erm_finetune_contrastive_hard_negative_weight),
        "erm_finetune_seed": None if erm_finetune_seed is None else int(erm_finetune_seed),
        "feature_decomposition": _canonical_feature_decomposition(feature_decomposition),
        "erm_env_adv_weight": float(erm_env_adv_weight),
        "erm_env_adv_hidden_dim": int(erm_env_adv_hidden_dim),
        "erm_env_adv_loss_weight": float(erm_env_adv_loss_weight),
        "erm_finetune_warmup_epochs": int(erm_finetune_warmup_epochs),
        "erm_finetune_warmup_mode": erm_finetune_warmup_mode,
        "weights_variant": weights_variant,
        "eval_transform_style": eval_transform_style,
        "feature_extractor_suffix": feature_extractor_suffix,
    }
    preset = (erm_finetune_preset or "").strip().lower()
    if not preset:
        return settings
    if preset != "official":
        raise ValueError("erm_finetune_preset must be empty or 'official'.")
    official = _official_erm_settings()
    settings.update(official)
    settings["feature_decomposition"] = _canonical_feature_decomposition(feature_decomposition)
    return settings


def evaluate_waterbirds_model(
    model: torch.nn.Module,
    dataset_dir: Path,
    metadata: pd.DataFrame,
    transform: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}
    for split_name in ("train", "val", "test"):
        split_metadata = metadata[metadata["split"] == split_name].reset_index(drop=True)
        dataset = WaterbirdsImageDataset(dataset_dir, split_metadata, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        predictions: list[torch.Tensor] = []
        with torch.inference_mode():
            for batch, _ in loader:
                predictions.append(model(batch.to(device)).argmax(dim=1).cpu())
        pred = torch.cat(predictions, dim=0)
        target = torch.tensor(split_metadata["y"].to_numpy(), dtype=torch.long)
        group = torch.tensor(split_metadata["group"].to_numpy(), dtype=torch.long)
        metrics[f"{split_name}/accuracy"] = float((pred == target).float().mean().item())
        for label_id in sorted(int(value) for value in torch.unique(target)):
            metrics[f"{split_name}/label_{label_id}_count"] = float((target == label_id).sum().item())
        for label_id in sorted(int(value) for value in torch.unique(pred)):
            metrics[f"{split_name}/predicted_label_{label_id}_count"] = float((pred == label_id).sum().item())
        scores: list[float] = []
        for group_id in torch.unique(group):
            mask = group == group_id
            group_accuracy = float((pred[mask] == target[mask]).float().mean().item())
            group_key = int(group_id.item())
            metrics[f"{split_name}/group_{group_key}_accuracy"] = group_accuracy
            metrics[f"{split_name}/group_{group_key}_count"] = float(mask.sum().item())
            scores.append(group_accuracy)
        metrics[f"{split_name}/worst_group_accuracy"] = min(scores) if scores else float("nan")
    return metrics


def extract_feature_matrix_from_model(
    model: torch.nn.Module,
    dataset_dir: Path,
    metadata: pd.DataFrame,
    transform: Any,
    *,
    device: torch.device,
    batch_size: int,
    feature_decomposition: str = "none",
) -> torch.Tensor:
    mode = _canonical_feature_decomposition(feature_decomposition)
    if mode != "none":
        dataset = WaterbirdsImageViewDataset(dataset_dir, metadata, transform, mode)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        outputs: list[torch.Tensor] = []
        model.eval()
        with torch.inference_mode():
            for batch, _ in loader:
                batch = batch.to(device)
                batch_count, view_count = int(batch.shape[0]), int(batch.shape[1])
                flat_batch = batch.view(batch_count * view_count, *batch.shape[2:])
                flat_features = model(flat_batch).detach().cpu()
                view_features = flat_features.view(batch_count, view_count, -1)
                outputs.append(_compose_decomposed_view_features(view_features, mode))
        return torch.cat(outputs, dim=0)
    dataset = WaterbirdsImageDataset(dataset_dir, metadata, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch, _ in loader:
            outputs.append(model(batch.to(device)).detach().cpu())
    return torch.cat(outputs, dim=0)


def extract_protocol_feature_matrix(
    dataset_dir: Path,
    metadata: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
    erm_finetune_epochs: int,
    erm_finetune_lr: float,
    erm_finetune_weight_decay: float,
    erm_finetune_mode: str,
    erm_finetune_optimizer: str,
    erm_finetune_momentum: float,
    erm_finetune_augment: bool,
    erm_finetune_balance_groups: bool,
    erm_finetune_sample_mode: str = "natural",
    erm_finetune_minority_weight: float = 1.0,
    erm_finetune_sample_warmup_epochs: int = 0,
    erm_finetune_contrastive_weight: float = 0.0,
    erm_finetune_contrastive_temperature: float = 0.1,
    erm_finetune_contrastive_hard_negative_weight: float = 1.0,
    erm_finetune_seed: int | None = None,
    erm_env_adv_weight: float = 0.0,
    erm_env_adv_hidden_dim: int = 0,
    erm_env_adv_loss_weight: float = 1.0,
    erm_finetune_warmup_epochs: int = 0,
    erm_finetune_warmup_mode: str = "head",
    weights_variant: str = "imagenet1k_v2",
    eval_transform_style: str = "weights",
    backbone_name: str = "resnet50",
    feature_extractor_suffix: str = "",
    erm_finetune_preset: str | None = None,
    feature_decomposition: str = "none",
    training_checkpoint_path: Path | None = None,
) -> tuple[torch.Tensor, str, dict[str, float], dict[str, Any]]:
    settings = _resolve_erm_settings(
        batch_size=batch_size,
        erm_finetune_epochs=erm_finetune_epochs,
        erm_finetune_lr=erm_finetune_lr,
        erm_finetune_weight_decay=erm_finetune_weight_decay,
        erm_finetune_mode=erm_finetune_mode,
        erm_finetune_optimizer=erm_finetune_optimizer,
        erm_finetune_momentum=erm_finetune_momentum,
        erm_finetune_augment=erm_finetune_augment,
        erm_finetune_balance_groups=erm_finetune_balance_groups,
        erm_finetune_sample_mode=erm_finetune_sample_mode,
        erm_finetune_minority_weight=erm_finetune_minority_weight,
        erm_finetune_sample_warmup_epochs=erm_finetune_sample_warmup_epochs,
        erm_finetune_contrastive_weight=erm_finetune_contrastive_weight,
        erm_finetune_contrastive_temperature=erm_finetune_contrastive_temperature,
        erm_finetune_contrastive_hard_negative_weight=erm_finetune_contrastive_hard_negative_weight,
        erm_finetune_seed=erm_finetune_seed,
        erm_env_adv_weight=erm_env_adv_weight,
        erm_env_adv_hidden_dim=erm_env_adv_hidden_dim,
        erm_env_adv_loss_weight=erm_env_adv_loss_weight,
        erm_finetune_warmup_epochs=erm_finetune_warmup_epochs,
        erm_finetune_warmup_mode=erm_finetune_warmup_mode,
        weights_variant=weights_variant,
        eval_transform_style=eval_transform_style,
        feature_extractor_suffix=feature_extractor_suffix,
        erm_finetune_preset=erm_finetune_preset,
        feature_decomposition=feature_decomposition,
    )
    backbone_key = backbone_name.strip().lower()
    if int(settings["erm_finetune_epochs"]) > 0 and backbone_key != "resnet50":
        raise ValueError("ERM fine-tuning is currently supported only for the resnet50 backbone.")
    if backbone_key == "resnet50":
        model, transform, model_name = build_resnet50_model(
            device,
            weights_variant=str(settings["weights_variant"]),
            eval_transform_style=str(settings["eval_transform_style"]),
        )
    elif backbone_key in {
        "hf",
        "hf_auto",
        "huggingface",
        "hf_patch_components",
        "hf_patch",
        "hf_patch_cls_components",
        "hf_patch_cls",
        "hf_patch_norm_components",
        "hf_patch_norm",
    }:
        if backbone_key in {"hf_patch_components", "hf_patch"}:
            hf_pooling = "patch_center_background"
        elif backbone_key in {"hf_patch_cls_components", "hf_patch_cls"}:
            hf_pooling = "patch_cls_similarity"
        elif backbone_key in {"hf_patch_norm_components", "hf_patch_norm"}:
            hf_pooling = "patch_token_norm"
        else:
            hf_pooling = "cls"
        model, transform, model_name = build_frozen_hf_vision_backbone(
            model_id=str(settings["weights_variant"]),
            local_files_only=str(settings["eval_transform_style"]).strip().lower() == "local",
            pooling=hf_pooling,
        )
        model.to(device)
    else:
        model, transform, model_name = build_frozen_hub_backbone(
            backbone_name=backbone_key,
            weights_variant=str(settings["weights_variant"]),
            eval_transform_style=str(settings["eval_transform_style"]),
        )
        model.to(device)
    base_metrics: dict[str, float] = {}
    if int(settings["erm_finetune_epochs"]) > 0:
        train_transform = build_train_transform(
            bool(settings["erm_finetune_augment"]),
            official=(str(settings["eval_transform_style"]).strip().lower() == "official"),
        )
        train_erm_featurizer(
            model,
            dataset_dir,
            metadata,
            train_transform,
            device=device,
            batch_size=int(settings["batch_size"]),
            epochs=int(settings["erm_finetune_epochs"]),
            lr=float(settings["erm_finetune_lr"]),
            weight_decay=float(settings["erm_finetune_weight_decay"]),
            mode=str(settings["erm_finetune_mode"]),
            optimizer_name=str(settings["erm_finetune_optimizer"]),
            momentum=float(settings["erm_finetune_momentum"]),
            balance_groups=bool(settings["erm_finetune_balance_groups"]),
            sample_mode=str(settings["erm_finetune_sample_mode"]),
            minority_weight=float(settings["erm_finetune_minority_weight"]),
            sample_warmup_epochs=int(settings["erm_finetune_sample_warmup_epochs"]),
            contrastive_weight=float(settings["erm_finetune_contrastive_weight"]),
            contrastive_temperature=float(settings["erm_finetune_contrastive_temperature"]),
            contrastive_hard_negative_weight=float(settings["erm_finetune_contrastive_hard_negative_weight"]),
            seed=settings["erm_finetune_seed"],
            env_adv_weight=float(settings["erm_env_adv_weight"]),
            env_adv_hidden_dim=int(settings["erm_env_adv_hidden_dim"]),
            env_adv_loss_weight=float(settings["erm_env_adv_loss_weight"]),
            warmup_epochs=int(settings["erm_finetune_warmup_epochs"]),
            warmup_mode=str(settings["erm_finetune_warmup_mode"]),
            drop_head=False,
            checkpoint_path=training_checkpoint_path,
        )
        base_metrics = evaluate_waterbirds_model(
            model,
            dataset_dir,
            metadata,
            transform,
            device=device,
            batch_size=int(settings["batch_size"]),
        )
        model.fc = torch.nn.Identity()
        suffix = str(settings["feature_extractor_suffix"]).strip()
        if suffix:
            extractor_name = f"{model_name}_{suffix}"
        else:
            aug_tag = "aug" if bool(settings["erm_finetune_augment"]) else "noaug"
            sample_mode = str(settings["erm_finetune_sample_mode"])
            if sample_mode == "natural":
                balance_tag = "erm"
            elif sample_mode == "group_balanced":
                balance_tag = "groupbalanced"
            else:
                balance_tag = sample_mode
            optimizer_tag = str(settings["erm_finetune_optimizer"]).strip().lower()
            adv_tag = ""
            if float(settings["erm_env_adv_weight"]) > 0.0:
                adv_tag = f"_envadv{settings['erm_env_adv_weight']}"
            warmup_tag = ""
            if int(settings["erm_finetune_warmup_epochs"]) > 0:
                warmup_tag = f"_warm{settings['erm_finetune_warmup_mode']}{settings['erm_finetune_warmup_epochs']}"
            extractor_name = (
                f"{model_name}_waterbirds_{balance_tag}_{settings['erm_finetune_mode']}_{optimizer_tag}_{aug_tag}"
                f"{adv_tag}{warmup_tag}_e{settings['erm_finetune_epochs']}_penultimate"
            )
    else:
        model.fc = torch.nn.Identity()
        model.eval()
        extractor_name = model_name if str(model_name).endswith("_penultimate") else f"{model_name}_penultimate"
    decomposition_tag = _feature_decomposition_tag(str(settings["feature_decomposition"]))
    if decomposition_tag and decomposition_tag not in extractor_name:
        extractor_name = f"{extractor_name}{decomposition_tag}"
    return (
        extract_feature_matrix_from_model(
            model,
            dataset_dir,
            metadata,
            transform,
            device=device,
            batch_size=int(settings["batch_size"]),
            feature_decomposition=str(settings["feature_decomposition"]),
        ),
        extractor_name,
        base_metrics,
        {**settings, "erm_finetune_preset": (erm_finetune_preset or ""), "backbone_name": backbone_key},
    )


def update_benchmark_config(
    config_path: Path,
    *,
    features_path: Path,
    feature_extractor: str,
    feature_source: str,
    split_definition: str,
) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config.setdefault("dataset", {})
    config["dataset"]["path"] = str(features_path)
    config.setdefault("benchmark", {})
    config["benchmark"].setdefault("provenance", {})
    config["benchmark"]["provenance"]["feature_extractor"] = feature_extractor
    config["benchmark"]["provenance"]["feature_source"] = feature_source
    config["benchmark"]["provenance"]["split_definition"] = split_definition
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _feature_manifest_path(features_csv: Path) -> Path:
    return features_csv.with_suffix(features_csv.suffix + ".manifest.json")


def _load_prepared_artifact(features_csv: Path) -> PreparedWaterbirdsFeatures | None:
    manifest_path = _feature_manifest_path(features_csv)
    if not features_csv.exists() or not manifest_path.exists():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return PreparedWaterbirdsFeatures(
        features_csv=features_csv,
        manifest_path=manifest_path,
        feature_extractor=str(payload.get("feature_extractor", "")),
        feature_source=str(payload.get("feature_source", "")),
        split_definition=str(payload.get("split_definition", "")),
        base_metrics={str(key): float(value) for key, value in dict(payload.get("base_metrics", {})).items()},
        resolved_settings=dict(payload.get("resolved_settings", {})),
        feature_columns=[str(value) for value in payload.get("feature_columns", [])],
        feature_components={
            str(key): [str(item) for item in value]
            for key, value in dict(payload.get("feature_components", {})).items()
        },
    )


def _store_prepared_artifact(artifact: PreparedWaterbirdsFeatures) -> None:
    artifact.manifest_path.write_text(
        json.dumps(
            {
                "feature_extractor": artifact.feature_extractor,
                "feature_source": artifact.feature_source,
                "split_definition": artifact.split_definition,
                "base_metrics": artifact.base_metrics,
                "resolved_settings": artifact.resolved_settings,
                "feature_columns": artifact.feature_columns,
                "feature_components": artifact.feature_components,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def prepare_waterbirds_features_artifact(
    *,
    download_dir: Path,
    raw_dir: Path,
    features_csv: Path,
    config_path: Path,
    device_name: str,
    batch_size: int,
    limit: int | None,
    force_download: bool,
    force_extract: bool,
    overwrite_features: bool,
    erm_finetune_epochs: int = 0,
    erm_finetune_lr: float = 1e-4,
    erm_finetune_weight_decay: float = 1e-4,
    erm_finetune_mode: str = "layer4",
    erm_finetune_optimizer: str = "adam",
    erm_finetune_momentum: float = 0.9,
    erm_finetune_augment: bool = False,
    erm_finetune_balance_groups: bool = False,
    erm_finetune_sample_mode: str = "natural",
    erm_finetune_minority_weight: float = 1.0,
    erm_finetune_sample_warmup_epochs: int = 0,
    erm_finetune_contrastive_weight: float = 0.0,
    erm_finetune_contrastive_temperature: float = 0.1,
    erm_finetune_contrastive_hard_negative_weight: float = 1.0,
    erm_finetune_seed: int | None = None,
    erm_env_adv_weight: float = 0.0,
    erm_env_adv_hidden_dim: int = 0,
    erm_env_adv_loss_weight: float = 1.0,
    erm_finetune_warmup_epochs: int = 0,
    erm_finetune_warmup_mode: str = "head",
    weights_variant: str = "imagenet1k_v2",
    eval_transform_style: str = "weights",
    backbone_name: str = "resnet50",
    feature_extractor_suffix: str = "",
    erm_finetune_preset: str | None = None,
    feature_decomposition: str = "none",
) -> PreparedWaterbirdsFeatures:
    if features_csv.exists() and not overwrite_features:
        cached = _load_prepared_artifact(features_csv)
        if cached is not None:
            return cached

    dataset_dir = raw_dir / DATASET_DIRNAME
    if not dataset_dir.joinpath("metadata.csv").exists() or force_extract:
        archive_path = ensure_downloaded(download_dir, force=force_download)
        dataset_dir = ensure_extracted(archive_path, raw_dir, force=force_extract)
    metadata = load_metadata(dataset_dir, limit=limit)
    device = choose_device(device_name)
    training_checkpoint_path = features_csv.with_suffix(features_csv.suffix + ".training.pt")
    feature_matrix, extractor_name, base_metrics, resolved_settings = extract_protocol_feature_matrix(
        dataset_dir,
        metadata,
        device=device,
        batch_size=batch_size,
        erm_finetune_epochs=erm_finetune_epochs,
        erm_finetune_lr=erm_finetune_lr,
        erm_finetune_weight_decay=erm_finetune_weight_decay,
        erm_finetune_mode=erm_finetune_mode,
        erm_finetune_optimizer=erm_finetune_optimizer,
        erm_finetune_momentum=erm_finetune_momentum,
        erm_finetune_augment=erm_finetune_augment,
        erm_finetune_balance_groups=erm_finetune_balance_groups,
        erm_finetune_sample_mode=erm_finetune_sample_mode,
        erm_finetune_minority_weight=erm_finetune_minority_weight,
        erm_finetune_sample_warmup_epochs=erm_finetune_sample_warmup_epochs,
        erm_finetune_contrastive_weight=erm_finetune_contrastive_weight,
        erm_finetune_contrastive_temperature=erm_finetune_contrastive_temperature,
        erm_finetune_contrastive_hard_negative_weight=erm_finetune_contrastive_hard_negative_weight,
        erm_finetune_seed=erm_finetune_seed,
        erm_env_adv_weight=erm_env_adv_weight,
        erm_env_adv_hidden_dim=erm_env_adv_hidden_dim,
        erm_env_adv_loss_weight=erm_env_adv_loss_weight,
        erm_finetune_warmup_epochs=erm_finetune_warmup_epochs,
        erm_finetune_warmup_mode=erm_finetune_warmup_mode,
        weights_variant=weights_variant,
        eval_transform_style=eval_transform_style,
        backbone_name=backbone_name,
        feature_extractor_suffix=feature_extractor_suffix,
        erm_finetune_preset=erm_finetune_preset,
        feature_decomposition=feature_decomposition,
        training_checkpoint_path=training_checkpoint_path,
    )
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_columns, feature_components = _feature_columns_for_components(
        int(feature_matrix.shape[1]),
        _feature_component_names(resolved_settings=resolved_settings, extractor_name=extractor_name),
    )
    feature_frame = build_feature_frame(metadata, feature_matrix, feature_columns=feature_columns)
    feature_frame.to_csv(features_csv, index=False)
    split_definition = "official Stanford Waterbirds metadata split: 0=train, 1=val, 2=test"
    feature_source = (
        "official Stanford Waterbirds tarball "
        "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz "
        "featurized locally by scripts/prepare_waterbirds_features.py"
    )
    update_benchmark_config(
        config_path,
        features_path=features_csv,
        feature_extractor=extractor_name,
        feature_source=feature_source,
        split_definition=split_definition,
    )
    artifact = PreparedWaterbirdsFeatures(
        features_csv=features_csv,
        manifest_path=_feature_manifest_path(features_csv),
        feature_extractor=extractor_name,
        feature_source=feature_source,
        split_definition=split_definition,
        base_metrics=base_metrics,
        resolved_settings=resolved_settings,
        feature_columns=feature_columns,
        feature_components=feature_components,
    )
    _store_prepared_artifact(artifact)
    if training_checkpoint_path.exists():
        training_checkpoint_path.unlink()
    return artifact


def prepare_waterbirds_features(
    *,
    download_dir: Path,
    raw_dir: Path,
    features_csv: Path,
    config_path: Path,
    device_name: str,
    batch_size: int,
    limit: int | None,
    force_download: bool,
    force_extract: bool,
    overwrite_features: bool,
    erm_finetune_epochs: int = 0,
    erm_finetune_lr: float = 1e-4,
    erm_finetune_weight_decay: float = 1e-4,
    erm_finetune_mode: str = "layer4",
    erm_finetune_optimizer: str = "adam",
    erm_finetune_momentum: float = 0.9,
    erm_finetune_augment: bool = False,
    erm_finetune_balance_groups: bool = False,
    erm_finetune_sample_mode: str = "natural",
    erm_finetune_minority_weight: float = 1.0,
    erm_finetune_sample_warmup_epochs: int = 0,
    erm_finetune_contrastive_weight: float = 0.0,
    erm_finetune_contrastive_temperature: float = 0.1,
    erm_finetune_contrastive_hard_negative_weight: float = 1.0,
    erm_finetune_seed: int | None = None,
    erm_env_adv_weight: float = 0.0,
    erm_env_adv_hidden_dim: int = 0,
    erm_env_adv_loss_weight: float = 1.0,
    erm_finetune_warmup_epochs: int = 0,
    erm_finetune_warmup_mode: str = "head",
    weights_variant: str = "imagenet1k_v2",
    eval_transform_style: str = "weights",
    backbone_name: str = "resnet50",
    feature_extractor_suffix: str = "",
    erm_finetune_preset: str | None = None,
    feature_decomposition: str = "none",
) -> Path:
    artifact = prepare_waterbirds_features_artifact(
        download_dir=download_dir,
        raw_dir=raw_dir,
        features_csv=features_csv,
        config_path=config_path,
        device_name=device_name,
        batch_size=batch_size,
        limit=limit,
        force_download=force_download,
        force_extract=force_extract,
        overwrite_features=overwrite_features,
        erm_finetune_epochs=erm_finetune_epochs,
        erm_finetune_lr=erm_finetune_lr,
        erm_finetune_weight_decay=erm_finetune_weight_decay,
        erm_finetune_mode=erm_finetune_mode,
        erm_finetune_optimizer=erm_finetune_optimizer,
        erm_finetune_momentum=erm_finetune_momentum,
        erm_finetune_augment=erm_finetune_augment,
        erm_finetune_balance_groups=erm_finetune_balance_groups,
        erm_finetune_sample_mode=erm_finetune_sample_mode,
        erm_finetune_minority_weight=erm_finetune_minority_weight,
        erm_finetune_sample_warmup_epochs=erm_finetune_sample_warmup_epochs,
        erm_finetune_contrastive_weight=erm_finetune_contrastive_weight,
        erm_finetune_contrastive_temperature=erm_finetune_contrastive_temperature,
        erm_finetune_contrastive_hard_negative_weight=erm_finetune_contrastive_hard_negative_weight,
        erm_finetune_seed=erm_finetune_seed,
        erm_env_adv_weight=erm_env_adv_weight,
        erm_env_adv_hidden_dim=erm_env_adv_hidden_dim,
        erm_env_adv_loss_weight=erm_env_adv_loss_weight,
        erm_finetune_warmup_epochs=erm_finetune_warmup_epochs,
        erm_finetune_warmup_mode=erm_finetune_warmup_mode,
        weights_variant=weights_variant,
        eval_transform_style=eval_transform_style,
        backbone_name=backbone_name,
        feature_extractor_suffix=feature_extractor_suffix,
        erm_finetune_preset=erm_finetune_preset,
        feature_decomposition=feature_decomposition,
    )
    return artifact.features_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--erm-finetune-epochs", type=int, default=0)
    parser.add_argument("--erm-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--erm-finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--erm-finetune-mode", choices=("head", "layer4", "all"), default="layer4")
    parser.add_argument("--erm-finetune-optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--erm-finetune-momentum", type=float, default=0.9)
    parser.add_argument("--erm-finetune-augment", action="store_true")
    parser.add_argument("--erm-finetune-balance-groups", action="store_true")
    parser.add_argument("--erm-finetune-sample-mode", default="natural", choices=tuple(sorted(ERM_SAMPLE_MODES)))
    parser.add_argument("--erm-finetune-minority-weight", type=float, default=1.0)
    parser.add_argument("--erm-finetune-sample-warmup-epochs", type=int, default=0)
    parser.add_argument("--erm-finetune-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--erm-finetune-contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--erm-finetune-contrastive-hard-negative-weight", type=float, default=1.0)
    parser.add_argument("--erm-finetune-seed", type=int, default=None)
    parser.add_argument("--erm-env-adv-weight", type=float, default=0.0)
    parser.add_argument("--erm-env-adv-hidden-dim", type=int, default=0)
    parser.add_argument("--erm-env-adv-loss-weight", type=float, default=1.0)
    parser.add_argument("--erm-finetune-warmup-epochs", type=int, default=0)
    parser.add_argument("--erm-finetune-warmup-mode", choices=("head", "layer4", "all"), default="head")
    parser.add_argument("--erm-finetune-preset", choices=("official",), default=None)
    parser.add_argument("--backbone-name", default="resnet50")
    parser.add_argument("--feature-decomposition", default="none", choices=tuple(sorted(FEATURE_DECOMPOSITION_MODES)))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    args = parser.parse_args()

    out = prepare_waterbirds_features(
        download_dir=Path(args.download_dir),
        raw_dir=Path(args.raw_dir),
        features_csv=Path(args.features_csv),
        config_path=Path(args.config),
        device_name=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        force_download=args.force_download,
        force_extract=args.force_extract,
        overwrite_features=args.overwrite_features,
        erm_finetune_epochs=args.erm_finetune_epochs,
        erm_finetune_lr=args.erm_finetune_lr,
        erm_finetune_weight_decay=args.erm_finetune_weight_decay,
        erm_finetune_mode=args.erm_finetune_mode,
        erm_finetune_optimizer=args.erm_finetune_optimizer,
        erm_finetune_momentum=args.erm_finetune_momentum,
        erm_finetune_augment=args.erm_finetune_augment,
        erm_finetune_balance_groups=args.erm_finetune_balance_groups,
        erm_finetune_sample_mode=args.erm_finetune_sample_mode,
        erm_finetune_minority_weight=args.erm_finetune_minority_weight,
        erm_finetune_sample_warmup_epochs=args.erm_finetune_sample_warmup_epochs,
        erm_finetune_contrastive_weight=args.erm_finetune_contrastive_weight,
        erm_finetune_contrastive_temperature=args.erm_finetune_contrastive_temperature,
        erm_finetune_contrastive_hard_negative_weight=args.erm_finetune_contrastive_hard_negative_weight,
        erm_finetune_seed=args.erm_finetune_seed,
        erm_env_adv_weight=args.erm_env_adv_weight,
        erm_env_adv_hidden_dim=args.erm_env_adv_hidden_dim,
        erm_env_adv_loss_weight=args.erm_env_adv_loss_weight,
        erm_finetune_warmup_epochs=args.erm_finetune_warmup_epochs,
        erm_finetune_warmup_mode=args.erm_finetune_warmup_mode,
        erm_finetune_preset=args.erm_finetune_preset,
        backbone_name=args.backbone_name,
        feature_decomposition=args.feature_decomposition,
    )
    print(out)


if __name__ == "__main__":
    main()
