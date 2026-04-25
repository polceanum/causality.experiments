from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
from typing import Any
import urllib.request

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def build_featurizer(device: torch.device) -> tuple[torch.nn.Module, Any, str]:
    try:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        transform = weights.transforms()
        extractor_name = "torchvision_resnet50_imagenet1k_v2_penultimate"
    except ModuleNotFoundError:
        model, transform, extractor_name = build_hub_resnet50()
    except Exception:
        model, transform, extractor_name = build_hub_resnet50(prefer_pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model, transform, extractor_name


def build_hub_resnet50(prefer_pretrained: bool = False) -> tuple[torch.nn.Module, Any, str]:
    ensure_torchvision_nms_stub()
    try:
        model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")
        extractor_name = "torch_hub_resnet50_imagenet1k_v2_penultimate"
    except Exception:
        model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
        extractor_name = "torch_hub_resnet50_pretrained_penultimate"
    return model, imagenet_transform, extractor_name


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


def load_metadata(dataset_dir: Path, limit: int | None = None) -> pd.DataFrame:
    metadata = pd.read_csv(dataset_dir / "metadata.csv")
    metadata["split"] = metadata["split"].map(SPLIT_MAP)
    metadata["group"] = metadata["place"].astype(int) * 2 + metadata["y"].astype(int)
    if limit is not None:
        metadata = metadata.iloc[:limit].copy()
    return metadata


def build_feature_frame(metadata: pd.DataFrame, feature_matrix: torch.Tensor) -> pd.DataFrame:
    frame = metadata.loc[:, ["split", "y", "place", "group", "img_filename", "place_filename"]].copy()
    features = feature_matrix.detach().cpu().numpy()
    feature_frame = pd.DataFrame(
        features,
        columns=[f"feature_{index}" for index in range(features.shape[1])],
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
) -> Path:
    if features_csv.exists() and not overwrite_features:
        return features_csv

    archive_path = ensure_downloaded(download_dir, force=force_download)
    dataset_dir = ensure_extracted(archive_path, raw_dir, force=force_extract)
    metadata = load_metadata(dataset_dir, limit=limit)
    device = choose_device(device_name)
    feature_matrix, extractor_name = extract_feature_matrix(
        dataset_dir,
        metadata,
        device=device,
        batch_size=batch_size,
    )
    features_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_frame = build_feature_frame(metadata, feature_matrix)
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
    return features_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR))
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--features-csv", default=str(DEFAULT_FEATURES_CSV))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
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
    )
    print(out)


if __name__ == "__main__":
    main()