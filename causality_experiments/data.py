from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    task: str
    splits: dict[str, dict[str, torch.Tensor]]
    input_dim: int
    output_dim: int
    causal_mask: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def split(self, name: str) -> dict[str, torch.Tensor]:
        try:
            return self.splits[name]
        except KeyError as exc:
            raise KeyError(f"Dataset {self.name!r} has no split {name!r}.") from exc


DatasetFactory = Callable[[dict[str, Any]], DatasetBundle]


def _rng(config: dict[str, Any]) -> np.random.Generator:
    return np.random.default_rng(int(config.get("seed", 0)))


def _classification_splits(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    env: np.ndarray,
    group: np.ndarray,
    causal_mask: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> DatasetBundle:
    n = len(x)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    idxs = {
        "train": np.arange(0, train_end),
        "val": np.arange(train_end, val_end),
        "test": np.arange(val_end, n),
    }
    splits: dict[str, dict[str, torch.Tensor]] = {}
    for split, idx in idxs.items():
        splits[split] = {
            "x": torch.tensor(x[idx], dtype=torch.float32),
            "y": torch.tensor(y[idx], dtype=torch.long),
            "env": torch.tensor(env[idx], dtype=torch.long),
            "group": torch.tensor(group[idx], dtype=torch.long),
        }
    return DatasetBundle(
        name=name,
        task="classification",
        splits=splits,
        input_dim=int(x.shape[1]),
        output_dim=int(np.max(y)) + 1,
        causal_mask=None
        if causal_mask is None
        else torch.tensor(causal_mask, dtype=torch.float32),
        metadata=metadata or {},
    )


def _spurious_tabular(config: dict[str, Any], *, nonlinear: bool = False) -> DatasetBundle:
    rng = _rng(config)
    n = int(config.get("n", 900))
    rho_train = float(config.get("rho_train", 0.9))
    rho_test = float(config.get("rho_test", -0.7))
    x1 = rng.normal(size=n)
    env = np.zeros(n, dtype=np.int64)
    env[int(0.6 * n) : int(0.8 * n)] = 1
    env[int(0.8 * n) :] = 2
    rho = np.where(env == 2, rho_test, rho_train)
    x2 = rho * x1 + np.sqrt(np.maximum(1.0 - rho**2, 0.0)) * rng.normal(size=n)
    signal = np.sin(2.0 * x1) if nonlinear else x1
    y = (signal + 0.25 * rng.normal(size=n) > 0.0).astype(np.int64)
    x = np.stack([x1, x2], axis=1).astype(np.float32)
    group = (env * 2 + y).astype(np.int64)
    return _classification_splits(
        "synthetic_nonlinear" if nonlinear else "synthetic_linear",
        x,
        y,
        env,
        group,
        np.array([1.0, 0.0], dtype=np.float32),
        {"causal_features": ["x1"], "spurious_features": ["x2"]},
    )


def _factor_fixture(config: dict[str, Any], name: str, factors: int = 6) -> DatasetBundle:
    rng = _rng(config)
    n = int(config.get("n", 960))
    z = rng.integers(0, 4, size=(n, factors)).astype(np.float32)
    env = z[:, 1].astype(np.int64)
    y = ((z[:, 0] + z[:, 2]) % 2).astype(np.int64)
    nuisance = (env == y).astype(np.float32)[:, None]
    x = np.concatenate([z / 3.0, nuisance, rng.normal(scale=0.05, size=(n, 4))], axis=1)
    group = (env.astype(np.int64) * 2 + y).astype(np.int64)
    causal_mask = np.zeros(x.shape[1], dtype=np.float32)
    causal_mask[[0, 2]] = 1.0
    return _classification_splits(
        name,
        x.astype(np.float32),
        y,
        env,
        group,
        causal_mask,
        {"factors": [f"z{i}" for i in range(factors)], "fixture": True},
    )


def _waterbirds_fixture(config: dict[str, Any]) -> DatasetBundle:
    rng = _rng(config)
    n = int(config.get("n", 800))
    bird = rng.integers(0, 2, size=n)
    background = np.where(rng.random(n) < 0.85, bird, 1 - bird)
    env = background.astype(np.int64)
    texture = rng.normal(size=(n, 6))
    x = np.concatenate([bird[:, None], background[:, None], texture], axis=1).astype(np.float32)
    group = (background * 2 + bird).astype(np.int64)
    return _classification_splits(
        "waterbirds_tiny",
        x,
        bird.astype(np.int64),
        env,
        group,
        np.array([1.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        {"labels": ["landbird", "waterbird"], "fixture": True},
    )


def _sequence_fixture(config: dict[str, Any], name: str, ner: bool = False) -> DatasetBundle:
    rng = _rng(config)
    n = int(config.get("n", 900))
    length = int(config.get("length", 10))
    tokens = rng.integers(0, 12, size=(n, length))
    cause_pos = 2 if not ner else 4
    conf_pos = 7 if length > 7 else length - 1
    y = (tokens[:, cause_pos] % 2).astype(np.int64)
    tokens[:, conf_pos] = np.where(rng.random(n) < 0.85, y, 1 - y)
    env = tokens[:, conf_pos].astype(np.int64)
    group = (env * 2 + y).astype(np.int64)
    x = tokens.astype(np.float32)
    causal_mask = np.zeros(length, dtype=np.float32)
    causal_mask[cause_pos] = 1.0
    return _classification_splits(
        name,
        x,
        y,
        env,
        group,
        causal_mask,
        {
            "cause_position": cause_pos,
            "confounder_position": conf_pos,
            "fixture": True,
            "modality": "sequence",
            "vocab_size": 12,
        },
    )


def _local_dataset(config: dict[str, Any]) -> DatasetBundle:
    path = Path(str(config.get("path", ""))).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Real dataset path {path} does not exist. Run with a fixture config or set "
            "dataset.path to a prepared local benchmark mirror."
        )
    raise NotImplementedError(
        "Real dataset adapters are scaffolded for v1. Add a loader that returns DatasetBundle "
        f"for {path}."
    )


def _first_existing(columns: set[str], candidates: tuple[str, ...], context: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    joined = ", ".join(candidates)
    raise ValueError(f"{context} requires one of these columns: {joined}")


def _waterbirds_features(config: dict[str, Any]) -> DatasetBundle:
    import pandas as pd

    path = Path(str(config.get("path", ""))).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Waterbirds feature table {path} does not exist. Provide a local CSV with "
            "split, label, place/background, and feature columns."
        )
    frame = pd.read_csv(path)
    columns = set(frame.columns)
    split_col = _first_existing(columns, ("split", "fold"), "Waterbirds feature adapter")
    label_col = _first_existing(columns, ("y", "label", "target", "bird_label"), "Waterbirds feature adapter")
    env_col = _first_existing(columns, ("place", "background", "env", "spurious"), "Waterbirds feature adapter")
    group_col = next((col for col in ("group", "group_id") if col in columns), None)
    feature_cols = [
        col
        for col in frame.columns
        if col.startswith("feature_") or col.startswith("x")
    ]
    if not feature_cols:
        ignored = {split_col, label_col, env_col}
        if group_col is not None:
            ignored.add(group_col)
        feature_cols = [
            col
            for col in frame.columns
            if col not in ignored and pd.api.types.is_numeric_dtype(frame[col])
        ]
    if not feature_cols:
        raise ValueError("Waterbirds feature adapter could not infer any numeric feature columns.")

    splits: dict[str, dict[str, torch.Tensor]] = {}
    for split_name in ("train", "val", "test"):
        part = frame[frame[split_col].astype(str).str.lower() == split_name]
        if part.empty:
            raise ValueError(f"Waterbirds feature table has no rows for split {split_name!r}.")
        y = part[label_col].astype(int).to_numpy()
        env = part[env_col].astype(int).to_numpy()
        if group_col is None:
            group = env * 2 + y
        else:
            group = part[group_col].astype(int).to_numpy()
        splits[split_name] = {
            "x": torch.tensor(part[feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.long),
            "env": torch.tensor(env, dtype=torch.long),
            "group": torch.tensor(group, dtype=torch.long),
        }
    return DatasetBundle(
        name="waterbirds_features",
        task="classification",
        splits=splits,
        input_dim=len(feature_cols),
        output_dim=int(frame[label_col].max()) + 1,
        causal_mask=None,
        metadata={
            "fixture": False,
            "modality": "features",
            "source_path": str(path),
            "feature_columns": feature_cols,
            "label_column": label_col,
            "environment_column": env_col,
            "group_column": group_col,
        },
    )


DATASETS: dict[str, DatasetFactory] = {
    "synthetic_linear": lambda config: _spurious_tabular(config, nonlinear=False),
    "synthetic_nonlinear": lambda config: _spurious_tabular(config, nonlinear=True),
    "dsprites_tiny": lambda config: _factor_fixture(config, "dsprites_tiny"),
    "causal3d_tiny": lambda config: _factor_fixture(config, "causal3d_tiny", factors=8),
    "waterbirds_tiny": _waterbirds_fixture,
    "shapes_spurious_tiny": lambda config: _factor_fixture(config, "shapes_spurious_tiny"),
    "text_toy": lambda config: _sequence_fixture(config, "text_toy"),
    "fewshot_ner_tiny": lambda config: _sequence_fixture(config, "fewshot_ner_tiny", ner=True),
    "waterbirds_features": _waterbirds_features,
    "local": _local_dataset,
}


def load_dataset(config: dict[str, Any]) -> DatasetBundle:
    dataset_config = dict(config.get("dataset", {}))
    kind = str(dataset_config.get("kind", "synthetic_linear"))
    if "seed" not in dataset_config:
        dataset_config["seed"] = config.get("seed", 0)
    try:
        return DATASETS[kind](dataset_config)
    except KeyError as exc:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset kind {kind!r}. Known datasets: {known}") from exc


def write_fixture_npz(output_dir: str | Path, seed: int = 0) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for kind in sorted(k for k in DATASETS if k != "local"):
        bundle = DATASETS[kind]({"seed": seed, "n": 120}) 
        path = output / f"{kind}.npz"
        arrays: dict[str, np.ndarray] = {}
        for split, tensors in bundle.splits.items():
            for key, value in tensors.items():
                arrays[f"{split}_{key}"] = value.detach().cpu().numpy()
        if bundle.causal_mask is not None:
            arrays["causal_mask"] = bundle.causal_mask.detach().cpu().numpy()
        np.savez(path, **arrays)
        written.append(path)
    return written
