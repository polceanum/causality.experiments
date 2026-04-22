from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetBundle


class FittedModel(Protocol):
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def feature_importance(self) -> torch.Tensor | None:
        ...


@dataclass
class ConstantModel:
    label: int
    output_dim: int

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((len(x), self.output_dim), dtype=torch.float32)
        logits[:, self.label] = 1.0
        return logits

    def feature_importance(self) -> torch.Tensor | None:
        return None


@dataclass
class OracleMaskModel:
    causal_mask: torch.Tensor
    output_dim: int

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        score = (x * self.causal_mask.to(x.device)).sum(dim=1)
        logits = torch.stack([-score, score], dim=1)
        if self.output_dim > 2:
            pad = torch.zeros((len(x), self.output_dim - 2), device=x.device)
            logits = torch.cat([logits, pad], dim=1)
        return logits

    def feature_importance(self) -> torch.Tensor | None:
        return self.causal_mask


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TorchClassifier:
    model: nn.Module
    device: torch.device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x.to(self.device)).cpu()

    def feature_importance(self) -> torch.Tensor | None:
        first = next((m for m in self.model.modules() if isinstance(m, nn.Linear)), None)
        if first is None:
            return None
        return first.weight.detach().abs().mean(dim=0).cpu()


def _device(name: str | None) -> torch.device:
    if name in (None, "auto"):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def fit_constant(bundle: DatasetBundle, config: dict[str, Any]) -> ConstantModel:
    y = bundle.split("train")["y"]
    label = int(torch.mode(y).values.item())
    return ConstantModel(label=label, output_dim=bundle.output_dim)


def fit_oracle(bundle: DatasetBundle, config: dict[str, Any]) -> OracleMaskModel:
    if bundle.causal_mask is None:
        raise ValueError("Oracle baseline requires dataset.causal_mask.")
    return OracleMaskModel(bundle.causal_mask, bundle.output_dim)


def fit_erm(bundle: DatasetBundle, config: dict[str, Any]) -> TorchClassifier:
    method = dict(config.get("method", {}))
    training = dict(config.get("training", {}))
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = _device(str(training.get("device", "auto")))
    model = MLP(
        bundle.input_dim,
        bundle.output_dim,
        hidden_dim=int(method.get("hidden_dim", 64)),
    ).to(device)
    train = bundle.split("train")
    dataset = TensorDataset(train["x"], train["y"])
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 64)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training.get("epochs", 20))
    clip_grad = float(training.get("clip_grad", 1.0))
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
    return TorchClassifier(model, device)


def fit_adapter_only(bundle: DatasetBundle, config: dict[str, Any]) -> FittedModel:
    name = str(config.get("method", {}).get("kind", "adapter"))
    raise NotImplementedError(
        f"Method {name!r} is registered as an adapter contract in v1. "
        "Implement fit/predict against MethodAdapter before using it in runnable configs."
    )


METHODS = {
    "constant": fit_constant,
    "oracle": fit_oracle,
    "erm": fit_erm,
    "irm": fit_adapter_only,
    "counterfactual_augmentation": fit_adapter_only,
    "causal_probe": fit_adapter_only,
    "beta_vae": fit_adapter_only,
    "ivae": fit_adapter_only,
    "citris": fit_adapter_only,
    "csml": fit_adapter_only,
    "deepiv": fit_adapter_only,
}


def fit_method(bundle: DatasetBundle, config: dict[str, Any]) -> FittedModel:
    kind = str(config.get("method", {}).get("kind", "erm"))
    try:
        return METHODS[kind](bundle, config)
    except KeyError as exc:
        known = ", ".join(sorted(METHODS))
        raise ValueError(f"Unknown method kind {kind!r}. Known methods: {known}") from exc
