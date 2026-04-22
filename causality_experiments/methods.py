from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        hidden_dim: int = 64,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.long().clamp_min(0)
        embedded = self.embedding(tokens)
        pooled = embedded.mean(dim=1)
        return self.net(pooled)


@dataclass
class TorchClassifier:
    model: nn.Module
    device: torch.device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x.to(self.device)).cpu()

    def feature_importance(self) -> torch.Tensor | None:
        if isinstance(self.model, SequenceClassifier):
            return None
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


def _is_sequence(bundle: DatasetBundle) -> bool:
    return bool((bundle.metadata or {}).get("modality") == "sequence")


def _make_model(bundle: DatasetBundle, method: dict[str, Any]) -> nn.Module:
    if _is_sequence(bundle):
        vocab_size = int((bundle.metadata or {}).get("vocab_size", 12))
        return SequenceClassifier(
            vocab_size=vocab_size,
            output_dim=bundle.output_dim,
            hidden_dim=int(method.get("hidden_dim", 64)),
            embedding_dim=int(method.get("embedding_dim", 16)),
        )
    return MLP(
        bundle.input_dim,
        bundle.output_dim,
        hidden_dim=int(method.get("hidden_dim", 64)),
    )


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
    model = _make_model(bundle, method).to(device)
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


def _fit_minibatch_classifier(
    bundle: DatasetBundle,
    config: dict[str, Any],
    *,
    balanced_groups: bool = False,
) -> TorchClassifier:
    method = dict(config.get("method", {}))
    training = dict(config.get("training", {}))
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = _device(str(training.get("device", "auto")))
    model = _make_model(bundle, method).to(device)
    train = bundle.split("train")
    dataset = TensorDataset(train["x"], train["y"])
    generator = torch.Generator().manual_seed(seed)
    if balanced_groups:
        groups = train["group"]
        counts = torch.bincount(groups)
        weights = 1.0 / counts.clamp_min(1).float()[groups]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(training.get("batch_size", 64)),
            sampler=sampler,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=int(training.get("batch_size", 64)),
            shuffle=True,
            generator=generator,
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
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
    return TorchClassifier(model, device)


def fit_group_balanced_erm(bundle: DatasetBundle, config: dict[str, Any]) -> TorchClassifier:
    return _fit_minibatch_classifier(bundle, config, balanced_groups=True)


def fit_group_dro(bundle: DatasetBundle, config: dict[str, Any]) -> TorchClassifier:
    method = dict(config.get("method", {}))
    training = dict(config.get("training", {}))
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = _device(str(training.get("device", "auto")))
    model = _make_model(bundle, method).to(device)
    train = bundle.split("train")
    x = train["x"].to(device)
    y = train["y"].to(device)
    group = train["group"].to(device)
    groups = torch.unique(group)
    group_weights = torch.ones(len(groups), device=device) / len(groups)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training.get("epochs", 60))
    eta = float(method.get("dro_eta", 0.1))
    clip_grad = float(training.get("clip_grad", 1.0))
    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        losses = []
        for group_id in groups:
            mask = group == group_id
            losses.append(F.cross_entropy(model(x[mask]), y[mask]))
        group_losses = torch.stack(losses)
        with torch.no_grad():
            group_weights *= torch.exp(eta * group_losses.detach())
            group_weights /= group_weights.sum().clamp_min(1e-12)
        loss = torch.sum(group_weights.detach() * group_losses)
        loss.backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        opt.step()
    return TorchClassifier(model, device)


def fit_counterfactual_augmentation(bundle: DatasetBundle, config: dict[str, Any]) -> TorchClassifier:
    if bundle.causal_mask is None:
        raise ValueError("Counterfactual augmentation requires dataset.causal_mask.")
    method = dict(config.get("method", {}))
    training = dict(config.get("training", {}))
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = _device(str(training.get("device", "auto")))
    model = _make_model(bundle, method).to(device)
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
    causal_mask = bundle.causal_mask.to(device)
    nuisance_mask = 1.0 - causal_mask
    consistency_weight = float(method.get("consistency_weight", 0.2))
    epochs = int(training.get("epochs", 20))
    clip_grad = float(training.get("clip_grad", 1.0))
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            perm = torch.randperm(len(xb), device=device)
            if _is_sequence(bundle):
                x_cf = xb.clone()
                conf_pos = int((bundle.metadata or {}).get("confounder_position", -1))
                if conf_pos >= 0:
                    x_cf[:, conf_pos] = xb[perm, conf_pos]
                else:
                    x_cf = xb * causal_mask + xb[perm] * nuisance_mask
            else:
                x_cf = xb * causal_mask + xb[perm] * nuisance_mask
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            cf_logits = model(x_cf)
            label_loss = F.cross_entropy(logits, yb) + F.cross_entropy(cf_logits, yb)
            consistency = F.mse_loss(F.softmax(logits, dim=1), F.softmax(cf_logits, dim=1))
            loss = label_loss + consistency_weight * consistency
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
    return TorchClassifier(model, device)


def _irm_penalty(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)


def fit_irm(bundle: DatasetBundle, config: dict[str, Any]) -> TorchClassifier:
    method = dict(config.get("method", {}))
    training = dict(config.get("training", {}))
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = _device(str(training.get("device", "auto")))
    model = _make_model(bundle, method).to(device)
    train = bundle.split("train")
    x = train["x"].to(device)
    y = train["y"].to(device)
    env = train["env"].to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training.get("epochs", 100))
    penalty_weight = float(method.get("penalty_weight", 100.0))
    anneal_epochs = int(method.get("anneal_epochs", 10))
    clip_grad = float(training.get("clip_grad", 1.0))
    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        risks = []
        penalties = []
        for env_id in torch.unique(env):
            mask = env == env_id
            logits = model(x[mask])
            risks.append(F.cross_entropy(logits, y[mask]))
            penalties.append(_irm_penalty(logits, y[mask]))
        risk = torch.stack(risks).mean()
        penalty = torch.stack(penalties).mean()
        weight = penalty_weight if epoch >= anneal_epochs else 1.0
        loss = risk + weight * penalty
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
    "group_balanced_erm": fit_group_balanced_erm,
    "group_dro": fit_group_dro,
    "irm": fit_irm,
    "counterfactual_augmentation": fit_counterfactual_augmentation,
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
