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

    def representations(self, x: torch.Tensor) -> torch.Tensor | None:
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

    def representations(self, x: torch.Tensor) -> torch.Tensor | None:
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

    def representations(self, x: torch.Tensor) -> torch.Tensor | None:
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
        )
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(
            *layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for layer in list(self.net.children())[:-1]:
            hidden = layer(hidden)
        return hidden


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
        return self.net(self.encode(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.long().clamp_min(0)
        embedded = self.embedding(tokens)
        return embedded.mean(dim=1)


class FeatureGate(nn.Module):
    def __init__(
        self,
        causal_mask: torch.Tensor,
        *,
        causal_input_weight: float = 1.0,
        nuisance_input_weight: float = 1.0,
        learned: bool = False,
        group_size: int = 1,
        score_prior: torch.Tensor | None = None,
        score_weight: float = 0.0,
        score_conditioned: bool = False,
        contextual: bool = False,
        representation_conditioned: bool = False,
        context_dim: int | None = None,
    ) -> None:
        super().__init__()
        causal_mask = causal_mask.detach().float()
        nuisance_mask = 1.0 - causal_mask
        base_gate = causal_input_weight * causal_mask + nuisance_input_weight * nuisance_mask
        self.register_buffer("base_gate", base_gate)
        self.group_size = max(int(group_size), 1)
        self.score_weight = max(float(score_weight), 0.0)
        self.score_conditioned = bool(score_conditioned)
        self.contextual = bool(contextual)
        self.representation_conditioned = bool(representation_conditioned)
        if score_prior is not None:
            score_prior = score_prior.detach().float()
            if score_prior.shape != base_gate.shape:
                raise ValueError("FeatureGate score_prior must match the causal mask shape.")
            score_min = float(score_prior.min().item())
            score_max = float(score_prior.max().item())
            if score_max <= score_min + 1e-12:
                normalized_scores = torch.ones_like(score_prior)
            else:
                normalized_scores = (score_prior - score_min) / (score_max - score_min)
            self.register_buffer("normalized_scores", normalized_scores)
        else:
            self.register_buffer("normalized_scores", torch.zeros_like(base_gate))
        if score_prior is not None and self.score_weight > 0.0:
            score_scale = 1.0 + causal_mask * self.score_weight * (2.0 * normalized_scores - 1.0)
            self.register_buffer("score_scale", score_scale.clamp_min(0.05))
        else:
            self.register_buffer("score_scale", torch.ones_like(base_gate))
        if learned:
            if self.group_size == 1:
                self.feature_logits = nn.Parameter(torch.zeros_like(base_gate))
            else:
                group_count = (len(base_gate) + self.group_size - 1) // self.group_size
                self.feature_logits = nn.Parameter(torch.zeros(group_count, dtype=base_gate.dtype))
        else:
            self.register_parameter("feature_logits", None)
        if self.score_conditioned:
            parameter_shape = (len(base_gate) + self.group_size - 1) // self.group_size if self.group_size > 1 else len(base_gate)
            self.score_alpha = nn.Parameter(torch.zeros(parameter_shape, dtype=base_gate.dtype))
            self.score_beta = nn.Parameter(torch.zeros(parameter_shape, dtype=base_gate.dtype))
        else:
            self.register_parameter("score_alpha", None)
            self.register_parameter("score_beta", None)
        if self.contextual:
            parameter_shape = (len(base_gate) + self.group_size - 1) // self.group_size if self.group_size > 1 else len(base_gate)
            self.context_alpha = nn.Parameter(torch.zeros(parameter_shape, dtype=base_gate.dtype))
        else:
            self.register_parameter("context_alpha", None)
        if self.representation_conditioned:
            parameter_shape = (len(base_gate) + self.group_size - 1) // self.group_size if self.group_size > 1 else len(base_gate)
            if context_dim is None or int(context_dim) <= 0:
                raise ValueError("FeatureGate representation-conditioned mode requires a positive context_dim.")
            self.context_projection = nn.Linear(int(context_dim), parameter_shape)
        else:
            self.context_projection = None

    def _expanded_feature_logits(self) -> torch.Tensor | None:
        if self.feature_logits is None:
            return None
        if self.group_size == 1:
            return self.feature_logits
        return self.feature_logits.repeat_interleave(self.group_size)[: len(self.base_gate)]

    def _expanded_score_condition_logits(self) -> torch.Tensor | None:
        if self.score_alpha is None or self.score_beta is None:
            return None
        if self.group_size == 1:
            alpha = self.score_alpha
            beta = self.score_beta
        else:
            alpha = self.score_alpha.repeat_interleave(self.group_size)[: len(self.base_gate)]
            beta = self.score_beta.repeat_interleave(self.group_size)[: len(self.base_gate)]
        centered_scores = 2.0 * self.normalized_scores - 1.0
        return alpha * centered_scores + beta

    def _group_context(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.context_alpha is None:
            return None
        if x.ndim != 2:
            return None
        activations = x.abs()
        if self.group_size == 1:
            grouped = activations
            alpha = self.context_alpha
        else:
            group_count = len(self.context_alpha)
            target_dim = group_count * self.group_size
            if activations.shape[1] < target_dim:
                activations = F.pad(activations, (0, target_dim - activations.shape[1]))
            grouped = activations[:, :target_dim].reshape(len(x), group_count, self.group_size).mean(dim=2)
            alpha = self.context_alpha
        centered = grouped - grouped.mean(dim=1, keepdim=True)
        normalized = centered / grouped.std(dim=1, keepdim=True).clamp_min(1e-6)
        dynamic = normalized * alpha.unsqueeze(0)
        if self.group_size == 1:
            return dynamic
        return dynamic.repeat_interleave(self.group_size, dim=1)[:, : len(self.base_gate)]

    def _representation_condition_logits(self, context: torch.Tensor | None) -> torch.Tensor | None:
        if self.context_projection is None or context is None:
            return None
        logits = self.context_projection(context)
        if self.group_size == 1:
            return logits
        return logits.repeat_interleave(self.group_size, dim=1)[:, : len(self.base_gate)]

    def gate(self, x: torch.Tensor | None = None, context: torch.Tensor | None = None) -> torch.Tensor:
        gate = self.base_gate * self.score_scale
        feature_logits = self._expanded_feature_logits()
        score_condition_logits = self._expanded_score_condition_logits()
        if score_condition_logits is not None:
            feature_logits = score_condition_logits if feature_logits is None else feature_logits + score_condition_logits
        context_logits = self._group_context(x) if x is not None else None
        representation_logits = self._representation_condition_logits(context)
        if representation_logits is not None:
            context_logits = representation_logits if context_logits is None else context_logits + representation_logits
        if context_logits is not None:
            gate = gate.unsqueeze(0).expand(len(context_logits), -1)
            if feature_logits is not None:
                context_logits = context_logits + feature_logits.unsqueeze(0)
            return gate * (2.0 * torch.sigmoid(context_logits))
        if feature_logits is not None:
            gate = gate * (2.0 * torch.sigmoid(feature_logits))
        return gate

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        return x * self.gate(x, context=context).to(x.device)


class GatedModel(nn.Module):
    def __init__(self, model: nn.Module, input_gate: FeatureGate) -> None:
        super().__init__()
        self.model = model
        self.input_gate = input_gate

    def _gate_context(self, x: torch.Tensor) -> torch.Tensor | None:
        if not self.input_gate.representation_conditioned:
            return None
        return self.model.encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.input_gate(x, context=self._gate_context(x)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(self.input_gate(x, context=self._gate_context(x)))

    def feature_importance(self) -> torch.Tensor | None:
        first = next((m for m in self.model.modules() if isinstance(m, nn.Linear)), None)
        if first is None:
            return None
        weight_importance = first.weight.detach().abs().mean(dim=0)
        gate = self.input_gate.gate().detach().to(weight_importance.device)
        return (weight_importance * gate).cpu()


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


def _scheduled_adv_weight(
    base_weight: float,
    *,
    epoch_idx: int,
    total_epochs: int,
    schedule: str,
    warmup_frac: float,
) -> float:
    if base_weight <= 0:
        return 0.0
    if total_epochs <= 1:
        return base_weight
    schedule_name = schedule.strip().lower()
    if not schedule_name or schedule_name == "constant":
        return base_weight

    progress = epoch_idx / max(total_epochs - 1, 1)
    warmup = min(max(warmup_frac, 0.0), 0.95)
    if progress <= warmup:
        scaled = 0.0
    else:
        scaled = (progress - warmup) / max(1.0 - warmup, 1e-12)

    if schedule_name == "linear":
        return base_weight * min(max(scaled, 0.0), 1.0)
    raise ValueError(f"Unknown adversarial schedule {schedule!r}.")


def _make_nuisance_head(
    repr_dim: int,
    output_dim: int,
    method: dict[str, Any],
) -> nn.Module:
    hidden_dim = int(method.get("nuisance_hidden_dim", 0))
    dropout = float(method.get("nuisance_dropout", 0.0))
    if hidden_dim <= 0:
        return nn.Linear(repr_dim, output_dim)

    layers: list[nn.Module] = [
        nn.Linear(repr_dim, hidden_dim),
        nn.ReLU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


@dataclass
class TorchClassifier:
    model: nn.Module
    device: torch.device

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            return self.model(x.to(self.device)).cpu()

    def feature_importance(self) -> torch.Tensor | None:
        if hasattr(self.model, "feature_importance"):
            importance = self.model.feature_importance()
            if importance is not None:
                return importance
        if isinstance(self.model, SequenceClassifier):
            return None
        first = next((m for m in self.model.modules() if isinstance(m, nn.Linear)), None)
        if first is None:
            return None
        return first.weight.detach().abs().mean(dim=0).cpu()

    def representations(self, x: torch.Tensor) -> torch.Tensor | None:
        if not hasattr(self.model, "encode"):
            return None
        self.model.eval()
        with torch.no_grad():
            return self.model.encode(x.to(self.device)).cpu()


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
        return SequenceClassifier(
            vocab_size=int((bundle.metadata or {}).get("vocab_size", 12)),
            output_dim=bundle.output_dim,
            hidden_dim=int(method.get("hidden_dim", 64)),
            embedding_dim=int(method.get("embedding_dim", 16)),
        )
    model = MLP(
        bundle.input_dim,
        bundle.output_dim,
        hidden_dim=int(method.get("hidden_dim", 64)),
        dropout=float(method.get("dropout", 0.0)),
    )
    if bundle.causal_mask is None or not _uses_learned_input_gate(method):
        return model
    return GatedModel(
        model,
        FeatureGate(
            bundle.causal_mask,
            causal_input_weight=float(method.get("causal_input_weight", 1.0)),
            nuisance_input_weight=float(method.get("nuisance_input_weight", 1.0)),
            learned=True,
            group_size=int(method.get("input_gate_group_size", 1)),
            score_prior=_feature_score_prior(bundle, method),
            score_weight=float(method.get("input_gate_score_weight", 0.0)),
            score_conditioned=bool(method.get("input_gate_score_conditioned", False)),
            contextual=bool(method.get("input_gate_contextual", False)),
            representation_conditioned=bool(method.get("input_gate_representation_conditioned", False)),
            context_dim=int(method.get("hidden_dim", 64)),
        ),
    )


def _uses_learned_input_gate(method: dict[str, Any]) -> bool:
    gate_name = str(method.get("input_gate", "fixed")).strip().lower()
    if gate_name in {"learned", "feature_gate", "learned_feature_gate"}:
        return True
    return bool(method.get("learned_input_gate", False))


def _make_fixed_input_gate(bundle: DatasetBundle, method: dict[str, Any]) -> FeatureGate | None:
    if bundle.causal_mask is None or _is_sequence(bundle):
        return None
    return FeatureGate(
        bundle.causal_mask,
        causal_input_weight=float(method.get("causal_input_weight", 1.0)),
        nuisance_input_weight=float(method.get("nuisance_input_weight", 1.0)),
        learned=False,
        score_prior=_feature_score_prior(bundle, method),
        score_weight=float(method.get("input_gate_score_weight", 0.0)),
        score_conditioned=bool(method.get("input_gate_score_conditioned", False)),
        contextual=bool(method.get("input_gate_contextual", False)),
        representation_conditioned=bool(method.get("input_gate_representation_conditioned", False)),
        context_dim=int(method.get("hidden_dim", 64)),
    )


def _feature_score_prior(bundle: DatasetBundle, method: dict[str, Any]) -> torch.Tensor | None:
    if not bool(method.get("input_gate_use_scores", False)):
        return None
    metadata = bundle.metadata or {}
    values = metadata.get("causal_feature_scores")
    if not values:
        return None
    prior = torch.tensor(values, dtype=torch.float32)
    if bundle.causal_mask is not None and prior.shape != bundle.causal_mask.shape:
        return None
    return prior


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
    train = bundle.split("train")
    indices = torch.arange(len(train["y"]))
    return _fit_minibatch_classifier_subset(
        bundle,
        config,
        indices,
        balanced_groups=balanced_groups,
    )


def _fit_minibatch_classifier_subset(
    bundle: DatasetBundle,
    config: dict[str, Any],
    indices: torch.Tensor,
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
    x = train["x"][indices]
    y = train["y"][indices]
    dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(seed)
    if balanced_groups:
        groups = train["group"][indices]
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


def fit_jtt(bundle: DatasetBundle, config: dict[str, Any]) -> TorchClassifier:
    method = dict(config.get("method", {}))
    training = dict(config.get("training", {}))
    seed = int(config.get("seed", 0))
    stage1_epochs = int(method.get("stage1_epochs", max(1, int(training.get("epochs", 20)) // 3)))
    jtt_folds = max(2, int(method.get("jtt_folds", 4)))
    upweight = float(method.get("upweight", 5.0))
    stage1_config = {
        **config,
        "training": {**training, "epochs": stage1_epochs},
    }
    train = bundle.split("train")
    train_size = len(train["y"])
    fold_count = min(jtt_folds, train_size)
    permutation = torch.randperm(train_size, generator=torch.Generator().manual_seed(seed))
    fold_ids = torch.arange(train_size) % fold_count
    pred = torch.empty(train_size, dtype=torch.long)
    for fold in range(fold_count):
        holdout_idx = permutation[fold_ids == fold]
        fit_idx = permutation[fold_ids != fold]
        stage1 = _fit_minibatch_classifier_subset(
            bundle,
            stage1_config,
            fit_idx,
            balanced_groups=False,
        )
        with torch.no_grad():
            pred[holdout_idx] = stage1.predict(train["x"][holdout_idx]).argmax(dim=1).cpu()
    weights = torch.ones(len(pred), dtype=torch.float32)
    weights[pred != train["y"]] = upweight

    torch.manual_seed(seed)
    device = _device(str(training.get("device", "auto")))
    model = _make_model(bundle, method).to(device)
    dataset = TensorDataset(train["x"], train["y"])
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 64)),
        sampler=WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(seed + 1),
        ),
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


def fit_adversarial_probe(bundle: DatasetBundle, config: dict[str, Any]) -> TorchClassifier:
    method = dict(config.get("method", {}))
    training = dict(config.get("training", {}))
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    device = _device(str(training.get("device", "auto")))
    model = _make_model(bundle, method).to(device)
    train = bundle.split("train")
    dataset = TensorDataset(train["x"], train["y"], train["env"])
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 64)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    env_dim = int(train["env"].max().item()) + 1
    repr_dim = int(model.encode(train["x"][:1].to(device)).shape[1])
    nuisance_head = _make_nuisance_head(repr_dim, env_dim, method).to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(nuisance_head.parameters()),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training.get("epochs", 30))
    adv_weight = float(method.get("adv_weight", 0.2))
    nuisance_loss_weight = float(method.get("nuisance_loss_weight", 1.0))
    clip_grad = float(training.get("clip_grad", 1.0))
    for _ in range(epochs):
        model.train()
        nuisance_head.train()
        for xb, yb, envb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            envb = envb.to(device)
            opt.zero_grad(set_to_none=True)
            z = model.encode(xb)
            label_loss = F.cross_entropy(model(xb), yb)
            nuisance_loss = F.cross_entropy(nuisance_head(_grad_reverse(z, adv_weight)), envb)
            loss = label_loss + nuisance_loss_weight * nuisance_loss
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(nuisance_head.parameters()),
                    clip_grad,
                )
            opt.step()
    return TorchClassifier(model, device)


def _counterfactual_batch(bundle: DatasetBundle, xb: torch.Tensor) -> torch.Tensor:
    if bundle.causal_mask is None:
        raise ValueError("Counterfactual batch construction requires dataset.causal_mask.")
    device = xb.device
    perm = torch.randperm(len(xb), device=device)
    if _is_sequence(bundle):
        x_cf = xb.clone()
        conf_pos = int((bundle.metadata or {}).get("confounder_position", -1))
        if conf_pos >= 0:
            x_cf[:, conf_pos] = xb[perm, conf_pos]
            return x_cf
    causal_mask = bundle.causal_mask.to(device)
    nuisance_mask = 1.0 - causal_mask
    return xb * causal_mask + xb[perm] * nuisance_mask


def _counterfactual_disagreement_weights(
    logits: torch.Tensor,
    cf_logits: torch.Tensor,
    *,
    scale: float,
    floor: float,
) -> torch.Tensor:
    if scale <= 0:
        return torch.ones(len(logits), device=logits.device, dtype=logits.dtype)
    probs = F.softmax(logits.detach(), dim=1)
    cf_probs = F.softmax(cf_logits.detach(), dim=1)
    disagreement = (probs - cf_probs).abs().mean(dim=1)
    mean_disagreement = disagreement.mean()
    if torch.allclose(mean_disagreement, torch.zeros_like(mean_disagreement)):
        return torch.ones_like(disagreement)
    normalized = disagreement / mean_disagreement.clamp_min(1e-6)
    floor_value = min(max(float(floor), 0.0), 1.0)
    weights = (1.0 - scale) + scale * normalized
    return weights.clamp_min(floor_value)


def _update_instability_ema(
    ema: torch.Tensor,
    indices: torch.Tensor,
    disagreement: torch.Tensor,
    *,
    decay: float,
) -> torch.Tensor:
    updated = ema.clone()
    decay_value = min(max(float(decay), 0.0), 0.9999)
    updated[indices] = decay_value * updated[indices] + (1.0 - decay_value) * disagreement.detach().to(updated.device)
    return updated


def _select_replay_indices(
    ema: torch.Tensor,
    indices: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    if len(indices) == 0:
        return indices
    fraction_value = min(max(float(fraction), 0.0), 1.0)
    if fraction_value <= 0.0:
        return indices[:0]
    count = max(1, int(round(len(indices) * fraction_value)))
    ema_values = ema[indices].to(indices.device)
    order = torch.argsort(ema_values, descending=True)
    return indices[order[:count]]


def _instability_sample_weights(
    instability: torch.Tensor,
    *,
    top_fraction: float,
    upweight: float,
) -> torch.Tensor:
    weights = torch.ones(len(instability), dtype=torch.float32)
    fraction = min(max(float(top_fraction), 0.0), 1.0)
    factor = max(float(upweight), 1.0)
    if fraction <= 0.0 or factor <= 1.0 or len(instability) == 0:
        return weights
    count = max(1, int(round(len(instability) * fraction)))
    top_idx = torch.argsort(instability, descending=True)[:count]
    weights[top_idx] = factor
    return weights


def _estimate_counterfactual_instability(
    model: TorchClassifier,
    bundle: DatasetBundle,
    *,
    passes: int,
    seed: int,
    score_mode: str = "mean",
) -> torch.Tensor:
    train = bundle.split("train")
    xb = train["x"]
    yb = train["y"]
    group = train.get("group")
    pass_count = max(1, int(passes))
    logits = model.predict(xb)
    per_pass_scores: list[torch.Tensor] = []
    factual_losses = F.cross_entropy(logits, yb.to(logits.device), reduction="none")
    per_pass_excess_losses: list[torch.Tensor] = []
    for pass_idx in range(pass_count):
        torch.manual_seed(seed + pass_idx)
        x_cf = _counterfactual_batch(bundle, xb)
        cf_logits = model.predict(x_cf)
        disagreement = (F.softmax(logits, dim=1) - F.softmax(cf_logits, dim=1)).abs().mean(dim=1)
        per_pass_scores.append(disagreement.cpu())
        cf_losses = F.cross_entropy(cf_logits, yb.to(cf_logits.device), reduction="none")
        per_pass_excess_losses.append((cf_losses - factual_losses).clamp_min(0.0).detach().cpu())
    stacked = torch.stack(per_pass_scores)
    mean_scores = stacked.mean(dim=0)
    mean_excess_losses = torch.stack(per_pass_excess_losses).mean(dim=0)
    mode = str(score_mode).strip().lower()
    if mode in {"", "mean"}:
        return mean_scores
    if mode in {"counterfactual_loss_increase_mean", "cf_loss_increase_mean", "loss_delta_mean"}:
        return mean_excess_losses
    if mode in {
        "group_loss_weighted_counterfactual_loss_increase_mean",
        "group_loss_weighted_cf_loss_increase_mean",
        "group_loss_weighted_loss_delta_mean",
    }:
        if group is None:
            return mean_excess_losses
        losses = factual_losses.detach().cpu()
        group_cpu = group.detach().cpu().long()
        group_weights = torch.ones(len(losses), dtype=torch.float32)
        overall_mean = losses.mean().clamp_min(1e-6)
        for group_id in torch.unique(group_cpu):
            group_mask = group_cpu == group_id
            group_mean = losses[group_mask].mean().clamp_min(1e-6)
            group_weights[group_mask] = group_mean / overall_mean
        return mean_excess_losses * group_weights
    if mode in {"loss_weighted_mean", "hardness_weighted_mean"}:
        losses = factual_losses.detach().cpu()
        normalized_losses = losses / losses.mean().clamp_min(1e-6)
        return mean_scores * normalized_losses
    if mode in {"group_loss_weighted_mean", "group_conditional_failure", "group_failure_weighted_mean"}:
        if group is None:
            return mean_scores
        losses = F.cross_entropy(logits, yb.to(logits.device), reduction="none").detach().cpu()
        group_cpu = group.detach().cpu().long()
        group_weights = torch.ones(len(losses), dtype=torch.float32)
        overall_mean = losses.mean().clamp_min(1e-6)
        for group_id in torch.unique(group_cpu):
            group_mask = group_cpu == group_id
            group_mean = losses[group_mask].mean().clamp_min(1e-6)
            group_weights[group_mask] = group_mean / overall_mean
        return mean_scores * group_weights
    if mode in {"mean_minus_std", "lower_confidence_bound", "lcb"}:
        if pass_count == 1:
            return mean_scores
        penalty = stacked.std(dim=0, unbiased=False)
        return (mean_scores - penalty).clamp_min(0.0)
    raise ValueError(f"Unknown counterfactual instability score mode: {score_mode}")


def _apply_causal_input_gate(
    bundle: DatasetBundle,
    xb: torch.Tensor,
    method: dict[str, Any],
) -> torch.Tensor:
    if bundle.causal_mask is None or _is_sequence(bundle) or _uses_learned_input_gate(method):
        return xb
    input_gate = _make_fixed_input_gate(bundle, method)
    if input_gate is None:
        return xb
    gate = input_gate.gate()
    if torch.allclose(gate, torch.ones_like(gate)):
        return xb
    return input_gate(xb)


def fit_counterfactual_adversarial(bundle: DatasetBundle, config: dict[str, Any]) -> TorchClassifier:
    if bundle.causal_mask is None:
        raise ValueError("Counterfactual adversarial training requires dataset.causal_mask.")
    method = dict(config.get("method", {}))
    training = dict(config.get("training", {}))
    seed = int(config.get("seed", 0))
    stage1_epochs = int(method.get("counterfactual_instability_stage1_epochs", 0))
    instability_passes = int(method.get("counterfactual_instability_passes", 3))
    instability_score_mode = str(method.get("counterfactual_instability_score_mode", "mean"))
    instability_top_fraction = float(method.get("counterfactual_instability_top_fraction", 0.0))
    instability_upweight = float(method.get("counterfactual_instability_upweight", 1.0))
    sample_weights: torch.Tensor | None = None
    if stage1_epochs > 0 and instability_top_fraction > 0.0 and instability_upweight > 1.0:
        stage1_method = dict(method)
        stage1_method["counterfactual_instability_stage1_epochs"] = 0
        stage1_method["counterfactual_instability_top_fraction"] = 0.0
        stage1_method["counterfactual_instability_upweight"] = 1.0
        stage1_method["counterfactual_instability_replay_fraction"] = 0.0
        stage1_method["counterfactual_instability_replay_weight"] = 0.0
        stage1_method["counterfactual_disagreement_weight"] = 0.0
        stage1_config = {
            **config,
            "method": stage1_method,
            "training": {**training, "epochs": stage1_epochs},
        }
        stage1_model = fit_counterfactual_adversarial(bundle, stage1_config)
        instability = _estimate_counterfactual_instability(
            stage1_model,
            bundle,
            passes=instability_passes,
            seed=seed,
            score_mode=instability_score_mode,
        )
        sample_weights = _instability_sample_weights(
            instability,
            top_fraction=instability_top_fraction,
            upweight=instability_upweight,
        )
    torch.manual_seed(seed)
    device = _device(str(training.get("device", "auto")))
    model = _make_model(bundle, method).to(device)
    train = bundle.split("train")
    train_indices = torch.arange(len(train["y"]))
    dataset = TensorDataset(train["x"], train["y"], train["env"], train_indices)
    batch_size = int(training.get("batch_size", 64))
    loader_generator = torch.Generator().manual_seed(seed)
    if sample_weights is not None:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
                generator=torch.Generator().manual_seed(seed + 1),
            ),
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=loader_generator,
        )
    env_dim = int(train["env"].max().item()) + 1
    repr_dim = int(model.encode(train["x"][:1].to(device)).shape[1])
    nuisance_head = _make_nuisance_head(repr_dim, env_dim, method).to(device)
    model_opt = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    nuisance_steps = max(0, int(method.get("nuisance_steps", 0)))
    nuisance_lr_scale = float(method.get("nuisance_lr_scale", 1.0))
    nuisance_opt = torch.optim.Adam(
        nuisance_head.parameters(),
        lr=float(training.get("lr", 1e-3)) * nuisance_lr_scale,
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    epochs = int(training.get("epochs", 30))
    adv_weight = float(method.get("adv_weight", 0.05))
    adv_schedule = str(method.get("adv_schedule", "constant"))
    adv_warmup_frac = float(method.get("adv_warmup_frac", 0.0))
    nuisance_loss_weight = float(method.get("nuisance_loss_weight", 1.0))
    consistency_weight = float(method.get("consistency_weight", 0.2))
    disagreement_weight = float(method.get("counterfactual_disagreement_weight", 0.0))
    disagreement_floor = float(method.get("counterfactual_disagreement_floor", 0.5))
    instability_replay_fraction = float(method.get("counterfactual_instability_replay_fraction", 0.0))
    instability_replay_weight = float(method.get("counterfactual_instability_replay_weight", 0.0))
    instability_ema_decay = float(method.get("counterfactual_instability_ema_decay", 0.9))
    instability_ema = torch.zeros(len(train["y"]), dtype=torch.float32)
    clip_grad = float(training.get("clip_grad", 1.0))
    for epoch_idx in range(epochs):
        model.train()
        nuisance_head.train()
        scheduled_adv_weight = _scheduled_adv_weight(
            adv_weight,
            epoch_idx=epoch_idx,
            total_epochs=epochs,
            schedule=adv_schedule,
            warmup_frac=adv_warmup_frac,
        )
        for xb, yb, envb, idxb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            envb = envb.to(device)
            idxb = idxb.long()
            x_cf = _counterfactual_batch(bundle, xb)
            x_main = _apply_causal_input_gate(bundle, xb, method)
            x_cf_main = _apply_causal_input_gate(bundle, x_cf, method)
            if nuisance_steps > 0:
                for _ in range(nuisance_steps):
                    nuisance_opt.zero_grad(set_to_none=True)
                    z_detached = model.encode(x_main).detach()
                    nuisance_train_loss = F.cross_entropy(nuisance_head(z_detached), envb)
                    nuisance_train_loss.backward()
                    if clip_grad > 0:
                        nn.utils.clip_grad_norm_(list(nuisance_head.parameters()), clip_grad)
                    nuisance_opt.step()

            model_opt.zero_grad(set_to_none=True)
            nuisance_opt.zero_grad(set_to_none=True)
            logits = model(x_main)
            cf_logits = model(x_cf_main)
            z = model.encode(x_main)
            sample_weights = _counterfactual_disagreement_weights(
                logits,
                cf_logits,
                scale=disagreement_weight,
                floor=disagreement_floor,
            )
            current_disagreement = (F.softmax(logits.detach(), dim=1) - F.softmax(cf_logits.detach(), dim=1)).abs().mean(dim=1)
            instability_ema = _update_instability_ema(
                instability_ema,
                idxb,
                current_disagreement,
                decay=instability_ema_decay,
            )
            label_loss = (
                sample_weights * F.cross_entropy(logits, yb, reduction="none")
                + sample_weights * F.cross_entropy(cf_logits, yb, reduction="none")
            ).mean()
            consistency = (
                sample_weights
                * F.mse_loss(F.softmax(logits, dim=1), F.softmax(cf_logits, dim=1), reduction="none").mean(dim=1)
            ).mean()
            nuisance_loss = F.cross_entropy(nuisance_head(_grad_reverse(z, scheduled_adv_weight)), envb)
            replay_loss = torch.zeros((), device=device)
            replay_indices = _select_replay_indices(
                instability_ema,
                idxb,
                fraction=instability_replay_fraction,
            )
            if len(replay_indices) > 0 and instability_replay_weight > 0.0:
                replay_mask = torch.isin(idxb, replay_indices.to(idxb.device))
                replay_logits = logits[replay_mask]
                replay_cf_logits = cf_logits[replay_mask]
                replay_targets = yb[replay_mask]
                replay_loss = (
                    F.cross_entropy(replay_logits, replay_targets)
                    + F.cross_entropy(replay_cf_logits, replay_targets)
                )
            loss = (
                label_loss
                + consistency_weight * consistency
                + nuisance_loss_weight * nuisance_loss
                + instability_replay_weight * replay_loss
            )
            loss.backward()
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(list(model.parameters()), clip_grad)
                nn.utils.clip_grad_norm_(list(nuisance_head.parameters()), clip_grad)
            model_opt.step()
            nuisance_opt.step()
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
            x_cf = _counterfactual_batch(bundle, xb)
            x_main = _apply_causal_input_gate(bundle, xb, method)
            x_cf_main = _apply_causal_input_gate(bundle, x_cf, method)
            opt.zero_grad(set_to_none=True)
            logits = model(x_main)
            cf_logits = model(x_cf_main)
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
    "jtt": fit_jtt,
    "adversarial_probe": fit_adversarial_probe,
    "counterfactual_adversarial": fit_counterfactual_adversarial,
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
