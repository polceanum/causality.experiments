from __future__ import annotations

from typing import Any

import torch

from .data import DatasetBundle
from .methods import FittedModel


def _pred_labels(model: FittedModel, x: torch.Tensor) -> torch.Tensor:
    return model.predict(x).argmax(dim=1)


def accuracy(model: FittedModel, split: dict[str, torch.Tensor]) -> float:
    pred = _pred_labels(model, split["x"])
    return float((pred == split["y"]).float().mean().item())


def worst_group_accuracy(model: FittedModel, split: dict[str, torch.Tensor]) -> float:
    pred = _pred_labels(model, split["x"])
    scores: list[float] = []
    for group in torch.unique(split["group"]):
        mask = split["group"] == group
        if int(mask.sum()) > 0:
            scores.append(float((pred[mask] == split["y"][mask]).float().mean().item()))
    return min(scores) if scores else float("nan")


def support_recovery(model: FittedModel, bundle: DatasetBundle) -> float:
    if bundle.causal_mask is None:
        return float("nan")
    importance = model.feature_importance()
    if importance is None:
        return float("nan")
    k = int(bundle.causal_mask.sum().item())
    if k <= 0:
        return float("nan")
    top = torch.zeros_like(bundle.causal_mask)
    top[torch.topk(importance, k=k).indices] = 1.0
    return float((top == bundle.causal_mask).float().mean().item())


def ate_proxy_error(model: FittedModel, bundle: DatasetBundle) -> float:
    metadata = bundle.metadata or {}
    if "cause_position" in metadata:
        idx = int(metadata["cause_position"])
    elif bundle.causal_mask is not None and int(bundle.causal_mask.sum().item()) > 0:
        idx = int(torch.argmax(bundle.causal_mask).item())
    else:
        return float("nan")
    split = bundle.split("test")
    x0 = split["x"].clone()
    x1 = split["x"].clone()
    x0[:, idx] = 0.0
    x1[:, idx] = 1.0
    p0 = torch.softmax(model.predict(x0), dim=1)[:, 1]
    p1 = torch.softmax(model.predict(x1), dim=1)[:, 1]
    estimated = float((p1 - p0).mean().item())
    return abs(1.0 - estimated)


def evaluate(model: FittedModel, bundle: DatasetBundle, config: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for split_name in ("train", "val", "test"):
        split = bundle.split(split_name)
        out[f"{split_name}/accuracy"] = accuracy(model, split)
        out[f"{split_name}/worst_group_accuracy"] = worst_group_accuracy(model, split)
    out["support_recovery"] = support_recovery(model, bundle)
    out["ate_proxy_error"] = ate_proxy_error(model, bundle)
    return out
