from __future__ import annotations

import csv
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path


METHOD_SUFFIXES = (
    "_counterfactual_augmentation",
    "_counterfactual_adversarial",
    "_official_causal_shrink_dfr_val_tr",
    "_official_dfr_val_tr",
    "_counterfactual_causal_dfr",
    "_representation_dfr",
    "_causal_dfr",
    "_adversarial_probe",
    "_group_balanced_erm",
    "_group_dro",
    "_loss_weighted_dfr",
    "_dfr",
    "_irm",
    "_jtt",
    "_erm",
)

REPORT_METHODS = {
    "erm",
    "group_balanced_erm",
    "group_dro",
    "irm",
    "jtt",
    "dfr",
    "official_dfr_val_tr",
    "official_causal_shrink_dfr_val_tr",
    "causal_dfr",
    "representation_dfr",
    "adversarial_probe",
    "counterfactual_adversarial",
    "counterfactual_augmentation",
}

PROPOSED_METHODS = {
    "causal_dfr",
    "official_causal_shrink_dfr_val_tr",
    "representation_dfr",
    "counterfactual_adversarial",
    "counterfactual_augmentation",
}

VALIDATION_GROUP_METHODS = {
    "dfr",
    "official_dfr_val_tr",
    "official_causal_shrink_dfr_val_tr",
    "causal_dfr",
    "representation_dfr",
}


def experiment_name(config_name: str) -> str:
    for suffix in METHOD_SUFFIXES:
        if config_name.endswith(suffix):
            return config_name[: -len(suffix)]
        marker = f"{suffix}_"
        if marker in config_name:
            return config_name.split(marker, 1)[0]
    return config_name


def method_family(method: str) -> str:
    return "proposed" if method in PROPOSED_METHODS else "baseline"


def validation_usage(method: str) -> str:
    if method in VALIDATION_GROUP_METHODS:
        return "trains_on_validation_groups"
    return "holdout_validation_metrics"


def is_ad_hoc_config(
    config_name: str,
    *,
    include_sweep_prefix: bool = True,
    include_waterbirds_tune: bool = True,
) -> bool:
    return (
        "_seed" in config_name
        or "_irm_w" in config_name
        or "_w0p" in config_name
        or "_w1p" in config_name
        or (include_sweep_prefix and config_name.startswith("sweep_"))
        or (include_waterbirds_tune and config_name.startswith("waterbirds_tune_"))
    )


def safe_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def metric_value(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def format_metric_value(value: float | None) -> str:
    if value is None or value != value:
        return ""
    return f"{value:.3f}"


def format_row_metric(row: dict[str, str], key: str) -> str:
    return format_metric_value(metric_value(row, key))


def format_reference_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value / 100.0:.3f}"


def format_delta(ours: float | None, reference_percent: float | None) -> str:
    if ours is None or reference_percent is None:
        return ""
    return f"{ours - (reference_percent / 100.0):.3f}"


def delta_to_reference(row: dict[str, str], key: str, reference_percent: float | None) -> str:
    ours = metric_value(row, key)
    if ours != ours:
        return ""
    return format_delta(ours, reference_percent)


def latest_by_config(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        config_name = row.get("config", "")
        if not config_name:
            continue
        if config_name not in latest or row.get("run", "") > latest[config_name].get("run", ""):
            latest[config_name] = row
    return latest


def latest_by_key(
    rows: Iterable[dict[str, str]],
    key_fn: Callable[[dict[str, str]], Hashable],
) -> dict[Hashable, dict[str, str]]:
    latest: dict[Hashable, dict[str, str]] = {}
    for row in rows:
        key = key_fn(row)
        if key not in latest or row.get("run", "") > latest[key].get("run", ""):
            latest[key] = row
    return latest


def write_csv_rows(path: str | Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
