from __future__ import annotations

from dataclasses import dataclass
from typing import Any


PROVENANCE_FIELDS = (
    "feature_extractor",
    "feature_source",
    "split_definition",
)


@dataclass(frozen=True)
class LiteratureResult:
    benchmark_id: str
    method: str
    worst_group_accuracy: float | None
    average_accuracy: float | None
    source: str
    notes: str


LITERATURE_RESULTS: dict[str, list[LiteratureResult]] = {
    "waterbirds": [
        LiteratureResult(
            benchmark_id="waterbirds",
            method="ERM",
            worst_group_accuracy=75.3,
            average_accuracy=98.1,
            source="Kim et al. 2025 table reproducing original-paper baselines",
            notes="Real Waterbirds, percent units; not comparable to tiny fixture.",
        ),
        LiteratureResult(
            benchmark_id="waterbirds",
            method="JTT",
            worst_group_accuracy=86.7,
            average_accuracy=93.3,
            source="Liu et al. 2021 / reproduced in Kim et al. 2025",
            notes="Real Waterbirds, percent units; JTT does not require training group labels.",
        ),
        LiteratureResult(
            benchmark_id="waterbirds",
            method="GroupDRO",
            worst_group_accuracy=91.4,
            average_accuracy=93.5,
            source="LaBonte et al. 2023 Group Robust Classification",
            notes="Real Waterbirds, percent units; group-aware reference.",
        ),
        LiteratureResult(
            benchmark_id="waterbirds",
            method="DFR",
            worst_group_accuracy=92.9,
            average_accuracy=94.2,
            source="Kim et al. 2025 table of original-paper baselines",
            notes="Real Waterbirds, percent units; uses validation group information.",
        ),
    ]
}


def _has_provenance_value(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in {"todo", "tbd", "unknown", "set_me"}


def benchmark_provenance(config: dict[str, Any]) -> dict[str, str]:
    benchmark = dict(config.get("benchmark", {}))
    provenance = dict(benchmark.get("provenance", {}))
    return {key: str(provenance.get(key, "")).strip() for key in PROVENANCE_FIELDS}


def benchmark_provenance_complete(config: dict[str, Any]) -> bool:
    benchmark = dict(config.get("benchmark", {}))
    if not bool(benchmark.get("comparable_to_literature", False)):
        return False
    provenance = benchmark_provenance(config)
    return all(_has_provenance_value(provenance.get(key)) for key in PROVENANCE_FIELDS)


def benchmark_metadata(config: dict[str, Any]) -> dict[str, Any]:
    benchmark = dict(config.get("benchmark", {}))
    if not benchmark:
        dataset_kind = str(config.get("dataset", {}).get("kind", ""))
        benchmark = {
            "kind": "fixture",
            "id": dataset_kind,
            "comparable_to_literature": False,
        }
    benchmark.setdefault("kind", "fixture")
    benchmark.setdefault("comparable_to_literature", benchmark["kind"] == "real")
    benchmark["provenance"] = benchmark_provenance(config)
    benchmark["provenance_complete"] = benchmark_provenance_complete(config)
    return benchmark


def literature_rows(benchmark_id: str) -> list[LiteratureResult]:
    return LITERATURE_RESULTS.get(benchmark_id, [])


def literature_row_map(benchmark_id: str) -> dict[str, LiteratureResult]:
    return {row.method.lower(): row for row in literature_rows(benchmark_id)}


def best_literature_wga(benchmark_id: str) -> float | None:
    rows = [row.worst_group_accuracy for row in literature_rows(benchmark_id)]
    values = [value for value in rows if value is not None]
    return max(values) if values else None


def literature_wga(benchmark_id: str, method: str) -> float | None:
    row = literature_row_map(benchmark_id).get(method.lower())
    if row is None:
        return None
    return row.worst_group_accuracy


def literature_avg_accuracy(benchmark_id: str, method: str) -> float | None:
    row = literature_row_map(benchmark_id).get(method.lower())
    if row is None:
        return None
    return row.average_accuracy
