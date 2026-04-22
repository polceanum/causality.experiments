from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    return benchmark


def literature_rows(benchmark_id: str) -> list[LiteratureResult]:
    return LITERATURE_RESULTS.get(benchmark_id, [])


def best_literature_wga(benchmark_id: str) -> float | None:
    rows = [row.worst_group_accuracy for row in literature_rows(benchmark_id)]
    values = [value for value in rows if value is not None]
    return max(values) if values else None
