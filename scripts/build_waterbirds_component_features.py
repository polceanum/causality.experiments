from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causality_experiments.component_representation import (
    build_component_clue_rows,
    columns_for_components,
    component_summary_rows,
    component_test_rows,
    feature_columns_from_frame,
    filter_feature_components,
    infer_feature_components,
    load_feature_components_from_manifest,
    ordered_feature_components,
    write_component_manifest,
)
from causality_experiments.data import load_dataset


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def build_component_feature_artifacts(
    *,
    input_csv: Path,
    output_csv: Path,
    output_clues_csv: Path,
    output_summary_csv: Path | None = None,
    output_tests_csv: Path | None = None,
    manifest_path: Path | None = None,
    component_names: tuple[str, ...] = (),
    include_components: tuple[str, ...] = (),
    exclude_components: tuple[str, ...] = (),
    split_name: str = "train",
) -> dict[str, object]:
    frame = pd.read_csv(input_csv)
    feature_columns = feature_columns_from_frame(frame)
    source_manifest = manifest_path or input_csv.with_suffix(input_csv.suffix + ".manifest.json")
    manifest_components = load_feature_components_from_manifest(source_manifest)
    feature_components = (
        ordered_feature_components(feature_columns, component_names)
        if component_names
        else infer_feature_components(feature_columns, manifest_components)
    )
    feature_components = filter_feature_components(
        feature_components,
        include_components=include_components,
        exclude_components=exclude_components,
    )
    feature_columns = columns_for_components(feature_components)
    metadata_columns = [column for column in frame.columns if column not in feature_columns_from_frame(frame)]
    frame = frame.loc[:, metadata_columns + feature_columns]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    write_component_manifest(
        output_csv=output_csv,
        feature_columns=feature_columns,
        feature_components=feature_components,
        source_path=input_csv,
        extra={"component_source_manifest": str(source_manifest) if source_manifest.exists() else ""},
    )
    bundle = load_dataset({"dataset": {"kind": "waterbirds_features", "path": str(output_csv)}})
    clue_rows = build_component_clue_rows(bundle, feature_components=feature_components, split_name=split_name)
    summary_rows = component_summary_rows(bundle, feature_components=feature_components, split_name=split_name)
    test_rows = component_test_rows(bundle, feature_components=feature_components, split_name=split_name)
    _write_rows(output_clues_csv, clue_rows)
    if output_summary_csv is not None:
        _write_rows(output_summary_csv, summary_rows)
    if output_tests_csv is not None:
        _write_rows(output_tests_csv, test_rows)
    return {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "output_clues_csv": str(output_clues_csv),
        "feature_count": len(feature_columns),
        "component_count": len(feature_components),
        "components": {key: len(value) for key, value in feature_components.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-clues-csv", required=True)
    parser.add_argument("--output-summary-csv", default="")
    parser.add_argument("--output-tests-csv", default="")
    parser.add_argument("--manifest-path", default="")
    parser.add_argument("--component-names", nargs="*", default=[])
    parser.add_argument("--include-components", nargs="*", default=[])
    parser.add_argument("--exclude-components", nargs="*", default=[])
    parser.add_argument("--split-name", default="train")
    args = parser.parse_args()
    result = build_component_feature_artifacts(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        output_clues_csv=Path(args.output_clues_csv),
        output_summary_csv=Path(args.output_summary_csv) if args.output_summary_csv else None,
        output_tests_csv=Path(args.output_tests_csv) if args.output_tests_csv else None,
        manifest_path=Path(args.manifest_path) if args.manifest_path else None,
        component_names=tuple(args.component_names),
        include_components=tuple(args.include_components),
        exclude_components=tuple(args.exclude_components),
        split_name=args.split_name,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
