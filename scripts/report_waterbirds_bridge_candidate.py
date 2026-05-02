from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for path in paths:
        if path.exists() and path.is_file():
            manifests.append({"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size})
    return manifests


def _trace_manifest(trace_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in trace_dir.glob("**/*") if path.is_file()) if trace_dir.exists() else []
    manifest_files = [path for path in files if path.name == "manifest.json"]
    training_trace_files = [path for path in files if path.name == "training_traces.jsonl"]
    packet_files = [path for path in files if path.name == "latent_clue_packets.jsonl"]
    return {
        "path": str(trace_dir),
        "exists": trace_dir.exists(),
        "file_count": len(files),
        "manifest_count": len(manifest_files),
        "training_trace_count": len(training_trace_files),
        "latent_packet_count": len(packet_files),
        "files": _file_manifest([*manifest_files, *training_trace_files, *packet_files]),
    }


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _row_type_summary(rows: list[dict[str, str]], row_type: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("row_type") == row_type]
    wgas = [float(row["test_wga"]) for row in selected]
    baseline_deltas = [float(row["delta_to_baseline"]) for row in selected if row.get("delta_to_baseline")]
    stats_deltas = [float(row["delta_to_stats"]) for row in selected if row.get("delta_to_stats")]
    return {
        "count": len(selected),
        "mean_test_wga": _mean(wgas),
        "min_test_wga": min(wgas) if wgas else 0.0,
        "max_test_wga": max(wgas) if wgas else 0.0,
        "mean_delta_to_baseline": _mean(baseline_deltas),
        "min_delta_to_baseline": min(baseline_deltas) if baseline_deltas else 0.0,
        "mean_delta_to_stats": _mean(stats_deltas),
        "min_delta_to_stats": min(stats_deltas) if stats_deltas else 0.0,
    }


def _candidate_vs_best_random(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_seed_top_k: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("seed", ""), row.get("top_k", ""))
        bucket = by_seed_top_k.setdefault(key, {"random": []})
        if row.get("row_type") == "candidate":
            bucket["candidate"] = row
        elif row.get("row_type") == "random_control":
            bucket["random"].append(row)
    paired: list[dict[str, Any]] = []
    for (seed, top_k), bucket in sorted(by_seed_top_k.items()):
        candidate = bucket.get("candidate")
        random_rows = list(bucket.get("random", []))
        if not candidate or not random_rows:
            continue
        best_random = max(random_rows, key=lambda row: float(row["test_wga"]))
        candidate_wga = float(candidate["test_wga"])
        best_random_wga = float(best_random["test_wga"])
        paired.append(
            {
                "seed": int(seed),
                "top_k": int(top_k),
                "candidate_test_wga": candidate_wga,
                "best_random_label": best_random["label"],
                "best_random_test_wga": best_random_wga,
                "delta_to_best_random": candidate_wga - best_random_wga,
            }
        )
    deltas = [float(row["delta_to_best_random"]) for row in paired]
    return {
        "paired": paired,
        "mean_delta_to_best_random": _mean(deltas),
        "min_delta_to_best_random": min(deltas) if deltas else 0.0,
        "non_negative_best_random_seeds": sum(delta >= 0.0 for delta in deltas),
    }


def _source_summaries(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "baseline": _row_type_summary(rows, "baseline"),
        "stats_control": _row_type_summary(rows, "stats_control"),
        "candidate": _row_type_summary(rows, "candidate"),
        "random_controls": _random_control_summaries(rows),
    }


def _random_control_summaries(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("row_type") == "random_control":
            grouped.setdefault(row.get("label", ""), []).append(row)
    summaries = [{"label": label, **_row_type_summary(items, "random_control")} for label, items in grouped.items()]
    summaries.sort(key=lambda item: float(item["mean_test_wga"]), reverse=True)
    return summaries


def _markdown_report(summary: dict[str, Any]) -> str:
    candidate = summary["rows"]["candidate"]
    stats = summary["rows"]["stats_control"]
    best_random = summary["rows"]["random_controls"][0] if summary["rows"]["random_controls"] else {}
    best_random_gate = summary["candidate_vs_best_random"]
    lines = [
        "# Waterbirds Bridge Candidate Report",
        "",
        f"- Candidate: `{summary['candidate_label']}`",
        f"- Paired CSV: `{summary['input_csv']}`",
        f"- Trace snapshot: `{summary['trace_manifest']['path']}`",
        "",
        "## Promotion Metrics",
        "",
        "| Comparator | Mean Test WGA | Min Test WGA | Mean Delta | Min Delta |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Official DFR | {summary['rows']['baseline']['mean_test_wga']:.10f} | {summary['rows']['baseline']['min_test_wga']:.10f} |  |  |",
        f"| Stats top-k | {stats['mean_test_wga']:.10f} | {stats['min_test_wga']:.10f} | {candidate['mean_delta_to_stats']:.10f} | {candidate['min_delta_to_stats']:.10f} |",
        f"| Best random score | {float(best_random.get('mean_test_wga', 0.0)):.10f} | {float(best_random.get('min_test_wga', 0.0)):.10f} | {best_random_gate['mean_delta_to_best_random']:.10f} | {best_random_gate['min_delta_to_best_random']:.10f} |",
        f"| Bridge candidate | {candidate['mean_test_wga']:.10f} | {candidate['min_test_wga']:.10f} | {candidate['mean_delta_to_baseline']:.10f} | {candidate['min_delta_to_baseline']:.10f} |",
        "",
        f"Candidate non-negative vs best random by seed: `{best_random_gate['non_negative_best_random_seeds']}/{len(best_random_gate['paired'])}`.",
        "",
        "## Artifact Checksums",
        "",
    ]
    for item in summary["score_files"]:
        lines.append(f"- `{item['path']}`: `{item['sha256']}`")
    return "\n".join(lines) + "\n"


def build_report(
    *,
    input_csv: Path,
    input_json: Path | None,
    score_dir: Path,
    trace_dir: Path,
    candidate_label: str,
) -> dict[str, Any]:
    rows = _read_rows(input_csv)
    score_files = sorted(score_dir.glob("scores_*.csv")) if score_dir.exists() else []
    summary = {
        "input_csv": str(input_csv),
        "input_json": str(input_json) if input_json else "",
        "candidate_label": candidate_label,
        "rows": _source_summaries(rows),
        "candidate_vs_best_random": _candidate_vs_best_random(rows),
        "score_files": _file_manifest(score_files),
        "trace_manifest": _trace_manifest(trace_dir),
    }
    if input_json and input_json.exists():
        summary["runner_summary"] = json.loads(input_json.read_text(encoding="utf-8"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="outputs/dfr_sweeps/bridge-fused-refreshed-random-controls.csv")
    parser.add_argument("--input-json", default="outputs/dfr_sweeps/bridge-fused-refreshed-random-controls.json")
    parser.add_argument("--score-dir", default="outputs/dfr_sweeps/bridge_fused_refreshed_random_controls")
    parser.add_argument("--trace-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments_20260502_refreshed")
    parser.add_argument("--candidate-label", default="bridge_fused_w0p3_top512")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/bridge-fused-candidate-report.json")
    parser.add_argument("--output-md", default="outputs/dfr_sweeps/bridge-fused-candidate-report.md")
    args = parser.parse_args()
    report = build_report(
        input_csv=Path(args.input_csv),
        input_json=Path(args.input_json) if args.input_json else None,
        score_dir=Path(args.score_dir),
        trace_dir=Path(args.trace_dir),
        candidate_label=str(args.candidate_label),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
