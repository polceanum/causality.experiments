from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() and path.is_file() else "",
        "bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _support_by_label(report: dict[str, Any], label: str) -> dict[str, Any]:
    for row in report.get("supports", []):
        if row.get("label") == label:
            return dict(row)
    return {}


def build_package(
    *,
    candidate_report_path: Path,
    support_report_path: Path,
    critical_audit_path: Path,
    runner_csv_path: Path,
    runner_json_path: Path,
    score_dir: Path,
    trace_dir: Path,
    candidate_label: str,
) -> dict[str, Any]:
    candidate = _read_json(candidate_report_path)
    support = _read_json(support_report_path)
    audit = _read_json(critical_audit_path)
    incumbent_support = _support_by_label(support, "bridge_fused_w0p3")
    stats_support = _support_by_label(support, "stats")
    random_supports = [row for row in support.get("supports", []) if str(row.get("label", "")).startswith("random_score_")]
    random_env_counts = [float(row.get("env_ge_label_count", 0.0)) for row in random_supports]
    package = {
        "package_version": "waterbirds_bridge_incumbent/v1",
        "candidate_label": candidate_label,
        "verdict": "keep_bridge_fused_w0p3_top512_as_active_incumbent",
        "headline": {
            "mean_test_wga": candidate.get("rows", {}).get("candidate", {}).get("mean_test_wga"),
            "min_test_wga": candidate.get("rows", {}).get("candidate", {}).get("min_test_wga"),
            "mean_delta_to_official": candidate.get("rows", {}).get("candidate", {}).get("mean_delta_to_baseline"),
            "mean_delta_to_stats": candidate.get("rows", {}).get("candidate", {}).get("mean_delta_to_stats"),
            "mean_delta_to_best_random": candidate.get("candidate_vs_best_random", {}).get("mean_delta_to_best_random"),
            "min_delta_to_best_random": candidate.get("candidate_vs_best_random", {}).get("min_delta_to_best_random"),
            "non_negative_best_random_seeds": candidate.get("candidate_vs_best_random", {}).get("non_negative_best_random_seeds"),
        },
        "seed_rows": audit.get("incumbent_seed_rows", candidate.get("candidate_vs_best_random", {}).get("paired", [])),
        "critical_read": audit.get("critical_read", []),
        "support_summary": {
            "top_k": support.get("top_k", 512),
            "reference_label": support.get("reference_label", "stats"),
            "bridge_overlap_with_stats": incumbent_support.get("overlap_with_reference"),
            "bridge_jaccard_with_stats": incumbent_support.get("jaccard_with_reference"),
            "bridge_env_ge_label_count": incumbent_support.get("env_ge_label_count"),
            "stats_env_ge_label_count": stats_support.get("env_ge_label_count"),
            "random_env_ge_label_count_min": min(random_env_counts) if random_env_counts else None,
            "random_env_ge_label_count_max": max(random_env_counts) if random_env_counts else None,
        },
        "artifacts": {
            "candidate_report_json": _file_entry(candidate_report_path),
            "support_report_json": _file_entry(support_report_path),
            "critical_audit_json": _file_entry(critical_audit_path),
            "runner_csv": _file_entry(runner_csv_path),
            "runner_json": _file_entry(runner_json_path),
            "score_files": [_file_entry(path) for path in sorted(score_dir.glob("scores_*.csv"))],
            "trace_manifests": [_file_entry(path) for path in sorted(trace_dir.glob("**/manifest.json"))],
        },
        "reproduction_commands": [
            "conda run -n orpheus python scripts/report_waterbirds_bridge_candidate.py --input-csv outputs/dfr_sweeps/bridge-fused-refreshed-random-controls.csv --input-json outputs/dfr_sweeps/bridge-fused-refreshed-random-controls.json --score-dir outputs/dfr_sweeps/bridge_fused_refreshed_random_controls --trace-dir outputs/dfr_sweeps/llm_clue_fixture_experiments_20260502_refreshed --candidate-label bridge_fused_w0p3_top512 --output-json outputs/dfr_sweeps/bridge-fused-candidate-report.json --output-md outputs/dfr_sweeps/bridge-fused-candidate-report.md",
            "conda run -n orpheus python scripts/report_waterbirds_bridge_support.py --output-json outputs/dfr_sweeps/bridge-fused-support-report.json",
            "conda run -n orpheus python scripts/package_waterbirds_bridge_incumbent.py",
        ],
        "claim_language": {
            "safe": "bridge_fused_w0p3_top512 is the active local Waterbirds incumbent under the five-seed official/stats/best-random gate.",
            "caveat": "The margin is modest and seed 104 ties the best random control, so this should be reported as a local reproducible improvement rather than a broad SOTA claim.",
        },
    }
    return package


def _markdown(package: dict[str, Any]) -> str:
    headline = package["headline"]
    support = package["support_summary"]
    lines = [
        "# Waterbirds Bridge Incumbent Package",
        "",
        f"Verdict: `{package['verdict']}`.",
        "",
        "## Headline",
        "",
        f"- Candidate: `{package['candidate_label']}`",
        f"- Mean WGA: `{headline['mean_test_wga']}`",
        f"- Min WGA: `{headline['min_test_wga']}`",
        f"- Mean delta to official DFR: `{headline['mean_delta_to_official']}`",
        f"- Mean delta to stats: `{headline['mean_delta_to_stats']}`",
        f"- Mean delta to best random: `{headline['mean_delta_to_best_random']}`",
        f"- Best-random non-negative seeds: `{headline['non_negative_best_random_seeds']}/5`",
        "",
        "## Critical Read",
        "",
    ]
    lines.extend(f"- {item}" for item in package.get("critical_read", []))
    lines.extend(
        [
            "",
            "## Support Summary",
            "",
            f"- Bridge overlap with stats: `{support['bridge_overlap_with_stats']}`",
            f"- Bridge Jaccard with stats: `{support['bridge_jaccard_with_stats']}`",
            f"- Bridge env>=label features: `{support['bridge_env_ge_label_count']}`",
            f"- Random env>=label feature range: `{support['random_env_ge_label_count_min']}`-`{support['random_env_ge_label_count_max']}`",
            "",
            "## Claim Boundary",
            "",
            package["claim_language"]["safe"],
            "",
            package["claim_language"]["caveat"],
            "",
            "## Reproduction Commands",
            "",
        ]
    )
    lines.extend(f"```bash\n{command}\n```" for command in package.get("reproduction_commands", []))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", default="outputs/dfr_sweeps/bridge-fused-candidate-report.json")
    parser.add_argument("--support-report", default="outputs/dfr_sweeps/bridge-fused-support-report.json")
    parser.add_argument("--critical-audit", default="outputs/dfr_sweeps/bridge-fused-critical-audit.json")
    parser.add_argument("--runner-csv", default="outputs/dfr_sweeps/bridge-fused-refreshed-random-controls.csv")
    parser.add_argument("--runner-json", default="outputs/dfr_sweeps/bridge-fused-refreshed-random-controls.json")
    parser.add_argument("--score-dir", default="outputs/dfr_sweeps/bridge_fused_refreshed_random_controls")
    parser.add_argument("--trace-dir", default="outputs/dfr_sweeps/llm_clue_fixture_experiments_20260502_refreshed")
    parser.add_argument("--candidate-label", default="bridge_fused_w0p3_top512")
    parser.add_argument("--output-json", default="outputs/dfr_sweeps/waterbirds-bridge-incumbent-package.json")
    parser.add_argument("--output-md", default="outputs/dfr_sweeps/waterbirds-bridge-incumbent-package.md")
    args = parser.parse_args()
    package = build_package(
        candidate_report_path=Path(args.candidate_report),
        support_report_path=Path(args.support_report),
        critical_audit_path=Path(args.critical_audit),
        runner_csv_path=Path(args.runner_csv),
        runner_json_path=Path(args.runner_json),
        score_dir=Path(args.score_dir),
        trace_dir=Path(args.trace_dir),
        candidate_label=str(args.candidate_label),
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(package, indent=2, sort_keys=True), encoding="utf-8")
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown(package), encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
