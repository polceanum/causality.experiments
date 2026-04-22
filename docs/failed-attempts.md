# Failed Attempts Log

Record dead ends, bugs, bad assumptions, and debugging notes so future work does
not quietly repeat them.

## 2026-04-22

- `pdftotext` was not installed locally.
  - Resolution: used `pypdf`, which was already available in `orpheus`.
- Running `python scripts/run_all_fixtures.py` initially failed with
  `ModuleNotFoundError: No module named 'causality_experiments'`.
  - Cause: direct script execution put `scripts/` on `sys.path`, not the repo
    root.
  - Resolution: scripts now insert the repo root into `sys.path`.
- Attempted to create a branch named `codex/causal-experiments-harness`.
  - Failure: git could not create the nested ref path.
  - Resolution: user later clarified direct pushes to `main` are acceptable.
- First `git push` failed with `Could not resolve hostname github.com`.
  - Cause: sandboxed network restriction.
  - Resolution: reran with approved `git push` escalation.
- `gh auth status` reported an invalid GitHub token for user `polceanum`.
  - Impact: skipped GitHub CLI PR flow.
  - Resolution: used direct git SSH push to `main`.
- Initial IRM sweep used `penalty_weight=50.0`.
  - Failure mode: many runs collapsed to poor worst-group accuracy and low
    average accuracy.
  - Resolution: quick tuning on synthetic linear and Waterbirds-style fixtures
    showed lower penalties are safer; sweep default changed to `1.0`.
- The first version of `scripts/report_best_methods.py` grouped runs by parsing
  timestamped run directory names.
  - Failure mode: noisy grouping and ad hoc tuning runs polluted the report.
  - Resolution: `summary.csv` now includes the config name, and the report skips
    `_irm_w...` ad hoc tuning configs.
