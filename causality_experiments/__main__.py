from __future__ import annotations

from pathlib import Path

import typer

from .data import write_fixture_npz
from .run import run_experiment, summarize_runs

app = typer.Typer(help="Run causal extraction experiments.")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Experiment YAML config."),
    output_dir: Path | None = typer.Option(None, help="Override output root."),
) -> None:
    run_dir = run_experiment(config, output_dir)
    typer.echo(f"Wrote run outputs to {run_dir}")


@app.command("make-fixtures")
def make_fixtures(
    output_dir: Path = typer.Option(Path("data/fixtures"), help="Fixture output directory."),
    seed: int = typer.Option(0, help="Fixture seed."),
) -> None:
    written = write_fixture_npz(output_dir, seed=seed)
    for path in written:
        typer.echo(path)


@app.command()
def summarize(
    runs: Path = typer.Option(Path("outputs/runs"), help="Run output directory."),
) -> None:
    out = summarize_runs(runs)
    typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    app()
