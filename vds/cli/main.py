"""CLI entry point (System Design §2.13).

Phase-1 first-class client: drives the same service layer as the API (in-process,
not over HTTP), so the pipeline is provable end to end before any UI exists.
Bootstrap scope: the command surface plus the framework-level commands that work
today — `version`, `config-check`, `serve`, `worker`. Pipeline commands are
stubbed and land in Phase 1.
"""

from __future__ import annotations

import typer

from vds import __version__
from vds.container import build_container

app = typer.Typer(help="AutoDataForge", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"vds {__version__}")


@app.command("config-check")
def config_check() -> None:
    """Load and validate configuration, print the resolved model plugins."""
    container = build_container()
    s = container.settings
    typer.echo(f"environment : {s.environment}")
    typer.echo(f"gpu         : {s.gpu.device} (budget {s.gpu.vram_budget_mb} MB)")
    typer.echo(f"database    : {s.storage.database_url}")
    typer.echo(f"cas_root    : {s.storage.cas_root}")
    typer.echo(f"llm         : {s.llm.provider} (model={s.llm.model})")
    typer.echo("models      :")
    for capability, path in container.models.describe().items():
        typer.echo(f"  {capability:<13}-> {path}")
    typer.echo("OK: configuration valid")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI app."""
    import uvicorn

    uvicorn.run("vds.api.app:app", host=host, port=port)


@app.command()
def worker() -> None:
    """Run the job worker loop."""
    from vds.jobs.worker import Worker

    build_container()  # configures logging + validates config
    Worker().run()


@app.command()
def gui() -> None:
    """Launch the desktop application (PySide6)."""
    from vds.gui.app import main as gui_main

    raise typer.Exit(gui_main())


# --- Phase 1 pipeline ---
@app.command()
def run(
    source: str,
    fmt: str = typer.Option("coco", "--format", help="coco | yolo"),
    dest: str = "export",
    name: str = "phase1",
) -> None:
    """Run the full pipeline on an image folder: import -> plan -> detect ->
    segment -> verify -> export. Prints an execution + benchmark report."""
    container = build_container()
    report = container.pipeline.run(source, name=name, export_format=fmt, dest=dest)
    b = report.benchmark
    typer.echo("")
    typer.echo("=== Execution Report ===")
    typer.echo(f"source            : {report.source}")
    typer.echo(f"imported          : {report.imported}")
    typer.echo(f"duplicates_skipped: {report.duplicates_skipped}")
    typer.echo(f"quarantined       : {report.quarantined}")
    typer.echo(f"detections        : {report.detections}")
    typer.echo(f"  approved        : {report.verified_approved}")
    typer.echo(f"  needs_review    : {report.needs_review}")
    typer.echo(f"  rejected        : {report.rejected}")
    typer.echo(f"export            : {report.export.format} -> {report.export.output_path}"
               f" (validated={report.export.validated})")
    typer.echo("=== Benchmark ===")
    typer.echo(f"images/sec        : {b.images_per_second}")
    typer.echo(f"total_seconds     : {b.total_seconds}")
    typer.echo(f"avg_inference_ms  : {b.avg_inference_ms}")
    typer.echo(f"stage_seconds     : {b.stage_seconds}")
    typer.echo(f"peak_ram_mb       : {b.peak_ram_mb}")
    typer.echo(f"gpu_util_percent  : {b.gpu_util_percent}")


if __name__ == "__main__":
    app()
