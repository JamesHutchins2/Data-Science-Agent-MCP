from __future__ import annotations

import asyncio
import base64
import uuid
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from .agentic_pipeline import run_pipeline
from .models import RunRecord
from .runtime import RUNS, get_run_dirs


class CsvFilePayload(BaseModel):
    filename: str = Field(description="CSV filename")
    content_base64: str = Field(description="Base64-encoded CSV content")


class NotebookGenerationResult(BaseModel):
    run_id: str
    status: str
    notebook_filename: str | None = None
    notebook_base64: str | None = None
    output_files: list[str] = Field(default_factory=list)
    report_files: list[str] = Field(default_factory=list)
    error: str | None = None


mcp = FastMCP(
    "DataCleaningNotebookService",
    instructions=(
        "Generate data-cleaning notebooks from CSV inputs. "
        "Upload CSV content and instructions via tools, then retrieve notebook outputs."
    ),
    stateless_http=True,
    json_response=True,
    streamable_http_path="/",
)


def _decode_and_write_csvs(run_id: str, csv_files: list[CsvFilePayload]) -> list[str]:
    input_dir, _, _, _, _ = get_run_dirs(run_id)
    written_files: list[str] = []
    for csv_file in csv_files:
        if not csv_file.filename.lower().endswith(".csv"):
            raise ValueError(f"Only .csv files are supported: {csv_file.filename}")
        raw = base64.b64decode(csv_file.content_base64)
        out_path = input_dir / csv_file.filename
        out_path.write_bytes(raw)
        written_files.append(csv_file.filename)
    return written_files


def _collect_run_outputs(run_id: str) -> tuple[list[str], list[str]]:
    _, eda_dir, output_dir, _, _ = get_run_dirs(run_id)
    output_files = sorted([p.name for p in output_dir.glob("*") if p.is_file()])
    report_files = sorted([p.name for p in eda_dir.glob("*.html")])
    return output_files, report_files


def _read_notebook_payload(run_id: str, prefer_executed: bool = True) -> tuple[str | None, str | None]:
    _, _, output_dir, _, _ = get_run_dirs(run_id)
    executed = output_dir / "agentic_notebook_executed.ipynb"
    draft = output_dir / "agentic_notebook.ipynb"

    chosen: Path | None = None
    if prefer_executed and executed.exists():
        chosen = executed
    elif draft.exists():
        chosen = draft
    elif executed.exists():
        chosen = executed

    if not chosen:
        return None, None

    encoded = base64.b64encode(chosen.read_bytes()).decode("utf-8")
    return chosen.name, encoded


@mcp.tool()
async def generate_notebook_from_csvs(
    instructions: str,
    csv_files: list[CsvFilePayload],
    ml_prompt: str = "",
    system_prompt: str = "You are an expert data cleaning agent.",
    timeout_seconds: int = 1800,
    prefer_executed_notebook: bool = True,
) -> NotebookGenerationResult:
    """Run the full notebook generation pipeline from CSV files and return notebook output."""
    if not instructions.strip():
        raise ValueError("instructions is required")
    if not csv_files:
        raise ValueError("csv_files must contain at least one CSV payload")

    run_id = str(uuid.uuid4())
    RUNS[run_id] = RunRecord(run_id=run_id)
    run_record = RUNS[run_id]

    run_record.files = _decode_and_write_csvs(run_id, csv_files)

    try:
        await asyncio.wait_for(
            run_pipeline(run_id, instructions, ml_prompt, system_prompt),
            timeout=max(1, timeout_seconds),
        )
    except asyncio.TimeoutError:
        run_record.status = "failed"
        run_record.error = f"Pipeline timed out after {timeout_seconds} seconds"

    output_files, report_files = _collect_run_outputs(run_id)
    notebook_name, notebook_b64 = _read_notebook_payload(run_id, prefer_executed=prefer_executed_notebook)

    return NotebookGenerationResult(
        run_id=run_id,
        status=run_record.status,
        notebook_filename=notebook_name,
        notebook_base64=notebook_b64,
        output_files=output_files,
        report_files=report_files,
        error=run_record.error,
    )


@mcp.tool()
def get_run_status(run_id: str) -> dict[str, Any]:
    """Get status and basic metadata for a previously created run."""
    record = RUNS.get(run_id)
    if not record:
        raise ValueError("Run not found")
    output_files, report_files = _collect_run_outputs(run_id)
    return {
        "run_id": record.run_id,
        "status": record.status,
        "started_at": record.started_at,
        "completed_at": record.completed_at,
        "error": record.error,
        "input_files": record.files,
        "output_files": output_files,
        "report_files": report_files,
    }
