from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import RunRecord


DATA_ROOT = Path(os.getenv("DATA_ROOT", "/data"))
RUNS_DIR = DATA_ROOT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

RUNS: dict[str, RunRecord] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_run_dirs(run_id: str) -> tuple[Path, Path, Path, Path, Path]:
    base = RUNS_DIR / run_id
    input_dir = base / "input"
    eda_dir = base / "eda"
    output_dir = base / "output"
    plan_dir = base / "plan"
    logs_dir = base / "logs"
    for path in [input_dir, eda_dir, output_dir, plan_dir, logs_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return input_dir, eda_dir, output_dir, plan_dir, logs_dir


def append_jsonl(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(file_path: Path, payload: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_text(file_path: Path, text: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(text)


async def emit(run_id: str, stage: str, message: str, payload: dict[str, Any] | None = None) -> None:
    if run_id not in RUNS:
        return
    event = {
        "time": now_iso(),
        "stage": stage,
        "message": message,
        "payload": payload or {},
    }
    RUNS[run_id].events.append(event)
    _, _, _, _, logs_dir = get_run_dirs(run_id)
    append_jsonl(logs_dir / "events.jsonl", event)


def emit_sync(run_id: str, stage: str, message: str, payload: dict[str, Any] | None = None) -> None:
    if run_id not in RUNS:
        return
    event = {
        "time": now_iso(),
        "stage": stage,
        "message": message,
        "payload": payload or {},
    }
    RUNS[run_id].events.append(event)
    _, _, _, _, logs_dir = get_run_dirs(run_id)
    append_jsonl(logs_dir / "events.jsonl", event)
