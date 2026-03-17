from __future__ import annotations

import asyncio
import json
import os
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from .agentic_pipeline import run_pipeline
from .models import RunRecord
from .runtime import RUNS, emit, get_run_dirs
from .ui import UI


router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
	return {"status": "ok"}


@router.post("/pipeline/start")
async def start_pipeline(
	files: list[UploadFile] = File(...),
	prompt: str = Form(...),
	ml_prompt: str = Form(""),
	system_prompt: str = Form("You are an expert data cleaning agent."),
):
	if not os.getenv("OPENAI_API_KEY", ""):
		raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured.")
	if not files:
		raise HTTPException(status_code=400, detail="At least one CSV file is required")

	run_id = str(uuid.uuid4())
	RUNS[run_id] = RunRecord(run_id=run_id)
	input_dir, _, _, _, _ = get_run_dirs(run_id)

	for file in files:
		if not file.filename.lower().endswith(".csv"):
			raise HTTPException(status_code=400, detail=f"Only CSV files are supported: {file.filename}")
		out_path = input_dir / file.filename
		out_path.write_bytes(await file.read())
		RUNS[run_id].files.append(file.filename)

	asyncio.create_task(run_pipeline(run_id, prompt, ml_prompt, system_prompt))
	await emit(run_id, "pipeline", "Run created")
	return {
		"run_id": run_id,
		"status_url": f"/pipeline/status/{run_id}",
		"events_url": f"/pipeline/events/{run_id}",
		"reports_url": f"/pipeline/reports/{run_id}",
		"outputs_url": f"/pipeline/outputs/{run_id}",
	}


@router.get("/pipeline/status/{run_id}")
async def pipeline_status(run_id: str):
	record = RUNS.get(run_id)
	if not record:
		raise HTTPException(status_code=404, detail="Run not found")
	return {
		"run_id": record.run_id,
		"status": record.status,
		"started_at": record.started_at,
		"completed_at": record.completed_at,
		"error": record.error,
		"input_files": record.files,
	}


@router.get("/pipeline/events/{run_id}")
async def pipeline_events(run_id: str):
	if run_id not in RUNS:
		raise HTTPException(status_code=404, detail="Run not found")

	async def event_generator():
		index = 0
		while True:
			record = RUNS.get(run_id)
			if not record:
				break
			while index < len(record.events):
				payload = record.events[index]
				index += 1
				yield f"data: {json.dumps(payload)}\n\n"
			if record.status in {"completed", "completed_with_validation_errors", "failed"} and index >= len(record.events):
				break
			await asyncio.sleep(0.5)

	return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/pipeline/reports/{run_id}")
async def list_reports(run_id: str):
	if run_id not in RUNS:
		raise HTTPException(status_code=404, detail="Run not found")
	_, eda_dir, _, _, _ = get_run_dirs(run_id)
	reports = sorted([p.name for p in eda_dir.glob("*.html")])
	return {"run_id": run_id, "reports": reports}


@router.get("/pipeline/outputs/{run_id}")
async def list_outputs(run_id: str):
	if run_id not in RUNS:
		raise HTTPException(status_code=404, detail="Run not found")
	_, _, output_dir, plan_dir, logs_dir = get_run_dirs(run_id)
	outputs = sorted([p.name for p in output_dir.glob("*") if p.is_file()])
	plans = sorted([p.name for p in plan_dir.glob("*.json") if p.is_file()])
	logs = sorted([str(p.relative_to(logs_dir)) for p in logs_dir.rglob("*") if p.is_file()])
	return {"run_id": run_id, "outputs": outputs, "plans": plans, "logs": logs}


@router.get("/reports/{run_id}/{report_name}")
async def get_report(run_id: str, report_name: str):
	_, eda_dir, _, _, _ = get_run_dirs(run_id)
	report_path = eda_dir / report_name
	if not report_path.exists():
		raise HTTPException(status_code=404, detail="Report not found")
	return FileResponse(report_path)


@router.get("/files/{run_id}/input/{file_name}")
async def get_input_file(run_id: str, file_name: str):
	input_dir, _, _, _, _ = get_run_dirs(run_id)
	file_path = input_dir / file_name
	if not file_path.exists():
		raise HTTPException(status_code=404, detail="Input file not found")
	return FileResponse(file_path, filename=file_name)


@router.get("/files/{run_id}/output/{file_name}")
async def get_output_file(run_id: str, file_name: str):
	_, _, output_dir, plan_dir, _ = get_run_dirs(run_id)
	output_path = output_dir / file_name
	if output_path.exists():
		return FileResponse(output_path, filename=file_name)
	plan_path = plan_dir / file_name
	if plan_path.exists():
		return FileResponse(plan_path, filename=file_name)
	raise HTTPException(status_code=404, detail="Output file not found")


@router.get("/files/{run_id}/logs/{log_path:path}")
async def get_log_file(run_id: str, log_path: str):
	_, _, _, _, logs_dir = get_run_dirs(run_id)
	requested_path = (logs_dir / log_path).resolve()
	if not str(requested_path).startswith(str(logs_dir.resolve())):
		raise HTTPException(status_code=400, detail="Invalid log path")
	if not requested_path.exists() or not requested_path.is_file():
		raise HTTPException(status_code=404, detail="Log file not found")
	return FileResponse(requested_path, filename=requested_path.name)


@router.get("/")
async def ui_home():
	return HTMLResponse(UI)


@router.get("/root")
async def root():
	return {"message": "ADCES MVP API is running", "health": "/health"}

