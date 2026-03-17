from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class NotebookCellSpec(BaseModel):
    cell_type: Literal["markdown", "code"]
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    title: str
    objective: str
    success_criteria: str


class NotebookPlan(BaseModel):
    summary: str
    steps: list[PlanStep] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class PlanValidation(BaseModel):
    approved: bool
    feedback: str = ""


class NotebookDraft(BaseModel):
    notes: str = ""
    cells: list[NotebookCellSpec] = Field(default_factory=list)


class RunReview(BaseModel):
    complete: bool
    feedback: str = ""


class PipelineState(TypedDict):
    run_id: str
    user_request: str
    system_prompt: str
    data_dir: str
    output_dir: str
    plan_dir: str
    logs_dir: str
    eda_context: str
    general_skills: str
    plan: dict[str, Any]
    plan_feedback: str
    notebook_cells: list[dict[str, Any]]
    coder_feedback: str
    notebook_path: str
    executed_notebook_path: str
    execution_summary: dict[str, Any]
    review_complete: bool
    review_feedback: str
    plan_attempts: int
    coding_attempts: int
    max_plan_attempts: int
    max_coding_attempts: int


@dataclass
class RunRecord:
    run_id: str
    status: str = "created"
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    reports: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
