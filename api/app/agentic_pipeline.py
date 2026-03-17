from __future__ import annotations

import json
import os
import traceback
from urllib.parse import quote
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .models import NotebookDraft, NotebookPlan, PipelineState, PlanValidation, RunReview
from .prompts import GENERAL_DATA_SCIENCE_SKILLS_PROMPT, load_local_skill_prompt
from .runtime import RUNS, append_jsonl, append_text, emit, emit_sync, get_run_dirs, now_iso, write_json
from .tools import NotebookToolbox


def llm_client() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    return ChatOpenAI(model=model, temperature=0)


def build_agentic_graph(llm: ChatOpenAI, notebook_tools: NotebookToolbox):
    planner_llm = llm.with_structured_output(NotebookPlan, method="function_calling")
    validator_llm = llm.with_structured_output(PlanValidation, method="function_calling")
    coder_llm = llm.with_structured_output(NotebookDraft, method="function_calling")
    reviewer_llm = llm.with_structured_output(RunReview, method="function_calling")

    def _log_agent_io(state: PipelineState, agent_name: str, attempt: int, request_payload: dict[str, Any], response_payload: dict[str, Any]) -> None:
        logs_dir = Path(state["logs_dir"])
        payload = {
            "time": now_iso(),
            "agent": agent_name,
            "attempt": attempt,
            "request": request_payload,
            "response": response_payload,
        }
        append_jsonl(logs_dir / "agents" / f"{agent_name}.jsonl", payload)
        write_json(logs_dir / "agents" / f"{agent_name}_attempt_{attempt}.json", payload)

    def planner_node(state: PipelineState) -> dict[str, Any]:
        attempt = state.get("plan_attempts", 0) + 1
        emit_sync(state["run_id"], "planner", f"Planner attempt {attempt} started")
        feedback = state.get("plan_feedback", "")
        request_payload = {
            "system": (
                "You are a planner agent for data science notebook construction. "
                "Create a concrete, testable plan aligned to the request and provided EDA."
            ),
            "user_request": state["user_request"],
            "general_skills": state["general_skills"],
            "eda_context": state["eda_context"],
            "plan_feedback": feedback,
        }
        result = planner_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a planner agent for data science notebook construction. "
                        "Create a concrete, testable plan aligned to the request and provided EDA."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User request:\n{state['user_request']}\n\n"
                        f"General skills:\n{state['general_skills']}\n\n"
                        f"Auto-EDA context (compact):\n{state['eda_context']}\n\n"
                        f"Plan feedback from orchestrator:\n{feedback}"
                    )
                ),
            ]
        )
        _log_agent_io(
            state,
            "planner",
            attempt,
            request_payload=request_payload,
            response_payload=result.model_dump(),
        )
        emit_sync(state["run_id"], "planner", f"Planner attempt {attempt} completed")
        return {
            "plan": result.model_dump(),
            "plan_attempts": attempt,
        }

    def plan_validator_node(state: PipelineState) -> dict[str, Any]:
        emit_sync(state["run_id"], "validator", "Validating planner output")
        request_payload = {
            "system": (
                "You are an orchestrator-validator for a planner agent. "
                "Approve only if the plan fully addresses request, includes EDA, cleaning, modeling, and validation."
            ),
            "user_request": state["user_request"],
            "plan": state.get("plan", {}),
        }
        validation = validator_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an orchestrator-validator for a planner agent. "
                        "Approve only if the plan fully addresses request, includes EDA, cleaning, modeling, and validation."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User request:\n{state['user_request']}\n\n"
                        f"Plan JSON:\n{json.dumps(state.get('plan', {}), indent=2)}"
                    )
                ),
            ]
        )
        _log_agent_io(
            state,
            "validator",
            state.get("plan_attempts", 0),
            request_payload=request_payload,
            response_payload=validation.model_dump(),
        )
        emit_sync(state["run_id"], "validator", "Validation completed", {"approved": bool(validation.approved)})
        return {
            "plan_feedback": validation.feedback,
            "review_complete": bool(validation.approved),
        }

    def route_plan(state: PipelineState) -> str:
        approved = bool(state.get("review_complete", False))
        if approved:
            return "approved"
        if state.get("plan_attempts", 0) >= state.get("max_plan_attempts", 3):
            return "approved"
        return "revise"

    def coder_node(state: PipelineState) -> dict[str, Any]:
        attempt = state.get("coding_attempts", 0) + 1
        emit_sync(state["run_id"], "coder", f"Coder attempt {attempt} started")
        prior_cells = state.get("notebook_cells", [])
        prior_execution = state.get("execution_summary", {})
        request_payload = {
            "system": (
                "You are a coder agent that writes Jupyter notebook cells. "
                "Return notebook cells only; keep workflow reproducible and executable."
            ),
            "user_request": state["user_request"],
            "general_skills": state["general_skills"],
            "plan": state.get("plan", {}),
            "eda_context": state["eda_context"],
            "coder_feedback": state.get("coder_feedback", ""),
            "prior_execution_summary": prior_execution,
            "prior_notebook_cells": prior_cells,
            "requirements": [
                "Use markdown and code cells.",
                "Include EDA, cleaning, train/validation split, model training, evaluation.",
                "Save key artifacts into output directory where relevant.",
                "Assume CSVs are in ./input when notebook executes from run root.",
                "For attempts after 1, revise the existing notebook incrementally; keep working cells and only change what validator feedback requests.",
            ],
        }
        draft = coder_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a coder agent that writes Jupyter notebook cells. "
                        "Return notebook cells only; keep workflow reproducible and executable."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User request:\n{state['user_request']}\n\n"
                        f"General skills:\n{state['general_skills']}\n\n"
                        f"Plan:\n{json.dumps(state.get('plan', {}), indent=2)}\n\n"
                        f"Auto-EDA context:\n{state['eda_context']}\n\n"
                        f"Current coder feedback:\n{state.get('coder_feedback', '')}\n\n"
                        f"Previous execution summary:\n{json.dumps(prior_execution, indent=2)}\n\n"
                        f"Previous notebook cells JSON:\n{json.dumps(prior_cells, indent=2)}\n\n"
                        "Notebook requirements:\n"
                        "- Use markdown and code cells.\n"
                        "- Include EDA, cleaning, train/validation split, model training, evaluation.\n"
                        "- Save key artifacts into output directory where relevant.\n"
                        "- Assume CSVs are in ./input when notebook executes from run root.\n"
                        "- If this is a revision attempt, do not restart from zero: keep valid prior cells and only modify/add cells needed to satisfy validator feedback and execution issues."
                    )
                ),
            ]
        )
        cell_models = notebook_tools.normalize_cells([cell.model_dump() for cell in draft.cells])
        notebook_cells = [c.model_dump() for c in cell_models]
        _log_agent_io(
            state,
            "coder",
            attempt,
            request_payload=request_payload,
            response_payload={
                "notes": draft.notes,
                "cells": notebook_cells,
            },
        )
        emit_sync(
            state["run_id"],
            "coder",
            f"Coder attempt {attempt} completed",
            {"generated_cells": len(notebook_cells)},
        )
        return {
            "notebook_cells": notebook_cells,
            "coding_attempts": attempt,
        }

    async def execute_node(state: PipelineState) -> dict[str, Any]:
        output_dir = Path(state["output_dir"])
        run_root_dir = Path(state["data_dir"]).parent
        logs_dir = Path(state["logs_dir"])
        notebook_path = output_dir / "agentic_notebook.ipynb"
        executed_path = output_dir / "agentic_notebook_executed.ipynb"
        execution_backend = os.getenv("NOTEBOOK_EXECUTION_BACKEND", "papermill").strip().lower()

        emit_sync(state["run_id"], "execute", "Notebook write and execution started")

        cell_specs = notebook_tools.normalize_cells(state.get("notebook_cells", []))
        notebook_tools.write_notebook(notebook_path, cell_specs)
        write_json(
            logs_dir / "actions" / f"execute_attempt_{state.get('coding_attempts', 0)}.json",
            {
                "time": now_iso(),
                "action": "execute_notebook",
                "backend": execution_backend,
                "notebook_path": str(notebook_path),
                "executed_path": str(executed_path),
                "cwd": str(run_root_dir),
                "kernel_name": "python3",
                "timeout": 900,
                "allow_errors": False,
            },
        )
        if execution_backend == "nbclient":
            execution = notebook_tools.execute_notebook_nbclient(
                notebook_path=notebook_path,
                executed_path=executed_path,
                cwd=run_root_dir,
                kernel_name="python3",
                timeout=900,
                allow_errors=False,
            )
        else:
            execution = notebook_tools.execute_notebook_papermill(
                notebook_path=notebook_path,
                executed_path=executed_path,
                cwd=run_root_dir,
                parameters=None,
                kernel_name="python3",
            )

        (output_dir / "agentic_execution_summary.json").write_text(
            json.dumps(execution, indent=2),
            encoding="utf-8",
        )
        append_jsonl(
            logs_dir / "actions" / "execution_summary.jsonl",
            {
                "time": now_iso(),
                "coding_attempt": state.get("coding_attempts", 0),
                "execution_summary": execution,
            },
        )
        emit_sync(
            state["run_id"],
            "execute",
            "Notebook execution completed",
            {
                "error_cells": execution.get("error_cells", 0),
                "first_error": execution.get("first_error", ""),
                "backend": execution_backend,
            },
        )
        return {
            "notebook_path": str(notebook_path),
            "executed_notebook_path": str(executed_path),
            "execution_summary": execution,
        }

    def reviewer_node(state: PipelineState) -> dict[str, Any]:
        emit_sync(state["run_id"], "coder_validator", "Coder validator started")
        request_payload = {
            "system": (
                "You are a validator for coder revisions. "
                "Decide if the notebook is complete and provide precise, actionable change requests when not complete."
            ),
            "user_request": state["user_request"],
            "plan": state.get("plan", {}),
            "execution_summary": state.get("execution_summary", {}),
            "current_notebook_cells": state.get("notebook_cells", []),
        }
        execution_summary = state.get("execution_summary", {})
        if not execution_summary.get("success", True):
            feedback = (
                "Execution failed. Fix notebook execution first before content refinements. "
                f"Root error: {execution_summary.get('error', '')}. "
                "Preserve working sections and only make targeted fixes needed for successful execution."
            )
            _log_agent_io(
                state,
                "coder_validator",
                state.get("coding_attempts", 0),
                request_payload=request_payload,
                response_payload={"complete": False, "feedback": feedback},
            )
            emit_sync(state["run_id"], "coder_validator", "Execution failure sent as validator feedback")
            return {
                "review_complete": False,
                "review_feedback": feedback,
                "coder_feedback": feedback,
            }

        review = reviewer_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a validator for coder revisions. "
                        "Approve only if user request and plan are fully implemented with an executable notebook. "
                        "If not approved, give concrete change requests that can be applied incrementally."
                    )
                ),
                HumanMessage(
                    content=(
                        f"User request:\n{state['user_request']}\n\n"
                        f"Plan:\n{json.dumps(state.get('plan', {}), indent=2)}\n\n"
                        f"Execution summary:\n{json.dumps(state.get('execution_summary', {}), indent=2)}\n\n"
                        f"Current notebook cells JSON:\n{json.dumps(state.get('notebook_cells', []), indent=2)}"
                    )
                ),
            ]
        )
        _log_agent_io(
            state,
            "coder_validator",
            state.get("coding_attempts", 0),
            request_payload=request_payload,
            response_payload=review.model_dump(),
        )
        if state.get("execution_summary", {}).get("error_cells", 0) > 0:
            emit_sync(state["run_id"], "coder_validator", "Validator requested revision due to execution errors")
            return {
                "review_complete": False,
                "review_feedback": "Execution has failing cells. Fix all failures before completion.",
                "coder_feedback": f"Notebook execution failed: {state.get('execution_summary', {}).get('first_error', '')}",
            }
        emit_sync(state["run_id"], "coder_validator", "Coder validator completed", {"complete": bool(review.complete)})
        return {
            "review_complete": bool(review.complete),
            "review_feedback": review.feedback,
            "coder_feedback": review.feedback,
        }

    def route_review(state: PipelineState) -> str:
        if state.get("review_complete"):
            return "done"
        if state.get("coding_attempts", 0) >= state.get("max_coding_attempts", 4):
            return "done"
        return "revise"

    graph = StateGraph(PipelineState)
    graph.add_node("planner", planner_node)
    graph.add_node("plan_validator", plan_validator_node)
    graph.add_node("coder", coder_node)
    graph.add_node("execute", execute_node)
    graph.add_node("coder_validator", reviewer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "plan_validator")
    graph.add_conditional_edges("plan_validator", route_plan, {"approved": "coder", "revise": "planner"})
    graph.add_edge("coder", "execute")
    graph.add_edge("execute", "coder_validator")
    graph.add_conditional_edges("coder_validator", route_review, {"done": END, "revise": "coder"})
    return graph.compile()


async def run_pipeline(run_id: str, prompt: str, ml_prompt: str, system_prompt: str) -> None:
    record = RUNS[run_id]
    record.status = "running"
    record.started_at = now_iso()

    input_dir, eda_dir, output_dir, plan_dir, logs_dir = get_run_dirs(run_id)
    notebook_tools = NotebookToolbox(output_dir=output_dir)

    try:
        write_json(
            logs_dir / "run_context.json",
            {
                "run_id": run_id,
                "started_at": record.started_at,
                "prompt": prompt,
                "ml_prompt": ml_prompt,
                "system_prompt": system_prompt,
            },
        )
        await emit(run_id, "pipeline", "Building compact auto-EDA summary")
        compact_eda = notebook_tools.compact_auto_eda_summary(input_dir)
        eda_context = notebook_tools.format_eda_context(compact_eda)
        (eda_dir / "summary_compact.json").write_text(json.dumps(compact_eda, indent=2), encoding="utf-8")

        await emit(run_id, "pipeline", "Generating EDA reports")
        report_names = notebook_tools.generate_eda_html_reports(input_dir=input_dir, eda_dir=eda_dir)
        record.reports = report_names
        for report_name in report_names:
            await emit(
                run_id,
                "pipeline",
                f"EDA report generated: {report_name}",
                {"report_url": f"/reports/{run_id}/{quote(report_name)}", "report_name": report_name},
            )

        skills_prompt = GENERAL_DATA_SCIENCE_SKILLS_PROMPT
        local_skill = load_local_skill_prompt()
        if local_skill:
            skills_prompt = f"{skills_prompt}\n\nLocal skill context:\n{local_skill}"

        llm = llm_client()
        graph = build_agentic_graph(llm=llm, notebook_tools=notebook_tools)

        state: PipelineState = {
            "run_id": run_id,
            "user_request": prompt,
            "system_prompt": system_prompt,
            "data_dir": str(input_dir),
            "output_dir": str(output_dir),
            "plan_dir": str(plan_dir),
            "logs_dir": str(logs_dir),
            "eda_context": eda_context,
            "general_skills": skills_prompt,
            "plan": {},
            "plan_feedback": "",
            "notebook_cells": [],
            "coder_feedback": "",
            "notebook_path": "",
            "executed_notebook_path": "",
            "execution_summary": {},
            "review_complete": False,
            "review_feedback": "",
            "plan_attempts": 0,
            "coding_attempts": 0,
            "max_plan_attempts": int(os.getenv("MAX_PLAN_REVISIONS", "3")),
            "max_coding_attempts": int(os.getenv("MAX_CODER_REVISIONS", "4")),
        }

        await emit(run_id, "pipeline", "Running planner-validator-orchestrator workflow")
        final_state = await graph.ainvoke(state)

        write_json(logs_dir / "final_state.json", final_state)

        (plan_dir / "agentic_plan.json").write_text(
            json.dumps(final_state.get("plan", {}), indent=2),
            encoding="utf-8",
        )
        (plan_dir / "review_feedback.json").write_text(
            json.dumps(
                {
                    "plan_feedback": final_state.get("plan_feedback", ""),
                    "review_feedback": final_state.get("review_feedback", ""),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        outputs = sorted([p.name for p in output_dir.glob("*") if p.is_file()])
        record.outputs = outputs
        if final_state.get("review_complete") and final_state.get("execution_summary", {}).get("error_cells", 0) == 0:
            record.status = "completed"
            await emit(run_id, "pipeline", "Agentic pipeline completed")
        else:
            record.status = "completed_with_validation_errors"
            await emit(
                run_id,
                "pipeline",
                "Agentic pipeline ended with unresolved review feedback",
                {"review_feedback": final_state.get("review_feedback", "")},
            )
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)
        trace_text = traceback.format_exc()
        append_text(logs_dir / "errors" / "traceback.txt", trace_text)
        write_json(
            logs_dir / "errors" / "error.json",
            {
                "time": now_iso(),
                "error": str(exc),
                "traceback": trace_text,
            },
        )
        await emit(
            run_id,
            "pipeline",
            "Pipeline failed",
            {"error": str(exc), "traceback": trace_text},
        )
    finally:
        record.completed_at = now_iso()
        write_json(
            logs_dir / "run_status.json",
            {
                "run_id": run_id,
                "status": record.status,
                "started_at": record.started_at,
                "completed_at": record.completed_at,
                "error": record.error,
                "outputs": record.outputs,
            },
        )
