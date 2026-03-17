from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nbformat
import pandas as pd
import papermill as pm
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from nbformat import ValidationError

from .models import NotebookCellSpec


class NotebookToolbox:
	def __init__(self, output_dir: Path):
		self.output_dir = output_dir

	def compact_auto_eda_summary(
		self,
		input_dir: Path,
		max_columns: int = 40,
		top_unique_values: int = 5,
	) -> dict[str, Any]:
		summary: dict[str, Any] = {}
		for csv_path in sorted(input_dir.glob("*.csv")):
			df = pd.read_csv(csv_path)
			cols = list(df.columns)
			selected_cols = cols[:max_columns]
			file_summary: dict[str, Any] = {
				"shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
				"sample_columns": selected_cols,
				"dtypes": {column: str(dtype) for column, dtype in df[selected_cols].dtypes.to_dict().items()},
				"missing_count": {
					column: int(count)
					for column, count in df[selected_cols].isna().sum().to_dict().items()
				},
				"duplicates": int(df.duplicated().sum()),
				"uniques": {column: int(df[column].nunique(dropna=False)) for column in selected_cols},
				"value_preview": {},
			}
			for column in selected_cols:
				vals = df[column].dropna().astype(str).value_counts().head(top_unique_values).index.tolist()
				file_summary["value_preview"][column] = vals
			summary[csv_path.name] = file_summary
		return summary

	@staticmethod
	def format_eda_context(summary: dict[str, Any], max_chars: int = 18000) -> str:
		text = json.dumps(summary, indent=2)
		return text[:max_chars]

	def generate_eda_html_reports(
		self,
		input_dir: Path,
		eda_dir: Path,
		max_columns: int = 80,
	) -> list[str]:
		report_files: list[str] = []
		import sweetviz as sv
		from ydata_profiling import ProfileReport

		for csv_path in sorted(input_dir.glob("*.csv")):
			df = pd.read_csv(csv_path)
			selected_columns = list(df.columns)[:max_columns]
			df_selected = df[selected_columns].copy()

			target_col = "Exited" if "Exited" in df_selected.columns else None

			sweetviz_report = sv.analyze(df_selected, target_feat=target_col)
			sweetviz_name = f"{csv_path.stem}_sweetviz.html"
			sweetviz_path = eda_dir / sweetviz_name
			sweetviz_report.show_html(filepath=str(sweetviz_path), open_browser=False, layout="widescreen")
			report_files.append(sweetviz_name)

			ydata_name = f"{csv_path.stem}_ydata_profiling.html"
			ydata_path = eda_dir / ydata_name
			profile = ProfileReport(
				df_selected,
				title=f"YData Profiling Report - {csv_path.name}",
				explorative=True,
			)
			profile.to_file(str(ydata_path))
			report_files.append(ydata_name)

		return report_files

	@staticmethod
	def normalize_cells(cell_specs: list[dict[str, Any]]) -> list[NotebookCellSpec]:
		normalized: list[NotebookCellSpec] = []
		for raw in cell_specs:
			try:
				spec = NotebookCellSpec.model_validate(raw)
			except Exception:
				continue
			language = str(spec.metadata.get("language", "")).strip().lower()
			if not language:
				spec.metadata["language"] = "python" if spec.cell_type == "code" else "markdown"
			normalized.append(spec)
		return normalized

	def write_notebook(self, notebook_path: Path, cell_specs: list[NotebookCellSpec]) -> Path:
		notebook = nbformat.v4.new_notebook()
		notebook.metadata.setdefault(
			"kernelspec",
			{
				"display_name": "Python 3",
				"language": "python",
				"name": "python3",
			},
		)
		notebook.metadata.setdefault(
			"language_info",
			{
				"name": "python",
			},
		)
		notebook.cells = []

		for spec in cell_specs:
			if spec.cell_type == "code":
				notebook.cells.append(
					nbformat.v4.new_code_cell(
						source=spec.source,
						metadata=spec.metadata,
					)
				)
			else:
				notebook.cells.append(
					nbformat.v4.new_markdown_cell(
						source=spec.source,
						metadata=spec.metadata,
					)
				)

		nbformat.validate(notebook)
		nbformat.write(notebook, str(notebook_path))
		return notebook_path

	@staticmethod
	def validate_notebook(notebook_path: Path) -> tuple[bool, str]:
		try:
			nb = nbformat.read(str(notebook_path), as_version=4)
			nbformat.validate(nb)
			return True, "ok"
		except ValidationError as exc:
			return False, str(exc)
		except Exception as exc:
			return False, str(exc)

	def execute_notebook_nbclient(
		self,
		notebook_path: Path,
		executed_path: Path,
		cwd: Path,
		kernel_name: str = "python3",
		timeout: int = 600,
		allow_errors: bool = False,
	) -> dict[str, Any]:
		notebook = nbformat.read(str(notebook_path), as_version=4)
		client = NotebookClient(
			notebook,
			kernel_name=kernel_name,
			timeout=timeout,
			resources={"metadata": {"path": str(cwd)}},
			allow_errors=allow_errors,
		)

		success = True
		error = ""
		try:
			client.execute()
		except CellExecutionError as exc:
			success = False
			error = str(exc)
		finally:
			nbformat.write(notebook, str(executed_path))

		summary = self.execution_summary(executed_path)
		summary["success"] = success
		summary["error"] = error
		return summary

	def execute_notebook_papermill(
		self,
		notebook_path: Path,
		executed_path: Path,
		cwd: Path,
		parameters: dict[str, Any] | None = None,
		kernel_name: str = "python3",
	) -> dict[str, Any]:
		success = True
		error = ""
		try:
			pm.execute_notebook(
				str(notebook_path),
				str(executed_path),
				parameters=parameters or {},
				cwd=str(cwd),
				kernel_name=kernel_name,
			)
		except Exception as exc:
			success = False
			error = str(exc)

		summary = self.execution_summary(executed_path)
		summary["success"] = success
		summary["error"] = error
		return summary

	@staticmethod
	def execution_summary(executed_path: Path) -> dict[str, Any]:
		if not executed_path.exists():
			return {
				"cells_total": 0,
				"code_cells": 0,
				"error_cells": 0,
				"first_error": "executed_notebook_missing",
			}

		notebook = nbformat.read(str(executed_path), as_version=4)
		code_cells = 0
		error_cells = 0
		first_error = ""
		cell_summaries: list[dict[str, Any]] = []

		for index, cell in enumerate(notebook.cells, start=1):
			if cell.get("cell_type") != "code":
				continue
			code_cells += 1
			has_error = False
			error_text = ""
			stream_tail = ""
			for output in cell.get("outputs", []):
				otype = output.get("output_type")
				if otype == "error":
					has_error = True
					ename = str(output.get("ename", ""))
					evalue = str(output.get("evalue", ""))
					error_text = f"{ename}: {evalue}".strip(": ")
				if otype == "stream":
					stream_tail = str(output.get("text", ""))[-400:]

			if has_error:
				error_cells += 1
				if not first_error:
					first_error = error_text or f"error_in_cell_{index}"

			cell_summaries.append(
				{
					"cell_number": index,
					"has_error": has_error,
					"error": error_text,
					"stream_tail": stream_tail,
				}
			)

		return {
			"cells_total": len(notebook.cells),
			"code_cells": code_cells,
			"error_cells": error_cells,
			"first_error": first_error,
			"cells": cell_summaries,
		}

