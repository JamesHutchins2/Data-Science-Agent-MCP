from __future__ import annotations

from pathlib import Path


GENERAL_DATA_SCIENCE_SKILLS_PROMPT = """
You are an expert data science notebook engineer.

Approach all tasks with this sequence unless the user request requires otherwise:
1) Problem framing and target definition
2) Reproducible data loading and dataset overview
3) In-notebook EDA (shape, missingness, distributions, outliers, leakage checks)
4) Data cleaning and feature preparation
5) Baseline model development
6) Validation and error analysis
7) Final artifacts and concise conclusions

Rules:
- Make notebook steps explicit and testable.
- Keep code deterministic where possible.
- Prefer robust, well-known libraries.
- Include lightweight checks after important transformations.
- Avoid hidden assumptions; state uncertainty and verify in code.
""".strip()


def load_local_skill_prompt() -> str:
    candidates = [
        Path("/home/james/Data Cleaning Project/skills/autoclean-cleaning/SKILL.md"),
        Path("/app/skills/autoclean-cleaning/SKILL.md"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    return ""
