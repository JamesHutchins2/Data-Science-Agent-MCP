# Data Science Agent/MCP Service

The folllowing is the implementation of an agentic workflow designed to execute data science tasks.

Specifically the agent is designed to complete a series of steps such as: 

    - EDA
    - Data Cleaning
    - Feature Engineering
    - Model Training
    - Model Validation
    - PCA 

As a standard data scientist might do. 


## Workflow Process

The following diagram provides a high level overview of the workflow implemented within this repository.

```mermaid
flowchart TD
    A[POST /pipeline/start] --> B[Create run_id and run directories]
    B --> C["Save uploaded CSV files to input/"]
    C --> D[Emit event: Run created]
    D --> E[run_pipeline starts]

    E --> F[Build compact auto-EDA summary]
    F --> G["Write eda/summary_compact.json"]
    G --> H["Generate EDA HTML reports per input CSV<br/>Sweetviz + YData Profiling"]
    H --> I["Emit report_url events and persist eda/*.html"]

    I --> J[Build LangGraph state]
    J --> K[Planner agent]
    K --> L[Plan validator]
    L --> M{Plan approved?}
    M -- No, attempts < max --> K
    M -- Yes or max reached --> N[Coder agent]

    N --> O["Write output/agentic_notebook.ipynb"]
    O --> P["Execute notebook<br/>papermill default, nbclient optional"]
    P --> Q[Persist execution summary + action logs]
    Q --> R[Coder validator]
    R --> S{Complete?}
    S -- No, attempts < max --> N
    S -- Yes --> T[Status: completed]
    S -- Max attempts reached --> U[Status: completed_with_validation_errors]

    T --> V[Write final artifacts<br/>plan, review feedback, run status, logs]
    U --> V

    E --> W{Exception?}
    W -- Yes --> X[Status: failed]
    X --> Y["Write error.json + traceback.txt"]
    Y --> V

```

## Skills 

Taking after anthropic's new skills concept, this was designed with a similar idea in mind. Providing md files that outline specific library usage. Currently only added for the use of the py-autoclean library, such a skill base could be expanded to contain core machine learning concepts. For example, the addition of how to apply a bayesian predictor etc. 


## Output 

The result of this workflow is the output of a re-producable jupyter notebook as well as auto-generated EDA Reports. 

It completes this process using an orchistrator, coder, validator pattern. 


## MCP Integration

This project now exposes an MCP server so external agents can call the workflow by sending instructions and CSV files, then receiving notebook artifacts.

### MCP Endpoint

- Streamable HTTP MCP endpoint: `/mcp`
- Full URL in local docker compose setup: `http://localhost:8001/mcp`

### MCP Tools Implemented

1) `generate_notebook_from_csvs`
- Inputs:
    - `instructions` (string)
    - `csv_files` (array of objects with `filename`, `content_base64`)
    - `ml_prompt` (optional string)
    - `system_prompt` (optional string)
    - `timeout_seconds` (optional int)
    - `prefer_executed_notebook` (optional bool)
- Behavior:
    - Creates a new run
    - Decodes and stores input CSV files
    - Executes the full planner/validator/coder pipeline
    - Returns run metadata plus notebook artifact as base64

2) `get_run_status`
- Inputs:
    - `run_id` (string)
- Behavior:
    - Returns current status, input/output/report files, timestamps, and error fields

### Design Notes

- Uses MCP Python SDK `FastMCP` with Streamable HTTP transport.
- Mounted into existing FastAPI app via `app.mount("/mcp", mcp.streamable_http_app())`.
- Reuses the same internal `run_pipeline(...)` orchestration path as the REST API.
- Returns notebook content in base64 to support direct client-side file reconstruction.