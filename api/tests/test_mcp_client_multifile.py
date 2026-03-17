from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any
from urllib import request


PROTOCOL_VERSION = "2025-06-18"


class McpHttpClient:
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip("/") + "/"
        self.session_id: str | None = None

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], request.addinfourl]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.endpoint,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json,text/event-stream",
                "MCP-Protocol-Version": PROTOCOL_VERSION,
                **({"Mcp-Session-Id": self.session_id} if self.session_id else {}),
            },
        )
        resp = request.urlopen(req, timeout=900)
        body = resp.read().decode("utf-8")
        parsed = json.loads(body) if body else {}
        return parsed, resp

    def initialize(self) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "mcp-multifile-test", "version": "0.1.0"},
            },
        }
        response, raw = self._post(payload)
        session_id = raw.headers.get("Mcp-Session-Id")
        if session_id:
            self.session_id = session_id
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        return response

    def call_tool(self, name: str, arguments: dict[str, Any], request_id: int = 2) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        }
        response, _ = self._post(payload)
        return response


def parse_tool_result(response: dict[str, Any]) -> dict[str, Any]:
    if "error" in response:
        raise RuntimeError(f"MCP protocol error: {response['error']}")

    result = response.get("result", {})
    if result.get("isError"):
        text = "\n".join(
            item.get("text", "")
            for item in result.get("content", [])
            if item.get("type") == "text"
        )
        raise RuntimeError(f"Tool execution error: {text or result}")

    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured

    text_parts = [
        item.get("text", "")
        for item in result.get("content", [])
        if item.get("type") == "text"
    ]
    if not text_parts:
        return {}
    joined = "\n".join(text_parts).strip()
    try:
        return json.loads(joined)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Unable to parse tool output as JSON: {joined[:500]}...") from exc


def to_csv_payload(path: Path) -> dict[str, str]:
    return {
        "filename": path.name,
        "content_base64": base64.b64encode(path.read_bytes()).decode("utf-8"),
    }


def expected_report_names(csv_paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in csv_paths:
        stem = path.stem
        names.add(f"{stem}_sweetviz.html")
        names.add(f"{stem}_ydata_profiling.html")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-file MCP report validation test")
    parser.add_argument("--endpoint", default="http://localhost:8001/mcp/", help="MCP endpoint URL")
    parser.add_argument(
        "--csv",
        action="append",
        default=["../testing_data/Bank_Churn.csv", "../testing_data/account_info.csv"],
        help="CSV file path (repeat flag for multiple files)",
    )
    parser.add_argument(
        "--instructions",
        default="Clean these files for downstream analytics while preserving useful business columns.",
        help="Instruction sent to the MCP tool",
    )
    args = parser.parse_args()

    csv_paths = [Path(p).resolve() for p in args.csv]
    missing = [str(p) for p in csv_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"CSV files not found: {missing}")

    client = McpHttpClient(args.endpoint)
    init_result = client.initialize()
    print("Initialized:", json.dumps(init_result.get("result", {}), indent=2)[:500])

    call_response = client.call_tool(
        "generate_notebook_from_csvs",
        {
            "instructions": args.instructions,
            "csv_files": [to_csv_payload(p) for p in csv_paths],
        },
        request_id=3,
    )

    result = parse_tool_result(call_response)
    print("Run status:", result.get("status"), "run_id:", result.get("run_id"))

    report_files = set(result.get("report_files") or [])
    expected = expected_report_names(csv_paths)
    missing_reports = sorted(expected - report_files)

    print("Expected report files:", sorted(expected))
    print("Returned report files:", sorted(report_files))

    if result.get("status") not in {"completed", "completed_with_validation_errors"}:
        raise AssertionError(f"Unexpected run status: {result.get('status')} | error={result.get('error')}")

    if missing_reports:
        raise AssertionError(f"Missing expected report files: {missing_reports}")

    notebook_b64 = result.get("notebook_base64")
    if not notebook_b64:
        raise AssertionError("Notebook payload missing in MCP response")

    print("PASS: Multi-file MCP call returned notebook and both report types per CSV input.")


if __name__ == "__main__":
    main()
