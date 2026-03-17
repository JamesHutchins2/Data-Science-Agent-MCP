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
        resp = request.urlopen(req, timeout=600)
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
                "clientInfo": {"name": "mcp-test-client", "version": "0.1.0"},
            },
        }
        response, raw = self._post(payload)

        session_id = raw.headers.get("Mcp-Session-Id")
        if session_id:
            self.session_id = session_id

        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        self._post(initialized_notification)
        return response

    def list_tools(self) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        response, _ = self._post(payload)
        return response

    def call_tool(self, name: str, arguments: dict[str, Any], request_id: int = 3) -> dict[str, Any]:
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
        raise RuntimeError(f"MCP error: {response['error']}")

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

    text_blocks = [
        item.get("text", "")
        for item in result.get("content", [])
        if item.get("type") == "text"
    ]
    if not text_blocks:
        return {}

    joined = "\n".join(text_blocks).strip()
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return {"text": joined}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test MCP notebook generation tool")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8001/mcp/",
        help="MCP streamable HTTP endpoint",
    )
    parser.add_argument(
        "--csv",
        default="../testing_data/Bank_Churn.csv",
        help="Path to CSV file to upload",
    )
    parser.add_argument(
        "--instructions",
        default="Clean these files for downstream analytics while preserving useful business columns.",
        help="User instruction for notebook generation",
    )
    parser.add_argument(
        "--output-notebook",
        default="./mcp_test_output.ipynb",
        help="Path where the returned notebook will be written",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    client = McpHttpClient(args.endpoint)

    init_response = client.initialize()
    print("Initialized MCP:", json.dumps(init_response.get("result", {}), indent=2)[:800])

    tools_response = client.list_tools()
    tool_names = [t.get("name") for t in tools_response.get("result", {}).get("tools", [])]
    print("Available tools:", tool_names)

    csv_payload = {
        "filename": csv_path.name,
        "content_base64": base64.b64encode(csv_path.read_bytes()).decode("utf-8"),
    }

    call_response = client.call_tool(
        "generate_notebook_from_csvs",
        {
            "instructions": args.instructions,
            "csv_files": [csv_payload],
        },
        request_id=4,
    )
    output = parse_tool_result(call_response)
    print("Tool output:", json.dumps(output, indent=2)[:1500])

    notebook_b64 = output.get("notebook_base64")
    notebook_name = output.get("notebook_filename")
    if not notebook_b64:
        raise RuntimeError("No notebook returned by generate_notebook_from_csvs")

    output_path = Path(args.output_notebook).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(notebook_b64))
    print(f"Notebook written to: {output_path}")
    if notebook_name:
        print(f"Notebook filename from server: {notebook_name}")


if __name__ == "__main__":
    main()
