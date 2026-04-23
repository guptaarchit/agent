"""Discovers the usable tool catalog from the MCP gateway.

The gateway at `MCP_SERVER_ENDPOINT` is an aggregator — `session.list_tools()`
only returns a handful of meta-tools. The real catalog is fetched by calling a
meta-tool (default name: `list_internal_tools`, override with
`MCP_LIST_TOOLS_TOOL`) and parsing its JSON payload.

`fetch_mcp_tools(...)` is the public entry point; `build_catalog_text(...)`
formats the result for the system prompt. The `_*` helpers are pure parsers and
easy to unit-test in isolation.
"""

import asyncio
import json
import logging
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from ..schemas import ToolInfo

log = logging.getLogger(__name__)


def _extract_text(result: Any) -> str:
    texts = [
        block.text
        for block in (getattr(result, "content", None) or [])
        if getattr(block, "text", None)
    ]
    combined = "\n".join(texts).strip()
    if not combined:
        raise RuntimeError("list-tools meta-tool returned no text content")
    return combined


def _unwrap_list(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("tools", "items", "data", "result"):
            if isinstance(data.get(key), list):
                return data[key]
        raise RuntimeError(f"Unexpected object shape from list-tools: keys={list(data)[:10]}")
    raise RuntimeError(f"list-tools did not return a list: {type(data).__name__}")


def _item_to_tool(item: Any) -> ToolInfo | None:
    if isinstance(item, str):
        return ToolInfo(name=item, description="")
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("tool_name") or item.get("id")
    if not name:
        return None
    desc = item.get("description") or item.get("summary") or item.get("doc") or ""
    schema = item.get("inputSchema") or item.get("input_schema") or item.get("parameters")
    return ToolInfo(
        name=str(name),
        description=str(desc).strip(),
        input_schema=schema if isinstance(schema, dict) else None,
    )


def _parse_list_tools_result(result: Any) -> list[ToolInfo]:
    combined = _extract_text(result)
    try:
        data = json.loads(combined)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"list-tools output is not JSON: {combined[:500]}") from e

    tools = [t for item in _unwrap_list(data) if (t := _item_to_tool(item))]
    if not tools:
        raise RuntimeError(f"list-tools returned no parseable tools: {combined[:500]}")
    return tools


async def _hydrate_schemas(
    session: ClientSession,
    schema_tool_name: str,
    tools: list[ToolInfo],
    concurrency: int,
) -> None:
    """Fills in `tool.input_schema` for each tool that lacks one, via the schema meta-tool.

    Runs up to `concurrency` calls in flight at once on the shared MCP session
    (JSON-RPC request IDs keep responses routed correctly). Failures are logged
    and left as `None` so hydration never blocks the whole startup.
    """
    pending = [t for t in tools if t.input_schema is None]
    if not pending:
        return

    sem = asyncio.Semaphore(concurrency)

    async def hydrate_one(tool: ToolInfo) -> bool:
        async with sem:
            try:
                raw = await session.call_tool(schema_tool_name, {"tool_name": tool.name})
                tool.input_schema = _parse_schema_payload(raw, schema_tool_name, tool.name)
                return True
            except Exception as e:
                log.warning("Schema fetch failed for '%s': %s", tool.name, e)
                return False

    results = await asyncio.gather(*(hydrate_one(t) for t in pending))
    log.info("Cached schemas for %d/%d tools (concurrency=%d)", sum(results), len(tools), concurrency)


async def fetch_mcp_tools(
    *,
    endpoint: str,
    timeout_seconds: float,
    list_tool_name: str,
    schema_tool_name: str | None = None,
    schema_concurrency: int = 10,
) -> list[ToolInfo]:
    async with (
        asyncio.timeout(timeout_seconds),
        streamablehttp_client(endpoint) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        gateway = await session.list_tools()
        gateway_names = [t.name for t in gateway.tools]
        log.info("Gateway tools (%d): %s", len(gateway_names), gateway_names)

        if list_tool_name not in gateway_names:
            log.warning(
                "Meta-tool '%s' not on gateway — falling back to gateway tools",
                list_tool_name,
            )
            return [
                ToolInfo(name=t.name, description=(t.description or "").strip())
                for t in gateway.tools
            ]

        result = await session.call_tool(list_tool_name, {})
        tools = _parse_list_tools_result(result)
        log.info("Discovered %d tools behind gateway", len(tools))

        if schema_tool_name and schema_tool_name in gateway_names:
            await _hydrate_schemas(session, schema_tool_name, tools, schema_concurrency)
        elif schema_tool_name:
            log.warning("Schema meta-tool '%s' not on gateway; skipping schema cache", schema_tool_name)

        return tools


def _parse_schema_payload(raw: Any, schema_tool_name: str, tool_name: str) -> dict[str, Any]:
    text = _extract_text(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{schema_tool_name}('{tool_name}') returned non-JSON: {text[:500]}") from e

    if isinstance(data, dict):
        for key in ("inputSchema", "input_schema", "schema", "parameters"):
            if isinstance(data.get(key), dict):
                return data[key]
        return data
    raise RuntimeError(f"{schema_tool_name}('{tool_name}') returned non-object: {type(data).__name__}")


async def probe_tool_schema(
    *,
    endpoint: str,
    timeout_seconds: float,
    schema_tool_name: str,
    tool_name: str,
) -> dict[str, Any]:
    """Diagnostic call — never raises; returns raw + parsed info for manual inspection."""
    out: dict[str, Any] = {"schema_tool": schema_tool_name, "tool_name": tool_name}
    try:
        async with (
            asyncio.timeout(timeout_seconds),
            streamablehttp_client(endpoint) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            raw = await session.call_tool(schema_tool_name, {"tool_name": tool_name})
    except Exception as e:
        out["call_error"] = f"{type(e).__name__}: {e}"
        return out

    out["is_error"] = bool(getattr(raw, "isError", False))
    out["raw_texts"] = [
        b.text for b in (getattr(raw, "content", None) or []) if getattr(b, "text", None)
    ]
    try:
        out["parsed_schema"] = _parse_schema_payload(raw, schema_tool_name, tool_name)
    except Exception as e:
        out["parse_error"] = f"{type(e).__name__}: {e}"
    return out


async def fetch_tool_schema(
    *,
    endpoint: str,
    timeout_seconds: float,
    schema_tool_name: str,
    tool_name: str,
) -> dict[str, Any]:
    """Fetches the input schema for a single tool via the gateway's schema meta-tool.

    Primarily kept for manual / diagnostic use; `/chat` uses the schema cached on
    each `ToolInfo` via `fetch_mcp_tools(..., schema_tool_name=...)`.
    """
    async with (
        asyncio.timeout(timeout_seconds),
        streamablehttp_client(endpoint) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        raw = await session.call_tool(schema_tool_name, {"tool_name": tool_name})
    return _parse_schema_payload(raw, schema_tool_name, tool_name)


def build_catalog_text(tools: list[ToolInfo]) -> str:
    lines: list[str] = []
    for t in tools:
        lines.append(f"- name: {t.name}")
        if t.description:
            lines.append(f"  description: {t.description}")
        if t.input_schema:
            lines.append(f"  input_schema: {json.dumps(t.input_schema, separators=(',', ':'))}")
    return "\n".join(lines)


def build_planner_catalog(tools: list[ToolInfo]) -> str:
    """Lightweight catalog for the planner — just names + descriptions.

    Schemas are added per-tool later (after the planner selects relevant tools)
    to keep the planner prompt small.
    """
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)
