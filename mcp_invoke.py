"""Invokes a selected tool on the MCP gateway and returns its response.

Because the MCP endpoint is an aggregator, invocation goes through a meta-tool
(default `call_internal_tool`, override with `MCP_CALL_TOOL_TOOL`) rather than
`session.call_tool(<real name>)` directly. Payload shape follows MCP's own
convention: `{name, arguments}`.

`invoke_tool(...)` returns an `InvocationResult` with the parsed text/JSON
output and an `is_error` flag reflecting the MCP result. Parse failures and
transport errors raise `InvocationError` for the route layer to translate.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

log = logging.getLogger(__name__)


class InvocationError(Exception):
    """Raised when the tool invocation cannot be completed."""


@dataclass
class InvocationResult:
    result: Any
    is_error: bool


def _extract_text(result: Any) -> str:
    texts = [
        block.text
        for block in (getattr(result, "content", None) or [])
        if getattr(block, "text", None)
    ]
    return "\n".join(texts).strip()


def _parse_result(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def invoke_tool(
    *,
    endpoint: str,
    timeout_seconds: float,
    call_tool_name: str,
    tool_name: str,
    arguments: dict[str, Any],
    auth_header: str | None = None,
) -> InvocationResult:
    payload = {"tool_name": tool_name, "arguments": arguments}
    log.info("Invoking %s via %s with args=%s", tool_name, call_tool_name, arguments)

    raw: Any = None
    gateway_names: set[str] = set()
    headers = {"Authorization": auth_header} if auth_header else None

    try:
        async with (
            asyncio.timeout(timeout_seconds),
            streamablehttp_client(endpoint, headers=headers) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()

            gateway_names = {t.name for t in (await session.list_tools()).tools}
            if call_tool_name in gateway_names:
                raw = await session.call_tool(call_tool_name, payload)
            elif tool_name in gateway_names:
                log.info("Call meta-tool '%s' absent; invoking '%s' directly", call_tool_name, tool_name)
                raw = await session.call_tool(tool_name, arguments)
    except asyncio.TimeoutError as e:
        raise InvocationError(f"MCP invocation timed out after {timeout_seconds}s") from e

    if raw is None:
        raise InvocationError(
            f"Gateway exposes {sorted(gateway_names)} — set MCP_CALL_TOOL_TOOL "
            f"to the one that invokes '{tool_name}' (tried '{call_tool_name}')."
        )

    text = _extract_text(raw)
    return InvocationResult(result=_parse_result(text), is_error=bool(getattr(raw, "isError", False)))
