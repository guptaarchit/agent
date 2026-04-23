"""MCP integration for a gateway that aggregates many tools behind meta-tools.

Gateway pattern:
  - `session.list_tools()` returns only meta-tools (list / schema / call)
  - The real tool catalog comes from calling the list meta-tool
  - Per-tool schemas come from the schema meta-tool (one call per tool, fanned
    out concurrently)
  - Invocation goes through the call meta-tool with
    `{"tool_name": "<real>", "arguments": {...}}`

On startup, one MCP session (authenticated with MCP_BOOTSTRAP_AUTH) fetches
the catalog and hydrates schemas, then everything is cached in-process. Each
`/chat` request builds StructuredTools from the cache; invocations open a
fresh MCP session with the caller's own auth header.

If the gateway exposes tools directly (the list meta-tool is missing), we
fall back to using gateway tools as-is — the same LangChain tool shape, just
routed differently on the wire.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field, create_model

from config import get_settings


logger = logging.getLogger("agent")


# Bump this whenever the on-disk cache shape changes to force a refresh.
_CACHE_VERSION = 2

# Populated by init_startup_cache or the first /chat request as a fallback.
_tool_specs: List[dict] = []

# Whether to route tool calls through the call meta-tool. Determined at startup
# by checking which tools the gateway advertises.
_invoke_via_meta_tool: bool = False

# Resolved call-meta-tool name (may differ from settings default via auto-detect).
_active_call_meta_tool: Optional[str] = None

# Names the gateway advertises on tools/list — surfaced via /admin/gateway
# so users can discover the right MCP_CALL_TOOL_TOOL name if auto-detect misses.
_gateway_names: List[str] = []


# ---------- public API ----------

def cache_is_warm() -> bool:
    return bool(_tool_specs)


def cached_specs() -> List[dict]:
    return list(_tool_specs)


async def init_startup_cache(bootstrap_auth: str) -> None:
    """Populate the in-memory cache at app startup.

    Priority:
      1. Fresh disk cache (mcp_cache.json) if within TTL — loads instantly and
         survives auto-reloads without re-hitting the gateway.
      2. Live fetch using MCP_BOOTSTRAP_AUTH if provided.
      3. Otherwise defer — first /chat call will populate using caller's auth.
    """
    if _load_cache_from_disk():
        return
    if not bootstrap_auth:
        logger.warning(
            "No disk cache and MCP_BOOTSTRAP_AUTH not set — skipping startup "
            "cache. The first /chat call will populate it."
        )
        return
    try:
        await refresh_cache(bootstrap_auth)
    except Exception:
        logger.exception("Startup cache failed — per-request discovery will be used")


def _load_cache_from_disk() -> bool:
    """Return True if a fresh cache was found and loaded."""
    global _tool_specs, _invoke_via_meta_tool, _active_call_meta_tool, _gateway_names
    settings = get_settings()
    path = settings.mcp_cache_file
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("version") != _CACHE_VERSION:
            logger.info(
                "Disk cache schema version mismatch (got %s, want %s) — discarding",
                data.get("version"),
                _CACHE_VERSION,
            )
            return False
        age = time.time() - float(data.get("timestamp", 0))
        if age > settings.mcp_cache_ttl:
            logger.info(
                "Disk cache at %s is %.0fs old (> %ds TTL) — will refresh",
                path,
                age,
                settings.mcp_cache_ttl,
            )
            return False
        specs = data.get("specs") or []
        if not specs:
            return False
        for s in specs:
            if not isinstance(s.get("input_schema"), dict):
                s["input_schema"] = {"type": "object", "properties": {}}
            model, name_map = _jsonschema_to_pydantic(s["input_schema"], s["name"])
            s["_args_model"] = model
            s["_name_map"] = name_map
        _tool_specs = specs
        _invoke_via_meta_tool = bool(data.get("invoke_via_meta_tool", False))
        _active_call_meta_tool = data.get("active_call_meta_tool")
        _gateway_names = data.get("gateway_names") or []
        logger.info(
            "Loaded MCP cache from disk: %d tools, age=%.0fs, "
            "call_meta=%s, route=%s",
            len(specs),
            age,
            _active_call_meta_tool,
            "meta-tool" if _invoke_via_meta_tool else "direct",
        )
        return True
    except Exception as e:
        logger.warning("Failed to load cache from %s: %s", path, e)
        return False


def _save_cache_to_disk() -> None:
    settings = get_settings()
    path = settings.mcp_cache_file
    payload = {
        "version": _CACHE_VERSION,
        "timestamp": time.time(),
        "invoke_via_meta_tool": _invoke_via_meta_tool,
        "active_call_meta_tool": _active_call_meta_tool,
        "gateway_names": _gateway_names,
        "specs": [
            {
                "name": s["name"],
                "description": s["description"],
                "input_schema": s["input_schema"],
            }
            for s in _tool_specs
        ],
    }
    try:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
        logger.info("Persisted MCP cache to %s (%d tools)", path, len(_tool_specs))
    except Exception as e:
        logger.warning("Failed to persist cache to %s: %s", path, e)


_CALL_META_CANDIDATES = [
    "call_internal_tool",
    "run_tool_with_params",
    "run_internal_tool",
    "invoke_internal_tool",
    "execute_internal_tool",
    "dispatch_internal_tool",
    "call_tool",
    "run_tool",
    "invoke_tool",
    "execute_tool",
]


def _auto_detect_call_meta(gateway_names: List[str], configured: str) -> Optional[str]:
    """Pick the gateway's call meta-tool: user override > known names > heuristic."""
    if configured in gateway_names:
        return configured
    for name in _CALL_META_CANDIDATES:
        if name in gateway_names:
            return name
    # Heuristic: something with "tool" and one of call/run/invoke/execute/dispatch
    for gn in gateway_names:
        low = gn.lower()
        if "tool" in low and any(
            kw in low for kw in ("call", "run", "invoke", "execute", "dispatch")
        ):
            return gn
    return None


async def refresh_cache(authorization: str) -> List[dict]:
    """(Re)discover the catalog and hydrate schemas, using the given auth."""
    global _tool_specs, _invoke_via_meta_tool, _active_call_meta_tool, _gateway_names

    settings = get_settings()
    headers = {"Authorization": authorization} if authorization else {}
    list_meta = settings.mcp_list_tools_tool
    schema_meta = settings.mcp_schema_tool

    async with (
        asyncio.timeout(settings.mcp_server_timeout),
        streamablehttp_client(
            settings.mcp_server_endpoint, headers=headers
        ) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        gateway = await session.list_tools()
        gateway_names = [t.name for t in gateway.tools]
        _gateway_names = gateway_names
        logger.info(
            "Gateway advertises %d tools: %s", len(gateway_names), gateway_names
        )

        call_meta = _auto_detect_call_meta(
            gateway_names, settings.mcp_call_tool_tool
        )
        if call_meta:
            logger.info("Call meta-tool resolved to '%s'", call_meta)
        _active_call_meta_tool = call_meta

        if list_meta in gateway_names:
            # Aggregator pattern — follow the meta-tool chain
            _invoke_via_meta_tool = call_meta is not None
            if not _invoke_via_meta_tool:
                logger.warning(
                    "No call meta-tool found on gateway (tried '%s' + %d candidates). "
                    "Gateway names: %s. Set MCP_CALL_TOOL_TOOL=<real_name> in .env.",
                    settings.mcp_call_tool_tool,
                    len(_CALL_META_CANDIDATES),
                    gateway_names,
                )

            raw = await session.call_tool(list_meta, {})
            specs = _parse_list_result(raw)
            logger.info("Catalog via '%s': %d tools", list_meta, len(specs))

            if schema_meta in gateway_names:
                await _hydrate_schemas(
                    session, schema_meta, specs, settings.mcp_schema_concurrency
                )
            else:
                logger.warning(
                    "Schema meta-tool '%s' not on gateway — schemas unavailable",
                    schema_meta,
                )
        else:
            # Direct pattern — expose gateway tools as first-class
            logger.info(
                "List meta-tool '%s' absent — exposing %d gateway tools directly",
                list_meta,
                len(gateway.tools),
            )
            _invoke_via_meta_tool = False
            specs = [
                {
                    "name": t.name,
                    "description": (t.description or "").strip(),
                    "input_schema": t.inputSchema or None,
                }
                for t in gateway.tools
            ]

    # Ensure every spec has a dict schema so tool construction can't fail,
    # then precompute pydantic models so /chat requests don't re-do this work.
    for s in specs:
        if not isinstance(s.get("input_schema"), dict):
            s["input_schema"] = {"type": "object", "properties": {}}
        model, name_map = _jsonschema_to_pydantic(s["input_schema"], s["name"])
        s["_args_model"] = model
        s["_name_map"] = name_map
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug(
                    "Tool %s schema: %s", s["name"], model.model_json_schema()
                )
            except Exception:
                pass

    _tool_specs = specs
    logger.info(
        "MCP cache ready: %d tools, routing via %s (call_meta=%s)",
        len(specs),
        "meta-tool" if _invoke_via_meta_tool else "direct",
        _active_call_meta_tool,
    )
    _save_cache_to_disk()
    return _tool_specs


async def load_mcp_tools(authorization: str) -> List[BaseTool]:
    if not _tool_specs:
        logger.warning(
            "Cache cold on /chat — will fetch catalog + schemas now. "
            "Set MCP_BOOTSTRAP_AUTH or wait for the first successful request to warm it."
        )
        await refresh_cache(authorization)
    else:
        logger.info("Cache hit: using %d cached tools", len(_tool_specs))
    return [_build_tool(spec, authorization) for spec in _tool_specs]


# ---------- MCP response parsing ----------

def _extract_text(result: Any) -> str:
    texts = [
        block.text
        for block in (getattr(result, "content", None) or [])
        if getattr(block, "text", None)
    ]
    return "\n".join(texts).strip()


def _unwrap_list(data: Any) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("tools", "items", "data", "result"):
            if isinstance(data.get(key), list):
                return data[key]
    raise RuntimeError(
        f"list meta-tool returned unexpected shape: {type(data).__name__}"
    )


def _item_to_spec(item: Any) -> Optional[dict]:
    if isinstance(item, str):
        return {"name": item, "description": "", "input_schema": None}
    if not isinstance(item, dict):
        return None
    name = item.get("name") or item.get("tool_name") or item.get("id")
    if not name:
        return None
    desc = (
        item.get("description")
        or item.get("summary")
        or item.get("doc")
        or ""
    )
    schema = (
        item.get("inputSchema")
        or item.get("input_schema")
        or item.get("parameters")
    )
    return {
        "name": str(name),
        "description": str(desc).strip(),
        "input_schema": schema if isinstance(schema, dict) else None,
    }


def _parse_list_result(raw: Any) -> List[dict]:
    text = _extract_text(raw)
    if not text:
        raise RuntimeError("list meta-tool returned no text content")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"list meta-tool output is not JSON: {text[:400]}"
        ) from e
    return [s for item in _unwrap_list(data) if (s := _item_to_spec(item))]


def _parse_schema_payload(raw: Any, tool_name: str) -> dict:
    text = _extract_text(raw)
    if not text:
        raise RuntimeError(f"schema for '{tool_name}' was empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"schema for '{tool_name}' is not JSON: {text[:400]}"
        ) from e
    if isinstance(data, dict):
        for key in ("inputSchema", "input_schema", "schema", "parameters"):
            if isinstance(data.get(key), dict):
                return data[key]
        return data
    raise RuntimeError(
        f"schema for '{tool_name}' is not an object: {type(data).__name__}"
    )


async def _hydrate_schemas(
    session: ClientSession,
    schema_tool: str,
    specs: List[dict],
    concurrency: int,
) -> None:
    pending = [s for s in specs if s.get("input_schema") is None]
    if not pending:
        return
    sem = asyncio.Semaphore(concurrency)

    async def one(spec: dict) -> bool:
        async with sem:
            try:
                raw = await session.call_tool(
                    schema_tool, {"tool_name": spec["name"]}
                )
                spec["input_schema"] = _parse_schema_payload(raw, spec["name"])
                return True
            except Exception as e:
                logger.warning("Schema fetch failed for '%s': %s", spec["name"], e)
                return False

    results = await asyncio.gather(*(one(s) for s in pending))
    logger.info(
        "Hydrated schemas: %d/%d (concurrency=%d)",
        sum(results),
        len(pending),
        concurrency,
    )


# ---------- per-tool wrappers ----------

def _flatten_exception(e: BaseException) -> str:
    """Unwrap ExceptionGroup (Py 3.11+ asyncio TaskGroup wrapper) to a readable string."""
    parts: list[str] = []

    def walk(exc: BaseException) -> None:
        if isinstance(exc, BaseExceptionGroup):
            for sub in exc.exceptions:
                walk(sub)
        else:
            parts.append(f"{type(exc).__name__}: {exc}")

    walk(e)
    return "; ".join(parts) if parts else repr(e)


def _build_tool(spec: dict, authorization: str) -> BaseTool:
    """Build a StructuredTool that closes over the caller's auth.

    Pydantic model + name_map are precomputed at startup; this function
    is a near-constant-time factory per request.
    """
    settings = get_settings()
    tool_name: str = spec["name"]
    headers = {"Authorization": authorization} if authorization else {}
    use_meta = _invoke_via_meta_tool
    call_meta = _active_call_meta_tool or settings.mcp_call_tool_tool
    args_schema = spec.get("_args_model") or _jsonschema_to_pydantic(
        spec["input_schema"], tool_name
    )[0]
    name_map: dict = spec.get("_name_map") or {}

    async def _invoke(**kwargs: Any) -> str:
        # Drop values the LLM didn't set — sending explicit null to gateways
        # expecting "omitted" causes spurious validation errors.
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        # Restore original property names for fields we renamed to keep
        # pydantic happy (e.g. leading-underscore keys).
        if name_map:
            cleaned = {name_map.get(k, k): v for k, v in cleaned.items()}
        logger.info(
            "MCP call: %s args=%s (route=%s%s)",
            tool_name,
            cleaned,
            "meta-tool via " + call_meta if use_meta else "direct",
            "",
        )
        raw = None
        try:
            async with (
                asyncio.timeout(settings.mcp_server_timeout),
                streamablehttp_client(
                    settings.mcp_server_endpoint, headers=headers
                ) as (read, write, _),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                if use_meta:
                    payload = {"tool_name": tool_name, "arguments": cleaned}
                    raw = await session.call_tool(call_meta, payload)
                else:
                    raw = await session.call_tool(tool_name, cleaned)
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"MCP call for {tool_name} timed out") from e
        except BaseException as e:
            detail = _flatten_exception(e)
            logger.warning("MCP transport error for %s: %s", tool_name, detail)
            raise RuntimeError(
                f"MCP transport error for {tool_name}: {detail}"
            ) from e

        text = _extract_text(raw)
        if getattr(raw, "isError", False):
            # Log the full raw response so we can see what the gateway actually said
            logger.warning(
                "MCP tool %s returned isError=True. Response text: %s",
                tool_name,
                text[:800] or "<empty>",
            )
            raise RuntimeError(
                text or f"MCP tool {tool_name} returned an error"
            )
        return text or ""

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=tool_name,
        description=spec["description"],
        args_schema=args_schema,
    )


# ---------- JSON Schema → pydantic ----------

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
}


def _py_type(prop: dict) -> Any:
    t = prop.get("type", "string")
    if isinstance(t, list):
        # e.g. ["string", "null"] — pick the first non-null
        t = next((x for x in t if x != "null"), "string")
    if t == "array":
        items = prop.get("items") or {}
        inner = _py_type(items) if items else Any
        return List[inner]
    return _JSON_TO_PY.get(t, str)


def _sanitize_field_name(original: str) -> str:
    """Make a JSON-schema property name safe as a pydantic field.

    Pydantic refuses leading underscores; Python refuses non-identifier chars.
    We mangle the name and keep a reverse map so we can restore it when
    calling MCP.
    """
    safe = original
    if safe.startswith("_"):
        safe = "arg" + safe
    if not safe.isidentifier():
        safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in safe)
        if safe and safe[0].isdigit():
            safe = "_" + safe
    return safe or "arg"


def _jsonschema_to_pydantic(
    schema: dict, name: str
) -> tuple[Type[BaseModel], dict[str, str]]:
    """Return (pydantic model, {python_name: original_name}).

    The caller uses the map to translate kwargs back to original MCP keys
    before invocation, so renames are invisible on the wire.
    """
    properties = (schema or {}).get("properties") or {}
    required = set((schema or {}).get("required") or [])
    fields: dict[str, tuple] = {}
    name_map: dict[str, str] = {}

    for prop_name, prop_schema in properties.items():
        prop_schema = prop_schema or {}
        py_type = _py_type(prop_schema)
        description = prop_schema.get("description") or ""
        enum = prop_schema.get("enum")
        if enum:
            description = (description + f" Allowed values: {enum}.").strip()

        field_name = _sanitize_field_name(prop_name)
        if field_name != prop_name:
            name_map[field_name] = prop_name

        if prop_name in required:
            fields[field_name] = (py_type, Field(..., description=description))
        else:
            fields[field_name] = (
                Optional[py_type],
                Field(default=None, description=description),
            )

    safe_model_name = (
        "".join(ch if ch.isalnum() else "_" for ch in name) or "Tool"
    )
    if not fields:
        return create_model(f"{safe_model_name}_Args"), name_map
    return create_model(f"{safe_model_name}_Args", **fields), name_map
