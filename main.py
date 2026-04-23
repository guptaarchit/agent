import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import build_llm, run_turn
from config import get_settings
from mcp_client import (
    cache_is_warm,
    cached_specs,
    init_startup_cache,
    load_mcp_tools,
    refresh_cache,
)
import mcp_client

from langgraph.checkpoint.memory import MemorySaver

# Preferred: Redis Stack saver (requires RediSearch module on the Redis server)
try:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
    _REDIS_STACK_IMPORT_ERROR = None
except Exception as e:
    AsyncRedisSaver = None
    _REDIS_STACK_IMPORT_ERROR = e

# Fallback 1: our plain-Redis saver (any Redis 5+)
from redis_saver import AsyncPlainRedisSaver

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    _SQLITE_IMPORT_ERROR = None
except Exception as e:
    AsyncSqliteSaver = None
    _SQLITE_IMPORT_ERROR = e


# Configure logging at import time, with force=True so we override any
# root handlers uvicorn may have installed. Without this, logger.info
# calls in our modules are silently swallowed.
_settings = get_settings()
logging.basicConfig(
    level=_settings.log_level,
    format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger("agent")
logger.setLevel(_settings.log_level)


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question")
    session_id: Optional[str] = Field(
        None, description="Existing session id; a new one is generated if omitted"
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str


def require_auth(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    return authorization


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    app.state.llm = build_llm()
    app.state.checkpointer_cm = None
    app.state.checkpointer_kind = None
    app.state.checkpointer_path = None

    kind = settings.checkpointer.lower()

    if kind == "redis" and settings.redis_url:
        # Preferred path — Redis Stack saver (full-featured, needs RediSearch)
        if AsyncRedisSaver is not None:
            try:
                cm = AsyncRedisSaver.from_conn_string(settings.redis_url)
                checkpointer = await cm.__aenter__()
                app.state.checkpointer_cm = cm
                app.state.checkpointer = checkpointer
                app.state.checkpointer_kind = "redis-stack"
                app.state.checkpointer_path = settings.redis_url
                logger.info(
                    "Using Redis Stack checkpointer at %s", settings.redis_url
                )
            except Exception as e:
                logger.warning(
                    "Redis Stack saver failed (%s) — needs RediSearch module. "
                    "Falling back to plain-Redis saver.",
                    e,
                )
        else:
            logger.info(
                "langgraph-checkpoint-redis not installed (%s); skipping Redis Stack path",
                _REDIS_STACK_IMPORT_ERROR,
            )

        # Fallback 1 — plain Redis, no RediSearch needed
        if app.state.checkpointer_cm is None:
            try:
                cm = AsyncPlainRedisSaver.from_url(settings.redis_url)
                checkpointer = await cm.__aenter__()
                app.state.checkpointer_cm = cm
                app.state.checkpointer = checkpointer
                app.state.checkpointer_kind = "redis-plain"
                app.state.checkpointer_path = settings.redis_url
                logger.info(
                    "Using plain-Redis checkpointer at %s", settings.redis_url
                )
            except Exception as e:
                logger.warning(
                    "Plain-Redis saver failed (%s); falling back to SQLite.", e
                )
                kind = "sqlite"

    if app.state.checkpointer_cm is None and kind == "sqlite":
        if AsyncSqliteSaver is None:
            logger.warning(
                "AsyncSqliteSaver unavailable: %s. Install `langgraph-checkpoint-sqlite` "
                "and `aiosqlite` to enable persistent sessions.",
                _SQLITE_IMPORT_ERROR,
            )
        else:
            try:
                abs_sqlite = os.path.abspath(settings.sqlite_path)
                cm = AsyncSqliteSaver.from_conn_string(settings.sqlite_path)
                checkpointer = await cm.__aenter__()
                app.state.checkpointer_cm = cm
                app.state.checkpointer = checkpointer
                app.state.checkpointer_kind = "sqlite"
                app.state.checkpointer_path = abs_sqlite
                logger.info("Using SQLite checkpointer at %s", abs_sqlite)
            except Exception as e:
                logger.warning(
                    "SQLite checkpointer init failed (%s); falling back to memory", e
                )

    if app.state.checkpointer_cm is None:
        app.state.checkpointer = MemorySaver()
        app.state.checkpointer_kind = "memory"
        logger.warning("Using in-memory checkpointer (sessions lost on restart)")

    try:
        await init_startup_cache(settings.mcp_bootstrap_auth)
    except Exception as e:
        logger.warning(
            "MCP startup cache failed (%s). Falling back to per-request discovery.",
            e,
        )

    # Summary banner — one place to see everything on startup.
    banner = [
        "=" * 68,
        "  Opensource MCP Agent — startup complete",
        f"  env           : {settings.environment}",
        f"  listening on  : http://{settings.host}:{settings.port}",
        f"  checkpointer  : {app.state.checkpointer_kind} "
        f"({app.state.checkpointer_path or 'in-process'})",
        f"  mcp endpoint  : {settings.mcp_server_endpoint}",
        f"  mcp cache     : {'WARM (' + str(len(cached_specs())) + ' tools)' if cache_is_warm() else 'COLD (set MCP_BOOTSTRAP_AUTH to warm at startup)'}",
        "=" * 68,
    ]
    for line in banner:
        logger.info(line)

    try:
        yield
    finally:
        cm = app.state.checkpointer_cm
        if cm is not None:
            await cm.__aexit__(None, None, None)


app = FastAPI(title="Opensource MCP Agent", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "environment": get_settings().environment}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, authorization: str = Depends(require_auth)):
    session_id = req.session_id or str(uuid.uuid4())
    logger.info(
        "chat request: session=%s cache=%s",
        session_id,
        "warm" if cache_is_warm() else "cold",
    )
    try:
        tools = await load_mcp_tools(authorization)
    except Exception as e:
        logger.exception("Failed to load MCP tools")
        raise HTTPException(status_code=502, detail=f"MCP connection failed: {e}")

    try:
        answer = await run_turn(
            llm=app.state.llm,
            tools=tools,
            checkpointer=app.state.checkpointer,
            thread_id=session_id,
            user_message=req.message,
        )
    except Exception as e:
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return ChatResponse(session_id=session_id, answer=answer)


@app.get("/admin/tools")
async def list_cached_tools(_: str = Depends(require_auth)):
    """Return the MCP tool schemas cached at startup."""
    return {"warm": cache_is_warm(), "tools": cached_specs()}


@app.get("/admin/tools/{tool_name}/schema")
async def tool_schema(tool_name: str, _: str = Depends(require_auth)):
    """Return both the raw MCP schema and the pydantic-generated schema
    that the LLM actually sees for a given tool."""
    from mcp_client import _build_tool  # local import to avoid cycles at module load

    spec = next((s for s in cached_specs() if s["name"] == tool_name), None)
    if not spec:
        raise HTTPException(404, f"Tool {tool_name!r} not in cache")
    tool = _build_tool(spec, "dummy")
    return {
        "name": tool_name,
        "mcp_input_schema": spec["input_schema"],
        "llm_visible_schema": tool.args_schema.model_json_schema(),
        "required": (spec["input_schema"] or {}).get("required", []),
    }


@app.post("/admin/tools/refresh")
async def refresh_cached_tools(authorization: str = Depends(require_auth)):
    """Re-fetch tool schemas from the MCP gateway using the caller's auth."""
    specs = await refresh_cache(authorization)
    return {"warm": True, "count": len(specs), "tools": [s["name"] for s in specs]}


@app.get("/admin/gateway")
async def gateway_info(_: str = Depends(require_auth)):
    """Show what the gateway exposes and which call-meta-tool we resolved to.

    Useful when tool invocation fails with "Unknown tool" — compare the
    gateway names here against the default in MCP_CALL_TOOL_TOOL.
    """
    return {
        "gateway_names": mcp_client._gateway_names,
        "invoke_via_meta_tool": mcp_client._invoke_via_meta_tool,
        "active_call_meta_tool": mcp_client._active_call_meta_tool,
        "list_tool_configured": get_settings().mcp_list_tools_tool,
        "schema_tool_configured": get_settings().mcp_schema_tool,
        "call_tool_configured": get_settings().mcp_call_tool_tool,
    }


@app.get("/admin/sessions")
async def list_sessions(_: str = Depends(require_auth)):
    """List all persisted session thread ids from the checkpointer."""
    checkpointer = app.state.checkpointer
    seen: set[str] = set()
    entries = []
    if hasattr(checkpointer, "alist"):
        async for tup in checkpointer.alist(None):
            tid = (tup.config or {}).get("configurable", {}).get("thread_id")
            if tid and tid not in seen:
                seen.add(tid)
                entries.append({"thread_id": tid})
    return {
        "storage": app.state.checkpointer_kind,
        "path": app.state.checkpointer_path,
        "count": len(entries),
        "sessions": entries,
    }


@app.get("/sessions/{session_id}/history")
async def history(session_id: str, _: str = Depends(require_auth)):
    """Return persisted messages for a session (for UI rehydration)."""
    checkpointer = app.state.checkpointer
    config = {"configurable": {"thread_id": session_id}}
    snapshot = await checkpointer.aget_tuple(config) if hasattr(checkpointer, "aget_tuple") else checkpointer.get_tuple(config)
    if not snapshot:
        return {"session_id": session_id, "messages": []}
    messages = snapshot.checkpoint.get("channel_values", {}).get("messages", [])
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": getattr(m, "type", "unknown"),
                "content": getattr(m, "content", str(m)),
            }
            for m in messages
        ],
    }


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "main:app",
        host=s.host,
        port=s.port,
        log_level=s.log_level.lower(),
        reload=s.reload,
        reload_dirs=["."],
        reload_excludes=[
            "sessions.sqlite*",
            "mcp_cache.json*",
            "*.pyc",
            "__pycache__",
        ],
    )
