import json
import logging
import re
from typing import List

from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver

from config import get_settings


logger = logging.getLogger("agent")


SYSTEM_PROMPT = (
    "You are an assistant with access to MCP tools. Your job is to USE the tools "
    "to answer the user, not to interview the user about parameters.\n\n"
    "PARAMETER RULES — very important:\n"
    "• Each tool has a JSON schema. Fields listed under `required` are the ONLY "
    "  ones that must be provided.\n"
    "• If a tool has NO required fields (empty `required` array, or every field is "
    "  Optional / has a default of null), CALL IT WITH `{}` — an empty arguments "
    "  object. Do not ask the user for optional filters unless the first attempt "
    "  returned something that clearly needs narrowing.\n"
    "• For listing/discovery tools (names like list_*, get_*, describe_*, search_*), "
    "  always try with `{}` first. Only pass parameters the user explicitly mentioned.\n"
    "• Never invent values. If a REQUIRED field is missing and cannot be inferred "
    "  from the conversation, then (and only then) ask the user for it.\n\n"
    "ERROR HANDLING:\n"
    "• If a tool returns an error about missing params, look at its schema — if the "
    "  schema says those params are optional, the gateway itself is broken; report "
    "  that plainly to the user and stop. Don't loop retrying.\n"
    "• If the gateway returns 5xx, report it and stop.\n\n"
    "Be concise. Cite tool results. Never fabricate output."
)


TOOL_SELECTOR_PROMPT = (
    "You are a tool-selection assistant. Given a catalog of available tools and "
    "the user's latest message, return the tools that are relevant for answering "
    "it. Include tools for the exact operation AND closely related ones (for "
    "example, include list/get/describe siblings when the user asks for a listing, "
    "and include update/delete siblings if the user is likely to act on the results). "
    "Prefer recall over precision — extra tools don't hurt, but missing ones do.\n\n"
    "Return ONLY a JSON array of tool-name strings. No prose, no markdown."
)


def _handle_tool_error(exc: Exception) -> str:
    logger.warning("Tool error surfaced to agent: %s", exc)
    return (
        f"Tool call failed: {exc}. Do not retry this exact call; either adjust "
        f"parameters or explain the failure to the user."
    )


def build_llm() -> AzureChatOpenAI:
    s = get_settings()
    return AzureChatOpenAI(
        azure_endpoint=s.azure_openai_endpoint,
        azure_deployment=s.azure_openai_deployment,
        api_version=s.azure_openai_api_version,
        api_key=s.azure_openai_api_key,
        temperature=0,
    )


def _keyword_rank(tools: List[BaseTool], query: str) -> List[BaseTool]:
    words = set(re.findall(r"\w+", (query or "").lower()))
    if not words:
        return list(tools)

    def score(t: BaseTool) -> int:
        text = (t.name + " " + (t.description or "")).lower()
        return sum(1 for w in words if w in text)

    return sorted(tools, key=score, reverse=True)


async def select_relevant_tools(
    *,
    llm: AzureChatOpenAI,
    all_tools: List[BaseTool],
    user_message: str,
    max_tools: int,
) -> List[BaseTool]:
    """Narrow the tool set before handing it to the react agent.

    Uses a fast LLM-based planner; falls back to keyword scoring if the
    planner fails or returns nothing usable.
    """
    if len(all_tools) <= max_tools:
        return all_tools

    catalog = "\n".join(
        f"- {t.name}: {(t.description or '')[:150]}" for t in all_tools
    )
    prompt = (
        f"Catalog ({len(all_tools)} tools):\n{catalog}\n\n"
        f"User message: {user_message}\n\n"
        f"Return a JSON array of up to {max_tools} tool names."
    )

    try:
        resp = await llm.ainvoke(
            [
                SystemMessage(content=TOOL_SELECTOR_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        text = (resp.content or "").strip()
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        names = json.loads(match.group()) if match else []
        available = {t.name: t for t in all_tools}
        selected = [available[n] for n in names if isinstance(n, str) and n in available]
        if selected:
            selected = selected[:max_tools]
            logger.info(
                "Tool planner: %d/%d selected → %s",
                len(selected),
                len(all_tools),
                [t.name for t in selected[:15]],
            )
            return selected
        logger.warning("Tool planner returned no usable names; falling back to keywords")
    except Exception as e:
        logger.warning("Tool planner failed (%s); falling back to keywords", e)

    ranked = _keyword_rank(all_tools, user_message)[:max_tools]
    logger.info(
        "Keyword fallback: %d tools → %s",
        len(ranked),
        [t.name for t in ranked[:15]],
    )
    return ranked


async def run_turn(
    *,
    llm: AzureChatOpenAI,
    tools: List[BaseTool],
    checkpointer: BaseCheckpointSaver,
    thread_id: str,
    user_message: str,
) -> str:
    settings = get_settings()

    selected = await select_relevant_tools(
        llm=llm,
        all_tools=tools,
        user_message=user_message,
        max_tools=settings.agent_max_tools,
    )

    tool_node = ToolNode(selected, handle_tool_errors=_handle_tool_error)
    agent = create_react_agent(
        model=llm,
        tools=tool_node,
        checkpointer=checkpointer,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.agent_max_iterations,
    }
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    final = result["messages"][-1]
    return final.content if hasattr(final, "content") else str(final)
