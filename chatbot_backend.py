import asyncio
import os
import sys
import threading
from typing import Annotated
from typing_extensions import Literal, TypedDict

import aiosqlite
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from chatbot_tools import tools as base_tools

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment / .env file")

if sys.platform == "win32":
    # Ensure standard Windows Proactor policy is active
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def _create_event_loop() -> asyncio.AbstractEventLoop:
    """Create a standard event loop safely, even inside non-main threads on Windows."""
    if sys.platform == "win32":
        import signal

        orig_wakeup = getattr(signal, "set_wakeup_fd", None)
        if orig_wakeup is not None:
            try:
                signal.set_wakeup_fd = lambda *args, **kwargs: -1
                return asyncio.new_event_loop()
            finally:
                signal.set_wakeup_fd = orig_wakeup
    return asyncio.new_event_loop()


_ASYNC_LOOP: asyncio.AbstractEventLoop = _create_event_loop()


def _run_loop():
    asyncio.set_event_loop(_ASYNC_LOOP)
    _ASYNC_LOOP.run_forever()


_ASYNC_THREAD = threading.Thread(target=_run_loop, daemon=True)
_ASYNC_THREAD.start()


def _submit(coro):
    """Schedule *coro* on the backend loop and return a Future."""
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    """Block the calling thread until *coro* completes; return its result."""
    return _submit(coro).result()


def submit_async_task(coro):
    """Fire-and-forget: schedule *coro* without waiting.  Returns the Future."""
    return _submit(coro)



llm = ChatMistralAI(
    model="ministral-14b-2512",
    api_key=MISTRAL_API_KEY,
    temperature=0,
)


MCP_SERVERS: dict = {
    "expense": {
        "transport": "streamable_http",
        "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
    },
    }

mcp_client = MultiServerMCPClient(MCP_SERVERS)


async def _fetch_mcp_tools() -> list[BaseTool]:
    """Return MCP tools; fall back to [] on any error."""
    try:
        return await mcp_client.get_tools()
    except Exception as exc:
        print(f"[MCP] Could not load tools: {exc}")
        return []


mcp_tools: list[BaseTool] = run_async(_fetch_mcp_tools())
if mcp_tools:
    print(f"[MCP] Loaded {len(mcp_tools)} tool(s): {[t.name for t in mcp_tools]}")
else:
    print("[MCP] No MCP tools loaded (servers may be offline).")

_seen: set[str] = set()
tools: list[BaseTool] = []
for t in [*base_tools, *mcp_tools]:
    if t.name not in _seen:
        _seen.add(t.name)
        tools.append(t)

llm_with_tools = llm.bind_tools(tools)

class RouterState(BaseModel):
    """Decide how to handle the latest user message."""

    route: Literal["chat", "tools"] = Field(
        description=(
            "'tools' — web search, Wikipedia, arXiv, PubMed, YouTube, URL reading, "
            "current date/time, weather, currency conversion, stock news, calculator, "
            "unit conversion, or any MCP-server capability. "
            "'chat' — greetings, opinions, writing, coding help, concept explanations, "
            "or follow-ups that need no outside data."
        )
    )


router_llm = llm.with_structured_output(RouterState)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: Literal["tools", "chat"]


async def router_node(state: ChatState) -> dict:
    messages = state["messages"]
    latest_text = str(messages[-1].content)

    # URLs always need the fetch_url tool
    if "https://" in latest_text or "http://" in latest_text:
        return {"route": "tools"}

    previous = [
        str(m.content)[:200] for m in messages[:-1] if isinstance(m, HumanMessage)
    ][-3:]
    user_block = (
        "Previous user messages:\n" + ("\n".join(previous) or "(none)")
        + f"\n\nLatest user message:\n{latest_text}"
    )

    try:
        decision = await router_llm.ainvoke([HumanMessage(content=user_block)])
        route = decision.route
    except Exception as exc:
        print(f"[router] failed ({exc}); defaulting to tools path")
        route = "tools"

    print(f"[router] → {route}")
    return {"route": route}


def route_selector(state: ChatState) -> str:
    return "tool_agent" if state["route"] == "tools" else "chat_node"


async def chat_node(state: ChatState) -> dict:
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}


async def tool_agent(state: ChatState) -> dict:
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


async def _init_checkpointer() -> AsyncSqliteSaver:
    conn = await aiosqlite.connect(database="chatbot.db")
    return AsyncSqliteSaver(conn)


checkpointer: AsyncSqliteSaver = run_async(_init_checkpointer())


graph = StateGraph(ChatState)
graph.add_node("router", router_node)
graph.add_node("chat_node", chat_node)
graph.add_node("tool_agent", tool_agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_selector)   # → chat_node | tool_agent
graph.add_edge("chat_node", END)
graph.add_conditional_edges("tool_agent", tools_condition)  # → tools | END
graph.add_edge("tools", "tool_agent")

chatbot = graph.compile(checkpointer=checkpointer)


async def _alist_threads() -> list[str]:
    all_threads: set[str] = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads() -> list[str]:
    return run_async(_alist_threads())
