from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os
import sqlite3

from chatbot_tools import tools

load_dotenv()

## read API
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if MISTRAL_API_KEY:
    print("MISTRAL_API_KEY found")
else:
    raise ValueError("MISTRAL_API_KEY not found")

## llm call
llm = ChatMistralAI(
    model="ministral-14b-2512",
    api_key=MISTRAL_API_KEY,
    temperature=0,
)

## tool binding
llm_with_tools = llm.bind_tools(tools=tools)


## Schema for the router's decision
class RouterState(BaseModel):
    """Decide how to handle the latest user message."""

    route: Literal["chat", "tools"] = Field(
        description=(
            "'tools' if the latest message needs: current/recent info (news, prices, 'latest', "
            "'today'), weather, currency rates, stock news, exact maths or unit conversion, "
            "date/time, a link or YouTube videos, research papers, a Wikipedia lookup, "
            "translation, to-dos/reminders, sending email, or a CSV/Excel file. "
            "'chat' for greetings, small talk, explanations of well-known concepts, writing, "
            "coding help, opinions, or follow-ups needing no outside data. "
            "Use previous messages only to understand follow-ups like 'and in Mumbai?'."
        )
    )


router_llm = llm.with_structured_output(RouterState)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: Literal["tools", "chat"]


def router_node(state: ChatState):
    messages = state["messages"]
    latest_text = str(messages[-1].content)

    # a link always needs the URL tool
    if "https://" in latest_text or "http://" in latest_text:
        return {"route": "tools"}

    previous = [str(m.content)[:200] for m in messages[:-1] if isinstance(m, HumanMessage)][-3:]
    user_block = (
        "Previous user messages:\n" + ("\n".join(previous) or "(none)")
        + f"\n\nLatest user message:\n{latest_text}"
    )

    try:
        decision = router_llm.invoke([HumanMessage(content=user_block)])
        route = decision.route
    except Exception as e:
        print(f"Router failed ({e}); defaulting to tools path")
        route = "tools"  # safe default: the tool agent can still answer normally

    print(f"[router] -> {route}")
    return {"route": route}


def route_selector(state: ChatState):
    return "tool_agent" if state["route"] == "tools" else "chat_node"


def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def tool_agent(state: ChatState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


connection = sqlite3.connect(database="chatbot.db", check_same_thread=False)

# Checkpointer
checkpointer = SqliteSaver(conn=connection)

graph = StateGraph(ChatState)
graph.add_node("router", router_node)
graph.add_node("chat_node", chat_node)
graph.add_node("tool_agent", tool_agent)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "router")
graph.add_conditional_edges("router", route_selector)       # -> chat_node or tool_agent
graph.add_edge("chat_node", END)
graph.add_conditional_edges("tool_agent", tools_condition)  # -> tools or END
graph.add_edge("tools", "tool_agent")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)