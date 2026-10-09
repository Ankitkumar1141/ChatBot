from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

load_dotenv()

## read API
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if MISTRAL_API_KEY:
    print("MISTRAL_API_KEY found")
else:
    raise ValueError("MISTRAL_API_KEY not found")

llm = ChatMistralAI(
    model="ministral-14b-2512", 
    api_key=MISTRAL_API_KEY, 
    temperature=0)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

# Checkpointer
checkpointer = InMemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)