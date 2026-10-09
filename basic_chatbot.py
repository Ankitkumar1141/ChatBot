## import libraries
from langgraph.graph import StateGraph, START,END
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

## load environment variables
load_dotenv()

## read API
try:
    MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
    print("MISTRAL_API_KEY found")
except:
    print("MISTRAL_API_KEY not found")

### create llm instance
llm = ChatMistralAI(
    model="ministral-14b-2512", 
    api_key=MISTRAL_API_KEY, 
    temperature=0
)

## add message operator
from langgraph.graph.message import add_messages

## create state
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]


## chat_node function
def chat_node(state: ChatState):
    message = state["messages"]
    response = llm.invoke(message)
    return {"messages":[response]}

## object of StateGraph
graph = StateGraph(ChatState)

## add nodes
graph.add_node("chat_node", chat_node)

## add_edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile()


initial_state = {
    'messages': [HumanMessage(content='What is Photosynthesis?')]
}

final_response = chatbot.invoke(initial_state)["messages"][-1].content
print(final_response)