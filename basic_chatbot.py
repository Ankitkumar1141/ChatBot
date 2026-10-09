## import libraries
from langgraph.graph import StateGraph, START,END
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os
from langgraph.checkpoint.memory import MemorySaver

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

## create checkpointer object
checkpointer = MemorySaver()

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

chatbot = graph.compile(checkpointer=checkpointer)

thread_id  ="user_1111"
while True:
    user_input = input("Enter your Query: ")
    print("User Input: ", user_input)
    if user_input.lower().strip() in ["exit","quit","bye"]:
        print("Exiting the chat. Goodbye!")
        break
    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": thread_id}})
    print("Response: ", response["messages"][-1].content)