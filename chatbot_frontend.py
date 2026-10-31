import streamlit as st
from chatbot_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk
import uuid


## utility functions
def generate_thread_id():
    return str(uuid.uuid4())  # string, so it matches the thread ids stored in SQLite


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()  # load old chats from SQLite

add_thread(st.session_state["thread_id"])

### sidebar
st.sidebar.title("Personal Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Past Conversations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                temp_messages.append({"role": "user", "content": msg.content})
            # keep only AI replies with text (skip tool calls and tool results)
            elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                temp_messages.append({"role": "assistant", "content": msg.content})
        st.session_state["message_history"] = temp_messages

# loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    ## add the user message to the message history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    ## observability
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "Next_turn",
    }

    ## stream the assistant reply
    with st.chat_message("assistant"):
        status_box = st.empty()  # shows "Using <tool>..." while a tool runs

        def ai_only_stream():
            for chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                node = metadata.get("langgraph_node")

                if node == "tools":
                    status_box.caption(f"Using {getattr(chunk, 'name', 'tool')}...")
                # only the answer nodes; this hides the router's own LLM call
                elif (
                    node in ("chat_node", "tool_agent")
                    and isinstance(chunk, AIMessageChunk)
                    and chunk.content
                ):
                    status_box.empty()
                    yield chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})