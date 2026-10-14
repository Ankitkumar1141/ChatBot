import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import uuid

## utility functions
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []

def load_conversation(thread_id):
    state = chatbot.get_state(config = {"configurable": {"thread_id": thread_id}})
    return state.values.get("messages",[])


if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])

### sidebar
st.sidebar.title("Personal Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Past Conservations")
for thread_id in st.session_state["chat_threads"][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg,HumanMessage):
                role = "user"
            else:
                role = "assistant"
            temp_messages.append({"role": "user","content":msg.content})
        st.session_state["message_history"] = temp_messages

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    ## add the user message to the message history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {"configurable": {"thread_id":st.session_state["thread_id"]}}

    ## add the ai assistant message to the message history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})