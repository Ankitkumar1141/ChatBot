import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage

# st.session_state -> dict -> 
CONFIG = {'configurable': {'thread_id': 'user1111'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

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

    ## add the ai assistant message to the message history
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, meta_data in chatbot.stream(
                {"messages":[HumanMessage(content=user_input)]},
                config = {"configurable": {"thread_id":"user1111"}},
                stream_mode = "messages"
            )
        )

    st.session_state["message_history"].append({"role":"assistant","content":"ai_message"})
    