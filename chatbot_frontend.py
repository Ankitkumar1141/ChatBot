import queue
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from chatbot_backend import chatbot, retrieve_all_threads, submit_async_task


st.set_page_config(page_title="Multi-Tool AI Agent", page_icon="🧠", layout="centered")


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def add_thread(thread_id: str) -> None:
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat() -> None:
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def load_conversation(thread_id: str) -> list[dict]:
    """Load a past thread from the checkpointer and return display-safe dicts."""
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    raw: list = state.values.get("messages", [])
    result = []
    for msg in raw:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            result.append({"role": "assistant", "content": str(msg.content)})
    return result


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])


st.sidebar.title("🧠 Multi-Tool AI Agent")
st.sidebar.caption("Mistral AI · LangGraph · MCP · LangSmith")

if st.sidebar.button("➕  New Chat", use_container_width=True):
    reset_chat()

st.sidebar.divider()
st.sidebar.subheader("Past Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    label = f"💬 …{str(thread_id)[-8:]}"   # show last 8 chars for readability
    if st.sidebar.button(label, key=f"thread_{thread_id}", use_container_width=True):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)


st.title("Personal Chatbot")

# Render existing history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Type your message…")

if user_input:
    
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata":     {"thread_id": st.session_state["thread_id"]},
        "run_name":     "chat_turn",
    }

    # ── Assistant streaming block ───────────────────────────────────────────
    with st.chat_message("assistant"):
        # Mutable holder so the inner generator can reference the status box
        status_holder: dict = {"box": None}

        def ai_only_stream():
            """
            Bridge async astream → sync generator for st.write_stream.

            Events are passed through a thread-safe Queue from the async
            loop thread to this Streamlit (sync) generator.
            """
            event_queue: queue.Queue = queue.Queue()

            async def run_stream():
                try:
                    async for chunk, metadata in chatbot.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=CONFIG,
                        stream_mode="messages",
                    ):
                        event_queue.put(("chunk", chunk, metadata))
                except Exception as exc:
                    event_queue.put(("error", exc, None))
                finally:
                    event_queue.put(("done", None, None))

            submit_async_task(run_stream())

            while True:
                kind, payload, metadata = event_queue.get()

                if kind == "done":
                    break

                if kind == "error":
                    raise payload

                chunk, meta = payload, metadata
                node = meta.get("langgraph_node", "") if meta else ""

                # ── Tool running: update/create status widget ───────────────
                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Running `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Running `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # ── Stream only real assistant answer tokens ────────────────
                if (
                    node in ("chat_node", "tool_agent")
                    and isinstance(chunk, AIMessageChunk)
                    and chunk.content
                ):
                    yield chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalise the status widget after streaming is done
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # ── Persist assistant reply ─────────────────────────────────────────────
    if ai_message:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )
