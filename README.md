<div align="center">

# Multi-Tool Conversational AI Agent

**A routed, tool-augmented LLM agent built on LangGraph and Mistral AI, with MCP integration, persistent memory, streaming responses and LangSmith observability.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)
![Mistral](https://img.shields.io/badge/LLM-Mistral%20AI-FF7000)
![MCP](https://img.shields.io/badge/Protocol-MCP-6A0DAD)
![LangSmith](https://img.shields.io/badge/Tracing-LangSmith-1C3C3C)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

</div>

---

## Overview

This project is a conversational agent that decides *how* to answer each message rather than sending everything through a single tool-calling loop. A lightweight **intent router** classifies every turn as either a direct chat response or a tool-assisted one. The agent supports both built-in tools and externally hosted **MCP (Model Context Protocol) servers**, making it extensible without touching core code. Conversation state is checkpointed per thread, so sessions are resumable, and every run is traced for debugging and cost/latency analysis.

> **Status:** Core agent, routing, tooling, MCP integration, async streaming, persistence, and tracing are implemented. RAG, automated evaluation and deployment are in active development (see [Roadmap](#roadmap)).


### LangSmith trace

![LangSmith trace overview showing router, tool call and final response](assets/langsmith-trace-1.png)

![LangSmith trace detail showing latency and token usage](assets/langsmith-trace-2.png)

*A single turn traced end to end: router, tool agent, tool execution and final answer, with latency and token usage per step.*

## Table of Contents

- [Key Features](#key-features)
- [Architecture](#architecture)
- [Design Decisions](#design-decisions)
- [Tool Catalog](#tool-catalog)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Observability](#observability)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

## Key Features

| Capability | Description |
|---|---|
| **Intent routing** | Structured-output (Pydantic) router sends each turn to a plain chat path or a tool-calling path, avoiding unnecessary tool-enabled calls on simple queries. |
| **MCP integration** | `MultiServerMCPClient` loads tools from any number of remote MCP servers at startup. New servers are added with a one-line config entry — no code changes required. |
| **Deterministic overrides** | Messages containing URLs bypass the LLM router and go straight to the tool path. |
| **Fail-safe behavior** | If router inference fails, the system defaults to the tool agent, which can still answer normally. If an MCP server is offline, it is silently skipped and built-in tools continue working. |
| **Tool-calling loop** | Agent ↔ `ToolNode` cycle runs until the model returns a final answer with no pending tool calls. |
| **Async everywhere** | Graph nodes, checkpointer and MCP calls are all async. A dedicated background event loop bridges Streamlit's sync runtime to the async graph via a thread-safe queue. |
| **Persistent memory** | Per-thread checkpointing via LangGraph `AsyncSqliteSaver`; prior threads can be listed and resumed. |
| **Streaming** | Responses are streamed token-by-token to the UI. A live `st.status()` widget shows which tool is running and collapses when done. |
| **Tracing** | End-to-end LangSmith traces covering router decisions, tool calls, latency and token usage. |

## Architecture

```mermaid
flowchart TD
    U([User message]) --> R[Router node<br/>structured-output LLM]
    R -->|route = chat| C[chat_node<br/>LLM, no tools]
    R -->|route = tools| A[tool_agent<br/>LLM with bound tools]
    A -->|tool calls present| T[ToolNode<br/>built-in + MCP tools]
    T --> A
    A -->|no tool calls| E([END])
    C --> E

    subgraph MCP["MCP Servers (streamable_http / sse)"]
      M1[expense tracker]
      M2[custom server ...]
    end
    T --> MCP

    subgraph Persistence
      S[(AsyncSqliteSaver<br/>chatbot.db)]
    end
    R -.state.-> S
    A -.state.-> S
    C -.state.-> S
```

### Graph state

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]   # reducer appends messages
    route: Literal["tools", "chat"]                        # router decision for the current turn
```

### Request lifecycle

1. **Route.** If the latest message contains `http://` or `https://`, route directly to `tools`. Otherwise the router LLM receives the last three user messages (truncated) plus the latest message and returns a typed `RouterState(route=...)`.
2. **Execute.** `chat` → single async LLM call. `tools` → async agent call with all bound tools (built-in + MCP); `tools_condition` loops through `ToolNode` until no further tool calls are requested.
3. **Persist.** The `AsyncSqliteSaver` checkpointer writes state per step under the conversation's `thread_id`.
4. **Stream.** Token chunks flow from the async loop through a `queue.Queue` to Streamlit's sync `write_stream` generator.
5. **Trace.** The full run is recorded in LangSmith.

## Design Decisions

| Decision | Rationale | Trade-off |
|---|---|---|
| **Separate router instead of always binding tools** | Many turns (greetings, explanations, coding help) need no tools. Routing them to a tool-free path reduces prompt size and avoids spurious tool calls. | One extra LLM call per turn; routing errors can misdirect a query. |
| **Structured output for routing** | A Pydantic schema constrains the decision to a closed set (`chat` / `tools`), making the router easy to validate and test. | Depends on the model's structured-output reliability. |
| **Router fails open to `tools`** | The tool agent is a superset: it can answer directly or call tools, so a router failure degrades gracefully instead of erroring. | Failures cost more tokens than the chat path would have. |
| **Deterministic URL rule** | Links almost always require fetching content; a rule is cheaper and more predictable than asking an LLM. | Pasted URLs that need no tool still take the tool path. |
| **MCP via `streamable_http` / `sse` only** | Stdio transport requires spawning a child process, which is unreliable from inside a venv on Windows. HTTP transports work from any thread on any OS. | Requires MCP servers to be network-accessible (local or remote). |
| **Dedicated async event loop** | Streamlit is synchronous; LangGraph MCP is async. A single background thread runs `asyncio` forever; `submit_async_task` bridges the two without blocking the UI thread. | Slightly more complex than a purely sync setup. |
| **MCP tools deduplicated by name** | Built-in and MCP tools are merged into one list at startup, with name-based deduplication, so there are no conflicts if a tool appears in both. | First-registered tool wins on name collision. |
| **MCP failures are silent at startup** | If a server is unreachable, `_fetch_mcp_tools()` logs a warning and returns `[]`. The chatbot still starts with its full built-in toolset. | A silently missing MCP tool may confuse users who expect it. |
| **`temperature=0`** | Stable routing and tool-argument generation. | Less varied phrasing in free-form answers. |
| **`AsyncSqliteSaver`** | Zero-infrastructure async persistence compatible with the async graph. Sufficient for local and single-user use. | Not designed for concurrent multi-user production; Postgres is the natural upgrade path. |

## Tool Catalog

### Built-in tools (`chatbot_tools.py`)

| Tool | Purpose |
|---|---|
| `web_search` | Current events, breaking news, recent prices — anything that changes frequently |
| `wikipedia` | Encyclopedic look-ups: people, places, history, concepts, organisations |
| `arxiv_search` | Research papers in physics, CS, mathematics, AI / ML |
| `pubmed_search` | Medical and life-science literature, clinical studies, drug information |
| `youtube_search` | Finding YouTube videos on any topic |
| `fetch_url` | Read and summarise a web page from a user-pasted `http(s)://` link |
| `get_current_datetime` | Current date, time, and day of the week (any IANA timezone) |
| `get_weather` | Current conditions and 3-day forecast for any city |
| `convert_currency` | Live exchange-rate conversion between any two currencies |
| `stock_news` | Recent news articles for a stock ticker (e.g. `AAPL`, `TCS.NS`) |
| `calculator` | Exact arithmetic, algebra, percentages, powers, roots |
| `unit_converter` | Length, mass, temperature, volume, speed, area and more |

### MCP tools (pluggable)

Configured in `MCP_SERVERS` inside `chatbot_backend.py`. Add a new entry and restart — no other changes needed.

| Server (example) | Transport | Capability |
|---|---|---|
| `expense` (FastMCP) | `streamable_http` | Expense tracking and querying |
| *(your server)* | `streamable_http` / `sse` | Any custom capability |

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (`StateGraph`, `ToolNode`, `AsyncSqliteSaver`) |
| LLM | Mistral AI via `langchain-mistralai` (`ministral-14b-2512`) |
| MCP | `langchain-mcp-adapters`, `mcp` |
| Schema validation | Pydantic v2 (`typing_extensions.Literal` for model fields) |
| UI | Streamlit (async-bridged via `queue.Queue` + background event loop) |
| Persistence | SQLite (`aiosqlite`, async) |
| Observability | LangSmith |
| Config | `python-dotenv` |

## Project Structure

```
.
├── chatbot_backend.py     # Graph definition: router, chat node, tool agent, checkpointer
├── chatbot_tools.py       # Tool implementations bound to the agent
├── chatbot_frontend.py    # Streamlit UI with streaming and thread management
├── assets/
│   ├── chat-ui.png          # Chat interface screenshot
│   └── langsmith-trace.png  # Example LangSmith trace
├── basic_chatbot.py       # Simpler baseline version
├── requirements.txt
└── .env.example           # Template for required environment variables
```

## Getting Started

**Prerequisites:** Python 3.10+, a [Mistral API key](https://console.mistral.ai/), and optionally a [LangSmith](https://smith.langchain.com/) account.

```bash
# 1. Clone
git clone https://github.com/Ankitkumar1141/ChatBot.git
cd ChatBot

# 2. Create and activate a virtual environment
python -m venv chatbot_venv
# Windows:
chatbot_venv\Scripts\activate
# macOS / Linux:
source chatbot_venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env              # then fill in your keys

# 5. Run
streamlit run chatbot_frontend.py
```

### Adding an MCP server

Open `chatbot_backend.py`, find `MCP_SERVERS`, and add an entry:

```python
MCP_SERVERS = {
    "my_server": {
        "transport": "streamable_http",   # or "sse"
        "url": "https://your-server.example.com/mcp",
    },
}
```

Restart the app — the new server's tools are loaded automatically at startup.

## Configuration

| Variable | Required | Description |
|---|---|---|
| `MISTRAL_API_KEY` | Yes | Mistral AI API key. The backend raises an error at startup if missing. |
| `LANGSMITH_TRACING` | No | Set to `true` to enable tracing. |
| `LANGSMITH_API_KEY` | No | LangSmith API key. |
| `LANGSMITH_PROJECT` | No | Project name under which traces are grouped. |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint. Defaults to `https://api.smith.langchain.com`; use `https://eu.api.smith.langchain.com` for EU workspaces. |

## Observability

With tracing enabled, each conversation turn produces a LangSmith trace showing:

- the router's decision and the input it saw
- every LLM call, tool invocation (including MCP) and tool result in order
- latency and token usage per step

This makes it straightforward to diagnose misrouted queries, malformed tool arguments, slow tool calls, and MCP server issues.

See the [example trace](#langsmith-trace) above.

## Known Limitations

- **Router quality is unmeasured.** Routing accuracy has not yet been benchmarked on a labeled set (planned below).
- **Limited routing context.** The router sees only the last three user messages, truncated, so long-range follow-ups may be misclassified.
- **MCP tools loaded once at startup.** If a server comes online after the app starts, a restart is needed to pick up its tools.
- **MCP transport: HTTP only.** Stdio transport is intentionally excluded to ensure Windows compatibility. MCP servers must be network-accessible.
- **Single-node persistence.** `AsyncSqliteSaver` is appropriate for local / single-user use, not concurrent multi-user deployment.
- **No automated tests or CI yet.**

## Roadmap

**Completed**
- [x] LangGraph agent with LLM-based intent router
- [x] Multi-tool calling loop (12 built-in tools)
- [x] Persistent multi-thread memory (AsyncSqliteSaver / SQLite)
- [x] Async graph with token-level streaming
- [x] LangSmith tracing
- [x] **MCP integration** — `MultiServerMCPClient` with `streamable_http` / `sse` transport; graceful fallback if servers are offline; tool deduplication; Windows-compatible async event-loop bridge

**In progress / planned**
- [ ] **RAG pipeline:** document ingestion, chunking, embeddings, vector store, retrieval exposed as a graph tool, with source citations
- [ ] **Evaluation suite:** labeled set for router accuracy, tool-call success rate and RAG answer quality (e.g., RAGAS or LLM-as-judge)
- [ ] **Human-in-the-loop approval** for side-effect tools using LangGraph interrupts
- [ ] **Tests and CI:** unit tests for tools and routing logic, GitHub Actions workflow
- [ ] **Containerization and deployment:** Dockerfile and a hosted demo
- [ ] **Persistence upgrade:** Postgres checkpointer for multi-user scenarios

## Author

**Ankit Kumar** · [GitHub](https://github.com/Ankitkumar1141)
