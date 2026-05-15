# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About This Repo

LangChain Academy's **Introduction to LangChain** course — a Python learning repo built around Jupyter notebooks. There are no tests or CI pipelines; the primary artifact is runnable notebooks, not a deployed application.

## Environment Setup

Python >=3.12, <3.14 is required. `uv` is the recommended package manager.

```bash
cp example.env .env          # then add API keys
uv sync                      # install all dependencies
uv run python env_utils.py   # verify setup
```

Required API keys (in `.env`): `OPENAI_API_KEY`, `TAVILY_API_KEY`.  
Optional: `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `LANGSMITH_API_KEY`.

## Running Things

**Jupyter notebooks** (main development interface):
```bash
uv run jupyter lab
```

**LangGraph Studio** (for modules 1 and 3 — must run from the module directory):
```bash
cd notebooks/module-1   # or module-3
uv run langgraph dev
```

Each module with Studio support has a `langgraph.json` pointing to its agent `.py` file.

**Agent Chat UI** (bonus, Module 3 only — requires LangGraph dev server running):
```bash
cd notebooks/module-3/agent-chat-ui
pnpm install
pnpm dev
```

## Architecture

The repo is organized into 3 modules under `notebooks/`, each a progression:

- **Module 1** (`notebooks/module-1/`): Foundational models, tools, memory, multimodal. Culminates in `1.5_personal_chef.py` — a LangGraph agent deployable via Studio.
- **Module 2** (`notebooks/module-2/`): MCP (uses `uvx` to run MCP servers), runtime context/state, multi-agent systems, RAG bonus, SQL bonus.
- **Module 3** (`notebooks/module-3/`): Middleware, long-conversation management, HITL (human-in-the-loop), dynamic agents. Culminates in `3.5_email_agent.py`. Includes `agent-chat-ui/` — a Next.js 15 + React 19 frontend (pnpm, Tailwind v4) that connects to LangGraph's API server.

The standalone `.py` files (`1.5_personal_chef.py`, `3.5_email_agent.py`) are the LangGraph graph entry points referenced by `langgraph.json`. Notebooks are self-contained lessons; the `.py` files expose the `agent` graph object for Studio and the Chat UI.

`env_utils.py` at the root validates Python version, virtual environment activation, installed packages, and `.env` key configuration.
