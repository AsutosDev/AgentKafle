# AgentKafle v1.0.0

AgentKafle is an AI detective assistant built in Python. It receives a case,
investigates it, and reasons its way toward a final report.

**Features:**

- **Local accounts** — create/login/logout with secure PBKDF2 password hashing
- **Persistent chat history** — conversations survive restarts, per-user isolation
- **Multiple conversations** — new chat, rename, delete, auto-titling
- **Provider-agnostic LLM layer** — Ollama (local) + Gemini (cloud) supported
- **ProviderRouter** — single active provider, instant switching, shared by Agent & Detective
- **Detective reasoning** — structured JSON analysis of cases & evidence
- **Cases & evidence** — JSON persistence, status tracking (VERIFIED/UNVERIFIED/DISPUTED)
- **Long-term memory** — per-user semantic/episodic memory, separate from chat history
- **Desktop GUI** — Tkinter-based, non-blocking, braille loading spinner

## Requirements

- macOS / Linux / Windows
- Python 3.10+
- For local Ollama: [Ollama](https://ollama.com) with a model pulled, e.g.:

```bash
ollama pull llama3.2:3b
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` (defaults point at local Ollama). Never commit real API keys; `.env` is gitignored.

**Desktop GUI:**
```bash
python gui.py
```

**Terminal REPL:**
```bash
python main.py
```

## Creating an Account

On first launch, the GUI shows a login screen:

1. Enter a username and password
2. Click **Create Account** (or **Login** if you already have one)
3. Your account is stored locally in `data/agentkafle.db` — passwords are PBKDF2-hashed with a unique salt, never stored in plaintext

## Where Local Data Lives

```
AgentKafle/
├── data/
│   ├── agentkafle.db          # SQLite: users, conversations, messages
│   └── memories/
│       └── user_<id>.json     # Per-user long-term memory
├── cases/
│   ├── CASE-001.json          # Case metadata
│   └── CASE-001.evidence.json # Evidence for that case
└── memory/
    └── memories.json          # Global memory (legacy, no longer used by GUI)
```

All data is local to your machine. No cloud sync, no telemetry.

## Configuration

All model settings live in `.env`:

| Variable         | Default                    | Purpose                        |
|------------------|----------------------------|--------------------------------|
| `LLM_BASE_URL`   | `http://localhost:11434/v1`| OpenAI-compatible API address  |
| `LLM_MODEL`      | `llama3.2:3b`              | Model name                     |
| `LLM_API_KEY`    | `ollama`                   | Key (any value for Ollama)     |
| `GEMINI_API_KEY` | *(empty)*                  | Required for Gemini provider   |
| `GEMINI_MODEL`   | `gemini-3.6-flash`         | Gemini model name              |
| `PERSONA_FILE`   | *(empty)*                  | Optional custom system prompt  |

Switch providers in the GUI header pill (OLLAMA · llama3.2:3b → click to change). The active model is displayed and remembered per conversation.

## Architecture Overview

```
gui.py                 →  Tkinter desktop GUI
    ├─ auth.py         →  UserManager, Session (PBKDF2-HMAC-SHA256)
    ├─ history.py      →  ChatStore (conversations, messages, ownership)
    ├─ database.py     →  SQLite connection, schema, transactions
    ├─ agent.py        →  AgentKafle (routing, cases, evidence, memory)
    │    ├─ llm.py     →  ProviderRouter, LLMInterface, OllamaProvider, GeminiProvider
    │    ├─ detective.py →  Detective (investigation, JSON reasoning)
    │    ├─ cases.py   →  CaseManager (JSON persistence)
    │    ├─ evidence.py →  EvidenceManager (JSON persistence)
    │    ├─ memory.py  →  MemoryStore (per-user JSON)
    │    └─ persona.py →  Detective system prompt
    └─ tools.py        →  (reserved)
```

**Key design decisions:**

- **ProviderRouter** owns the single active LLM — Agent and Detective share it, so they can never drift onto different backends
- **Registry pattern** in `llm.py` — providers self-register; adding a new provider is a one-file change
- **Ownership-enforced queries** in `history.py` — every conversation/message lookup includes `user_id`, so users can never access each other's data
- **SQLite transactions** — message insert + conversation timestamp bump are atomic; crash-safe
- **Per-user memory** — `data/memories/user_<id>.json` keeps long-term facts separate from chat history

## Supported Providers

| Provider | Type | Setup |
|----------|------|-------|
| Ollama | Local (OpenAI-compatible) | `ollama serve` + model pulled |
| Gemini | Cloud (Google GenAI) | `GEMINI_API_KEY` in `.env` |

## Detective Features

- **Cases**: create, list, open, solve, close, delete
- **Evidence**: add, list, view, verify, dispute, delete (per case)
- **Evidence status**: VERIFIED (fact), UNVERIFIED (claim), DISPUTED (contested) — model is instructed to respect these
- **Structured reasoning**: Detective calls `chat_json()` for deterministic analysis output

## Long-term Memory

- **Separate from chat history** — facts explicitly stored via `remember` command or extraction
- **Per-user files** — `data/memories/user_<id>.json` (not shared between accounts)
- **Types**: EPISODIC, SEMANTIC, CASE, CONVERSATION
- **Importance**: LOW, MEDIUM, HIGH (affects recall ranking)
- **KeywordRetriever** — dependency-free token overlap + importance + recency (pluggable for embeddings later)

## Development / Testing

```bash
# Run all tests
.venv/bin/python test_provider_switch.py
.venv/bin/python test_chat_history.py

# Syntax check
.venv/bin/python -m py_compile *.py
```

**Test coverage:**

- `test_provider_switch.py` — provider registry, exceptions, switching, detective routing, model selection, lifecycle
- `test_chat_history.py` — auth, conversations, isolation, persistence, agent integration, provider-switch preservation

## Current Limitations (v1.0)

- **No model dropdown** — `available_models()` is wired but not yet exposed in the GUI (planned for v1.1)
- **No password reset** — local app; delete `data/agentkafle.db` and recreate account
- **No multi-process safety** — SQLite uses `RLock`; concurrent processes would need WAL mode
- **Background thread needs mainloop** — headless tests can't fully exercise the LLM worker thread (GUI works fine)
- **No conversation export/import** — planned for v1.1
- **No per-message timestamps in UI** — messages show order, not wall-clock time

## Roadmap (v1.1+)

- Model selector dropdown in provider pill
- Conversation export (JSON/Markdown)
- Password reset via email or local hint
- Embedding-based memory retrieval (swap KeywordRetriever)
- Multi-process SQLite (WAL mode)
- Android client feasibility (see below)

## Android Feasibility Assessment

**Goal:** Evaluate whether AgentKafle's architecture can support a future Android client.

| Component | Reusability | Notes |
|-----------|-------------|-------|
| `llm.py` (ProviderRouter, providers) | ✅ **High** | Pure Python; Ollama client needs network access to host machine; Gemini works directly on Android via GenAI SDK |
| `agent.py` / `detective.py` | ✅ **High** | Pure Python logic; no GUI dependencies |
| `cases.py` / `evidence.py` / `memory.py` | ✅ **High** | JSON persistence; would need file-path adaptation for Android sandbox |
| `auth.py` / `database.py` / `history.py` | ⚠️ **Medium** | SQLite works on Android; password hashing reusable; schema portable. Would need to expose as a local service (Kivy/Chaos) or backend API |
| `gui.py` | ❌ **None** | Tkinter is desktop-only |

**Three architectural paths for Android:**

1. **Native Android client (Kotlin/Jetpack Compose)** + **AgentKafle backend API**
   - Python backend runs on user's computer/server (FastAPI/Flask)
   - Android app talks REST/WebSocket to backend
   - Ollama runs on backend host; Gemini key stored on backend
   - Sync: SQLite → API → Room/SQLDelight on device
   - Best UX, most work

2. **Python/Kivy (or BeeWare/Toga) app**
   - Bundle Python + dependencies as APK
   - Reuse `llm.py`, `agent.py`, `detective.py`, `auth.py`, `database.py`, `history.py` almost directly
   - Ollama: must run on same device (heavy) or connect to remote host
   - Gemini: works directly via GenAI SDK
   - Single codebase, but Python-on-Android has size/performance tradeoffs

3. **Hybrid: local-first sync**
   - Android app has its own SQLite (Room) + memory JSON
   - Periodic sync with desktop AgentKafle via local network (WebDAV, Syncthing, custom protocol)
   - Conflicts resolved by timestamps/UUIDs
   - True offline-first, but sync logic is complex

**Recommendation for v1.x:** Keep the desktop architecture clean (ProviderRouter, registry, ownership-enforced DB, per-user memory files) so path 1 or 2 is straightforward later. No Android-specific code in v1.0.

**Ollama on Android:** Not practical on-device (RAM/CPU). Android client would either:
- Connect to a remote Ollama instance (user's desktop/server) via LAN/VPN/Tailscale, or
- Use Gemini/cloud providers exclusively on mobile

**Gemini on Android:** Works natively via `google-genai` SDK — just needs the API key.

## Version

**v1.0.0** — First stable release with persistent auth, chat history, provider switching, Detective AI, and desktop GUI.

Version constant available at runtime:

```python
from agentkafle import __version__
print(__version__)  # "1.0.0"
```

---

*AgentKafle — local, private, detective AI. No cloud required.*