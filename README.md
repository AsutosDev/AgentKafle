# AgentKafle

AgentKafle is an AI detective assistant built in Python. It receives a case,
investigates it, and reasons its way toward a final report.

This project is built in 7 steps:

1. Foundation — LLM integration, project structure
2. Detective identity & behavior — persona, reasoning categories
3. Case management — cases, statuses, active case, JSON persistence (current)
4. Evidence tracking — items, connections, contradictions
5. Memory — long-term store and retrieval for case knowledge
6. Reasoning — hypotheses, suspect analysis, final report
7. Tools & GUI

AgentKafle connects to a pre-trained language model, behaves like an
analytical AI detective, and can manage real investigation cases that persist
between sessions. Evidence, reasoning, and the GUI arrive in later steps.

## Requirements

- macOS / Linux / Windows
- Python 3.10+
- [Ollama](https://ollama.com) with an LLM pulled, e.g.:

```bash
ollama pull llama3.2:3b
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` (defaults already point at local Ollama). Never commit real
API keys; `.env` is gitignored.

## Usage

```bash
python main.py
```

## Configuration

All model settings live in `.env`:

| Variable         | Default                    | Purpose                        |
|------------------|----------------------------|--------------------------------|
| `LLM_BASE_URL`   | `http://localhost:11434/v1`| OpenAI-compatible API address  |
| `LLM_MODEL`      | `llama3.2:3b`              | Model name                     |
| `LLM_API_KEY`    | `ollama`                   | Key (any value for Ollama)     |
| `PERSONA_FILE`   | *(empty)*                  | Optional custom system prompt  |

Switch to a cloud provider by changing the first three values — no code
changes.

## Persona

The detective system prompt lives in `persona.py` (the `Persona` class).
It defines AgentKafle's role and the reasoning categories the investigator
uses to separate what is known from what is claimed:

- **FACT** — explicitly provided or verified information
- **CLAIM** — something a person says happened, not yet verified
- **INFERENCE** — logically suggested by evidence
- **HYPOTHESIS** — a possible explanation, not established
- **UNKNOWN** — information currently unavailable

AgentKafle labels analyses with these categories and never presents a
hypothesis as a confirmed fact.

To use a different persona, set `PERSONA_FILE` in `.env` to a text file
containing your own system prompt — no code changes needed.

## Case management

Cases are managed by `cases.py`:

- `Case` — a dataclass with `id`, `title`, `description`, `status`,
  `created_at`, and `updated_at`.
- `CaseStatus` — `OPEN`, `SOLVED`, `CLOSED` (plain strings, easy to extend).
- `CaseManager` — creates, loads, lists, updates, and deletes cases.

Each case is stored as its own file under `cases/`:

```
cases/
    CASE-001.json
    CASE-002.json
```

This keeps cases independent and makes it easy to attach evidence files to a
case in Step 4. The storage is plain JSON so it can be swapped for a database
later without changing the rest of the app.

### Case commands

Case commands are handled directly in application code — the LLM never writes
case data. In the REPL:

```
new case Missing Laptop | A laptop disappeared from Room 204.
list cases
open case CASE-001
current case
solve case
close case
delete case CASE-001
```

### Active case

AgentKafle tracks one *active case* at a time. The active case is attached as
text to the system prompt when reasoning, so the model always knows which
investigation it is working on. The model only receives a copy; CaseManager
remains the source of truth.

## Architecture

- `main.py` — command-line entry point
- `agent.py` — the AgentKafle class: case operations, command routing, and
  LLM reasoning
- `llm.py` — LLMInterface: the only module that talks to the model
- `persona.py` — the detective system prompt and Persona loader
- `cases.py` — Case, CaseStatus, and CaseManager (JSON persistence)
- `memory.py`, `evidence.py`, `detective.py`, `tools.py`, `gui.py` —
  placeholder modules for later steps