# AgentKafle

AgentKafle is an AI detective assistant built in Python. It receives a case,
investigates it, and reasons its way toward a final report.

This project is built in 7 steps:

1. Foundation — LLM integration, project structure
2. Detective identity & behavior — persona, reasoning categories
3. Case management — cases, statuses, active case, JSON persistence
4. Evidence tracking — items, types, statuses, JSON persistence (current)
5. Memory — long-term store and retrieval for case knowledge
6. Reasoning — hypotheses, suspect analysis, final report
7. Tools & GUI

AgentKafle connects to a pre-trained language model, behaves like an
analytical AI detective, and manages cases and evidence that persist between
sessions. Reasoning, memory, and the GUI arrive in later steps.

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

## Evidence management

Evidence is managed by `evidence.py` and attached to the **active case**:

- `EvidenceType` — `PHYSICAL`, `DIGITAL`, `TESTIMONY`, `DOCUMENT`, `OTHER`.
- `EvidenceStatus` — `UNVERIFIED`, `VERIFIED`, `DISPUTED`.
- `Evidence` — a dataclass with `id`, `case_id`, `title`, `description`,
  `evidence_type`, `status`, `source`, `created_at`, and `updated_at`.
- `EvidenceManager` — adds, loads, updates, and deletes evidence.

Evidence IDs (`EVD-001`, `EVD-002`, …) are unique across all cases.

Each case's evidence is stored in its own file:

```
cases/
    CASE-001.json
    CASE-001.evidence.json
    CASE-002.json
    CASE-002.evidence.json
```

Splitting evidence per case keeps cases independent: deleting one case never
touches another's evidence, and the plain JSON can later be swapped for a
database.

### Evidence commands

Evidence commands operate on the active case and are handled in application
code — the LLM never writes evidence data:

```
add evidence Broken window | Glass fragments found inside Room 204
add evidence Camera footage | Someone entering Room 204 at 8:17 PM | type=DIGITAL | source=Security camera
list evidence
view evidence EVD-001
verify evidence EVD-001
dispute evidence EVD-001
delete evidence EVD-001
```

`type=` and `source=` are optional. Without them, evidence is created as
`OTHER` / `UNVERIFIED`.

### How evidence status affects reasoning

When AgentKafle reasons, the active case's evidence is grouped by status and
sent to the model with an explicit rule:

- **VERIFIED** — may be treated as established.
- **UNVERIFIED** — only a claim; must not be treated as fact.
- **DISPUTED** — contested; must not be relied upon.

The model is told never to present UNVERIFIED or DISPUTED evidence as
confirmed. The status is stored by `EvidenceManager`; the LLM only reads it.

## Architecture

- `main.py` — command-line entry point
- `agent.py` — the AgentKafle class: case/evidence operations, command
  routing, and LLM reasoning
- `llm.py` — LLMInterface: the only module that talks to the model
- `persona.py` — the detective system prompt and Persona loader
- `cases.py` — Case, CaseStatus, and CaseManager (JSON persistence)
- `evidence.py` — Evidence, EvidenceType, EvidenceStatus, and
  EvidenceManager (JSON persistence)
- `memory.py`, `detective.py`, `tools.py`, `gui.py` — placeholder modules for
  later steps