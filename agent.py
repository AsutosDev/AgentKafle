"""The main AgentKafle class.

AgentKafle is the top-level object that the rest of the app talks to. It sits
between the user and four collaborators:

  User → AgentKafle → CaseManager      (structured case data, no LLM)
                    → EvidenceManager  (structured evidence data, no LLM)
                    → MemoryStore      (persistent long-term memory, no LLM)
                    → Persona + LLM    (language and reasoning)

Key rule: the LLM never reads or writes case, evidence, or memory files.
CaseManager, EvidenceManager, and MemoryStore are the single sources of
truth for structured data. The LLM only receives text context when reasoning
is required, and it can never write memory directly.

The detective persona (system prompt) lives in persona.py, not here.
"""

import re

from llm import LLMInterface
from persona import Persona
from cases import CaseManager, CaseStatus
from evidence import EvidenceManager, EvidenceType, EvidenceStatus
from memory import MemoryStore, MemoryType, MemoryImportance


# Maps user-friendly type names to EvidenceType values.
EVIDENCE_TYPES = {
    "physical": EvidenceType.PHYSICAL,
    "digital": EvidenceType.DIGITAL,
    "testimony": EvidenceType.TESTIMONY,
    "document": EvidenceType.DOCUMENT,
    "other": EvidenceType.OTHER,
}


class AgentKafle:
    def __init__(self, name="AgentKafle"):
        self.name = name
        self.llm = LLMInterface()
        self.persona = Persona(agent_name=name)
        self.case_manager = CaseManager()
        self.evidence_manager = EvidenceManager()
        self.memory = MemoryStore()

    # ── Case operations (application logic, never sent to the LLM) ────────

    def create_case(self, title, description=""):
        """Create a case and make it the active one."""
        case, message = self.case_manager.create(title, description)
        if not case:
            return message

        self.case_manager.set_active(case.id)
        return f"{message}\nActive case: {case.id}"

    def list_cases(self):
        """Return a formatted list of all cases."""
        cases = self.case_manager.list_all()

        if not cases:
            return "No cases yet. Create one with: new case <title>"

        lines = []
        for case in cases:
            marker = "  [active]" if case.id == self.case_manager.active_case_id else ""
            lines.append(f"{case.id} — {case.title} — {case.status}{marker}")

        return "\n".join(lines)

    def select_case(self, case_id):
        """Set the active case by ID."""
        case, message = self.case_manager.set_active(case_id)
        return message

    def view_case(self, case_id=None):
        """Show details for a case (the active one by default)."""
        if case_id:
            case = self.case_manager.get(case_id)
            if not case:
                return f"Error: Case {case_id} not found."
        else:
            case = self.case_manager.get_active()
            if not case:
                return "No active case. Create or select a case first."

        return self._format_case(case)

    def solve_case(self, case_id=None):
        """Mark a case (active by default) as SOLVED."""
        return self._change_status(case_id, CaseStatus.SOLVED)

    def close_case(self, case_id=None):
        """Mark a case (active by default) as CLOSED."""
        return self._change_status(case_id, CaseStatus.CLOSED)

    def delete_case(self, case_id):
        """Delete a case by ID."""
        _, message = self.case_manager.delete(case_id)
        return message

    def _change_status(self, case_id, new_status):
        # Fall back to the active case when no ID is given.
        if case_id is None:
            case_id = self.case_manager.active_case_id

        if case_id is None:
            return "No active case. Create or select a case first."

        _, message = self.case_manager.update_status(case_id, new_status)
        return message

    def _format_case(self, case):
        return (
            f"Case: {case.id}\n"
            f"Title: {case.title}\n"
            f"Status: {case.status}\n"
            f"Description: {case.description or '(no description)'}\n"
            f"Created: {case.created_at}\n"
            f"Updated: {case.updated_at}"
        )

    # ── Evidence operations (active case only, never sent to the LLM) ─────

    def add_evidence(self, title, description="", evidence_type=EvidenceType.OTHER, source=""):
        """Add evidence to the active case."""
        case, error = self._require_active_case()
        if error:
            return error

        _, message = self.evidence_manager.add(
            case.id, title, description, evidence_type, source
        )
        return message

    def list_evidence(self):
        """List the active case's evidence."""
        case, error = self._require_active_case()
        if error:
            return error

        items = self.evidence_manager.list_for_case(case.id)
        if not items:
            return f"No evidence yet for {case.id}."

        lines = []
        for item in items:
            lines.append(
                f"{item.id} — {item.title} "
                f"[{item.evidence_type.value} / {item.status.value}]"
            )
        return "\n".join(lines)

    def view_evidence(self, evidence_id):
        """Show one evidence item from the active case."""
        case, error = self._require_active_case()
        if error:
            return error

        item = self.evidence_manager.get(case.id, evidence_id)
        if not item:
            return f"Error: Evidence {evidence_id} not found."

        return self._format_evidence(item)

    def verify_evidence(self, evidence_id):
        """Mark an item VERIFIED on the active case."""
        return self._change_evidence_status(evidence_id, EvidenceStatus.VERIFIED)

    def dispute_evidence(self, evidence_id):
        """Mark an item DISPUTED on the active case."""
        return self._change_evidence_status(evidence_id, EvidenceStatus.DISPUTED)

    def delete_evidence(self, evidence_id):
        """Delete an item from the active case."""
        case, error = self._require_active_case()
        if error:
            return error

        _, message = self.evidence_manager.delete(case.id, evidence_id)
        return message

    def _change_evidence_status(self, evidence_id, new_status):
        case, error = self._require_active_case()
        if error:
            return error

        _, message = self.evidence_manager.update_status(case.id, evidence_id, new_status)
        return message

    def _require_active_case(self):
        """Return (case, None) or (None, error_message)."""
        case = self.case_manager.get_active()
        if not case:
            return None, "No active case. Create or select a case first."
        return case, None

    def _format_evidence(self, item):
        return (
            f"Evidence: {item.id}\n"
            f"Title: {item.title}\n"
            f"Type: {item.evidence_type.value}\n"
            f"Status: {item.status.value}\n"
            f"Description: {item.description or '(no description)'}\n"
            f"Source: {item.source or '(unknown)'}\n"
            f"Case: {item.case_id}\n"
            f"Created: {item.created_at}\n"
            f"Updated: {item.updated_at}"
        )

    # ── Memory operations ────────────────────────────────────────────────
    # Long-term memory lives in MemoryStore; the agent only forwards calls.
    # The LLM never writes memory directly.

    def remember(self, content, memory_type=MemoryType.SEMANTIC,
                 source="", importance=MemoryImportance.MEDIUM,
                 case_id="", confidence="", tags=None):
        """Persist a new memory and return the confirmation message."""
        _, message = self.memory.add(
            content, memory_type=memory_type, source=source,
            importance=importance, case_id=case_id, confidence=confidence,
            tags=tags,
        )
        return message

    def recall(self, query, top_k=5):
        """Return the best relevant memories as a formatted text block."""
        block, message = self.memory.recall(query, top_k=top_k)
        return block if block else message

    def list_memories(self):
        """Return a formatted list of every stored memory."""
        memories = self.memory.list_all()
        if not memories:
            return "No memories stored yet."
        lines = []
        for m in memories:
            flag = "" if m.active else " [forgotten]"
            lines.append(f"{m.id} [{m.type.value}/{m.importance.value}] {m.content}{flag}")
        return "\n".join(lines)

    # ── Command router ────────────────────────────────────────────────────
    # Detects case and evidence commands in plain language. Returns
    # (handled, response). If handled is False, the caller sends the text to
    # the LLM.

    def handle_command(self, text):
        lower = text.lower().strip()

        # ── Evidence commands (checked first so "add evidence" etc. match) ─
        if lower.startswith("add evidence"):
            return True, self._cmd_add_evidence(text)

        if lower in ("evidence", "list evidence", "show evidence", "all evidence"):
            return True, self.list_evidence()

        if lower.startswith("view evidence") or lower.startswith("show evidence"):
            return True, self._cmd_view_evidence(text)

        if lower.startswith("verify evidence"):
            return True, self._cmd_verify_evidence(text)

        if lower.startswith("dispute evidence"):
            return True, self._cmd_dispute_evidence(text)

        if lower.startswith("delete evidence") or lower.startswith("remove evidence"):
            return True, self._cmd_delete_evidence(text)

        # ── Case commands ─────────────────────────────────────────────────
        if lower.startswith(("new case", "create case")):
            return True, self._cmd_create_case(text)

        if lower in ("cases", "list cases", "show cases", "all cases"):
            return True, self.list_cases()

        if lower.startswith(("open case", "select case", "load case", "switch case")):
            return True, self._cmd_select_case(text)

        if lower in (
            "current case",
            "active case",
            "case details",
            "view case",
            "show case",
            "show me the current case",
            "show current case",
        ):
            return True, self.view_case()

        if lower.startswith(("view case ", "show case ")):
            return True, self.view_case(self._extract_case_id(text))

        if lower.startswith(("solve case", "close case", "mark solved", "mark solved case")):
            return True, self._cmd_change_status(text)

        if lower.startswith("delete case"):
            case_id = self._extract_case_id(text)
            if not case_id:
                return True, "Please include a case ID. Example: delete case CASE-001"
            return True, self.delete_case(case_id)

        return False, None

    # ── command helpers ───────────────────────────────────────────────────

    def _cmd_create_case(self, text):
        # Strip the command phrase, then optional punctuation.
        rest = re.sub(r"(?i)^\s*(new|create)\s+case\b", "", text).strip()
        rest = rest.lstrip(":- ").strip()

        if not rest:
            return "Please provide a title. Example: new case Missing Laptop"

        if "|" in rest:
            title, description = rest.split("|", 1)
        else:
            title, description = rest, ""

        return self.create_case(title.strip(), description.strip())

    def _cmd_select_case(self, text):
        case_id = self._extract_case_id(text)
        if not case_id:
            return "Please include a case ID. Example: open case CASE-001"
        return self.select_case(case_id)

    def _cmd_change_status(self, text):
        case_id = self._extract_case_id(text)
        if "solve" in text.lower() or "solved" in text.lower():
            return self.solve_case(case_id)
        return self.close_case(case_id)

    def _cmd_add_evidence(self, text):
        # Syntax:
        #   add evidence <title> | <description> | type=PHYSICAL | source=Guard
        # Only the title is required. type= and source= are optional.
        rest = re.sub(r"(?i)^\s*add\s+evidence\b", "", text).strip()
        rest = rest.lstrip(":- ").strip()

        if not rest:
            return (
                "Please provide a title. Example:\n"
                "add evidence Broken window | Glass fragments found inside Room 204"
            )

        title = None
        description = ""
        evidence_type = EvidenceType.OTHER
        source = ""

        for segment in rest.split("|"):
            segment = segment.strip()
            if not segment:
                continue

            key, sep, value = segment.partition("=")
            key_lower = key.strip().lower()

            if sep and key_lower in ("type", "source"):
                value = value.strip()
                if key_lower == "type":
                    evidence_type = EVIDENCE_TYPES.get(value.lower())
                    if evidence_type is None:
                        valid = ", ".join(t.upper() for t in EVIDENCE_TYPES)
                        return f"Error: Unknown type '{value}'. Valid types: {valid}."
                else:
                    source = value
            elif title is None:
                title = segment
            elif not description:
                description = segment
            else:
                description += " " + segment

        if not title:
            return "Please provide a title. Example: add evidence Broken window | Glass ..."

        return self.add_evidence(title, description, evidence_type, source)

    def _cmd_view_evidence(self, text):
        evidence_id = self._extract_evidence_id(text)
        if not evidence_id:
            return "Please include an evidence ID. Example: view evidence EVD-001"
        return self.view_evidence(evidence_id)

    def _cmd_verify_evidence(self, text):
        evidence_id = self._extract_evidence_id(text)
        if not evidence_id:
            return "Please include an evidence ID. Example: verify evidence EVD-001"
        return self.verify_evidence(evidence_id)

    def _cmd_dispute_evidence(self, text):
        evidence_id = self._extract_evidence_id(text)
        if not evidence_id:
            return "Please include an evidence ID. Example: dispute evidence EVD-001"
        return self.dispute_evidence(evidence_id)

    def _cmd_delete_evidence(self, text):
        evidence_id = self._extract_evidence_id(text)
        if not evidence_id:
            return "Please include an evidence ID. Example: delete evidence EVD-001"
        return self.delete_evidence(evidence_id)

    def _extract_case_id(self, text):
        match = re.search(r"(?i)\bcase-\d+\b", text)
        return match.group(0).upper() if match else None

    def _extract_evidence_id(self, text):
        match = re.search(r"(?i)\bevd-\d+\b", text)
        return match.group(0).upper() if match else None

    # ── LLM reasoning ─────────────────────────────────────────────────────

    def respond(self, user_message, history=None, top_k=5):
        """Get an LLM-generated response.

        Adds four kinds of context, in order:
          1. The persona system prompt.
          2. Active case + evidence context (existing behaviour).
          3. Bounded recent conversation (history, optional — the GUI
             passes its _history so this session's past turns are visible).
          4. Relevant long-term memories recalled from MemoryStore.

        Case/evidence data and memory are passed as text only. The model
        cannot modify them; all changes go through the managers above.
        """
        system_prompt = self.persona.system_prompt

        active_case = self.case_manager.get_active()
        if active_case:
            system_prompt += self._build_case_context(active_case)

        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Short-term context: the last few conversation turns (if given).
        if history:
            messages += self._recent_history_messages(history)

        # Long-term context: memories relevant to what the user just asked.
        recalled, _ = self.memory.recall(user_message, top_k=top_k)
        if recalled:
            messages.append({"role": "system", "content": recalled})

        messages.append({"role": "user", "content": user_message})

        return self.llm.chat(messages)

    def _recent_history_messages(self, history, max_turns=6):
        """Turn the last few (role, text) turns into LLM chat messages.

        Bounds the amount of conversation sent so the context window does
        not grow forever. This is short-term context, not long-term memory.
        """
        recent = history[-max_turns * 2:]
        messages = []
        for role, text in recent:
            llm_role = "assistant" if role == "agent" else "user"
            messages.append({"role": llm_role, "content": text})
        return messages

    def _build_case_context(self, case):
        """Build the case + evidence text block for the system prompt."""
        lines = [
            "",
            "",
            "━━ ACTIVE CASE CONTEXT ━━",
            f"Case ID: {case.id}",
            f"Title: {case.title}",
            f"Status: {case.status}",
            f"Description: {case.description or '(no description)'}",
        ]

        items = self.evidence_manager.list_for_case(case.id)

        if items:
            lines.append("")
            lines.append(f"EVIDENCE ({len(items)} item(s)):")

            # Group by status so the model can see trust levels immediately.
            for status in (EvidenceStatus.VERIFIED,
                           EvidenceStatus.UNVERIFIED,
                           EvidenceStatus.DISPUTED):
                group = [i for i in items if i.status == status]
                if not group:
                    continue
                lines.append(f"[{status.value}]")
                for item in group:
                    lines.append(f"  {item.id} — {item.title}")
                    lines.append(f"    Type: {item.evidence_type.value}")
                    lines.append(f"    Description: {item.description or '(none)'}")
                    lines.append(f"    Source: {item.source or '(unknown)'}")

            lines.append("")
            lines.append(
                "Evidence status matters: VERIFIED evidence may be treated as "
                "established. UNVERIFIED evidence is only a claim and must NOT "
                "be treated as fact. DISPUTED evidence is contested and must "
                "not be relied upon. Never present UNVERIFIED or DISPUTED "
                "evidence as confirmed."
            )
        else:
            lines.append("")
            lines.append("EVIDENCE: (none recorded yet)")

        lines.append("Use this context when reasoning.")
        return "\n".join(lines)