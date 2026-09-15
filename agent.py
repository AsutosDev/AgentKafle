"""The main AgentKafle class.

AgentKafle is the top-level object that the rest of the app talks to. It sits
between the user and three collaborators:

  User → AgentKafle → CaseManager    (structured case data, no LLM)
                    → Persona + LLM  (language and reasoning)

Key rule: the LLM never reads or writes case files. CaseManager is the single
source of truth for structured case data. The LLM only receives case context
when reasoning is required, as text inside the system prompt.

The detective persona (system prompt) lives in persona.py, not here.
"""

import re

from llm import LLMInterface
from persona import Persona
from cases import CaseManager, CaseStatus


class AgentKafle:
    def __init__(self, name="AgentKafle"):
        self.name = name
        self.llm = LLMInterface()
        self.persona = Persona(agent_name=name)
        self.case_manager = CaseManager()

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

    # ── Command router ────────────────────────────────────────────────────
    # Detects case commands in plain language. Returns (handled, response).
    # If handled is False, the caller should send the text to the LLM.

    def handle_command(self, text):
        lower = text.lower().strip()

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

    def _extract_case_id(self, text):
        match = re.search(r"(?i)\bcase-\d+\b", text)
        return match.group(0).upper() if match else None

    # ── LLM reasoning ─────────────────────────────────────────────────────

    def respond(self, user_message):
        """Get an LLM-generated response, with active case context attached.

        The case data is passed as text only. The model cannot modify it;
        all changes go through CaseManager above.
        """
        system_prompt = self.persona.system_prompt

        active_case = self.case_manager.get_active()
        if active_case:
            system_prompt += (
                "\n\n━━ ACTIVE CASE CONTEXT ━━\n"
                f"Case ID: {active_case.id}\n"
                f"Title: {active_case.title}\n"
                f"Status: {active_case.status}\n"
                f"Description: {active_case.description or '(no description)'}\n"
                "Use this context when reasoning. It is the current investigation."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        return self.llm.chat(messages)