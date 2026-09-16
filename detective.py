"""Detective reasoning engine for AgentKafle.

The Detective is the brains of the operation. It sits between the structured
data (CaseManager + EvidenceManager) and the LLM:

  AgentKafle → Detective → CaseManager / EvidenceManager   (structured data)
                        → reason() → LLMInterface          (LLM reasoning)

The Detective has two layers:

1. Investigation layer — investigate() collects a case and its evidence into
   one structured Investigation object, with evidence grouped by status.
2. Reasoning layer — reason() sends that Investigation to the LLM and turns
   the model's reply into a structured ReasoningResult (summary, facts,
   hypotheses, confidence, missing information, etc.).

WHY THIS DESIGN?

- The Detective never reads or writes JSON files directly. It only talks to
  CaseManager and EvidenceManager, which stay the single sources of truth.
- investigate() returns an Investigation object (data), not printed text.
  That way the GUI, the CLI, or the future LLM can each use the same data
  however they like.
- Evidence is grouped by status so anyone reading the object instantly sees
  what can be trusted (VERIFIED), what is only a claim (UNVERIFIED), and
  what is contested (DISPUTED).
- reason() returns a ReasoningResult object (data), not just printed text.
  The LLM is asked for JSON, and a small parser turns that text into fields.

LLM communication stays inside llm.py: the Detective only calls
LLMInterface.chat() and never talks to the model provider directly.
"""

import json
import re
from dataclasses import dataclass, field

from cases import CaseManager
from evidence import EvidenceManager, EvidenceStatus
from llm import LLMInterface


# ── Investigation context ─────────────────────────────────────────────────
# One Investigation holds everything the reasoning step needs about a case.
# It is plain data: no file access, no printing, no side effects.
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class Investigation:
    """A structured snapshot of one case and all of its evidence.

    The evidence lists hold the same Evidence objects that come from
    EvidenceManager; the three grouped lists are simply views of the full
    ``evidence`` list, split by status.

    Required fields: case_id, case_title, case_description, evidence,
    verified_evidence, unverified_evidence, disputed_evidence.
    """

    case_id: str
    case_title: str
    case_description: str = ""
    case_status: str = ""
    evidence: list = field(default_factory=list)
    verified_evidence: list = field(default_factory=list)
    unverified_evidence: list = field(default_factory=list)
    disputed_evidence: list = field(default_factory=list)

    @property
    def evidence_count(self):
        """How many pieces of evidence the case has in total."""
        return len(self.evidence)

    def to_dict(self):
        """Convert to plain dicts so it can be saved or sent over JSON."""
        return {
            "case_id": self.case_id,
            "case_title": self.case_title,
            "case_description": self.case_description,
            "case_status": self.case_status,
            "evidence_count": self.evidence_count,
            "evidence": [item.to_dict() for item in self.evidence],
            "verified_evidence": [item.to_dict() for item in self.verified_evidence],
            "unverified_evidence": [item.to_dict() for item in self.unverified_evidence],
            "disputed_evidence": [item.to_dict() for item in self.disputed_evidence],
        }


# ── Reasoning result ─────────────────────────────────────────────────────
# One ReasoningResult is the structured output of a single reason() call.
# It is the LLM's analysis, cleaned into plain Python fields so the GUI and
# the final report can rely on a fixed shape instead of free-form text.
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ReasoningResult:
    """Structured reasoning output produced from an Investigation.

    ``confidence`` is a judgment by the model (LOW, MEDIUM, or HIGH), never
    a calculated number. ``raw_text`` keeps the model's exact reply so the
    analysis can be audited later.
    """

    summary: str = ""
    established_facts: list = field(default_factory=list)
    unverified_claims: list = field(default_factory=list)
    disputed_information: list = field(default_factory=list)
    contradictions: list = field(default_factory=list)
    hypotheses: list = field(default_factory=list)
    conclusion: str = ""
    confidence: str = "LOW"                     # LOW, MEDIUM, or HIGH
    missing_information: list = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self):
        """Convert to plain dicts for saving or displaying."""
        return {
            "summary": self.summary,
            "established_facts": self.established_facts,
            "unverified_claims": self.unverified_claims,
            "disputed_information": self.disputed_information,
            "contradictions": self.contradictions,
            "hypotheses": self.hypotheses,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "missing_information": self.missing_information,
        }

    @classmethod
    def from_dict(cls, data, raw_text=""):
        """Build a clean ReasoningResult from a parsed JSON dict.

        Missing or oddly-typed fields are defaulted instead of crashing, so
        a slightly imperfect model reply still produces a usable result.
        """

        def clean_list(value):
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            if isinstance(value, str) and value.strip():
                return [value.strip()]
            return []

        def clean_text(value):
            return value.strip() if isinstance(value, str) else ""

        confidence = clean_text(data.get("confidence")).upper()
        if confidence not in ("LOW", "MEDIUM", "HIGH"):
            confidence = "LOW"

        return cls(
            summary=clean_text(data.get("summary")),
            established_facts=clean_list(data.get("established_facts")),
            unverified_claims=clean_list(data.get("unverified_claims")),
            disputed_information=clean_list(data.get("disputed_information")),
            contradictions=clean_list(data.get("contradictions")),
            hypotheses=clean_list(data.get("hypotheses")),
            conclusion=clean_text(data.get("conclusion")),
            confidence=confidence,
            missing_information=clean_list(data.get("missing_information")),
            raw_text=raw_text,
        )

    def as_text(self):
        """Format the reasoning result as readable text for display."""
        lines = [f"SUMMARY: {self.summary}", ""]

        lines.append(f"ESTABLISHED FACTS ({len(self.established_facts)}):")
        lines += [f"  - {item}" for item in self.established_facts] or ["  (none)"]

        lines.append("")
        lines.append(f"UNVERIFIED CLAIMS ({len(self.unverified_claims)}):")
        lines += [f"  - {item}" for item in self.unverified_claims] or ["  (none)"]

        lines.append("")
        lines.append(f"DISPUTED INFORMATION ({len(self.disputed_information)}):")
        lines += [f"  - {item}" for item in self.disputed_information] or ["  (none)"]

        lines.append("")
        lines.append(f"CONTRADICTIONS ({len(self.contradictions)}):")
        lines += [f"  - {item}" for item in self.contradictions] or ["  (none)"]

        lines.append("")
        lines.append(f"HYPOTHESES ({len(self.hypotheses)}):")
        lines += [f"  - {item}" for item in self.hypotheses] or ["  (none)"]

        lines.append("")
        lines.append(f"CONCLUSION: {self.conclusion}")

        lines.append("")
        lines.append(f"CONFIDENCE: {self.confidence}")

        lines.append("")
        lines.append(f"MISSING INFORMATION ({len(self.missing_information)}):")
        lines += [f"  - {item}" for item in self.missing_information] or ["  (none)"]

        return "\n".join(lines)


# ── Reasoning system prompt ───────────────────────────────────────────────
# This is the ONLY set of reasoning rules for the reasoning layer. It tells
# the model exactly how evidence status must be respected and that it must
# answer in JSON, which reason() then parses.
# ───────────────────────────────────────────────────────────────────────────

REASONING_SYSTEM_PROMPT = """\
You are an analytical AI detective assisting with an investigation.

You must reason ONLY from the case description and evidence supplied in the
user's message. Follow these rules strictly:

1. Never invent evidence, sources, or events that were not supplied.
2. You have no ability to execute commands or browse the internet. Never
   claim that you did either.
3. Evidence status decides how much weight an item may carry:
   - VERIFIED evidence may be treated as established evidence.
   - UNVERIFIED evidence must remain explicitly identified as unverified.
     It may be considered only as a claim or as input to a hypothesis.
   - DISPUTED evidence must be identified as disputed. It must not
     independently support the conclusion.
4. Contradictions: identify places where pieces of evidence conflict. Do
   not arbitrarily choose which piece is true. Explain what additional
   information would resolve the conflict.
5. Hypotheses: present possible explanations consistent with the evidence.
   Clearly distinguish hypotheses from facts and never state a hypothesis
   as proven.
6. Conclusion: summarize what can reasonably be concluded from the
   available evidence. If the evidence is insufficient, explicitly say so.
7. Confidence: use only LOW, MEDIUM, or HIGH. Base it on evidence quality
   and consistency. This is a judgment, not a mathematical calculation.
8. Missing information: list the most important information needed to
   strengthen or resolve the investigation.

Respond with ONLY valid JSON. Do not add prose outside the JSON and do not
wrap the JSON in markdown code fences.
"""


# ── Detective ─────────────────────────────────────────────────────────────
# Builds structured contexts and runs LLM reasoning on them.
# ───────────────────────────────────────────────────────────────────────────

class Detective:
    """Builds investigation contexts and runs LLM reasoning on them.

    Like the managers, investigate() and reason() return (result, message)
    tuples so callers can branch on success/failure without try/except
    blocks. investigate() returns an Investigation; reason() returns a
    ReasoningResult.
    """

    def __init__(self, case_manager, evidence_manager, llm=None):
        """Store the managers and the LLM. The Detective never touches files.

        llm is an optional LLMInterface instance. If omitted, a default
        LLMInterface is created so reason() works out of the box. Callers
        that already own an LLM (like AgentKafle) can pass it in to share a
        single connection.
        """
        self.case_manager = case_manager
        self.evidence_manager = evidence_manager
        self.llm = llm if llm is not None else LLMInterface()

    def set_llm(self, llm):
        """Point this Detective at a different LLM (e.g. after a provider switch).

        reason() always uses whatever LLM is set here, which keeps structured
        detective reasoning on the same provider the agent has selected.
        """
        self.llm = llm

    def get_active_case(self):
        """Return the active Case object, or None if there is none."""
        return self.case_manager.get_active()

    def investigate(self, case_id=None):
        """Collect one case and its evidence into an Investigation.

        - case_id is optional. If given, that case must exist. If omitted,
          the active case is used instead.
        - Evidence is gathered via EvidenceManager and split into three
          status groups: VERIFIED, UNVERIFIED, DISPUTED.

        Returns (Investigation, message) on success or (None, error_message)
        when the case does not exist / there is no active case.
        """
        # Pick the case: by ID, or fall back to the active case.
        if case_id is not None:
            case = self.case_manager.get(case_id)
            if case is None:
                return None, f"Error: Case {case_id} not found."
        else:
            case = self.case_manager.get_active()
            if case is None:
                return None, "Error: No active case. Select or create one first."

        # The case description is part of the context (see Investigation).
        description = case.description

        # Gather all evidence for this case (order they were added).
        items = self.evidence_manager.list_for_case(case.id)

        # Split by status so trust levels are visible at a glance.
        verified = [i for i in items if i.status == EvidenceStatus.VERIFIED]
        unverified = [i for i in items if i.status == EvidenceStatus.UNVERIFIED]
        disputed = [i for i in items if i.status == EvidenceStatus.DISPUTED]

        investigation = Investigation(
            case_id=case.id,
            case_title=case.title,
            case_description=description,
            case_status=case.status,
            evidence=items,
            verified_evidence=verified,
            unverified_evidence=unverified,
            disputed_evidence=disputed,
        )

        message = (
            f"Investigation context ready for {case.id} "
            f"({investigation.evidence_count} evidence item(s))."
        )
        return investigation, message

    # ── helper: format the context ───────────────────────────────────────
    def format_context(self, investigation):
        """Format an Investigation as readable text.

        This text is exactly what reason() sends to the LLM as the case
        context, and it lets the GUI/CLI display the same context in a
        human-friendly way.
        """
        lines = [
            f"Case: {investigation.case_id}",
            f"Title: {investigation.case_title}",
            f"Status: {investigation.case_status}",
            f"Description: {investigation.case_description or '(no description)'}",
            "",
            f"EVIDENCE ({investigation.evidence_count} item(s)):",
        ]

        # Data is a list of (label, evidence_list) pairs so the ordering of
        # the status groups is explicit and easy to read.
        groups = [
            ("VERIFIED", investigation.verified_evidence),
            ("UNVERIFIED", investigation.unverified_evidence),
            ("DISPUTED", investigation.disputed_evidence),
        ]

        for label, group in groups:
            if not group:
                continue
            lines.append(f"[{label}]")
            for item in group:
                lines.append(f"  {item.id} — {item.title}")
                lines.append(f"    Type: {item.evidence_type.value}")
                lines.append(f"    Description: {item.description or '(none)'}")
                lines.append(f"    Source: {item.source or '(unknown)'}")

        if not investigation.evidence:
            lines.append("  (no evidence recorded yet)")

        return "\n".join(lines)

    # ── LLM reasoning step ────────────────────────────────────────────────
    # reason() builds the messages, calls the LLM through llm.py, and turns
    # the model's JSON reply into a structured ReasoningResult.
    # ───────────────────────────────────────────────────────────────────────

    def reason(self, investigation):
        """Send an Investigation to the LLM and return structured reasoning.

        Returns (ReasoningResult, message) on success, or (None, error) if
        the LLM call itself fails (for example Ollama is not running). The
        reasoning is returned as data (see ReasoningResult), never printed.

        LLM failures are caught here so a flaky model does not crash the
        caller; the prompt rules above keep the model from inventing
        evidence or over-claiming from UNVERIFIED/DISPUTED items.
        """
        messages = self._build_reasoning_messages(investigation)

        try:
            raw_text = self.llm.chat(messages, json_mode=True)
        except Exception as exc:
            return None, f"Reasoning failed: LLM error ({exc})"

        # The model replies in text; parse it into structured fields.
        data = self._extract_json(raw_text)
        result = ReasoningResult.from_dict(data, raw_text=raw_text)

        message = f"Reasoning complete for {investigation.case_id}."
        return result, message

    def _build_reasoning_messages(self, investigation):
        """Build the system + user messages for the reasoning LLM call."""
        user_prompt = (
            "Investigate the case below using ONLY the supplied "
            "information.\n\n"
            f"{self.format_context(investigation)}\n\n"
            'Return your analysis as a single JSON object with these keys: '
            '"summary", "established_facts", "unverified_claims", '
            '"disputed_information", "contradictions", "hypotheses", '
            '"conclusion", "confidence", "missing_information".\n'
            'Shape rules:\n'
            '- "summary" and "conclusion": short text strings.\n'
            '- "established_facts", "unverified_claims", '
            '"disputed_information", "contradictions", "hypotheses", and '
            '"missing_information": arrays of strings (empty arrays if no '
            'items apply).\n'
            '- "confidence": exactly one of "LOW", "MEDIUM", "HIGH".\n'
            "If the evidence is insufficient to conclude anything, say so "
            "in the conclusion and choose a LOW or MEDIUM confidence."
        )

        return [
            {"role": "system", "content": REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _extract_json(self, text):
        """Parse JSON from the model's reply, tolerating a few common issues.

        Handles: an empty reply (returns {}), replies wrapped in markdown
        code fences, and replies that include extra prose around the JSON.
        """
        if not text or not text.strip():
            return {}

        cleaned = text.strip()

        # Remove a surrounding ```json ... ``` fence if the model added one.
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Last resort: grab the first { ... } block in the reply.
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

        return {}