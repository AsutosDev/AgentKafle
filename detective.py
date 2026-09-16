"""Detective reasoning engine for AgentKafle.

The Detective is the brains of the operation. It sits between the structured
data (CaseManager + EvidenceManager) and the LLM:

  AgentKafle → Detective → CaseManager / EvidenceManager   (structured data)
                        → reason() → LLMInterface          (later step)

Right now the Detective only has its *investigation layer*: it collects a
case and its evidence into one structured object. The full reasoning step
(hypotheses, contradictions, suspects, the final report) will be built on
top of that object in a later step.

WHY THIS DESIGN?

- The Detective never reads or writes JSON files directly. It only talks to
  CaseManager and EvidenceManager, which stay the single sources of truth.
- investigate() returns an Investigation object (data), not printed text.
  That way the GUI, the CLI, or the future LLM can each use the same data
  however they like.
- Evidence is grouped by status so anyone reading the object instantly sees
  what can be trusted (VERIFIED), what is only a claim (UNVERIFIED), and
  what is contested (DISPUTED).
"""

from dataclasses import dataclass, field

from cases import CaseManager
from evidence import EvidenceManager, EvidenceStatus


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


# ── Detective ─────────────────────────────────────────────────────────────
# Only builds structured contexts for now. The reasoning step (reason()) is
# added later and will consume the Investigation this class produces.
# ───────────────────────────────────────────────────────────────────────────

class Detective:
    """Builds structured investigation contexts for a case.

    Like the managers, investigate() returns a (result, message) tuple so
    callers can branch on success/failure without try/except blocks. On
    success the result is an Investigation object.
    """

    def __init__(self, case_manager, evidence_manager):
        """Store the managers. The Detective never touches files directly."""
        self.case_manager = case_manager
        self.evidence_manager = evidence_manager

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

        This text is what will be sent to the LLM as context in a later step.
        It is kept here so the GUI/CLI can also display the same context in a
        human-friendly way before any AI reasoning exists.
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

    # ── placeholder for the LLM reasoning step ─────────────────────────────
    def reason(self, investigation):
        """Run the reasoning step on an Investigation.

        NOT IMPLEMENTED YET. This is the next milestone. When it lands, it
        will:
          1. Turn the Investigation into a prompt (format_context above).
          2. Send it to the LLM via LLMInterface.
          3. Return the model's hypotheses / contradictions / report.

        Raising NotImplementedError keeps the design obvious without adding
        half-working AI code.
        """
        raise NotImplementedError(
            "Detective.reason() is the next step. It will pass the "
            "Investigation context to the LLM and return the analysis."
        )