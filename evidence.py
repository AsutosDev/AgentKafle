"""Evidence management for AgentKafle.

This module provides:

1. EvidenceType    — the kind of evidence (PHYSICAL, DIGITAL, TESTIMONY,
                     DOCUMENT, OTHER).
2. EvidenceStatus  — how trustworthy the evidence is (UNVERIFIED, VERIFIED,
                     DISPUTED).
3. Evidence        — the data model for one piece of evidence.
4. EvidenceManager — adds, loads, updates, and deletes evidence. Each case's
                     evidence is stored in its own file, for example:
                         cases/CASE-001.evidence.json

WHY THIS DESIGN?

- One file per case keeps evidence independent, just like cases.
- The LLM never touches these files. EvidenceManager is the single source of
  truth for evidence data.
- Evidence IDs (EVD-001, EVD-002, ...) are unique across every case, so an ID
  can never refer to two different items.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum


# ── Evidence type ─────────────────────────────────────────────────────────
# Deriving from str as well as Enum lets us serialise these directly to JSON
# while still keeping the readable Enum API in Python.

class EvidenceType(str, Enum):
    PHYSICAL = "PHYSICAL"
    DIGITAL = "DIGITAL"
    TESTIMONY = "TESTIMONY"
    DOCUMENT = "DOCUMENT"
    OTHER = "OTHER"


# ── Evidence status ───────────────────────────────────────────────────────
# Status tells the detective how much weight the evidence can carry:
#   VERIFIED   — confirmed, may support conclusions.
#   UNVERIFIED — reported but not confirmed; treat as a claim.
#   DISPUTED   — actively contested; do not rely on it.

class EvidenceStatus(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    DISPUTED = "DISPUTED"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# ── Evidence data model ───────────────────────────────────────────────────

@dataclass
class Evidence:
    id: str
    case_id: str
    title: str
    description: str = ""
    evidence_type: EvidenceType = EvidenceType.OTHER
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    source: str = ""
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self):
        data = asdict(self)
        # Store the readable string values in JSON, not the Enum objects.
        data["evidence_type"] = self.evidence_type.value
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        # Rebuild Enum values from the strings stored on disk.
        data["evidence_type"] = EvidenceType(data.get("evidence_type", "OTHER"))
        data["status"] = EvidenceStatus(data.get("status", "UNVERIFIED"))
        return cls(**data)


# ── EvidenceManager ───────────────────────────────────────────────────────
# Every public method returns a (result, message) tuple, matching CaseManager,
# so callers can branch on success/failure without try/except.

class EvidenceManager:
    def __init__(self, storage_dir="cases"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._next_num = self._scan_next_id()

    # ── public API ────────────────────────────────────────────────────────

    def add(self, case_id, title, description="", evidence_type=EvidenceType.OTHER, source=""):
        """Add one piece of evidence to a case."""
        title = title.strip()
        if not title:
            return None, "Error: Evidence title cannot be empty."

        if not isinstance(evidence_type, EvidenceType):
            return None, f"Error: Unknown evidence type '{evidence_type}'."

        evidence_id = self._make_id()
        item = Evidence(
            id=evidence_id,
            case_id=case_id,
            title=title,
            description=description.strip(),
            evidence_type=evidence_type,
            source=source.strip(),
        )

        items = self._load_case(case_id)
        items.append(item)
        self._save_case(case_id, items)

        return item, (
            f"Added {item.id}: {item.title} "
            f"[{item.evidence_type.value} / {item.status.value}]"
        )

    def list_for_case(self, case_id):
        """Return all evidence for a case, in the order added."""
        return self._load_case(case_id)

    def get(self, case_id, evidence_id):
        """Look up one evidence item.  Returns Evidence or None."""
        evidence_id = evidence_id.upper()
        for item in self._load_case(case_id):
            if item.id == evidence_id:
                return item
        return None

    def update_status(self, case_id, evidence_id, new_status):
        """Change an item's status.  Returns (item, message) or (None, error)."""
        if not isinstance(new_status, EvidenceStatus):
            return None, f"Error: Invalid status '{new_status}'."

        items = self._load_case(case_id)
        for item in items:
            if item.id == evidence_id.upper():
                old_status = item.status.value
                item.status = new_status
                item.updated_at = _now_iso()
                self._save_case(case_id, items)
                return item, (
                    f"{item.id} status changed: {old_status} → {new_status.value}"
                )

        return None, f"Error: Evidence {evidence_id} not found."

    def update(self, case_id, evidence_id, title=None, description=None,
               evidence_type=None, source=None):
        """Update fields on an item.  Only non-None fields are changed."""
        items = self._load_case(case_id)
        for item in items:
            if item.id == evidence_id.upper():
                if title is not None:
                    item.title = title.strip()
                if description is not None:
                    item.description = description.strip()
                if evidence_type is not None:
                    item.evidence_type = evidence_type
                if source is not None:
                    item.source = source.strip()
                item.updated_at = _now_iso()
                self._save_case(case_id, items)
                return item, f"Updated {item.id}: {item.title}"

        return None, f"Error: Evidence {evidence_id} not found."

    def delete(self, case_id, evidence_id):
        """Delete one item.  Returns (True, message) or (None, error)."""
        items = self._load_case(case_id)
        for item in items:
            if item.id == evidence_id.upper():
                items.remove(item)
                self._save_case(case_id, items)
                return True, f"Deleted {item.id}: {item.title}"

        return None, f"Error: Evidence {evidence_id} not found."

    # ── internals ─────────────────────────────────────────────────────────

    def _filepath(self, case_id):
        return os.path.join(self.storage_dir, f"{case_id}.evidence.json")

    def _load_case(self, case_id):
        """Read a case's evidence file.  Returns a list of Evidence."""
        filepath = self._filepath(case_id)
        if not os.path.exists(filepath):
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Evidence.from_dict(raw) for raw in data.get("evidence", [])]
        except (json.JSONDecodeError, KeyError, ValueError):
            return []

    def _save_case(self, case_id, items):
        """Write a case's evidence list to disk."""
        payload = {
            "case_id": case_id,
            "evidence": [item.to_dict() for item in items],
        }
        with open(self._filepath(case_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _make_id(self):
        self._next_num += 1
        return f"EVD-{self._next_num:03d}"

    def _scan_next_id(self):
        """Find the highest existing EVD number so IDs stay unique."""
        highest = 0
        for filename in os.listdir(self.storage_dir):
            if not re.match(r"^CASE-\d+\.evidence\.json$", filename):
                continue

            filepath = os.path.join(self.storage_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            for raw in data.get("evidence", []):
                match = re.match(r"EVD-(\d+)", raw.get("id", ""))
                if match:
                    highest = max(highest, int(match.group(1)))

        return highest