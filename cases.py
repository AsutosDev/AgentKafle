"""Case management for AgentKafle.

This module provides three things:

1. CaseStatus  — the possible states of a case (OPEN, SOLVED, CLOSED).
2. Case        — the data model for a single investigation.
3. CaseManager — creates, loads, updates, and deletes cases. Stores each
                  case as its own JSON file under a cases/ directory so
                  that no single file holds all case data.

WHY THIS DESIGN?

- One file per case means adding evidence later (Step 4) can attach
  files alongside the case without touching other cases.
- CaseManager is completely separate from AgentKafle. The agent calls
  CaseManager methods to get or change data but never writes JSON directly.
- CaseStatus is a plain class with string constants rather than a Python
  enum, so it stays easy to understand and extend.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


# ── Case statuses ─────────────────────────────────────────────────────────
# Simple string constants.  To add a new status later, just add another
# line here and check it in CaseManager.update_status().
# ───────────────────────────────────────────────────────────────────────────

class CaseStatus:
    OPEN = "OPEN"
    SOLVED = "SOLVED"
    CLOSED = "CLOSED"


# ── Case data model ───────────────────────────────────────────────────────
# A dataclass keeps the fields visible in one place and gives us free
# __init__, __repr__, and to_dict / from_dict helpers.
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class Case:
    id: str
    title: str
    description: str = ""
    status: str = CaseStatus.OPEN
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# ── CaseManager ───────────────────────────────────────────────────────────
# Owns the cases/ directory.  Every public method returns a
# (result, message) tuple so the caller can branch on success/failure
# without try/except blocks.
# ───────────────────────────────────────────────────────────────────────────

class CaseManager:
    def __init__(self, storage_dir="cases"):
        self.storage_dir = storage_dir
        self.cases = {}            # id string → Case object
        self.active_case_id = None  # id string or None
        os.makedirs(storage_dir, exist_ok=True)
        self._load_all()

    # ── public API ────────────────────────────────────────────────────────

    def create(self, title, description=""):
        """Create a new case and save it to disk."""
        title = title.strip()
        if not title:
            return None, "Error: Case title cannot be empty."

        case_id = self._next_id()
        case = Case(id=case_id, title=title, description=description.strip())
        self.cases[case_id] = case
        self._save(case)
        return case, f"Created {case_id}: {case.title}"

    def list_all(self):
        """Return every case, newest first."""
        return sorted(self.cases.values(), key=lambda c: c.created_at, reverse=True)

    def get(self, case_id):
        """Look up a case by its ID.  Returns the Case or None."""
        return self.cases.get(case_id.upper())

    def update_status(self, case_id, new_status):
        """Change a case's status.  Returns (case, message) or (None, error)."""
        case = self.get(case_id)
        if not case:
            return None, f"Error: Case {case_id} not found."

        valid = (CaseStatus.OPEN, CaseStatus.SOLVED, CaseStatus.CLOSED)
        if new_status not in valid:
            return None, f"Error: Invalid status '{new_status}'. Use OPEN, SOLVED, or CLOSED."

        old_status = case.status
        case.status = new_status
        case.updated_at = _now_iso()
        self._save(case)
        return case, f"{case_id} status changed: {old_status} → {new_status}"

    def delete(self, case_id):
        """Delete a case from disk.  Returns (True, message) or (None, error)."""
        case = self.get(case_id)
        if not case:
            return None, f"Error: Case {case_id} not found."

        filepath = self._filepath(case.id)
        if os.path.exists(filepath):
            os.remove(filepath)

        del self.cases[case_id]

        if self.active_case_id == case_id:
            self.active_case_id = None

        return True, f"Deleted {case_id}: {case.title}"

    def set_active(self, case_id):
        """Select the active case.  Returns (case, message) or (None, error)."""
        case = self.get(case_id)
        if not case:
            return None, f"Error: Case {case_id} not found."

        self.active_case_id = case_id
        return case, f"Active case: {case.id} — {case.title} ({case.status})"

    def get_active(self):
        """Return the active Case object, or None."""
        if not self.active_case_id:
            return None
        case = self.get(self.active_case_id)
        if not case:
            # Active case was deleted externally.
            self.active_case_id = None
            return None
        return case

    # ── internals ─────────────────────────────────────────────────────────

    def _next_id(self):
        """Generate the next sequential case ID."""
        existing = []
        for cid in self.cases:
            match = re.match(r"CASE-(\d+)", cid)
            if match:
                existing.append(int(match.group(1)))

        next_num = max(existing, default=0) + 1
        return f"CASE-{next_num:03d}"

    def _filepath(self, case_id):
        return os.path.join(self.storage_dir, f"{case_id}.json")

    def _save(self, case):
        """Write one case to its own JSON file."""
        filepath = self._filepath(case.id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(case.to_dict(), f, indent=2)

    def _load_all(self):
        """Read every case file from the storage directory."""
        for filename in sorted(os.listdir(self.storage_dir)):
            # Match only CASE-001.json, not CASE-001.evidence.json (Step 4).
            if not re.match(r"^CASE-\d+\.json$", filename):
                continue

            filepath = os.path.join(self.storage_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                case = Case.from_dict(data)
                self.cases[case.id] = case
            except (json.JSONDecodeError, KeyError):
                # Skip corrupt files rather than crashing.
                continue