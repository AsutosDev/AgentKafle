"""Memory 2.0 — long-term persistent memory for AgentKafle.

This module provides three things:

1. MemoryType       — what kind of memory this is (EPISODIC, SEMANTIC,
                      CASE, CONVERSATION).
2. MemoryImportance — how much weight a memory should carry (LOW, MEDIUM,
                      HIGH).
3. Memory           — the data model for one stored memory.
4. MemoryStore      — the subsystem that owns persistent memory. Memories
                      are stored in one JSON file: memory/memories.json.

WHY THIS DESIGN?

- MemoryStore owns storage, exactly like CaseManager and EvidenceManager.
  Nothing else (agent, detective, GUI, LLM) reads or writes memory files.
  The LLM must NEVER write memory directly; it can only suggest memories,
  and the agent writes them through MemoryStore.
- The Memory dataclass carries metadata (type, importance, source, case_id,
  confidence, links) so retrieval can rank memories beyond raw keyword hits.
- Retrieval is delegated to a small KeywordRetriever (a "strategy"). Phase 2
  can swap in an embedding/vector retriever behind the same interface
  without touching the rest of AgentKafle.

Short-term conversation is a SEPARATE system from long-term memory:
- Conversation history lives in the GUI's _history and is only passed to the
  LLM for the current session.
- MemoryStore is the single source of truth for persisted memories.

This is Phase 1: persistent storage, retrieval, update, soft-delete, linking,
and recall formatting. Automatic extraction/summarization is Phase 2.
"""

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum


# Project root (directory containing this file) so the default memory path is
# stable regardless of the process's current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MEMORY_DIR = os.path.join(_PROJECT_ROOT, "memory")


# ── Memory type & importance ──────────────────────────────────────────────
# Plain str-Enums so these serialise cleanly to JSON, matching EvidenceType.

class MemoryType(str, Enum):
    EPISODIC = "EPISODIC"          # a specific past event or interaction
    SEMANTIC = "SEMANTIC"          # a fact or piece of general knowledge
    CASE = "CASE"                  # investigation-specific information
    CONVERSATION = "CONVERSATION"  # a past conversation or exchange


class MemoryImportance(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# ── Memory data model ─────────────────────────────────────────────────────

@dataclass
class Memory:
    """One stored memory.

    ``links`` connects this memory to cases, evidence, conversations, or
    other memories. Each entry is a dict: {"kind": ..., "target_id": ...}.

    ``embedding`` is reserved for Phase 2 vector search and stays None here.
    ``active`` supports soft-delete: forgetting sets it to False instead of
    erasing the row.
    """

    id: str
    type: MemoryType
    content: str
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())
    source: str = ""
    importance: MemoryImportance = MemoryImportance.MEDIUM
    case_id: str = ""
    confidence: str = ""                 # "", LOW, MEDIUM, or HIGH
    tags: list = field(default_factory=list)
    links: list = field(default_factory=list)
    embedding: list = None               # reserved for Phase 2
    active: bool = True

    def to_dict(self):
        data = asdict(self)
        # Store readable string values, not Enum objects.
        data["type"] = self.type.value
        data["importance"] = self.importance.value
        return data

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["type"] = MemoryType(data.get("type", MemoryType.SEMANTIC.value))
        data["importance"] = MemoryImportance(
            data.get("importance", MemoryImportance.MEDIUM.value)
        )
        return cls(**data)


# ── Keyword retrieval strategy ────────────────────────────────────────────
# Phase 1 retrieval: token overlap + importance boost + recency factor.
# A vector retriever with the same retrieve() signature replaces this later.

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "of", "to",
    "in", "on", "at", "for", "from", "with", "without", "about", "into",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "it", "this", "that", "these", "those", "i",
    "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "our", "their", "his", "as", "by", "not", "no", "so",
    "just", "what", "when", "where", "which", "who", "how", "can", "will",
    "would", "should", "could", "more", "most", "very", "also", "there",
}


# ── Word families ─────────────────────────────────────────────────────────
# A tiny built-in, dependency-free thesaurus for common investigation words
# so related wording can match even when simple stemming would not (e.g.
# "explain" ↔ "explanations"). The key is one member; the value is the whole
# family. Add more groups here as vocabulary grows.
# ───────────────────────────────────────────────────────────────────────────

WORD_FAMILIES = {
    "explain": {"explain", "explains", "explained", "explaining",
                "explanation", "explanations"},
    "prefer": {"prefer", "prefers", "preferred", "preference",
               "preferences", "preferring"},
    "remember": {"remember", "remembers", "remembered", "remembering",
                 "memory", "memories", "recall", "recalls", "recalled",
                 "recollection"},
    "understand": {"understand", "understands", "understood", "understanding"},
    "concise": {"concise", "concisely", "brief", "briefly", "summary",
                "summarize", "summarized", "summaries"},
    "suspect": {"suspect", "suspects", "suspected", "suspecting", "suspicion"},
    "witness": {"witness", "witnesses", "witnessed", "witnessing"},
    "evidence": {"evidence", "evidences", "evidentiary"},
    "verify": {"verify", "verified", "verifying", "verification"},
    "dispute": {"dispute", "disputed", "disputing", "disputes"},
    "investigate": {"investigate", "investigates", "investigated",
                    "investigating", "investigation", "investigations"},
    "conclude": {"conclude", "concludes", "concluded", "concluding",
                 "conclusion", "conclusions"},
}


class KeywordRetriever:
    """Ranks memories by query-term overlap, importance, and recency.

    The MemoryStore keeps an instance of this class as its retriever. Keep
    the retrieve() signature stable so an embedding retriever can replace it
    later without changing callers.
    """

    IMPORTANCE_WEIGHT = {
        MemoryImportance.HIGH: 2.0,
        MemoryImportance.MEDIUM: 1.4,
        MemoryImportance.LOW: 1.0,
    }

    def __init__(self, stop_words=None):
        self.stop_words = set(stop_words) if stop_words else STOP_WORDS

    def retrieve(self, query, memories, top_k=5):
        """Return the best-matching memories, already active-filtered.

        Score = number of matching query terms x importance weight x
        recency factor. Only memories with at least one term match rank.
        """
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        results = []
        for memory in memories:
            content_terms = set(self._tokenize(memory.content))
            tag_terms = set(self._tokenize(" ".join(memory.tags)))
            usable_terms = content_terms | tag_terms

            overlap = sum(1 for term in query_terms
                          if self._matches(term, usable_terms))
            if overlap == 0:
                continue

            score = (
                overlap
                * self.IMPORTANCE_WEIGHT.get(memory.importance, 1.0)
                * self._recency_factor(memory)
            )
            results.append((score, memory))

        results.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in results[:top_k]]

    def _matches(self, term, usable_terms):
        """True if a query term relates to any memory term.

        Matches in four dependency-free ways:
          1. Exact word ("prototype" == "prototype").
          2. Same word family ("explain" ~ "explanations").
          3. Prefix ("prefer" ~ "prefers").
          4. Substring of 5+ chars ("console" ~ "consoles").
        """
        if term in usable_terms:
            return True

        family = WORD_FAMILIES.get(term)
        if family and any(candidate in family for candidate in usable_terms):
            return True

        for candidate in usable_terms:
            if len(term) >= 4 and len(candidate) >= 4:
                if term.startswith(candidate) or candidate.startswith(term):
                    return True
            if len(term) >= 5 and len(candidate) >= 5:
                if term in candidate or candidate in term:
                    return True
        return False

    def _tokenize(self, text):
        """Lowercase, strip punctuation, split, and drop stop words."""
        words = re.findall(r"[a-z0-9]+", (text or "").lower())
        return [word for word in words if word not in self.stop_words]

    def _recency_factor(self, memory):
        """Return a multiplier between 0 and 1 favouring recent memories."""
        try:
            updated = datetime.fromisoformat(memory.updated_at)
        except (TypeError, ValueError):
            return 1.0

        days_old = max(0.0, (datetime.now(timezone.utc) - updated).total_seconds() / 86400)
        if days_old < 7:
            return 1.0
        if days_old < 30:
            return 0.9
        if days_old < 180:
            return 0.75
        return 0.6


# ── MemoryStore ───────────────────────────────────────────────────────────
# Owns memory/memories.json. Every public method returns a
# (result, message) tuple so callers can branch without try/except, except
# get() / list_all() which return plain data like CaseManager does.

class MemoryStore:
    def __init__(self, storage_dir=_DEFAULT_MEMORY_DIR, filename="memories.json"):
        self.storage_dir = storage_dir
        self.filename = filename
        self.memories = {}          # id string → Memory object
        self.retriever = KeywordRetriever()
        os.makedirs(storage_dir, exist_ok=True)
        self._load()

    # ── public API ────────────────────────────────────────────────────────

    def add(self, content, memory_type=MemoryType.SEMANTIC,
            source="", importance=MemoryImportance.MEDIUM,
            case_id="", confidence="", tags=None):
        """Create and persist a new memory."""
        content = content.strip()
        if not content:
            return None, "Error: Memory content cannot be empty."

        memory_type = self._coerce_type(memory_type)
        importance = self._coerce_importance(importance)
        if memory_type is None or importance is None:
            return (None, "Error: Invalid memory type or importance value.")

        confidence = self._coerce_confidence(confidence)
        if confidence is None:
            return None, "Error: Confidence must be LOW, MEDIUM, or HIGH."

        if tags is None:
            tags = self._derive_tags(content)

        memory = Memory(
            id=self._next_id(),
            type=memory_type,
            content=content,
            source=source.strip(),
            importance=importance,
            case_id=case_id.strip().upper(),
            confidence=confidence,
            tags=[tag for tag in tags if tag],
        )

        self.memories[memory.id] = memory
        self._save()
        return memory, f"Stored {memory.id} [{memory.type.value}]"

    def get(self, memory_id):
        """Look up a memory by ID. Returns the Memory or None."""
        return self.memories.get(memory_id.upper())

    def list_all(self):
        """Return every memory, newest first (including forgotten ones)."""
        return sorted(self.memories.values(), key=lambda m: m.created_at, reverse=True)

    def search(self, query, top_k=5, case_id=None, memory_type=None):
        """Find active memories that best match a query.

        Optional filters narrow the pool: case_id restricts to memories for
        one case; memory_type restricts to one kind of memory.
        Returns (list_of_Memory, message).
        """
        if not query or not query.strip():
            return [], "Please provide a search query."

        pool = [m for m in self.memories.values() if m.active]

        if case_id:
            pool = [m for m in pool if m.case_id == case_id.strip().upper()]

        memory_type = self._coerce_type(memory_type) if memory_type else None
        if memory_type:
            pool = [m for m in pool if m.type == memory_type]

        results = self.retriever.retrieve(query, pool, top_k=top_k)

        if not results:
            return [], "No relevant memories found."
        return results, f"Found {len(results)} relevant memory/memories."

    def recall(self, query, top_k=5, case_id=None, memory_type=None):
        """Return a formatted, LLM-friendly block of relevant memories.

        The block carries a clear warning: past memories are historical
        context and are NOT current evidence. Returns (block_or_None,
        message).
        """
        results, message = self.search(query, top_k=top_k,
                                       case_id=case_id, memory_type=memory_type)
        if not results:
            return None, message
        return self._format_recall(results), message

    def update(self, memory_id, content=None, memory_type=None,
               source=None, importance=None, confidence=None,
               case_id=None, tags=None):
        """Change fields on a memory. Only non-None fields change."""
        memory = self.get(memory_id)
        if not memory:
            return None, f"Error: Memory {memory_id} not found."

        if content is not None:
            clean = content.strip()
            if not clean:
                return None, "Error: Memory content cannot be empty."
            memory.content = clean

        if memory_type is not None:
            value = self._coerce_type(memory_type)
            if value is None:
                return None, "Error: Invalid memory type."
            memory.type = value

        if source is not None:
            memory.source = source.strip()

        if importance is not None:
            value = self._coerce_importance(importance)
            if value is None:
                return None, "Error: Invalid importance value."
            memory.importance = value

        if confidence is not None:
            value = self._coerce_confidence(confidence)
            if value is None:
                return None, "Error: Confidence must be LOW, MEDIUM, or HIGH."
            memory.confidence = value

        if case_id is not None:
            memory.case_id = case_id.strip().upper()

        if tags is not None:
            memory.tags = [tag for tag in tags if tag]

        memory.updated_at = _now_iso()
        self._save()
        return memory, f"Updated {memory.id}."

    def forget(self, memory_id):
        """Soft-delete a memory: it stops appearing in search/recall but
        still exists on disk with active=False."""
        memory = self.get(memory_id)
        if not memory:
            return None, f"Error: Memory {memory_id} not found."

        memory.active = False
        memory.updated_at = _now_iso()
        self._save()
        return memory, f"Forgot {memory.id}."

    def link(self, source_id, kind, target_id):
        """Connect one memory to a case, evidence, conversation, or memory.

        kind is one of "memory", "case", "evidence", "conversation". The
        target does NOT need to exist yet (e.g. a not-yet-created case).
        """
        memory = self.get(source_id)
        if not memory:
            return None, f"Error: Memory {source_id} not found."

        kind = kind.strip().lower()
        if kind not in ("memory", "case", "evidence", "conversation"):
            return None, f"Error: Link kind must be memory, case, evidence, or conversation."

        target_id = target_id.strip().upper()
        new_link = {"kind": kind, "target_id": target_id}

        if new_link not in memory.links:
            memory.links.append(new_link)
            memory.updated_at = _now_iso()
            self._save()

        return memory, f"Linked {source_id} → {kind}:{target_id}."

    # ── recall formatting ─────────────────────────────────────────────────

    def _format_recall(self, memories):
        """Turn ranked memories into the system-context block for the LLM."""
        lines = [
            "RELEVANT PAST MEMORIES",
            "These memories are historical context and may be outdated.",
            "They are not current case evidence.",
            "",
        ]
        for memory in memories:
            date = (memory.created_at or "")[:10] or "?"
            ref = memory.case_id or memory.source or "—"
            lines.append(
                f"[{memory.importance.value} · {memory.type.value} · {date} · {ref}]"
            )
            lines.append(memory.content.strip())
            lines.append("")

        return "\n".join(lines).rstrip()

    # ── internals ─────────────────────────────────────────────────────────

    def _filepath(self):
        return os.path.join(self.storage_dir, self.filename)

    def _load(self):
        """Read all memories from disk. Missing/corrupt files = empty."""
        self.memories = {}
        if not os.path.exists(self._filepath()):
            return

        try:
            with open(self._filepath(), "r", encoding="utf-8") as f:
                data = json.load(f)
            for raw in data.get("memories", []):
                memory = Memory.from_dict(raw)
                self.memories[memory.id] = memory
        except (json.JSONDecodeError, KeyError, ValueError):
            # Corrupt store: start empty rather than crashing.
            self.memories = {}

    def _save(self):
        """Write every memory to disk."""
        payload = {
            "memories": [m.to_dict() for m in self.memories.values()],
        }
        with open(self._filepath(), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _next_id(self):
        """Generate the next sequential memory ID (MEM-001, MEM-002...)."""
        existing = []
        for memory_id in self.memories:
            match = re.match(r"MEM-(\d+)", memory_id)
            if match:
                existing.append(int(match.group(1)))
        return f"MEM-{max(existing, default=0) + 1:03d}"

    @staticmethod
    def _derive_tags(content):
        """Pull a handful of keyword tags out of the content."""
        words = re.findall(r"[a-z0-9]+", (content or "").lower())
        seen = []
        for word in words:
            if word not in STOP_WORDS and word not in seen:
                seen.append(word)
            if len(seen) == 5:
                break
        return seen

    @staticmethod
    def _coerce_type(value):
        if isinstance(value, MemoryType):
            return value
        try:
            return MemoryType(str(value).upper())
        except ValueError:
            return None

    @staticmethod
    def _coerce_importance(value):
        if isinstance(value, MemoryImportance):
            return value
        try:
            return MemoryImportance(str(value).upper())
        except ValueError:
            return None

    @staticmethod
    def _coerce_confidence(value):
        clean = (value or "").strip().upper()
        if clean in ("", "LOW", "MEDIUM", "HIGH"):
            return clean
        return None