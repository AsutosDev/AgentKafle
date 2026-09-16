"""SQLite data layer for AgentKafle.

This module owns the connection to the local SQLite database and the schema.
It is the ONLY module that speaks SQL directly — auth.py and history.py are
thin, plain-Python APIs on top of it.

Storage layout (per-user, writable location):
    macOS:   ~/Library/Application Support/AgentKafle/agentkafle.db
    Windows: %APPDATA%\\AgentKafle\\agentkafle.db
    Linux:   ~/.local/share/AgentKafle/agentkafle.db

Why SQLite (and not more JSON files)?
- The app already persists cases and evidence as JSON. Conversations and
  messages are different: they are relational (users → conversations →
  messages), grow over time, and must be scoped per user. SQLite gives us
  real foreign keys, indexes, and safe transactions without adding a server.

Concurrency note:
- The GUI runs LLM calls on a background thread but does all of its database
  work on the main thread. To be safe regardless of caller, the connection is
  opened with check_same_thread=False and every public method guards access
  with a re-entrant lock.
- ``transaction()`` is an explicit BEGIN/COMMIT/ROLLBACK block so a crash in
  the middle of a multi-statement write can never leave the database half
  updated (e.g. a message inserted but the conversation timestamp not bumped).

Schema (created automatically on first open):
- users            — accounts with PBKDF2 hashes (never plaintext)
- conversations    — one chat thread per user
- messages         — individual user/assistant turns inside a conversation
"""

import os
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone


def _get_data_dir() -> str:
    """Return the platform-appropriate user data directory for AgentKafle.

    This is a writable location that persists across application updates.
    """
    if sys.platform == "darwin":
        # macOS: ~/Library/Application Support/AgentKafle
        base = os.path.expanduser("~/Library/Application Support")
    elif sys.platform == "win32":
        # Windows: %APPDATA%\AgentKafle
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        # Linux/other: ~/.local/share/AgentKafle (XDG)
        base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))

    data_dir = os.path.join(base, "AgentKafle")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


# Project root (the directory containing this file), used as fallback for
# development runs from source. In a packaged app (PyInstaller), sys._MEIPASS
# points to the bundle resources, but we want user data in the platform dir.
if getattr(sys, "frozen", False):
    # Running in a PyInstaller bundle
    _PROJECT_ROOT = os.path.dirname(sys.executable)
else:
    # Running from source
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Default fallback (used only in development)
_DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
DB_FILENAME = "agentkafle.db"


def get_data_dir() -> str:
    """Public accessor for the data directory."""
    return _get_data_dir()


def get_db_path() -> str:
    """Return the full path to the SQLite database file."""
    return os.path.join(get_data_dir(), DB_FILENAME)


def now_iso():
    """UTC timestamp in ISO format, used for created_at/updated_at columns."""
    return datetime.now(timezone.utc).isoformat()


# ── Schema ───────────────────────────────────────────────────────────────
# Usernames are matched case-insensitively (COLLATE NOCASE + unique index),
# so "Alice", "alice", and "ALICE" all refer to the same account.
# Foreign keys cascade: deleting a conversation deletes its messages, and
# deleting a user deletes their conversations.

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL COLLATE NOCASE,
    password_hash TEXT    NOT NULL,
    password_salt TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_username
    ON users(username COLLATE NOCASE);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT    NOT NULL DEFAULT 'New Chat',
    created_at TEXT    NOT NULL,
    updated_at TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id
    ON conversations(user_id);

CREATE INDEX IF NOT EXISTS idx_conversations_updated_at
    ON conversations(updated_at);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
    ON messages(conversation_id);
"""


class Database:
    """A single reusable SQLite connection with a small safe API.

    Usage:
        db = Database()                      # platform data dir/agentkafle.db
        user = UserManager(db).create_user("Alice", "secret")
        ChatStore(db).create_conversation(user.id)

    Public helpers:
        query_one(sql, params)      → one row as dict, or None
        query_all(sql, params)      → rows as list of dicts
        execute(sql, params)        → cursor (for single statements)
        transaction()               → context manager for atomic multi-statement writes
        close()                     → close the connection
    """

    def __init__(self, path=None):
        self.path = path or get_db_path()

        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Re-entrant lock so any thread may safely talk to the DB.
        self.lock = threading.RLock()

        # isolation_level=None puts SQLite in autocommit mode: statements
        # apply immediately unless wrapped in an explicit transaction(). This
        # keeps BEGIN/COMMIT/ROLLBACK fully under our control.
        self.conn = sqlite3.connect(
            self.path,
            isolation_level=None,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row

        with self.lock:
            # Foreign keys are per-connection and must be enabled explicitly.
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.executescript(_SCHEMA)

    # ── helpers ────────────────────────────────────────────────────────────

    def query_one(self, sql, params=()):
        """Return the first row of a SELECT as a dict, or None."""
        with self.lock:
            row = self.conn.execute(sql, params).fetchone()
            return dict(row) if row is not None else None

    def query_all(self, sql, params=()):
        """Return every row of a SELECT as a list of dicts."""
        with self.lock:
            return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def execute(self, sql, params=()):
        """Execute a single statement (autocommitted). Returns the cursor."""
        with self.lock:
            return self.conn.execute(sql, params)

    @contextmanager
    def transaction(self):
        """Run several statements atomically.

        On success the block is COMMITTED; on any exception it is ROLLED BACK
        and the exception re-raised, so a crash mid-write never corrupts the
        data.
        """
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def close(self):
        """Close the underlying connection."""
        with self.lock:
            self.conn.close()