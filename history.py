"""Conversation (chat history) persistence for AgentKafle.

Every method that touches a conversation requires the owning ``user_id`` and
checks ownership against a ``WHERE id = ? AND user_id = ?`` query before
reading, writing, renaming, or deleting. A caller can therefore never reach
another user's conversation by guessing an ID — the lookup simply returns
None/False instead.

This is CHAT HISTORY, not AgentKafle's long-term memory. Agent memory lives in
memory.py (MemoryStore) and is a separate, per-user file. This module only
persists conversations and their chronological messages.
"""

from database import now_iso

DEFAULT_CONVERSATION_TITLE = "New Chat"
# When a conversation's first user message arrives and it still has the
# default title, the title is auto-set to that message, truncated to this
# many characters (nice-to-have, keeps the sidebar readable).
AUTO_TITLE_LIMIT = 40


class Conversation:
    """A single chat thread belonging to one user."""

    def __init__(self, conversation_id, user_id, title, created_at, updated_at):
        self.id = conversation_id
        self.user_id = user_id
        self.title = title
        self.created_at = created_at
        self.updated_at = updated_at

    @classmethod
    def from_row(cls, row):
        return cls(
            conversation_id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def __repr__(self):
        return f"Conversation(id={self.id!r}, user_id={self.user_id!r}, title={self.title!r})"


class Message:
    """One message inside a conversation."""

    def __init__(self, message_id, conversation_id, role, content, created_at):
        self.id = message_id
        self.conversation_id = conversation_id
        self.role = role           # "user" or "assistant"
        self.content = content
        self.created_at = created_at

    @classmethod
    def from_row(cls, row):
        return cls(
            message_id=row["id"],
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
        )

    def __repr__(self):
        return f"Message(id={self.id!r}, role={self.role!r})"


class ChatStore:
    """CRUD for conversations and messages, scoped to an owning user."""

    def __init__(self, db):
        self.db = db

    # ── ownership guard ──
    # Every public method funnels through _owns() so foreign conversation ids
    # can never leak another user's data.

    def _owns(self, user_id, conversation_id) -> bool:
        row = self.db.query_one(
            "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        return row is not None

    # ── conversations ──

    def create_conversation(self, user_id, title=DEFAULT_CONVERSATION_TITLE) -> Conversation:
        """Create a new, empty conversation for a user."""
        now = now_iso()
        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (user_id, title, created_at, updated_at)"
                " VALUES (?, ?, ?, ?)",
                (user_id, title, now, now),
            )
            conversation_id = cur.lastrowid
        # Return the canonical row via get_conversation (which checks ownership).
        return self.get_conversation(user_id, conversation_id)

    def list_conversations(self, user_id):
        """Return the user's conversations, most recently updated first."""
        rows = self.db.query_all(
            "SELECT id, user_id, title, created_at, updated_at"
            " FROM conversations WHERE user_id = ?"
            " ORDER BY updated_at DESC, id DESC",
            (user_id,),
        )
        return [Conversation.from_row(row) for row in rows]

    def get_conversation(self, user_id, conversation_id):
        """Return one conversation, but only if it belongs to the user."""
        row = self.db.query_one(
            "SELECT id, user_id, title, created_at, updated_at"
            " FROM conversations WHERE id = ? AND user_id = ?",
            (conversation_id, user_id),
        )
        return Conversation.from_row(row) if row else None

    def rename_conversation(self, user_id, conversation_id, title) -> bool:
        """Rename a conversation owned by the user. Returns False if not owned."""
        title = (title or "").strip()
        if not title:
            return False
        with self.db.transaction() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ?"
                " WHERE id = ? AND user_id = ?",
                (title, now_iso(), conversation_id, user_id),
            )
        return cur.rowcount > 0

    def delete_conversation(self, user_id, conversation_id) -> bool:
        """Delete a conversation (and its messages via CASCADE).

        Returns False if the conversation does not belong to the user.
        """
        with self.db.transaction() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            )
        return cur.rowcount > 0

    # ── messages ──

    def add_message(self, user_id, conversation_id, role, content) -> Message:
        """Append a message to a conversation owned by the user.

        The message insert and the conversation's updated_at bump happen in
        one transaction, so a crash cannot leave the pair half-updated.

        Returns the stored Message, or None if the user does not own the
        conversation.
        """
        if not self._owns(user_id, conversation_id):
            return None

        content = "" if content is None else str(content)
        now = now_iso()

        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at)"
                " VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
        message_id = cur.lastrowid

        row = self.db.query_one(
            "SELECT id, conversation_id, role, content, created_at"
            " FROM messages WHERE id = ?",
            (message_id,),
        )
        return Message.from_row(row)

    def list_messages(self, user_id, conversation_id):
        """Return the conversation's messages in chronological order.

        Returns an empty list if the user does not own the conversation.
        """
        if not self._owns(user_id, conversation_id):
            return []

        rows = self.db.query_all(
            "SELECT id, conversation_id, role, content, created_at"
            " FROM messages WHERE conversation_id = ?"
            " ORDER BY created_at ASC, id ASC",
            (conversation_id,),
        )
        return [Message.from_row(row) for row in rows]

    def message_count(self, user_id, conversation_id) -> int:
        """Number of messages in a user's conversation (0 if not owned)."""
        if not self._owns(user_id, conversation_id):
            return 0
        row = self.db.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        return int(row["n"]) if row else 0

    def set_title_from_first_message(self, user_id, conversation_id, content) -> bool:
        """Auto-title a brand-new chat from its first user message.

        Only fires while the conversation still has DEFAULT_CONVERSATION_TITLE;
        afterwards the title is left alone.
        """
        conversation = self.get_conversation(user_id, conversation_id)
        if conversation is None or conversation.title != DEFAULT_CONVERSATION_TITLE:
            return False

        title = " ".join((content or "").split())
        if not title:
            return False
        if len(title) > AUTO_TITLE_LIMIT:
            title = title[:AUTO_TITLE_LIMIT].rstrip() + "…"

        return self.rename_conversation(user_id, conversation_id, title)