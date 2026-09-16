"""Test suite for AgentKafle's local account + chat-history system.

Runs against a throwaway SQLite file in a temp directory — no real user data,
no Ollama/Gemini, and no GUI. Run with:

    .venv/bin/python test_chat_history.py

or discover it with pytest. Covers:

- Authentication: create/login, duplicate rejection, hashing (no plaintext),
  correct/incorrect password, empty credentials, case-insensitive usernames.
- Conversations: create, list, get, rename, delete, add message,
  chronological retrieval, auto-title.
- Isolation: User A can never read/write/rename/delete User B's chats.
- Persistence: close and reopen the database, everything is still there.
- Agent integration: a new chat has empty context; an old chat restores its
  history into the prompt the model receives; new messages persist.
"""

import os
import tempfile

import agent as agent_mod
from database import Database
from auth import AuthError, UserManager
from history import ChatStore


# ── Helpers ──────────────────────────────────────────────────────────────

def _temp_db():
    """Create a Database in a fresh temp directory."""
    directory = tempfile.mkdtemp(prefix="agentkafle_test_")
    return Database(os.path.join(directory, "agentkafle.db"))


class FakeProvider:
    """Provider stand-in that records the last prompt it received."""

    def __init__(self, name="fake", label="FAKE", model="fake-model"):
        self.name = name
        self.label = label
        self.model = model
        self.last_messages = None

    def chat(self, messages, json_mode=False):
        self.last_messages = messages
        return "FAKE REPLY"

    def chat_json(self, messages):
        self.last_messages = messages
        return (
            '{"summary": "x", "established_facts": [], '
            '"unverified_claims": [], "disputed_information": [], '
            '"contradictions": [], "hypotheses": [], "conclusion": "x", '
            '"confidence": "LOW", "missing_information": []}'
        )

    def complete(self, prompt):
        return "FAKE REPLY"

    def close(self):
        pass


class FakeRouter:
    """Minimal ProviderRouter stand-in for agent-level tests."""

    def __init__(self):
        self.provider = FakeProvider()

    @property
    def name(self):
        return self.provider.name

    @property
    def label(self):
        return self.provider.label

    @property
    def model(self):
        return self.provider.model

    def chat(self, messages):
        return self.provider.chat(messages)

    def chat_json(self, messages):
        return self.provider.chat_json(messages)

    def complete(self, prompt):
        return self.provider.complete(prompt)

    def switch(self, provider, model=None):
        self.provider = FakeProvider(name=provider, label=provider.upper())
        return self.provider

    def configured_providers(self):
        return {
            "fake": {"label": "FAKE", "default_model": "fake-model",
                     "env_keys": (), "configured": True},
        }

    def available_models(self):
        return []

    def close(self):
        pass


def _prompt_text(messages):
    """Join the LLM prompt message contents for easy assertions."""
    return "\n".join(m.get("content", "") for m in messages)


# ── 1. Authentication ────────────────────────────────────────────────────

def test_auth():
    db = _temp_db()
    users = UserManager(db)

    # Create a user and log back in with the correct password.
    alice = users.create_user("Alice", "hunter2")
    assert alice.id is not None
    assert alice.username == "Alice"
    assert users.authenticate("Alice", "hunter2").id == alice.id

    # Usernames are case-insensitive for matching.
    assert users.authenticate("alice", "hunter2").id == alice.id
    assert users.authenticate("ALICE", "hunter2").id == alice.id

    # Duplicate usernames are rejected, regardless of case.
    for dup in ("Alice", "alice", "ALICE", "aLiCe"):
        try:
            users.create_user(dup, "another-password")
        except AuthError:
            pass
        else:
            raise AssertionError(f"duplicate username {dup!r} was accepted")

    # Wrong password fails; the message never reveals which part was wrong.
    for wrong in ("Hunter2", "hunter3", ""):
        try:
            users.authenticate("Alice", wrong)
        except AuthError as exc:
            msg = str(exc).lower()
            assert "invalid" in msg or "password" in msg or "enter both" in msg
        else:
            raise AssertionError(f"wrong password {wrong!r} was accepted")

    # Empty credentials are rejected on login.
    try:
        users.authenticate("", "password1")
    except AuthError:
        pass
    else:
        raise AssertionError("empty username was accepted on login")

    # Empty credentials are rejected on signup.
    for username, password in (("", "pass"), ("alice", ""), ("   ", "pass")):
        try:
            users.create_user(username, password)
        except AuthError:
            pass
        else:
            raise AssertionError(f"signup ({username!r}, {password!r}) succeeded")

    # The stored hash must NOT contain the plaintext password, and the salt
    # and hash must be distinct values.
    row = db.query_one(
        "SELECT password_hash, password_salt FROM users WHERE username = ?",
        ("Alice",),
    )
    assert row["password_hash"] != "hunter2"
    assert "hunter2" not in row["password_hash"]
    assert row["password_salt"] != "hunter2"
    assert row["password_hash"] != row["password_salt"]

    # Two accounts get different salts/hashes even for the same password.
    bob = users.create_user("Bob", "shared-pass")
    row_alice = db.query_one(
        "SELECT password_hash, password_salt FROM users WHERE username = 'Alice'"
    )
    row_bob = db.query_one(
        "SELECT password_hash, password_salt FROM users WHERE username = 'Bob'"
    )
    assert row_alice["password_salt"] != row_bob["password_salt"]
    assert row_alice["password_hash"] != row_bob["password_hash"]
    db.close()
    print("OK: authentication")


# ── 2. Conversations ─────────────────────────────────────────────────────

def test_conversations():
    db = _temp_db()
    users = UserManager(db)
    chats = ChatStore(db)
    user = users.create_user("Carol", "secret")

    # Create, list, get.
    conv1 = chats.create_conversation(user.id)
    conv2 = chats.create_conversation(user.id, "Research")
    assert conv1.id != conv2.id
    listed = chats.list_conversations(user.id)
    assert len(listed) == 2
    assert chats.get_conversation(user.id, conv1.id).id == conv1.id

    # Rename.
    assert chats.rename_conversation(user.id, conv1.id, "AgentKafle Project")
    assert chats.get_conversation(user.id, conv1.id).title == "AgentKafle Project"

    # Add messages; retrieve in chronological order.
    m1 = chats.add_message(user.id, conv1.id, "user", "first")
    m2 = chats.add_message(user.id, conv1.id, "assistant", "second")
    m3 = chats.add_message(user.id, conv1.id, "user", "third")
    msgs = chats.list_messages(user.id, conv1.id)
    assert [m.content for m in msgs] == ["first", "second", "third"]
    assert [m.role for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0].id == m1.id and msgs[-1].id == m3.id
    assert chats.message_count(user.id, conv1.id) == 3

    # Delete removes the conversation and (via FK cascade) its messages.
    assert chats.delete_conversation(user.id, conv1.id)
    assert chats.get_conversation(user.id, conv1.id) is None
    assert chats.list_messages(user.id, conv1.id) == []
    assert chats.message_count(user.id, conv1.id) == 0
    assert chats.list_conversations(user.id)[0].id == conv2.id

    # Auto-title: a fresh chat takes its title from the first user message.
    conv3 = chats.create_conversation(user.id)
    assert chats.message_count(user.id, conv3.id) == 0
    chats.add_message(user.id, conv3.id, "user", "Tell me about the agent")
    assert chats.set_title_from_first_message(user.id, conv3.id, "Tell me about the agent")
    assert chats.get_conversation(user.id, conv3.id).title == "Tell me about the agent"
    db.close()
    print("OK: conversations")


# ── 3. Isolation between users ───────────────────────────────────────────

def test_isolation():
    db = _temp_db()
    users = UserManager(db)
    chats = ChatStore(db)

    alice = users.create_user("Alice", "pw-a")
    bob = users.create_user("Bob", "pw-b")
    conv = chats.create_conversation(alice.id, "Alice's private chat")
    chats.add_message(alice.id, conv.id, "user", "secret plans")

    # B never sees A's conversations.
    assert chats.list_conversations(bob.id) == []
    assert chats.get_conversation(bob.id, conv.id) is None
    assert chats.message_count(bob.id, conv.id) == 0
    assert chats.list_messages(bob.id, conv.id) == []

    # B cannot rename, delete, or write into A's conversation.
    assert chats.rename_conversation(bob.id, conv.id, "hijacked") is False
    assert chats.get_conversation(alice.id, conv.id).title == "Alice's private chat"
    assert chats.delete_conversation(bob.id, conv.id) is False
    assert chats.get_conversation(alice.id, conv.id) is not None
    assert chats.add_message(bob.id, conv.id, "user", "intrusion") is None

    # A's data is untouched and B's attempts produced no rows.
    msgs = chats.list_messages(alice.id, conv.id)
    assert len(msgs) == 1
    assert msgs[0].content == "secret plans"

    # The reverse direction also holds.
    b_conv = chats.create_conversation(bob.id, "Bob's chat")
    assert chats.get_conversation(alice.id, b_conv.id) is None
    assert chats.delete_conversation(alice.id, b_conv.id) is False
    assert chats.add_message(alice.id, b_conv.id, "user", "sneak") is None
    db.close()
    print("OK: isolation")


# ── 4. Persistence across close/reopen ───────────────────────────────────

def test_persistence():
    directory = tempfile.mkdtemp(prefix="agentkafle_persist_")
    path = os.path.join(directory, "agentkafle.db")

    # First "session".
    db1 = Database(path)
    users1 = UserManager(db1)
    chats1 = ChatStore(db1)
    user = users1.create_user("Dana", "rock-solid")
    conv = chats1.create_conversation(user.id, "Persistent Chat")
    chats1.add_message(user.id, conv.id, "user", "hello there")
    chats1.add_message(user.id, conv.id, "assistant", "howdy")
    db1.close()

    # Reopen as if the app had been closed and restarted.
    db2 = Database(path)
    users2 = UserManager(db2)
    chats2 = ChatStore(db2)

    again = users2.authenticate("DANA", "rock-solid")   # case-insensitive login
    assert again.id == user.id
    listed = chats2.list_conversations(again.id)
    assert len(listed) == 1
    assert listed[0].title == "Persistent Chat"
    msgs = chats2.list_messages(again.id, listed[0].id)
    assert [m.content for m in msgs] == ["hello there", "howdy"]
    db2.close()
    print("OK: persistence")


# ── 5. Agent integration ─────────────────────────────────────────────────

def test_agent_integration():
    db = _temp_db()
    users = UserManager(db)
    chats = ChatStore(db)
    user = users.create_user("Erin", "pw")

    # A brand-new chat starts with empty context.
    fresh_conv = chats.create_conversation(user.id, "Fresh")
    assert chats.list_messages(user.id, fresh_conv.id) == []
    agent = agent_mod.AgentKafle(router=FakeRouter())
    reply = agent.respond("hello", history=[])
    assert reply == "FAKE REPLY"
    prompt = _prompt_text(agent.router.provider.last_messages)
    assert "old context" not in prompt.lower()

    # An old chat restores its history into the model's prompt.
    old_conv = chats.create_conversation(user.id, "Old Chat")
    chats.add_message(user.id, old_conv.id, "user", "What was the password?")
    chats.add_message(user.id, old_conv.id, "assistant", "The bolt is #4821")
    stored = chats.list_messages(user.id, old_conv.id)
    history = [("agent" if m.role == "assistant" else "user", m.content) for m in stored]

    agent2 = agent_mod.AgentKafle(router=FakeRouter())
    reply = agent2.respond("continue", history=history)
    assert reply == "FAKE REPLY"
    prompt = _prompt_text(agent2.router.provider.last_messages)
    assert "What was the password?" in prompt
    assert "The bolt is #4821" in prompt

    # New messages are persisted to the open conversation.
    chats.add_message(user.id, old_conv.id, "user", "thanks")
    chats.add_message(user.id, old_conv.id, "assistant", "anytime")
    after = chats.list_messages(user.id, old_conv.id)
    assert len(after) == len(stored) + 2
    assert after[-1].role == "assistant" and after[-1].content == "anytime"
    db.close()
    print("OK: agent integration")


# ── 6. Provider switching keeps the conversation intact (backend layer) ──

def test_provider_switch_keeps_history():
    db = _temp_db()
    users = UserManager(db)
    chats = ChatStore(db)
    user = users.create_user("Finn", "pw")

    conv = chats.create_conversation(user.id, "Switch Chat")
    chats.add_message(user.id, conv.id, "user", "before")
    chats.add_message(user.id, conv.id, "assistant", "first reply")

    agent = agent_mod.AgentKafle(router=FakeRouter())
    agent.set_provider("gemini")
    assert agent.router.name == "gemini"
    chat = chats.get_conversation(user.id, conv.id)
    assert chat is not None and chat.title == "Switch Chat"

    # Provider switches never touch the stored conversation.
    agent.set_provider("ollama")
    agent.set_provider("gemini")
    msgs = chats.list_messages(user.id, conv.id)
    assert [m.content for m in msgs] == ["before", "first reply"]
    db.close()
    print("OK: provider switch preserves history")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_auth()
    test_conversations()
    test_isolation()
    test_persistence()
    test_agent_integration()
    test_provider_switch_keeps_history()
    print("OK: all chat-history checks passed")