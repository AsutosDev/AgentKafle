"""Local user accounts and session state for AgentKafle.

This module is a small dedicated authentication layer. It contains NO GUI
code: the GUI only calls UserManager.create_user() / authenticate() and reads
the Session object.

Security notes:
- Passwords are hashed with PBKDF2-HMAC-SHA256 and a unique random salt.
  The raw password is never stored or returned anywhere.
- Password verification uses hmac.compare_digest (constant-time).
- All SQL is parameterized; usernames are matched case-insensitively.
- Passwords and password hashes are never printed or logged.
"""

import hashlib
import hmac
import secrets
import sqlite3

from database import Database, now_iso

# PBKDF2 iteration count. OWASP's minimum for PBKDF2-HMAC-SHA256 is 600k, but
# 200k keeps login snappy on modest hardware while still being robust for a
# local desktop app. Raise this constant any time; old hashes are re-verified
# against the salt stored with them, so nothing breaks.
PBKDF2_ITERATIONS = 200_000
_SALT_BYTES = 16


class AuthError(Exception):
    """Raised for any user-facing authentication problem.

    Messages are intentionally generic (e.g. "Invalid username or password")
    so a response never reveals whether a username exists.
    """


class User:
    """A single account. Never contains the password or its hash."""

    def __init__(self, user_id, username, created_at):
        self.id = user_id
        self.username = username
        self.created_at = created_at

    @classmethod
    def from_row(cls, row):
        return cls(
            user_id=row["id"],
            username=row["username"],
            created_at=row["created_at"],
        )

    def __repr__(self):
        return f"User(id={self.id!r}, username={self.username!r})"


def _hash_password(password: str, salt_hex=None):
    """Hash a password with PBKDF2-HMAC-SHA256.

    Returns (salt_hex, hash_hex). With a salt_hex supplied this rehashes the
    same salt (used during verification); otherwise a fresh random salt is
    created.
    """
    salt = secrets.token_bytes(_SALT_BYTES) if salt_hex is None else bytes.fromhex(salt_hex)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return salt.hex(), digest.hex()


def _verify_password(password: str, stored_hash: str, salt_hex: str) -> bool:
    """Constant-time check of password against a stored hash + salt."""
    _, computed_hash = _hash_password(password, salt_hex)
    return hmac.compare_digest(computed_hash, stored_hash)


class UserManager:
    """Create and authenticate users against the SQLite database."""

    def __init__(self, db: Database):
        self.db = db

    # ── creation ──

    def create_user(self, username, password) -> User:
        """Create a new account. Logs the caller in implicitly on success.

        Raises AuthError for empty/invalid input or a duplicated username.
        """
        username = (username or "").strip()
        if not username:
            raise AuthError("Username cannot be empty.")
        if len(username) > 50:
            raise AuthError("Username must be 50 characters or fewer.")

        if not password:
            raise AuthError("Password cannot be empty.")

        # Never store the plaintext password — only a salted PBKDF2 hash.
        salt_hex, hash_hex = _hash_password(password)

        try:
            with self.db.transaction() as conn:
                cur = conn.execute(
                    "INSERT INTO users (username, password_hash, password_salt, created_at)"
                    " VALUES (?, ?, ?, ?)",
                    (username, hash_hex, salt_hex, now_iso()),
                )
                user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            # The unique (case-insensitive) username index fired.
            raise AuthError(
                "That username is already taken. Choose a different one."
            )

        row = self.db.query_one(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        return User.from_row(row)

    # ── authentication ──

    def authenticate(self, username, password) -> User:
        """Verify credentials. Raises AuthError on any failure.

        Usernames are matched case-insensitively, so logging in as "alice"
        works for the account created as "Alice".
        """
        username = (username or "").strip()
        if not username or not password:
            raise AuthError("Please enter both a username and a password.")

        row = self.db.query_one(
            "SELECT id, username, password_hash, password_salt, created_at"
            " FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        )
        if row is None:
            raise AuthError("Invalid username or password.")

        if not _verify_password(password, row["password_hash"], row["password_salt"]):
            raise AuthError("Invalid username or password.")

        return User.from_row(row)

    # ── lookups ──

    def get_user(self, user_id) -> User:
        """Look up an account by numeric id. Returns None if missing."""
        row = self.db.query_one(
            "SELECT id, username, created_at FROM users WHERE id = ?",
            (user_id,),
        )
        return User.from_row(row) if row else None

    def user_exists(self, username) -> bool:
        """True if a username exists (case-insensitive)."""
        row = self.db.query_one(
            "SELECT id FROM users WHERE username = ? COLLATE NOCASE",
            ((username or "").strip(),),
        )
        return row is not None


class Session:
    """Application-level login + active-chat state.

    Explicitly does NOT hold the plaintext password or any credential — only
    the user_id/username and which conversation is currently open.
    """

    def __init__(self):
        self.user_id = None
        self.username = None
        self.active_conversation_id = None

    @property
    def logged_in(self) -> bool:
        return self.user_id is not None

    def start(self, user: User):
        """Begin a session for a user (clears the previous one)."""
        self.user_id = user.id
        self.username = user.username
        self.active_conversation_id = None

    def clear(self):
        """End the session: no user, no active conversation."""
        self.user_id = None
        self.username = None
        self.active_conversation_id = None

    def set_active_conversation(self, conversation_id):
        """Point the session at an open conversation (or None to detach)."""
        self.active_conversation_id = conversation_id