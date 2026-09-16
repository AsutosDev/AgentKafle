"""AgentKafle — Detective AI desktop GUI.

Run with:
    python gui.py

The GUI now starts with a local-login screen (accounts are stored in SQLite,
passwords PBKDF2-hashed — see auth.py / database.py), then shows the main
Detective interface. Chat conversations are persisted per user automatically
(history.py) and survive restarts, while the existing provider selector,
active-case bar, Detective reasoning, evidence, and long-term memory keep
working unchanged.

Architecture:

  GUI → auth.UserManager        (login / create account)
     → history.ChatStore        (conversations, messages, auto-save)
     → agent.handle_command()   (fast, same thread)
     → agent.respond() via threading  (slow, background thread)
     → root.after()             (posts result back to main thread)

LLM calls run on a background thread so the interface stays responsive. The
SQLite layer (database.py) is thread-safe, and all chat-history writes happen
on the main thread via root.after()/command handlers.
"""

import os
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog

from agent import AgentKafle
from llm import ProviderConfigError, ProviderConnectionError
from auth import AuthError, Session, UserManager
from database import Database, get_data_dir
from history import ChatStore
from memory import MemoryStore


# ── Braille spinner frames ────────────────────────────────────────────────
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_DELAY_MS = 80

# Message bubbles never grow wider than this.
BUBBLE_MAX_FRACTION = 0.72
BUBBLE_ABSOLUTE_MAX = 640


# ── Provider metadata ────────────────────────────────────────────────────
# Single source of truth: the provider registry inside llm.py describes every
# provider (label, default model, configured status). The GUI consumes that
# metadata through the Agent's ProviderRouter instead of maintaining its own
# provider dict, so adding a provider is a one-file change (llm.py).

def _provider_info(agent):
    """Return provider key -> metadata dict from the Agent's router."""
    return agent.router.configured_providers()


class AgentKafleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AgentKafle — Detective AI")
        self.geometry("1050x680")
        self.minsize(760, 460)

        # Persistence layer (SQLite, created/opened automatically on start).
        self.db = Database()
        self.user_manager = UserManager(self.db)
        self.chat_store = ChatStore(self.db)

        # Current login + active-chat state (no credentials stored here).
        self.session = Session()

        # Provider metadata is (re)populated after login, when the agent exists.
        self._providers = {}

        # Conversation history of the OPEN conversation: list of (role, text)
        # tuples. role is "user" or "agent". Loaded from SQLite when a chat is
        # opened and passed to the LLM so the model sees the conversation.
        self._history = []

        self._input_placeholder = "Ask AgentKafle…"
        self._entry_is_placeholder = True
        self._busy = False            # True while an LLM call is running
        self._spinner_idx = 0         # current frame index
        self._spinner_id = None       # after() id so we can cancel it
        self._loading_conversations = False  # guard against listbox callbacks

        self._setup_style()
        self._show_login()

    # ── Style ─────────────────────────────────────────────────────────────

    def _setup_style(self):
        # One consistent contrast system. Every text colour is chosen so it
        # keeps roughly >= 4.5:1 contrast against the surface it sits on
        # (WCAG AA for normal text). The ratios are noted per token:
        #   on_accent  #ffffff on accent #2563eb  -> 5.2:1
        #   muted      #a5aebc on any panel       -> >= 6.3:1
        #   placeholder #9aa4b4 on surface        -> 6.4:1
        #   user_fg    #ffffff on user_bg        -> 5.7:1
        #   error      #ff7a7a on error_bg        -> 5.7:1
        self._c = {
            "bg":       "#17181c",   # window background
            "surface":  "#1e2126",   # raised panels (case bar, input box)
            "surface2": "#272b33",   # agent message bubble
            "border":   "#3b4252",   # hairlines and field outlines
            "text":     "#eceff4",   # primary text (15.4:1 on background)
            "muted":    "#a5aebc",   # secondary text (>= 6.3:1 everywhere)
            "placeholder": "#9aa4b4",  # input ghost text
            "accent":   "#2563eb",   # action colour — darker for contrast
            "accent_d": "#1d4ed8",   # accent pressed / hovered
            "accent_disabled": "#333a47",  # send button while busy
            "on_accent": "#ffffff",  # text drawn on the accent colour
            "user_bg":  "#2f5fd0",   # user message bubble
            "user_fg":  "#ffffff",
            "error_bg": "#3a2426",   # error message bubble
            "error":    "#ff7a7a",
            "status":   {            # case status → dot colour
                "OPEN":   "#4cd964",
                "SOLVED": "#5ac8fa",
                "CLOSED": "#aab2bd",
            },
        }
        self._f = {
            "title":   ("Helvetica Neue", 18, "bold"),
            "app":     ("Helvetica Neue", 13),
            "app_mid": ("Helvetica Neue", 12),
            "small":   ("Helvetica Neue", 11),
            "xsmall":  ("Helvetica Neue", 10),
            "eyebrow": ("Helvetica Neue", 9, "bold"),
            "badge":   ("Helvetica Neue", 11, "bold"),
        }
        self.configure(bg=self._c["bg"])

    # ── Styled button helper ──────────────────────────────────────────────
    # Rendered with a Label (not tk.Button) so colours match the rest of the
    # app on every platform (default buttons can ignore fg on macOS).

    def _make_button(self, parent, text, command, primary=False, small=False):
        if primary:
            bg, fg, hover = self._c["accent"], self._c["on_accent"], self._c["accent_d"]
        else:
            bg, fg, hover = self._c["surface2"], self._c["text"], self._c["border"]
        btn = tk.Label(
            parent, text=text, bg=bg, fg=fg,
            font=self._f["small"] if small else self._f["app"],
            padx=10, pady=6, cursor="hand2",
        )
        btn.bind("<Button-1>", lambda e: command())
        btn.bind("<Enter>", lambda e: btn.config(bg=hover))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    # ── Login screen ──────────────────────────────────────────────────────

    def _show_login(self):
        self._login_main = tk.Frame(self, bg=self._c["bg"])
        self._login_main.pack(fill=tk.BOTH, expand=True)
        self._login_main.columnconfigure(0, weight=1)
        self._login_main.rowconfigure(0, weight=1)

        card = tk.Frame(
            self._login_main, bg=self._c["surface"],
            highlightbackground=self._c["border"], highlightthickness=1,
        )
        card.grid(row=0, column=0)
        card.configure(padx=34, pady=28)

        tk.Label(card, text="AgentKafle", bg=self._c["surface"],
                 fg=self._c["text"], font=self._f["title"],
                 anchor="w").pack(anchor="w")
        tk.Label(card, text="Detective AI", bg=self._c["surface"],
                 fg=self._c["muted"], font=self._f["small"],
                 anchor="w").pack(anchor="w", pady=(1, 20))

        tk.Label(card, text="Username", bg=self._c["surface"],
                 fg=self._c["muted"], font=self._f["xsmall"],
                 anchor="w").pack(anchor="w")
        self._login_user = tk.Entry(
            card, bg=self._c["bg"], fg=self._c["text"],
            insertbackground=self._c["text"], font=self._f["app"],
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=self._c["border"],
            highlightcolor=self._c["accent"],
        )
        self._login_user.pack(fill=tk.X, ipady=7, pady=(3, 12))

        tk.Label(card, text="Password", bg=self._c["surface"],
                 fg=self._c["muted"], font=self._f["xsmall"],
                 anchor="w").pack(anchor="w")
        self._login_pass = tk.Entry(
            card, show="•", bg=self._c["bg"], fg=self._c["text"],
            insertbackground=self._c["text"], font=self._f["app"],
            relief=tk.FLAT, highlightthickness=1,
            highlightbackground=self._c["border"],
            highlightcolor=self._c["accent"],
        )
        self._login_pass.pack(fill=tk.X, ipady=7, pady=(3, 14))

        self._login_error_lbl = tk.Label(
            card, text="", bg=self._c["surface"], fg=self._c["error"],
            font=self._f["xsmall"], anchor="w", wraplength=300,
            justify="left",
        )
        self._login_error_lbl.pack(anchor="w", pady=(0, 10))

        buttons = tk.Frame(card, bg=self._c["surface"])
        buttons.pack(fill=tk.X)
        login_btn = self._make_button(buttons, "Login", self._do_login, primary=True)
        login_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        create_btn = self._make_button(buttons, "Create Account", self._do_create_account)
        create_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0))

        # Enter submits the login from either field.
        self._login_user.bind("<Return>", self._do_login)
        self._login_user.bind("<KP_Enter>", self._do_login)
        self._login_pass.bind("<Return>", self._do_login)
        self._login_pass.bind("<KP_Enter>", self._do_login)

        self._login_user.focus_set()

    def _set_login_error(self, message):
        self._login_error_lbl.config(text=message)

    def _do_login(self, _event=None):
        try:
            user = self.user_manager.authenticate(
                self._login_user.get(), self._login_pass.get()
            )
        except AuthError as exc:
            self._set_login_error(str(exc))
            self._login_pass.delete(0, tk.END)
            return
        except Exception:
            # Database-level failure: stay on the login screen, never crash.
            self._set_login_error("Could not log in right now. Please try again.")
            self._login_pass.delete(0, tk.END)
            return
        self._start_session(user)

    def _do_create_account(self, _event=None):
        try:
            user = self.user_manager.create_user(
                self._login_user.get(), self._login_pass.get()
            )
        except AuthError as exc:
            self._set_login_error(str(exc))
            return
        except Exception:
            self._set_login_error("Could not create the account right now. Please try again.")
            return
        # A brand-new account is logged in immediately.
        self._start_session(user)

    def _start_session(self, user):
        self.session.start(user)
        self._login_main.destroy()
        self._login_main = None

        # Per-user long-term memory (separate from chat history). The old
        # global memory/memories.json is left untouched on disk.
        memory_store = MemoryStore(
            storage_dir=os.path.join(get_data_dir(), "memories"),
            filename=f"user_{user.id}.json",
        )
        self.agent = AgentKafle(memory=memory_store)
        self._providers = _provider_info(self.agent)

        self._history = []
        self._conv_ids = []
        self._busy = False
        self._spinner_idx = 0
        self._spinner_id = None

        self._build_ui()
        self._refresh_case_info()
        self._refresh_conversation_list()
        self._append_agent(
            f"Hello, {user.username}. I am AgentKafle.\n"
            "Type 'help' to see commands, or ask me anything.\n"
        )
        self._entry.focus_set()

    # ── Logout ────────────────────────────────────────────────────────────

    def _logout(self):
        if self._busy:
            return
        self._stop_spinner()
        self.session.clear()
        self._history = []
        self._conv_ids = []
        self.agent = None
        self._providers = {}

        self._main.destroy()
        self._main = None
        self._show_login()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        self._build_sidebar()
        self._build_header()
        self._build_case_bar()
        self._build_chat()
        self._build_input()
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _build_sidebar(self):
        self._main = tk.Frame(self, bg=self._c["bg"])
        self._main.pack(fill=tk.BOTH, expand=True)

        self._sidebar = tk.Frame(self._main, bg=self._c["surface"], width=232)
        self._sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self._sidebar.pack_propagate(False)

        # Wordmark + user.
        tk.Label(self._sidebar, text="AgentKafle", bg=self._c["surface"],
                 fg=self._c["text"], font=self._f["title"],
                 anchor="w").pack(anchor="w", padx=14, pady=(14, 0))
        tk.Label(self._sidebar, text="Detective AI", bg=self._c["surface"],
                 fg=self._c["muted"], font=self._f["small"],
                 anchor="w").pack(anchor="w", padx=14, pady=(0, 12))

        # New chat.
        new_btn = self._make_button(self._sidebar, "+  New Chat",
                                    self._new_chat, primary=True)
        new_btn.pack(fill=tk.X, padx=12, pady=(0, 12))

        # Conversation list.
        tk.Label(self._sidebar, text="CONVERSATIONS", bg=self._c["surface"],
                 fg=self._c["muted"], font=self._f["eyebrow"],
                 anchor="w").pack(anchor="w", padx=14, pady=(0, 4))
        self._conv_list = tk.Listbox(
            self._sidebar, bg=self._c["surface"], fg=self._c["text"],
            selectbackground=self._c["accent"], selectforeground=self._c["on_accent"],
            highlightthickness=0, bd=0, relief=tk.FLAT,
            font=self._f["small"], exportselection=False, activestyle="none",
        )
        self._conv_list.pack(fill=tk.BOTH, expand=True, padx=10)
        self._conv_list.bind("<<ListboxSelect>>", self._on_conversation_select)

        # Rename / delete the selected conversation.
        row = tk.Frame(self._sidebar, bg=self._c["surface"])
        row.pack(fill=tk.X, padx=12, pady=(8, 10))
        rename_btn = self._make_button(row, "Rename", self._rename_conversation, small=True)
        rename_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)
        delete_btn = self._make_button(row, "Delete", self._delete_conversation, small=True)
        delete_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(6, 0))

        # Footer: user + logout.
        footer = tk.Frame(self._sidebar, bg=self._c["surface"])
        footer.pack(fill=tk.X, side=tk.BOTTOM, padx=12, pady=(6, 14))
        tk.Label(footer, text=f"Logged in as {self.session.username}",
                 bg=self._c["surface"], fg=self._c["muted"],
                 font=self._f["xsmall"], anchor="w").pack(anchor="w", pady=(0, 6))
        self._make_button(footer, "Log Out", self._logout).pack(fill=tk.X)

    def _build_header(self):
        header = tk.Frame(self._main, bg=self._c["bg"])
        header.pack(fill=tk.X, padx=20, pady=(14, 8))

        left = tk.Frame(header, bg=self._c["bg"])
        left.pack(side=tk.LEFT)
        self._conv_title_lbl = tk.Label(
            left, text="New Chat", bg=self._c["bg"], fg=self._c["text"],
            font=self._f["app_mid"], anchor="w",
        )
        self._conv_title_lbl.pack(anchor="w")
        tk.Label(left, text="Detective AI", bg=self._c["bg"],
                 fg=self._c["muted"], font=self._f["xsmall"],
                 anchor="w").pack(anchor="w", pady=(1, 0))

        # Provider / model selector. Clicking the pill opens a small menu
        # (see _open_provider_menu).
        self._provider_pill = tk.Frame(
            header, bg=self._c["surface"],
            highlightbackground=self._c["border"],
            highlightthickness=1, cursor="hand2",
        )
        self._provider_pill.pack(side=tk.RIGHT, anchor="s", pady=(0, 4))

        self._provider_lbl = tk.Label(
            self._provider_pill, bg=self._c["surface"], fg=self._c["muted"],
            font=self._f["xsmall"], cursor="hand2",
        )
        self._provider_lbl.pack(side=tk.LEFT, padx=(10, 0), pady=5)

        self._model_lbl = tk.Label(
            self._provider_pill, bg=self._c["surface"], fg=self._c["text"],
            font=self._f["xsmall"], cursor="hand2",
        )
        self._model_lbl.pack(side=tk.LEFT, pady=5)

        self._chevron_lbl = tk.Label(
            self._provider_pill, text="⌄", bg=self._c["surface"],
            fg=self._c["muted"], font=self._f["small"], cursor="hand2",
        )
        self._chevron_lbl.pack(side=tk.LEFT, padx=(8, 10), pady=5)

        # Bind click + hover on the frame and every label inside it, since
        # tkinter events do not bubble up to a parent widget on their own.
        for widget in (self._provider_pill, self._provider_lbl,
                       self._model_lbl, self._chevron_lbl):
            widget.bind("<Button-1>", self._open_provider_menu)
            widget.bind("<Enter>", lambda e: self._on_provider_hover(True))
            widget.bind("<Leave>", lambda e: self._on_provider_hover(False))

        self._refresh_provider_pill()

    def _build_case_bar(self):
        bar = tk.Frame(self._main, bg=self._c["surface"])
        bar.pack(fill=tk.X, padx=20, pady=(0, 10))

        left = tk.Frame(bar, bg=self._c["surface"])
        left.pack(side=tk.LEFT, fill=tk.X, expand=True,
                  padx=(12, 8), pady=(8, 10))

        tk.Label(left, text="ACTIVE CASE", bg=self._c["surface"],
                 fg=self._c["muted"], font=self._f["eyebrow"],
                 anchor="w").pack(anchor="w")

        row = tk.Frame(left, bg=self._c["surface"])
        row.pack(fill=tk.X, pady=(3, 0))
        self._case_title_lbl = tk.Label(
            row, text="No active case", bg=self._c["surface"],
            fg=self._c["text"], font=self._f["app_mid"], anchor="w",
        )
        self._case_title_lbl.pack(side=tk.LEFT)
        self._case_id_lbl = tk.Label(
            row, text="", bg=self._c["surface"], fg=self._c["muted"],
            font=self._f["xsmall"], anchor="w",
        )
        self._case_id_lbl.pack(side=tk.LEFT, padx=(8, 0))

        status = tk.Frame(bar, bg=self._c["surface"])
        status.pack(side=tk.RIGHT, padx=(8, 12), pady=(10, 10))
        self._case_dot = tk.Label(status, text="●", bg=self._c["surface"],
                                  fg=self._c["muted"], font=self._f["small"])
        self._case_dot.pack(side=tk.LEFT)
        self._case_status_lbl = tk.Label(
            status, text="—", bg=self._c["surface"], fg=self._c["muted"],
            font=self._f["badge"],
        )
        self._case_status_lbl.pack(side=tk.LEFT, padx=(4, 0))

    def _build_chat(self):
        tk.Frame(self._main, bg=self._c["border"], height=1).pack(fill=tk.X)

        shell = tk.Frame(self._main, bg=self._c["bg"])
        shell.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self._chat = tk.Canvas(shell, bg=self._c["bg"],
                               highlightthickness=0, bd=0)
        scrollbar = tk.Scrollbar(shell, orient=tk.VERTICAL,
                                 command=self._chat.yview)
        self._chat_inner = tk.Frame(self._chat, bg=self._c["bg"])
        self._chat_inner_id = self._chat.create_window(
            (0, 0), window=self._chat_inner, anchor="nw")

        self._chat.configure(yscrollcommand=scrollbar.set)
        self._chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._chat.bind("<Configure>", self._on_chat_resize)
        self._chat_inner.bind("<Configure>", self._on_inner_config)

    def _build_input(self):
        bottom = tk.Frame(self._main, bg=self._c["bg"])
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=(8, 16))

        # Thinking status line — animated while the model works.
        self._spinner_lbl = tk.Label(
            bottom, text="", bg=self._c["bg"], fg=self._c["muted"],
            font=self._f["app_mid"], anchor="w",
        )
        self._spinner_lbl.pack(fill=tk.X)

        box = tk.Frame(bottom, bg=self._c["surface"],
                       highlightbackground=self._c["border"],
                       highlightthickness=1)
        box.pack(fill=tk.X, pady=(4, 0))

        self._entry = tk.Entry(
            box, bg=self._c["surface"], fg=self._c["text"],
            insertbackground=self._c["text"], font=self._f["app"],
            relief=tk.FLAT, highlightthickness=0, bd=0,
        )
        self._entry.pack(side=tk.LEFT, fill=tk.BOTH, expand=True,
                         padx=(14, 6), ipady=7, pady=5)
        self._entry.bind("<Return>", self._on_enter)
        self._entry.bind("<KP_Enter>", self._on_enter)
        self._entry.bind("<Key>", self._on_entry_key)
        self._entry.bind("<FocusOut>", self._on_entry_focus_out)

        # Send is drawn with a Label (not tk.Button) so the background and
        # text colours render identically on every platform — the default
        # macOS button can ignore fg and make white text invisible.
        self._send_btn = tk.Label(
            box, text="Send", bg=self._c["accent"], fg=self._c["on_accent"],
            font=("Helvetica Neue", 13, "bold"), relief=tk.FLAT,
            padx=18, pady=8, cursor="hand2",
        )
        self._send_btn.pack(side=tk.RIGHT, padx=(6, 6), pady=5)
        self._send_btn.bind("<Button-1>", lambda e: self._on_send())
        self._send_btn.bind("<Enter>", lambda e: self._on_send_hover(True))
        self._send_btn.bind("<Leave>", lambda e: self._on_send_hover(False))

        self._show_placeholder()
        self._entry.focus_set()

    # ── Conversations (sidebar + persistence) ─────────────────────────────

    def _update_conversation_header(self, title):
        if self._conv_title_lbl is not None:
            self._conv_title_lbl.config(text=title or "New Chat")

    def _refresh_conversation_list(self, select_id=None):
        """Repopulate the sidebar list from SQLite, keeping the open chat selected."""
        try:
            conversations = self.chat_store.list_conversations(self.session.user_id)
        except Exception as exc:
            self._append_error(f"Could not load conversations: {exc}")
            return

        if select_id is None:
            select_id = self.session.active_conversation_id

        self._conv_ids = [c.id for c in conversations]
        self._loading_conversations = True
        try:
            self._conv_list.delete(0, tk.END)
            for conversation in conversations:
                self._conv_list.insert(tk.END, conversation.title or "New Chat")
            if select_id in self._conv_ids:
                index = self._conv_ids.index(select_id)
                self._conv_list.selection_clear(0, tk.END)
                self._conv_list.selection_set(index)
                self._conv_list.activate(index)
                self._conv_list.see(index)
        finally:
            self._loading_conversations = False

    def _on_conversation_select(self, _event=None):
        if self._busy or self._loading_conversations:
            return
        selection = self._conv_list.curselection()
        if not selection:
            return
        index = selection[0]
        if index >= len(self._conv_ids):
            return
        conversation_id = self._conv_ids[index]
        if conversation_id == self.session.active_conversation_id:
            return
        self._load_conversation(conversation_id)

    def _load_conversation(self, conversation_id):
        """Open a conversation: load its messages into the active chat context."""
        try:
            conversation = self.chat_store.get_conversation(
                self.session.user_id, conversation_id
            )
            if conversation is None:
                self._refresh_conversation_list()
                return
            messages = self.chat_store.list_messages(
                self.session.user_id, conversation_id
            )
        except Exception as exc:
            self._append_error(f"Could not load conversation: {exc}")
            return

        # Restore the full history into the context the model will see. The
        # provider call still bounds how much of it is actually sent per turn.
        self._history = []
        for message in messages:
            role = "user" if message.role == "user" else "agent"
            self._history.append((role, message.content))

        self.session.set_active_conversation(conversation_id)
        self._update_conversation_header(conversation.title)
        self._render_history()
        self._refresh_conversation_list(select_id=conversation_id)

    def _new_chat(self):
        if self._busy:
            return
        try:
            conversation = self.chat_store.create_conversation(self.session.user_id)
        except Exception as exc:
            self._append_error(f"Could not create a new chat: {exc}")
            return

        self.session.set_active_conversation(conversation.id)
        self._history = []
        self._update_conversation_header(conversation.title)
        self._clear_chat()
        self._refresh_conversation_list(select_id=conversation.id)
        self._entry.focus_set()
        self._restore_placeholder()

    def _rename_conversation(self):
        if self._busy:
            return
        conversation_id = self.session.active_conversation_id
        if conversation_id is None:
            self._append_error("Select a conversation to rename first.")
            return
        conversation = self.chat_store.get_conversation(
            self.session.user_id, conversation_id
        )
        if conversation is None:
            self._refresh_conversation_list()
            return

        title = simpledialog.askstring(
            "Rename conversation", "New title:",
            initialvalue=conversation.title, parent=self,
        )
        if title is None:      # cancelled
            return
        title = title.strip()
        if not title:
            self._append_error("Conversation title cannot be empty.")
            return
        try:
            renamed = self.chat_store.rename_conversation(
                self.session.user_id, conversation_id, title
            )
        except Exception as exc:
            self._append_error(f"Could not rename conversation: {exc}")
            return
        if not renamed:
            self._append_error("Conversation not found.")
            self._refresh_conversation_list()
            return
        self._update_conversation_header(title)
        self._refresh_conversation_list(select_id=conversation_id)

    def _delete_conversation(self):
        if self._busy:
            return
        conversation_id = self.session.active_conversation_id
        if conversation_id is None:
            self._append_error("Select a conversation to delete first.")
            return
        if not messagebox.askyesno(
            "Delete conversation",
            "Delete this conversation and all of its messages?",
            parent=self,
        ):
            return
        try:
            deleted = self.chat_store.delete_conversation(
                self.session.user_id, conversation_id
            )
        except Exception as exc:
            self._append_error(f"Could not delete conversation: {exc}")
            return
        self.session.set_active_conversation(None)
        self._history = []
        self._clear_chat()
        self._update_conversation_header("New Chat")
        self._refresh_conversation_list()
        if not deleted:
            self._append_error("Conversation not found.")

    # ── persistence helpers ───────────────────────────────────────────────

    def _save_message(self, role, content):
        """Persist one message to the open conversation. Never crashes the GUI."""
        conversation_id = self.session.active_conversation_id
        if conversation_id is None:
            return
        try:
            self.chat_store.add_message(
                self.session.user_id, conversation_id, role, content
            )
        except Exception as exc:
            self._append_error(f"Could not save the message: {exc}")

    def _maybe_autotitle(self):
        """Give a brand-new chat a readable title from its first user message."""
        if len(self._history) != 1:
            return
        role, text = self._history[0]
        if role != "user":
            return
        try:
            self.chat_store.set_title_from_first_message(
                self.session.user_id, self.session.active_conversation_id, text
            )
        except Exception:
            pass
        self._refresh_conversation_list()

    # ── Chat rendering ────────────────────────────────────────────────────

    def _bubble_width(self):
        self.update_idletasks()
        width = self._chat.winfo_width()
        if width < 200:             # window not laid out yet
            width = 780
        return min(BUBBLE_ABSOLUTE_MAX,
                   max(220, int(width * BUBBLE_MAX_FRACTION)))

    def _clear_chat(self):
        for child in self._chat_inner.winfo_children():
            child.destroy()
        self._on_inner_config()

    def _render_history(self):
        """Redraw every bubble from the current in-memory history."""
        self._clear_chat()
        for role, text in self._history:
            if role == "user":
                self._append_user(text)
            else:
                self._append_agent(text)
        self._chat.update_idletasks()
        self._chat.yview_moveto(1.0)

    def _append(self, speaker, text, tag="agent"):
        del speaker  # kept for compatibility; role comes from the tag
        if tag == "user":
            self._append_user(text)
        elif tag == "error":
            self._append_error(text)
        else:
            self._append_agent(text)

    def _append_user(self, text):
        self._append_msg(text, align_right=True, bg=self._c["user_bg"],
                         fg=self._c["user_fg"], label="You")

    def _append_agent(self, text):
        self._append_msg(text, align_right=False, bg=self._c["surface2"],
                         fg=self._c["text"], label="AgentKafle")

    def _append_error(self, text):
        self._append_msg(text, align_right=False, bg=self._c["error_bg"],
                         fg=self._c["error"], label="AgentKafle")

    def _append_msg(self, text, align_right, bg, fg, label):
        row = tk.Frame(self._chat_inner, bg=self._c["bg"])
        row.pack(fill=tk.X, pady=(10, 2), padx=16)

        tk.Label(row, text=label, bg=self._c["bg"], fg=self._c["muted"],
                 font=self._f["xsmall"]
                 ).pack(anchor="e" if align_right else "w", pady=(0, 2))

        bubble = tk.Frame(row, bg=bg, padx=14, pady=9)
        bubble.pack(anchor="e" if align_right else "w")

        tk.Label(bubble, text=text, bg=bg, fg=fg, font=self._f["app"],
                 justify="left", anchor="w",
                 wraplength=max(180, self._bubble_width() - 28)).pack()

        self._on_inner_config()
        self._chat.update_idletasks()
        self._chat.yview_moveto(1.0)

    def _on_chat_resize(self, event):
        self._chat.itemconfigure(self._chat_inner_id, width=event.width)
        self._on_inner_config()

    def _on_inner_config(self, _event=None):
        self._chat.configure(scrollregion=self._chat.bbox("all"))

    def _on_mousewheel(self, event):
        delta = event.delta
        units = int(delta / 120) if abs(delta) >= 100 else int(delta)
        self._chat.yview_scroll(-units, "units")

    def _on_mousewheel_linux(self, event):
        self._chat.yview_scroll(-1 if event.num == 4 else 1, "units")

    # ── Provider / model selector ─────────────────────────────────────────

    def _provider_pill_text(self):
        """Return (label, model) of the actually active provider.

        Reads straight from the router so the pill always reflects the real
        backend that will serve the next request — never a decorative label.
        """
        return self.agent.router.label, self.agent.router.model

    def _refresh_provider_pill(self):
        """Update the header pill text to the selected provider."""
        label, model = self._provider_pill_text()
        self._provider_lbl.config(text=label or "")
        self._model_lbl.config(text=" · " + (model or ""))

    def _on_provider_hover(self, hovering):
        """Lighten the pill while the mouse is over it (hover feedback)."""
        bg = self._c["surface2"] if hovering else self._c["surface"]
        self._provider_pill.config(bg=bg)
        for widget in (self._provider_lbl, self._model_lbl, self._chevron_lbl):
            widget.config(bg=bg)

    def _select_provider(self, key):
        """Handle picking a provider from the menu.

        Switches the Agent's actual LLM backend (via the shared router) first
        and only then updates the GUI pill, so the displayed provider always
        matches the backend that will serve the next request (both respond()
        and reason()). The conversation history is untouched, so provider
        switching inside an open chat keeps it intact.
        """
        if key not in self._providers:
            return

        if key == self.agent.provider_name:
            return

        info = self._providers[key]
        if not info["configured"]:
            # Provider not configured - show error but don't switch
            self._append_error(
                f"{info['label']} is not configured. "
                "Set the required environment variables and restart."
            )
            return

        # Switch the Agent's real LLM backend before touching the GUI so a
        # failed switch can never be reported as successful. A failed switch
        # leaves the router on the previous provider (confirmed by the router).
        try:
            self.agent.set_provider(key)
        except ProviderConfigError as e:
            self._append_error(f"{info['label']} is not configured: {e}")
            return
        except Exception as e:
            self._append_error(f"Failed to switch to {info['label']}: {e}")
            return

        self._refresh_provider_pill()

        model = self.agent.router.model
        self._append_agent(
            f"Provider set to {info['label']} ({model})."
        )

    def _open_provider_menu(self, _event=None):
        """Show the provider chooser menu under the header pill.

        Entries come from the router's registry metadata; configured status is
        shown next to each provider.
        """
        menu = tk.Menu(self, tearoff=0)
        active = self.agent.provider_name
        for key, info in self._providers.items():
            status = " ✓" if key == active else ""
            status = status + " (not configured)" if not info["configured"] else status
            label = f"{info['label']} · {info['default_model']}{status}"
            menu.add_command(
                label=label, command=lambda k=key: self._select_provider(k)
            )

        # Post the menu just below the pill.
        try:
            menu.tk_popup(
                self._provider_pill.winfo_rootx(),
                self._provider_pill.winfo_rooty()
                + self._provider_pill.winfo_height() + 4,
            )
        finally:
            menu.grab_release()

    # ── Case info display ─────────────────────────────────────────────────

    def _refresh_case_info(self):
        case = self.agent.case_manager.get_active()
        if case:
            self._case_title_lbl.config(text=case.title)
            self._case_id_lbl.config(text=case.id)
            self._case_status_lbl.config(text=str(case.status))
            self._case_dot.config(
                fg=self._c["status"].get(case.status, self._c["muted"]),
            )
        else:
            self._case_title_lbl.config(text="No active case")
            self._case_id_lbl.config(text="")
            self._case_status_lbl.config(text="—")
            self._case_dot.config(fg=self._c["muted"])

    # ── Input state ───────────────────────────────────────────────────────

    def _set_busy(self, busy):
        self._busy = busy

        # The send control is a Label, so the busy state is drawn by
        # swapping its colours (tk.Button's state= option is not needed).
        if busy:
            self._send_btn.config(bg=self._c["accent_disabled"], cursor="arrow")
        else:
            self._send_btn.config(bg=self._c["accent"], cursor="hand2")

        state = tk.DISABLED if busy else tk.NORMAL
        self._entry.config(state=state)
        if not busy:
            self._entry.focus_set()
            self._restore_placeholder()

    # ── Loading spinner ───────────────────────────────────────────────────

    def _start_spinner(self):
        self._spinner_idx = 0
        self._spinner_lbl.config(
            text=f"{SPINNER_FRAMES[0]} AgentKafle is thinking...")
        self._tick_spinner()

    def _tick_spinner(self):
        if not self._busy:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(SPINNER_FRAMES)
        self._spinner_lbl.config(
            text=f"{SPINNER_FRAMES[self._spinner_idx]} AgentKafle is thinking...")
        self._spinner_id = self.after(SPINNER_DELAY_MS, self._tick_spinner)

    def _stop_spinner(self):
        if self._spinner_id:
            self.after_cancel(self._spinner_id)
            self._spinner_id = None
        self._spinner_lbl.config(text="")

    # ── Input placeholder (ghost text) ───────────────────────────────────

    def _show_placeholder(self):
        """Show dimmed placeholder text when the entry is empty."""
        self._entry.config(fg=self._c["placeholder"])
        self._entry.delete(0, tk.END)
        self._entry.insert(0, self._input_placeholder)
        self._entry_is_placeholder = True

    def _on_entry_key(self, _event=None):
        """Remove the placeholder as soon as the user types anything.

        Only clears if the current text is the placeholder. If the user has
        already typed (or text was inserted programmatically), do nothing.
        """
        if self._entry_is_placeholder:
            self._entry.delete(0, tk.END)
            self._entry.config(fg=self._c["text"])
            self._entry_is_placeholder = False

    def _on_entry_focus_out(self, _event=None):
        """Bring the placeholder back when the field is emptied and left."""
        if not self._entry.get().strip() and not self._entry_is_placeholder:
            self._show_placeholder()

    def _restore_placeholder(self):
        """Re-show the placeholder after a message is sent."""
        if not self._entry.get().strip() and not self._entry_is_placeholder:
            self._show_placeholder()

    def _on_send_hover(self, hovering):
        """Darken the Send button while hovered (skipped while busy)."""
        if self._busy:
            return
        self._send_btn.config(
            bg=self._c["accent_d"] if hovering else self._c["accent"],
        )

    # ── Send / Enter ──────────────────────────────────────────────────────

    def _on_enter(self, _event=None):
        if not self._busy:
            self._on_send()

    def _on_send(self):
        if self._busy:
            return

        if self._entry_is_placeholder:
            return

        user_text = self._entry.get().strip()
        if not user_text:
            return

        self._entry.delete(0, tk.END)
        self._entry.config(fg=self._c["text"])
        self._append_user(user_text)
        self._history.append(("user", user_text))

        # Persist the user message immediately (auto-save). If the app closes
        # during generation, the user's words are never lost.
        self._save_message("user", user_text)
        self._maybe_autotitle()

        # Fast commands: handled on the main thread.
        handled, response = self.agent.handle_command(user_text)
        if handled:
            self._history.append(("agent", response))
            self._save_message("assistant", response)
            self._refresh_conversation_list()
            self._append_agent(response)
            self._refresh_case_info()
            self._restore_placeholder()
            return

        # LLM call: run in background thread so GUI stays responsive.
        self._set_busy(True)
        self._start_spinner()
        threading.Thread(
            target=self._llm_worker,
            args=(user_text,),
            daemon=True,
        ).start()

    def _llm_worker(self, user_text):
        """Runs on a background thread.  Posts the result back with root.after()."""
        try:
            # Pass the open conversation's history so the LLM sees this
            # conversation's context (bounded inside AgentKafle.respond).
            response = self.agent.respond(user_text, history=self._history)
        except Exception as exc:
            self.after(0, self._on_llm_error, exc)
            return
        self.after(0, self._on_llm_done, response)

    def _on_llm_done(self, response):
        self._stop_spinner()
        self._set_busy(False)
        self._history.append(("agent", response))
        self._save_message("assistant", response)
        self._refresh_conversation_list()
        self._append_agent(response)

    def _on_llm_error(self, error):
        self._stop_spinner()
        self._set_busy(False)

        # Connection failures are normalized by the provider into
        # ProviderConnectionError — no brittle string-matching needed. The
        # Ollama-specific hint only applies when Ollama is the active backend.
        if (
            self.agent is not None
            and self.agent.provider_name == "ollama"
            and isinstance(error, ProviderConnectionError)
        ):
            friendly = (
                "Could not reach Ollama. Please ensure it is running:\n"
                "  ollama serve"
            )
        else:
            friendly = f"LLM request failed: {error}"

        self._append_error(friendly)


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    app = AgentKafleGUI()
    app.mainloop()


if __name__ == "__main__":
    main()