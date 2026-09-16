"""AgentKafle — Detective AI desktop GUI.

Run with:
    python gui.py

The GUI reuses the existing AgentKafle class and command router.  LLM calls
run on a background thread so the interface stays responsive, and a braille
spinner animates while the model is thinking.

Architecture:

  GUI  →  handle_command()         (fast, same thread)
       →  respond() via threading   (slow, background thread)
       →  root.after()              (posts result back to main thread)
"""

import threading
import tkinter as tk

from agent import AgentKafle


# ── Braille spinner frames ────────────────────────────────────────────────
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_DELAY_MS = 80

# Message bubbles never grow wider than this.
BUBBLE_MAX_FRACTION = 0.72
BUBBLE_ABSOLUTE_MAX = 640


# ── Provider registry ─────────────────────────────────────────────────────
# Single place that describes the providers the header can select.
# "configured" is determined at runtime based on API key availability.
# A real Gemini backend plugs in here without redesigning any GUI code.
# ───────────────────────────────────────────────────────────────────────────

def _check_provider_configured(provider: str) -> bool:
    """Check if a provider has its required configuration."""
    if provider == "ollama":
        return True  # Ollama runs locally, no API key needed
    elif provider == "gemini":
        import os
        return bool(os.getenv("GEMINI_API_KEY", "").strip())
    return False


PROVIDERS = {
    "ollama": {"label": "OLLAMA", "value": "llama3.2:3b", "configured": True},
    "gemini": {"label": "GEMINI", "value": "gemini-3.6-flash", "configured": _check_provider_configured("gemini")},
}


class AgentKafleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AgentKafle — Detective AI")
        self.geometry("920x680")
        self.minsize(600, 460)

        self.agent = AgentKafle()
        self._busy = False          # True while an LLM call is running
        self._spinner_idx = 0       # current frame index
        self._spinner_id = None     # after() id so we can cancel it

        # Conversation history: list of (role, text) tuples.
        # role is "user" or "agent".  Sent to the LLM each time so the
        # model sees the full conversation.
        self._history = []

        # Provider selection state. "ollama" is the default and the only
        # configured backend for now. See PROVIDERS above.
        self._active_provider = "ollama"

        # Input field placeholder ("ghost text") state.
        self._entry_placeholder = "Ask AgentKafle…"
        self._entry_is_placeholder = True

        self._setup_style()
        self._build_ui()
        self._refresh_case_info()
        self._append_agent(
            "Hello. I am AgentKafle.\n"
            "Type 'help' to see commands, or ask me anything.\n"
        )

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

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._build_case_bar()
        self._build_chat()
        self._build_input()
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _build_header(self):
        header = tk.Frame(self, bg=self._c["bg"])
        header.pack(fill=tk.X, padx=20, pady=(14, 8))

        wordmark = tk.Frame(header, bg=self._c["bg"])
        wordmark.pack(side=tk.LEFT)
        tk.Label(wordmark, text="AgentKafle", bg=self._c["bg"],
                 fg=self._c["text"], font=self._f["title"],
                 anchor="w").pack(anchor="w")
        tk.Label(wordmark, text="Detective AI", bg=self._c["bg"],
                 fg=self._c["muted"], font=self._f["small"],
                 anchor="w").pack(anchor="w", pady=(1, 0))

        # Provider / model selector. Clicking the pill opens a small menu
        # (see _open_provider_menu). Only Ollama is configured for now;
        # Gemini shows a "coming soon" note instead of making fake requests.
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
        bar = tk.Frame(self, bg=self._c["surface"])
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
        tk.Frame(self, bg=self._c["border"], height=1).pack(fill=tk.X)

        shell = tk.Frame(self, bg=self._c["bg"])
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
        bottom = tk.Frame(self, bg=self._c["bg"])
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

    # ── Chat rendering ────────────────────────────────────────────────────

    def _bubble_width(self):
        self.update_idletasks()
        width = self._chat.winfo_width()
        if width < 200:             # window not laid out yet
            width = 780
        return min(BUBBLE_ABSOLUTE_MAX,
                   max(220, int(width * BUBBLE_MAX_FRACTION)))

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

    def _provider_model(self):
        """Return (label, model) of the currently selected provider."""
        info = PROVIDERS[self._active_provider]
        return info["label"], info["value"]

    def _refresh_provider_pill(self):
        """Update the header pill text to the selected provider."""
        label, model = self._provider_model()
        self._provider_lbl.config(text=label)
        self._model_lbl.config(text=" · " + model)

    def _on_provider_hover(self, hovering):
        """Lighten the pill while the mouse is over it (hover feedback)."""
        bg = self._c["surface2"] if hovering else self._c["surface"]
        self._provider_pill.config(bg=bg)
        for widget in (self._provider_lbl, self._model_lbl, self._chevron_lbl):
            widget.config(bg=bg)

    def _select_provider(self, key):
        """Handle picking a provider from the menu.

        Switches the active LLM provider for the AgentKafle instance.
        """
        if key not in PROVIDERS:
            return

        if key == self._active_provider:
            return

        info = PROVIDERS[key]
        if not info["configured"]:
            # Provider not configured - show error but don't switch
            self._append_error(
                f"{info['label']} is not configured. "
                "Set GEMINI_API_KEY environment variable and restart."
            )
            return

        # Create new agent with the selected provider
        try:
            new_agent = AgentKafle(provider=key)
            # Preserve the existing history and case state
            new_agent.case_manager = self.agent.case_manager
            new_agent.evidence_manager = self.agent.evidence_manager
            new_agent.memory = self.agent.memory
            new_agent.detective = self.agent.detective
            self.agent = new_agent
        except Exception as e:
            self._append_error(f"Failed to switch to {info['label']}: {e}")
            return

        self._active_provider = key
        self._refresh_provider_pill()

        self._append_agent(
            f"Provider set to {info['label']} ({info['value']})."
        )

    def _open_provider_menu(self, _event=None):
        """Show the provider chooser menu under the header pill."""
        menu = tk.Menu(self, tearoff=0)
        for key, info in PROVIDERS.items():
            label = f"{info['label']} · {info['value']}"
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
        self._entry.insert(0, self._entry_placeholder)
        self._entry_is_placeholder = True

    def _on_entry_key(self, _event=None):
        """Remove the placeholder as soon as the user types anything."""
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

        # Fast commands: handled on the main thread.
        handled, response = self.agent.handle_command(user_text)
        if handled:
            self._history.append(("agent", response))
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
            # Pass the session's recent conversation so the LLM can see this
            # conversation's context (bounded inside AgentKafle.respond).
            response = self.agent.respond(user_text, history=self._history)
        except Exception as exc:
            self.after(0, self._on_llm_error, str(exc))
            return
        self.after(0, self._on_llm_done, response)

    def _on_llm_done(self, response):
        self._stop_spinner()
        self._set_busy(False)
        self._history.append(("agent", response))
        self._append_agent(response)

    def _on_llm_error(self, error):
        self._stop_spinner()
        self._set_busy(False)

        if any(word in error.lower() for word in ("connect", "refused", "url")):
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