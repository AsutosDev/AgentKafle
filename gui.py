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
from urllib.parse import urlparse

from agent import AgentKafle


# ── Braille spinner frames ────────────────────────────────────────────────
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_DELAY_MS = 80

# Message bubbles never grow wider than this.
BUBBLE_MAX_FRACTION = 0.72
BUBBLE_ABSOLUTE_MAX = 640


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

        self._setup_style()
        self._build_ui()
        self._refresh_case_info()
        self._append_agent(
            "Hello. I am AgentKafle.\n"
            "Type 'help' to see commands, or ask me anything.\n"
        )

    # ── Style ─────────────────────────────────────────────────────────────

    def _setup_style(self):
        self._c = {
            "bg":       "#17181c",   # window background
            "surface":  "#1e2126",   # raised panels
            "surface2": "#272b33",   # agent message bubble
            "border":   "#353b46",   # hairlines and field outlines
            "text":     "#eceff4",   # primary text
            "muted":    "#8b95a5",   # secondary text
            "accent":   "#3b82f6",   # mac-blue accent
            "accent_d": "#2f69cf",   # pressed accent
            "user_bg":  "#2f5fd0",   # user message bubble
            "user_fg":  "#ffffff",
            "error_bg": "#3a2426",   # error message bubble
            "error":    "#ff7a7a",
            "status":   {            # case status → dot colour
                "OPEN":   "#4cd964",
                "SOLVED": "#5ac8fa",
                "CLOSED": "#9aa0a6",
            },
        }
        self._f = {
            "title":   ("Helvetica Neue", 17, "bold"),
            "app":     ("Helvetica Neue", 12),
            "app_mid": ("Helvetica Neue", 11),
            "small":   ("Helvetica Neue", 10),
            "xsmall":  ("Helvetica Neue", 9),
            "eyebrow": ("Helvetica Neue", 8, "bold"),
            "badge":   ("Helvetica Neue", 10, "bold"),
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

        # Provider / model indicator.  The chevron is reserved for a future
        # provider selector and is intentionally inert for now.
        provider, model = self._provider_model()
        pill = tk.Frame(header, bg=self._c["surface"],
                        highlightbackground=self._c["border"],
                        highlightthickness=1)
        pill.pack(side=tk.RIGHT, anchor="s", pady=(0, 4))
        tk.Label(pill, text=provider, bg=self._c["surface"],
                 fg=self._c["muted"], font=self._f["xsmall"]
                 ).pack(side=tk.LEFT, padx=(10, 0), pady=5)
        tk.Label(pill, text=" · " + model, bg=self._c["surface"],
                 fg=self._c["text"], font=self._f["xsmall"]
                 ).pack(side=tk.LEFT, pady=5)
        tk.Label(pill, text="⌄", bg=self._c["surface"],
                 fg=self._c["muted"], font=self._f["small"]
                 ).pack(side=tk.LEFT, padx=(8, 10), pady=5)

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

        self._send_btn = tk.Button(
            box, text="Send", bg=self._c["accent"], fg=self._c["user_fg"],
            activebackground=self._c["accent_d"], activeforeground="#ffffff",
            font=("Helvetica Neue", 11, "bold"), relief=tk.FLAT,
            highlightthickness=0, bd=0, padx=18, pady=7, cursor="hand2",
            command=self._on_send,
        )
        self._send_btn.pack(side=tk.RIGHT, padx=(6, 6), pady=5)

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

    # ── Provider / model ──────────────────────────────────────────────────

    def _provider_model(self):
        llm = self.agent.llm
        host = urlparse(llm.base_url).hostname or "localhost"
        if host in ("localhost", "127.0.0.1"):
            provider = "OLLAMA"
        else:
            provider = host.upper()
        return provider, llm.model

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
        state = tk.DISABLED if busy else tk.NORMAL
        self._send_btn.config(state=state)
        self._entry.config(state=state)
        if not busy:
            self._entry.focus_set()

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

    # ── Send / Enter ──────────────────────────────────────────────────────

    def _on_enter(self, _event=None):
        if not self._busy:
            self._on_send()

    def _on_send(self):
        if self._busy:
            return

        user_text = self._entry.get().strip()
        if not user_text:
            return

        self._entry.delete(0, tk.END)
        self._append_user(user_text)
        self._history.append(("user", user_text))

        # Fast commands: handled on the main thread.
        handled, response = self.agent.handle_command(user_text)
        if handled:
            self._history.append(("agent", response))
            self._append_agent(response)
            self._refresh_case_info()
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
            response = self.agent.respond(user_text)
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