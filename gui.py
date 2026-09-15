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
from tkinter import scrolledtext

from agent import AgentKafle


# ── Braille spinner frames ────────────────────────────────────────────────
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_DELAY_MS = 80


class AgentKafleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AgentKafle — Detective AI")
        self.geometry("950x720")
        self.minsize(640, 480)
        self.configure(bg="#1e1e1e")

        self.agent = AgentKafle()
        self._busy = False          # True while an LLM call is running
        self._spinner_idx = 0       # current frame index
        self._spinner_id = None     # after() id so we can cancel it

        # Conversation history: list of (role, text) tuples.
        # role is "user" or "agent".  Sent to the LLM each time so the
        # model sees the full conversation.
        self._history = []

        self._build_ui()
        self._refresh_case_info()
        self._append_agent(
            "Hello. I am AgentKafle.\n"
            "Type 'help' to see commands, or ask me anything.\n"
        )

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        colours = {
            "bg":        "#1e1e1e",
            "panel":     "#2a2a2a",
            "input_bg":  "#333333",
            "text_fg":   "#e0e0e0",
            "accent":    "#569cd6",
            "user":      "#b5cea8",
            "agent":     "#e0e0e0",
            "label":     "#808080",
            "heading":   "#d4a843",
        }
        self._c = colours

        # ── Top bar: case info ────────────────────────────────────────────
        top = tk.Frame(self, bg=colours["bg"], height=48)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Label(top, text="ACTIVE CASE", fg=colours["heading"],
                 bg=colours["bg"], font=("Helvetica", 8, "bold")).pack(
            side=tk.LEFT, padx=(14, 4), pady=10, anchor=tk.W,
        )
        self._case_lbl = tk.Label(
            top, text="NO ACTIVE CASE", fg=colours["label"],
            bg=colours["bg"], font=("Helvetica", 10), anchor=tk.W,
        )
        self._case_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=10)

        tk.Label(top, text="AgentKafle", fg=colours["accent"],
                 bg=colours["bg"], font=("Helvetica", 9, "bold")).pack(
            side=tk.RIGHT, padx=14, pady=10,
        )

        tk.Frame(self, bg=colours["panel"], height=1).pack(fill=tk.X)

        # ── Chat area ─────────────────────────────────────────────────────
        self._chat = scrolledtext.ScrolledText(
            self, wrap=tk.WORD, state=tk.DISABLED,
            bg=colours["bg"], fg=colours["text_fg"],
            insertbackground=colours["text_fg"],
            font=("Menlo", 11), relief=tk.FLAT,
            borderwidth=0, padx=12, pady=8,
        )
        self._chat.pack(fill=tk.BOTH, expand=True)
        self._chat.config(state=tk.NORMAL)
        self._chat.tag_configure("you",    foreground=colours["user"])
        self._chat.tag_configure("agent",  foreground=colours["agent"])
        self._chat.tag_configure("bold",   font=("Menlo", 11, "bold"))
        self._chat.tag_configure("dim",    foreground=colours["label"])
        self._chat.tag_configure("error",  foreground="#f44747")
        self._chat.config(state=tk.DISABLED)

        # ── Bottom bar: input + send + spinner ────────────────────────────
        tk.Frame(self, bg=colours["panel"], height=1).pack(fill=tk.X)

        bottom = tk.Frame(self, bg=colours["input_bg"], height=52)
        bottom.pack(fill=tk.X)
        bottom.pack_propagate(False)

        self._entry = tk.Entry(
            bottom, bg=colours["input_bg"], fg=colours["text_fg"],
            insertbackground=colours["text_fg"],
            font=("Helvetica", 12), relief=tk.FLAT,
            borderwidth=0,
        )
        self._entry.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14, 8), pady=10,
        )
        self._entry.bind("<Return>", self._on_enter)
        self._entry.focus_set()

        self._spinner_lbl = tk.Label(
            bottom, text="", fg=colours["accent"],
            bg=colours["input_bg"], font=("Menlo", 13),
            width=3, anchor=tk.W,
        )
        self._spinner_lbl.pack(side=tk.RIGHT, padx=(0, 6), pady=10)

        self._send_btn = tk.Button(
            bottom, text="Send", bg=colours["accent"], fg="#ffffff",
            activebackground="#4a8cbf", activeforeground="#ffffff",
            font=("Helvetica", 11, "bold"), relief=tk.FLAT,
            borderwidth=0, padx=16, pady=4,
            command=self._on_send,
        )
        self._send_btn.pack(side=tk.RIGHT, padx=(0, 10), pady=10)

    # ── Chat helpers ──────────────────────────────────────────────────────

    def _append(self, speaker, text, tag="agent"):
        self._chat.config(state=tk.NORMAL)
        self._chat.insert(tk.END, f"{speaker}:\n", "bold")
        self._chat.insert(tk.END, text + "\n\n", tag)
        self._chat.see(tk.END)
        self._chat.config(state=tk.DISABLED)

    def _append_user(self, text):
        self._append("You", text, tag="you")

    def _append_agent(self, text):
        self._append("AgentKafle", text, tag="agent")

    def _append_error(self, text):
        self._chat.config(state=tk.NORMAL)
        self._chat.insert(tk.END, f"AgentKafle:\n", "bold")
        self._chat.insert(tk.END, text + "\n\n", "error")
        self._chat.see(tk.END)
        self._chat.config(state=tk.DISABLED)

    # ── Case info display ─────────────────────────────────────────────────

    def _refresh_case_info(self):
        case = self.agent.case_manager.get_active()
        if case:
            self._case_lbl.config(
                text=f"{case.id} — {case.title} — {case.status}",
            )
        else:
            self._case_lbl.config(text="NO ACTIVE CASE")

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
        self._spinner_lbl.config(text=SPINNER_FRAMES[0])
        self._tick_spinner()

    def _tick_spinner(self):
        if not self._busy:
            return
        self._spinner_idx = (self._spinner_idx + 1) % len(SPINNER_FRAMES)
        self._spinner_lbl.config(text=SPINNER_FRAMES[self._spinner_idx])
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