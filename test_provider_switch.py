"""Regression test for the AgentKafle provider-switching bug.

Ensures that switching the provider really swaps the backend that serves
every subsequent LLM request (agent.respond, detective.reason, and any
DETECTIVE-mode routing), not just the GUI label.

The test replaces llm.create_llm with fake providers that identify
themselves, so no real Ollama/Gemini connection is needed.

Run with:
    python3 test_provider_switch.py
"""

import sys

import agent as agent_mod


class FakeLLM:
    """Provider stand-in that answers with its own name."""

    def __init__(self, name):
        self.name = name

    def chat(self, messages, json_mode=False):
        _calls.append((self.name, json_mode))
        if json_mode:
            return (
                '{"summary": "x", "established_facts": [], '
                '"unverified_claims": [], "disputed_information": [], '
                '"contradictions": [], "hypotheses": [], "conclusion": "x", '
                '"confidence": "LOW", "missing_information": []}'
            )
        return f"REPLY-FROM-{self.name}"

    def complete(self, prompt):
        return f"REPLY-FROM-{self.name}"


_calls = []


def fake_create(provider):
    provider = provider.lower().strip()
    if provider == "ollama":
        return FakeLLM("OLLAMA")
    if provider == "gemini":
        return FakeLLM("GEMINI")
    raise ValueError(f"Unknown provider: {provider}")


agent_mod.create_llm = fake_create


def _history(previous_provider):
    """Conversation containing an old Gemini-era assistant reply."""
    return [
        ("user", "hello"),
        ("agent", f"REPLY-FROM-{previous_provider}"),
        ("user", "What model are you using?"),
    ]


def _assert_same_llm(agent, label):
    assert agent.llm is agent.detective.llm, (
        f"{label}: agent.llm is not detective.llm"
    )


agent = agent_mod.AgentKafle(provider="gemini")
_assert_same_llm(agent, "init")
assert agent.llm.name == "GEMINI"

case_manager, evidence_manager, memory = (
    agent.case_manager,
    agent.evidence_manager,
    agent.memory,
)

reply = agent.respond("What model are you using?", history=_history("GEMINI"))
assert "REPLY-FROM-GEMINI" in reply

# Gemini → Ollama
agent.set_provider("ollama")
_assert_same_llm(agent, "gemini->ollama")
assert agent.llm.name == "OLLAMA"
assert agent.case_manager is case_manager
assert agent.evidence_manager is evidence_manager
assert agent.memory is memory
reply = agent.respond("What model are you using?", history=_history("GEMINI"))
assert "REPLY-FROM-OLLAMA" in reply

# Ollama → Gemini
agent.set_provider("gemini")
_assert_same_llm(agent, "ollama->gemini")
assert agent.llm.name == "GEMINI"
reply = agent.respond("What model are you using?", history=_history("OLLAMA"))
assert "REPLY-FROM-GEMINI" in reply

# Gemini → Ollama again
agent.set_provider("ollama")
_assert_same_llm(agent, "gemini->ollama(2)")
assert agent.llm.name == "OLLAMA"
reply = agent.respond("What model are you using?", history=_history("GEMINI"))
assert "REPLY-FROM-OLLAMA" in reply

# DETECTIVE-mode routing must hit the newly selected provider on BOTH the
# detective.reason() call (json_mode) and the final respond() chat call.
agent.set_provider("gemini")
calls_before = len(_calls)
agent.case_manager.create("The Case", "desc")
agent.case_manager.set_active(agent.case_manager.list_all()[0].id)
reply = agent.respond("investigate", history=_history("OLLAMA"))
assert "REPLY-FROM-GEMINI" in reply
step_calls = _calls[calls_before:]
assert all(name == "GEMINI" for name, _ in step_calls)
assert any(json_mode for _, json_mode in step_calls)

print("OK: all provider-switching checks passed")