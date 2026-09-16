"""Regression + architecture tests for the AgentKafle provider layer.

Covers, without needing real Ollama/Gemini credentials:

- Provider registry: providers registered, metadata available, unknown
  providers fail cleanly with ProviderConfigError.
- Shared exception taxonomy: config vs connection failures are
  distinguishable and share a common base.
- Provider switching: every pair/sequence the GUI can produce, verifying the
  active provider, the active model, and that the Agent and the Detective
  always share the same backend through the ProviderRouter.
- Detective: normal and JSON reasoning both hit the active provider, and a
  switch cannot leave the Detective on the old backend.
- Model selection: providers are constructed with an explicit model and a
  switch updates the active model metadata.
- State preservation: cases, evidence, and memory survive every switch.
- Lifecycle: a switched-out provider is closed.

The test replaces the real providers in the registry with fake ones (the
registry is explicitly designed to allow that), so no network or API key is
ever required.

Run with:
    python3 test_provider_switch.py
"""

import os
import sys

import llm as llm_mod

import agent as agent_mod


# ── Fake providers ──────────────────────────────────────────────────────
# Registered under the real provider keys so the app uses these instead of
# touching Ollama/Gemini. _calls records every LLM invocation so tests can
# assert WHICH backend served each request.

JSON_BODY = (
    '{"summary": "x", "established_facts": [], '
    '"unverified_claims": [], "disputed_information": [], '
    '"contradictions": [], "hypotheses": [], "conclusion": "x", '
    '"confidence": "LOW", "missing_information": []}'
)

_calls = []      # (provider_name, model, json_mode)
_closed = []     # provider names that have been closed


class _FakeBase(llm_mod.LLMInterface):
    """Provider stand-in that answers with its own name and model."""

    def __init__(self, model=None):
        super().__init__(model)
        self.closed = False

    def chat(self, messages, json_mode=False):
        _calls.append((self.name, self.model, json_mode))
        if json_mode:
            return JSON_BODY
        return f"REPLY-FROM-{self.label}"

    def chat_json(self, messages):
        return self.chat(messages, json_mode=True)

    def complete(self, prompt):
        return f"REPLY-FROM-{self.label}"

    def close(self):
        self.closed = True
        _closed.append(self.name)

    def available_models(self):
        return [f"{self.name}-one", f"{self.name}-two"]


class FakeOllama(_FakeBase):
    name = "ollama"
    label = "OLLAMA"
    default_model = "llama3.2:3b"
    env_keys = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


class FakeGemini(_FakeBase):
    name = "gemini"
    label = "GEMINI"
    default_model = "gemini-3.6-flash"
    env_keys = ("GEMINI_API_KEY", "GEMINI_MODEL")


class FakeUnconfigured(_FakeBase):
    """A provider that is never configured; construction must fail cleanly."""
    name = "cloudsecret"
    label = "CLOUDSECRET"
    default_model = "secret-model"
    env_keys = ("CLOUD_API_KEY",)

    @classmethod
    def configured(cls):
        return False

    def __init__(self, model=None):
        raise llm_mod.ProviderConfigError(
            "cloudsecret: missing CLOUD_API_KEY"
        )


class FakeFlaky(_FakeBase):
    """A provider whose requests fail with a connection error."""
    name = "flaky"
    label = "FLAKY"
    default_model = "flaky-model"

    def chat(self, messages, json_mode=False):
        _calls.append((self.name, self.model, json_mode))
        raise llm_mod.ProviderConnectionError(
            f"{self.name}: connection refused"
        )


llm_mod.register_provider(FakeOllama)
llm_mod.register_provider(FakeGemini)
llm_mod.register_provider(FakeUnconfigured)
llm_mod.register_provider(FakeFlaky)


# ── Helpers ──────────────────────────────────────────────────────────────

def _history(previous_provider):
    """Conversation containing an old-era assistant reply from a previous backend."""
    return [
        ("user", "hello"),
        ("agent", f"REPLY-FROM-{previous_provider}"),
        ("user", "What model are you using?"),
    ]


def _assert_shared_backend(agent, label):
    """Agent and Detective must resolve to the SAME active provider now."""
    assert agent.router is agent.detective.router, (
        f"{label}: agent.router is not detective.router"
    )
    assert agent.llm is agent.detective.llm, (
        f"{label}: agent.llm is not detective.llm"
    )
    assert agent.llm is agent.router.provider, (
        f"{label}: agent.llm is not router.provider"
    )


def _assert_switch_ok(agent, expect_name, label):
    _assert_shared_backend(agent, label)
    assert agent.router.name == expect_name, (
        f"{label}: router.name {agent.router.name!r} != {expect_name!r}"
    )
    assert agent.llm.name == expect_name, (
        f"{label}: agent.llm.name {agent.llm.name!r} != {expect_name!r}"
    )


# ── 1. Provider registry ─────────────────────────────────────────────────

def test_registry():
    providers = llm_mod.list_providers()
    assert "ollama" in providers, "ollama not registered"
    assert "gemini" in providers, "gemini not registered"

    ollama = providers["ollama"]
    gemini = providers["gemini"]
    assert ollama.label == "OLLAMA"
    assert gemini.label == "GEMINI"
    assert ollama.default_model == "llama3.2:3b"
    assert gemini.default_model == "gemini-3.6-flash"
    assert "GEMINI_API_KEY" in gemini.env_keys

    # Unknown provider fails cleanly with the shared config error type.
    try:
        llm_mod.get_provider_class("not-a-real-provider")
    except llm_mod.ProviderConfigError:
        pass
    else:
        raise AssertionError("unknown provider did not raise ProviderConfigError")

    assert FakeOllama.configured() is True
    assert FakeUnconfigured.configured() is False
    print("OK: registry")


# ── 2. Exceptions ────────────────────────────────────────────────────────

def test_exceptions():
    assert issubclass(llm_mod.ProviderConfigError, llm_mod.ProviderError)
    assert issubclass(llm_mod.ProviderConnectionError, llm_mod.ProviderError)
    assert not issubclass(llm_mod.ProviderConnectionError, llm_mod.ProviderConfigError)

    # Config failure on construction surfaces the shared type.
    try:
        FakeUnconfigured()
    except llm_mod.ProviderConfigError as exc:
        assert "cloudsecret" in str(exc)
    else:
        raise AssertionError("unconfigured provider did not raise")

    # Connection failure surfaces the shared type and is distinguishable.
    try:
        FakeFlaky().chat([{"role": "user", "content": "hi"}])
    except llm_mod.ProviderConnectionError as exc:
        assert "connection refused" in str(exc)
    except Exception as exc:
        raise AssertionError(f"flaky raised wrong type: {type(exc).__name__}")

    # A caller that only knows about ProviderError can still catch both.
    try:
        FakeUnconfigured()
    except llm_mod.ProviderError:
        pass
    else:
        raise AssertionError("ProviderError base did not catch config error")
    print("OK: exceptions")


# ── 3. Provider switching (the original regression + every pair) ─────────

def test_switching():
    agent = agent_mod.AgentKafle(provider="gemini")
    _assert_switch_ok(agent, "gemini", "init")
    assert agent.router.model == "gemini-3.6-flash"

    case_manager, evidence_manager, memory = (
        agent.case_manager,
        agent.evidence_manager,
        agent.memory,
    )

    reply = agent.respond("What model are you using?", history=_history("GEMINI"))
    assert "REPLY-FROM-GEMINI" in reply

    # Gemini → Ollama
    agent.set_provider("ollama")
    _assert_switch_ok(agent, "ollama", "gemini->ollama")
    assert agent.router.model == "llama3.2:3b"
    assert agent.case_manager is case_manager
    assert agent.evidence_manager is evidence_manager
    assert agent.memory is memory
    reply = agent.respond("What model are you using?", history=_history("GEMINI"))
    assert "REPLY-FROM-OLLAMA" in reply

    # Ollama → Gemini
    agent.set_provider("gemini")
    _assert_switch_ok(agent, "gemini", "ollama->gemini")
    reply = agent.respond("What model are you using?", history=_history("OLLAMA"))
    assert "REPLY-FROM-GEMINI" in reply

    # Gemini → Ollama again
    agent.set_provider("ollama")
    _assert_switch_ok(agent, "ollama", "gemini->ollama(2)")
    reply = agent.respond("What model are you using?", history=_history("GEMINI"))
    assert "REPLY-FROM-OLLAMA" in reply

    # Ollama → Gemini → Ollama full round trip
    agent.set_provider("gemini")
    agent.set_provider("ollama")
    _assert_switch_ok(agent, "ollama", "o->g->o")
    reply = agent.respond("What model are you using?", history=_history("GEMINI"))
    assert "REPLY-FROM-OLLAMA" in reply
    print("OK: switching")


# ── 4. Detective routing ─────────────────────────────────────────────────

def test_detective_uses_active_provider():
    agent = agent_mod.AgentKafle(provider="gemini")

    agent.case_manager.create("Why Detective", "desc")
    case = agent.case_manager.list_all()[0]
    agent.case_manager.set_active(case.id)
    agent.evidence_manager.add(case.id, "A note", "handwritten")

    try:
        # DETECTIVE-mode routing must hit the newly selected provider on BOTH the
        # detective.reason() call (JSON) and the final respond() chat call.
        calls_before = len(_calls)
        reply = agent.respond("investigate", history=_history("OLLAMA"))
        assert "REPLY-FROM-GEMINI" in reply
        step_calls = _calls[calls_before:]
        assert all(name == "gemini" for name, _model, _json in step_calls)
        assert any(json_mode for _name, _model, json_mode in step_calls)

        # Direct Detective.reason() also uses the active (Gemini) provider.
        inv, _ = agent.detective.investigate(case.id)
        calls_before = len(_calls)
        result, _ = agent.detective.reason(inv)
        assert result is not None
        direct_calls = _calls[calls_before:]
        assert all(name == "gemini" for name, _model, _json in direct_calls)
        assert all(json_mode for _name, _model, json_mode in direct_calls)

        # Switching cannot leave the Detective on the old backend: the Detective
        # holds the same router, so reasoning now hits Ollama.
        agent.set_provider("ollama")
        calls_before = len(_calls)
        result, _ = agent.detective.reason(inv)
        assert result is not None
        direct_calls = _calls[calls_before:]
        assert all(name == "ollama" for name, _model, _json in direct_calls)
        assert agent.detective.router is agent.router
    finally:
        # Clean up the files this test created.
        for path in (
            os.path.join("cases", f"{case.id}.json"),
            os.path.join("cases", f"{case.id}.evidence.json"),
        ):
            if os.path.exists(path):
                os.remove(path)
    print("OK: detective routing")


# ── 5. Model selection ───────────────────────────────────────────────────

def test_model_selection():
    # create_llm accepts an explicit model.
    prov = llm_mod.create_llm("ollama", model="llama3.2:3b")
    assert prov.name == "ollama"
    assert prov.model == "llama3.2:3b"
    assert prov.available_models() == ["ollama-one", "ollama-two"]

    prov2 = llm_mod.create_llm("gemini", model="gemini-2.5-flash")
    assert prov2.name == "gemini"
    assert prov2.model == "gemini-2.5-flash"

    # create_llm with no model falls back to the provider default.
    assert llm_mod.create_llm("ollama").model == "llama3.2:3b"

    # Understanding unknown provider via create_llm fails cleanly too.
    try:
        llm_mod.create_llm("nope")
    except llm_mod.ProviderConfigError:
        pass
    else:
        raise AssertionError("create_llm(unknown) did not raise")

    # Switching provider/model updates the active metadata on the router.
    agent = agent_mod.AgentKafle(provider="ollama")
    assert agent.router.model == "llama3.2:3b"
    agent.set_provider("gemini", model="gemini-2.5-flash")
    assert agent.router.name == "gemini"
    assert agent.router.model == "gemini-2.5-flash"
    assert agent.llm.model == "gemini-2.5-flash"

    reply = agent.respond("What model are you using?", history=_history("OLLAMA"))
    assert "REPLY-FROM-GEMINI" in reply

    # A failed switch leaves the previous provider and model active.
    try:
        agent.set_provider("cloudsecret")
    except llm_mod.ProviderConfigError:
        pass
    else:
        raise AssertionError("unconfigured switch did not raise")
    assert agent.router.name == "gemini"
    assert agent.router.model == "gemini-2.5-flash"
    _assert_shared_backend(agent, "failed-switch")
    print("OK: model selection")


# ── 6. State preservation ────────────────────────────────────────────────

def test_state_preserved_across_switches():
    agent = agent_mod.AgentKafle(provider="ollama")
    case_manager, evidence_manager, memory = (
        agent.case_manager,
        agent.evidence_manager,
        agent.memory,
    )

    agent.create_case("State Case", "evidence state survives switching")
    agent.add_evidence("Clue", "found at the scene", source="Lab")
    agent.remember("The lab report was redacted.", source="case-notes")

    active = case_manager.get_active()
    created_id = active.id

    for _ in range(4):
        agent.set_provider("gemini" if agent.provider_name == "ollama" else "ollama")

    # The manager/memory OBJECTS were never recreated.
    assert agent.case_manager is case_manager
    assert agent.evidence_manager is evidence_manager
    assert agent.memory is memory

    # Their DATA survived.
    assert case_manager.get(created_id) is not None
    assert evidence_manager.list_for_case(created_id)
    assert memory.search("redacted")[0]

    return created_id


# ── 7. Lifecycle ─────────────────────────────────────────────────────────

def test_old_provider_closed_on_switch():
    agent = agent_mod.AgentKafle(provider="ollama")
    old = agent.llm
    assert old.closed is False

    agent.set_provider("gemini")
    assert old.closed is True, "old provider was not closed on switch"
    assert "ollama" in _closed

    old2 = agent.llm
    agent.set_provider("ollama")
    assert old2.closed is True, "gemini provider was not closed on switch"
    print("OK: lifecycle")


# ── Run ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    memory_file = os.path.join("memory", "memories.json")
    memory_existed_before = os.path.exists(memory_file)

    test_registry()
    test_exceptions()
    test_switching()
    test_detective_uses_active_provider()
    test_model_selection()
    created_id = test_state_preserved_across_switches()
    test_old_provider_closed_on_switch()

    # Clean up the files this test session created so the repo stays tidy.
    for path in (
        os.path.join("cases", f"{created_id}.json"),
        os.path.join("cases", f"{created_id}.evidence.json"),
    ):
        if os.path.exists(path):
            os.remove(path)
    # Same for the memory file, but only if this run created it.
    if not memory_existed_before and os.path.exists(memory_file):
        os.remove(memory_file)

    print("OK: all provider-switching checks passed")