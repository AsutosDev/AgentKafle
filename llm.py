"""LLM interface for AgentKafle.

This module is the ONLY place that talks to the language model.
Everything else (agent, detective, GUI) calls LLMInterface and gets back text.

Why this design?
- If we later switch from Ollama to a cloud provider, we change ONE file.
- The detective reasoning code (added in later steps) stays independent of
  which model is running underneath.

Provider abstraction:
- LLMInterface is the base class defining the chat() contract and the shared
  provider metadata (name, label, default model, configuration keys).
- OllamaProvider implements it using the OpenAI-compatible client.
- GeminiProvider implements it using Google's GenAI SDK.
- Providers self-register in a global registry, so adding a third provider is
  a single-file change.
- ProviderRouter owns the single "active" provider. The agent and the
  detective both talk through the router, so they can never hold different
  providers at the same time.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ── Shared exceptions ────────────────────────────────────────────────────
# A small, provider-independent error taxonomy. Callers (GUI, agent,
# detective) catch these instead of string-matching raw SDK errors.

class ProviderError(Exception):
    """Base class for every provider error."""


class ProviderConfigError(ProviderError):
    """The provider is missing required configuration (e.g. an API key).

    Raised on construction and for unknown provider names, so callers can
    tell "you must configure this" from "the service is unreachable."
    """


class ProviderConnectionError(ProviderError):
    """The provider could not be reached or returned a failed request."""


# ── Provider registry ────────────────────────────────────────────────────
# Single source of truth: providers register themselves with metadata, and
# the GUI reads its provider menu from here instead of a duplicated dict.

_PROVIDER_REGISTRY = {}


def register_provider(cls):
    """Class decorator: add a provider class to the global registry.

    The class must define ``name`` (the provider key, e.g. "ollama").
    Re-registering a key replaces the previous entry, which is what the test
    suite uses to inject fake providers.
    """
    key = getattr(cls, "name", "")
    if not key:
        raise ValueError(f"Provider class {cls.__name__} must define a name")
    _PROVIDER_REGISTRY[key.lower().strip()] = cls
    return cls


def get_provider_class(provider: str):
    """Look up a provider class by key. Unknown keys raise ProviderConfigError."""
    key = (provider or "").lower().strip()
    cls = _PROVIDER_REGISTRY.get(key)
    if cls is None:
        known = ", ".join(sorted(_PROVIDER_REGISTRY)) or "none"
        raise ProviderConfigError(
            f"Unknown provider: '{provider}'. Registered providers: {known}."
        )
    return cls


def list_providers() -> dict:
    """Return a copy of the registry: provider key -> provider class."""
    return dict(_PROVIDER_REGISTRY)


# ── Base interface ────────────────────────────────────────────────────────

class LLMInterface(ABC):
    """Abstract base class for LLM providers.

    Providers describe themselves with class attributes (``name``, ``label``,
    ``default_model``, ``env_keys``) and implement chat(). The base class
    provides the model-aware constructor, identity helpers, configuration
    status, lazy model discovery, a JSON chat contract, and a close() hook.
    """

    name: Optional[str] = None          # provider key, e.g. "ollama"
    label: Optional[str] = None         # display name, e.g. "OLLAMA"
    default_model: Optional[str] = None # model used when none is specified
    env_keys: tuple = ()                # env variables used to configure this

    def __init__(self, model: Optional[str] = None):
        self.model = model if model is not None else self.default_model

    # ── identity ──

    @property
    def provider_name(self) -> Optional[str]:
        """The provider key (e.g. "ollama"). Same as ``name``."""
        return self.name

    # ── configuration / discovery ──

    @classmethod
    def configured(cls) -> bool:
        """True when required configuration is present.

        Must never make network calls. Subclasses that need an API key
        override this to check their environment variables.
        """
        return True

    def available_models(self) -> list:
        """Return available model IDs, or [] when discovery is impossible.

        Discovery is lazy and failure-safe by design: subclasses that hit a
        network API catch every error and fall back to an empty list so the
        GUI can degrade gracefully (e.g. when Ollama is down).
        """
        return []

    # ── text contracts ──

    @abstractmethod
    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        """Send a list of chat messages and return the model's reply text.

        messages is a list of dicts, e.g.:
            [
                {"role": "system", "content": "You are a detective."},
                {"role": "user", "content": "What is your name?"},
            ]

        When json_mode is True, the provider is asked to return one valid
        JSON object only. The caller must still parse the reply; prompt
        instructions (not just this flag) tell the model which keys to use.
        """
        pass

    def chat_json(self, messages: list[dict]) -> str:
        """Request a structured (single JSON object) reply.

        This is the provider-independent structured-output contract. Each
        provider translates it into its own mechanism internally (Ollama:
        ``response_format``, Gemini: ``response_mime_type``). Subclasses may
        override for a provider-specific implementation.
        """
        return self.chat(messages, json_mode=True)

    def complete(self, prompt: str) -> str:
        """Convenience: send a single user prompt, get back a reply."""
        return self.chat([{"role": "user", "content": prompt}])

    # ── lifecycle ──

    def close(self):
        """Release any client/connection resources. No-op by default."""
        pass


# ── Ollama provider ──────────────────────────────────────────────────────

@register_provider
class OllamaProvider(LLMInterface):
    """Ollama provider using OpenAI-compatible API."""

    name = "ollama"
    label = "OLLAMA"
    default_model = "llama3.2:3b"
    env_keys = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")

    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI

        # Defaults assume Ollama running locally.
        # An explicitly requested model (from the router) wins; otherwise
        # fall back to the env default, then the built-in default.
        model = model or os.getenv("LLM_MODEL") or self.default_model
        super().__init__(model)

        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv("LLM_API_KEY", "ollama")

        # Create the connection. With Ollama, api_key is ignored but OpenAI's
        # client requires the argument, so we pass it anyway. Client creation
        # does not connect to the network; no request is made here.
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @classmethod
    def configured(cls) -> bool:
        """Ollama runs locally; no API key is required."""
        return True

    def available_models(self) -> list:
        """List models from the local Ollama API. Failure-safe."""
        try:
            if self._client is None:
                return []
            response = self._client.models.list()
            return sorted(m.id for m in response.data)
        except Exception:
            return []

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        if self._client is None:
            raise ProviderConnectionError("Ollama provider is closed")

        kwargs = {"model": self.model, "messages": messages}

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Normalize raw SDK failures so callers never have to inspect
            # openai-specific exception types.
            raise ProviderConnectionError(
                f"Ollama request failed: {exc}"
            ) from exc

        content = response.choices[0].message.content
        if content is None:
            raise ProviderConnectionError("Ollama returned an empty response")
        return content

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def close(self):
        """Release the OpenAI client handle."""
        self._client = None


# ── Gemini provider ──────────────────────────────────────────────────────

@register_provider
class GeminiProvider(LLMInterface):
    """Gemini provider using Google GenAI SDK."""

    name = "gemini"
    label = "GEMINI"
    default_model = "gemini-3.6-flash"
    env_keys = ("GEMINI_API_KEY", "GEMINI_MODEL")

    def __init__(self, model: Optional[str] = None):
        from google import genai
        from google.genai import types

        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise ProviderConfigError(
                "Gemini API key is not configured. Set GEMINI_API_KEY and try again."
            )

        model = model or os.getenv("GEMINI_MODEL") or self.default_model
        super().__init__(model)

        self._client = genai.Client(api_key=self.api_key)
        self._types = types

    @classmethod
    def configured(cls) -> bool:
        """Gemini needs a GEMINI_API_KEY to be present."""
        return bool(os.getenv("GEMINI_API_KEY", "").strip())

    def available_models(self) -> list:
        """List Gemini models via the GenAI SDK. Failure-safe."""
        try:
            items = list(self._client.models.list())
            return sorted(getattr(m, "name", "") for m in items if getattr(m, "name", ""))
        except Exception:
            return []

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        if self._client is None:
            raise ProviderConnectionError("Gemini provider is closed")

        # Convert OpenAI-style messages to GenAI format
        # GenAI expects: system_instruction + contents (list of Content)
        system_instruction = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(self._types.Content(role="user", parts=[self._types.Part(text=content)]))
            elif role == "assistant":
                contents.append(self._types.Content(role="model", parts=[self._types.Part(text=content)]))
            else:
                # Treat unknown roles as user
                contents.append(self._types.Content(role="user", parts=[self._types.Part(text=content)]))

        config_kwargs = {}
        if json_mode:
            # Use structured output for JSON mode
            config_kwargs["response_mime_type"] = "application/json"
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = None
        if config_kwargs:
            config = self._types.GenerateContentConfig(**config_kwargs)

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # Handle potential empty response
            if response.text is None:
                raise ProviderConnectionError("Gemini returned an empty response")

            return response.text

        except ProviderError:
            raise
        except Exception as e:
            # Normalize provider errors with context under the shared type.
            raise ProviderConnectionError(f"Gemini API error: {e}") from e

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def close(self):
        """Release the GenAI client handle."""
        self._client = None


# ── Provider factory ─────────────────────────────────────────────────────

def create_llm(provider: str = "ollama", model: Optional[str] = None) -> LLMInterface:
    """Factory function to create the appropriate LLM provider.

    Args:
        provider: any registered provider key (e.g. "ollama", "gemini").
                  Case-insensitive.
        model: optional model name. When omitted, the provider's env default
               (or its built-in default) is used.

    Returns:
        An LLMInterface implementation.

    Raises:
        ProviderConfigError: if the provider is unknown or its required
            configuration (e.g. the Gemini API key) is missing.
    """
    cls = get_provider_class(provider)
    return cls(model=model)


# ── Active provider router ──────────────────────────────────────────────
# The router owns the single active provider. The agent and the detective
# both hold the SAME router and route every request through it, which makes
# "agent and detective somehow use different providers" structurally
# impossible.

class ProviderRouter:
    """Owns the one authoritative LLM provider for the whole app.

    Usage:
        router = ProviderRouter(default="ollama")
        router.switch("gemini", model="gemini-3.6-flash")
        reply = router.chat(messages)
        structured = router.chat_json(messages)

    Switching creates the new provider first and only swaps it in if that
    succeeds, so a failed switch (unknown provider, missing API key) leaves
    the currently active provider untouched.
    """

    def __init__(self, default: str = "ollama", model: Optional[str] = None):
        self._provider: Optional[LLMInterface] = None
        self.switch(default, model=model)

    # ── identity of the active provider ──

    @property
    def provider(self) -> Optional[LLMInterface]:
        """The currently active provider instance (or None after close())."""
        return self._provider

    @property
    def name(self) -> Optional[str]:
        """Provider key of the active provider (e.g. "ollama")."""
        return self._provider.provider_name if self._provider else None

    @property
    def label(self) -> Optional[str]:
        """Display label of the active provider (e.g. "OLLAMA")."""
        return self._provider.label if self._provider else None

    @property
    def model(self) -> Optional[str]:
        """The active provider's current model name."""
        return self._provider.model if self._provider else None

    # ── registry metadata for UI consumption ──

    def configured_providers(self) -> dict:
        """Return provider key -> metadata dict for menus/displays.

        Never makes network calls. Each entry carries the display label,
        default model, required env keys, and the configured() flag, all
        sourced from the registry (single source of truth).
        """
        result = {}
        for key, cls in list_providers().items():
            result[key] = {
                "label": cls.label,
                "default_model": cls.default_model,
                "env_keys": tuple(cls.env_keys or ()),
                "configured": cls.configured(),
            }
        return result

    def available_models(self) -> list:
        """Available models for the active provider (lazy, failure-safe)."""
        if self._provider is None:
            return []
        return self._provider.available_models()

    # ── switching ──

    def switch(self, provider: str, model: Optional[str] = None) -> LLMInterface:
        """Replace the active provider.

        Creates the new provider first. Only after a successful creation is
        the old provider closed and the new one made active; any exception
        propagates and the previous provider stays in place.

        Returns the newly created provider instance.
        """
        cls = get_provider_class(provider)
        new_provider = cls(model=model)

        old_provider = self._provider
        self._provider = new_provider

        if old_provider is not None:
            try:
                old_provider.close()
            except Exception:
                # Closing must never break or roll back a successful switch.
                pass

        return new_provider

    # ── request routing ──

    def chat(self, messages: list[dict]) -> str:
        """Route a normal chat request to the active provider."""
        return self._provider.chat(messages)

    def chat_json(self, messages: list[dict]) -> str:
        """Route a structured/JSON chat request to the active provider."""
        return self._provider.chat_json(messages)

    def complete(self, prompt: str) -> str:
        """Route a single-prompt completion to the active provider."""
        return self._provider.complete(prompt)

    # ── lifecycle ──

    def close(self):
        """Close the active provider and forget it."""
        if self._provider is not None:
            try:
                self._provider.close()
            finally:
                self._provider = None