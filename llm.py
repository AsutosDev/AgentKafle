"""LLM interface for AgentKafle.

This module is the ONLY place that talks to the language model.
Everything else (agent, detective, GUI) calls LLMInterface and gets back text.

Why this design?
- If we later switch from Ollama to a cloud provider, we change ONE file.
- The detective reasoning code (added in later steps) stays independent of
  which model is running underneath.

Provider abstraction:
- LLMInterface is the base class defining the chat() contract.
- OllamaProvider implements it using the OpenAI-compatible client.
- GeminiProvider implements it using Google's GenAI SDK.
- A factory function creates the appropriate provider based on configuration.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# ── Base interface ────────────────────────────────────────────────────────

class LLMInterface(ABC):
    """Abstract base class for LLM providers.

    Implementations must provide chat() and complete().
    """

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

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Convenience: send a single user prompt, get back a reply."""
        pass


# ── Ollama provider ──────────────────────────────────────────────────────

class OllamaProvider(LLMInterface):
    """Ollama provider using OpenAI-compatible API."""

    def __init__(self):
        from openai import OpenAI

        # Defaults assume Ollama running locally.
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv("LLM_API_KEY", "ollama")
        self.model = os.getenv("LLM_MODEL", "llama3.2:3b")

        # Create the connection. With Ollama, api_key is ignored but OpenAI's
        # client requires the argument, so we pass it anyway.
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        kwargs = {"model": self.model, "messages": messages}

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])


# ── Gemini provider ──────────────────────────────────────────────────────

class GeminiProvider(LLMInterface):
    """Gemini provider using Google GenAI SDK."""

    def __init__(self):
        from google import genai
        from google.genai import types

        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not configured. Set GEMINI_API_KEY and try again."
            )

        self.model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self.client = genai.Client(api_key=self.api_key)
        self._types = types

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
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
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )

            # Handle potential empty response
            if response.text is None:
                raise ValueError("Gemini returned an empty response")

            return response.text

        except Exception as e:
            # Wrap provider errors with context
            raise RuntimeError(f"Gemini API error: {e}") from e

    def complete(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])


# ── Provider factory ─────────────────────────────────────────────────────

def create_llm(provider: str = "ollama") -> LLMInterface:
    """Factory function to create the appropriate LLM provider.

    Args:
        provider: "ollama" or "gemini" (case-insensitive)

    Returns:
        An LLMInterface implementation.

    Raises:
        ValueError: if provider is unknown or Gemini API key is missing.
    """
    provider = provider.lower().strip()
    if provider == "ollama":
        return OllamaProvider()
    elif provider == "gemini":
        return GeminiProvider()
    else:
        raise ValueError(f"Unknown provider: '{provider}'. Use 'ollama' or 'gemini'.")