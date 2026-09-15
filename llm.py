"""LLM interface for AgentKafle.

This module is the ONLY place that talks to the language model.
Everything else (agent, detective, GUI) calls LLMInterface and gets back text.

Why this design?
- If we later switch from Ollama to a cloud provider, we change ONE file.
- The detective reasoning code (added in later steps) stays independent of
  which model is running underneath.

We use the OpenAI Python client because Ollama (and most other providers)
expose an "OpenAI-compatible" API. That means the same code works whether the
model runs locally on your Mac or on someone's cloud servers.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI


class LLMInterface:
    """A thin wrapper around any OpenAI-compatible language model."""

    def __init__(self):
        # Load values from the .env file (never hardcoded in code).
        # Settings like the model name and server address live in .env so
        # you can change providers without touching this file.
        load_dotenv()

        # Defaults assume Ollama running locally.
        self.base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv("LLM_API_KEY", "ollama")
        self.model = os.getenv("LLM_MODEL", "llama3.2:3b")

        # Create the connection. With Ollama, api_key is ignored but OpenAI's
        # client requires the argument, so we pass it anyway.
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, messages):
        """Send a list of chat messages and return the model's reply text.

        messages is a list of dicts, e.g.:
            [
                {"role": "system", "content": "You are a detective."},
                {"role": "user", "content": "What is your name?"},
            ]
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        # The reply is nested inside the response. We just want the text.
        return response.choices[0].message.content

    def complete(self, prompt):
        """Convenience: send a single user prompt, get back a reply."""
        return self.chat([{"role": "user", "content": prompt}])