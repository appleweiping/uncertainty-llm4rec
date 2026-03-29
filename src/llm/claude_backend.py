# src/llm/claude_backend.py

from __future__ import annotations

import os
from anthropic import Anthropic

from src.llm.base import LLMBackend


class ClaudeBackend(LLMBackend):
    def __init__(
        self,
        model_name: str = "claude-haiku-4-5-20251001",
        temperature: float = 0.0,
        max_tokens: int = 300,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment variables.")

        self.client = Anthropic(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        texts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                texts.append(block.text)

        return "\n".join(texts).strip()