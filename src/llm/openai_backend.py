# src/llm/openai_backend.py
from __future__ import annotations

import os

from openai import OpenAI

from src.llm.base import LLMBackend


class OpenAIBackend(LLMBackend):
    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in environment variables.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        return response.output_text