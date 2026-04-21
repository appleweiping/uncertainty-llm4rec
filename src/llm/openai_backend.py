from __future__ import annotations

from src.llm.api_backend import APIBackend


class OpenAIBackend(APIBackend):
    def __init__(
        self,
        *,
        model_name: str = "gpt-4o-mini",
        base_url: str | None = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int = 300,
        timeout: float | None = None,
        batch_size: int = 1,
        max_concurrency: int | None = None,
    ) -> None:
        super().__init__(
            provider="openai",
            model_name=model_name,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
