from __future__ import annotations

from src.llm.api_backend import APIBackend


class DeepSeekBackend(APIBackend):
    def __init__(
        self,
        *,
        model_name: str = "deepseek-chat",
        temperature: float = 0.0,
        max_tokens: int = 300,
        base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        timeout: float | None = None,
        batch_size: int = 1,
        max_concurrency: int | None = None,
    ) -> None:
        super().__init__(
            provider="deepseek",
            model_name=model_name,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
        )
