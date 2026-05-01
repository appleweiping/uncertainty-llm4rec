"""Local Hugging Face causal LM provider interface."""

from __future__ import annotations

from llm4rec.llm.base import LLMRequest, LLMResponse


class HFLocalProvider:
    provider_name = "hf_local"
    supports_logprobs = True
    supports_seed = True

    def __init__(self, *, model_name_or_path: str, device: str = "auto") -> None:
        model_name_or_path = str(model_name_or_path or "").strip()
        if not model_name_or_path or model_name_or_path.casefold() in {"none", "null"}:
            raise ValueError("model_name_or_path is required")
        self.model_name = model_name_or_path
        self.model_name_or_path = model_name_or_path
        self.device = device

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "HFLocalProvider requires optional dependencies torch and transformers. "
                "Install them and provide a local model path; Phase 3 smoke tests do not download models."
            ) from exc
        raise NotImplementedError(
            "HFLocalProvider is an interface scaffold in Phase 3; local model loading/generation is reserved for a later approved phase."
        )
