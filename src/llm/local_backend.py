from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.llm.base import GenerationResult, LLMBackend


def _maybe_path(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class LocalHFBackend(LLMBackend):
    def __init__(
        self,
        *,
        model_name: str,
        model_path: str,
        tokenizer_path: str | None = None,
        provider: str = "local_hf",
        dtype: str = "auto",
        device_map: str | dict[str, Any] | None = "auto",
        max_tokens: int = 300,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: bool | None = None,
        use_chat_template: bool = True,
        trust_remote_code: bool = False,
    ) -> None:
        self.provider = provider
        self.backend_type = "local_hf"
        self.model_name = model_name
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.dtype = dtype
        self.device_map = device_map
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.use_chat_template = use_chat_template
        self.trust_remote_code = trust_remote_code

        self._model = None
        self._tokenizer = None
        self._torch = None

    def _resolve_dtype(self, torch_module):
        dtype_name = str(self.dtype or "auto").strip().lower()
        if dtype_name in {"", "auto"}:
            return "auto"
        mapping = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported local backend dtype: {self.dtype}")
        return mapping[dtype_name]

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LocalHFBackend requires `transformers` and `torch`. "
                "Install them before using backend_name=local_hf."
            ) from exc

        self._torch = torch

        model_path = _maybe_path(self.model_path)
        tokenizer_path = _maybe_path(self.tokenizer_path)
        if not model_path:
            raise ValueError("LocalHFBackend requires a non-empty model_path.")

        tokenizer_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(self.trust_remote_code),
        }
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path, **tokenizer_kwargs)

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(self.trust_remote_code),
        }
        resolved_dtype = self._resolve_dtype(torch)
        if resolved_dtype != "auto":
            model_kwargs["torch_dtype"] = resolved_dtype
        if self.device_map not in {None, "", "none"}:
            model_kwargs["device_map"] = self.device_map

        self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self._model.eval()

    def _build_model_input(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt

        tokenizer = self._tokenizer
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            return prompt

        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def _resolve_do_sample(self, temperature: float, do_sample: bool | None) -> bool:
        if do_sample is not None:
            return bool(do_sample)
        return temperature > 0.0

    def generate(self, prompt: str, **kwargs) -> dict[str, Any]:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None
        assert self._torch is not None

        generation_model_name = str(kwargs.get("model_name", self.model_name) or self.model_name)
        max_new_tokens = int(kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_tokens)))
        temperature = float(kwargs.get("temperature", self.temperature))
        top_p = float(kwargs.get("top_p", self.top_p))
        do_sample = self._resolve_do_sample(temperature, kwargs.get("do_sample", self.do_sample))

        prompt_text = self._build_model_input(prompt)
        tokenizer_outputs = self._tokenizer(prompt_text, return_tensors="pt")
        if not hasattr(self._model, "hf_device_map"):
            model_device = getattr(self._model, "device", None)
            if model_device is not None:
                tokenizer_outputs = {
                    key: value.to(model_device) if hasattr(value, "to") else value
                    for key, value in tokenizer_outputs.items()
                }

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        start = time.perf_counter()
        with self._torch.inference_mode():
            output_ids = self._model.generate(**tokenizer_outputs, **generate_kwargs)
        latency = time.perf_counter() - start

        input_ids = tokenizer_outputs["input_ids"]
        prompt_length = input_ids.shape[-1]
        generated_ids = output_ids[0][prompt_length:]
        raw_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        result = GenerationResult(
            raw_text=raw_text,
            latency=latency,
            model_name=generation_model_name,
            provider=self.provider,
            backend_type=self.backend_type,
            usage={
                "prompt_tokens": int(prompt_length),
                "completion_tokens": int(generated_ids.shape[-1]),
                "total_tokens": int(prompt_length + generated_ids.shape[-1]),
            },
            raw_response=None,
        )
        return result.to_dict()
