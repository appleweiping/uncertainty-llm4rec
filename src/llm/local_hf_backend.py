from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.llm.base import GenerationResult, LLMBackend


class LocalHFBackend(LLMBackend):
    """Transformers backend intended for the server-side main experiment path."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer_name_or_path: str | None = None,
        model_name: str | None = None,
        provider: str = "local_hf",
        device: str = "cuda",
        device_map: str | dict[str, Any] | None = "auto",
        dtype: str = "auto",
        batch_size: int = 1,
        max_new_tokens: int = 300,
        temperature: float = 0.0,
        top_p: float = 1.0,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        adapter_path: str | None = None,
        use_chat_template: bool = True,
        enable_thinking: bool | None = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.tokenizer_name_or_path = str(tokenizer_name_or_path or model_name_or_path)
        self.model_name = str(model_name or Path(self.model_name_or_path).name)
        self.provider = provider
        self.device = device
        self.device_map = device_map
        self.dtype = dtype
        self.batch_size = int(batch_size)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
        self.load_in_4bit = bool(load_in_4bit)
        self.load_in_8bit = bool(load_in_8bit)
        self.adapter_path = str(adapter_path).strip() if adapter_path else None
        self.use_chat_template = bool(use_chat_template)
        self.enable_thinking = enable_thinking

        self._torch = None
        self._tokenizer = None
        self._model = None

    def _uses_single_device(self) -> bool:
        if self.device_map is None:
            return True
        if isinstance(self.device_map, str):
            normalized = self.device_map.strip().lower()
            if normalized in {"", "none"}:
                return True
            if normalized in {"auto", "balanced", "balanced_low_0", "sequential"}:
                return False
            if normalized.startswith(("cuda", "cpu", "mps", "xpu")):
                return True
        return False

    def _get_input_device(self):
        assert self._model is not None
        embedding_layer = getattr(self._model, "get_input_embeddings", None)
        if callable(embedding_layer):
            try:
                embeddings = embedding_layer()
                if embeddings is not None and getattr(embeddings, "weight", None) is not None:
                    return embeddings.weight.device
            except Exception:  # noqa: BLE001 - fall back to parameter device if embeddings are unavailable
                pass

        try:
            return next(self._model.parameters()).device
        except StopIteration as exc:
            raise RuntimeError("Unable to determine LocalHFBackend input device.") from exc

    def _torch_dtype(self):
        torch = self._torch
        dtype = self.dtype.lower()
        if dtype in {"auto", ""}:
            return "auto"
        if dtype in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        if dtype in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported local HF dtype: {self.dtype}")

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LocalHFBackend requires `torch` and `transformers` on the execution server."
            ) from exc

        self._torch = torch
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if self.device_map is not None:
            model_kwargs["device_map"] = self.device_map
        torch_dtype = self._torch_dtype()
        if torch_dtype != "auto":
            model_kwargs["dtype"] = torch_dtype
        else:
            model_kwargs["dtype"] = "auto"
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        if self.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError("adapter_path was set, but `peft` is not installed.") from exc
            model = PeftModel.from_pretrained(model, self.adapter_path)

        if self._uses_single_device() and self.device != "auto":
            model = model.to(self.device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    def _format_prompt(self, prompt: str) -> str:
        tokenizer = self._tokenizer
        if (
            self.use_chat_template
            and tokenizer is not None
            and getattr(tokenizer, "chat_template", None)
        ):
            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.enable_thinking is not None:
                template_kwargs["enable_thinking"] = bool(self.enable_thinking)
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **template_kwargs,
                )
            except TypeError:
                template_kwargs.pop("enable_thinking", None)
                return tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **template_kwargs,
                )
        return prompt

    def generate(self, prompt: str, **kwargs) -> dict[str, Any]:
        return self.batch_generate([prompt], **kwargs)[0]

    def batch_generate(self, prompts: list[str], **kwargs) -> list[dict[str, Any]]:
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        max_new_tokens = int(kwargs.get("max_new_tokens", self.max_new_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))
        top_p = float(kwargs.get("top_p", self.top_p))
        batch_size = int(kwargs.get("batch_size", self.batch_size))

        results: list[dict[str, Any]] = []
        for start_idx in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start_idx : start_idx + batch_size]
            formatted_prompts = [self._format_prompt(prompt) for prompt in batch_prompts]
            start = time.perf_counter()
            inputs = self._tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_device = self._get_input_device()
            inputs = {
                key: value.to(input_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "top_p": top_p,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature

            with self._torch.inference_mode():
                outputs = self._model.generate(**inputs, **generation_kwargs)
            latency = time.perf_counter() - start

            input_lengths = inputs["input_ids"].shape[1]
            decoded = self._tokenizer.batch_decode(
                outputs[:, input_lengths:],
                skip_special_tokens=True,
            )
            per_item_latency = latency / max(len(batch_prompts), 1)
            for text in decoded:
                results.append(
                    GenerationResult(
                        raw_text=str(text).strip(),
                        latency=per_item_latency,
                        model_name=self.model_name,
                        provider=self.provider,
                        usage={},
                        raw_response=None,
                    ).to_dict()
                )

        return results
