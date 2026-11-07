"""On-device chat model. No network calls."""
from __future__ import annotations

import os
import threading
from pathlib import Path

_LOCK = threading.Lock()
_LLM = None
_LOADED_PATH = ""


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


def resolve_model_path(path: str | None = None) -> Path:
    raw = path or os.getenv(
        "LOCAL_LLM_PATH",
        "models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    )
    model = Path(raw).expanduser()
    if not model.is_absolute():
        model = Path(__file__).resolve().parent.parent / model
    return model


def get_llama(path: str | None = None):
    """Load the GGUF once and reuse it for scripts and live replies."""
    global _LLM, _LOADED_PATH
    model = resolve_model_path(path)
    if not model.is_file():
        raise FileNotFoundError(
            f"Local answer model was not found: {model}. "
            "Download qwen2.5-1.5b-instruct-q4_k_m.gguf into models/llm."
        )
    with _LOCK:
        if _LLM is not None and _LOADED_PATH == str(model):
            return _LLM
        from llama_cpp import Llama

        _LLM = Llama(
            model_path=str(model),
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            n_batch=128,
            verbose=False,
        )
        _LOADED_PATH = str(model)
        return _LLM


class LocalChatClient:
    """Small stand-in for the Groq chat client, backed by llama.cpp."""

    def __init__(self, model_path: str | None = None):
        self._model_path = model_path
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer: "LocalChatClient"):
            self.completions = outer._Completions(outer)

    class _Completions:
        def __init__(self, outer: "LocalChatClient"):
            self._outer = outer

        def create(self, **kwargs) -> _Completion:
            messages = list(kwargs.get("messages") or [])
            max_tokens = int(kwargs.get("max_tokens") or 80)
            temperature = float(kwargs.get("temperature") if kwargs.get("temperature") is not None else 0.7)
            llm = get_llama(self._outer._model_path)
            result = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = ""
            choices = result.get("choices") if isinstance(result, dict) else None
            if choices:
                message = choices[0].get("message") or {}
                content = str(message.get("content") or "").strip()
            return _Completion(content)
