"""HTTP judge backends: Ollama, OpenAI-compatible, and Anthropic.

Standard library only — no requests/httpx dependency, so the package installs
clean and runs anywhere.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import Judge, RawCompletion


def _post(url: str, payload: dict, headers: dict[str, str], timeout: float) -> tuple[dict, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), ""
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8")[:400]
        except Exception:
            pass
        return {}, f"HTTP {exc.code}: {body}"
    except Exception as exc:  # network, DNS, timeout
        return {}, f"{type(exc).__name__}: {exc}"


class OllamaJudge(Judge):
    """Local models via the native Ollama endpoint."""

    id = "ollama"
    family = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.id = f"ollama:{model}"
        self.family = model

    def _call(self, prompt: str, temperature: float) -> RawCompletion:
        t0 = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": temperature},
        }
        data, err = _post(f"{self.host}/api/chat", payload, {"Content-Type": "application/json"}, self.timeout)
        ms = (time.perf_counter() - t0) * 1000
        if err:
            return RawCompletion("", "", ms, error=err)
        text = (data.get("message") or {}).get("content", "")
        return RawCompletion(text, data.get("model", self.model), ms)


class OpenAICompatJudge(Judge):
    """Any OpenAI-compatible chat endpoint: OpenAI, DeepSeek, Qwen, Groq, vLLM, together."""

    id = "openai_compat"
    family = "unknown"

    def __init__(
        self,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        family: str | None = None,
        timeout: float = 120.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.id = f"openai:{model}"
        self.family = family or model

    def _call(self, prompt: str, temperature: float) -> RawCompletion:
        t0 = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        data, err = _post(f"{self.base_url}/chat/completions", payload, headers, self.timeout)
        ms = (time.perf_counter() - t0) * 1000
        if err:
            return RawCompletion("", "", ms, error=err)
        choices = data.get("choices") or []
        text = ""
        if choices:
            text = (choices[0].get("message") or {}).get("content", "") or ""
        return RawCompletion(text, data.get("model", self.model), ms)


class AnthropicJudge(Judge):
    """Claude via the native Messages API."""

    id = "anthropic"
    family = "claude"

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.id = f"anthropic:{model}"
        self.family = "claude"

    def _call(self, prompt: str, temperature: float) -> RawCompletion:
        t0 = time.perf_counter()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        data, err = _post("https://api.anthropic.com/v1/messages", payload, headers, self.timeout)
        ms = (time.perf_counter() - t0) * 1000
        if err:
            return RawCompletion("", "", ms, error=err)
        content = data.get("content") or []
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        return RawCompletion(text, data.get("model", self.model), ms)


__all__ = ["OllamaJudge", "OpenAICompatJudge", "AnthropicJudge", "_post"]
