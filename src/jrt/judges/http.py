"""HTTP judge backends: Ollama, OpenAI-compatible, and Anthropic.

Standard library only — no requests/httpx dependency, so the package installs
clean and runs anywhere.

Failure handling
----------------
Transient failures (HTTP 429, 5xx, timeouts) are retried with capped exponential
backoff inside `_post`; permanent failures (401, 400, malformed bodies) fail
immediately. Error strings are scrubbed through `redact_secrets` before they are
attached to a RawCompletion, so an API key can never leak into stored data.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request

from .base import Judge, RawCompletion

# 429 and the 5xx codes that LLM gateways actually emit under load. 401/403 are
# permanent (retrying cannot fix a bad key) and are deliberately excluded.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_SLEEP_SECONDS = 30.0

# Patch point for tests. Production code uses urllib directly.
_urlopen = urllib.request.urlopen


def redact_secrets(text: str, secrets) -> str:
    """Scrub secret values out of an error string.

    Error bodies and exception messages get stored next to the data. If a proxy
    or a misconfigured gateway ever echoes credentials back, this keeps them out
    of the persisted record.
    """
    for s in secrets:
        if s and s in text:
            text = text.replace(s, "[redacted]")
    return text


def _is_transient_exception(exc: Exception) -> bool:
    """True for timeouts — the one non-HTTP failure worth another attempt."""
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (TimeoutError, socket.timeout))


def _post(
    url: str,
    payload: dict,
    headers: dict[str, str],
    timeout: float,
    max_retries: int = 3,
    backoff: float = 1.5,
    sleep=None,
) -> tuple[dict, str]:
    """POST JSON with capped exponential backoff on transient failures.

    Retries: 429, 5xx gateway errors, and timeouts. Fails immediately on auth
    errors and malformed bodies — those will not get better by waiting.
    `sleep` is injectable so tests can run without real delays; resolved at call
    time (not definition time) so tests can monkeypatch it.
    """
    if sleep is None:
        sleep = time.sleep
    data = json.dumps(payload).encode("utf-8")
    last_err = ""
    for attempt in range(max_retries):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with _urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), ""
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8")[:400]
            except Exception:
                pass
            last_err = f"HTTP {exc.code}: {body}"
            if exc.code not in _RETRYABLE_STATUS:
                return {}, last_err
        except Exception as exc:  # network, DNS, timeout
            last_err = f"{type(exc).__name__}: {exc}"
            if not _is_transient_exception(exc):
                return {}, last_err
        if attempt < max_retries - 1:
            sleep(min(_MAX_SLEEP_SECONDS, backoff * (2**attempt)))
    return {}, last_err


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
        # _post owns transient-failure retries; the base-class retry would only
        # multiply the wait on permanent errors. Override with max_retries=N if
        # outer retries are wanted anyway.
        super().__init__(
            max_retries=kwargs.pop("max_retries", 1),
            backoff=kwargs.pop("backoff", 1.5),
        )
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
        data, err = _post(
            f"{self.host}/api/chat",
            payload,
            {"Content-Type": "application/json"},
            self.timeout,
            max_retries=self.max_retries,
        )
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
        super().__init__(
            max_retries=kwargs.pop("max_retries", 1),
            backoff=kwargs.pop("backoff", 1.5),
        )
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
        data, err = _post(
            f"{self.base_url}/chat/completions",
            payload,
            headers,
            self.timeout,
            max_retries=self.max_retries,
        )
        ms = (time.perf_counter() - t0) * 1000
        if err:
            return RawCompletion("", "", ms, error=redact_secrets(err, [self.api_key]))
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
        super().__init__(
            max_retries=kwargs.pop("max_retries", 1),
            backoff=kwargs.pop("backoff", 1.5),
        )
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
        data, err = _post(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers,
            self.timeout,
            max_retries=self.max_retries,
        )
        ms = (time.perf_counter() - t0) * 1000
        if err:
            return RawCompletion("", "", ms, error=redact_secrets(err, [self.api_key]))
        content = data.get("content") or []
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        return RawCompletion(text, data.get("model", self.model), ms)


__all__ = ["OllamaJudge", "OpenAICompatJudge", "AnthropicJudge", "_post", "redact_secrets"]
