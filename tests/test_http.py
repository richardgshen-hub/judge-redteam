"""Tests for the HTTP judge backends.

Problem 5 (engineering quality). The transport layer must retry transient
failures (429, 5xx, timeouts) with capped exponential backoff, fail fast on
permanent ones (401, malformed bodies), and never let an API key leak into an
error string that gets persisted. All sleeps are injected, so these tests run
instantly and deterministically.
"""

from __future__ import annotations

import io
import json
import socket
import urllib.error

import pytest

from jrt.judges import http as jhttp
from jrt.judges.http import OpenAICompatJudge, _post, redact_secrets


class FakeResp:
    def __init__(self, body: bytes):
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(code: int, body: bytes = b"") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://test", code=code, msg="err", hdrs={}, fp=io.BytesIO(body)
    )


class Recorder:
    def __init__(self, script):
        # script: list of outcomes, one per call — a FakeResp or an Exception.
        self.script = list(script)
        self.calls = 0

    def __call__(self, request, timeout=None):
        self.calls += 1
        outcome = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def no_network(monkeypatch):
    """Point the module's urlopen at a Recorder; returns (recorder, sleeps)."""
    sleeps: list[float] = []

    def install(script):
        rec = Recorder(script)
        monkeypatch.setattr(jhttp, "_urlopen", rec)
        monkeypatch.setattr(jhttp.time, "sleep", lambda s: sleeps.append(s))
        return rec

    install.sleeps = sleeps
    return install


URL = "http://test/endpoint"
HEADERS = {"Content-Type": "application/json"}


def test_success_parses_body(no_network):
    rec = no_network([FakeResp(json.dumps({"ok": True}).encode())])
    data, err = _post(URL, {}, HEADERS, 5.0)
    assert data == {"ok": True}
    assert err == ""
    assert rec.calls == 1


def test_401_fails_immediately_without_retry(no_network):
    rec = no_network([_http_error(401, b'{"error":"bad key"}')])
    data, err = _post(URL, {}, HEADERS, 5.0)
    assert data == {}
    assert "HTTP 401" in err
    assert rec.calls == 1, "auth errors must not be retried"
    assert install_sleeps(no_network) == []


def install_sleeps(no_network):
    return no_network.sleeps


def test_429_retries_then_succeeds(no_network):
    rec = no_network([_http_error(429), FakeResp(json.dumps({"ok": 1}).encode())])
    data, err = _post(URL, {}, HEADERS, 5.0)
    assert data == {"ok": 1}
    assert rec.calls == 2
    assert install_sleeps(no_network) == [1.5], "one backoff sleep between attempts"


def test_persistent_500_exhausts_retries_with_backoff(no_network):
    rec = no_network([_http_error(500, b"boom")] * 9)
    data, err = _post(URL, {}, HEADERS, 5.0, max_retries=3)
    assert data == {}
    assert "HTTP 500" in err
    assert rec.calls == 3
    # Exponential: 1.5, then 3.0. Capped, never unbounded.
    assert install_sleeps(no_network) == [1.5, 3.0]


def test_timeout_is_retried(no_network):
    timeout_err = urllib.error.URLError(socket.timeout("timed out"))
    rec = no_network([timeout_err, FakeResp(json.dumps({"ok": 1}).encode())])
    data, err = _post(URL, {}, HEADERS, 5.0)
    assert data == {"ok": 1}
    assert rec.calls == 2


def test_malformed_body_fails_without_retry(no_network):
    rec = no_network([FakeResp(b"this is not json")])
    data, err = _post(URL, {}, HEADERS, 5.0)
    assert data == {}
    assert "JSONDecodeError" in err or "Expecting" in err
    assert rec.calls == 1


def test_backoff_is_capped(no_network):
    rec = no_network([_http_error(503)] * 9)
    _post(URL, {}, HEADERS, 5.0, max_retries=8, backoff=10.0)
    sleeps = install_sleeps(no_network)
    assert all(s <= jhttp._MAX_SLEEP_SECONDS for s in sleeps)


def test_redact_secrets_scrubs_key():
    text = "HTTP 401: key sk-secret123 rejected by upstream"
    out = redact_secrets(text, ["sk-secret123"])
    assert "sk-secret123" not in out
    assert "[redacted]" in out


def test_judge_error_never_contains_api_key(no_network):
    # A misconfigured gateway that echoes the credential back must not leak it
    # into the persisted error string.
    leaked = b'{"error":"invalid key sk-secret1234"}'
    no_network([_http_error(401, leaked)])
    judge = OpenAICompatJudge(model="gpt-test", api_key="sk-secret1234", base_url="http://test/v1")
    comp = judge._call("prompt", 0.7)
    assert comp.error
    assert "sk-secret1234" not in comp.error
    assert "[redacted]" in comp.error


def test_judge_default_retries_transient_transport_error(no_network):
    rec = no_network([
        _http_error(429),
        FakeResp(json.dumps({"model": "gpt-test", "choices": []}).encode()),
    ])
    judge = OpenAICompatJudge(model="gpt-test", base_url="http://test/v1")
    comp = judge._call("prompt", 0.7)
    assert comp.error == ""
    assert rec.calls == 2


def test_judge_identity_never_contains_api_key():
    judge = OpenAICompatJudge(model="gpt-test", api_key="sk-secret1234")
    identity = json.dumps(judge.identity_record())
    assert "sk-secret1234" not in identity
