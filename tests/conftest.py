"""Shared test fixtures.

The default pytest tmp_path root (`/private/var/folders/.../pytest-of-*`) is
unreliable in some sandboxed environments (its mkdir gets brokered and denied).
This override hands tests a plain /tmp directory instead, which keeps the suite
self-contained and deterministic.
"""

from __future__ import annotations

import shutil
import tempfile

import pytest


@pytest.fixture
def tmp_path():
    d = tempfile.mkdtemp(prefix="jrt-test-", dir="/tmp")
    yield d
    shutil.rmtree(d, ignore_errors=True)
