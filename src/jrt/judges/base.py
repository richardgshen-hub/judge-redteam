"""Judge backend interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..prompts import build_prompt
from ..types import Presentation


@dataclass
class RawCompletion:
    text: str
    model_version: str
    latency_ms: float
    error: str = ""


class Judge(ABC):
    """A backend that can be asked to compare two candidates."""

    id: str = "base"
    family: str = "unknown"

    def __init__(self, max_retries: int = 3, backoff: float = 1.5) -> None:
        self.max_retries = max_retries
        self.backoff = backoff

    def identity_record(self) -> dict[str, object]:
        """Public, non-secret settings that determine what judge was run."""
        return {"id": self.id, "family": self.family, "type": type(self).__name__}

    @abstractmethod
    def _call(self, prompt: str, temperature: float) -> RawCompletion:
        """One attempt. Implementations must never raise."""
        ...

    def complete(self, prompt: str, temperature: float = 0.7) -> RawCompletion:
        last = RawCompletion("", "", 0.0, "no attempts")
        for attempt in range(self.max_retries):
            last = self._call(prompt, temperature)
            if last.text and not last.error:
                return last
            if attempt < self.max_retries - 1:
                time.sleep(self.backoff**attempt)
        return last

    def judge_presentation(
        self,
        presentation: Presentation,
        question: str,
        template: str = "v1_standard",
        extra_instruction: str = "",
        temperature: float = 0.7,
    ) -> RawCompletion:
        """Text-in, text-out path used by real backends.

        Simulated backends override this to score the structured presentation
        directly, which avoids round-tripping ground truth through the prompt.
        """
        prompt = build_prompt(question, presentation, template, extra_instruction)
        return self.complete(prompt, temperature)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} id={self.id} family={self.family}>"


__all__ = ["Judge", "RawCompletion"]
