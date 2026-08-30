"""A judge with injected, known biases.

This exists for one reason: to calibrate the measuring instrument. If we inject
a bias of known magnitude and the statistical pipeline recovers it, we have
evidence that the pipeline detects real bias rather than manufacturing
significance out of noise. If we inject *no* bias and the pipeline reports
"no significant net bias", we have evidence it does not cry wolf.

Both directions matter and both are exercised in `scripts/selfcheck.py`.

The judge scores candidates on surface features only. Its ability to prefer the
ground-truth side is governed by `competence` (0.5 = coin flip, 1.0 = perfect),
which stands in for whatever internal machinery a real model uses.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, replace

from ..types import Presentation
from .base import Judge, RawCompletion

_AUTHORITY_MARKERS = (
    "it is well established",
    "research consistently",
    "clearly,",
    "without serious dispute",
    "any careful analysis",
)
_COT_MARKERS = ("let me work through", "step 1", "step 2", "step 3", "therefore:")
_ABSTENTION_MARKERS = (
    "i am not confident",
    "i don't know",
    "i do not know",
    "rather than guess",
    "would want to verify",
)
_MD_MARKERS = ("## ", "### ", "**", "\n- ")


@dataclass
class BiasProfile:
    """Injected bias magnitudes. All default to zero (an unbiased judge)."""

    competence: float = 0.85
    competence_spread: float = 0.15
    position: float = 0.0
    length: float = 0.0
    authority: float = 0.0
    format: float = 0.0
    verbose_cot: float = 0.0
    abstention: float = 0.0
    self_preference: float = 0.0
    noise: float = 0.15
    parse_fail_rate: float = 0.0

    def scaled(self, factor: float) -> "BiasProfile":
        kwargs = {}
        for k, v in self.__dict__.items():
            if k in ("competence", "competence_spread", "noise", "parse_fail_rate"):
                kwargs[k] = v
            else:
                kwargs[k] = v * factor
        return BiasProfile(**kwargs)


def _has(text: str, markers: tuple[str, ...]) -> float:
    low = text.lower()
    return 1.0 if any(m in low for m in markers) else 0.0


def surface_score(text: str, label: str, profile: BiasProfile, own_family: str) -> float:
    s = 0.0
    # Length bias saturates: past roughly a screen of text, extra words stop
    # buying extra credit. Without the cap, any perturbation that makes text
    # longer scales without bound and the comparison between the length axis and
    # the verbose-CoT axis becomes meaningless.
    s += profile.length * min(len(text) / 600.0, 1.0)
    s += profile.authority * _has(text, _AUTHORITY_MARKERS)
    s += profile.format * _has(text, _MD_MARKERS)
    s += profile.verbose_cot * _has(text, _COT_MARKERS)
    s -= profile.abstention * _has(text, _ABSTENTION_MARKERS)
    s += profile.self_preference * (1.0 if own_family.lower() in text.lower() else 0.0)
    if label == "A":
        s += profile.position
    return s


class SimulatedJudge(Judge):
    id = "simulated"
    family = "Simulated"

    def __init__(
        self,
        profile: BiasProfile | None = None,
        seed: int = 20260830,
        name: str = "",
        family: str = "Simulated",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.profile = profile or BiasProfile()
        self._rng = random.Random(seed)
        self.id = name or type(self).id
        self.family = family
        self._own_family = family

    def _item_competence(self, question: str) -> float:
        """Per-item discrimination ability, derived deterministically from the question.

        Real judges are not uniformly good: near ceiling on easy items, near
        chance on hard ones. A flat competence is unrealistic and, more
        importantly, it hides positional effects — a constant large quality gap
        swamps any slot preference. Item-level spread is what lets a weak bias
        show up precisely where the judge is genuinely uncertain, which is where
        real-world bias bites.
        """
        if not question:
            return self.profile.competence
        h = hashlib.sha256(question.encode("utf-8")).hexdigest()
        u = int(h[:8], 16) / 0xFFFFFFFF
        jitter = (u - 0.5) * 2.0 * self.profile.competence_spread
        return min(1.0, max(0.0, self.profile.competence + jitter))

    def score_presentation(
        self, presentation: Presentation, profile: BiasProfile | None = None
    ) -> dict[str, float]:
        prof = profile or self.profile
        scores: dict[str, float] = {}
        for label, text in presentation.blocks:
            quality = 1.0 if label == presentation.preferred else 0.0
            base = prof.competence * quality
            noise = self._rng.gauss(0.0, prof.noise)
            scores[label] = base + surface_score(text, label, prof, self._own_family) + noise
        return scores

    def judge_presentation(
        self,
        presentation: Presentation,
        question: str = "",
        template: str = "v1_standard",
        extra_instruction: str = "",
        temperature: float = 0.7,
    ) -> RawCompletion:
        """Score the structured presentation directly.

        Overriding here rather than round-tripping through the prompt means
        ground truth never has to be smuggled into the text the judge reads —
        which would make the simulation meaningless.
        """
        t0 = time.perf_counter()
        elapsed = lambda: (time.perf_counter() - t0) * 1000  # noqa: E731
        del template, extra_instruction
        if self._rng.random() < self.profile.parse_fail_rate:
            return RawCompletion(
                text="I'm afraid I can't compare those.",
                model_version="simulated-1.0",
                latency_ms=elapsed(),
            )
        # Higher sampling temperature means more sampling variance.
        comp = self._item_competence(question)
        effective = replace(
            self.profile, noise=self.profile.noise * (0.4 + temperature), competence=comp
        )
        scores = self.score_presentation(presentation, effective)
        best = max(scores, key=lambda k: scores[k])
        margin = abs(scores.get("A", 0.0) - scores.get("B", 0.0))
        verdict = best if margin >= 0.05 else "TIE"
        return RawCompletion(
            text=f"VERDICT: {verdict}\nREASON: simulated preference.",
            model_version=f"simulated-1.0(temp={temperature})",
            latency_ms=elapsed(),
        )

    def _call(self, prompt: str, temperature: float = 0.7) -> RawCompletion:
        raise NotImplementedError(
            "SimulatedJudge cannot operate on raw prompt text: ground truth is not "
            "recoverable from a prompt. Use judge_presentation() instead."
        )


__all__ = ["BiasProfile", "SimulatedJudge", "surface_score"]
