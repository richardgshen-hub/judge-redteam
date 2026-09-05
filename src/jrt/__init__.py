"""judge-redteam — auditing LLM-as-Judge under surface-form perturbation.

The headline idea: disagreement is not bias. Every verdict flip is decomposed by
direction so that systematic bias (P(correct->wrong) > P(wrong->correct)) can be
separated from sampling variance. See PREREGISTRATION.md.
"""

from .axes import AXES, HYPOTHESES, get_axis
from .runner import Experiment, RunConfig
from .stats import AxisResult, summarise_axis
from .types import Item, Judgment, load_items

__version__ = "0.2.0"

__all__ = [
    "AXES",
    "HYPOTHESES",
    "get_axis",
    "Experiment",
    "RunConfig",
    "AxisResult",
    "summarise_axis",
    "Item",
    "Judgment",
    "load_items",
    "__version__",
]
