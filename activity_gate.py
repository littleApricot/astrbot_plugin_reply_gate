"""Accumulate reply interest without clocks or random sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_ACTIVITY = 0.5


def validate_activity(value: object) -> float:
    """Accept a finite activity setting between zero and one."""
    try:
        if isinstance(value, bool):
            raise TypeError
        activity = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid_reply_activity") from exc
    if not math.isfinite(activity) or not 0 <= activity <= 1:
        raise ValueError("invalid_reply_activity")
    return activity


@dataclass(frozen=True)
class ActivityDecision:
    """Keep both the trigger value and the balance after consuming a reply."""

    reply: bool
    before: float
    change: float | None
    accumulated: float
    remaining: float


class ReplyAccumulator:
    """Keep independent balances for AstrBot conversations in this process."""

    def __init__(self) -> None:
        self._balances: dict[tuple[str, str], tuple[float, float]] = {}

    def reset(self, key: tuple[str, str]) -> None:
        """Discard credit after an explicit or fallback reply."""
        self._balances.pop(key, None)

    def evaluate(
        self, key: tuple[str, str], probability: float, activity: float
    ) -> ActivityDecision:
        """Apply one user-message score, consuming credit on admission.

        The exact score endpoints are hard decisions. Activity endpoints take
        precedence over them. A changed activity starts a fresh balance.
        """
        activity = validate_activity(activity)
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("invalid_decision_probability")
        previous_activity, before = self._balances.get(key, (activity, 0.0))
        if previous_activity != activity:
            before = 0.0
        if activity == 0 or activity == 1 or probability == 0 or probability == 1:
            reply = activity == 1 or (activity > 0 and probability == 1)
            self.reset(key)
            return ActivityDecision(reply, before, None, 0.0, 0.0)

        # Equality is mathematically neutral; avoid tiny floating-point drift.
        change = (
            0.0
            if probability == 1 - activity
            else math.log(probability)
            - math.log1p(-probability)
            + math.log(activity)
            - math.log1p(-activity)
        )
        accumulated = max(0.0, before + change)
        reply = change > 0 and accumulated >= 1.0
        remaining = 0.0 if reply else accumulated
        if remaining:
            self._balances[key] = (activity, remaining)
        else:
            self.reset(key)
        return ActivityDecision(reply, before, change, accumulated, remaining)
