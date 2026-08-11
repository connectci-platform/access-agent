"""A single replay sample: the response text plus whether the replay errored.

Distinguishing 'the agent errored' from 'the agent returned empty text' is
load-bearing — an errored sample must never be scored as a defense (see the
gate's handling), which a bare '' string could not express.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleResult:
    text: str
    errored: bool = False

    @classmethod
    def error(cls) -> SampleResult:
        return cls(text="", errored=True)
