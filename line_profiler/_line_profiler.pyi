from __future__ import annotations

from typing import Any, Mapping

class LineStats:
    timings: Mapping[tuple[str, int, str], list[tuple[int, int, int]]]
    unit: float

    def __init__(
        self,
        timings: Mapping[tuple[str, int, str], list[tuple[int, int, int]]],
        unit: float,
        adjustment_factors: (
            Mapping[tuple[tuple[str, int, str], int], int] | None
        ) = None,
    ) -> None:
        ...

    @property
    def adjusted_timings(self) -> dict[
        tuple[str, int, str], list[tuple[int, int, int]]
    ]:
        ...


class LineProfiler:
    def enable_by_count(self) -> None: ...
    def disable_by_count(self) -> None: ...
    def add_function(self, func: Any) -> None: ...
    def get_stats(self) -> LineStats: ...
    def dump_stats(self, filename: str) -> None: ...


def label(code: Any) -> Any: ...
