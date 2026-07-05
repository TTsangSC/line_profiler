# SPDX-License-Identifier: Apache-2.0
"""Small time helpers. Never compare transcript ('...Z') and git ('+00:00') ISO strings
lexicographically — the 'Z' vs '+00:00' suffix and fractional seconds both break order
across the two sources; parse to aware datetimes with to_dt first."""
from __future__ import annotations

from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_stamp() -> str:
    """Compact UTC stamp (YYYYMMDDTHHMMSS) for lexically-sortable shard archive names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def to_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def span_seconds(lo: str | None, hi: str | None) -> float | None:
    dlo, dhi = to_dt(lo), to_dt(hi)
    if not dlo or not dhi:
        return None
    return round((dhi - dlo).total_seconds(), 1)
