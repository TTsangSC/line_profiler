# SPDX-License-Identifier: Apache-2.0
"""Agent-backend interface.

Everything backend-specific — where session transcripts live, how to parse tokens out of
them, whether the agent has a compaction concept — lives behind a Backend. The core
(record/reconcile/rollup, the ledger, git wiring) is backend-agnostic and works with the
NORMALIZED shapes below, so adding Codex or another agent is a new Backend, nothing else.

A `parse_turns` result is a list of turns, each:
    {"id": str, "ts": iso8601, "type": str, "model": str,
     "usage": {"input_tokens", "cache_creation_input_tokens",
               "cache_read_input_tokens", "output_tokens"},
     "web_search": int, "web_fetch": int}

A `parse_compaction_events` result is a list (empty if the backend has no compaction):
    {"boundary_ts": iso8601, "model": str,
     "peak_context_tokens": int, "summary_chars": int}
"""
from __future__ import annotations


class Backend:
    #: value stored in each row's `agent` field
    name = "?"

    def default_projects_dir(self) -> str:
        """Where this backend's session logs live by default."""
        raise NotImplementedError

    def find_transcript(self, projects_dir: str, session: str | None,
                        strict: bool = False) -> str | None:
        """The current (or named) session transcript for this repo. With ``strict=True``,
        match ONLY this repo (or the named session) and return ``None`` if there is no
        confident match — never a global "most recent session" fallback and never exit. The
        passive multi-backend hook uses strict mode so an unrelated session (e.g. a Codex
        session in another repo) is never mis-attributed to this commit."""
        raise NotImplementedError

    def session_transcripts(self, projects_dir: str) -> list[str]:
        """All session transcripts attributable to this repo (for `reconcile` to sweep)."""
        raise NotImplementedError

    def parse_turns(self, transcript: str) -> list[dict]:
        raise NotImplementedError

    def parse_compaction_events(self, transcript: str) -> list[dict]:
        return []  # backends without a compaction concept override nothing
