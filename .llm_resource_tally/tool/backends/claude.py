# SPDX-License-Identifier: Apache-2.0
"""Claude Code backend: read `~/.claude/projects/<encoded-cwd>/<session>.jsonl` transcripts.

Everything here is specific to Claude Code's on-disk format and layout. Other agents
(Codex, etc.) get their own module implementing the same Backend interface.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

from .base import Backend
from ..gitutil import superproject_root
from ..schema import TOKEN_KEYS


def munged_project_dir(path: str) -> str:
    """Reproduce Claude Code's project-dir encoding: transcripts live under
    `<projects>/<encoded-cwd>/` where the cwd is encoded by replacing EVERY
    non-alphanumeric character (`/`, `_`, `.`, space, ...) with `-`, not just `/`.
    (So `/home/u/llm_resource_tally` -> `-home-u-llm-resource-tally`.) Getting this exact
    matters: reconcile globs this dir directly. Encoding is lossy — `_` and `-` both map
    to `-` (Claude's own limitation)."""
    return re.sub(r"[^A-Za-z0-9-]", "-", path)


def default_projects_dir() -> str:
    """`CLAUDE_PROJECTS_DIR` overrides outright; else `<CLAUDE_CONFIG_DIR or ~/.claude>/projects`."""
    env = os.environ.get("CLAUDE_PROJECTS_DIR")
    if env:
        return os.path.expanduser(env)
    cfg = os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude"))
    return os.path.join(cfg, "projects")


def _context_size(usage: dict) -> int:
    return sum(int(usage.get(k, 0) or 0)
               for k in ("input_tokens", "cache_creation_input_tokens",
                         "cache_read_input_tokens"))


def _summary_text(rec: dict) -> str:
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c
                       if isinstance(p, dict) and p.get("type") == "text")
    return ""


class ClaudeBackend(Backend):
    name = "claude-code"

    def default_projects_dir(self) -> str:
        return default_projects_dir()

    def find_transcript(self, projects_dir: str, session: str | None,
                        strict: bool = False) -> str | None:
        """Prefer the project dir munged from the *superproject* cwd (the agent's cwd is
        usually the top-level repo); fall back to a repo-wide scan. Most-recently-modified
        `.jsonl` unless `--session` is given. In ``strict`` mode use ONLY the repo's own
        project dir (no repo-wide fallback) and return ``None`` rather than exiting."""
        proj = os.path.join(projects_dir, munged_project_dir(superproject_root()))
        candidates = sorted(glob.glob(os.path.join(proj, "*.jsonl")),
                            key=os.path.getmtime, reverse=True)
        if not candidates and not strict:
            candidates = sorted(glob.glob(os.path.join(projects_dir, "**", "*.jsonl"),
                                          recursive=True),
                                key=os.path.getmtime, reverse=True)
        if session:
            for c in candidates:
                if os.path.splitext(os.path.basename(c))[0] == session:
                    return c
            if strict:
                return None
            sys.exit(f"error: no transcript for session {session} under {projects_dir}")
        if not candidates:
            if strict:
                return None
            sys.exit(f"error: no session transcripts found under {projects_dir}")
        return candidates[0]

    def session_transcripts(self, projects_dir: str) -> list[str]:
        proj = os.path.join(projects_dir, munged_project_dir(superproject_root()))
        return sorted(glob.glob(os.path.join(proj, "*.jsonl")))

    def parse_turns(self, transcript: str) -> list[dict]:
        """Every billed turn bearing a usage object, deduped by message id. We do NOT
        filter on `type == "assistant"`: any record carrying a `usage` object is a real
        billed API call. Streaming can emit a message id more than once; last wins."""
        by_id: dict[str, dict] = {}
        with open(transcript, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
                usage = (msg.get("usage") or rec.get("usage")) or {}
                if not usage:
                    continue
                mid = (msg.get("id") or rec.get("requestId")
                       or rec.get("uuid") or rec.get("timestamp"))
                st = usage.get("server_tool_use") or {}
                by_id[mid] = {
                    "id": mid, "ts": rec.get("timestamp"),
                    "type": rec.get("type", "?"), "model": msg.get("model", "?"),
                    "usage": {k: int(usage.get(k, 0) or 0) for k in TOKEN_KEYS},
                    "web_search": int(st.get("web_search_requests", 0) or 0),
                    "web_fetch": int(st.get("web_fetch_requests", 0) or 0),
                }
        turns = [t for t in by_id.values() if t["ts"]]
        turns.sort(key=lambda t: t["ts"])
        return turns

    def parse_compaction_events(self, transcript: str) -> list[dict]:
        """`/compact` performs a genuine LLM call but the harness logs NO usage object —
        only a compact_boundary marker + isCompactSummary text. Capture the measured
        signals: peak pre-boundary context read, and summary length in chars."""
        recs: list[dict] = []
        with open(transcript, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        events: list[dict] = []
        peak = 0
        last_model = "?"
        for i, rec in enumerate(recs):
            msg = rec.get("message") if isinstance(rec.get("message"), dict) else {}
            usage = (msg.get("usage") or rec.get("usage")) or {}
            if usage:
                peak = max(peak, _context_size(usage))
                last_model = msg.get("model", last_model)
            if rec.get("type") == "system" and rec.get("subtype") == "compact_boundary":
                summary = ""
                for j in range(i, min(i + 8, len(recs))):
                    if recs[j].get("isCompactSummary"):
                        summary = _summary_text(recs[j])
                        break
                events.append({"boundary_ts": rec.get("timestamp"), "model": last_model,
                               "peak_context_tokens": peak, "summary_chars": len(summary)})
                peak = 0
        return [e for e in events if e["boundary_ts"]]
