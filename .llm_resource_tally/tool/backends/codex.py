# SPDX-License-Identifier: Apache-2.0
"""Codex backend: read ``~/.codex/sessions/**/rollout-*.jsonl`` transcripts.

Codex records token measurements as ``event_msg`` records whose payload type is
``token_count``. The ``last_token_usage`` field is already the per-call delta, so we
aggregate those events directly rather than differencing cumulative totals.
"""
from __future__ import annotations

import glob
import json
import os
import sys

from .base import Backend
from ..gitutil import superproject_root
from ..schema import TOKEN_KEYS


def default_sessions_dir() -> str:
    """``CODEX_SESSIONS_DIR`` overrides outright; else ``<CODEX_HOME or ~/.codex>/sessions``."""
    env = os.environ.get("CODEX_SESSIONS_DIR")
    if env:
        return os.path.expanduser(env)
    cfg = os.path.expanduser(os.environ.get("CODEX_HOME", "~/.codex"))
    return os.path.join(cfg, "sessions")


def _real(path: str | None) -> str | None:
    return os.path.realpath(os.path.expanduser(path)) if path else None


def _path_in(path: str | None, root: str) -> bool:
    path = _real(path)
    root = _real(root)
    return bool(path and root and (path == root or path.startswith(root + os.sep)))


def _iter_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _session_meta(transcript: str) -> dict:
    """Return lightweight metadata needed for discovery without exposing message text."""
    out = {"session_ids": set(), "cwd": None, "workspace_roots": [], "models": []}
    for rec in _iter_jsonl(transcript):
        payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
        rtype = rec.get("type")
        if rtype == "session_meta":
            sid = payload.get("session_id") or payload.get("id")
            if sid:
                out["session_ids"].add(sid)
            if payload.get("cwd") and not out["cwd"]:
                out["cwd"] = payload.get("cwd")
        elif rtype == "turn_context":
            sid = payload.get("turn_id")
            if sid:
                out["session_ids"].add(sid)
            if payload.get("cwd") and not out["cwd"]:
                out["cwd"] = payload.get("cwd")
            roots = payload.get("workspace_roots")
            if isinstance(roots, list):
                out["workspace_roots"].extend(r for r in roots if isinstance(r, str))
            if payload.get("model"):
                out["models"].append(payload.get("model"))
    return out


def _matches_repo(transcript: str, root: str) -> bool:
    meta = _session_meta(transcript)
    if _path_in(meta.get("cwd"), root):
        return True
    for ws in meta.get("workspace_roots", []):
        if _real(ws) == _real(root) or _path_in(root, ws):
            return True
    return False


class CodexBackend(Backend):
    name = "codex"

    def default_projects_dir(self) -> str:
        return default_sessions_dir()

    def _candidates(self, projects_dir: str) -> list[str]:
        return sorted(glob.glob(os.path.join(projects_dir, "**", "*.jsonl"),
                                recursive=True),
                      key=os.path.getmtime, reverse=True)

    def find_transcript(self, projects_dir: str, session: str | None,
                        strict: bool = False) -> str | None:
        candidates = self._candidates(projects_dir)
        if session:
            for c in candidates:
                stem = os.path.splitext(os.path.basename(c))[0]
                if stem == session or session in stem or session in _session_meta(c)["session_ids"]:
                    return c
            if strict:
                return None
            sys.exit(f"error: no Codex transcript for session {session} under {projects_dir}")

        root = superproject_root()
        matches = [c for c in candidates if _matches_repo(c, root)]
        if matches:
            return matches[0]
        if strict:
            return None                      # never fall back to an unrelated Codex session
        if candidates:
            return candidates[0]
        sys.exit(f"error: no Codex session transcripts found under {projects_dir}")

    def session_transcripts(self, projects_dir: str) -> list[str]:
        root = superproject_root()
        return sorted(c for c in self._candidates(projects_dir) if _matches_repo(c, root))

    def parse_turns(self, transcript: str) -> list[dict]:
        by_id: dict[str, dict] = {}
        current_model = "?"
        for rec in _iter_jsonl(transcript):
            payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
            if rec.get("type") == "turn_context" and payload.get("model"):
                current_model = payload["model"]
            if payload.get("type") != "token_count":
                continue
            info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
            usage = info.get("last_token_usage") if isinstance(info.get("last_token_usage"), dict) else {}
            if not usage:
                continue
            ts = rec.get("timestamp")
            if not ts:
                continue
            input_total = int(usage.get("input_tokens", 0) or 0)
            cached = int(usage.get("cached_input_tokens", 0) or 0)
            uncached = max(0, input_total - cached)
            normalized = {
                "input_tokens": uncached,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": cached,
                "output_tokens": int(usage.get("output_tokens", 0) or 0),
            }
            mid = f"{ts}:{input_total}:{cached}:{normalized['output_tokens']}"
            by_id[mid] = {
                "id": mid, "ts": ts, "type": payload.get("type", "?"),
                "model": current_model,
                "usage": {k: normalized.get(k, 0) for k in TOKEN_KEYS},
                "web_search": 0, "web_fetch": 0,
            }
        turns = [t for t in by_id.values() if t["ts"]]
        turns.sort(key=lambda t: t["ts"])
        return turns
