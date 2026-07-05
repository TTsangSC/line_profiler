# SPDX-License-Identifier: Apache-2.0
"""Claude Code PostToolUse(Bash) hook handler — the precise cross-repo recorder.

A git post-commit hook can't identify the Claude session (the CLI exposes no session id
to subprocesses), so when a session in repo A commits into repo B it can't attribute
correctly. Claude Code's *native* PostToolUse hook CAN: it delivers session_id +
transcript_path + cwd (the repo the commit landed in) on stdin. This handler consumes
that payload and records the exact session against the exact repo. Claude-specific.
"""
from __future__ import annotations

import json
import os
import re
import sys
import types

_GIT_COMMIT_RE = re.compile(r"\bgit\b(?:\s+-C\s+\S+)?(?:\s+-c\s+\S+)*\s+commit\b")


def is_git_commit(command: str) -> bool:
    """Heuristic: does this shell command create a commit? (Excludes dry-run/help.)"""
    if not command or not _GIT_COMMIT_RE.search(command):
        return False
    return not re.search(r"--dry-run|--help|(?:^|\s)-h(?:\s|$)", command)


def commit_repo_dir(command: str, cwd: str) -> str:
    """The repo the commit lands in: `git -C <path>` if present, else the tool-call cwd."""
    m = re.search(r"\bgit\s+-C\s+(\"[^\"]+\"|'[^']+'|\S+)", command)
    d = m.group(1).strip("\"'") if m else cwd
    d = os.path.expanduser(d)
    return d if os.path.isabs(d) else os.path.normpath(os.path.join(cwd, d))


def cmd_hook(args) -> None:
    """Read the PostToolUse JSON on stdin; if the call was a `git commit`, record that
    session's turns against the commit in whatever repo it landed in. MUST NEVER raise or
    print — a hook must not disrupt the session — so everything is wrapped and silenced."""
    from ..record import cmd_record
    from ..gitutil import repo_root
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return
    try:
        command = (payload.get("tool_input") or {}).get("command") or ""
        if not is_git_commit(command):
            return
        cwd = payload.get("cwd") or os.getcwd()
        repo_dir = commit_repo_dir(command, cwd)
        transcript = payload.get("transcript_path")
        if not transcript or not os.path.exists(transcript):
            return
        ns = types.SimpleNamespace(
            backend="claude", commit="HEAD", transcript=transcript, session=None,
            projects_dir=args.projects_dir, label=None, force=False,
            no_estimate_compaction=False)
        old_cwd, old_out = os.getcwd(), sys.stdout
        try:
            os.chdir(repo_dir)
            repo_root()                    # raises if repo_dir isn't a git repo -> skip
            sys.stdout = open(os.devnull, "w")
            cmd_record(ns)
        finally:
            try:
                sys.stdout.close()
            except Exception:
                pass
            sys.stdout = old_out
            os.chdir(old_cwd)
    except Exception:
        return
