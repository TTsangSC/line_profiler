# SPDX-License-Identifier: Apache-2.0
"""Command-line interface: argument parsing and dispatch only."""
from __future__ import annotations

import argparse

from .backends import DEFAULT_BACKEND, backend_names
from .backends.claude_hook import cmd_hook
from .install import cmd_install, cmd_uninstall, cmd_update
from .record import cmd_record, cmd_reconcile
from .rollup import cmd_rollup, cmd_show
from .version import CANONICAL_REPO


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="llm_resource_tally",
                                description="Measured LLM-usage accounting.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--backend", default=None,
                        help=f"agent backend (default {DEFAULT_BACKEND}; known: "
                             f"{', '.join(backend_names())})")
        sp.add_argument("--transcript", help="explicit session transcript path")
        sp.add_argument("--session", help="session id (transcript filename stem)")
        sp.add_argument("--projects-dir", default=None,
                        help="transcripts base dir (default: backend-specific)")
        sp.add_argument("--label", default=None,
                        help="tag this work (e.g. planning, implementation, review); "
                             "stored as the row's `activity`")
        sp.add_argument("--no-estimate-compaction", action="store_true",
                        help="do not add reconstructed rows for context-compaction events")

    r = sub.add_parser("record", help="attribute new turns to a commit")
    r.add_argument("--commit", default="HEAD")
    r.add_argument("--force", action="store_true")
    common(r)
    r.set_defaults(func=cmd_record)

    rc = sub.add_parser("reconcile", help="sweep un-recorded trailing turns")
    common(rc)
    rc.set_defaults(func=cmd_reconcile)

    ru = sub.add_parser("rollup", help="refresh lifetime totals from the ledger")
    ru.set_defaults(func=cmd_rollup)

    sh = sub.add_parser("show", help="print the ledger")
    sh.set_defaults(func=cmd_show)

    ins = sub.add_parser("install", help="wire git hook + AGENTS.md (offline, idempotent)")
    ins.add_argument("--dir", default=None,
                     help="package dir relative to repo root (default: auto/.llm_resource_tally/tool)")
    ins.add_argument("--hook-mode", choices=["auto", "hookspath", "append", "none"],
                     default="auto", help="how to install the post-commit hook")
    ins.add_argument("--agents-file", default="AGENTS.md",
                     help="doc to carry the managed block (default AGENTS.md)")
    ins.add_argument("--claude", action="store_true",
                     help="also wire a Claude Code PostToolUse(Bash) hook into "
                          ".claude/settings.json for correct cross-repo attribution")
    ins.add_argument("--backend", default=None,
                     help="register a backend the passive hook should record; unioned into "
                          ".llm_resource_tally/settings.json (fresh repos default to claude+codex)")
    ins.set_defaults(func=cmd_install)

    un = sub.add_parser("uninstall", help="remove hook wiring + AGENTS.md block (keeps data)")
    un.add_argument("--dir", default=None)
    un.add_argument("--agents-file", default="AGENTS.md")
    un.set_defaults(func=cmd_uninstall)

    up = sub.add_parser("update", help="re-vendor the latest version (needs network)")
    up.add_argument("--repo", default=CANONICAL_REPO, help="GitHub owner/name source")
    up.add_argument("--ref", default="main", help="tag/branch/sha to install (default main)")
    up.set_defaults(func=cmd_update)

    hk = sub.add_parser("hook", help="internal: Claude PostToolUse handler (reads JSON on stdin)")
    hk.add_argument("--projects-dir", default=None)
    hk.set_defaults(func=cmd_hook)

    args = p.parse_args(argv)
    args.func(args)
