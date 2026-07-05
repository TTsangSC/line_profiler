# SPDX-License-Identifier: Apache-2.0
"""llm_resource_tally — measured, per-commit LLM resource accounting for a git repo.

Stdlib-only. Package layout:
  cli.py            argument parsing / dispatch
  record.py         record / reconcile (backend-agnostic)
  rollup.py         rollup / show (post-hoc passes over the ledger)
  install.py        install / uninstall / update / vendor + git-hook & AGENTS wiring
  ledger.py         rolling JSONL shards, read/dedup/append, aggregate
  schema.py         compact on-disk row codec (<-> rich in-memory rows)
  gitutil.py        git helpers (repo_root anchors the ledger)
  backends/         agent-specific transcript readers
"""
from .version import tool_version                       # noqa: F401
from .backends import get_backend                       # noqa: F401
from .backends.claude import munged_project_dir         # noqa: F401
from .ledger import read_ledger                         # noqa: F401
from .cli import main                                   # noqa: F401

__all__ = ["main", "tool_version", "get_backend", "munged_project_dir", "read_ledger"]
