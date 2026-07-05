# SPDX-License-Identifier: Apache-2.0
"""Per-repo settings, committed at `.llm_resource_tally/settings.json`.

Chiefly the list of **backends the passive git hook should record**. A bare `record` /
`reconcile` (no `--backend`) walks this list and records whichever agent actually has a
session matching the repo — so a Codex repo (or a mixed one) auto-records without baking a
backend name into the hook. The file is a committed, hand-editable JSON object; stdlib only.
"""
from __future__ import annotations

import json
import os

from .backends import backend_names
from .ledger import data_dir, ensure_data_dir

#: Backends a fresh repo records passively out of the box. Both agentic CLIs we support are
#: on by default; strict matching means a backend with no session for this repo simply records
#: nothing, so enabling one you don't use is harmless. A curated settings.json is respected.
DEFAULT_BACKENDS = ["claude", "codex"]


def settings_path() -> str:
    return os.path.join(data_dir(), "settings.json")


def read_settings() -> dict:
    try:
        with open(settings_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def registered_backends() -> list[str]:
    """Backends the passive hook should try, in order. A repo with no settings yet defaults to
    DEFAULT_BACKENDS. Unknown names are dropped (forward/back compatible)."""
    known = set(backend_names())
    names = read_settings().get("backends")
    if isinstance(names, list):
        valid = [n for n in names if isinstance(n, str) and n in known]
        if valid:
            return list(dict.fromkeys(valid))
    return list(DEFAULT_BACKENDS)


def register_backend(name: str | None) -> list[str]:
    """Add `name` (if given) to the repo's registered backends; write settings.json and return
    the list. A repo with no settings yet is seeded with DEFAULT_BACKENDS (both on by default);
    an existing, curated list is respected and only unioned with `name`. Union/idempotent —
    safe to re-run. Only known backend names are kept."""
    known = set(backend_names())
    data = read_settings()
    existing = data.get("backends")
    names = ([n for n in existing if isinstance(n, str)] if isinstance(existing, list)
             else list(DEFAULT_BACKENDS))
    if name and name not in names:
        names.append(name)
    names = [n for n in dict.fromkeys(names) if n in known] or list(DEFAULT_BACKENDS)
    data["backends"] = names
    ensure_data_dir()
    with open(settings_path(), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return names
