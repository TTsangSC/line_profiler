# SPDX-License-Identifier: Apache-2.0
"""Version + canonical source. The VENDORED copy is the source of truth: everything
needed to (re)install is committed into the host repo, so it works with zero network.
Hosting is only a convenience for the first fetch and for `update`."""
from __future__ import annotations

import os

CANONICAL_REPO = "Erotemic/llm_resource_tally"


def tool_version() -> str:
    """Vendored/submodule: the VERSION file written next to the package. Pip: that file
    isn't installed, so fall back to the installed distribution's metadata."""
    try:
        with open(os.path.join(os.path.dirname(__file__), "VERSION"), encoding="utf-8") as fh:
            return fh.read().strip() or "0.0.0"
    except OSError:
        try:
            from importlib.metadata import version, PackageNotFoundError
            try:
                return version("llm_resource_tally")
            except PackageNotFoundError:
                return "0.0.0"
        except Exception:
            return "0.0.0"
