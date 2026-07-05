# SPDX-License-Identifier: Apache-2.0
"""Entry point for three invocation styles:

  * ``python -m llm_resource_tally``     (pip / package context) — ``__package__`` is set
  * ``python3 <vendored-dir>``           (run the dir by path)    — no package context
  * the ``llm_resource_tally`` console script                     — bypasses this file

Running the dir by path executes THIS file as top-level ``__main__`` with no package
context, and the vendored directory may have ANY basename (default ``.llm_resource_tally/
tool``, or whatever ``--dir`` chose). So instead of relying on the directory being named
``llm_resource_tally``, we load its files as the canonical package regardless of the
directory's name — which makes relative imports inside the package resolve correctly.
"""
import importlib
import os
import sys

_CANON = "llm_resource_tally"


def _load_main():
    if __package__:                              # -m or genuine package context
        from .cli import main
        return main
    # Run by path: register this directory as the canonical package from its files, so the
    # basename ('tool', 'rt', …) is irrelevant and the package's relative imports work.
    here = os.path.dirname(os.path.abspath(__file__))
    if _CANON not in sys.modules:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            _CANON, os.path.join(here, "__init__.py"),
            submodule_search_locations=[here])
        module = importlib.util.module_from_spec(spec)
        sys.modules[_CANON] = module
        spec.loader.exec_module(module)
    return importlib.import_module(_CANON + ".cli").main


if __name__ == "__main__":
    _load_main()()
