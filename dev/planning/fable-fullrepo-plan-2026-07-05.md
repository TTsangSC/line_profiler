# Fable plan — full-repo remediation & modernization

**This is a living coordination document.** Evidence lives in
`fable-review-fullrepo-2026-07-05.md` (frozen). Same rules as the PR-431
plan: claim tasks on the status board, log what you did in the Agent log,
post decisions you can't make to Open questions. Tasks marked **[hard]**
need a strong agent + human review.

These tasks are independent of PR #431 and target `main`; they are staged
on top of the PR-431 branch only for sequencing convenience (see the
maintainer's merge plan in the Agent log).

## Status board

| ID | Task | Difficulty | Owner | Status |
|----|------|-----------|-------|--------|
| FR-1 | sysmon backend: process-global registration refcount | **[hard]** | fable | in-progress 2026-07-05 |
| FR-2 | `GlobalProfiler.show()`: replace `ppid == 1` heuristic | small | — | todo |
| FR-3 | Generator `@profile` preserves return value | trivial | — | todo |
| FR-4 | AST: skip `*` aliases in `_visit_import` | trivial | — | todo (folds into AST-ROBUST) |
| FR-5 | AST: multi-name import statements profile all matched names | small | — | todo (folds into AST-ROBUST) |
| FR-6 | `LineStats` mixed-unit merge converts to smallest unit | small | — | todo |
| FR-7 | `_line_profiler.pyi` drift | — | — | superseded by TYPE program |
| FR-8 | `c_trace_callbacks.c` refcount fixes | **[hard]** | fable | in-progress 2026-07-05 |
| FR-9 | `_StrEnumBase.__hash__` on 3.10 | trivial | — | todo |
| FR-10 | Guard monitoring callbacks during interpreter finalization | small | — | todo |
| FR-11 | Cosmetic/dead-code bundle (see review F11: docstring env var, dead guards, `if basename`→`if parent`, util_static twins, pyx try/else/finally, inspect monkeypatch lock) | small | — | todo |
| XC-1 | xcookie: step `timeout-minutes` in github_actions builder | small | — | todo (in xcookie repo) |
| XC-2 | xcookie: emit `py_modules` + extra entry-point groups from config; mirror into `[tool.xcookie]` | medium | — | todo (partially mooted by PR431-8) |
| XC-3 | xcookie: pin/bound `ty` in lint job | trivial | — | todo (in xcookie repo) |
| PKG-1 | MANIFEST hygiene: tests completeness, explicit root includes, `global-exclude *.py[cod]`, drop `run_tests.sh` + bogus `py_modules` entry | small | — | todo |
| PKG-2 | `package_data['line_profiler.rc'] = ['*.toml']` | trivial | — | todo |
| PKG-3 | Delete or wire `tests/cython_example/MANIFEST.in` (currently untracked + dead) | trivial | — | todo |
| AST-ROBUST | AST-pipeline robustness program (below) | medium, staged | — | todo |
| TYPE | Remove `.pyi` stubs; modern inline annotations (below) | medium, staged | — | todo |

## Task specs (point fixes)

### FR-1 — sysmon process-global refcount [hard]

Evidence: review F1 (verified repro: cross-thread disable loses data and
raises `ValueError: tool 2 is not in use` inside user `finally` blocks).
Fix in `line_profiler/_line_profiler.pyx`:

- Keep per-thread `_LineProfilerManager`s, but make
  register/deregister of the *global* monitoring state refcounted across
  all threads: only `sys.monitoring.register_callback`/`set_events` on the
  0→1 transition of a process-wide active count, and only
  clear events + `free_tool_id` on the 1→0 transition.
- `deregister()` must tolerate an already-freed tool (no raise).
- `disable()` should clear `_c_last_time` for all threads, not just the
  caller's.
- Regression test FIRST (it exists conceptually in the review repro):
  thread A enable→disable while thread B still profiling; assert B's hits
  are complete and no exception surfaces in B. Must pass under both
  cores (`LINE_PROFILER_CORE=legacy` and sysmon).

### FR-8 — `c_trace_callbacks.c` refcounts [hard]

Evidence: review F8. In `call_callback`: `Py_DECREF` the result of
`PyObject_CallOneArg`/`PyObject_CallMethodOneArg` after
`PyObject_SetAttrString` stores it (SetAttr takes its own reference); on
NULL result, propagate the error instead of calling SetAttr with a live
exception. Same in `set_local_trace`. Remove dead `mod`/`dle` locals.
Verify with a refcount-stability test under `sys.settrace`-style foreign
local traces if feasible, else careful review + valgrind/`--with-pydebug`
spot check.

### FR-2 … FR-6, FR-9 … FR-11, XC-*, PKG-*

Each is fully specified by its review entry (`fable-review-fullrepo-…` A/B
parts); one reviewable commit each. XC tasks are changes to
`/home/joncrall/code/xcookie` (then regenerate this repo's CI); do not
hand-edit generated workflow files here except as a stopgap noted in the
commit message.

## Program: AST-ROBUST — autoprofile AST pipeline robustness

Goal: `--prof-mod`/`--prof-imports`/autoprofile should never crash or
silently skip on legal Python; failures must be loud warnings that name
the file and construct. Maintainer priority (2026-07-05).

Stages:

1. **Fix the known crashers/droppers** (FR-4 star imports, FR-5 multi-name
   imports) with targeted regression tests.
2. **Failure policy:** wrap per-node transformation in a
   warn-and-skip-this-node policy (never abort the whole rewrite for one
   construct); add a `strict` debug switch that re-raises for development.
3. **Corpus test:** a test that runs `AstTreeProfiler`/
   `AstTreeModuleProfiler` over a corpus (every `.py` in `line_profiler/`
   itself + a curated set of gnarly constructs: star imports, multi-name
   imports, conditional imports, `try/except ImportError`, relative
   imports, `__all__` manipulation, walrus, match, async, decorators with
   arguments, nested classes) asserting the rewritten source `compile()`s
   and — for the curated set — executes with identical observable
   behavior to the original.
4. **Coverage of the extractor:** `profmod_extractor` currently only scans
   top-level statements; document that boundary in docstrings, then decide
   (Open question) whether to extend to function-local imports.
5. **De-vendor drift:** `util_static.py` is vendored from
   ubelt/xdoctest with local drift (see F11.4). Either re-sync with
   upstream and record the upstream commit, or claim ownership and delete
   the unreachable branches. Do not leave it ambiguous.

## Program: TYPE — retire `.pyi`, modern inline annotations

Maintainer decision (2026-07-05): **all `.pyi` files go away** in favor of
modern inline type annotations. Current inventory: exactly one stub,
`line_profiler/_line_profiler.pyi` (for the C extension; already drifted —
review F7).

Plan:

1. Pure-Python modules already use inline annotations — audit with a
   strict `mypy`/`ty` pass over `line_profiler/` and fix gaps (several
   modules still have untyped defs; `kernprof.py` uses comment-style hints
   in places).
2. The C extension cannot carry inline annotations that type checkers can
   read. Chosen direction per the maintainer: delete
   `_line_profiler.pyi` and make the *typed public surface* the Python
   layer (`line_profiler/line_profiler.py` subclass + wrappers), treating
   `line_profiler._line_profiler` as private/untyped. Concretely:
   - delete the stub; remove `'*.pyi'` from `package_data` in `setup.py`
     and from MANIFEST.in patterns;
   - annotate the Python `LineProfiler` subclass fully (including members
     it inherits and re-exposes: `enable_count`, `timer_unit`,
     `functions`, `enable/disable`, `__init__(*functions, wrap_trace=…)`),
     so downstream checkers see correct types without the stub;
   - keep `py.typed` (the package remains typed);
   - CI: type-check step must pass with the stub gone (watch for
     `unresolved-import` suppressions in pyproject `[tool.ty]` hiding
     regressions — tighten those once green).
3. Open question below re: whether any downstream users import
   `line_profiler._line_profiler` directly and rely on the stub; if so a
   deprecation note in the changelog is enough (underscore module, no
   compatibility promise).

## Agent log

- **2026-07-04 (fable):** full-repo review performed alongside the PR-431
  review; all findings verified (see review doc).
- **2026-07-05 (fable):** split planning docs; recorded maintainer
  directives: AST-pipeline robustness is a priority; `.pyi` files are to
  be removed in favor of modern annotations. Claimed FR-1 and FR-8 (the
  two [hard] repo-side tasks). Maintainer merge plan for sequencing:
  current branch → PR into `TTsangSC:profile-child-processes` → main;
  repo-level fixes (FR-*) staged as separate commits at the branch tip so
  they can be split into a follow-up PR onto main.

## Open questions

- **TYPE-Q1 (2026-07-05, fable):** removing `_line_profiler.pyi` makes
  `line_profiler._line_profiler` untyped for anyone importing it directly.
  It is underscore-private, so I propose: delete without deprecation
  cycle, one CHANGELOG line. Confirm?
- **AST-Q1 (2026-07-05, fable):** should `profmod_extractor` learn to see
  function-local imports (`def f(): import numpy`), or is top-level-only
  a documented boundary? Extending it changes what `--prof-mod` matches
  and could surprise existing users.
