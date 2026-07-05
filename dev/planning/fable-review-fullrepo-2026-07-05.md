# Fable review — full repo (`line_profiler` beyond PR #431)

- **Reviewer:** Claude (Fable 5). Review performed 2026-07-04; reorganized
  into `dev/planning/` 2026-07-05. Findings here exist on `main`
  independently of PR #431 (except Part B, which is generated-CI hygiene
  surfaced by the PR). PR-specific findings live in
  `fable-review-pr431-2026-07-05.md`. Remediation task specs and live
  status live in `fable-fullrepo-plan-2026-07-05.md` — update THAT file;
  this review is frozen evidence.
- Every finding below was verified by a local repro or by direct reading of
  the cited lines; several were re-verified independently after being
  surfaced by a delegated deep-dive.

---

## Part A — Core library findings

### F1 (HIGH): `sys.monitoring` backend loses data and raises in user code under threads

Enable/disable state is per-thread (`_line_profiler.pyx:1066` `_managers`
keyed by thread id; `:1248-1271` `_manager` property) but
`sys.monitoring.set_events()`/`free_tool_id()` are process-global
(`:436-449` `_SysMonitoringState.deregister`). When the first-enabling
thread's `enable_count` reaches zero while another thread is still
profiling, `deregister()` clears the global event set and frees tool id 2.

**Verified repro (3.12, default sysmon core):** thread A enables and later
disables while thread B is still profiling → (a) B's remaining line hits
are silently lost (2000 hits recorded vs 3000 expected; legacy-trace core
records all 3000), and (b) B's subsequent `disable_by_count()` raises
`ValueError: tool 2 is not in use` — and since `@profile` wrappers call
`disable_by_count()` in a `finally:`, the exception surfaces *inside user
code* (e.g. `ThreadPoolExecutor` tasks). Also `disable()` (`:1360`) clears
`_c_last_time` only for the calling thread.

This is the highest-value core fix: 3.12+ defaults to sysmon, and PR #431's
threading story sits on top of this machinery.

### F2 (HIGH): `GlobalProfiler.show()` silently discards all output when orphaned

`explicit_profiler.py:466-469` (from PR #415):
`if os.getppid() == 1 and owner_env == str(os.getpid()): return` skips the
atexit output writer whenever the owner process's parent has died or is
init/PID 1. **Verified repro:** `LINE_PROFILE=1` script whose parent exits
first (daemonized/`nohup`/ssh-disconnect/container-PID-1 patterns) writes
*no* `profile_output.txt`/`.lprof`/stdout summary, with no diagnostic. The
guard was meant to suppress duplicate output from mp helpers; it should
test for that condition directly (`multiprocessing.parent_process()`), not
via the `ppid == 1` heuristic.

### F3 (MEDIUM): `@profile` on a generator loses the generator's return value

`profiler_mixin.py:550-551`: `except StopIteration: return` discards
`StopIteration.value`. **Verified repro:** a profiled generator with
`return 'RETVAL'` yields `None` to a `yield from` caller — profiling
changes program semantics. Fix: `except StopIteration as e: return e.value`
(PEP 380). (`wrap_async_generator` is fine; async generators cannot return
values.)

### F4 (MEDIUM): `--prof-imports` crashes scripts containing star imports

`ast_profile_transformer.py:146-152` builds `ast.Name(id='*')` for
`from x import *`, generating `profile.add_imported_function_or_module(*)`
→ `NameError` at runtime. **Verified repro.**

### F5 (MEDIUM): multi-name imports drop all but the last matched target

`profmod_extractor.py:222-243` keys results by AST statement index
(`modnames_found_in_tree[tree_index] = name`), so
`import os, json` with both in `prof_mod` profiles only `json`.
**Verified repro.**

### F6 (MEDIUM): mixed-unit `LineStats` merges quantize to the coarsest unit

`line_profiler.py:486-505` (`_get_aggregated_timings`): ascending sort then
`unit = stats_objs[-1].unit` takes the *largest* unit; entries recorded in
finer units are divided down and `int(round(...))`ed — a 400 ns entry
(unit 1e-9) merged with a 1e-6-unit stats object becomes 0. The comment
says the sort exists "to minimize rounding errors"; it does the opposite.
Same-host merges (PR #431's gather path) are unaffected; cross-platform
`python -m line_profiler` file merges are. **Verified repro.**

### F7 (MEDIUM): `_line_profiler.pyi` stub has drifted from the extension

The stub declares `dump_stats()` on the C `LineProfiler` (does not exist —
`hasattr` verified False; it lives on the Python subclass) while omitting
real public surface (`enable`/`disable`, `enable_count`, `functions`,
`timer_unit`, `wrap_trace`, actual `__init__` signature). Note: the
full-repo plan's TYPE program (see plan) removes this stub entirely in
favor of typed Python wrappers, which subsumes the narrow fix.

### F8 (LOW severity, HIGH subtlety): reference leaks and C-API misuse in `c_trace_callbacks.c`

`c_trace_callbacks.c:212-231` (`call_callback`) and `:263-269`
(`set_local_trace`): results of `PyObject_CallOneArg`/
`PyObject_CallMethodOneArg` (new references) are stored via
`PyObject_SetAttrString` (which takes its own reference) and never
released — the in-code comment claiming nothing else holds the reference
is wrong once SetAttr succeeds. One wrapper object leaks per `'call'`
event when a foreign frame-local trace (debugger, coverage) is active.
Additionally, in `set_local_trace` a NULL call result is passed to
`PyObject_SetAttrString`, turning it into an attribute *delete* executed
with a live exception set (C-API misuse). `mod`/`dle` at `:185` are dead.

### F9 (LOW): `_StrEnumBase` unhashable on Python 3.10

`line_profiler_utils.py:55-56` defines `__eq__` without `__hash__`
(implicit `__hash__ = None`); the fallback class is used exactly where
`enum.StrEnum` is missing (3.10), so `ScopingPolicy` members are
unhashable there and hashable on 3.11+. Add `__hash__ = str.__hash__`.

### F10 (LOW): shutdown noise from monitoring callbacks during finalization

`Exception ignored in ... handle_raise_event ... 'NoneType' object has no
attribute 'monitoring'` (`_line_profiler.pyx:721/733`) whenever a process
dies with an enabled profiler while the interpreter finalizes (module
globals already cleared). Observed constantly while reproducing PR-431 P3.
Guard the callbacks against finalization (`sys is None` check or
unregister at shutdown).

### F11 (LOW): cosmetic / dead / fragile code

1. `explicit_profiler.py:18-19` documents the wrong env var
   (`LINE_PROFILER=1`) and flag (`--profile`); actual: `LINE_PROFILE`,
   `--line-profile`.
2. Dead guard logic in `LineProfiler.add_callable`
   (line_profiler.py:584-591): `guard` is defaulted two lines above, so
   the `is None` ternary branch is unreachable and `_get_wrapper_info`
   runs twice per impl.
3. `scoping_policy.py:308-317` and `:334-343`: `if basename` is always
   true for valid module names; the intended test was `if parent`. The
   written fallbacks are unreachable; current behavior is correct only by
   coincidence.
4. `util_static.py:475-478`: identical `if`/`else` branches (vendored
   drift).
5. `_line_profiler.pyx:495-507`: `try/else/finally` without `except` is a
   `SyntaxError` in CPython and compiles only via a Cython grammar quirk;
   `events_before` (`:470`) would be unbound in `finally` if
   `mon.get_events()` raised.
6. `line_profiler.py:179-187` (`get_code_block`): monkeypatches
   `inspect.getblock.__globals__['BlockFinder']` and restores in
   `finally`; two concurrent `show_text()` calls can interleave
   save/restore and leave `_CythonBlockFinder` permanently installed in
   the `inspect` module. Needs a lock (or a tokeneater that doesn't
   mutate `inspect`).

---

## Part B — Packaging / CI / xcookie

1. **(must fix in xcookie, not this repo)** The three `timeout-minutes`
   additions in `.github/workflows/tests.yml` (lines 94/158/431 on the PR
   branch) live in a file xcookie regenerates with `overwrite: 1`
   (`xcookie/main.py:922-929`), and `timeout` appears nowhere in
   `xcookie/builders/github_actions.py` — the next regen silently reverts
   them. Upstream: `timeout-minutes: 10` on "Test full loose sdist"
   (`github_actions.py:1083-1107`), `timeout-minutes: 60` on
   `Actions.cibuildwheel` (`:359-401`), `timeout-minutes: 10` on the
   "Test wheel" step (`:1823` via `common_ci.py:345+`).
2. Mirror the PR's new entry-point group and `_line_profiler_hooks`
   py_module into `[tool.xcookie.entry_points]`/xcookie config so a future
   setup.py regen doesn't drop them (xcookie's setup.py builder emits
   neither `py_modules` nor extra entry-point groups today).
3. `_line_profiler_hooks.py` ships correctly in both sdists
   (`MANIFEST.in:4`) and wheels (`py_modules`, all build paths) — verified
   via egg-info SOURCES/top_level. No action.
4. sdists cannot run the full test suite: `tests/cython_example/*.pyx/.toml`
   are excluded (`MANIFEST.in` only pulls `tests/**.py`), so
   `test_cython.py` fails from an sdist. Add
   `recursive-include tests *.pyx *.pxd *.toml`.
5. `tests/cython_example/MANIFEST.in` is untracked *and* dead (the test's
   copy filter never propagates it). Delete or commit-and-wire it.
6. Dirty-tree sdists leak private files: `include *.md` / `*.py` / `*.txt`
   at the root pulled `AGENTS.md`, `prompt.txt`, `review.md`, `test.py`
   into this tree's egg-info. Enumerate root includes explicitly, add
   `global-exclude *.py[cod]` + `prune **/__pycache__`, drop the stale
   `include run_tests.sh`, and drop the bogus `'line_profiler'` entry from
   `py_modules`.
7. `package_data` can't cover `line_profiler/rc/*.toml` (subpackage needs
   its own key); works today only via MANIFEST + `include_package_data`.
   Add `'line_profiler.rc': ['*.toml']`.
8. Lint job installs `ty` unpinned (generated); the PR branch's
   `[tool.ty.terminal] error-on-warning=false` is a workaround for exactly
   that drift. Pin/bound `ty` in `xcookie/builders/common_ci.py:100`.
