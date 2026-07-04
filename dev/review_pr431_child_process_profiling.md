# Review & remediation plan: PR #431 (`--prof-child-procs`) and adjacent code

- **Reviewer:** Claude (Fable 5), 2026-07-04, on branch `profile-child-processes`
  (head `c818cc3`, merge-base with `main` = `a5ec7fd`).
- **Method:** full read of the PR diff (~8.5k insertions, 34 files); the test
  suite executed locally in a fresh venv (156 tests, all pass with a correctly
  installed `kernprof`); targeted end-to-end experiments against the built
  package for fork/spawn/forkserver, `Pool`, `Process`,
  `concurrent.futures.ProcessPoolExecutor`, `subprocess`, `terminate()`, and a
  cross-interpreter pool; three delegated deep-dives (core modules,
  packaging/CI/xcookie, test suite) whose load-bearing claims were
  independently re-verified before inclusion here.
- **Verdict up front:** the architecture of PR #431 is sound and a large
  improvement over earlier iterations, but the PR **must not merge yet**:
  merged profile numbers are quantitatively wrong for fork-based children
  (P1), a silently-failed child hook deadlocks patched pools (P2), and the
  feature hard-crashes `kernprof` on any unwritable site-packages (P3). All
  three were reproduced locally, with recipes below. Everything else is
  fixable in follow-ups, but P1–P3 are disqualifying for the primary output
  of a profiler: trustworthy numbers and no new hangs.

Repro environment used throughout: Python 3.12, Linux, editable install in a
venv (`pip install -e . --no-build-isolation` after `pip install cython`),
`kernprof` resolved from the venv `bin/`.

---

## Part A — What the PR gets right (verified, keep as-is)

These design decisions were questioned during PR review and are, on
inspection and measurement, correct. Do not regress them while fixing the
issues below.

1. **Activation/finalization split.** Activation (env vars + `.pth` +
   serialized `LineProfilingCache`) is cleanly separated from finalization
   (`gather_stats` with warn-on-empty/defective). This is the right shape.
2. **Cooperative flush instead of termination handling.** Pool workers flush
   stats before each result `put()`
   (`_profiling_patches.py::wrap_worker`), direct `Process` children flush at
   `_bootstrap` exit (`wrap_bootstrap`), pool workers are marked so the two
   paths don't double-fire. Verified: spawn and forkserver produce *exact*
   hit counts (100,001/100,000 in a 100k-loop workload; 27/27 task
   executions across all three start methods).
3. **The `.pth` hook is cheap and guarded.** `_line_profiler_hooks.py`
   imports only `os` on the fast path and early-returns unless
   `LINE_PROFILER_PROFILE_CHILD_PROCESSES_CACHE_PID` matches the PID baked
   into the `.pth` line. Stale hook + PID reuse combinations were analyzed:
   worst case is a harmless duplicate no-op setup, not misprofiling.
4. **Cleanup discipline.** After a clean run: no `.pth` left in
   site-packages, no env vars left in the parent, cache dir removed
   (verified by inspection after end-to-end runs). Concurrent sessions
   coexist correctly by construction (per-PID env var + per-file PID guard).
5. **`terminate()` semantics match the intended contract.** A terminated
   child's empty stats file produces a `UserWarning` at gather time rather
   than silence or a crash (verified; message-quality issues are P10).
6. **Determinism work paid off.** No retry/flaky markers remain anywhere in
   the suite; 3/3 consecutive full runs pass in ~15 s; exact-count
   assertions would catch stats-loss races loudly. The suite's history shows
   races were fixed in the product, not papered over.
7. **`multiprocessing.dummy`/`ThreadPool` is safe**: the per-put dump is a
   no-op in the parent process (`cache._stats_dumper is None` there), and the
   PID tag round-trips correctly.
8. **Reviewer-raised issues from the PR thread are addressed**: the tiny
   deferred-import hook module exists; empty/defective child files default to
   `warn`; expected-empty helper PIDs (resource tracker, idle pool workers)
   are suppressed via a PID registry rather than blanket-ignoring empties.

---

## Part B — Blocking product defects in the PR

### P1 (HIGH, blocks merge): fork-based children double-count all pre-fork stats

**Repro (reduced):**

```python
# forkdup.py
import multiprocessing as mp
def child_work(): pass
def main():
    acc = 0
    for i in range(100000):
        acc += i
    ctx = mp.get_context('fork')
    p = ctx.Process(target=child_work)
    p.start(); p.join()
if __name__ == '__main__': main()
```

`kernprof -l --prof-mod=forkdup.py --prof-child-procs forkdup.py` reports
**200,002 hits** (and ~2x time) on the loop line; with three fork children it
reports **400,004** — one extra full copy per forked child. The same script
under `spawn` reports the exact 100,001. Since `fork` is the default start
method on Linux ≤3.13, the feature's merged numbers are wrong in the most
common configuration, for hits *and* times.

**Cause:** `cache._wrap_os_fork()`
([cache.py:528-571](../line_profiler/_child_process_profiling/cache.py#L528))
reuses the parent's profiler object in the fork
(`forked._setup_in_child_process(False, 'fork', self.profiler)`, cache.py:568).
That object's timings already contain everything the parent accumulated
before the fork. The child later dumps its *full* stats to its own file; the
parent also dumps its full stats; `gather_stats()` merges by summation.
Every `os.fork()`-descended child (direct `fork` Process, fork Pool workers,
and `ProcessPoolExecutor` with fork context) re-contributes the parent's
pre-fork counts.

**Fix (delta-baseline dumping — do NOT switch to a fresh profiler; the
inherited instance is what holds the already-wrapped functions):**

1. Add `LineStats.__sub__`/`__isub__` in
   `line_profiler/line_profiler.py`, the exact inverse of the existing
   `__add__`: per-key, per-lineno subtraction of `(nhits, time)`, dropping
   entries that reach `(0, 0)` and asserting no negative results (a negative
   result means the baseline is newer than the stats — raise `ValueError`).
   Symmetric doctests to the `__add__` ones.
2. In the fork wrapper's child branch (cache.py:546-569), snapshot
   `baseline = self.profiler.get_stats()` *before* any child work, and pass
   it to `_setup_in_child_process`, which forwards it to `_DumpStatsHelper`.
3. `_DumpStatsHelper` (cache.py:65-84) dumps
   `prof.get_stats() - baseline` when a baseline is present (it must
   re-derive the delta on *every* dump, since pool workers dump repeatedly).
4. Tests: (a) the reduced repro above as an integration test asserting the
   exact 100,001 count for `fork`, parametrized over
   fork/spawn/forkserver × Process/Pool/ProcessPoolExecutor; (b) a unit test
   for `LineStats.__sub__` including the negative-delta error; (c) a
   multi-fork test asserting counts don't scale with child count.

**Watch out for:** nested forks (grandchild baseline must be the profiler
state at *its* fork, which falls out naturally if step 2 runs in every fork);
the forkserver server process (its baseline is ~empty; harmless); and the
snapshot cost at fork time (acceptable; document it). This fix is subtle —
see "hand-off risk" at the end.

### P2 (HIGH, blocks merge): patched parent + unpatched child = pool deadlock

**Repro:** a pool whose workers run in an interpreter that doesn't execute
the session's `.pth` hook:

```python
# crosspy.py
import multiprocessing as mp
def task(x): return x * 2
def main():
    ctx = mp.get_context('spawn')
    ctx.set_executable('/usr/bin/python3.12')   # any interpreter without the session's .pth
    with ctx.Pool(1) as pool:
        res = pool.map(task, range(4))
    print(res)
if __name__ == '__main__': main()
```

Without `--prof-child-procs` this completes (`[0, 2, 4, 6]`). With the flag
it **hangs forever** (killed by timeout in testing).

**Cause:** the mandatory pool patch changes the parent↔worker result
protocol: patched workers `put((pid, result))`
(`_mandatory_patches.py::wrap_worker` + `PutWrapper(push_to_parent=True)`),
and the patched parent unconditionally unpacks
`pid, orig_result = result`
([_mandatory_patches.py:240](../line_profiler/_child_process_profiling/multiprocessing_patches/_mandatory_patches.py#L240)).
If the worker is *not* patched it puts the vanilla 3-tuple `(job, i, obj)`;
the 2-way unpack raises `ValueError` inside the pool's `_handle_results`
thread, the thread dies, and every `AsyncResult.get()` blocks forever.
Meanwhile `load_pth_hook` deliberately swallows all exceptions
(`_line_profiler_hooks.py:63-70`, silent unless `DEBUG`), so *every* silent
hook failure — different interpreter via `set_executable`, unreadable cache
file, exception during child preimports, cleared env — converts "child just
isn't profiled" into "the user's program deadlocks."

**Fix (make the protocol self-describing and the parent tolerant):**

1. In `_queue.PutWrapper` with `push_to_parent=True`, wrap as
   `(_LP_PID_TAG, pid, obj)` where `_LP_PID_TAG` is a module-level unique
   string constant (e.g. `'__line_profiler_pool_pid__'`).
2. In `_wrap_outqueue_quick_get`: if the result is a tuple whose first
   element `== _LP_PID_TAG`, strip and record the PID; otherwise pass the
   result through unchanged and emit a *once-per-process* warning
   ("worker PID unknown — child process appears unpatched; its profile data
   will be missing") via `warnings.warn` + `diagnostics.log.warning`.
3. Keep the sentinel `None` path as-is (already handled).
4. Test: monkeypatch a pool so its worker side is vanilla while the parent
   is patched (simplest: apply only the parent-side patches via
   `Registry.select`, or spawn the worker with the hook env var stripped)
   and assert `pool.map` completes, returns correct results, and the warning
   fires. Add a timeout guard so a regression fails fast instead of hanging
   CI.
5. Separately, make silent hook failure *observable*: `load_pth_hook`'s
   `except Exception` branch should always `warnings.warn` (not only under
   `DEBUG`) — a child that has the env vars set but cannot set up profiling
   is an abnormal condition the user asked to know about; keep the broad
   catch so the child still runs.

### P3 (HIGH, blocks merge): unwritable site-packages hard-crashes kernprof

**Repro:** any environment where `sysconfig.get_path('purelib')` is not
writable — stock Debian/Ubuntu system Python with a `pip install --user`
line_profiler, corporate read-only installs, Nix, etc.:

```
kernprof -l --prof-child-procs ... 
→ PermissionError: [Errno 13] .../dist-packages/_line_profiler-profiling-hook-....pth
```

The whole run dies before the profiled script starts (traceback verified
locally; this is also why 94/156 tests fail outside a venv — CI never sees
it because CI always uses venvs).

**Cause:** `write_pth_hook` defaults `dir` to `sysconfig.get_path('purelib')`
([cache.py:361](../line_profiler/_child_process_profiling/cache.py#L361)) and
nothing between it and `kernprof.main()` catches `OSError`.

**Fix (graceful degradation, in `LineProfilingCache.write_pth_hook` /
`_setup_in_main_process`):**

1. Candidate directories, in order: the directory containing the installed
   `_line_profiler_hooks` module **if** it is a site dir (this also fixes
   the uninstalled-checkout hazard below), `sysconfig.get_path('purelib')`,
   `site.getusersitepackages()` when `site.ENABLE_USER_SITE` is true.
   Attempt the write in each; first success wins.
2. If all fail: `warnings.warn` with an actionable message ("cannot install
   startup hook (…); profiling will not extend into `spawn`/`exec`ed
   children; fork-based children are still covered") and continue the run.
   Do not raise.
3. Related hazard (same function): when line_profiler runs from an
   *uninstalled checkout*, the parent imports `_line_profiler_hooks` from
   the repo root, but the written `.pth` does a bare
   `import _line_profiler_hooks` in every interpreter sharing that
   site-packages — which fails and makes `site.py` print an error in every
   unrelated Python process for the session's lifetime. Guard: before
   writing, verify `_line_profiler_hooks.__file__`'s directory is on the
   default `sys.path` of a bare interpreter (or simply that it equals the
   chosen site dir); otherwise warn and skip the `.pth`.
4. Tests: point `dir=` at a read-only tempdir and assert warn-and-continue;
   assert the run still profiles fork children. Also add the session-scoped
   pytest skip described in Part D so the suite degrades legibly.

### P4 (MEDIUM): `assert not worker.is_alive()` in a `finally` during pool termination

[_mandatory_patches.py:179](../line_profiler/_child_process_profiling/multiprocessing_patches/_mandatory_patches.py#L179),
inside `wrap_terminate_pool`'s `finally`. CPython's `_terminate_pool` has its
own raise path *before* workers are joined
(`AssertionError("Cannot have cache with result_handler not alive")` —
which fires in exactly the P2 scenario), plus `KeyboardInterrupt` during the
joins. In those cases the assert trips on a live worker and *shadows the
original exception*; under `python -O` it silently vanishes instead. An
assert about foreign-code invariants doesn't belong in a cleanup path.
**Fix:** replace with `if worker.is_alive(): cache._debug_output(...); continue`
(skip `_get_worker_ntasks` for live workers). Same treatment for
`assert callable(stop)` in `RebootForkserverPatch.reboot`
(_mandatory_patches.py:301): raise a clear `RuntimeError` or warn-and-skip,
since `-O` turns it into `TypeError: 'NoneType' is not callable`.

### P5 (MEDIUM): one failure in child-stats gathering destroys the *entire* profiling output

`_manage_profiler.__exit__` ([kernprof.py:1228-1240](../kernprof.py#L1228)):
`_post_profile` (which writes the parent's own stats) runs only after
`cache.cleanup()` and `cache.gather_stats()` succeed. On `main`, dumping was
in a `finally`. Any unexpected exception in cleanup/gather (e.g.
`FileNotFoundError` from `gather_stats`'s `is_empty` stat racing a deleted
file, an `OSError` from a half-removed cache dir) now discards even the
parent's data. **Fix:** wrap the cleanup+gather block in `try/except
Exception`, log + warn, and always call `_post_profile(options, prof,
extra_stats_or_None)` in a `finally` alongside `self._ctx.uninstall()`.
Additionally harden `gather_stats`: `is_empty` should treat
`FileNotFoundError` as "exclude the file".

### P6 (MEDIUM): `__enter__` failure leaks installed global state

`_manage_profiler.__enter__` ([kernprof.py:1183-1219](../kernprof.py#L1183))
installs `CuratedProfilerContext` (global `@profile` overwrite, builtins
injection, threading patch) before `_prepare_exec_script` /
`_prepare_child_profiling_cache`, both of which can raise (P3 was one such
path). If `__enter__` raises, `__exit__` never runs and the process-global
patches stay installed. Irrelevant when the CLI process exits, but
`kernprof.main()` is importable, called in-process by this repo's own tests
and by downstream tooling. **Fix:** wrap the body of `__enter__` after
`self._ctx.install()` in `try/except BaseException: self._ctx.uninstall();
raise`.

---

## Part C — Non-blocking issues in the PR

### P7 (MEDIUM): the support contract is not documented anywhere user-visible

`--prof-child-procs` help says only "Extend profiling into child Python
processes … (EXPERIMENTAL)". Nothing in `docs/` or the kernprof epilog
states the actual (correct, well-implemented!) boundary: cooperative Python
children that reach normal startup and exit/flush points are supported;
`SIGKILL`, `os._exit`, hard crashes, `python -S`, cleared environments,
non-Python and frozen/embedded children are not — and cannot be, since
`SIGKILL` is uncatchable and skipped startup never runs `.pth` hooks. The
`multiprocessing` pool caveat (unclosed pools → idle-worker warnings) is
documented only in an internal module docstring.
**Fix:** add a "Profiling child processes" section to the kernprof docs page
(and a paragraph in the `--prof-child-procs` help epilog) enumerating
supported / best-effort / unsupported cases, the flush points, and what the
empty-file warning means. Cheap, high value, and it directly answers the
"does the flag overclaim" concern: today the flag under-documents rather
than overclaims.

### P8 (MEDIUM): mandatory patches load exclusively via entry-point metadata

`Registry.from_entry_point()`
(_infrastructure.py:481) sources even the five `__`-prefixed *mandatory*
patches from `importlib.metadata.entry_points(group='line_profiler._multiproc_patches')`
declared in `setup.py:306-341`. Consequences: a stale editable install (or
any sys.path-based use without dist metadata) makes
`kernprof --prof-child-procs` die with a bare `AssertionError`
(multiprocessing_patches/__init__.py:119-120) — or, under `python -O`,
silently apply *no* patches; every profiled child pays an
all-distributions metadata scan at startup; and any third-party package can
inject patches into every kernprof session (extensibility, but also a
fragility/attack surface with only a warning on load failure).
**Fix:** register the built-in patches in code
(`Registry.get_default()` building directly from the
`_mandatory_patches`/`_optional_patches`/`_profiling_patches` objects), and
keep the entry-point group *only* as an extension mechanism layered on top.
Replace the `assert` loop with an explicit `RuntimeError` naming the missing
patch. This also deletes the setup.py↔code dual source of truth.

### P9 (MEDIUM): stale docstring describes a design that was removed

`multiprocessing_patches/__init__.py:57-71` promises Windows-specific
behavior — patching `Pool._get_tasks()`/`_guarded_task_generation()` and
`BaseProcess.terminate()` "on Windows" — none of which exists anywhere in
the package (no `sys.platform`/`os.name` branch at all; verified by grep).
This is left over from the abandoned delay-termination design and will send
future maintainers hunting for phantom code. **Fix:** rewrite the `Patches:`
section to describe the actual platform-uniform per-put/bootstrap-exit
design. Audit the other docstrings in the subpackage for the same drift
(e.g. `apply()`'s description of `'process'`: "…and are given enough time
for that" — also vestigial).

### P10 (LOW): warning-message defects at the primary user touchpoint

Observed real output:
`"1 file(s) out of 1 is/are empty and thus skipped:: ['/tmp/.../child-prof-output-86226-86230-....lprof']"`.
Three defects: the double colon (`description` already ends in `:` while
the format string adds another —
[line_profiler.py `from_files`](../line_profiler/line_profiler.py), the
`for problems, description, behavior in [...]` block); the second tuple's
description reads "…failed to load and is/are skipped" (inconsistent
grammar, no colon — same block); and the message gives the user no hint
*why* a file may be empty (terminated/killed child, unclosed pool) or that
it is often benign. Fix the format strings and append a one-line cause hint.
Also: the warning appears twice on stderr (once via `warnings`, once via the
diagnostics logger's stderr handler) — consider logging at `debug` level
when the warning is also user-visible.

### P11 (LOW): per-task flush cost is unbounded and undocumented

Pool workers serialize and rewrite their *entire accumulated stats file* on
every result `put()` (`dump_stats_quick` per task). For many small tasks
with a large profiled surface this is O(n_tasks x stats_size) of pickling
per worker. Acceptable for an experimental feature, but (a) document it in
the P7 docs section, and (b) consider a cheap throttle (dump at most once
per N seconds unless the pending task is the worker's last) as a follow-up.
Measure before optimizing; the current behavior is *correct* (flush-before-
result is what makes terminate-safe pools work).

### P12 (LOW): stale `.pth` files accumulate forever after hard kills

A SIGKILLed parent leaks its `.pth` (guard makes it a no-op, but it still
executes an import in every future interpreter). **Fix:** in
`write_pth_hook`, before writing, glob the target dir for
`{prefix}*{suffix}.pth`, parse the PID out of each file's
`load_pth_hook(<pid>)` line, and unlink files whose PID is not alive
(`os.kill(pid, 0)` → `ProcessLookupError`). Best-effort, wrapped in
`try/except OSError`.

### P13 (LOW): assorted small defects (one commit each, low risk)

1. `Cleanup.add_cleanup_with_priority` docstring: "a HIGHER value are
   invoked BEFORE those with **bigger** values" → "lower"
   ([cleanup.py:178-180](../line_profiler/cleanup.py#L178)).
2. Cleanup-callback failures are logged only at `debug` level
   (cleanup.py:146-156). Failures restoring `os.fork`/`multiprocessing`
   attributes or deleting the `.pth` deserve `log.warning`.
3. `make_tempfile` (line_profiler_utils.py:341-350) lost the old
   `_touch_tempfile` behavior of unlinking the file when `os.close` fails;
   restore it.
4. `LineStats.get_empty_instance()` instantiates a full `LineProfiler`
   (C state, tool registration) just to read `timer_unit`; read the unit
   constant from `_line_profiler` directly.
5. Typos: "implmentation", "extend profiling to therein", "befor"
   (`_line_profiler_hooks.py` docstring), "desinated" (kernprof.py:792),
   "profliing" (multiple), "automaticaly" (cache.py:405), "ppol"
   (_mandatory_patches.py:175), invalid `:path:`/`:cmd:`/`:py:method:` roles.
6. Dead `AnyStr` TypeVar in cache.py:42 (only used in
   `line_profiler_utils`).
7. `_line_profiler_hooks.load_pth_hook`: the double-load guard is checked
   *after* the heavy imports; move `getattr(load_pth_hook, 'called', False)`
   to before the `import warnings` block for the double-load case.
8. Nested kernprof sessions (kernprof profiling a script that itself runs
   kernprof `--prof-child-procs`) have under-specified attribution: the
   inner process was already set up as a *child* of the outer session, so
   `multiprocessing` is already marked patched and the inner session's
   `apply()` returns early; the inner session's own children then report to
   whichever cache the env vars point at. Not a crash, but undefined
   behavior. Minimum fix: detect the condition in
   `_prepare_child_profiling_cache` (env var already present with a
   different PID) and emit a warning that nested sessions have
   undefined attribution; add a test capturing whatever behavior is chosen.

---

## Part D — Test-suite findings (`tests/test_child_procs/`)

The suite gives trustworthy, deterministic signal on Linux (see Part A #6),
but has real gaps and hazards:

1. **(fixed-cost, do first)** Tests invoke the CLI as bare `'kernprof'`
   resolved from `PATH` (`_test_child_procs_utils.py:2038`), so the suite can
   silently test a *different* interpreter's installation than the one
   running pytest (this produced 5 confusing local failures during this
   review). Use `[sys.executable, '-m', 'kernprof']` as the default runner.
2. Dead product config in tests: `test_child_procs.py:505-511` writes a
   `[tool.line_profiler.child_processes.multiprocessing.polling]` table that
   the product no longer reads (polling was removed in `dcc8905`); the
   "easier to debug" safety net it claims to enable does not exist. Delete
   it — or better, make `MPConfig`/`get_subconfig` reject unknown keys so
   config typos fail loudly for users too.
3. `cleanup_extra_pth_files` (`_test_child_procs_utils.py:962-985`) deletes
   **every** `.pth` file that appeared in site-packages during the test —
   including e.g. a concurrently created `__editable__*.pth`. Restrict to
   the configured `_line_profiler-profiling-hook-` prefix.
4. No writability precheck: on unwritable purelib the suite fails with raw
   `PermissionError` x94 instead of skipping with a reason. Add a
   session-scoped fixture that skips the package (until P3 makes the
   product degrade gracefully, after which these tests should instead
   assert the warning).
5. 43 of 44 multiproc E2E variants run `kernprof.main()` **in-process in a
   daemon thread** rather than as a real subprocess; only one fork-context
   variant exercises the real CLI. The threaded harness must suppress
   CPython's "fork() in multi-threaded process" DeprecationWarning —
   masking a class of real regressions — and abandons the thread on
   timeout while restoring globals under it. Promote at least one spawn and
   one forkserver variant to `subproc=True`.
6. Coverage gaps to close: `concurrent.futures.ProcessPoolExecutor` (zero
   coverage today; manually verified working under spawn during this
   review — add the test, it will also catch P1 under fork);
   bare `os.fork()` without multiprocessing; `Process.terminate()` mid-task
   (the harness supports it, no test uses it); SIGKILLed child (assert
   graceful degradation + warning); corrupt non-empty `.lprof`
   (`is_valid_stats_file` currently *filters* defective files instead of
   asserting none exist); post-run `os.environ` cleanliness at the CLI
   level; nested sessions (see P13.8).
7. Patch-reversal verification is self-referential: tests assert
   restoration against the product's own `Patch.summary` metadata
   (`_test_child_procs_utils.py:1594-1757`), so an undeclared mutation is
   invisible to both sides. Add one hand-written list asserting restoration
   of the critical targets (`os.fork`, `BaseProcess._bootstrap`,
   `multiprocessing.pool.worker`, `Pool._handle_results`, env vars).
8. `conftest.py:382`: `(curr_pid - 42) % (2 * 16)` — `2 * 16` is almost
   certainly a typo for `2 ** 16`, can return 0, and the adjacent
   `assert pid != curr_pid` can never fire. Use `curr_pid + 1` (with the
   comment that only inequality matters).
9. Teardowns swallow exceptions silently (`conftest.py:353-361`,
   `preserve_object_attrs.__exit__`). At minimum re-raise or print; silent
   cleanup failure is exactly how order-dependent flakes are born.
10. The 2,189-line `_test_child_procs_utils.py` contains dead members
    (`ResultMismatch.rich_message`, `CheckWarnings.propagate_warnings`,
    unused `Sequence` protocol fillers) and re-implements pytest machinery
    (`Params` algebra ≈ stacked `parametrize`, `CheckWarnings` ≈
    `pytest.warns`). Delete the dead members now; consider shrinking the
    frameworks opportunistically, not as a big-bang rewrite.
11. `process_test_module.py:74-94` carries an `# xdoctest: +SKIP` for
    "currently unknown" pickling errors — the only Linux skip in the suite.
    File a tracking issue; unexplained pickling failures inside the harness
    that tests pickling paths deserve a root-cause.

---

## Part E — Packaging / CI / xcookie

1. **(must fix in xcookie, not here)** The three `timeout-minutes` additions
   in `.github/workflows/tests.yml` (lines 94/158/431) live in a file
   xcookie regenerates with `overwrite: 1` (`xcookie/main.py:922-929`), and
   `timeout` appears nowhere in `xcookie/builders/github_actions.py` — the
   next regen silently reverts them. Upstream: add `timeout-minutes: 10` to
   the "Test full loose sdist" step (`github_actions.py:1083-1107`),
   `timeout-minutes: 60` to `Actions.cibuildwheel` (`:359-401`), and
   `timeout-minutes: 10` to the "Test wheel" step (`:1823` via
   `common_ci.py:345+`). Given this PR's subject is process lifecycles,
   step timeouts are exactly the right defense — keep them, but in the
   generator.
2. Mirror the new entry-point group and `_line_profiler_hooks` py_module
   into `[tool.xcookie.entry_points]`/xcookie config so a future setup.py
   regen doesn't drop them (xcookie's setup.py builder emits neither
   `py_modules` nor extra entry-point groups today — teach it, or the P8
   refactor moots the entry-point half of this).
3. `_line_profiler_hooks.py` ships correctly in both sdists
   (`MANIFEST.in:4`) and wheels (`py_modules`, all build paths) — verified
   via egg-info SOURCES/top_level. No action.
4. sdists cannot run the full test suite: `tests/cython_example/*.pyx/.toml`
   are excluded (`MANIFEST.in` only pulls `tests/**.py`), so
   `test_cython.py` fails from an sdist. Add
   `recursive-include tests *.pyx *.pxd *.toml`.
5. `tests/cython_example/MANIFEST.in` is untracked *and* dead (the test's
   copy filter never propagates it). Delete it or commit-and-wire it.
6. Dirty-tree sdists leak private files: `include *.md` / `*.py` / `*.txt`
   at the root pulled `AGENTS.md`, `prompt.txt`, `review.md`, `test.py` into
   this tree's egg-info. Enumerate root includes explicitly, add
   `global-exclude *.py[cod]` + `prune **/__pycache__`, drop the stale
   `include run_tests.sh`, and drop the bogus `'line_profiler'` entry from
   `py_modules` while touching setup.py.
7. `package_data` can't cover `line_profiler/rc/*.toml` (subpackage needs
   its own key); works today only via MANIFEST + `include_package_data`.
   Add `'line_profiler.rc': ['*.toml']`.
8. Lint job installs `ty` unpinned (generated); this branch's
   `[tool.ty.terminal] error-on-warning=false` is a workaround for exactly
   that drift. Pin/bound `ty` in `xcookie/builders/common_ci.py:100`.

---

## Part F — Broader repo (pre-existing on `main`; fix independently of the PR)

Verified findings from the core-module deep-dive (each re-confirmed by repro
or direct code reading during this review):

1. **(HIGH)** `sys.monitoring` backend: enable/disable state is per-thread
   (`_line_profiler.pyx:1066`, `:1248-1271`) but
   `set_events()`/`free_tool_id()` are process-global (`:436-449`). First
   thread to hit zero `enable_count` kills profiling for all threads and
   makes their later `disable_by_count()` raise
   `ValueError: tool 2 is not in use` *inside user code* (repro'd with two
   threads on 3.12). Fix: process-wide refcount across `_managers` before
   deregistering, tolerant `deregister()`, and clear `_c_last_time` for all
   threads on `disable()`. This is the highest-value core fix since 3.12+
   defaults to sysmon — and it underlies this PR's threading story too.
2. **(HIGH)** `GlobalProfiler.show()` discards all output when
   `os.getppid() == 1` and the owner env var matches
   (explicit_profiler.py:466-469, from PR #415): any orphaned/daemonized/
   container-PID-1-parent run writes *nothing*, silently. Replace the
   heuristic with a real mp-bootstrap check (`multiprocessing.parent_process()`)
   or at least warn when skipping.
3. **(MEDIUM)** `@profile` on a generator loses its return value:
   `profiler_mixin.py:550` catches `StopIteration` without re-raising the
   value, so `yield from` under profiling yields `None` instead of the
   return value (repro'd). Fix: `except StopIteration as e: return e.value`.
4. **(MEDIUM)** `--prof-imports` crashes on star imports
   (`ast_profile_transformer.py:146-152` emits
   `add_imported_function_or_module(*)` → `NameError`); skip `alias.name == '*'`.
5. **(MEDIUM)** Multi-name imports drop all but the last matched target
   (`profmod_extractor.py:222-243` keys results by statement index);
   `import os, json` with both in `prof_mod` profiles only `json`.
6. **(MEDIUM)** `LineStats` merges with mixed units quantize to the
   *coarsest* unit (`line_profiler.py:486-505`: ascending sort then
   `stats_objs[-1].unit`), zeroing sub-unit times (400 ns @1e-9 + 1e-6 stats
   → 0). Convert to the smallest unit instead. (Same-host merges — the PR's
   gather path — are unaffected; cross-platform `python -m line_profiler`
   merges are.)
7. **(MEDIUM)** `_line_profiler.pyi` declares `dump_stats()` on the C class
   (doesn't exist) while omitting much of the real surface; fix the stub.
8. **(LOW)** `c_trace_callbacks.c:212-231/:263-269`: call results stored
   via `PyObject_SetAttrString` are never DECREF'd (leak per `'call'` event
   with a foreign frame-local trace), and a NULL result is passed to
   SetAttr with a live exception (C-API misuse). Needs a careful C fix.
9. **(LOW)** `_StrEnumBase` defines `__eq__` without `__hash__` →
   `ScopingPolicy` members unhashable on 3.10 only. Add
   `__hash__ = str.__hash__`.
10. **(LOW)** Shutdown noise: `Exception ignored in ...
    handle_raise_event ... 'NoneType' object has no attribute 'monitoring'`
    (`_line_profiler.pyx:721/733`) whenever a process dies with an enabled
    profiler during interpreter finalization (module globals already
    cleared) — observed constantly while reproducing P3. Guard the
    monitoring callbacks against `sys is None` / finalization.
11. **(LOW)** Cosmetic/dead code: `explicit_profiler.py:18-19` documents
    the wrong env var (`LINE_PROFILER=1`) and flag (`--profile`);
    dead guard logic in `LineProfiler.add_callable`
    (line_profiler.py:584-591); always-true `if basename` branches in
    `scoping_policy.py:308-343` (intended `if parent`); identical if/else
    in `util_static.py:475-478`; `try/else/finally`-without-`except` in
    `_line_profiler.pyx:495-507` compiles only via a Cython grammar quirk —
    restructure; `inspect` global monkeypatch in
    `line_profiler.py:179-187` is racy under concurrent `show_text` (use a
    lock).

---

## Part G — Execution order

| # | Task | Blocks merge? | Risk |
|---|------|---------------|------|
| 1 | P1 fork delta-baseline fix + `LineStats.__sub__` | yes | **high — see hand-off note** |
| 2 | P2 tagged pool protocol + tolerant parent + loud hook failure | yes | **high — see hand-off note** |
| 3 | P3 `.pth` write fallback + warn-and-degrade | yes | medium |
| 4 | P4 remove asserts from cleanup paths | yes (trivial) | low |
| 5 | P5 `__exit__` always dumps parent stats | yes (small) | low |
| 6 | D1 test runner via `sys.executable -m kernprof`; D4 skip fixture | with #3 | low |
| 7 | P6, P10, P13.1-7, P9 docstring rewrite | no | low |
| 8 | P7 contract documentation | before release | low |
| 9 | D2-D3, D6-D9 test hardening + new coverage | before de-experimentalizing | low-medium |
| 10 | P8 in-code patch registry | no | medium |
| 11 | E1-E2 xcookie upstreaming; E4-E7 MANIFEST/package_data hygiene | no | low |
| 12 | F1 sysmon refcount fix | independent | **high — see hand-off note** |
| 13 | F2-F7, F9-F11 core fixes | independent | low-medium |
| 14 | F8 C refcount fix | independent | **high — see hand-off note** |
| 15 | P11 flush throttling (measure first), P12 stale-.pth GC, P13.8 nested-session warning | no | medium |

**Hand-off risk (tasks that need a strong reviewer or should not be done
from this plan alone):** #1 (fork-time snapshot semantics interact with
monitoring state, nested forks, and repeated pool-worker dumps — the
acceptance tests above are necessary but may not be sufficient; get the
`LineStats.__sub__` invariants right first), #2 (protocol changes across a
process boundary where the failure mode is a hang; iterate under a hard
test timeout), #12/F1 (threading x sys.monitoring x Cython; the repro in
the finding must become a regression test before the fix), and #14/F8
(manual C refcounting). Everything else is mechanical enough to execute
directly from this document.
