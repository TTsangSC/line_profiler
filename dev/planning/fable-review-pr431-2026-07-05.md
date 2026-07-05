# Fable review — PR #431 (`--prof-child-procs`)

- **Reviewer:** Claude (Fable 5). Review performed 2026-07-04 on branch
  `profile-child-processes` (head `c818cc3`, merge-base with `main` =
  `a5ec7fd`); reorganized into `dev/planning/` 2026-07-05.
- **Scope of this file:** findings *specific to PR #431* (the feature code
  and its test suite). Repo-wide findings live in
  `fable-review-fullrepo-2026-07-05.md`. Remediation task specs and live
  status live in `fable-pr431-plan-2026-07-05.md` — update THAT file, not
  this one; this review is frozen evidence.
- **Method:** full read of the PR diff (~8.5k insertions, 34 files); the
  test suite executed locally in a fresh venv (156 tests, all pass with a
  correctly installed `kernprof`); targeted end-to-end experiments against
  the built package for fork/spawn/forkserver, `Pool`, `Process`,
  `concurrent.futures.ProcessPoolExecutor`, `subprocess`, `terminate()`,
  and a cross-interpreter pool; delegated deep-dives whose load-bearing
  claims were independently re-verified.
- **Verdict:** the architecture is sound and a large improvement over
  earlier iterations, but the PR **must not merge yet**: merged profile
  numbers are quantitatively wrong for fork-based children (P1), a
  silently-failed child hook deadlocks patched pools (P2), and the feature
  hard-crashes `kernprof` on any unwritable site-packages (P3). All three
  were reproduced locally, with recipes below.

Repro environment: Python 3.12, Linux, editable install in a venv
(`pip install -e . --no-build-isolation` after `pip install cython`),
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

## Part B — Blocking product defects

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
([cache.py:528-571](../../line_profiler/_child_process_profiling/cache.py#L528))
reuses the parent's profiler object in the fork
(`forked._setup_in_child_process(False, 'fork', self.profiler)`, cache.py:568).
That object's timings already contain everything the parent accumulated
before the fork. The child later dumps its *full* stats to its own file; the
parent also dumps its full stats; `gather_stats()` merges by summation.
Every `os.fork()`-descended child (direct `fork` Process, fork Pool workers,
and `ProcessPoolExecutor` with fork context) re-contributes the parent's
pre-fork counts.

**Remediation:** task PR431-1 in the plan (delta-baseline dumping).

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
([_mandatory_patches.py:240](../../line_profiler/_child_process_profiling/multiprocessing_patches/_mandatory_patches.py#L240)).
If the worker is *not* patched it puts the vanilla 3-tuple `(job, i, obj)`;
the 2-way unpack raises `ValueError` inside the pool's `_handle_results`
thread, the thread dies, and every `AsyncResult.get()` blocks forever.
Meanwhile `load_pth_hook` deliberately swallows all exceptions
(`_line_profiler_hooks.py:63-70`, silent unless `DEBUG`), so *every* silent
hook failure — different interpreter via `set_executable`, unreadable cache
file, exception during child preimports, cleared env — converts "child just
isn't profiled" into "the user's program deadlocks."

**Remediation:** task PR431-2 in the plan (self-describing tagged protocol,
tolerant parent, loud hook failure).

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
([cache.py:361](../../line_profiler/_child_process_profiling/cache.py#L361)) and
nothing between it and `kernprof.main()` catches `OSError`.

**Related hazard (same function):** when line_profiler runs from an
*uninstalled checkout*, the parent imports `_line_profiler_hooks` from the
repo root, but the written `.pth` does a bare `import _line_profiler_hooks`
in every interpreter sharing that site-packages — which fails and makes
`site.py` print an error in every unrelated Python process for the
session's lifetime.

**Remediation:** task PR431-3 in the plan (fallback dirs, warn-and-degrade).

### P4 (MEDIUM): `assert not worker.is_alive()` in a `finally` during pool termination

[_mandatory_patches.py:179](../../line_profiler/_child_process_profiling/multiprocessing_patches/_mandatory_patches.py#L179),
inside `wrap_terminate_pool`'s `finally`. CPython's `_terminate_pool` has its
own raise path *before* workers are joined
(`AssertionError("Cannot have cache with result_handler not alive")` —
which fires in exactly the P2 scenario), plus `KeyboardInterrupt` during the
joins. In those cases the assert trips on a live worker and *shadows the
original exception*; under `python -O` it silently vanishes instead. Same
class of problem: `assert callable(stop)` in
`RebootForkserverPatch.reboot` (_mandatory_patches.py:301).
**Remediation:** task PR431-4.

### P5 (MEDIUM): one failure in child-stats gathering destroys the *entire* profiling output

`_manage_profiler.__exit__` ([kernprof.py:1228-1240](../../kernprof.py#L1228)):
`_post_profile` (which writes the parent's own stats) runs only after
`cache.cleanup()` and `cache.gather_stats()` succeed. On `main`, dumping was
in a `finally`. Any unexpected exception in cleanup/gather (e.g.
`FileNotFoundError` from `gather_stats`'s `is_empty` stat racing a deleted
file) now discards even the parent's data. **Remediation:** task PR431-5.

### P6 (MEDIUM): `__enter__` failure leaks installed global state

`_manage_profiler.__enter__` ([kernprof.py:1183-1219](../../kernprof.py#L1183))
installs `CuratedProfilerContext` (global `@profile` overwrite, builtins
injection, threading patch) before `_prepare_exec_script` /
`_prepare_child_profiling_cache`, both of which can raise (P3 was one such
path). If `__enter__` raises, `__exit__` never runs and the process-global
patches stay installed. Irrelevant when the CLI process exits, but
`kernprof.main()` is importable, called in-process by this repo's own tests
and by downstream tooling. **Remediation:** task PR431-6.

---

## Part C — Non-blocking issues

### P7 (MEDIUM): the support contract is not documented anywhere user-visible

`--prof-child-procs` help says only "Extend profiling into child Python
processes … (EXPERIMENTAL)". Nothing in `docs/` or the kernprof epilog
states the actual (correct, well-implemented!) boundary: cooperative Python
children that reach normal startup and exit/flush points are supported;
`SIGKILL`, `os._exit`, hard crashes, `python -S`, cleared environments,
non-Python and frozen/embedded children are not — and cannot be, since
`SIGKILL` is uncatchable and skipped startup never runs `.pth` hooks. The
`multiprocessing` pool caveat (unclosed pools → idle-worker warnings) is
documented only in an internal module docstring. Today the flag
under-documents rather than overclaims. **Remediation:** task PR431-7.

### P8 (MEDIUM): mandatory patches load exclusively via entry-point metadata

`Registry.from_entry_point()` (_infrastructure.py:481) sources even the five
`__`-prefixed *mandatory* patches from
`importlib.metadata.entry_points(group='line_profiler._multiproc_patches')`
declared in `setup.py:306-341`. Consequences: a stale editable install (or
any sys.path-based use without dist metadata) makes
`kernprof --prof-child-procs` die with a bare `AssertionError`
(multiprocessing_patches/__init__.py:119-120) — or, under `python -O`,
silently apply *no* patches; every profiled child pays an all-distributions
metadata scan at startup; and any third-party package can inject patches
into every kernprof session with only a warning on load failure.
**Remediation:** task PR431-8.

### P9 (MEDIUM): stale docstring describes a design that was removed

`multiprocessing_patches/__init__.py:57-71` promises Windows-specific
behavior — patching `Pool._get_tasks()`/`_guarded_task_generation()` and
`BaseProcess.terminate()` "on Windows" — none of which exists anywhere in
the package (no `sys.platform`/`os.name` branch at all; verified by grep).
Leftover from the abandoned delay-termination design. **Remediation:** task
PR431-9.

### P10 (LOW): warning-message defects at the primary user touchpoint

Observed real output:
`"1 file(s) out of 1 is/are empty and thus skipped:: ['/tmp/.../child-prof-output-86226-86230-....lprof']"`.
Three defects: the double colon (`description` already ends in `:` while the
format string adds another — `LineStats.from_files` warning block in
`line_profiler/line_profiler.py`); the second tuple's description reads
"…failed to load and is/are skipped" (inconsistent grammar, no colon); and
the message gives no hint *why* a file may be empty (terminated/killed
child, unclosed pool) or that it is often benign. The warning also appears
twice on stderr (once via `warnings`, once via the diagnostics logger).
**Remediation:** task PR431-10.

### P11 (LOW): per-task flush cost is unbounded and undocumented

Pool workers serialize and rewrite their *entire accumulated stats file* on
every result `put()` (`dump_stats_quick` per task): O(n_tasks x stats_size)
of pickling per worker. Acceptable for an experimental feature, but should
be documented, and possibly throttled. The current behavior is *correct*
(flush-before-result is what makes terminate-safe pools work).
**Remediation:** task PR431-11.

### P12 (LOW): stale `.pth` files accumulate forever after hard kills

A SIGKILLed parent leaks its `.pth` (guard makes it a no-op, but it still
executes an import in every future interpreter). **Remediation:** task
PR431-12 (GC dead-PID hook files at next session start).

### P13 (LOW): assorted small defects

1. `Cleanup.add_cleanup_with_priority` docstring: "a HIGHER value are
   invoked BEFORE those with **bigger** values" → "lower"
   ([cleanup.py:178-180](../../line_profiler/cleanup.py#L178)).
2. Cleanup-callback failures are logged only at `debug` level
   (cleanup.py:146-156). Failures restoring `os.fork`/`multiprocessing`
   attributes or deleting the `.pth` deserve `log.warning`.
3. `make_tempfile` (line_profiler_utils.py:341-350) lost the old
   `_touch_tempfile` behavior of unlinking the file when `os.close` fails.
4. `LineStats.get_empty_instance()` instantiates a full `LineProfiler`
   (C state, tool registration) just to read `timer_unit`.
5. Typos: "implmentation", "extend profiling to therein", "befor"
   (`_line_profiler_hooks.py` docstring), "desinated" (kernprof.py:792),
   "profliing" (multiple), "automaticaly" (cache.py:405), "ppol"
   (_mandatory_patches.py:175), invalid `:path:`/`:cmd:`/`:py:method:` roles.
6. Dead `AnyStr` TypeVar in cache.py:42.
7. `_line_profiler_hooks.load_pth_hook`: the double-load guard is checked
   *after* the heavy imports; move it before.
8. Nested kernprof sessions (kernprof profiling a script that itself runs
   kernprof `--prof-child-procs`) have under-specified attribution: the
   inner process was already set up as a *child* of the outer session, so
   `multiprocessing` is already marked patched and the inner session's
   `apply()` returns early; the inner session's own children then report to
   whichever cache the env vars point at. Not a crash, but undefined
   behavior.

---

## Part D — Test-suite findings (`tests/test_child_procs/`)

The suite gives trustworthy, deterministic signal on Linux (Part A #6), but
has real gaps and hazards. Remediation: tasks PR431-T1 … T11 in the plan.

1. Tests invoke the CLI as bare `'kernprof'` resolved from `PATH`
   (`_test_child_procs_utils.py:2038`), so the suite can silently test a
   *different* interpreter's installation than the one running pytest (this
   produced 5 confusing local failures during this review).
2. Dead product config in tests: `test_child_procs.py:505-511` writes a
   `[tool.line_profiler.child_processes.multiprocessing.polling]` table that
   the product no longer reads (polling was removed in `dcc8905`); the
   "easier to debug" safety net it claims to enable does not exist.
3. `cleanup_extra_pth_files` (`_test_child_procs_utils.py:962-985`) deletes
   **every** `.pth` file that appeared in site-packages during the test —
   including e.g. a concurrently created `__editable__*.pth`.
4. No writability precheck: on unwritable purelib the suite fails with raw
   `PermissionError` x94 instead of skipping with a reason.
5. 43 of 44 multiproc E2E variants run `kernprof.main()` **in-process in a
   daemon thread** rather than as a real subprocess; only one fork-context
   variant exercises the real CLI. The threaded harness must suppress
   CPython's "fork() in multi-threaded process" DeprecationWarning —
   masking a class of real regressions — and abandons the thread on timeout
   while restoring globals under it.
6. Coverage gaps: `concurrent.futures.ProcessPoolExecutor` (zero coverage;
   manually verified working under spawn during this review); bare
   `os.fork()` without multiprocessing; `Process.terminate()` mid-task;
   SIGKILLed child; corrupt non-empty `.lprof` (`is_valid_stats_file`
   currently *filters* defective files instead of asserting none exist);
   post-run `os.environ` cleanliness at the CLI level; nested sessions.
7. Patch-reversal verification is self-referential: tests assert
   restoration against the product's own `Patch.summary` metadata
   (`_test_child_procs_utils.py:1594-1757`), so an undeclared mutation is
   invisible to both sides.
8. `conftest.py:382`: `(curr_pid - 42) % (2 * 16)` — `2 * 16` is almost
   certainly a typo for `2 ** 16`, can return 0, and the adjacent
   `assert pid != curr_pid` can never fire.
9. Teardowns swallow exceptions silently (`conftest.py:353-361`,
   `preserve_object_attrs.__exit__`).
10. The 2,189-line `_test_child_procs_utils.py` contains dead members
    (`ResultMismatch.rich_message`, `CheckWarnings.propagate_warnings`,
    unused `Sequence` protocol fillers) and re-implements pytest machinery
    (`Params` algebra ≈ stacked `parametrize`, `CheckWarnings` ≈
    `pytest.warns`).
11. `process_test_module.py:74-94` carries an `# xdoctest: +SKIP` for
    "currently unknown" pickling errors — the only Linux skip in the suite.
    Unexplained pickling failures inside the harness that tests pickling
    paths deserve a root-cause and a tracking issue.
