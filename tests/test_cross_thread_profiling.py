"""
Regression tests: disabling a profiler on one thread must not affect
profilers still active on other threads.

Under the ``sys.monitoring`` core, registration is process-global while
the manager bookkeeping is per-thread; the first thread's manager used
to tear down the global events/tool ID as soon as *its* profilers were
disabled, silently ending data collection for every other thread and
making their later ``disable()`` calls raise
``ValueError: tool ... is not in use`` from inside user ``finally``
blocks.

The core is chosen at import time, so the scenarios run in subprocesses
(once per core).
"""
import subprocess
import sys
import textwrap

import pytest


_SCENARIO = textwrap.dedent(
    """
    import threading

    from line_profiler import LineProfiler

    N = 2000


    def work_a():
        return len([None])


    def work_b():
        total = 0
        for i in range(N):
            total += i
        return total


    prof_a = LineProfiler(work_a)
    prof_b = LineProfiler(work_b)

    b_ready = threading.Event()
    a_disabled = threading.Event()
    errors = []


    def thread_b():
        try:
            prof_b.enable_by_count()
            try:
                work_b()
                b_ready.set()
                assert a_disabled.wait(timeout=30)
                # Profiling of this thread must survive thread A's
                # disable...
                work_b()
            finally:
                # ... and this must not raise
                prof_b.disable_by_count()
        except BaseException as e:
            errors.append(e)
            b_ready.set()


    # Thread A (the main thread) is the first to enable a profiler and
    # thus the one holding the global registration (when the core is
    # process-global); it bows out while thread B is still profiling.
    prof_a.enable_by_count()
    worker = threading.Thread(target=thread_b)
    worker.start()
    assert b_ready.wait(timeout=30)
    work_a()
    prof_a.disable_by_count()
    a_disabled.set()
    worker.join(timeout=30)
    assert not worker.is_alive()

    assert not errors, f'thread B errored: {errors!r}'
    import inspect

    source, start = inspect.getsourcelines(work_b)
    incr_lineno = start + next(
        offset for offset, line in enumerate(source) if 'total += i' in line
    )
    (entries,) = [
        e for key, e in prof_b.get_stats().timings.items()
        if key[2] == 'work_b'
    ]
    total_hits = sum(
        nhits for lineno, nhits, _ in entries if lineno == incr_lineno
    )
    assert total_hits == 2 * N, (
        f'expected {2 * N} hits on the loop body (2 executions), '
        f'got {total_hits}: thread B lost profiling data'
    )
    print('OK')
    """
)


@pytest.mark.parametrize('core', ['default', 'legacy'])
def test_disable_on_one_thread_keeps_other_threads_profiling(
    tmp_path, monkeypatch, core,
):
    script = tmp_path / 'cross_thread_scenario.py'
    script.write_text(_SCENARIO)
    if core == 'legacy':
        monkeypatch.setenv('LINE_PROFILER_CORE', 'legacy')
    else:
        monkeypatch.delenv('LINE_PROFILER_CORE', raising=False)
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert proc.stdout.strip() == 'OK'
