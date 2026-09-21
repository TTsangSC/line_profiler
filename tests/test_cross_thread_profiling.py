from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from textwrap import indent
from typing import Literal, cast

import pytest

from line_profiler import LineProfiler, LineStats


N = 2000


def _scenario(
    nprof: Literal[1, 2], outdir: str | os.PathLike[str] | None = None,
) -> None:
    def work_a() -> int:
        return len([None])  # grep: BODY (a)

    def work_b(n: int = N) -> int:
        total = 0
        for i in range(n):
            total += i  # grep: LOOP (b)
        return total

    if nprof == 1:
        prof_a = prof_b = LineProfiler(work_a, work_b)
    elif nprof == 2:
        prof_a = LineProfiler(work_a)
        prof_b = LineProfiler(work_b)
    else:
        assert False, f'{nprof=}'

    b_ready = threading.Event()
    a_disabled = threading.Event()
    errors = []

    def thread_b() -> None:
        try:
            with prof_b:
                work_b()
                b_ready.set()
                assert a_disabled.wait(timeout=30)
                # Profiling of this thread must survive thread A's
                # disable...
                work_b()
                # ... and exiting the ctx disables `prof_b`, which must
                # not raise
        except BaseException as e:
            errors.append(e)
            b_ready.set()

    # Thread A (the main thread) is the first to enable a profiler and
    # thus the one holding the global registration (when the core is
    # process-global); it bows out while thread B is still profiling.
    with prof_a:
        worker = threading.Thread(target=thread_b)
        worker.start()
        assert b_ready.wait(timeout=30)
        work_a()
    a_disabled.set()
    worker.join(timeout=30)
    assert not worker.is_alive()

    assert not errors, f'thread B errored: {errors!r}'

    # Write the profiling results for the test function to analyze
    if outdir is None:
        path = Path.cwd()
    else:
        path = Path(outdir)
    assert path.is_dir()
    prof_a.dump_stats(path / 'stats_a.lprof')
    prof_b.dump_stats(path / 'stats_b.lprof')


def _get_nhits(
    stats: LineStats, funcname: str, line_pattern: str | re.Pattern,
) -> int:
    source, start = inspect.getsourcelines(_scenario)
    if not isinstance(line_pattern, re.Pattern):
        line_pattern = re.compile(line_pattern)
    try:
        ln_match = start + next(
            offset for offset, line in enumerate(source)
            if line_pattern.search(line.rstrip('\n'))
        )
    except StopIteration:
        return 0

    try:
        (entries_match,) = [
            e for (*_, func), e in stats.timings.items()
            if func == funcname or func.rpartition('.')[-1] == funcname
        ]
    except ValueError:  # != 1 match
        return 0

    return sum(
        nhits for lineno, nhits, _ in entries_match if lineno == ln_match
    )


@pytest.mark.parametrize('core', ['sysmon', 'legacy'])
@pytest.mark.parametrize('nprof', [1, 2])
def test_disable_on_one_thread_keeps_other_threads_profiling(
    tmp_path_factory: pytest.TempPathFactory,
    core: Literal['sysmon', 'legacy'],
    nprof: Literal[1, 2],
) -> None:
    """
    Regression tests: disabling a profiler on one thread must not affect
    profilers still active on other threads, be it the same profiler or
    a different one.

    Under the :py:mod:`sys.monitoring` core, registration is
    process-global while the manager bookkeeping is per-thread; the
    first thread's manager used to tear down the global events/tool ID
    as soon as *its* profilers were disabled, silently ending data
    collection for every other thread and making their later
    :py:meth:`LineProfiler.disable` calls raise
    ``ValueError: tool ... is not in use`` from inside user ``finally``
    blocks.

    The core is chosen at import time, so the scenarios run in
    subprocesses (once per core).
    """
    tmp = tmp_path_factory.mktemp('mytmp')
    if core == 'sysmon' and sys.version_info[:2] < (3, 12):
        pytest.skip(
            'Cannot use `sys.monitoring` on Python '
            f'{".".join(str(v) for v in sys.version_info[:3])}'
        )
    env = dict(os.environ)
    env['LINE_PROFILER_CORE'] = core
    proc = subprocess.run(
        [sys.executable, __file__, str(nprof), str(tmp)],
        capture_output=True, env=env, text=True, timeout=120,
    )
    for stream in 'stdout', 'stderr':
        print(
            stream + ':',
            indent(getattr(proc, stream) or '<n/a>', '  '),
            sep='\n', end='', file=getattr(sys, stream),
        )
    assert proc.returncode == 0

    errors: list[str] = []
    for thread, filename, funcname, pattern, expected in [
        ('Main thread', 'stats_a.lprof', 'work_a', r'# grep: BODY \(a\)$', 1),
        ('Thread B', 'stats_b.lprof', 'work_b', r'# grep: LOOP \(b\)$', 2 * N),
    ]:
        stats = LineStats.from_files(tmp / filename)
        actual = _get_nhits(stats, funcname, pattern)
        if actual != expected:
            errors.append(
                f'{thread}: '
                f'expected {expected} hit(s) on the line {pattern}, '
                f'got {actual}',
            )
    assert not errors


if __name__ == '__main__':
    *_, n, outdir = sys.argv
    _scenario(cast(Literal[1, 2], int(n)), outdir)
