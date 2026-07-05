"""
Regression tests: profiling data must be attributed to the process that
produced it exactly once.

Forked children inherit the parent profiler's accumulated stats; if they
dump them verbatim, every line the parent executed before the fork is
counted once per forked child when the results are merged (the parent
dumps the same data itself).  See
``LineProfilingCache._wrap_os_fork`` and ``_dump_profiler_stats`` for
the subtractive-baseline fix these tests pin down.
"""
import multiprocessing
import subprocess
import sys

import pytest

from line_profiler import LineStats


LOOP_COUNT = 5000
NUM_TASKS = 4
TIMEOUT = 120

_API_SNIPPETS = {
    'process': """
    procs = [ctx.Process(target=child_work, args=(i,))
             for i in range(num_children)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
""",
    'pool': """
    with ctx.Pool(2) as pool:
        results = pool.map(child_work, range(num_tasks))
    assert results == [i * 2 for i in range(num_tasks)]
""",
    'cfut': """
    import concurrent.futures as cf

    with cf.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        results = list(ex.map(child_work, range(num_tasks)))
    assert results == [i * 2 for i in range(num_tasks)]
""",
}


def _write_workload(path, method, api, num_children=1):
    source = f"""
import multiprocessing as mp


def child_work(x):
    doubled = x * 2
    return doubled


def main():
    num_tasks = {NUM_TASKS}
    num_children = {num_children}
    acc = 0
    for i in range({LOOP_COUNT}):
        acc += i
    ctx = mp.get_context({method!r})
{_API_SNIPPETS[api]}

if __name__ == '__main__':
    main()
"""
    path.write_text(source)
    lines = source.splitlines()
    return {
        'acc': 1 + lines.index('        acc += i'),
        'child_work': 1 + lines.index('    doubled = x * 2'),
    }


def _run_kernprof(tmp_path, script):
    """
    Run the real CLI in a real subprocess (the .pth/env machinery only
    fully engages for a fresh interpreter) and load the merged stats.
    """
    outfile = tmp_path / 'out.lprof'
    proc = subprocess.run(
        [sys.executable, '-m', 'kernprof',
         '-l', f'--prof-mod={script}', '--prof-child-procs',
         f'--outfile={outfile}', str(script)],
        cwd=tmp_path, capture_output=True, text=True, timeout=TIMEOUT,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    return LineStats.from_files(outfile)


def _nhits(stats, func_name, lineno):
    matches = [
        entry
        for (_, _, func), entries in stats.timings.items()
        if func == func_name
        for entry in entries
        if entry[0] == lineno
    ]
    assert matches, (
        f'no timings recorded for line {lineno} of {func_name}(); '
        f'keys = {sorted(stats.timings)!r}'
    )
    return sum(nhits for _, nhits, _ in matches)


@pytest.mark.parametrize('api', sorted(_API_SNIPPETS))
@pytest.mark.parametrize('method', ['fork', 'forkserver', 'spawn'])
def test_exact_stats_across_start_methods(tmp_path, method, api):
    """
    The merged profile must show the parent's loop exactly once and the
    children's work exactly ``NUM_TASKS`` times, no matter how the
    children came into existence.
    """
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f'start method {method!r} unavailable')
    script = tmp_path / 'workload.py'
    expected_child_hits = 1 if api == 'process' else NUM_TASKS
    linenos = _write_workload(script, method, api)
    stats = _run_kernprof(tmp_path, script)
    assert _nhits(stats, 'main', linenos['acc']) == LOOP_COUNT
    assert (
        _nhits(stats, 'child_work', linenos['child_work'])
        == expected_child_hits
    )


def test_stats_do_not_scale_with_fork_children(tmp_path):
    """
    Pre-fork parent data must not be re-contributed once per child.
    """
    if 'fork' not in multiprocessing.get_all_start_methods():
        pytest.skip("start method 'fork' unavailable")
    script = tmp_path / 'workload.py'
    linenos = _write_workload(script, 'fork', 'process', num_children=3)
    stats = _run_kernprof(tmp_path, script)
    assert _nhits(stats, 'main', linenos['acc']) == LOOP_COUNT
    assert _nhits(stats, 'child_work', linenos['child_work']) == 3
