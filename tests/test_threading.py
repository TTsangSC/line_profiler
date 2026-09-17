"""
Test that our patch for :py:mod:`threading` is robust.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections.abc import Callable, Collection, Generator
from contextlib import ExitStack
from functools import partial
from inspect import unwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from threading import Condition, Thread
from types import FunctionType, ModuleType
from typing import Any, ClassVar, Literal, cast
from uuid import uuid4

import pytest

from line_profiler._line_profiler import label as code_to_timing_key
from line_profiler._threading_patches import apply as apply_patches
from line_profiler.line_profiler_utils import restore
from line_profiler.cleanup import Cleanup
from line_profiler.curated_profiling import CuratedProfilerContext
from line_profiler.line_profiler import LineProfiler, LineStats


LOOP_BODY_LINE: int = 0  # Placeholder to appease the linter


@pytest.fixture(scope='module')
def _my_module() -> Generator[Path, None, None]:
    module_text = dedent("""
    from __future__ import annotations

    from contextlib import nullcontext
    from os import PathLike


    def func(n: int, file: str | PathLike[str] | None = None) -> None:
        with (
            nullcontext(None) if file is None else open(file, mode='w')
        ) as fobj:
            for i in range(1, n + 1):
                print(i, file=fobj)
    """).strip('\n')

    global LOOP_BODY_LINE
    LOOP_BODY_LINE = len(module_text.splitlines())

    with TemporaryDirectory() as tmpdir:
        module = Path(tmpdir) / (
            next(propose_module_name('my_module')) + '.py'
        )
        module.write_text(module_text)
        yield module


@pytest.fixture
def my_module_path(_my_module: Path) -> Generator[Path, None, None]:
    with pytest.MonkeyPatch.context() as mp:
        tmpdir = _my_module.parent
        mp.syspath_prepend(str(tmpdir))
        mp.setenv('PYTHONPATH', str(tmpdir), os.pathsep)

        yield _my_module


@pytest.fixture
def my_module_obj(my_module_path: Path) -> Generator[ModuleType, None, None]:
    module = my_module_path.stem
    with restore.mapping(sys.modules, [module]):
        sys.modules.pop(module, None)
        importlib.invalidate_caches()

        yield importlib.import_module(module)


@pytest.fixture
def prof() -> Generator[LineProfiler, None, None]:
    try:
        prof = LineProfiler()
        yield prof
    finally:
        reset_enable_count(prof)


def propose_module_name(
    base: str, existing: Collection[str] | None = None,
) -> Generator[str, None, None]:
    if existing is None:
        existing = sys.modules
    if base not in existing:
        yield base
    while True:
        uuid = str(uuid4()).replace('-', '_')
        name = f'{base}_{uuid}'
        if name not in existing:
            yield name


def reset_enable_count(prof: LineProfiler, count: int = 0) -> None:
    delta = count - cast(int, getattr(prof, 'enable_count', None))
    if delta > 0:
        for _ in range(delta):
            prof.enable_by_count()
    elif delta < 0:
        for _ in range(-delta):
            prof.disable_by_count()
    assert getattr(prof, 'enable_count', None) == count


def get_nhits(stats: LineStats, func: FunctionType, lineno: int) -> int:
    timings: list[tuple[int, int, int]]
    timings = stats.timings.get(code_to_timing_key(func.__code__), [])
    return {lineno: nhits for lineno, nhits, _ in timings}.get(lineno, 0)


@pytest.mark.parametrize('count_at_creation', [0, 1, 2])
@pytest.mark.parametrize('count_at_start', [0, 1, 2])
@pytest.mark.parametrize('use_wrapper', [True, False])
@pytest.mark.parametrize('use_curated_ctx', [True, False])
def test_child_thread_profiling_toggle_by_count(
    my_module_obj: ModuleType,
    prof: LineProfiler,
    count_at_creation: int,
    count_at_start: int,
    use_wrapper: bool,
    use_curated_ctx: bool,
) -> None:
    """
    Test the profiling of child threads when the
    :py:attr:`LineProfiler.enable_count` has been manipulated in
    different manners.
    """
    n = 10
    target = my_module_obj.func
    if use_wrapper:
        target = prof(target)
        is_profiling = True
    else:
        prof.add_callable(target)
        is_profiling = count_at_start > 0

    with ExitStack() as stack:
        if use_curated_ctx:
            stack.enter_context(CuratedProfilerContext(prof))
        else:  # Basically the same, but more explicit
            cleanup = stack.enter_context(Cleanup())
            apply_patches(cleanup, prof)

        reset_enable_count(prof, count_at_creation)
        thread = Thread(target=target, args=(n,))

        reset_enable_count(prof, count_at_start)
        thread.start()
        thread.join()

        nhits = get_nhits(prof.get_stats(), unwrap(target), LOOP_BODY_LINE)
        assert nhits == (n if is_profiling else 0)


@pytest.mark.parametrize('use_wrapper', [True, False])
def test_child_thread_profiling_subclassed(
    my_module_obj: ModuleType, prof: LineProfiler, use_wrapper: bool,
) -> None:
    """
    Test profiling the workload in a :py:class:`Thread` subclass which
    directly overrides the :py:meth:`Thread.run` method.
    """
    target: Callable[[int], None] = my_module_obj.func

    class MyThread(Thread):
        def run(self) -> None:
            self.worker(self.n)

        @staticmethod
        def worker(n: int) -> None:
            target(n)

        @classmethod
        def use(cls) -> None:
            thread = cls()
            thread.start()
            thread.join()

        n: ClassVar[int] = 10

    prof.enable_by_count()
    with CuratedProfilerContext(prof):
        if use_wrapper:
            target = prof(target)
        else:
            prof.add_callable(target)

        MyThread.use()

        nhits = get_nhits(prof.get_stats(), unwrap(target), LOOP_BODY_LINE)
        assert nhits == MyThread.n


@pytest.mark.parametrize('use_wrapper', [True, False])
def test_child_thread_profiling_separate_creation_and_consumption(
    my_module_obj: ModuleType, prof: LineProfiler, use_wrapper: bool,
) -> None:
    """
    Test profiling in a thread whose instantiation and use happened on
    different threads.
    """
    target: Callable[[int], None] = my_module_obj.func

    def construct_thread(cond: Condition, namespace: dict[str, Any]) -> None:
        with cond:
            namespace['thread'] = Thread(target=target, args=(n,))
            cond.notify_all()

    def use_constructed_thread(
        cond: Condition, namespace: dict[str, Any],
    ) -> None:
        with cond:
            cond.wait()
            thread = namespace['thread']
            thread.start()
            thread.join()

    n = 10

    prof.enable_by_count()
    with CuratedProfilerContext(prof):
        if use_wrapper:
            target = prof(target)
        else:
            prof.add_callable(target)

        namespace: dict[str, Any] = {}
        new_thread = partial(Thread, args=(Condition(), namespace))
        constructor_thread = new_thread(target=construct_thread)
        consumer_thread = new_thread(target=use_constructed_thread)
        # Start the consumer first to ensure that the underlying
        # "physical" thread id isn't reused between the two
        consumer_thread.start()
        constructor_thread.start()
        assert (
            None
            is not consumer_thread.ident
            != constructor_thread.ident
            is not None
        )
        constructor_thread.join()
        consumer_thread.join()

        nhits = get_nhits(prof.get_stats(), unwrap(target), LOOP_BODY_LINE)
        assert nhits == n


@pytest.mark.parametrize('lp_core', ['old', 'new'])
def test_child_thread_profiling_in_kernprof(
    tmp_path_factory: pytest.TempPathFactory,
    my_module_obj: ModuleType,
    lp_core: Literal['old', 'new'],
) -> None:
    """
    End-to-end test for profiling multithreaded code with
    :py:mod:`kernprof`.
    """
    if (sys.version_info[:2]) < (3, 12) and lp_core == 'new':
        pytest.skip(
            f"Can't use `sys.monitoring` in Python "
            f'{".".join(str(v) for v in sys.version_info[:3])}'
        )
    tmpdir = tmp_path_factory.mktemp('mytemp')
    module = my_module_obj.__name__

    nfiles = 4
    test_code = dedent(f"""
    from __future__ import annotations

    import os
    from threading import Thread
    from {module} import func


    threads: list[Thread] = []

    for i in range(1, {nfiles + 1}):
        output = os.path.join({str(tmpdir)!r}, f'out-{{i}}.txt')
        threads.append(Thread(target=func, args=(i, output)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    """).strip('\n')
    stats_file = tmpdir / 'out.lprof'

    cmd = [
        'kernprof', '-l', f'--prof-mod={module}',
        f'--outfile={stats_file}',
        '-c', test_code,
    ]
    env = dict(os.environ)
    env['LINE_PROFILER_CORE'] = lp_core

    # Verify the code execution
    subprocess.run(cmd, check=True, env=env)
    for i in range(1, nfiles + 1):
        out = tmpdir / f'out-{i}.txt'
        assert out.exists()
        assert out.read_text() == ''.join(f'{n}\n' for n in range(1, i + 1))

    # Verify the profiling output
    total_nhits = (nfiles + 1) * nfiles / 2
    assert total_nhits == get_nhits(
        LineStats.from_files(stats_file),
        my_module_obj.func,
        LOOP_BODY_LINE,
    )
