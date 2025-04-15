import asyncio
import contextlib
import functools
import inspect
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import types
import pytest
from line_profiler import LineProfiler


def f(x):
    """A docstring."""
    y = x + 10
    return y


def g(x):
    y = yield x + 10
    yield y + 20


async def ag(delay, start=()):
    i = 0
    for x in start:
        yield i
        i += x
    while True:
        received = await asyncio.sleep(delay, (yield i))
        if received is None:
            return
        i += received


def get_profiling_tool_name():
    return sys.monitoring.get_tool(sys.monitoring.PROFILER_ID)


def strip(s):
    return textwrap.dedent(s).strip('\n')


@contextlib.contextmanager
def check_timings(prof):
    """
    Verify that the profiler starts without timing data and ends with
    some.
    """
    timings = prof.get_stats().timings
    assert not any(timings.values()), ('Expected no timing entries, '
                                       f'got {timings!r}')
    yield prof
    timings = prof.get_stats().timings
    assert any(timings.values()), f'Expected timing entries, got {timings!r}'


def run_in_subproc(code):
    """
    Run Python code in a subprocess so that it doesn't pollute the
    state of the current interpretor.

    Notes
    -----
    The code is written to a tempfile and run in a subprocess;
    compared with using `runpy.run_path()`, this results in more
    informative error messages.
    """
    code = strip(code)
    with tempfile.TemporaryDirectory() as tmpdir:
        curdir = os.path.abspath(os.curdir)
        os.chdir(tmpdir)
        try:
            fname = 'my_test.py'
            with open(fname, mode='w') as fobj:
                print(code, file=fobj)
            proc = subprocess.run([sys.executable, fname],
                                  capture_output=True, text=True)
        finally:
            os.chdir(curdir)
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    proc.check_returncode()


def test_init():
    lp = LineProfiler()
    assert lp.functions == []
    assert lp.code_map == {}
    lp = LineProfiler(f)
    assert lp.functions == [f]
    assert lp.code_map == {f.__code__: {}}
    lp = LineProfiler(f, g)
    assert lp.functions == [f, g]
    assert lp.code_map == {
        f.__code__: {},
        g.__code__: {},
    }


def test_enable_disable():
    lp = LineProfiler()
    assert lp.enable_count == 0
    lp.enable_by_count()
    assert lp.enable_count == 1
    lp.enable_by_count()
    assert lp.enable_count == 2
    lp.disable_by_count()
    assert lp.enable_count == 1
    lp.disable_by_count()
    assert lp.enable_count == 0
    assert lp.last_time == {}
    lp.disable_by_count()
    assert lp.enable_count == 0

    with lp:
        assert lp.enable_count == 1
        with lp:
            assert lp.enable_count == 2
        assert lp.enable_count == 1
    assert lp.enable_count == 0
    assert lp.last_time == {}

    with pytest.raises(RuntimeError):
        assert lp.enable_count == 0
        with lp:
            assert lp.enable_count == 1
            raise RuntimeError()
    assert lp.enable_count == 0
    assert lp.last_time == {}


def test_trace_callback_preservation():
    """
    Test in a subprocess that the profiler restores the active trace
    callback after it's disabled.
    """
    run_in_subproc(r"""
    import sys
    from typing import Any, Callable, Literal, Union
    from types import FrameType

    from line_profiler import LineProfiler


    Event = Literal['call', 'line', 'return', 'exception', 'opcode']
    TracingFunc = Callable[[FrameType, Event, Any], Union['TracingFunc', None]]


    def test_restoring_trace_callback(call: Union[TracingFunc, None]) -> None:
        sys.settrace(call)
        assert sys.gettrace() is call, f'can\'t set trace to {call!r}'
        profile = LineProfiler(wrap_trace=False)
        profile.enable_by_count()
        assert sys.gettrace() is profile, f'can\'t set trace to the profiler'
        profile.disable_by_count()
        assert sys.gettrace() is call, f'trace not restored to {call!r}'
        sys.settrace(None)


    def main() -> None:
        test_restoring_trace_callback(None)
        test_restoring_trace_callback(lambda frame, event, arg: None)


    if __name__ == '__main__':
        main()
    """)


@pytest.mark.parametrize('wrap_trace', [True, False])
def test_trace_callback_wrapping(wrap_trace: bool) -> None:
    """
    Test in a subprocess that the profiler can wrap around an existing
    trace callback such that we both profile the code and do whatever
    the existing callback does.
    """
    code = strip(r"""
    import contextlib
    import sys
    from io import StringIO
    from types import FrameType
    from typing import Any, Callable, Generator, List, Literal, Union

    from line_profiler import LineProfiler


    DEBUG = False

    Event = Literal['call', 'line', 'return', 'exception', 'opcode']
    TracingFunc = Callable[[FrameType, Event, Any], Union['TracingFunc', None]]


    def foo(n: int) -> int:
        result = 0
        for i in range(1, n + 1):
            result += i
        return result


    def get_logger(logs: List[str]) -> TracingFunc:
        '''
        Append a `foo: i = <...>` message whenever we hit the line in
        `foo()` containing the incrementation of `result`.
        '''
        def callback(frame: FrameType, event: Event,
                     _) -> Union[TracingFunc, None]:
            if DEBUG and callback.emit_debug:
                print('{0.co_filename}:{1.f_lineno} - {0.co_name} ({2})'
                      .format(frame.f_code, frame, event))
            if event == 'call':
                # Set up tracing for nested scopes
                return callback
            if event != 'line':
                return  # Only trace line events
            if frame.f_code.co_name == 'foo' and frame.f_lineno == _LINENO_:
                # Add log entry whenever the target line is hit
                logs.append(f'foo: i = {frame.f_locals.get("i")}')
            return callback

        callback.emit_debug = False
        return callback


    def main(wrap_trace: bool = _WRAP_TRACE_) -> None:
        logs = []
        my_callback = get_logger(logs)
        sys.settrace(my_callback)

        profile = LineProfiler(wrap_trace=wrap_trace)
        foo_wrapped = profile(foo)

        for stage, wrap, expect_output in [
                ('sanity check, no profiling', False, True),
                ('profiled', True, wrap_trace)]:
            assert sys.gettrace() is my_callback, (
                stage + ': can\'t set custom trace')
            foo_like = profile(foo) if wrap else foo
            my_callback.emit_debug = True
            x = foo_like(5)
            my_callback.emit_debug = False
            assert x == 15, f'{stage}: expected `foo(5) = 15`, got {x!r}'
            print(f'Logs ({stage}): {logs!r}')
            assert sys.gettrace() is my_callback, (
                stage + ': trace not restored afterwards')

            # Check that the existing trace function has been called
            # where appropriate
            expected = [f'foo: i = {i}' for i in range(1, 6)]
            if expect_output:
                assert logs == expected, (
                    f'{stage}: expected logs = {expected!r}, '
                    f'got {logs!r}')
            else:
                assert not logs, f'{stage}: expected no logs, got {logs!r}'
            logs.clear()
        sys.settrace(None)

        # Check that the profiling is as expected: 5 hits on the
        # incrementation line
        with StringIO() as sio:
            profile.print_stats(stream=sio, summarize=True)
            out = sio.getvalue()
        print(out)
        line, = (line for line in out.splitlines() if '+=' in line)
        nhits = int(line.split()[1])
        assert nhits == 5, f'expected 5 profiler hits, got {nhits!r}'


    if __name__ == '__main__':
        main()
    """)
    incr_lineno = 1 + next(i for i, line in enumerate(code.splitlines())
                           if line.endswith('result += i'))
    run_in_subproc(code
                   .replace('_LINENO_', str(incr_lineno))
                   .replace('_WRAP_TRACE_', str(wrap_trace)))


def test_double_decoration():
    """
    Test that wrapping the same function twice does not result in
    spurious profiling entries.
    """
    profile = LineProfiler()
    f_wrapped = profile(f)
    f_double_wrapped = profile(f_wrapped)
    assert f_double_wrapped is f_wrapped

    with check_timings(profile):
        assert profile.enable_count == 0
        value = f_wrapped(10)
        assert profile.enable_count == 0
        assert value == f(10)

    assert len(profile.get_stats().timings) == 1


def test_function_decorator():
    """
    Test for `LineProfiler.wrap_function()`.
    """
    profile = LineProfiler()
    f_wrapped = profile(f)
    assert f in profile.functions
    assert f_wrapped.__name__ == 'f'

    with check_timings(profile):
        assert profile.enable_count == 0
        value = f_wrapped(10)
        assert profile.enable_count == 0
        assert value == f(10)


def test_gen_decorator():
    """
    Test for `LineProfiler.wrap_generator()`.
    """
    profile = LineProfiler()
    g_wrapped = profile(g)
    assert inspect.isgeneratorfunction(g_wrapped)
    assert g in profile.functions
    assert g_wrapped.__name__ == 'g'

    with check_timings(profile):
        assert profile.enable_count == 0
        i = g_wrapped(10)
        assert profile.enable_count == 0
        assert next(i) == 20
        assert profile.enable_count == 0
        assert i.send(30) == 50
        assert profile.enable_count == 0
        with pytest.raises(StopIteration):
            next(i)
        assert profile.enable_count == 0


def test_coroutine_decorator():
    """
    Test for `LineProfiler.wrap_coroutine()`.
    """
    async def coro(delay=.015625):
        return (await asyncio.sleep(delay, 1))

    profile = LineProfiler()
    coro_wrapped = profile(coro)
    assert inspect.iscoroutinefunction(coro)
    assert coro in profile.functions

    with check_timings(profile):
        assert profile.enable_count == 0
        assert asyncio.run(coro_wrapped()) == 1
        assert profile.enable_count == 0


def test_async_gen_decorator():
    """
    Test for `LineProfiler.wrap_async_generator()`.
    """
    delay = .015625

    async def use_agen_complex(*args, delay=delay):
        results = []
        agen = ag_wrapped(delay)
        results.append(await agen.asend(None))  # Start the generator
        for send in args:
            with (pytest.raises(StopAsyncIteration)
                  if send is None else
                  contextlib.nullcontext()):
                results.append(await agen.asend(send))
            if send is None:
                break
        return results

    async def use_agen_simple(*args, delay=delay):
        results = []
        async for i in ag_wrapped(delay, args):
            results.append(i)
        return results

    profile = LineProfiler()
    ag_wrapped = profile(ag)
    assert inspect.isasyncgenfunction(ag_wrapped)
    assert ag in profile.functions

    with check_timings(profile):
        assert profile.enable_count == 0
        assert asyncio.run(use_agen_simple()) == [0]
        assert profile.enable_count == 0
        assert asyncio.run(use_agen_simple(1, 2, 3)) == [0, 1, 3, 6]
        assert profile.enable_count == 0
        assert asyncio.run(use_agen_complex(1, 2, 3)) == [0, 1, 3, 6]
        assert profile.enable_count == 0
        assert asyncio.run(use_agen_complex(1, 2, 3, None, 4)) == [0, 1, 3, 6]
        assert profile.enable_count == 0


def test_classmethod_decorator():
    """
    Test for `LineProfiler.wrap_classmethod()`.

    Notes
    -----
    This is testing for an edge case;
    for the best result, always use `@profile` as the innermost
    decorator, as auto-profile normally does.
    """
    profile = LineProfiler()

    class Object:
        @profile
        @classmethod
        def foo(cls) -> str:
            return cls.__name__ * 2

    assert isinstance(inspect.getattr_static(Object, 'foo'), classmethod)
    assert profile.enable_count == 0
    assert len(profile.functions) == 1
    assert Object.foo() == Object().foo() == 'ObjectObject'
    with io.StringIO() as sio:
        profile.print_stats(stream=sio, summarize=True)
        output = strip(sio.getvalue())
    print(output)
    # Check that we have profiled `Object.foo()`
    assert output.endswith('- foo')
    line, = (line for line in output.splitlines() if line.endswith('* 2'))
    # Check that it has been run twice
    assert int(line.split()[1]) == 2
    assert profile.enable_count == 0


def test_staticmethod_decorator():
    """
    Test for `LineProfiler.wrap_staticmethod()`.

    Notes
    -----
    This is testing for an edge case;
    for the best result, always use `@profile` as the innermost
    decorator, as auto-profile normally does.
    """
    profile = LineProfiler()

    class Object:
        @profile
        @staticmethod
        def foo(x: int) -> int:
            return x * 2

    assert isinstance(inspect.getattr_static(Object, 'foo'), staticmethod)
    assert profile.enable_count == 0
    assert len(profile.functions) == 1
    assert Object.foo(3) == Object().foo(3) == 6
    with io.StringIO() as sio:
        profile.print_stats(stream=sio, summarize=True)
        output = strip(sio.getvalue())
    print(output)
    # Check that we have profiled `Object.foo()`
    assert output.endswith('- foo')
    line, = (line for line in output.splitlines() if line.endswith('* 2'))
    # Check that it has been run twice
    assert int(line.split()[1]) == 2
    assert profile.enable_count == 0


def test_boundmethod_decorator():
    """
    Test for `LineProfiler.wrap_boundmethod()`.

    Notes
    -----
    This is testing for an edge case;
    for the best result, always use `@profile` as the innermost
    decorator, as auto-profile normally does.
    """
    profile = LineProfiler()

    class Object:
        def foo(self, x: int) -> int:
            return id(self) * x

    obj = Object()
    # Check that calls are aggregated
    profiled_foo_1 = profile(obj.foo)
    profiled_foo_2 = profile(obj.foo)
    assert isinstance(profiled_foo_1, types.MethodType)
    assert isinstance(profiled_foo_2, types.MethodType)
    assert profile.enable_count == 0
    # XXX: should we try do remove duplicates?
    assert profile.functions == [Object.foo, Object.foo]
    assert (profiled_foo_1(2)
            == profiled_foo_2(2)
            == obj.foo(2)
            == id(obj) * 2)
    with io.StringIO() as sio:
        profile.print_stats(stream=sio, summarize=True)
        output = strip(sio.getvalue())
    print(output)
    # Check that we have profiled `Object.foo()`
    assert output.endswith('- foo')
    line, = (line for line in output.splitlines() if line.endswith('* x'))
    # Check that the wrapped methods has been run twice in total
    assert int(line.split()[1]) == 2
    assert profile.enable_count == 0


def test_partialmethod_decorator():
    """
    Test for `LineProfiler.wrap_partialmethod()`

    Notes
    -----
    This is testing for an edge case;
    for the best result, always use `@profile` as the innermost
    decorator in a function definition, as auto-profile normally does.
    """
    profile = LineProfiler()

    class Object:
        def foo(self, x: int) -> int:
            return id(self) * x

        bar = profile(functools.partialmethod(foo, 1))

    assert isinstance(inspect.getattr_static(Object, 'bar'),
                      functools.partialmethod)
    obj = Object()
    assert profile.enable_count == 0
    assert profile.functions == [Object.foo]
    assert obj.foo(1) == obj.bar() == id(obj)
    with io.StringIO() as sio:
        profile.print_stats(stream=sio, summarize=True)
        output = strip(sio.getvalue())
    print(output)
    # Check that we have profiled `Object.foo()` (via `.bar()`)
    assert output.endswith('- foo')
    line, = (line for line in output.splitlines() if line.endswith('* x'))
    # Check that the wrapped method has been run once
    assert int(line.split()[1]) == 1
    assert profile.enable_count == 0


def test_partial_decorator() -> None:
    """
    Test for `LineProfiler.wrap_partial()`.

    Notes
    -----
    This is testing for an edge case;
    for the best result, always use `@profile` as the innermost
    decorator, as auto-profile normally does.
    """
    profile = LineProfiler()

    def foo(x: int, y: int) -> int:
        return x + y

    bar = functools.partial(foo, 2)
    profiled_bar_1 = profile(bar)
    profiled_bar_2 = profile(bar)
    assert isinstance(profiled_bar_1, functools.partial)
    assert isinstance(profiled_bar_2, functools.partial)
    assert profile.enable_count == 0
    # XXX: should we try do remove duplicates?
    assert profile.functions == [foo, foo]
    assert (profiled_bar_1(3)
            == profiled_bar_2(3)
            == bar(3)
            == foo(2, 3)
            == 5)
    with io.StringIO() as sio:
        profile.print_stats(stream=sio, summarize=True)
        output = strip(sio.getvalue())
    print(output)
    # Check that we have profiled `foo()`
    assert output.endswith('- foo')
    line, = (line for line in output.splitlines() if line.endswith('x + y'))
    # Check that the wrapped partials has been run twice in total
    assert int(line.split()[1]) == 2
    assert profile.enable_count == 0


def test_property_decorator():
    """
    Test for `LineProfiler.wrap_property()`.

    Notes
    -----
    This is testing for an edge case;
    for the best result, always use `@profile` as the innermost
    decorator, as auto-profile normally does.
    """
    profile = LineProfiler()

    class Object:
        def __init__(self, x: int) -> None:
            self.x = x

        @profile
        @property
        def foo(self) -> int:
            return self.x * 2

        # The profiler sees both the setter and the already-wrapped
        # getter here, but it shouldn't re-wrap the getter

        @profile
        @foo.setter
        def foo(self, foo) -> None:
            self.x = foo // 2

    assert isinstance(Object.foo, property)
    assert profile.enable_count == 0
    assert len(profile.functions) == 2
    obj = Object(3)
    assert obj.foo == 6  # Use getter
    obj.foo = 10  # Use setter
    assert obj.x == 5
    assert obj.foo == 10  # Use getter
    with io.StringIO() as sio:
        profile.print_stats(stream=sio, summarize=True)
        output = strip(sio.getvalue())
    print(output)
    # Check that we have profiled `Object.foo`
    assert output.endswith('- foo')
    getter_line, = (line for line in output.splitlines()
                    if line.endswith('* 2'))
    setter_line, = (line for line in output.splitlines()
                    if line.endswith('// 2'))
    # Check that the getter has been run twice and the setter once
    assert int(getter_line.split()[1]) == 2
    assert int(setter_line.split()[1]) == 1
    assert profile.enable_count == 0


def test_cached_property_decorator():
    """
    Test for `LineProfiler.wrap_cached_property()`

    Notes
    -----
    This is testing for an edge case;
    for the best result, always use `@profile` as the innermost
    decorator, as auto-profile normally does.
    """
    profile = LineProfiler()

    class Object:
        def __init__(self, x: int) -> None:
            self.x = x

        @profile
        @functools.cached_property
        def foo(self) -> int:
            return self.x * 2

    assert isinstance(Object.foo, functools.cached_property)
    assert profile.enable_count == 0
    assert len(profile.functions) == 1
    obj = Object(3)
    assert obj.foo == 6  # Use getter
    assert obj.foo == 6  # Getter not called because it's cached
    with io.StringIO() as sio:
        profile.print_stats(stream=sio, summarize=True)
        output = strip(sio.getvalue())
    print(output)
    # Check that we have profiled `Object.foo`
    assert output.endswith('- foo')
    line, = (line for line in output.splitlines() if line.endswith('* 2'))
    # Check that the getter has been run once
    assert int(line.split()[1]) == 1
    assert profile.enable_count == 0


def test_show_func_column_formatting():
    from line_profiler.line_profiler import show_func
    import line_profiler
    import io
    # Use a function in this module as an example
    func = line_profiler.line_profiler.show_text
    start_lineno = func.__code__.co_firstlineno
    filename = func.__code__.co_filename
    func_name = func.__name__

    def get_func_linenos(func):
        import sys
        if sys.version_info[0:2] >= (3, 10):
            return sorted(set([t[0] if t[2] is None else t[2]
                               for t in func.__code__.co_lines()]))
        else:
            import dis
            return sorted(set([t[1] for t in dis.findlinestarts(func.__code__)]))
    line_numbers = get_func_linenos(func)

    unit = 1.0
    output_unit = 1.0
    stripzeros = False

    # Build fake timeings for each line in the example function
    timings = [
        (lineno, idx * 1e13, idx * (2e10 ** (idx % 3)))
        for idx, lineno in enumerate(line_numbers, start=1)
    ]
    stream = io.StringIO()
    show_func(filename, start_lineno, func_name, timings, unit,
              output_unit, stream, stripzeros)
    text = stream.getvalue()
    print(text)

    timings = [
        (lineno, idx * 1e15, idx * 2e19)
        for idx, lineno in enumerate(line_numbers, start=1)
    ]
    stream = io.StringIO()
    show_func(filename, start_lineno, func_name, timings, unit,
              output_unit, stream, stripzeros)
    text = stream.getvalue()
    print(text)

    # TODO: write a check to verify columns are aligned nicely


@pytest.mark.skipif(not hasattr(sys, 'monitoring'),
                    reason='no `sys.monitoring` in version '
                    f'{".".join(str(v) for v in sys.version_info[:2])}')
def test_sys_monitoring():
    """
    Test that `LineProfiler` is properly registered with
    `sys.monitoring`.
    """
    profile = LineProfiler()
    get_name_wrapped = profile(get_profiling_tool_name)
    tool = get_profiling_tool_name()
    assert tool is None, (
        f'Expected no active profiling tool before profiling, got {tool!r}'
    )
    tool = get_name_wrapped()
    assert tool == 'line_profiler', (
        "Expected 'line_profiler' to be registered with `sys.monitoring` "
        f'when a profiled function is run, got {tool!r}'
    )
    tool = get_profiling_tool_name()
    assert tool is None, (
        f'Expected no active profiling tool after profiling, got {tool!r}'
    )
