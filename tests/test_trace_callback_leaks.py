"""
Regression tests: the C helpers in ``c_trace_callbacks.c`` must not
leak the wrapper objects they create.

``call_callback()`` wraps a foreign frame-local trace function with
``disable_line_events()`` when that function turns off
``frame.f_trace_lines``, and ``set_local_trace()`` wraps foreign local
trace functions with ``wrap_local_f_trace()``.  Both store the wrapper
on ``frame.f_trace``, whose setter takes its own reference; holding on
to the creation reference therefore leaked one wrapper per event,
which adds up on long profiled runs under a debugger/coverage tool.

The wrappers masquerade as the functions they wrap (``@wraps`` copies
``__qualname__``), so the checks capture ``frame.f_trace`` from inside
the profiled function and assert via weakref that the wrappers die
with their frames.  The scenario only exists under the legacy trace
core, so it runs in a subprocess with ``LINE_PROFILER_CORE=legacy``.
"""
import subprocess
import sys
import textwrap


_SCENARIO = textwrap.dedent(
    """
    import gc
    import sys
    import weakref

    from line_profiler import LineProfiler


    wrapper_refs = []


    def capture_frame_trace():
        # The frame-local trace of the *caller* of this function
        f_trace = sys._getframe(1).f_trace
        if f_trace is None or type(f_trace).__name__ == '_LineProfilerManager':
            # Only interested in the wrapper objects created around
            # foreign trace functions, not in the profiler itself
            return
        wrapper_refs.append(weakref.ref(f_trace))


    def foo(n):
        result = 0
        for spam in range(1, n + 1):
            result += spam
        capture_frame_trace()
        return result


    def make_callback(bugged):
        def callback(frame, event, arg):
            if (
                bugged
                and event == 'line'
                and frame.f_code.co_name == 'foo'
            ):
                # Turns off line events, triggering the
                # `disable_line_events()` wrapping in `call_callback()`
                frame.f_trace_lines = False
            return callback

        return callback


    def scenario(bugged, repeat=5):
        # Everything traced happens inside this function so that by the
        # time the caller checks the weakrefs, every traced frame is
        # dead and any live wrapper can only be held by a leaked
        # reference.
        sys.settrace(make_callback(bugged))
        try:
            profile = LineProfiler(
                wrap_trace=True, set_frame_local_trace=True,
            )
            foo_prof = profile(foo)
            for _ in range(repeat):
                assert foo_prof(5) == 15
        finally:
            sys.settrace(None)


    for bugged in (False, True):
        wrapper_refs.clear()
        scenario(bugged)
        gc.collect()
        alive = [ref for ref in wrapper_refs if ref() is not None]
        assert wrapper_refs, 'scenario failed to produce trace wrappers'
        assert not alive, (
            f'{len(alive)}/{len(wrapper_refs)} trace wrapper(s) still '
            f'alive after all traced frames died (bugged={bugged}): '
            f'{[ref() for ref in alive]!r}'
        )
    print('OK')
    """
)


def test_trace_wrappers_are_not_leaked(tmp_path, monkeypatch):
    script = tmp_path / 'leak_scenario.py'
    script.write_text(_SCENARIO)
    monkeypatch.setenv('LINE_PROFILER_CORE', 'legacy')
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert proc.stdout.strip() == 'OK'
