"""
Tests for the tagged pool result protocol.

Patched pool workers push ``(PID_TAG, pid, result)`` triplets so the
patched parent can attribute results to worker PIDs.  A worker that was
never patched (e.g. its interpreter never loaded the profiling startup
hook, or it runs via ``multiprocessing.set_executable()`` pointing at a
different Python) pushes vanilla results; the parent must pass those
through with a warning rather than crash its result-handler thread —
the crash presents to the user as ``pool.map()`` hanging forever.
"""
import math
import multiprocessing

import pytest

from line_profiler._child_process_profiling import (
    multiprocessing_patches as mp_patches,
)
from line_profiler._child_process_profiling.multiprocessing_patches import (
    _queue,
)
from line_profiler._child_process_profiling.multiprocessing_patches import (
    _mandatory_patches,
)


GET_TIMEOUT = 60


def test_put_wrapper_tags_results():
    pushed = []

    class FakeQueue:
        put = staticmethod(pushed.append)

    wrapper = _queue.PutWrapper(
        FakeQueue(), lambda: 1234, push_to_parent=True,
    )
    wrapper.put(('job', 0, 'result'))
    assert pushed == [(_queue.PID_TAG, 1234, ('job', 0, 'result'))]


@pytest.mark.parametrize(
    'incoming, expected, is_tagged',
    [
        ((_queue.PID_TAG, 1234, ('job', 0, 'ok')), ('job', 0, 'ok'), True),
        (('job', 0, 'ok'), ('job', 0, 'ok'), False),  # vanilla worker
        (None, None, True),  # queue sentinel, never warns
    ],
)
def test_quick_get_unwrapping(create_cache, incoming, expected, is_tagged):
    cache = create_cache(_use_curated_profiler=False)
    unwrap = _mandatory_patches._wrap_outqueue_quick_get
    if is_tagged:
        assert unwrap(cache, lambda: incoming) == expected
    else:
        with pytest.warns(UserWarning, match='without the profiling'):
            assert unwrap(cache, lambda: incoming) == expected
        # ... but only once per session
        assert unwrap(cache, lambda: incoming) == expected


def _run_pool_map(method):
    import os

    ctx = multiprocessing.get_context(method)
    with ctx.Pool(1) as pool:
        result = pool.map_async(math.sqrt, [0, 1, 4, 9])
        # A hard timeout so that a protocol regression fails fast
        # instead of hanging the suite (the historical failure mode)
        values = result.get(timeout=GET_TIMEOUT)
        # Guard against environments where the pool silently degrades:
        # the values must really come from another process
        worker_pid = pool.apply_async(os.getpid).get(timeout=GET_TIMEOUT)
    assert worker_pid != os.getpid()
    return values


def test_patched_parent_with_vanilla_spawn_worker(create_cache):
    """
    The parent is patched, but the spawn children know nothing of the
    profiling session (no env vars are injected, no .pth hook exists
    for them), so they push vanilla un-tagged results: the map must
    still complete, with a warning.
    """
    if 'spawn' not in multiprocessing.get_all_start_methods():
        pytest.skip("start method 'spawn' unavailable")
    cache = create_cache(_use_curated_profiler=False)
    # Make the wrapped methods resolve `LineProfilingCache.load()` to
    # this instance without injecting env vars (children stay vanilla)
    cache._replace_loaded_instance(force=True)
    mp_patches.apply(cache)
    with pytest.warns(UserWarning, match='without the profiling'):
        assert _run_pool_map('spawn') == [0.0, 1.0, 2.0, 3.0]


def test_patched_parent_with_patched_fork_worker(create_cache):
    """
    Control case: fork children inherit the parent's patches, results
    arrive tagged, and no warning fires.
    """
    if 'fork' not in multiprocessing.get_all_start_methods():
        pytest.skip("start method 'fork' unavailable")
    cache = create_cache()
    cache._replace_loaded_instance(force=True)
    mp_patches.apply(cache)
    import warnings as warnings_mod

    with warnings_mod.catch_warnings():
        warnings_mod.simplefilter('error')
        assert _run_pool_map('fork') == [0.0, 1.0, 2.0, 3.0]
