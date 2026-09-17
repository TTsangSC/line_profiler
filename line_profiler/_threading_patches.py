"""
Patch :py:mod:`threading` so that profiling extends consistenly into
processes it creates.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, TypeVar
from typing_extensions import ParamSpec, Concatenate

from ._line_profiler import (  # type: ignore
    USE_LEGACY_TRACE as SHOULD_PATCH_THREADING,
)
from .line_profiler import LineProfiler
from .cleanup import Cleanup


__all__ = ('apply', 'SHOULD_PATCH_THREADING')


T = TypeVar('T')
PS = ParamSpec('PS')

_PATCHED_MARKER = '__line_profiler_patched_threading__'


def make_syncing_wrapper(
    func: Callable[PS, T], prof: LineProfiler, enable_count: int,
) -> Callable[PS, T]:
    """
    Wrap the callable ``func`` so that when we spin up a new thread, we
    sync the
    :py:attr:`line_profiler.line_profiler.LineProfiler.enable_count`  of
    the active profiler (stored at the cache instance loaded from
    :py:meth:`LineProfilingCache.load`) with ``enable_count``.

    Note:
        This only seems to work as intended when using the legacy trace
        system...
    """
    @wraps(func)
    def wrapper(*args: PS.args, **kwargs: PS.kwargs) -> T:
        if TYPE_CHECKING:
            assert hasattr(prof, 'enable_count')
            assert isinstance(prof.enable_count, int)
        # Note: `prof.enable_count` is most likely to be zero on the new
        # thread
        thread_enable_count: int = prof.enable_count
        for _ in range(enable_count - thread_enable_count):
            prof.enable_by_count()
        try:
            return func(*args, **kwargs)
        finally:
            # Reset enable counts to avoid problems if the thread id is
            # ever reused
            for _ in range(prof.enable_count - thread_enable_count):
                prof.disable_by_count()

    return wrapper


def make_thread_start_wrapper(
    prof: LineProfiler,
    vanilla_impl: Callable[Concatenate[threading.Thread, PS], None],
) -> Callable[Concatenate[threading.Thread, PS], None]:
    """
    Wrap :py:meth:`threading.Thread.start` so that the profiler's
    :py:attr:`LineProfiler.enable_count` is synced up on newly spun-up
    threads.
    """
    @wraps(vanilla_impl)
    def wrapper(
        self: threading.Thread, *args: PS.args, **kwargs: PS.kwargs
    ) -> None:
        if TYPE_CHECKING:
            assert hasattr(self, '_bootstrap')
        enable_count: int | None = getattr(prof, 'enable_count', None)
        bootstrap: Callable[..., Any] | MethodType = self._bootstrap
        if enable_count:
            if isinstance(bootstrap, MethodType):
                unbound_wrapper = make_syncing_wrapper(
                    bootstrap.__func__, prof, enable_count,
                )
                bootstrap = MethodType(unbound_wrapper, bootstrap.__self__)
            else:
                bootstrap = make_syncing_wrapper(bootstrap, prof, enable_count)
            # `.start()` passes `._bootstrap()` to some lower-level
            # function to spin up the new thread.
            self._bootstrap = bootstrap  # type: ignore
        vanilla_impl(self, *args, **kwargs)

    return wrapper


def apply(cleanup: Cleanup, prof: LineProfiler) -> None:
    """
    Set up profiling in threads started by :py:mod:`threading` by
    applying patches to the module.

    Args:
        cleanup (Cleanup)
            Cleanup instance managing the profiling session

    Side effects:
        - :py:mod:`threading` marked as having been set up

        - The following methods and functions patched:

          - :py:meth:`threading.Thread.start`

        - Cleanup callbacks registered via ``cleanup.add_cleanup()``

    Note:
        This is a no-op when using :py:mod:`sys.monitoring`-based
        profiling.
    """
    if not SHOULD_PATCH_THREADING:
        return
    if getattr(threading, _PATCHED_MARKER, False):
        return
    start_wrapper = make_thread_start_wrapper(prof, threading.Thread.start)
    cleanup.patch(threading.Thread, 'start', start_wrapper)
    cleanup.patch(threading, _PATCHED_MARKER, True)
