"""
Patch :py:mod:`threading` so that profiling extends consistenly into
processes it creates.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from functools import wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload
from typing_extensions import ParamSpec, Concatenate

from .line_profiler import LineProfiler
from .cleanup import Cleanup


__all__ = ('apply',)


T = TypeVar('T')
PS = ParamSpec('PS')

_PATCHED_MARKER = '__line_profiler_patched_threading__'


@overload
def make_syncing_wrapper(
    func: Callable[PS, T], prof: LineProfiler, enable_count: int,
) -> Callable[PS, T]:
    ...


@overload
def make_syncing_wrapper(
    func: MethodType, prof: LineProfiler, enable_count: int,
) -> MethodType:
    ...


def make_syncing_wrapper(
    func: Callable[PS, T] | MethodType, prof: LineProfiler, enable_count: int,
) -> Callable[PS, T] | MethodType:
    """
    Wrap the callable ``func`` so that when we spin up a new thread, we
    sync the
    :py:attr:`line_profiler.line_profiler.LineProfiler.enable_count`  of
    the active profiler (stored at the cache instance loaded from
    :py:meth:`LineProfilingCache.load`) with ``enable_count``.
    """
    if isinstance(func, MethodType):
        impl = make_syncing_wrapper(func.__func__, prof, enable_count)
        return MethodType(impl, func.__self__)

    @wraps(func)
    def wrapper(*args: PS.args, **kwargs: PS.kwargs) -> T:
        if TYPE_CHECKING:
            assert hasattr(prof, 'enable_count')
            assert isinstance(prof.enable_count, int)
        # Note: `prof.enable_count` should be zero on the new thread
        for _ in range(enable_count):
            prof.enable_by_count()
        try:
            return func(*args, **kwargs)
        finally:
            # Reset enable counts to avoid problems if the "physical"
            # thread id is ever reused
            for _ in range(prof.enable_count):
                prof.disable_by_count()

    return wrapper


def wrap_thread_start(
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
        bootstrap: Callable[..., Any] | MethodType = cast(Any, self._bootstrap)
        if enable_count:
            # `.start()` passes `._bootstrap()` to some lower-level
            # function to spin up the new thread.
            bootstrap = wrap_thread_bootstrap(prof, bootstrap, enable_count)
            # This is a private method; we don't care about restoring it
            self._bootstrap = bootstrap  # type: ignore
        vanilla_impl(self, *args, **kwargs)

    return wrapper


@overload
def wrap_thread_bootstrap(
    prof: LineProfiler,
    vanilla_impl: Callable[Concatenate[threading.Thread, PS], None],
    enable_count: int,
) -> Callable[Concatenate[threading.Thread, PS], None]:
    ...


@overload
def wrap_thread_bootstrap(
    prof: LineProfiler, vanilla_impl: MethodType, enable_count: int,
) -> MethodType:
    ...


def wrap_thread_bootstrap(
    prof: LineProfiler,
    vanilla_impl: Callable[
        Concatenate[threading.Thread, PS], None
    ] | MethodType,
    enable_count: int,
) -> Callable[Concatenate[threading.Thread, PS], None] | MethodType:
    """
    Wrap :py:meth:`threading.Thread._bootstrap` so that the profiler's
    :py:attr:`LineProfiler.enable_count` is synced up on newly spun-up
    threads.

    Notes:
        This is separate from :py:func:`wrap_thread_start` because:

        - :py:func:`wrap_thread_start` is responsible for
          capturing ``prof.enable_count`` at startup on the parent
          thread.

        - However, if :py:func:`threading.settrace` is used, the
          supplied callable will override the legacy trace callback mid
          ``._bootstrap()`` (inside
          :py:meth:`threading.Thread._bootstrap_inner`, before calling
          :py:meth:`threading.Thread.run`), thus interfering with
          profiling (when the "legacy" core is used).

        - To circumvent that, we use a wrapper to reversibly
          monkey-patch :py:meth:`threading.Thread.run`, so that profiler
          activation happens after the call to :py:func:`sys.settrace`
          and can thus "wrap" the callable.
    """
    if isinstance(vanilla_impl, MethodType):
        impl = wrap_thread_bootstrap(prof, vanilla_impl.__func__, enable_count)
        return MethodType(impl, vanilla_impl.__self__)

    @wraps(vanilla_impl)
    def wrapper(
        self: threading.Thread, *args: PS.args, **kwargs: PS.kwargs
    ) -> None:
        with Cleanup() as cleanup:
            run = make_syncing_wrapper(self.run, prof, enable_count)
            cleanup.patch(self, 'run', run)
            vanilla_impl(self, *args, **kwargs)

    return wrapper


def apply(cleanup: Cleanup, prof: LineProfiler) -> None:
    """
    Set up profiling in threads started by :py:mod:`threading` by
    applying patches to the module.

    Args:
        cleanup (Cleanup):
            :py:class:`Cleanup` instance managing the profiling session

        prof (LineProfiler):
            :py:class:`LineProfiler` instance used in the session

    Side effects:
        - :py:mod:`threading` marked as having been set up

        - The following methods and functions patched:

          - :py:meth:`threading.Thread.start`

        - Cleanup callbacks registered via ``cleanup.add_cleanup()``
    """
    if getattr(threading, _PATCHED_MARKER, False):
        return
    start_wrapper = wrap_thread_start(prof, threading.Thread.start)
    cleanup.patch(threading.Thread, 'start', start_wrapper)
    cleanup.patch(threading, _PATCHED_MARKER, True)
