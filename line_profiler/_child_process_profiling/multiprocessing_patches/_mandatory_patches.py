from __future__ import annotations

import atexit
import os
import multiprocessing
from collections.abc import Callable
from functools import partial
from multiprocessing.process import BaseProcess
from types import MappingProxyType as mappingproxy, MethodType
from typing import Any, ClassVar, TypeVar, cast
from typing_extensions import Concatenate, ParamSpec

try:
    from multiprocessing import spawn
except ImportError:
    _CAN_USE_SPAWN = False
else:
    _CAN_USE_SPAWN = True
try:
    from multiprocessing import forkserver
except ImportError:
    _CAN_USE_FORKSERVER = False
else:
    _CAN_USE_FORKSERVER = (
        'forkserver' in multiprocessing.get_all_start_methods()
    )
try:
    from multiprocessing import resource_tracker
except ImportError:
    _CAN_USE_RESOURCE_TRACKER = False
else:
    _CAN_USE_RESOURCE_TRACKER = True

from .._patching_infrastructure import SingleModulePatch
from ..cache import LineProfilingCache
from ..runpy_patches import create_runpy_wrapper
from ._pool_patch_helpers import (
    get_per_task_callback_patch, get_worker_finalization_patch,
)


__all__ = (
    'POOL_WORKER_PID_PATCH', 'PROCESS_SETUP_PATCH',
    'RebootForkserverPatch', 'ResourceTrackerPatch', 'RunpyPatch',
    'wrap_bootstrap',
)

T = TypeVar('T')
P = TypeVar('P', bound=BaseProcess)
PS = ParamSpec('PS')

# ------------------------------ Helpers -------------------------------


def setup_mp_child(  # nocover
    cache: LineProfilingCache, proc: BaseProcess,
) -> None:
    """
    Perform :py:mod:`multiprocessing`-specific setup in a child process
    curated by the module. Currently it does the following:

    - Unregister the :py:mod:`atexit` hook associated with ``cache`` to
      avoid possible clashes with the profiling-file writing managed by
      this module.
    """
    if cache.main_pid == os.getpid():  # Not in a child process
        return
    xc: Exception | None = None
    msg = 'Performing setup for `multiprocessing` child processes...'
    cache._debug_output(msg)
    setup: Callable[[LineProfilingCache, BaseProcess], Any]
    for setup in [_unregister_atexit_hook]:
        try:
            setup(cache, proc)
        except Exception as e:
            xc = e
    if xc is None:
        msg = 'Setup for `multiprocessing` child process succeeded'
        cache._debug_output(msg)
    else:
        xc_str = type(xc).__name__
        if str(xc):
            xc_str = f'{xc_str}: {xc}'
        cache._debug_output(f'Setup failed: {xc_str}')
        raise xc


def _unregister_atexit_hook(  # nocover
    cache: LineProfilingCache, _,
) -> None:
    atexit.unregister(cache._atexit_hook)


# ----------- `multiprocessing.process.BaseProcess` patches ------------


@LineProfilingCache._method_wrapper  # nocover
def wrap_bootstrap(
    cache: LineProfilingCache,
    vanilla_impl: Callable[Concatenate[BaseProcess, PS], T],
    self: BaseProcess,
    /,
    *args: PS.args, **kwargs: PS.kwargs
) -> T:
    """
    Wrap around :py:meth:`.BaseProcess._bootstrap` to perform setups
    specific to :py:mod:`multiprocessing`-managed processes.
    """
    setup_mp_child(cache, self)
    return vanilla_impl(self, *args, **kwargs)


PROCESS_SETUP_PATCH = SingleModulePatch('multiprocessing.process', priority=1)
PROCESS_SETUP_PATCH.add_method('BaseProcess', '_bootstrap', wrap_bootstrap)

# ---------------------- PID bookkeeping patches -----------------------


def _get_worker_ntasks(cache: LineProfilingCache, worker: BaseProcess) -> int:
    """
    Check if the process has run any tasks; if not, report to the cache.

    Returns:
        Number of tasks run by ``worker``
    """
    pid: int | None = getattr(worker, 'pid', None)
    ntasks_finalized = _get_ntasks_finalized(cache)
    if pid is None:  # Dummy process
        return 0
    key = id(worker), pid
    try:
        return ntasks_finalized[key]
    except KeyError:
        pass
    ntasks = _get_ntasks(cache).pop(pid, 0)
    msg = 'Worker {0.name!r} (PID: {0.pid}) ran {1} task(s)'
    cache._debug_output(msg.format(worker, ntasks))
    if not ntasks:
        cache._warn_possible_lack_of_stats(pid)
    return ntasks_finalized.setdefault(key, ntasks)


def _increment_ntasks(cache: LineProfilingCache, pid: int) -> None:
    """
    Take and process the PID of the child process completing the task.
    """
    ntasks = _get_ntasks(cache)
    ntasks[pid] = ntasks.get(pid, 0) + 1


def _get_ntasks(cache: LineProfilingCache) -> dict[int, int]:
    key = 'mp_proc_ntasks'
    return cache._additional_data.setdefault(key, cast(dict[int, int], {}))


def _get_ntasks_finalized(
    cache: LineProfilingCache,
) -> dict[tuple[int, int], int]:
    key = 'mp_proc_ntasks_finalized'
    return cache._additional_data.setdefault(
        key, cast(dict[tuple[int, int], int], {})
    )


def _get_pid(_) -> int:
    return os.getpid()


POOL_WORKER_PID_PATCH = get_per_task_callback_patch(
    _get_pid, _increment_ntasks, '__line_profiler_pool_worker_pid__',
)
get_worker_finalization_patch(_get_worker_ntasks, POOL_WORKER_PID_PATCH)

# --------------------------- Misc. patches ----------------------------


class RebootForkserverPatch:
    """
    Reboot the process backing the global
    :py:class:`multiprocessing.forkserver.ForkServer` instance:

    - When the patch is applied, so as to ensure that child processes
      forked therefrom actually receives the active patches; and

    - When the session cache is cleaned up, so that child processes
      forked therefrom is no longer polluted by the patches.

    Note:
        This uses
        :py:meth:`multiprocessing.forkserver.ForkServer._stop()` which
        is private API, but it's the same hack used in Python's own test
        suite -- see the comment to said method.
    """
    summary: ClassVar[mappingproxy[str, frozenset[str]]] = mappingproxy({})
    priority: ClassVar[float | None] = None

    @classmethod
    def apply(cls, cache: LineProfilingCache, **_) -> None:
        if not _CAN_USE_FORKSERVER:
            return
        cls.reboot()
        cache.add_cleanup(cls.reboot)

    @staticmethod
    def reboot() -> None:
        fs_obj = forkserver._forkserver
        stop = getattr(fs_obj, '_stop', None)
        # Appease the type-checker since `._stop()` is not public API
        if not callable(stop):  # nocover
            msg = f'ForkServer._stop() (= {stop!r}) not callable'
            raise AssertionError(msg)  # Shouldn't happen
        stop()


class ResourceTrackerPatch:
    """
    Patch :py:mod:`multiprocessing.resource_tracker` so that
    :py:func:`multiprocessing.resource_tracker.ensure_running` and the
    eponymous method of
    :py:class:`multiprocessing.resource_tracker.ResourceTracker` report
    the resource-tracker server PIDs to the session cache.

    Note:
        The ``ResourceTracker`` server process is spawned when the first
        :py:mod:`multiprocessing` child process is created via the
        ``spawn`` or ``forkserver`` start methods. While this server
        process does not meaningfully contribute to the profiling result
        either way, since it can be created with profiling set up, its
        longevity means that :py:meth:`.LineProfilingCache.gather_stats`
        often catches empty .lprof files which it has occupied but not
        written to.

        To reduce noise while keeping the empty-file warning for other
        output files, we report the PIDs used by the server to the
        session cache so that they can be ignored if necessary.
    """
    if _CAN_USE_RESOURCE_TRACKER:
        summary: ClassVar[mappingproxy[str, frozenset[str]]] = mappingproxy({
            'multiprocessing.resource_tracker':
            frozenset({'ensure_running'}),
            'multiprocessing.resource_tracker.ResourceTracker':
            frozenset({'ensure_running'}),
        })
    else:
        summary = mappingproxy({})
    priority: ClassVar[float | None] = None

    @staticmethod
    @LineProfilingCache._method_wrapper
    def wrap_ensure_running(
        cache: LineProfilingCache,
        vanilla_impl: Callable[['resource_tracker.ResourceTracker'], None],
        self: 'resource_tracker.ResourceTracker',
    ) -> None:
        """
        Wrap around :py:meth:`multiprocessing.resource_tracker\
.ResourceTracker.ensure_running`
        so that the session cache can keep track of the PIDs used by the
        resource-tracer server.
        """
        maybe_pids: set[int | None] = {getattr(self, '_pid', None)}
        try:
            vanilla_impl(self)
        finally:
            maybe_pids.add(getattr(self, '_pid', None))
            pids = cast(set[int], maybe_pids - {None})
            if pids:
                cache._warn_possible_lack_of_stats(pids)

    @classmethod
    def apply(
        cls, cache: LineProfilingCache, *, cleanup: bool = True, **_,
    ) -> list[str]:
        if _CAN_USE_RESOURCE_TRACKER:
            patch = partial(cache.patch, cleanup=cleanup)
            # Patch the method on the class
            method = resource_tracker.ResourceTracker.ensure_running
            method = cls.wrap_ensure_running(method)
            patch(resource_tracker.ResourceTracker, 'ensure_running', method)
            # Patch the preexisting bound method on the module
            instance = resource_tracker._resource_tracker
            bound_method = MethodType(method, instance)
            patch(resource_tracker, 'ensure_running', bound_method)
        return list(cls.summary)


class RunpyPatch:
    """
    Patch the copy of :py:mod:`runpy` in the
    :py:mod:`multiprocessing.spawn` namespace so that subprocesses can
    perform rewrite-based profiling as with
    :py:func:`line_profiler.autoprofile.autoprofile.run`.

    See also:
        :py:mod:`line_profiler._child_process_profiling.runpy_patches`
    """
    summary: ClassVar[mappingproxy[str, frozenset[str]]]
    if _CAN_USE_SPAWN and hasattr(spawn, 'runpy'):
        summary = mappingproxy({'multiprocessing.spawn': frozenset({'runpy'})})
    else:
        summary = mappingproxy({})
    priority: ClassVar[float | None] = None

    @classmethod
    def apply(
        cls, cache: LineProfilingCache, *, cleanup: bool = True, **_,
    ) -> list[str]:
        if cls.summary:
            patch = partial(cache.patch, cleanup=cleanup)
            patch(spawn, 'runpy', create_runpy_wrapper(cache))
        return list(cls.summary)
