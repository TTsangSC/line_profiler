from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, Generic, Protocol, TypeVar, cast
from typing_extensions import Concatenate, ParamSpec

from ...cleanup import _CALLBACK_REPR
from ..cache import LineProfilingCache
from ._infrastructure import SingleModulePatch


__all__ = ('Queue', 'PutWrapper', 'QuickGetWrapper', 'get_mp_pool_patch')

T = TypeVar('T')
PS = ParamSpec('PS')

_POOL_PATCHES: set[str] = set()
_UNTAGGED_RESULT_WARNING_TEMPLATE = (
    'received a pool-task result `{result}` without the expected tag {tag!r}; '
    'the worker process appears not to have been set up for profiling '
    '(e.g. its interpreter never loaded the profiling startup hook), '
    'so its profiling data will be missing from the output'
)


class Queue(Protocol):
    """
    Protocol for methods common to e.g. :py:class:`queue.SimpleQueue`
    and :py:class:`multiprocessing.queues.SimpleQueue`.
    """
    def put(self, obj: Any) -> None:
        ...

    def get(self) -> Any:
        ...


class PutWrapper:
    """
    Wrap around a queue (the ``outqueue`` argument to
    :py:func:`multiprocessing.pool.worker`) so that each call to its
    ``.put()`` is preceded by calling a ``callback()``; if
    ``tag`` is given, the object pushed to the parent is replaced with
    the triplet ``(tag, callback(), obj)``.
    """
    def __init__(
        self,
        queue: Queue,
        callback: Callable[[], Any],
        tag: str | None = None,
    ) -> None:
        self._queue = queue
        self._callback = callback
        self._tag = tag

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._queue, attr)

    def put(self, obj: Any) -> None:
        data = self._callback()
        if self._tag is not None:
            obj = self._tag, data, obj
        self._queue.put(obj)

    def get(self) -> Any:
        return self._queue.get()


class QuickGetWrapper(Generic[PS, T]):
    """
    Wrap around a :py:attr:`multiprocessing.pool.Pool._quick_get` to
    intercept and process data slipped into the queue by a
    :py:class:`PutWrapper`.

    - If the result of the get is :py:const:`None` (i.e. a sentinel
      value), it is a no-op.

    - If the result is a 3-tuple consisting of the ``tag``, some
      ``data``, followed by the original result, the ``data`` is
      processed with
      ``callback(cache: LineProfilingCache, data: T) -> Any`` and the
      original result is returned.

    - Otherwise, a warning (once per ``tag``) is emitted and the result
      is returned as-is.
    """
    def __init__(
        self,
        cache: LineProfilingCache,
        get: Callable[PS, tuple[Any, ...] | None],
        callback: Callable[[LineProfilingCache, T], Any],
        tag: str,
    ) -> None:
        self._impl = get
        self._callback = partial(callback, cache)
        self._warn = partial(self._warn_untagged_result_once, cache, tag)
        self._tag = tag

    def __call__(
        self, /, *args: PS.args, **kwargs: PS.kwargs
    ) -> tuple[Any, ...] | None:
        """
        Note:
            A worker which was never patched (its interpreter didn't run
            the profiling startup hook) pushes vanilla un-tagged
            results; those are passed through untouched, with a
            once-per-session warning, so that a mixed
            patched-parent/vanilla-worker setup degrades to missing
            profile data instead of killing the pool's result-handler
            thread (and thereby deadlocking every
            :py:meth:`multiprocessing.pool.AsyncResult.get`).
        """
        result = self._impl(*args, **kwargs)
        if result is None:
            return None
        if (
            isinstance(result, tuple)
            and len(result) == 3
            and result[0] == self._tag
        ):
            _, data, orig_result = result
            self._callback(cast(T, data))
            return orig_result
        self._warn(result)
        return result

    @staticmethod
    def _warn_untagged_result_once(
        cache: LineProfilingCache, tag: str, result: Any,
    ) -> None:
        key = 'mp_result_handler_untagged_result_warnings'
        # No lock: a race just means an extra warning, and this runs on
        # the pool's single result-handler thread anyway
        has_warned = cache._additional_data.setdefault(key, {})
        if has_warned.get(tag):
            return
        has_warned[tag] = True
        # Log before warning in case the warning is promoted to an error
        msg = _UNTAGGED_RESULT_WARNING_TEMPLATE.format(
            result=_CALLBACK_REPR(result), tag=tag,
        )
        cache._debug_output(msg, 'warning')
        warnings.warn(msg)


def get_mp_pool_patch(
    get_data: Callable[[], T],
    process_data: Callable[[LineProfilingCache, T], Any],
    tag: str,
) -> SingleModulePatch:
    """
    Create a patch for :py:mod:`multiprocessing.pool` which:

    - Patches :py:func:`multiprocessing.pool.worker` so that extra data
      are created in child/worker processes after running EACH task by
      ``get_data()``, and pushed back to  parent process alongside said
      task's result.

    - Patches :py:meth:`multiprocessing.pool.Pool._handle_results` so
      that said extra data is, where possible, retrieved from
      interprocess communication and processed by the parent's active
      :py:class:`.LineProfilingCache` instance.
    """
    if tag in _POOL_PATCHES:
        raise RuntimeError(f'tag {tag!r} already in use')
    _POOL_PATCHES.add(tag)

    wrap_outqueue = partial(PutWrapper, callback=get_data, tag=tag)
    wrap_quick_get = partial(QuickGetWrapper, callback=process_data, tag=tag)

    @LineProfilingCache._method_wrapper
    def wrap_handle_results(
        cache: LineProfilingCache,
        vanilla_impl: Callable[
            Concatenate[Queue, Callable[[], tuple[Any, ...] | None], PS],
            None
        ],
        outqueue: Queue,
        # Since we patched `outqueue.put()` in the child process, the
        # result pushed to the parent is (normally) a `(tag, data, obj)`
        # triplet
        get: Callable[[], tuple[Any, ...] | None],
        *args: PS.args,
        **kwargs: PS.kwargs
    ) -> None:
        """
        Wrap around :py:meth:`multiprocessing.pool.Pool._handle_results`
        so that it handles the extra info (result of calling
        ``get_data()`` in a child process after each task) included by
        ``wrap_worker()`` with ``process_data(cache, data)``.

        Note:
            :py:meth:`multiprocessing.pool.Pool._handle_results` is a
            static method.
        """
        vanilla_impl(outqueue, wrap_quick_get(cache, get), *args, **kwargs)

    @LineProfilingCache._method_wrapper  # nocover
    def wrap_worker(
        # We don't need the cache instance, but `@_method_wrapper` does
        _,
        vanilla_impl: Callable[Concatenate[Queue, Queue, PS], None],
        inqueue: Queue,
        outqueue: Queue,
        *args: PS.args,
        **kwargs: PS.kwargs
    ) -> None:
        """
        Wrap around :py:func:`multiprocessing.pool.worker` so that child
        processes attach the result of ``get_data()`` as they pass the
        task results back to the parent.

        Note:
            This is only called in child processes and thus we can't
            reliably measure coverage thereon, hence the ``# nocover``.
        """
        return vanilla_impl(inqueue, wrap_outqueue(outqueue), *args, **kwargs)

    patch = SingleModulePatch('pool')
    patch.add_method('', 'worker', wrap_worker)
    patch.add_method('Pool', '_handle_results', wrap_handle_results, 'static')
    return patch
