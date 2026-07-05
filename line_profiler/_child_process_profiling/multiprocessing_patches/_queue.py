from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol


__all__ = ('Queue', 'PutWrapper', 'PID_TAG')

#: First element of the result triplets pushed to the parent by patched
#: pool workers (see ``PutWrapper`` with ``push_to_parent=True``); lets
#: the parent-side result handler distinguish tagged results from
#: vanilla ones coming from workers which were never patched (e.g.
#: because their interpreter never loaded the profiling startup hook),
#: so that a mixed setup degrades to missing profile data instead of
#: crashing the pool's result-handler thread.
PID_TAG = '__line_profiler_pool_worker_pid__'


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
    ``push_to_parent`` is true, the object pushed to the parent is
    replaced with the triplet ``(PID_TAG, callback(), obj)``.
    """
    def __init__(
        self,
        queue: Queue,
        callback: Callable[[], Any],
        push_to_parent: bool = False,
    ) -> None:
        self._queue = queue
        self._callback = callback
        self._push = push_to_parent

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._queue, attr)

    def put(self, obj: Any) -> None:
        data = self._callback()
        if self._push:
            obj = PID_TAG, data, obj
        self._queue.put(obj)

    def get(self) -> Any:
        return self._queue.get()
