from __future__ import annotations

import operator
import os
from collections.abc import (
    Callable, Collection, Generator, Mapping, MutableMapping, MutableSequence,
)
from dataclasses import dataclass
from functools import partial
from inspect import getattr_static
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, final, overload,
)
from typing_extensions import Concatenate, ParamSpec, Self

from line_profiler import _diagnostics as diagnostics
from line_profiler.line_profiler_utils import CallbackRepr, make_tempfile


__all__ = ('Transaction', 'Ledger')

_CALLBACK_REPR_HELPER = CallbackRepr(maxother=cast(int, float('inf')))
_CALLBACK_REPR = _CALLBACK_REPR_HELPER.repr

_Stacks = dict[float, list['Transaction']]
_StackContexts = list[_Stacks]
LogLevel = Literal['debug', 'info', 'warning', 'error', 'critical']

E = TypeVar('E')
E1 = TypeVar('E1')
R = TypeVar('R')
R1 = TypeVar('R1')
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
PS = ParamSpec('PS')


@dataclass
class _ExecState(Generic[E, R]):
    execute_result: E | None = None
    rollback_result: R | None = None
    executed: bool = False

    def handle_execute(self, result: E | None) -> None:
        if not self.executed:
            # New execution, clear the cached values
            self.execute_result, self.rollback_result = result, None
        self.executed = True

    def handle_rollback(self, result: R | None) -> None:
        if self.executed:
            # Note: we keep the last value handed to `.handle_execute()`
            # as a cache
            self.rollback_result = result
        self.executed = False


@final
class Transaction(Generic[E, R]):
    """
    Object representing a single transaction: a pair of setup and
    teardown callables.

    Init attributes:
        name:
            Optional name of the transaction to use in log messages.

        execute_callback:
            Optional setup/forward-direction callable, to be called
            without args when calling :py:meth:`.execute`.

        rollback_callback:
            Optional teardown/backward-direction callable, to be called
            when calling :py:meth:`.roll_back`.
            If :py:attr:`.rollback_uses_exec_result` is true, it should
            be a callable taking the result of
            :py:attr:`.execute_callback` as its only argument; else, it
            should allow being called without args.

        rollback_uses_exec_result
            Whether :py:attr:`.rollback_callback` should be called
            arg-less (false) or with the result of calling
            :py:attr:`.execute_callback` (true).

    Other attributes:
        can_execute:
            If false, :py:meth:`.execute` is a no-op.

        can_roll_back:
            If false, :py:meth:`.roll_back` is a no-op.

        last_execute_result
            The result of the last call to :py:meth:`.execute` which
            wasn't a no-op.

        last_rollback_result
            The result of the last call to :py:meth:`.roll_back` which
            wasn't a no-op.

        executed
            Whether the transaction has been :py:meth:`.execute`-d and
            has not yet :py:meth:`.roll_back`-ed.

    Select methods:
        execute():
            Run :py:attr:`.execute_callback` if it hasn't already been
            :py:attr:`.executed`.

        roll_back():
            Run :py:attr:`.rollback_callback` if it has already been
            :py:attr:`.executed`.

        with_name():
        with_execute():
        with_simple_rollback():
        with_parametrized_rollback():
        with_method_caller_rollback():
            Use the builder pattern to construct an instance.

    Notes:
        Prefer using the builder-pattern methods, especially the
        ``.with_*_rollback()`` ones, instead of going through the
        initializer.
    """
    @overload
    def __init__(
        self,
        name: str | None = None,
        *,
        execute_callback: Callable[[], E] | None = None,
        rollback_callback: None = None,
        rollback_uses_exec_result: bool | None = None,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        name: str | None = None,
        *,
        execute_callback: Callable[[], E] | None = None,
        rollback_callback: Callable[[], R],
        rollback_uses_exec_result: Literal[False],
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        name: str | None = None,
        *,
        execute_callback: Callable[[], E] | None = None,
        rollback_callback: Callable[[E], R],
        rollback_uses_exec_result: Literal[True],
    ) -> None:
        ...

    def __init__(
        self,
        name: str | None = None,
        *,
        execute_callback: Callable[[], E] | None = None,
        rollback_callback: Callable[[], R] | Callable[[E], R] | None = None,
        rollback_uses_exec_result: bool | None = None,
    ) -> None:
        self._name = name
        self._ex = execute_callback
        # Note: typing works better this way instead of if we stored
        # `.rollback_callback` and `.rollback_uses_exec_result`
        # separately
        self._rb: (
            tuple[Callable[[], R], Literal[False]]
            | tuple[Callable[[E], R], Literal[True]]
            | None
        )
        if rollback_callback is None:
            self._rb = None
        else:
            self._rb = cast(
                Any, (rollback_callback, bool(rollback_uses_exec_result)),
            )
        self._last_ex = _ExecState[E, R]()

    def __repr__(self) -> str:
        if self.name is None:
            desc_fields: dict[str, str] = {}
            if self.execute_callback is not None:
                desc_fields['ex'] = _CALLBACK_REPR(self.execute_callback)
            if self.rollback_callback is not None:
                desc_fields['rb'] = _CALLBACK_REPR(self.rollback_callback)
            if desc_fields:
                desc = ', '.join(
                    f'{abbr}={value}' for abbr, value in desc_fields.items()
                )
            else:
                desc = '(no-op)'
        else:
            desc = repr(self.name)
        return f'<Transaction @ {id(self):#x}: {desc}>'

    def execute(self) -> tuple[Literal[True], E] | tuple[Literal[False], None]:
        """
        Call :py:attr:`.execute_callback` if it hasn't been successfully
        called already.  Provided that the first call is successful,
        this method is idempotent; subsequent calls are "blocked" until
        the transaction is :py:meth:`.roll_back`-ed.

        Return:
            ``(has_executed_callback, return_value_of_callback)``;
            note that ``has_executed_callback`` is always false if
            :py:attr:`.execute_callback` is :py:const:`None`.

        Example:
            >>> log: list[str] = []
            >>> tr = Transaction().with_execute(log.append, 'hello')

            >>> tr.execute()
            (True, None)
            >>> log
            ['hello']

            Subsequent calls are no-op...:

            >>> tr.execute()
            (False, None)
            >>> log
            ['hello']

            >>> tr.execute()
            (False, None)
            >>> log
            ['hello']

            ... until we've rolled the transaction back:

            >>> tr.roll_back()
            (False, None)
            >>> tr.execute()
            (True, None)
            >>> log
            ['hello', 'hello']
        """
        result: E | None
        if self.can_execute:
            if TYPE_CHECKING:
                assert self.execute_callback is not None
            run, result = True, self.execute_callback()
        else:
            run, result = False, None
        self._last_ex.handle_execute(result)
        return cast(Any, (run, result))

    def roll_back(
        self,
    ) -> tuple[Literal[True], R] | tuple[Literal[False], None]:
        """
        Call :py:attr:`.rollback_callback` if it hasn't been
        successfully called already.  Provided that the first call is
        successful, this method is idempotent; subsequent calls are
        "blocked" until the transaction is re-:py:meth:`.execute`-d.

        Return:
            ``(has_executed_callback, return_value_of_callback)``;
            note that ``has_executed_callback`` is always false if
            :py:attr:`.rollback_callback` is :py:const:`None`.

        Example:
            >>> log: list[str] = []
            >>> tr = Transaction().with_simple_rollback(
            ...     log.append, 'hello',
            ... )

            :py:meth:`.roll_back` is a no-op if we haven't
            :py:meth:`.execute`-d:

            >>> tr.roll_back()
            (False, None)
            >>> log
            []
            >>> tr.execute()
            (False, None)
            >>> tr.roll_back()
            (True, None)
            >>> log
            ['hello']

            Subsequent calls are also no-op...:

            >>> tr.roll_back()
            (False, None)
            >>> log
            ['hello']

            >>> tr.roll_back()
            (False, None)
            >>> log
            ['hello']

            ... until we've executed the transaction again:

            >>> tr.execute()
            (False, None)
            >>> tr.roll_back()
            (True, None)
            >>> log
            ['hello', 'hello']
        """
        result: R | None
        if self.can_roll_back:
            if TYPE_CHECKING:
                assert self._rb is not None
            run = True
            if self._rb[1]:  # Using exec's result
                exec_result = cast(E, self.last_execute_result)
                # See `ty` issue #4674 for why we need the `cast()`s
                result = cast(Callable[[E], R], self._rb[0])(exec_result)
            else:  # Arg-less
                result = cast(Callable[[], R], self._rb[0])()
        else:
            run, result = False, None
        self._last_ex.handle_rollback(result)
        return cast(Any, (run, result))

    @overload
    def copy(self) -> Transaction[E, R]:
        ...

    @overload
    def copy(self, **replacements) -> Transaction[Any, Any]:
        ...

    def copy(self, **replacements) -> Transaction[Any, Any]:
        """
        Create a fresh copy: the internal states on e.g. whether the
        :py:class:`Transaction` has been executed and the results of the
        last calls to :py:meth:`.execute` and :py:meth:`.roll_back` are
        reset on the copy.
        """
        kwargs: dict[str, Any] = {}
        for name in {
            'name', 'execute_callback', 'rollback_callback',
            'rollback_uses_exec_result',
        }:
            kwargs[name] = replacements.get(name, getattr(self, name))
        return Transaction(**kwargs)

    def with_name(self, name: str) -> Transaction[E, R]:
        """
        Create a fresh copy with :py:attr:`.name`.

        Return:
            New instance.

        Example:
            >>> tr = Transaction().with_name('foo')
            >>> tr  # doctest: +ELLIPSIS
            <Transaction @ 0x...: 'foo'>
        """
        return self.copy(name=name)

    @overload
    def with_execute(
        self, func: Callable[PS, E1], /, *args: PS.args, **kwargs: PS.kwargs
    ) -> Transaction[E1, R]:
        ...

    @overload
    def with_execute(self, func: None, /) -> Transaction[None, R]:
        ...

    def with_execute(
        self,
        func: Callable[PS, E1] | None,
        /,
        *args: PS.args,
        **kwargs: PS.kwargs
    ) -> Transaction[E1, R] | Transaction[None, R]:
        """
        Create a fresh copy with an updated
        :py:attr:`.execute_callback`.

        Args:
            func()
                Base callable; if :py:const:`None`, set
                :py:attr:`.execute_callback` to :py:const:`None`
                instead.
            *args, **kwargs
                To be passed to ``func()``.

        Return:
            New instance.

        Example:
            >>> tr = (
            ...     Transaction()
            ...     .with_execute(print, 1, 2, 3)
            ...     .with_simple_rollback(print, 4, 5, 6)
            ... )
            >>> tr.execute()
            1 2 3
            (True, None)
            >>> tr.roll_back()
            4 5 6
            (True, None)

            No callback:

            >>> tr = tr.with_execute(None)
            >>> tr.execute()
            (False, None)
            >>> tr.roll_back()
            4 5 6
            (True, None)
        """
        if func is None:
            return self.copy(execute_callback=None)
        callback = self._consolidate_callback(func, *args, **kwargs)
        return self.copy(execute_callback=callback)

    @overload
    def with_simple_rollback(
        self, func: Callable[PS, R1], /, *args: PS.args, **kwargs: PS.kwargs
    ) -> Transaction[E, R1]:
        ...

    @overload
    def with_simple_rollback(self, func: None, /) -> Transaction[E, None]:
        ...

    def with_simple_rollback(
        self,
        func: Callable[PS, R1] | None,
        /,
        *args: PS.args,
        **kwargs: PS.kwargs
    ) -> Transaction[E, R1] | Transaction[E, None]:
        """
        Create a fresh copy with an updated
        :py:attr:`.rollback_callback`.

        Args:
            func()
                Base callable; it should NOT depend on the return value
                of :py:attr:`.execute_callback`; if :py:const:`None`,
                set :py:attr:`.rollback_callback` to :py:const:`None`
                instead.
            *args, **kwargs
                To be passed to ``func()``.

        Return:
            New instance.

        Example:
            >>> tr = (
            ...     Transaction()
            ...     .with_execute(print, 1, 2, 3)
            ...     .with_simple_rollback(print, 4, 5, 6)
            ... )
            >>> tr.execute()
            1 2 3
            (True, None)
            >>> tr.roll_back()
            4 5 6
            (True, None)

            No callback:

            >>> tr = tr.with_simple_rollback(None)
            >>> tr.execute()
            1 2 3
            (True, None)
            >>> tr.roll_back()
            (False, None)
        """
        if func is None:
            return self.copy(rollback_callback=None)
        callback = self._consolidate_callback(func, *args, **kwargs)
        return self.copy(
            rollback_callback=callback, rollback_uses_exec_result=False,
        )

    def with_parametrized_rollback(
        self, func: Callable[[E1], R1],
    ) -> Transaction[E1, R1]:
        """
        Create a fresh copy with an updated
        :py:attr:`.rollback_callback`.

        Args:
            func()
                Base callable; it should take exactly ONE argument: the
                return value of :py:attr:`.execute_callback`.

        Return:
            New instance.

        Example:
            >>> from functools import partial

            >>> tr = (
            ...     Transaction()
            ...     .with_execute(
            ...         lambda a: (print('Executed:', a), a)[-1],
            ...         'foo bar',
            ...     )
            ...     .with_parametrized_rollback(
            ...         partial(print, 'Rolled back:')
            ...     )
            ... )
            >>> tr.execute()
            Executed: foo bar
            (True, 'foo bar')
            >>> tr.roll_back()
            Rolled back: foo bar
            (True, None)
        """
        # Note: this is for when the rollback just takes the return
        # value of `.execute_callback`
        return self.copy(
            rollback_callback=func, rollback_uses_exec_result=True,
        )

    def with_method_caller_rollback(
        self, method: str, /, *args, **kwargs
    ) -> Transaction[E, Any]:
        """
        Create a fresh copy with an updated
        :py:attr:`.rollback_callback`.

        Args:
            method, *arg, **kwargs
                Passed to :py:func:`operator.methodcaller` to construct
                the rollback callback: at roll-back, call
                ``getattr(e, method)(*args, **kwargs)``, where ``e`` is
                the return value of :py:func:`.execute_callback`.

        Return:
            New instance.

        Example:
            >>> from functools import partial

            >>> tr = (
            ...     Transaction()
            ...     .with_execute(hex, 108)
            ...     .with_method_caller_rollback('upper')
            ... )
            >>> tr.execute()
            (True, '0x6c')
            >>> tr.roll_back()
            (True, '0X6C')
            >>> print(tr.last_execute_result, tr.last_rollback_result)
            0x6c 0X6C
        """
        # Note: this is for when the rollback is just a method call on
        # the return value of `.execute_callback`
        callback = cast(
            Callable[[E], Any],
            operator.methodcaller(method, *args, **kwargs),
        )
        return self.with_parametrized_rollback(callback)

    @staticmethod
    def _consolidate_callback(
        func: Callable[PS, T], /, *args: PS.args, **kwargs: PS.kwargs
    ) -> Callable[[], T]:
        if args or kwargs:
            return partial(func, *args, **kwargs)
        else:
            return cast(Callable[[], T], func)

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def execute_callback(self) -> Callable[[], E] | None:
        return self._ex

    @property
    def rollback_callback(self) -> Callable[[], R] | Callable[[E], R] | None:
        if self._rb is None:
            return None
        return self._rb[0]

    @property
    def rollback_uses_exec_result(self) -> bool | None:
        if self._rb is None:
            return None
        return self._rb[1]

    @property
    def can_execute(self) -> bool:
        return not self._last_ex.executed and self.execute_callback is not None

    @property
    def can_roll_back(self) -> bool:
        return self._last_ex.executed and self.rollback_callback is not None

    @property
    def last_execute_result(self) -> E | None:
        return self._last_ex.execute_result

    @property
    def last_rollback_result(self) -> R | None:
        return self._last_ex.rollback_result

    @property
    def executed(self) -> bool:
        return self._last_ex.executed


class Ledger:
    """
    Object managing stacks of :py:class:`Transaction` objects; can be
    used as a context manager which automatically:

    - Creates a set ("context") of prioritized stacks of
      :py:class:`Transaction` objects and push it to the internal stack
      of contexts on context-manager entry, and

    - Pops it from said internal stack and rolls back all its
      :py:class:`Transaction`s on context-manager exit.
    """
    # --------------------- Basic functionalities ----------------------

    def __init__(self, *_, name: str | None = None, **__) -> None:
        self.name = name
        self._contexts: _StackContexts = []

    def __repr__(self) -> str:
        base = f'{type(self).__name__} @ {id(self):#x}'
        if self.name is not None:
            base = f'{base}: {self.name!r}'
        return f'<{base}>'

    def __enter__(self) -> Self:
        """
        Returns:
            The instance

        Note:
            This context manager is reentrant; entering creates a new
            "context" of prioritized :py:class:`Transaction` stacks and
            pushes it to the internal stack, which is then popped and
            rolled back on :py:meth:`.__exit__`.

        Example:
            >>> from typing import ClassVar

            >>> strings = []
            >>> add = strings.append
            >>> with Ledger() as ledger:
            ...     ledger.on_rollback(add, 'one')
            ...     # Increased priority
            ...     ledger.on_rollback_with_priority(1, add, 'two')
            ...     add('three')
            ...     with ledger:
            ...         # Decreased priority
            ...         ledger.on_rollback_with_priority(
            ...             -1, add, 'four',
            ...         )
            ...         ledger.on_rollback(add, 'five')
            ...         add('six')
            ...     add('seven')
            ...     # Increased priority
            ...     ledger.on_rollback_with_priority(1, add, 'eight')
            ...
            >>> strings  # doctest: +NORMALIZE_WHITESPACE
            ['three', 'six', 'five', 'four', 'seven', 'eight', 'two',
             'one']
        """
        self._contexts.append({})
        return self

    def __exit__(self, *_, **__) -> Any:
        """
        Pop a "context" from the internal stack that was previously
        :py:meth:`.__enter__`-ed into, and
        :py:meth:`Transaction.roll_back` all its
        :py:class:Transaction`s.
        """
        self.roll_back_transactions(1, reason='context exit')

    def roll_back_transactions(
        self, levels: int | None = None, *, reason: str | None = None,
    ) -> None:
        """
        Roll back the :py:meth:`Transaction.execute`-d
        :py:class:`Transaction`s.

        Args:
            levels:
                Roll back (at most) this many stack levels of "contexts"
                of prioritized :py:class:`Transaction` stacks; default
                is to roll back ALL "contexts" on the internal stack.

            reason:
                Optional reason for the roll-back in log messages.
        """
        if levels is None:
            pop_contexts = self._pop_all_contexts(self._contexts)
        else:
            pop_contexts = self._pop_n_contexts(self._contexts, levels)
        for ctx in pop_contexts:
            self._roll_back_ctx(ctx, reason)

    # ------------------- `Transaction` manipulation -------------------

    def on_rollback(
        self, func: Callable[PS, T], /, *args: PS.args, **kwargs: PS.kwargs
    ) -> None:
        """
        Create a dummy :py:class:`Transaction` so that
        ``func(*args, **kwargs)`` is executed at
        :py:class:`.roll_back_transactions`-time with default priority
        (0).

        Args:
            func(), *args, **kwargs
                See :py:meth:`.on_rollback_with_priority`.

        Example:
            See :py:meth:`.__enter__()`.
        """
        self.on_rollback_with_priority(0, func, *args, **kwargs)

    def on_rollback_with_priority(
        self,
        priority: float,
        func: Callable[PS, T],
        /,
        *args: PS.args,
        **kwargs: PS.kwargs
    ) -> None:
        """
        Create a dummy :py:class:`Transaction` so that
        ``func(*args, **kwargs)`` is executed at
        :py:class:`.roll_back_transactions`-time with the specified
        rollback priority.

        Args:
            priority
                Numeric value for prioritizing the
                :py:class:`Transaction`'s roll-back (see
                ``rollback_priority`` in :py:meth:`.add_transactions`).

            func(), *args, **kwargs
                Callable and its arguments to construct the callback
                from; equivalent to passing
                ``partial(func, *args, **kwargs)``.

        Example:
            See :py:meth:`.__enter__()`.
        """
        transaction: Transaction[Any, T]
        transaction = Transaction().with_simple_rollback(func, *args, **kwargs)
        self.add_transactions(transaction, rollback_priority=priority)

    def add_transactions(
        self,
        trxns: Transaction[Any, Any] | Collection[Transaction[Any, Any]],
        *,
        execute: bool = True,
        rollback_priority: float = 0,
    ) -> None:
        """
        Explicitly add one or more :py:class:`Transaction`s to the
        current "context" of prioritized :py:class:`Transaction` stacks.

        Args:
            trxns:
                A single :py:class:`Transaction` or a collection
                thereof; if a collection, they are executed "atomically"
                (see :py:meth:`.create_atomic_transaction`).
            execute:
                Whether to immediately call
                :py:meth:`Transaction.execute` (default
                :py:const:`True`); if that fails, all
                :py:class:`Transaction`s in the current "context" will
                be rolled back.
            rollback_priority:
                Numeric value (default 0) for prioritizing ``trxns``'s
                roll-back; the more positive the value, the earlier
                their rollback callbacks will be called upon
                context-manager exit or an explicit
                :py:meth:`.roll_back_transactions` call.
                :py:class:`Transaction`s of the same
                ``rollback_priority`` are rolled back in LIFO (i.e.
                stack) order.

        Notes:
            Prefer using these higher-level convenience methods where
            possible, instead of explicitly constructing
            :py:class:`Transaction`s:

            - :py:meth:`.on_rollback`

            - :py:meth:`.on_rollback_with_priority`

            - :py:meth:`.update_mapping`

            - :py:meth:`.make_tempfile`

            - :py:meth:`.patch`

            - :py:meth:`.call_atomically`
        """
        if not isinstance(trxns, Transaction):
            trxns = self.create_atomic_transaction(trxns)
        try:
            ctx = self._current_context
            ctx.setdefault(rollback_priority, []).append(trxns)
            self._debug_output(
                f'Registered {trxns!r} '
                '(rollback priority: {rollback_priority})',
            )
            if execute:
                self._debug_output(f'Executing {trxns!r}...')
                state = 'success'
                try:
                    run, result = trxns.execute()
                except BaseException as e:
                    state = 'failure'
                    result_repr = self._format_exception(e)
                    raise
                else:
                    result_repr = repr(result) if run else '(no-op)'
                finally:
                    msg = f'Executed {trxns!r} ({state}): {result_repr}'
                    self._debug_output(msg)
        except BaseException:
            # If anything went sideways, immediately roll the current
            # context back
            self.roll_back_transactions(
                1, reason=f'transaction {trxns!r} failed to execute',
            )
            raise

    def update_mapping(
        self,
        mapping: MutableMapping[K, V],
        updates: Mapping[K, V],
        *,
        roll_back: bool = True,
        **kwargs
    ) -> None:
        """
        Transactionally update a mapping with another.

        Args:
            mapping:
                Mapping to be updated.
            updates:
                Mapping containing the updates.
            roll_back:
                Whether to reverse the changes on roll-back.
            **kwargs:
                Passed to :py:meth:`.add_transactions`.

        Example:
            >>> d1 = {1: 2, 3: 4}
            >>> d2 = d1.copy()
            >>> updates = {0: -1, 3: 5}
            >>> with Ledger() as ledger:
            ...     ledger.update_mapping(d1, updates)
            ...     for key, value in updates.items():
            ...         assert d1[key] == value
            ...
            >>> assert d1 == d2

            Without roll-backs:

            >>> with Ledger() as ledger:
            ...     ledger.update_mapping(d1, updates, roll_back=False)
            ...     for key, value in updates.items():
            ...         assert d1[key] == value
            ...
            >>> assert d1 != d2
            >>> for key, value in d1.items():
            ...     if key not in d2 or d2[key] != value:
            ...         assert updates[key] == value

            Atomicity:

            >>> from collections.abc import Callable, MutableMapping
            >>> from functools import partialmethod
            >>> from operator import getitem, setitem, delitem
            >>> from typing import ClassVar, Generic, TypeVar
            >>> from typing_extensions import (
            ...     Concatenate, ParamSpec, Self,
            ... )

            >>> PS = ParamSpec('PS')
            >>> T = ParamSpec('T')
            >>> K = ParamSpec('K')
            >>> V = ParamSpec('V')

            >>> class BuggedMapping(MutableMapping, Generic[K, V]):
            ...     _forbidden_keys: ClassVar[set[str]]
            ...     _forbidden_keys = {'spam', 'ham'}
            ...
            ...     def __init__(self, *args, **kwargs) -> None:
            ...         self._dict: dict[K, V] = dict(*args, **kwargs)
            ...
            ...     def __repr__(self) -> str:
            ...         return (
            ...             f'<{type(self).__name__} '
            ...             f'@ {id(self):#x}: {self._dict!r}>'
            ...         )
            ...
            ...     def copy(self) -> Self:
            ...         return type(self)(self)
            ...
            ...     def _defer(
            ...         self,
            ...         func: Callable[Concatenate[dict[K, V], PS], T],
            ...         /,
            ...         *args: PS.args,
            ...         **kwargs: PS.kwargs,
            ...     ) -> T:
            ...         return func(self._dict, *args, **kwargs)
            ...
            ...     def __setitem__(self, key: K, value: V, /) -> None:
            ...         if key in self._forbidden_keys:
            ...             raise KeyError(key)
            ...         self._defer(setitem, key, value)
            ...
            ...     __getitem__ = partialmethod(_defer, getitem)
            ...     __delitem__ = partialmethod(_defer, delitem)
            ...     __iter__ = partialmethod(_defer, iter)
            ...     __len__ = partialmethod(_defer, len)

            >>> d1 = BuggedMapping[str, int](foo=1, bar=2)
            >>> d2 = d1.copy()
            >>> with Ledger() as ledger:
            ...     try:
            ...         ledger.update_mapping(
            ...             d1,
            ...             {'bar': 3, 'baz': 4, 'spam': 0, 'eggs': 5},
            ...         )
            ...     except KeyError:
            ...         assert d1 == d2
            ...     else:
            ...         assert False, f'KeyError not raised: {d1=!r}'
        """
        transactions: list[Transaction[Any, Any]] = []
        for key, value in updates.items():
            trxn = Transaction[Any, Any]()
            if key in mapping:
                # Note: don't `.__getitem__()`, avoid triggering
                # `DefaultDict.__missing__()`
                old = mapping[key]
                change = f'{old!r} -> {value!r}'
                trxn = trxn.with_simple_rollback(
                    operator.setitem, mapping, key, old,
                )
            else:
                change = f'{value!r} (new)'
                trxn = trxn.with_simple_rollback(mapping.pop, key, None)
            msg = self._format_mapping_update_trxn(mapping, key, change)
            transactions.append(
                trxn
                .with_name(msg)
                .with_execute(operator.setitem, mapping, key, value)
            )
        trxn = self.create_atomic_transaction(transactions)
        if not roll_back:  # Disable roll-back
            trxn = trxn.with_simple_rollback(None)
        self.add_transactions(trxn, **kwargs)

    def make_tempfile(
        self,
        *,
        roll_back: bool = True,
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | os.PathLike[str] | None = None,
        **kwargs
    ) -> Path:
        """
        Transactionally create a fresh tempfile with
        :py:func:`make_tempfile`.

        Args:
            roll_back:
                Whether to remove the file on roll-back.
            suffix, prefix, dir:
                Passed to :py:func:`make_tempfile`.
            **kwargs:
                Passed to :py:meth:`.add_transactions`.

        Returns:
            path (Path):
                Path to the created file.

        Example:
            >>> prefix, suffix = 'my_file_', '.txt'
            >>> with Ledger() as ledger:
            ...     path = ledger.make_tempfile(
            ...         prefix=prefix, suffix=suffix,
            ...     )
            ...     assert path.exists()
            ...     assert path.name.startswith(prefix)
            ...     assert path.name.endswith(suffix)
            ...
            >>> assert not path.exists()

            Without roll-back:

            >>> from pathlib import Path

            >>> try:
            ...     path: Path | None = None
            ...     with Ledger() as ledger:
            ...         path = ledger.make_tempfile(
            ...             prefix=prefix,
            ...             suffix=suffix,
            ...             roll_back=False,
            ...         )
            ...         assert path.exists()
            ...         assert path.name.startswith(prefix)
            ...         assert path.name.endswith(suffix)
            ...     assert path.exists()
            ... finally:
            ...     if path is not None:
            ...         path.unlink()

        Notes:
            Since this method has to return the path to the created
            tempfile, the corresponding transaction is ALWAYS
            :py:meth:`Transaction.execute`-d immediately.
        """
        filename = '????'
        if suffix:
            filename = f'{filename}{suffix}'
        if prefix:
            filename = f'{prefix}{filename}'
        if dir:
            filename = os.path.join(dir, filename)

        trxn = (
            Transaction[Any, Any](f'Create tempfile {filename}')
            .with_execute(
                self._make_tempfile, suffix=suffix, prefix=prefix, dir=dir,
            )
        )
        if roll_back:
            trxn = trxn.with_method_caller_rollback('unlink', missing_ok=True)
        self.add_transactions(trxn, **{**kwargs, 'execute': True})
        path = trxn.last_execute_result
        if path is None:  # nocover
            raise AssertionError(
                f'{trxn=!r}: '
                'expected `.last_execute_result` to be a `Path`, got `None`',
            )
        return path

    def patch(
        self,
        obj: Any,
        attr: str,
        value: Any,
        *,
        name: str | None = None,
        roll_back: bool = True,
        **kwargs
    ) -> None:
        """
        Transactionally patch an attribute on an object.

        Args:
            obj:
                Object to be patched.
            attr:
                Name of the attribute.
            value:
                Value to be assigned to said attribute of ``obj``.
            name:
                Optional name for ``obj`` to be used in debug messages.
            roll_back:
                Whether to reverse the patch (by resetting or deleting
                the attribute) on rollback.
            **kwargs:
                Passed to :py:meth:`.add_transactions`.

        Example:
            >>> class Object:
            ...     pass  # Allow setting arbitrary attributes
            ...
            >>>
            >>> obj = Object()
            >>> obj.foo = 1
            >>> with Ledger() as ledger:
            ...     ledger.patch(obj, 'foo', 2)
            ...     ledger.patch(obj, 'bar', 3)
            ...     ledger.patch(obj, 'baz', 4, roll_back=False)
            ...     assert obj.foo == 2
            ...     assert obj.bar == 3
            ...     assert obj.baz == 4  # Not restored
            ...
            >>> assert obj.foo == 1
            >>> assert not hasattr(obj, 'bar')
            >>> assert obj.baz == 4
        """
        if name is None:
            name = self._get_name(obj)
        name = 'Patch `{}.{}` -> `{}`'.format(name, attr, value)
        trxn = (
            Transaction[Any, Any](name)
            .with_execute(setattr, obj, attr, value)
        )

        _, should_be_restored, old_value = self._get_attribute_info(obj, attr)
        if roll_back:
            if should_be_restored:
                trxn = trxn.with_simple_rollback(setattr, obj, attr, old_value)
            else:
                trxn = trxn.with_simple_rollback(delattr, obj, attr)

        self.add_transactions(trxn, **kwargs)

    # --------------------- "Atomic" transactions ----------------------

    def call_atomically(
        self,
        func: Callable[Concatenate[Self, PS], T],
        /,
        *args: PS.args,
        **kwargs: PS.kwargs
    ) -> T:
        """
        Wrap around ``func()`` which uses the instance to effect
        transactional-ized changes, and implement them in an "atomic"
        way.  This is similar to

        >>> with self:  # doctest: +SKIP
        ...     func(self, *args, **kwargs)

        except that the transactions are not (immediately) rolled back
        on successful exit (see Side effects).

        Args:
            func():
                Callable which takes the :py:class:`Ledger` instance
                (and other arguments), using its methods (e.g.
                :py:meth:`make_tempfile`) to implement transactional
                changes which are all-or-nothing.
            *args, **kwargs:
                Passed to ``func()``.

        Return:
            Result of ``func(self, *args, **kwargs)``

        Side effects:
            - A temporary "dummy" context is pushed to the internal
              stack of "contexts" for the time being of the call; be
              careful esp. if ``func()`` explicitly use
              :py:meth:`.roll_back_transactions` with the ``levels``
              argument.

            - If any (base) exception occurred when calling ``func()``,
              all the :py:class:`Transaction`s made therein via the
              instance will be rolled back IMMEDIATELY.
              will be rolled back IMMEDIATELY.

            - Otherwise, a :py:class:`Transaction` will be added to the
              current "context" at default priority, so that all the
              :py:class:`Transaction`s made in the call via the instance
              will be rolled back WHEN SAID :py:class:`Transaction` IS
              ITSELF ROLLED BACK, e.g. via context-manager exit or
              explicit :py:meth:`.roll_back_transactions`.

        Example:
            >>> def func(
            ...     ledger: Ledger,
            ...     dct: dict[str, int],
            ...     success: bool = True,
            ... ) -> str:
            ...     ledger.update_mapping(dct, {'spam': 1})
            ...     if not success:
            ...         raise RuntimeError('error')
            ...     ledger.update_mapping(dct, {'ham': 2})
            ...     return 'ok'

            Successful call:

            >>> d = {}
            >>> with Ledger() as ledger:
            ...     assert (
            ...         (res := ledger.call_atomically(func, d)) == 'ok'
            ...     ), f'{res=!r}'
            ...     assert (  # All changes preserved at this point
            ...         d == {'spam': 1, 'ham': 2}
            ...     ), f'{d=!r}'
            ...
            >>> assert not d, f'{d=!r}'  # All changes rolled back

            Failing call:

            >>> d = {}
            >>> Ledger().call_atomically(func, d, False)
            Traceback (most recent call last):
              ...
            RuntimeError: error
            >>> assert not d, f'{d=!r}'  # Changes bf. error rolled back

            Note that errors don't "leak": it can be contained by an
            appropriate try-except clause, without causing the current
            "context" to be rolled back:

            >>> with Ledger() as ledger:
            ...     ledger.on_rollback(print, 'rollback')
            ...     d = {}
            ...     try:
            ...         ledger.call_atomically(func, d, False)
            ...     except RuntimeError:
            ...         print('made failed call')
            ...         pass
            ...     else:
            ...         assert False, 'did not raise RuntimeError'
            ...     print('post-failed call')
            ...
            made failed call
            post-failed call
            rollback
        """
        call = '{}{}'.format(
            self._get_name(func),
            _CALLBACK_REPR_HELPER.format_call(self, *args, **kwargs),
        )
        dummy_ctx: _Stacks = {}
        trxn: Transaction[T, None] = (
            Transaction(f'call: {call}')
            .with_execute(func, self, *args, **kwargs)
            .with_simple_rollback(
                self._roll_back_ctx, dummy_ctx,
                reason=f'atomic roll-back: {call}',
            )
        )
        # This is just to keep the transaction in a real "context" in
        # the ledger (so it can be later `.roll_back()`-ed), instead of
        # the dummy context we pushed
        self.add_transactions(trxn, execute=False)
        self._contexts.append(dummy_ctx)
        try:
            self.add_transactions(trxn, execute=True)
            return cast(T, trxn.last_execute_result)
        finally:
            # We've inserted a dummy context so as to "isolate" the
            # transactions created inside the `func()` call; drop it
            is_ctx = partial(operator.is_, dummy_ctx)
            _del_items(self._contexts, is_ctx)

    @classmethod
    def create_atomic_transaction(
        cls, trxns: Collection[Transaction[Any, Any]],
    ) -> Transaction[Self, None]:
        """
        Explicitly construct an "atomic" :py:class:`Transaction` wrapper
        around multiple :py:class:`Transaction`s.

        Return:
            :py:class:`Transaction` object which handles ``trxns``
            "atomically".  That is to say,
            :py:meth:`Transaction.execute`-ing the return value would
            execute ``trxns`` one-by-one; if ANY of them failed, ALL of
            them will be rolled back.

        Note:
            ``trxns`` should all be "fresh" with
            :py:attr:`Transaction.executed` being false; otherwise they
            are silently dropped.

        Example:
            >>> from typing import NoReturn

            >>> def error(
            ...     *args,
            ...     Error: type[Exception] = Exception,
            ...     **kwargs
            ... ) -> NoReturn:
            ...     raise Error(*args, **kwargs)

            >>> log: list[str] = []
            >>> out = log.append

            >>> tr1 = (
            ...     Transaction()
            ...     .with_execute(out, 'exec 1')
            ...     .with_simple_rollback(out, 'roll back 1')
            ... )
            >>> tr2 = (
            ...     Transaction()
            ...     .with_execute(out, 'exec 2')
            ...     .with_simple_rollback(out, 'roll back 2')
            ... )
            >>> tr3 = (  # Fails, causing complete roll-back
            ...     Transaction()
            ...     .with_execute(error, 'exec 3', Error=RuntimeError)
            ...     .with_simple_rollback(out, 'roll back 3')
            ... )
            >>> tr4 = (  # Not executed nor rolled back
            ...     Transaction()
            ...     .with_execute(out, 'exec 4')
            ...     .with_simple_rollback(out, 'roll back 4')
            ... )

            >>> tr_atomic = Ledger.create_atomic_transaction([
            ...     tr1, tr2, tr3, tr4,
            ... ])
            >>> tr_atomic.execute()
            Traceback (most recent call last):
              ...
            RuntimeError: exec 3
            >>> log
            ['exec 1', 'exec 2', 'roll back 2', 'roll back 1']
        """
        name = f'atomic transaction ({len(trxns)} underlying transactions)'
        return (
            Transaction(name)
            .with_execute(cls._from_fresh_transactions, trxns, execute=True)
            .with_method_caller_rollback(
                '_roll_back_all_transactions', name,
                reason='rolling back parent transaction',
            )
        )

    # ------------------------- Helper methods -------------------------

    def _roll_back_ctx(self, ctx: _Stacks, reason: str | None) -> None:
        r"""
        Example:
            >>> import re
            >>> from collections.abc import Callable
            >>> from dataclasses import dataclass
            >>> from typing import Any
            >>> from typing_extensions import Self

            >>> @dataclass
            ... class LogEntry:
            ...     level: str
            ...     msg: str
            ...
            ...     def compare(
            ...         self,
            ...         *,
            ...         level: str | None = None,
            ...         msg: str | None = None,
            ...         regex: bool = False,
            ...     ) -> bool:
            ...         for field, pat in [
            ...             ('level', level), ('msg', msg),
            ...         ]:
            ...             if pat is None:
            ...                 continue
            ...             value = getattr(self, field)
            ...             if regex:
            ...                 if not re.search(pat, value):
            ...                     return False
            ...             elif value != pat:
            ...                 return False
            ...         return True

            >>> class MyLedger(Ledger):
            ...     def __init__(self, *args, **kwargs) -> None:
            ...         super().__init__(*args, **kwargs)
            ...         self.debug_log: list[LogEntry] = []
            ...
            ...     def _debug_output(
            ...         self, msg: str, level: str = 'debug',
            ...     ) -> None:
            ...         self.debug_log.append(LogEntry(level, msg))

            >>> def good_callback(msg: Any) -> None:
            ...     print(msg)

            >>> def bad_callback(msg: Any) -> None:
            ...     raise RuntimeError(msg)

            >>> with MyLedger() as ledger:
            ...     ledger.on_rollback(good_callback, 1)
            ...     ledger.on_rollback(bad_callback, 2)
            ...     ledger.on_rollback(good_callback, 3)
            3
            1

            >>> log = ledger.debug_log
            >>> assert len(log) >= 5, f'{log=!r}; {len(log)=!r}'

            >>> assert log[-5].compare(
            ...     level='debug',
            ...     msg='Starting roll-back '
            ...     '(context exit; 3 transaction(s))...',
            ... ), f'{log[-5]=!r}'
            >>> assert log[-4].compare(
            ...     level='debug',
            ...     msg=r'- Roll-back succeeded \(1/3\): '
            ...     '.*good_callback.*3',
            ...     regex=True,
            ... ), f'{log[-4]=!r}'
            >>> assert log[-3].compare(
            ...     level='warning',
            ...     msg=r'- Roll-back failed \(2/3\): '
            ...     '.*bad_callback.*2.*-> RuntimeError: 2',
            ...     regex=True,
            ... ), f'{log[-3]=!r}'
            >>> assert log[-2].compare(
            ...     level='debug',
            ...     msg=r'- Roll-back succeeded \(3/3\): '
            ...     '.*good_callback.*1.*',
            ...     regex=True,
            ... ), f'{log[-2]=!r}'
            >>> assert log[-1].compare(
            ...     level='debug',
            ...     msg='... roll-back completed '
            ...     '(context exit; 3 transaction(s))',
            ... ), f'{log[-1]=!r}'

            >>> with MyLedger() as ledger:
            ...     pass
            >>> log = ledger.debug_log
            >>> assert len(log) == 1, f'{log=!r}; {len(log)=!r}'
            >>> assert log[-1].compare(
            ...     msg='Nothing to roll back '
            ...     '(context exit; 0 transaction(s))',
            ... ), f'{log[-1]=!r}'
        """
        log = self._debug_output
        ntrxns_total = sum(
            sum(1 for trxn in stack if trxn.can_roll_back)
            for stack in ctx.values()
        )
        note = f'{ntrxns_total} transaction(s)'
        if reason:
            note = f'{reason}; {note}'
        if not ntrxns_total:
            log(f'Nothing to roll back ({note})')
            return
        log(f'Starting roll-back ({note})...')
        ntrxns_run = 0
        for priority, transactions in sorted(
            ctx.items(), reverse=True,  # Higher priority first
        ):
            for trxn in reversed(transactions):
                try:
                    run, result = trxn.roll_back()
                except Exception as e:
                    success, state = False, 'failed'
                    result = self._format_exception(e)
                else:
                    success, state, result = True, 'succeeded', repr(result)
                ntrxns_run += run
                if run:
                    log(
                        f'- Roll-back {state} ({ntrxns_run}/{ntrxns_total}): '
                        f'{trxn!r} -> {result}',
                        'debug' if success else 'warning',
                    )
        log(f'... roll-back completed ({note})')

    def _make_tempfile(self, /, **kwargs) -> Path:
        path = make_tempfile(**kwargs)
        self._debug_output(self._format_tempfile_creation_trxn(path))
        return path

    def _debug_output(self, msg: str, /, level: LogLevel = 'debug') -> None:
        """
        Write debugging output.

        Note:
            This default implementation just writes to the logger at the
            specified level.
        """
        log_func = getattr(diagnostics.log, level)
        log_func(msg)

    @classmethod
    def _from_fresh_transactions(
        cls,
        trxns: Collection[Transaction[Any, Any]],
        header: str | None = None,
        **kwargs
    ) -> Self:
        """
        Create a new instance to which un-executed ``trxns`` are added
        as they are :py:meth:`execute`-d; on errors all the
        added ``trxns`` are rolled back.

        Return:
            The instance.
        """
        ledger = cls()
        try:
            for transaction in trxns:
                if transaction.executed:
                    continue
                ledger.add_transactions(transaction, **kwargs)
        except BaseException as e:
            msg = (
                f'error executing sub-{transaction=!r}: '
                f'{ledger._format_exception(e)}'
            )
            ledger._roll_back_all_transactions(header, msg)
            raise
        return ledger

    def _roll_back_all_transactions(
        self, tr_name: str | None = None, reason: str | None = None,
    ) -> None:
        """
        Thin wrapper around :py:meth:`.roll_back_transactions` with a
        signature more readily used by
        :py:meth:`.create_atomic_transaction()` and the associated
        methods.
        """
        if tr_name is not None:
            if reason is None:
                reason = tr_name
            else:
                reason = f'{tr_name}: {reason}'
        self.roll_back_transactions(reason=reason)

    @staticmethod
    def _get_attribute_info(obj: Any, attr: str) -> tuple[bool, bool, Any]:
        """
        Returns:
            3-tuple of the following:

            - Whether the object has the attribute

            - Whether the present attribute value (if any) should be
              restored if changed

            - The attribute that should be restored (if any)
        """
        try:
            value = getattr(obj, attr)
        except AttributeError:
            return False, False, None

        try:
            static_value = getattr_static(obj, attr)
        except AttributeError:
            has_static = False
        else:
            has_static = True

        if isinstance(obj, type) and has_static:
            # Special-case classes: only restore if value not inherited
            # from a base class
            return True, (attr in vars(obj)), static_value

        try:
            inherited_value = getattr(type(obj), attr)
        except AttributeError:
            # Two cases:
            # - `has_static=True`: `value` is instance-bound (e.g. in
            #   `vars()`)
            #   -> Should be restored
            # - `has_static=False`: `value` is calculated dynamically
            #   (via `.__getattr[ibute]__()`)
            return True, has_static, value

        # If it is a settable data descriptor, reset it; else, leave it
        # alone (i.e. delete the instance-bound override we assign to)
        InheritedDescriptorType = type(inherited_value)
        if not callable(getattr(InheritedDescriptorType, '__set__', None)):
            is_settable = False
        elif issubclass(InheritedDescriptorType, property):
            # Special-case properties
            is_settable = inherited_value.fset is not None
        else:
            is_settable = True
        return True, is_settable, value

    @staticmethod
    def _get_name(obj: Any, /) -> str:
        """
        Get an appropriate name for an arbitrary object.

        Example:
            >>> import textwrap

            >>> Ledger._get_name(textwrap)
            'textwrap'
            >>> Ledger._get_name(textwrap.dedent)
            'textwrap.dedent'
            >>> Ledger._get_name(str)
            'str'
            >>> Ledger._get_name(print)
            'print'
            >>> Ledger._get_name(object())  # doctest: +ELLIPSIS
            '<object object at 0x...>'
        """
        if hasattr(obj, '__qualname__'):
            name = obj.__qualname__
        elif hasattr(obj, '__name__'):
            name = obj.__name__
        else:
            return repr(obj)
        if hasattr(obj, '__module__'):
            if obj.__module__ not in ('builtins', '__builtins__'):
                name = f'{obj.__module__}.{name}'
        return str(name)

    @staticmethod
    def _format_mapping_update_trxn[K](
        mapping: Mapping[K, Any], key: K, change: str,
    ) -> str:
        # Note: this supersedes
        # `Cleanup.make_tempfile(_format_debug_msg=...)`
        mapping_repr = object.__repr__(mapping)
        return f'{mapping_repr}[{key!r}]: {change}'

    @staticmethod
    def _format_tempfile_creation_trxn(path: Path) -> str:
        return f'Created tempfile: {path}'

    @staticmethod
    def _pop_n_contexts(
        contexts: _StackContexts, n: int,
    ) -> Generator[_Stacks]:
        for _ in range(n):
            try:
                yield contexts.pop()
            except IndexError:  # Nothing to pop
                return

    @staticmethod
    def _pop_all_contexts(contexts: _StackContexts) -> Generator[_Stacks]:
        while contexts:
            yield contexts.pop()

    @staticmethod
    def _format_exception(xc: BaseException) -> str:
        xc_repr = type(xc).__name__
        if (msg := str(xc)):
            xc_repr = f'{xc_repr}: {msg}'
        return xc_repr

    @property
    def _current_context(self) -> _Stacks:
        try:
            return self._contexts[-1]
        except IndexError:
            ctx: _Stacks = {}
            self._contexts.append(ctx)
            return ctx


def _del_items(
    seq: MutableSequence[T], criterion: Callable[[T], bool],
) -> None:
    """
    Example:
        >>> fib = [1, 1, 2, 3, 5, 8, 13]
        >>> _del_items(fib, lambda x: bool(x % 2))
        >>> fib
        [2, 8]
    """
    n = len(seq)
    for offset, item in enumerate(reversed(seq), 1):
        if criterion(item):
            del seq[n - offset]
