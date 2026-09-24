"""
Tools for setting up profiling in a curated environment (e.g. with
the use of :py:mod:`kernprof`).
"""
from __future__ import annotations

import builtins
import dataclasses
import os
import warnings
from collections.abc import Collection
from io import StringIO
from textwrap import indent
from typing import Any, TextIO, cast
from typing_extensions import Self

from . import _diagnostics as diagnostics, profile as _global_profiler
from ._threading_patches import apply as apply_threading_patches
from .autoprofile.autoprofile import (
    _extend_line_profiler_for_profiling_imports as upgrade_profiler,
)
from .autoprofile.util_static import modpath_to_modname
from .autoprofile.eager_preimports import (
    is_dotted_path, write_eager_import_module,
)
from .cleanup import Cleanup
from .cli_utils import short_string_path
from .line_profiler import LineProfiler
from .profiler_mixin import ByCountProfilerMixin


__all__ = ('ClassifiedPreimportTargets', 'CuratedProfilerContext')


@dataclasses.dataclass
class ClassifiedPreimportTargets:
    """
    Pre-import targets classified into three bins: ``regular`` targets,
    targets to ``recurse`` into, and ``invalid`` targets.
    """
    regular: list[str] = dataclasses.field(default_factory=list)
    recurse: list[str] = dataclasses.field(default_factory=list)
    invalid: list[str] = dataclasses.field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.regular or self.recurse)

    def write_preimport_module(
        self, fobj: TextIO, *, debug: bool | None = None, **kwargs
    ) -> None:
        r"""
        Convenience interface with
        :py:func:`~.write_eager_import_module`, writing a module which
        when imported sets up profiling of the targets.

        Args:
            fobj (TextIO):
                File object to write said module to.
            debug (Optional[bool]):
                Whether to generate debugging outputs.
            kwargs:
                Passed to :py:func:`~.write_eager_import_module`.

        Example:
            >>> from pathlib import Path
            >>> from contextlib import ExitStack, redirect_stdout
            >>> from os import devnull
            >>> from tempfile import TemporaryDirectory

            >>> import pytest

            >>> with TemporaryDirectory() as tmpdir_:
            ...     tmpdir = Path(tmpdir_)
            ...     file = tmpdir / 'preimports.py'
            ...     targets = ClassifiedPreimportTargets.from_targets([
            ...         'inspect.getdoc',
            ...         str(tmpdir / 'nonexistent.py'),
            ...     ])
            ...     with ExitStack() as stack:
            ...         enter = stack.enter_context
            ...         _ = enter(pytest.warns(
            ...             match='1 .* target cannot be converted .*'
            ...             r'nonexistent\.py',
            ...         ))
            ...         fobj = enter(file.open('w'))
            ...         _ = enter(
            ...             redirect_stdout(enter(open(devnull, 'a'))),
            ...         )
            ...         targets.write_preimport_module(fobj)
            ...     line_matcher = pytest.LineMatcher(
            ...         file.read_text().splitlines(),
            ...     )

            >>> line_matcher.re_match_lines([
            ...     r'\s*import inspect', r'\s*add\(.*\bgetdoc\)',
            ... ])
        """
        if self.invalid:
            invalid_targets = sorted(set(self.invalid))
            msg = (
                '{} profile-on-import target{} cannot be converted to '
                'dotted-path form: {!r}'.format(
                    len(invalid_targets),
                    '' if len(invalid_targets) == 1 else 's',
                    invalid_targets,
                )
            )
            # Log before warn in case the warning is raised
            diagnostics.log.warning(msg)
            warnings.warn(msg, stacklevel=2)

        if not self:
            return None
        # Note: `ty` (but not `mypy`) keeps complaining about our
        # splatting this dict; explicitly use `Any` to tell it to shut
        # up.
        write_module_kwargs: dict[str, Any] = {
            'dotted_paths': self.regular,
            'recurse': self.recurse,
            **kwargs,
        }
        if diagnostics.DEBUG if debug is None else debug:  # nocover
            with StringIO() as sio:
                write_eager_import_module(stream=sio, **write_module_kwargs)
                code = sio.getvalue()
            print(code, end='', file=fobj)
            if hasattr(fobj, 'name'):
                fobj_repr = repr(short_string_path(str(fobj.name)))
            else:
                fobj_repr = repr(fobj)  # Fall back
            diagnostics.log.debug(
                f'Wrote temporary module for pre-imports to {fobj_repr}:\n'
                + indent(code, '  ')
            )
        else:
            write_eager_import_module(stream=fobj, **write_module_kwargs)

    @classmethod
    def from_targets(
        cls,
        targets: Collection[str],
        exclude: Collection[os.PathLike[str] | str] = (),
    ) -> Self:
        """
        Create an instance based on a collection of targets
        (like what is supplied to ``kernprof --prof-mod=...``).

        Args:
            targets (Collection[str])
                Collection of dotted paths and filenames to profile.
            exclude (Collection[str])
                Collections of filenames which are explicitly excluded
                from being profiled.

        Return:
            New instance.

        Example:
            >>> import multiprocessing
            >>> import os.path
            >>> import textwrap
            >>> import xml
            >>> from importlib.util import find_spec
            >>> from tempfile import TemporaryDirectory

            >>> get_targets = ClassifiedPreimportTargets.from_targets
            >>> invalid_module = 'textwrapppppp'
            >>> assert find_spec(invalid_module) is None
            >>> nonexistent_target = textwrap.__file__.replace(
            ...     'textwrap', invalid_module,
            ... )

            >>> with TemporaryDirectory() as tmpdir:
            ...     malformed_target = os.path.join(tmpdir, 'b-a-r.py')
            ...     with open(malformed_target, mode='w'):
            ...         pass  # touch
            ...     excluded_target = os.path.join(tmpdir, 'excl.py')
            ...     with open(excluded_target, mode='w'):
            ...         pass  # touch
            ...
            ...     raw_targets = [
            ...         'sys',
            ...         # Could be invalid, but we don't know ATP
            ...         'foo',
            ...         # Resolved to `'textwrap'`
            ...         textwrap.__file__,
            ...         # Invalid targets
            ...         nonexistent_target, malformed_target,
            ...         # This is valid, but will be excluded
            ...         excluded_target,
            ...         # Resolved to `'xml'` (non-recursed)
            ...         xml.__file__,
            ...         # Resolved to `'multiprocessing'` (recursed)
            ...         os.path.dirname(multiprocessing.__file__),
            ...     ]
            ...     tar = get_targets(
            ...         raw_targets, exclude=[excluded_target],
            ...     )

            >>> all_targets = set().union(tar.regular, tar.recurse)
            >>> assert {
            ...     'sys', 'foo', 'textwrap', 'multiprocessing',
            ... } <= all_targets, f'{all_targets=!r}'
            >>> assert 'xml' in tar.regular, f'{tar.regular=!r}'
            >>> assert 'excl' not in all_targets
            >>> assert {
            ...     nonexistent_target, malformed_target,
            ... } == set(tar.invalid), f'{tar=!r}; {tar.invalid=!r}'

        Notes:
            The distinction between :py:attr:`.regular` and
            :py:attr:`.recurse` is that packages in the former are
            guaranteed to NOT be recursed into, while those in the
            latter can be recursed into where appropriate (see
            :py:func:`line_profiler.autoprofile.eager_preimports.\
resolve_profiling_targets`).
            Hence, modules and packages are classified into
            :py:attr:`.recurse` by default; recursion into packages is
            stopped (i.e. target classified into :py:attr:`.regular`) by
            supplying a dotted path suffixed with ``.__init__`` or a
            file path to the package's ``__init__.py``.
        """
        filtered_targets = []
        recurse_targets = []
        invalid_targets = []
        for target in targets:
            if is_dotted_path(target):
                modname = target
            else:
                # Paths already normalized by
                # `_normalize_profiling_targets()`
                if not os.path.exists(target):
                    invalid_targets.append(target)
                    continue
                if any(
                    os.path.samefile(target, excluded) for excluded in exclude
                ):
                    # Ignore the script to be run in eager importing
                    # (`line_profiler.autoprofile.autoprofile.run()`
                    # will handle it)
                    continue
                modname = modpath_to_modname(target, hide_init=False)
                if not is_dotted_path(modname):
                    invalid_targets.append(target)
                    continue
            if modname.endswith('.__init__'):
                modname = modname.rpartition('.')[0]
                filtered_targets.append(modname)
            else:
                recurse_targets.append(modname)
        return cls(filtered_targets, recurse_targets, invalid_targets)


class CuratedProfilerContext(Cleanup):
    """
    Context manager for handling various bookkeeping tasks when setting
    up and tearing down profiling:

    - Slipping ``prof`` into the builtin namespace (if
      ``insert_builtin`` is true) and :py::deco:`~.profile`
    - Patch :py:class:`threading.Thread` so that line-profiling is
      enabled on new threads if it is on the spawning threads
    - At exit, clearing the ``enable_count`` of ``prof``, properly
      disabling it

    Notes:

        - The attributes on this object are to be considered
          implementation details, but not its methods and their
          signatures.

        - In contrast to the base class (:py:class:`Cleanup`), while
          this context manager is still reentrant, reentering in nested
          `with: ...` statements is a no-op.
    """
    def __init__(
        self,
        prof: ByCountProfilerMixin,
        insert_builtin: bool = False,
        builtin_loc: str = 'profile',
    ) -> None:
        super().__init__()
        self.prof = prof
        self.insert_builtin = insert_builtin
        self.builtin_loc = builtin_loc
        self._installed = False
        self._kpo = _global_profiler._kernprof_overwrite

    def _global_install(self, prof: ByCountProfilerMixin | None) -> None:
        # Wrapper to convince type-checkers it is okay to pass these
        # stuff to `._kernprof_overwrite()`. We don't want to patch
        # that method's signature because passing non `LineProfiler`
        # objects to it should be the exception, not the norm.
        self._kpo(cast(LineProfiler, prof))

    @staticmethod
    def _disable_profiler(prof: ByCountProfilerMixin) -> None:
        for _ in range(getattr(prof, 'enable_count', 0)):
            prof.disable_by_count()

    def install(self) -> None:
        if self._installed:
            return
        # Equip the profiler instance with the
        # `.add_imported_function_or_module()` pseudo-method
        upgrade_profiler(self.prof)
        # Overwrite the explicit profiler (`@line_profiler.profile`)
        self._global_install(self.prof)
        self.add_cleanup(self._global_install, None)
        # Patch `threading`
        if isinstance(self.prof, LineProfiler):
            apply_threading_patches(self, self.prof)
        # Set up hooks to deal with inserting `.prof` as a builtin name
        if self.insert_builtin:
            self.patch(builtins, self.builtin_loc, self.prof)
        # Disable the profiler
        self.add_cleanup(self._disable_profiler, self.prof)

        self.patch(self, '_installed', True)

    def uninstall(self) -> None:
        self.cleanup(reason='uninstalling profiling context')

    def __enter__(self) -> Self:
        self.install()
        return self

    def __exit__(self, *_, **__) -> None:
        self.uninstall()
