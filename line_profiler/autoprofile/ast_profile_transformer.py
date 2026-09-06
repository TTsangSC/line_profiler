from __future__ import annotations

import ast
from collections.abc import Collection, Mapping, Sequence
from functools import partial, wraps
from os import PathLike
from types import MappingProxyType
from typing import TypeVar, cast, get_args
from warnings import warn

from .. import _diagnostics as diagnostics
from ..toml_config import ConfigSource
from ._import_targets import _DROPPED_STAR_IMPORTS_MSG_TEMPLATE, ImportTarget
from ._single_pass_transformer import (
    _CompoundNodeChecker,
    _ConcreteDuplicateImportChecker,
    CompoundNodeType,
    CompoundStatement,
    SinglePassTransformer,
    ast_create_profile_node, ast_create_star_import_node,
)
from .profmod_extractor import _should_profile_star_imports


__all__ = (
    'AstProfileTransformer',
    'ast_create_profile_node', 'ast_create_star_import_node',
)

_Import = TypeVar('_Import', ast.Import, ast.ImportFrom)

_PROFILE_IMPORTS_IN_DEFAULT: MappingProxyType[CompoundNodeType, bool]
_PROFILE_IMPORTS_IN_DEFAULT = MappingProxyType(dict.fromkeys(
    get_args(CompoundNodeType), True,
))


def _ast_create_node_from_import_target(
    target: ImportTarget,
    modnames_to_profile: Collection[str] | None = None,
    profile_star_imports: bool = False,
) -> ast.Expr | None:
    if target.resolved_name is None:  # Star-imports
        if not profile_star_imports:
            return None
        assert target.name.endswith('.*')
        return ast_create_star_import_node(
            target.name[:-2], modnames_to_profile,
        )
    return ast_create_profile_node(target.resolved_name)


_wrap_sig = partial(
    wraps, assigned=('__annotations__', '__type_params__'),
)


class _ContextAwareDuplicateChecker(_ConcreteDuplicateImportChecker):
    """
    This checker is context-aware and only deduplicates imports of the
    same object in the same scope.
    """
    def __init__(
        self,
        profiled_imports: (
            Mapping[Sequence[str | int], Sequence[ImportTarget]] | None
        ) = None,
    ) -> None:
        super().__init__()
        pi = self._profiled_imports
        for context, imports in (profiled_imports or {}).items():
            ctx, index = self._check_context(context)
            pi.setdefault(ctx, {})[index] = list(imports)


class _LegacyDuplicateChecker:
    """
    This checker replicates legacy behavior: all imports in the same
    module are treated on equal footing regardless of scoping, and each
    import target is only passed once to the profiler.

    Notes:
        Using this results in a :py:class:`DeprecationWarning`. This is
        because such deduplication can result in the profiler never
        getting passed an intended target. Consider the following
        example:

        >>> # doctest: +SKIP
        >>>
        >>>
        >>> def foo():
        ...     from spam import ham
        ...
        ...     return ham()
        ...
        >>>
        >>> def bar():
        ...     from spam import ham
        ...
        ...     return ham()
        ...
        >>>
        >>> if __name__ == '__main__':
        ...     bar()

        In the above example, ``spam.ham()`` is never profiled even
        after AST rewrite, because only the import in ``foo()`` has a
        post-import profiling statement inserted.
    """
    def __init__(self, profiled_imports: Collection[str]) -> None:
        msg = (
            'AstProfileTransformer(profiled_imports=<Collection[str]>) '
            'is deprecated because of erroneous resolution of duplicate '
            'imports; future code should use either '
            '`Mapping[Sequence[str | int], ImportTarget]` '
            '(e.g. the return value of `ProfmodExtractor.extract_all()`) '
            'or `None`'
        )
        diagnostics.log.warning(f'DeprecationWarning: {msg}')
        warn(msg, category=DeprecationWarning, stacklevel=2)  # Caller
        self._profiled_imports = set(profiled_imports)

    def should_profile_import(self, target: ImportTarget, _) -> bool:
        return target.name not in self._profiled_imports

    def record_profiled_import(self, target: ImportTarget, _) -> None:
        self._profiled_imports.add(target.name)


class AstProfileTransformer(SinglePassTransformer):
    """
    Transform an abstract syntax tree adding profiling to all of its
    objects, by:

    - Adding profiler decorators on all functions & methods that are not
      already decorated with the profiler.
    - If ``profile_imports`` is True, a profiler method call (see
      :py:func:`line_profiler.autoprofile.line_profiler_utils\
.add_imported_function_or_module`
      and
      :py:func:`line_profiler.autoprofile.line_profiler_utils\
.add_star_import`)
      is added to all imports immediately after the import.
    """
    def __init__(
        self,
        profile_imports: bool = False,
        profiled_imports: (
            Mapping[Sequence[str | int], Sequence[ImportTarget]]
            | Collection[str]
            | None
        ) = None,
        profiler_name: str = 'profile',
        *,
        profile_star_imports: bool = False,
        profile_imports_in: Mapping[
            CompoundNodeType, bool
        ] = _PROFILE_IMPORTS_IN_DEFAULT,
    ) -> None:
        """
        Initializes the AST transformer.

        Args:
            profile_imports (bool):
                if True, profile all concrete (non-star) imports.

            profiled_imports \
(Mapping[Sequence[str | int], Sequence[ImportTarget]] \
| Collection[str] | None):
                ``Mapping[Sequence[str | int], Sequence[ImportTarget]]``
                    mapping from the locations of already-profiled
                    import targets to those import targets themselves,
                    in the same format as the return value of
                    :py:meth:`line_profiler.autoprofile\
.ProfmodExtractor.extract_all`.
                ``Collection[str]``
                    DEPRECATED; dotted paths of imports to skip that
                    have already been added to profiler.
                :py:const:`None`
                    equivalent to ``{}``, i.e. no prior profiled
                    imports.

            profiler_name (str):
                the profiler name used as decorator and for the method
                call to add to the object to the profiler.

            profile_star_imports (bool):
                if this and ``profile_imports`` are True, also profile
                star-imports.

            profile_imports_in \
(Mapping[Literal['Module', 'Interactive', \
'FunctionDef', 'AsyncFunctionDef', 'ClassDef', \
'For', `AsyncFor`, `While`, 'If', 'match_case', \
'With', 'AsyncWith'. 'Try', 'TryStar', 'ExceptHandler'], bool]):
                for each of the compound-statement node type, whether to
                profile import statements residing therein.
        """
        super().__init__(
            profiler_name=profiler_name,
            prof_func_defs=True,
            prof_explicit_imports=profile_imports,
            prof_star_imports=profile_imports and profile_star_imports,
            prof_imports_in=profile_imports_in,
        )

        if profiled_imports is None:
            pass
        elif (
            isinstance(profiled_imports, Mapping)
            and all(
                isinstance(imports, Sequence)
                for imports in profiled_imports.values()
            )
        ):
            self._duplicate_import_checker = _ContextAwareDuplicateChecker(
                cast(
                    Mapping[Sequence[str | int], Sequence[ImportTarget]],
                    profiled_imports,
                ),
            )
        elif (
            isinstance(profiled_imports, Collection)
            and all(isinstance(imp, str) for imp in profiled_imports)
        ):
            self._duplicate_import_checker = _LegacyDuplicateChecker(
                cast(Collection[str], profiled_imports),
            )
        else:  # nocover
            raise TypeError(
                f'profiled_imports = {profiled_imports!r}: '
                'expected `Collection[str]`, '
                '`Mapping[Sequence[str | int], Sequence[ImportTarget]]`, '
                'or `None`',
            )

        self._dropped_star_imports: set[ImportTarget] = set()

    def _handle_new_import_target(
        self, target: ImportTarget,
    ) -> ast.Expr | None:
        """
        Add a node that profiles an import. If:

        - ``profile_imports`` is true,

        - The import statement isn't nested in a compound-statement node
          type explicitly excluded via ``profile_imports_in``, and

        - The import target is not in ``profiled_imports``,

        a node which calls the profiler method adding the object to the
        profiler is added immediately after the import.
        """
        if not (self._prof_star_imports or self._prof_explicit_imports):
            return None
        maybe_expr = _ast_create_node_from_import_target(
            target, profile_star_imports=bool(self._prof_star_imports),
        )
        if maybe_expr is None and target.resolved_name is None:
            self._dropped_star_imports.add(target)
        return maybe_expr

    @_wrap_sig(SinglePassTransformer.visit_Import)
    def visit_Import(self, /, *args, **kwargs):
        """
        Returns:
            node (_ast.Import | list[_ast.Import | _ast.Expr]):
                if ``profile_imports`` is False:
                    the import node
                if ``profile_imports`` is True:
                    a list containing the import node and the
                    profiling node(s)
        """
        # Thin wrapper; we just want to override the docstring
        return super().visit_Import(*args, **kwargs)

    @_wrap_sig(SinglePassTransformer.visit_Import)
    def visit_ImportFrom(self, /, *args, **kwargs):
        """
        Returns:
            node (_ast.Import | list[_ast.Import | _ast.Expr]):
                if ``profile_imports`` is False:
                    the import node
                if ``profile_imports`` is True:
                    a list containing the import node and the
                    profiling node(s)
        """
        # Thin wrapper; we just want to override the docstring
        return super().visit_ImportFrom(*args, **kwargs)

    @staticmethod
    def _get_profile_imports_in(
        config: ConfigSource | None = None,
        profile_nested_imports: Collection[CompoundStatement] | None = None,
    ) -> dict[CompoundNodeType, bool]:
        checker = _CompoundNodeChecker.from_config(
            config, profile_nested_imports,
        )
        return dict(checker.allowed)

    @classmethod
    def _transform(
        cls,
        node: ast.Module,
        filename: PathLike[str] | str | None = None,
        *,
        config: ConfigSource | None = None,
        profile_star_imports: bool | None = None,
        profile_nested_imports: Collection[CompoundStatement] | None = None,
        _known_dropped_star_imports: Collection[ImportTarget] | None = None,
        **kwargs,
    ) -> ast.Module:
        """
        Wrapper around ``<instance>.visit()`` with extra bookkeeping and
        convenience args.

        Args:
            node (ast.Module):
                AST module node

            filename (PathLike[str] | str | None):
                Optional filename to be used in error/warning messages

            config (ConfigSource | None):
                Optional :py:class:`.ConfigSource` to load options from,
                controlling whether an import should be profiled

            profile_star_imports (bool | None):
                Whether to profile star-imports (``from ... import *``);
                if :py:const:`None`, it is loaded from the ``config``
                (from ``autoprofile.prof_star_imports``)

            profile_nested_imports \
(Collection[Literal['func_defs', 'class_defs', \
'loops', 'conditionals', 'contexts', 'try_except']] | None):
                Which of the compound-statement types to look for nested
                imports in;
                if :py:const:`None`, it is loaded from the ``config``
                (from ``autoprofile.import_discovery``)

            **kwargs
                Passed to the initializer

        Returns:
            node (ast.Module):
                Input module node
        """
        if profile_star_imports is None:
            profile_star_imports = _should_profile_star_imports(config)
        kwargs.setdefault(
            'profile_imports_in',
            cls._get_profile_imports_in(config, profile_nested_imports),
        )
        transformer = cls(
            profile_star_imports=profile_star_imports, **kwargs,
        )
        dropped_star_imports = transformer._dropped_star_imports
        if filename is None:
            filename = '???'
        try:
            return cast(ast.Module, transformer.visit(node))
        finally:
            if _known_dropped_star_imports:
                # Don't double-warn on import targets that we already
                # know should be dropped
                dropped_star_imports.difference_update(
                    _known_dropped_star_imports,
                )
            ImportTarget._check_and_warn_dropped_imports(
                dropped_star_imports,
                _DROPPED_STAR_IMPORTS_MSG_TEMPLATE.format(
                    action='profiled',
                    argname='profile_star_imports',
                ),
                filename,
                stacklevel=2,  # Attribute warning to the caller
            )
