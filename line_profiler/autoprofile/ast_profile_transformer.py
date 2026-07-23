from __future__ import annotations

import ast
from collections.abc import Callable, Collection, Mapping, Sequence
from functools import partial
from os import PathLike
from types import MappingProxyType
from typing import TypeVar, cast, get_args

from ..toml_config import ConfigSource
from ._import_targets import ImportTarget
from .profmod_extractor import _CompoundNodeType, _ImportFinder


_Import = TypeVar('_Import', ast.Import, ast.ImportFrom)

_PROFILE_IMPORTS_IN_DEFAULT: MappingProxyType[_CompoundNodeType, bool]
_PROFILE_IMPORTS_IN_DEFAULT = MappingProxyType(dict.fromkeys(
    get_args(_CompoundNodeType), True,
))


def ast_create_profile_node(
    modname: str,
    profiler_name: str = 'profile',
    attr: str = 'add_imported_function_or_module',
) -> ast.Expr:
    """
    Create an abstract syntax tree node that adds an object to the
    profiler to be profiled, by calling the ``attr`` method of
    ``profile`` and passing ``modname`` to it.

    At runtime, this adds the object to the profiler so it can be
    profiled. This node must be added after the first instance of
    ``modname`` in the AST and before it is used.

    The node will look like:
        >>> # xdoctest: +SKIP
        >>> import foo.bar
        >>> profile.add_imported_function_or_module(foo.bar)

    Args:
        modname (str):
            name of the imported module.

        profiler_name (str):
            name of the :py:class:`line_profiler.LineProfiler` object.

        attr (str):
            name of the method of the :py:class:`LineProfiler` object to
            call on the imported module.

    Returns:
        (_ast.Expr): expr
            AST node that adds ``modname`` to profiler.
    """
    func = ast.Attribute(
        value=ast.Name(id=profiler_name, ctx=ast.Load()),
        attr=attr,
        ctx=ast.Load(),
    )
    names = modname.split('.')
    value: ast.expr = ast.Name(id=names[0], ctx=ast.Load())
    for name in names[1:]:
        value = ast.Attribute(attr=name, ctx=ast.Load(), value=value)
    expr = ast.Expr(value=ast.Call(func=func, args=[value], keywords=[]))
    return expr


def ast_create_star_import_node(
    modname: str,
    targets: Collection[str] | None,
    profiler_name: str = 'profile',
    attr: str = 'add_star_import',
) -> ast.Expr:
    """
    AST node similar to that created by
    :py:func:`.ast_create_profile_node`, except that it handles
    star-imports (``from ... import *``), like:

    >>> # doctest: +SKIP
    >>> from foo.bar import *
    >>> profile.add_star_import(
    ...     'foo.bar', ['foo.bar', 'spam.ham'], locals(),
    ... )

    Args:
        modname (str):
            name of the imported module.

        targets (Collection[str] | None):
            profile-on-import module targets; if :py:const:`None`, all
            the names imported by the star-import will be added to the
            profiler.

        profiler_name (str):
            name of the :py:class:`line_profiler.LineProfiler` object.

        attr (str):
            name of the method of the
            :py:class:`line_profiler.LineProfiler` object to call on the
            imported module.

    Returns:
        (_ast.Expr): expr
            AST node that adds ``modname`` to profiler.
    """
    func_node = ast.Attribute(
        value=ast.Name(id=profiler_name, ctx=ast.Load()),
        attr=attr,
        ctx=ast.Load(),
    )
    modname_node = ast.Constant(value=modname)
    if targets is None:
        targets_node: ast.Constant | ast.List = ast.Constant(value=None)
    else:
        targets_node = ast.List(
            elts=[ast.Constant(value=t) for t in targets],
            ctx=ast.Load(),
        )
    namespace_node = ast.Call(
        func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[],
    )
    expr = ast.Expr(value=ast.Call(
        func=func_node,
        args=[modname_node, targets_node, namespace_node],
        keywords=[],
    ))
    return expr


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


class AstProfileTransformer(ast.NodeTransformer):
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
        profiled_imports: Collection[str] | None = None,
        profiler_name: str = 'profile',
        *,
        profile_star_imports: bool = False,
        profile_imports_in: Mapping[
            _CompoundNodeType, bool
        ] = _PROFILE_IMPORTS_IN_DEFAULT,
    ) -> None:
        """Initializes the AST transformer with the profiler name.

        Args:
            profile_imports (bool):
                if True, profile all concrete (non-star) imports.

            profiled_imports (Collection[str]):
                list of dotted paths of imports to skip that have already been added to profiler.

            profiler_name (str):
                the profiler name used as decorator and for the method call to add to the object
                to the profiler.

            profile_star_imports (bool):
                if this and ``profile_imports`` are True, also profile
                star-imports.

            profile_imports_in \
(Mapping[Literal['Module', 'Interactive', \
'FunctionDef', 'AsyncFunctionDef', 'ClassDef', \
'For', `AsyncFor`, `While`, 'If', 'match_case', \
'With', 'AsyncWith'. 'Try', 'TryStar', 'ExceptHandler'], bool]):
                for each of the compound-statemnt node type, whether to
                profile import statements residing therein.
        """
        self._profile_imports = bool(profile_imports)
        self._profiled_imports = set(profiled_imports or ())
        self._profiler_name = profiler_name
        self._profile_star_imports = profile_star_imports
        self._should_visit_imports = dict(profile_imports_in)
        self._dropped_star_imports: set[ImportTarget] = set()
        self._current_loc: list[str] = []

    def _visit_func_def(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> ast.FunctionDef | ast.AsyncFunctionDef:
        """Decorate functions/methods with profiler.

        Checks if the function/method already has a profile_name decorator, if not, it will append
        profile_name to the end of the node's decorator list.
        The decorator is added to the end of the list to avoid conflicts with other decorators
        e.g. @staticmethod.

        Args:
            node (_ast.FunctionDef | _ast.AsyncFunctionDef):
                function/method in the AST

        Returns:
            node (_ast.FunctionDef | _ast.AsyncFunctionDef):
                function/method with profiling decorator
        """
        decor_ids = set()
        for decor in node.decorator_list:
            if isinstance(decor, ast.Name):
                decor_ids.add(decor.id)
        if self._profiler_name not in decor_ids:
            node.decorator_list.append(
                ast.Name(id=self._profiler_name, ctx=ast.Load())
            )
        self.generic_visit(node)
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = _visit_func_def

    def _visit_import(
        self,
        node: _Import,
        get_import_targets: Callable[[_Import], Sequence[ImportTarget]],
    ) -> _Import | list[_Import | ast.Expr]:
        """
        Add a node that profiles an import. If:

        - ``profile_imports`` is true,

        - The import statement isn't nested in a compound-statement node
          type explicitly excluded via ``profile_imports_in``, and

        - The import target is not in ``profiled_imports``,

        a node which calls the profiler method adding the object to the
        profiler is added immediately after the import.

        Args:
            node (_Import):
                import[-from] node in the AST
            get_import_targets \
(Callable[[_Import], Sequence[ImportTarget]]):
                helper callable for analyzing the node

        Returns:
            node (_Import | list[_Import | _ast.Expr]):
                if ``profile_imports`` is False:
                    the import node
                if ``profile_imports`` is True:
                    a list containing the import node and the profiling
                    node(s)
        """
        if not self._profile_imports:
            should_profile = False
        else:
            # Check if this node is nested inside compound statements
            # that we shouldn't look for imports in
            *ancestry, _ = self._current_loc
            svi = self._should_visit_imports
            should_profile = all(
                svi.get(cast(_CompoundNodeType, a_type), True)
                for a_type in ancestry
            )

        if not should_profile:
            self.generic_visit(node)
            return node

        this_visit = cast(_Import, self.generic_visit(node))
        visited: list[_Import | ast.Expr] = [this_visit]
        for target in get_import_targets(node):
            name = target.name
            if name in self._profiled_imports:
                continue
            expr = _ast_create_node_from_import_target(
                target, profile_star_imports=self._profile_star_imports,
            )
            if expr is None:  # Bookkeeping
                self._dropped_star_imports.add(target)
            else:
                self._profiled_imports.add(name)
                visited.append(expr)
        return visited

    def visit_Import(
        self, node: ast.Import,
    ) -> ast.Import | list[ast.Import | ast.Expr]:
        """
        Add nodes that profile objects imported using the
        ``import foo`` syntax.

        Args:
            node (_ast.Import):
                import in the AST

        Returns:
            node (_ast.Import | list[_ast.Import | _ast.Expr]):
                if ``profile_imports`` is False:
                    the import node
                if ``profile_imports`` is True:
                    a list containing the import node and the
                    profiling node(s)
        """
        # Note: we don't actually care about the `ImportTarget.index`
        # here; in fact, we're just reusing the name-resolution
        # machinery in `ImportTarget`
        get_targets = partial(ImportTarget._from_import_node, 0)
        return self._visit_import(node, get_targets)

    def visit(self, node: ast.AST) -> ast.AST | list[ast.AST]:
        # Bookkeeping
        loc = self._current_loc
        loc.append(type(node).__name__)
        try:
            return super().visit(node)
        finally:
            loc.pop()

    def visit_ImportFrom(
        self, node: ast.ImportFrom
    ) -> ast.ImportFrom | list[ast.ImportFrom | ast.Expr]:
        """
        Add nodes that profile objects imported using the
        ``from foo import bar`` syntax.

        Args:
            node (_ast.ImportFrom):
                import in the AST

        Returns:
            node (_ast.Import | list[_ast.Import | _ast.Expr]):
                if ``profile_imports`` is False:
                    the import node
                if ``profile_imports`` is True:
                    a list containing the import node and the
                    profiling node(s)
        """
        get_targets = partial(ImportTarget._from_import_from_node, 0)
        return self._visit_import(node, get_targets)

    @staticmethod
    def _get_profile_imports_in(
        config: ConfigSource | None = None,
    ) -> dict[_CompoundNodeType, bool]:
        if config is None:
            config = ConfigSource.from_default()
        return _ImportFinder.filter_node_types(
            **_ImportFinder._get_filter_args(config),
        )

    @classmethod
    def _transform(
        cls,
        node: ast.Module,
        filename: PathLike[str] | str | None = None,
        *,
        config: ConfigSource | None = None,
        **kwargs,
    ) -> ast.Module:
        """
        Wrapper around ``<instance>.visit()`` with extra bookkeeping.

        Args:
            node (ast.Module):
                AST module node
            filename (PathLike[str] | str | None):
                Optional filename to be used in error/warning messages
            config (ConfigSource | None):
                Optional :py:class:`.ConfigSource` to load options from,
                controlling whether an import should be profiled
            **kwargs
                Passed to the initializer

        Returns:
            node (ast.Module):
                Input module node
        """
        kwargs.setdefault(
            'profile_imports_in', cls._get_profile_imports_in(config),
        )
        transformer = cls(**kwargs)
        dropped_star_imports = transformer._dropped_star_imports
        if filename is None:
            filename = '???'
        try:
            return cast(ast.Module, transformer.visit(node))
        finally:
            ImportTarget._check_and_warn_dropped_imports(
                dropped_star_imports,
                "we don't currently handle `from ... import *` statements",
                filename,
                stacklevel=2,  # Attribute warning to the caller
            )
