"""
Single-pass AST transformer consolidating the functionalities of the
legacy :py:mod:`line_profiler.autoprofile.profmod_extractor` and
:py:mod:`line_profiler.autoprofile.ast_profile_transformer`.

Notes:
    The contents of this submodule should be considered private API and
    subject to changes without notice; documentations are provided for
    reference only.
"""
from __future__ import annotations

import ast
import os
from collections.abc import (
    Callable, Collection, Mapping, MutableSequence, Sequence,
)
from functools import cached_property, partial
from typing import (
    Any, ClassVar, Literal, Protocol, Self, TypeVar, cast, get_args,
)

from ..toml_config import ConfigSource
from ._import_targets import ImportTarget


__all__ = (
    'ContextAwareVisitor', 'SinglePassTransformer',
    'ast_create_profile_node', 'ast_create_star_import_node',
    'should_profile_regular_import', 'should_profile_star_import',
)

# Node types where code blocks can be found
CompoundNodeType = Literal[
    # Basic top-level nodes
    'Module', 'Interactive',
    # Function-definition nodes
    'FunctionDef', 'AsyncFunctionDef',
    # Class-definition nodes
    'ClassDef',
    # Loop nodes
    'For', 'AsyncFor', 'While',
    # Conditional nodes
    'If', 'match_case',
    # Context nodes
    'With', 'AsyncWith',
    # `try-except` nodes
    'Try', 'TryStar', 'ExceptHandler',
]
CompoundStatement = Literal[
    'func_defs', 'class_defs',
    'loops', 'conditionals', 'contexts', 'try_except',
]
ImportStatement = TypeVar('ImportStatement', ast.Import, ast.ImportFrom)
DefineStatement = TypeVar(
    'DefineStatement', ast.FunctionDef, ast.AsyncFunctionDef,
)
Node = TypeVar('Node', bound=ast.AST)
T = TypeVar('T')


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
        >>> import foo.bar  # doctest: +SKIP
        >>> profile.add_imported_function_or_module(  # doctest: +SKIP
        ...     foo.bar,
        ... )

    Args:
        modname (str):
            name of the imported module.

        profiler_name (str):
            name of the :py:class:`line_profiler.LineProfiler` object.

        attr (str):
            name of the method of the :py:class:`LineProfiler` object to
            call on the imported module.

Returns:
        expr (ast.Expr):
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

    >>> from foo.bar import *  # doctest: +SKIP
    >>> profile.add_star_import(  # doctest: +SKIP
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
        expr (ast.Expr):
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


def should_profile_regular_import(
    targets: Collection[str], modname: str,
) -> bool:
    """
    Check if either the parent module or submodule are in
    ``targets``
    """
    names = {modname, modname.rsplit('.', 1)[0]}
    return bool(names.intersection(targets))


def should_profile_star_import(
    targets: Collection[str], star_modname: str,
) -> bool:
    """
    Check if ``star_modname`` (should end in '.*') would match any of
    ``targets`` (should they actually exist)
    """
    assert star_modname.endswith('.*')
    modname = star_modname[:-2]
    if modname in targets:
        return True
    return any(
        target.rpartition('.')[0] == modname
        for target in targets if '.' in target
    )


def _return_value(value: T, /, *_, **__) -> T:
    return value


class _DuplicateImportChecker(Protocol):
    """
    Protocol for objects which helps with on-import profiling
    deduplication.
    """
    def should_profile_import(
        self, target: ImportTarget, context: Sequence[str | int], /,
    ) -> bool:
        ...

    def record_profiled_import(
        self, target: ImportTarget, context: Sequence[str | int], /,
    ) -> Any:
        ...


class _ConcreteDuplicateImportChecker:
    """
    Concrete implementation of :py:class:`~._DuplicateImportChecker`.
    """
    def __init__(self) -> None:
        self._profiled_imports: dict[
            tuple[str | int, ...], dict[int, list[ImportTarget]]
        ] = {}

    def should_profile_import(
        self, target: ImportTarget, context: Sequence[str | int],
    ) -> bool:
        # Note: the `index` shouldn't be needed since we're already
        # going through the import targets in order
        ctx, _ = self._check_context(context)
        if ctx not in self._profiled_imports:
            return True
        ctx_profiled_names = {
            imp.name
            for imports in self._profiled_imports[ctx].values()
            for imp in imports
        }
        return target.name not in ctx_profiled_names

    def record_profiled_import(
        self, target: ImportTarget, context: Sequence[str | int],
    ) -> None:
        ctx, index = self._check_context(context)
        (
            self._profiled_imports
            .setdefault(ctx, {})
            .setdefault(index, [])
            .append(target)
        )

    @staticmethod
    def _check_context(
        context: Sequence[str | int],
    ) -> tuple[tuple[str | int, ...], int]:
        *ctx, index = context
        if not isinstance(index, int):
            raise TypeError(
                f'context[-1] = {context[-1]!r}: expected an integer',
            )
        return tuple(ctx), index


class _CompoundNodeChecker:
    """
    Helper object for deciding on whether to look into a node.
    """
    _compound_node_types: ClassVar[set[CompoundNodeType]] = cast(
        set[CompoundNodeType], set(get_args(CompoundNodeType)),
    )

    def __init__(self, allowed: Mapping[CompoundNodeType, bool]) -> None:
        self.allowed = {
            node_type: allowed.get(node_type, False)
            for node_type in self._compound_node_types
        }

    def check(self, node: ast.AST) -> bool:
        node_type = cast(CompoundNodeType, type(node).__name__)
        return self.allowed.get(node_type, True)

    @classmethod
    def from_toggles(
        cls,
        *,
        conditionals: bool | None = None,
        try_except: bool | None = None,
        contexts: bool | None = None,
        loops: bool | None = None,
        func_defs: bool | None = None,
        class_defs: bool | None = None,
    ) -> Self:
        """
        Select the which of the compound node types (i.e. those than can
        contain nested code blocks, e.g. try-except statements) to
        visit.

        The default values are loaded from the
        `[tool.line_profiler.autoprofile.import_discovery]` table.
        """
        optional_toggles: dict[CompoundStatement, bool | None] = {
            'conditionals': conditionals,
            'try_except': try_except,
            'contexts': contexts,
            'loops': loops,
            'func_defs': func_defs,
            'class_defs': class_defs,
        }
        if any(t is None for t in optional_toggles.values()):
            defaults = cls._get_toggles()
            toggles: dict[CompoundStatement, bool] = {
                stmt: defaults[stmt] if toggle is None else toggle
                for stmt, toggle in optional_toggles.items()
            }
        else:
            toggles = cast(dict[CompoundStatement, bool], optional_toggles)

        allowed: set[CompoundNodeType] = {'Module', 'Interactive'}
        if toggles['conditionals']:
            allowed.update({'If', 'match_case'})
        if toggles['try_except']:
            allowed.update({'Try', 'TryStar', 'ExceptHandler'})
        if toggles['contexts']:
            allowed.update({'AsyncWith', 'With'})
        if toggles['loops']:
            allowed.update({'AsyncFor', 'For', 'While'})
        if toggles['func_defs']:
            allowed.update({'AsyncFunctionDef', 'FunctionDef'})
        if toggles['class_defs']:
            allowed.update({'ClassDef'})

        return cls({
            node_type: node_type in allowed
            for node_type in cls._compound_node_types
        })

    @classmethod
    def from_config(cls, *args, **kwargs) -> Self:
        return cls.from_toggles(**cls._get_toggles(*args, **kwargs))

    @staticmethod
    def _get_toggles(
        config: str | os.PathLike[str] | ConfigSource | None = None,
        find_nested_imports: Collection[CompoundStatement] | None = None,
    ) -> dict[CompoundStatement, bool]:
        if find_nested_imports is not None:
            return {
                cast(CompoundStatement, stmt): stmt in find_nested_imports
                for stmt in get_args(CompoundStatement)
            }
        if not isinstance(config, ConfigSource):
            config = ConfigSource.from_config(config)
        return cast(
            dict[CompoundStatement, bool],
            config
            .get_subconfig('autoprofile', 'import_discovery')
            .conf_dict,
        )


class ContextAwareVisitor(ast.NodeVisitor):
    """
    Helper class which keeps track of the chains of (1) nodes from the
    root node and (2) attribute and item accesses leading to the current
    one.

    The :py:meth:`._pre_visit_hook` (resp. :py:meth:`._post_visit_hook`)
    methods can be overridden to perform e.g. bookkeeping tasks before
    (resp. after) visiting a node and possibly transforming it.

    Node-type-specific visitor methods can make use of the following
    attributes:

    :py:attr:`._current_loc` (Sequence[int | str]):
        List of attribute/item accesses needed to reach the current node
        from the root node.

    :py:attr:`._node_stack` (Sequence[ast.AST]):
        Lineage of AST nodes starting from the root node and ending on
        the current node.

    Notes:
        Can be used as a mixin if transformation is required (see
        Examples).

    Examples:
        >>> import ast
        >>> import dataclasses
        >>> from collections.abc import Collection, Sequence

        Generic bookkeeping:

        >>> @dataclasses.dataclass
        ... class Record:
        ...     node: type[ast.AST]
        ...     ancestry: Sequence[str]
        ...     loc: Sequence[str | int]

        >>> class BaseRecorder(ContextAwareVisitor):
        ...     def __init__(
        ...         self, recorded: ast.AST | tuple[ast.AST, ...],
        ...     ) -> None:
        ...         super().__init__()
        ...         self.recorded = recorded
        ...         self.records: list[Record] = []
        ...
        ...     def _visit_hook(
        ...         self,
        ...         node: ast.AST,
        ...         ancestry: Sequence[ast.AST],
        ...         loc: Sequence[str | int],
        ...     ) -> None:
        ...         if not isinstance(node, self.recorded):
        ...             return
        ...         record = Record(
        ...             type(node),
        ...             [type(n).__name__ for n in ancestry],
        ...             list(loc),
        ...         )
        ...         self.records.append(record)

        >>> class PreRecorder(BaseRecorder):
        ...     _pre_visit_hook = BaseRecorder._visit_hook

        >>> class PostRecorder(BaseRecorder):
        ...     _post_visit_hook = BaseRecorder._visit_hook

        >>> class OnVisitRecorder(BaseRecorder):
        ...     def __init__(self) -> None:
        ...         super().__init__((ast.Import, ast.Return))
        ...
        ...     def _visit_handled(
        ...         self, node: ast.Import | ast.Return,
        ...     ) -> None:
        ...         self._visit_hook(
        ...             node, self._node_stack, self._current_loc,
        ...         )
        ...
        ...     visit_Import = visit_Return = _visit_handled

        >>> tree = ast.parse('''
        ... import foo
        ...
        ...
        ... def func(x: int) -> int:
        ...     return 1
        ... ''')

        >>> pre = PreRecorder((ast.Import, ast.Return))
        >>> post = PostRecorder((ast.Import, ast.Return))
        >>> on_visit = OnVisitRecorder()
        >>> expected_records = [
        ...     Record(ast.Import, ['Module', 'Import'], ['body', 0]),
        ...     Record(
        ...         ast.Return,
        ...         ['Module', 'FunctionDef', 'Return'],
        ...         ['body', 1, 'body', 0],
        ...     ),
        ... ]
        >>> pre.visit(tree)
        >>> post.visit(tree)
        >>> on_visit.visit(tree)
        >>> assert pre.records == expected_records, pre.records
        >>> assert post.records == expected_records, post.records
        >>> assert on_visit.records == expected_records, (
        ...     on_visit.records
        ... )

        Generic transforming:

        >>> import sys
        >>> from copy import deepcopy

        >>> @dataclasses.dataclass
        ... class PrintRecord:
        ...     node: str
        ...     ancestry: Sequence[str]
        ...     loc: Sequence[str | int]

        >>> class PrintStatementDoubler(
        ...     ContextAwareVisitor, ast.NodeTransformer,
        ... ):
        ...     '''
        ...     If the print statement has a single arg and it is in
        ...     ``drop_these``, remove the print;
        ...     otherwise, print it twice.
        ...     '''
        ...     def __init__(
        ...         self,
        ...         drop_these: Collection[
        ...             int | float | str | bytes | bool | None
        ...         ] = (),
        ...     ) -> None:
        ...         super().__init__()
        ...         self.before: list[PrintRecord] = []
        ...         self.after: list[PrintRecord] = []
        ...         self._dropped: set[
        ...             int | float | str | bytes | bool | None
        ...         ] = set(drop_these)
        ...
        ...     def visit_Expr(
        ...         self, expr: ast.Expr,
        ...     ) -> ast.Expr | list[ast.Expr] | None:
        ...         if not isinstance(expr.value, ast.Call):
        ...             return expr
        ...         call: ast.Call = expr.value
        ...         if not isinstance(call.func, ast.Name):
        ...             return expr
        ...         name: ast.Name = call.func
        ...         if name.id != 'print':
        ...             return expr
        ...
        ...         try:
        ...             arg, = call.args
        ...         except ValueError:  # Not single arg
        ...             return [expr, deepcopy(expr)]
        ...         if not isinstance(arg, ast.Constant):
        ...             return [expr, deepcopy(expr)]
        ...         const = arg.value
        ...         if const in self._dropped:  # Drop this print stmt
        ...             return None
        ...         return [expr, deepcopy(expr)]
        ...
        ...     @staticmethod
        ...     def _visit_hook(
        ...         node: ast.AST,
        ...         ancestry: Sequence[ast.AST],
        ...         loc: Sequence[str | int],
        ...         records: list[PrintRecord],
        ...     ) -> None:
        ...         if not isinstance(node, ast.Expr):
        ...             return
        ...         if not isinstance(node.value, ast.Call):
        ...             return
        ...         if not isinstance(node.value.func, ast.Name):
        ...             return
        ...         name: ast.Name = node.value.func
        ...         if name.id != 'print':
        ...             return
        ...         record = PrintRecord(
        ...             ast.unparse(node),
        ...             [type(n).__name__ for n in ancestry],
        ...             list(loc),
        ...         )
        ...         records.append(record)
        ...
        ...     def _pre_visit_hook(
        ...         self,
        ...         node: ast.AST,
        ...         ancestry: Sequence[ast.AST],
        ...         loc: Sequence[str | int],
        ...     ) -> None:
        ...         self._visit_hook(node, ancestry, loc, self.before)
        ...
        ...     def _post_visit_hook(
        ...         self,
        ...         node: ast.AST,
        ...         ancestry: Sequence[ast.AST],
        ...         loc: Sequence[str | int],
        ...     ) -> None:
        ...         self._visit_hook(node, ancestry, loc, self.after)

        >>> tree = ast.parse('''
        ... import sys as _sys
        ...
        ...
        ... print(1)
        ... print(2)
        ... str(3)
        ... print(4)
        ...
        ... for s in 'foo', 'bar', 'baz':
        ...     print(s)
        ... ''')
        >>> transformer = PrintStatementDoubler(
        ...     [2],  # Remove the 2nd print, double the others
        ... )
        >>> namespace = {}
        >>> exec(
        ...     compile(transformer.visit(tree), '<string>', 'exec'),
        ...     namespace,
        ... )
        1
        1
        4
        4
        foo
        foo
        bar
        bar
        baz
        baz
        >>> assert namespace['_sys'] is sys
        >>> assert namespace['s'] == 'baz'

        >>> assert transformer.before == [
        ...     PrintRecord(  # This is doubled
        ...         'print(1)', ['Module', 'Expr'], ['body', 1],
        ...     ),
        ...     PrintRecord(  # This is pruned
        ...         'print(2)', ['Module', 'Expr'], ['body', 3],
        ...     ),
        ...     PrintRecord(  # This is doubled
        ...         'print(4)', ['Module', 'Expr'], ['body', 4],
        ...     ),
        ...     PrintRecord(  # This is doubled
        ...         'print(s)',
        ...         ['Module', 'For', 'Expr'],
        ...         ['body', 6, 'body', 0],
        ...     ),
        ... ], transformer.before
        >>> assert transformer.after == [
        ...     PrintRecord(
        ...         'print(1)', ['Module', 'Expr'], ['body', 1],
        ...     ),
        ...     PrintRecord(  # This is new from doubling
        ...         'print(1)', ['Module', 'Expr'], ['body', 2],
        ...     ),
        ...     PrintRecord(
        ...         'print(4)', ['Module', 'Expr'], ['body', 4],
        ...     ),
        ...     PrintRecord(  # This is new from doubling
        ...         'print(4)', ['Module', 'Expr'], ['body', 5],
        ...     ),
        ...     PrintRecord(
        ...         'print(s)',
        ...         ['Module', 'For', 'Expr'],
        ...         ['body', 6, 'body', 0],
        ...     ),
        ...     PrintRecord(  # This is new from doubling
        ...         'print(s)',
        ...         ['Module', 'For', 'Expr'],
        ...         ['body', 6, 'body', 1],
        ...     ),
        ... ], transformer.after
    """
    def __init__(self, *_, **__) -> None:
        self._node_stack: list[ast.AST] = []
        self._current_loc: list[str | int] = []
        self._is_transformer = isinstance(self, ast.NodeTransformer)

    def _pre_visit_hook(
        self,
        node: ast.AST,
        /,
        ancestry: Sequence[ast.AST],
        loc: Sequence[str | int],
    ) -> Any:
        """
        Hook method to be executed BEFORE the
        :py:class:`ast.NodeVisitor` calls the node-type specific visit
        methods or :py:meth:`ast.NodeVisitor.generic_visit`.

        Args:
            node (ast.AST):
                The current node BEFORE any potential transformation.

            ancestry (Sequence[ast.AST]):
                The ancestry of the current node, starting from the root
                node and ending on itself.

            loc (Sequence[str | int]):
                The sequence of attribute/item accesses needed to reach
                the current node from the root node.

        Notes:
            - This base-class implementation is a no-op.

            - If the object is a :py:class:`ast.NodeTransformer` and the
              the ``node`` has been replaced, this is only called on the
              original node and NOT the new replacement nodes.
        """

    def _post_visit_hook(
        self,
        node: ast.AST,
        /,
        ancestry: Sequence[ast.AST],
        loc: Sequence[str | int],
    ) -> Any:
        """
        Hook method to be executed AFTER the
        :py:class:`ast.NodeVisitor` has called the node-type specific
        visit methods or :py:meth:`ast.NodeVisitor.generic_visit`.

        Args:
            node (ast.AST):
                The current node AFTER any potential transformation.

            ancestry (Sequence[ast.AST]):
                The ancestry of the current node, starting from the root
                node and ending on itself.

            loc (Sequence[str | int]):
                The sequence of attribute/item accesses needed to reach
                the current node from the root node.

        Notes:
            - This base-class implementation is a no-op.

            - If the object is a :py:class:`ast.NodeTransformer` and the
              the node has been replaced, this is called on the each of
              the new replacement ``node``s and NOT the original node.
        """

    def generic_visit(self, node: ast.AST) -> Any:
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                self._visit_generic_child(node, field, value)
            elif isinstance(value, MutableSequence):
                if not all(isinstance(item, ast.AST) for item in value):
                    continue
                # Compound node
                self._visit_generic_children(
                    node, field, cast(MutableSequence[ast.AST], value),
                )
        return node if self._is_transformer else None

    def visit(self, node: ast.AST) -> Any:
        self._node_stack.append(node)
        try:
            self._pre_visit_hook(
                node, self._node_stack, self._current_loc,
            )
            result = super().visit(node)
            if self._is_transformer:
                body = (
                    self._current_loc
                    and isinstance(self._current_loc[-1], int)
                )
                *ancestors, _ = self._node_stack
                result = cast(ast.AST | Sequence[ast.AST] | None, result)
                if body:  # Calculate new locations
                    if result is None:
                        replacements: Sequence[ast.AST] = []
                    elif isinstance(result, ast.AST):
                        replacements = [result]
                    else:
                        replacements = result
                    *loc, index = self._current_loc
                    for offset, node in enumerate(replacements):
                        self._post_visit_hook(
                            node,
                            ancestors + [node],
                            loc + [cast(int, index) + offset],
                        )
                elif result is not None:
                    node = cast(ast.AST, result)
                    self._post_visit_hook(
                        node, ancestors + [node], self._current_loc,
                    )
            else:
                self._post_visit_hook(
                    node, self._node_stack, self._current_loc,
                )
        finally:
            self._node_stack.pop()
        return result

    def _visit_generic_child(
        self, node: ast.AST, field: str, child: ast.AST,
    ) -> None:
        self._current_loc.append(field)
        try:
            replacement = self.visit(child)
            if self._is_transformer:
                if replacement is None or isinstance(replacement, ast.AST):
                    setattr(node, field, replacement)
                else:
                    raise RuntimeError(
                        f'node = {node!r}: invalid field `.{field}` '
                        f'replacement ({child!r} -> {replacement!r})'
                    )
        finally:
            self._current_loc.pop()

    def _visit_generic_children(
        self, node: ast.AST, field: str, children: MutableSequence[ast.AST],
    ) -> None:
        self._current_loc.append(field)
        try:
            new_children: list[ast.AST] = []
            offset = 0
            for i, item in enumerate(children):
                self._current_loc.append(i + offset)
                try:
                    if self._is_transformer:
                        replacement = cast(
                            ast.AST | Sequence[ast.AST] | None,
                            self.visit(item),
                        )
                        if replacement is None:
                            replacements: Sequence[ast.AST] = []
                        elif isinstance(replacement, ast.AST):
                            replacements = [replacement]
                        else:
                            replacements = replacement
                        offset += len(replacements) - 1
                        new_children.extend(replacements)
                    else:
                        self.visit(item)
                finally:
                    self._current_loc.pop()
            if self._is_transformer:
                children[:] = new_children
        finally:
            self._current_loc.pop()


class SinglePassTransformer(ContextAwareVisitor, ast.NodeTransformer):
    """
    Transform an abstract syntax tree adding profiling to all of its
    objects, by:

    - Decorating locally-defined functions/methods that are not already
      decorated with the profiler.

    - Adding profiler method calls (see
      :py:func:`line_profiler.autoprofile.line_profiler_utils\
.add_imported_function_or_module`
      and
      :py:func:`line_profiler.autoprofile.line_profiler_utils\
.add_star_import`)
      immediately after imports to profile the appropriate import
      targets.

    Args:
        profiler_name (str):
            Name under which the profiler instance should be found
            (default: ``'profile'``).

        prof_func_defs (bool | None):
            Whether to add a ``@profile`` decorator to all the
            locally-defined functions and methods that are not already
            decorated with the profiler.
            If :py:const:`None`, it is resolved from the active config's
            ``[tool.line_profiler.autoprofile]::prof_func_defs``.

        prof_explicit_imports (Collection[str] | bool | None):
            Which explicit-import targets (``import x.y`` or
            ``from x import y``) to profile;
            can also be a global boolean toggle.
            If :py:const:`None`, it is resolved from the active config's
            ``[tool.line_profiler.autoprofile]::prof_explicit_imports``.

        prof_star_imports (Collection[str] | bool | None):
            Which star-import targets (``from x import *``) to profile;
            can also be a global boolean toggle.
            If :py:const:`None`, it is resolved from the active config's
            ``[tool.line_profiler.autoprofile]::prof_star_imports``.

        prof_imports_in (Mapping[Literal[\
'Module', 'Interactive',\
'FunctionDef', 'AsyncFunctionDef',\
'ClassDef',\
'For', 'AsyncFor', 'While',\
'If', 'match_case',\
'With', 'AsyncWith',\
'Try', 'TryStar', 'ExceptHandler',\
], bool]):
            Which compound-statement types to descend into and look for
            imports;
            if :py:const:`None`, it is resolved from the active config's
            ``[tool.line_profiler.autoprofile]::import_discovery``.
            This can be used to control import-statement reachability;
            e.g. to avoid repeatedly calling the aforementioned
            ``add_imported_function_or_module()`` and
            ``add_star_import()`` within a function body.

        config (str | os.PathLike[str] | None):
            Source from which config options are loaded.
    """
    def __init__(
        self,
        profiler_name: str = 'profile',
        *,
        config: str | os.PathLike[str] | None = None,
        prof_func_defs: bool | None = None,
        prof_explicit_imports: Collection[str] | bool | None = None,
        prof_star_imports: Collection[str] | bool | None = None,
        prof_imports_in: Mapping[CompoundNodeType, bool] | None = None,
    ) -> None:
        super().__init__()

        conf = ConfigSource.from_config(config).get_subconfig('autoprofile')
        if prof_func_defs is None:
            prof_func_defs = conf.conf_dict['prof_func_defs']
        if prof_explicit_imports is None:
            prof_explicit_imports = conf.conf_dict['prof_explicit_imports']
        if prof_star_imports is None:
            prof_star_imports = conf.conf_dict['prof_star_imports']
        if prof_imports_in is None:
            prof_imports_in = _CompoundNodeChecker.from_config(config).allowed

        try:
            prof_explicit_imports = frozenset(cast(
                'Collection[str]', prof_explicit_imports,
            ))
        except TypeError:
            prof_explicit_imports = bool(prof_explicit_imports)
        try:
            prof_star_imports = frozenset(cast(
                'Collection[str]', prof_star_imports,
            ))
        except TypeError:
            prof_star_imports = bool(prof_star_imports)

        self._profiler_name = profiler_name
        self._prof_func_defs = bool(prof_func_defs)
        self._prof_explicit_imports = prof_explicit_imports
        self._prof_star_imports = prof_star_imports

        self._import_ancestry_checker = _CompoundNodeChecker(prof_imports_in)
        self._duplicate_import_checker: _DuplicateImportChecker
        self._duplicate_import_checker = _ConcreteDuplicateImportChecker()

    def _visit_import(
        self,
        node: ImportStatement,
        get_import_targets: Callable[
            [int, ImportStatement], Sequence[ImportTarget]
        ],
    ) -> ImportStatement | list[ImportStatement | ast.Expr]:
        # Should we be looking at this import statement?
        check_ancestor = self._import_ancestry_checker.check
        if not all(check_ancestor(node) for node in self._node_stack):
            return node

        *_, index = self._current_loc
        assert isinstance(index, int)
        result: list[ImportStatement | ast.Expr] = [node]
        dup_checker = self._duplicate_import_checker

        for target in get_import_targets(index, node):
            # Have we already profiled this import?
            if not dup_checker.should_profile_import(
                target, self._current_loc,
            ):
                continue

            maybe_expr = self._handle_new_import_target(target)
            if maybe_expr is not None:
                result.append(maybe_expr)
                dup_checker.record_profiled_import(target, self._current_loc)
        return result

    def _handle_new_import_target(
        self, target: ImportTarget,
    ) -> ast.Expr | None:
        """
        Returns:
            maybe_expr (ast.Expr | None):
                An expression responsible for setting up profiling of
                the import target, where appropriate.
        """
        if target.resolved_name is None:  # Star import
            create_node, should_profile = self._star_import_handlers
        else:  # Explicit import
            create_node, should_profile = self._expl_import_handlers

        if should_profile(target.name):
            return create_node(target.name)
        return None

    def visit_Import(
        self, node: ast.Import,
    ) -> ast.Import | list[ast.Import | ast.Expr]:
        """
        Where appropriate (as determined by ``prof_explicit_imports``
        and ``prof_imports_in``), add nodes that profile objects
        imported using the ``import foo`` syntax.

        Args:
            node (ast.Import):
                import in the AST

        Returns:
            nodes (ast.Import | list[ast.Import | ast.Expr]):
                ``node`` with subsequent profiling expressions inserted
                for on-import profiling of the appropriate import
                targets
        """
        return self._visit_import(node, ImportTarget._from_import_node)

    def visit_ImportFrom(
        self, node: ast.ImportFrom
    ) -> ast.ImportFrom | list[ast.ImportFrom | ast.Expr]:
        """
        Where appropriate (as determined by ``prof_explicit_imports``,
        ``prof_star_imports``, and ``prof_imports_in``), add nodes that
        profile objects imported using the ``from foo import bar``
        syntax.

        Args:
            node (ast.ImportFrom):
                import in the AST

        Returns:
            nodes (ast.ImportFrom | list[ast.ImportFrom | ast.Expr]):
                ``node`` with subsequent profiling expressions inserted
                for on-import profiling of the appropriate import
                targets
        """
        return self._visit_import(node, ImportTarget._from_import_from_node)

    def _visit_func_def(self, node: DefineStatement) -> DefineStatement:
        """
        Where appropriate (as determined by ``prof_func_defs``),
        decorate functions/methods with the profiler (at
        ``profiler_name``) if they aren't already decorated therewith.

        Args:
            node (ast.FunctionDef | ast.AsyncFunctionDef):
                function/method in the AST

        Returns:
            node (ast.FunctionDef | ast.AsyncFunctionDef):
                function/method with profiling decorator

        Notes:
            The added decorator is appended to decorator list to avoid
            conflicts with other decorators e.g. ``@staticmethod``.
        """
        if self._prof_func_defs:
            decor_ids: set[str] = {
                decor.id for decor in node.decorator_list
                if isinstance(decor, ast.Name)
            }
            if self._profiler_name not in decor_ids:
                node.decorator_list.append(
                    ast.Name(id=self._profiler_name, ctx=ast.Load())
                )
        self.generic_visit(node)
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = _visit_func_def

    @staticmethod
    def _create_star_import_node(
        target_name: str, /, *args, **kwargs
    ) -> ast.Expr:
        """
        Wrapper around :py:func:`ast_create_star_import_node` which
        directly takes the :py:attr:`ImportTarget.name` of a
        star-import.
        """
        if target_name.endswith('.*'):
            target_name = target_name[:-2]
        return ast_create_star_import_node(target_name, *args, **kwargs)

    @cached_property
    def _expl_import_handlers(self) -> tuple[
        Callable[[str], ast.Expr], Callable[[str], bool],
    ]:
        """
        Returns:
            (create_node, should_profile) \
(tuple[Callable[[str], ast.Expr], Callable[[str], bool]]):
                create_node()
                    Callable taking the :py:attr:`ImportTarget.name` and
                    returning an AST node responsible for setting up
                    profiling therefrom
                should_profile()
                    Callable taking the :py:attr:`ImportTarget.name` and
                    returning whether it should be profiled
        """
        profiled_imports = self._prof_explicit_imports
        should_profile: Callable[[str], bool]
        if profiled_imports in (True, False):
            should_profile = partial(_return_value, bool(profiled_imports))
        else:
            should_profile = partial(
                should_profile_regular_import,
                cast(Collection[str], profiled_imports),
            )
        create_node: Callable[[str], ast.Expr] = partial(
            ast_create_profile_node, profiler_name=self._profiler_name,
        )
        return create_node, should_profile

    @cached_property
    def _star_import_handlers(self) -> tuple[
        Callable[[str], ast.Expr], Callable[[str], bool],
    ]:
        """
        Returns:
            See :py:attr:`._expl_import_handlers`.
        """
        profiled_imports = self._prof_star_imports
        should_profile: Callable[[str], bool]
        if profiled_imports in (True, False):
            star_import_targets: Collection[str] | None = None
            should_profile = partial(_return_value, bool(profiled_imports))
        else:
            star_import_targets = cast(Collection[str], profiled_imports)
            should_profile = partial(
                should_profile_star_import, star_import_targets,
            )
        create_node: Callable[[str], ast.Expr] = partial(
            self._create_star_import_node,
            targets=star_import_targets,
            profiler_name=self._profiler_name,
        )
        return create_node, should_profile
