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
from collections.abc import Collection, Mapping, MutableSequence, Sequence
from typing import Any, ClassVar, Literal, Self, TypeVar, cast, get_args

from ..toml_config import ConfigSource
from ._import_targets import ImportTarget


__all__ = (
    'ContextAwareVisitor',
    'ast_create_profile_node', 'ast_create_star_import_node',
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
Node = TypeVar('Node', bound=ast.AST)


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


class _CompoundNodeChecker:
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
        ...     pre_visit_hook = BaseRecorder._visit_hook

        >>> class PostRecorder(BaseRecorder):
        ...     post_visit_hook = BaseRecorder._visit_hook

        >>> tree = ast.parse('''
        ... import foo
        ...
        ...
        ... def func(x: int) -> int:
        ...     return 1
        ... ''')

        >>> pre = PreRecorder((ast.Import, ast.Return))
        >>> post = PostRecorder((ast.Import, ast.Return))
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
        >>> assert pre.records == expected_records, pre.records
        >>> assert post.records == expected_records, post.records

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
        ...     def pre_visit_hook(
        ...         self,
        ...         node: ast.AST,
        ...         ancestry: Sequence[ast.AST],
        ...         loc: Sequence[str | int],
        ...     ) -> None:
        ...         self._visit_hook(node, ancestry, loc, self.before)
        ...
        ...     def post_visit_hook(
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

    def pre_visit_hook(
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

    def post_visit_hook(
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
        self._node_stack.append(node)
        try:
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
        finally:
            self._node_stack.pop()
        return node if self._is_transformer else None

    def visit(self, node: ast.AST) -> Any:
        self.pre_visit_hook(node, self._node_stack + [node], self._current_loc)
        result = super().visit(node)
        if self._is_transformer:
            body = self._current_loc and isinstance(self._current_loc[-1], int)
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
                    self.post_visit_hook(
                        node,
                        self._node_stack + [node],
                        loc + [cast(int, index) + offset],
                    )
            elif result is not None:
                self.post_visit_hook(
                    cast(ast.AST, result),
                    self._node_stack + [node],
                    self._current_loc,
                )
        else:
            self.post_visit_hook(
                node, self._node_stack + [node], self._current_loc,
            )
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
