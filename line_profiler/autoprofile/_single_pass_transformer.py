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
import sys
from collections.abc import (
    Callable, Collection, Mapping, MutableSequence, Sequence, Set,
)
from dataclasses import dataclass
from functools import cached_property, partial
from operator import attrgetter
from typing import (
    Any, ClassVar, Literal, ParamSpec, Protocol, Self, TypeVar, cast, get_args,
)

from ..toml_config import ConfigSource
from ._import_targets import _DROPPED_STAR_IMPORTS_MSG_TEMPLATE, ImportTarget
from .util_static import (
    modname_to_modpath, modpath_to_modname, package_modpaths,
)


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
PS = ParamSpec('PS')
T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')


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


def _chain_callables(
    func1: Callable[PS, T1], func2: Callable[[T1], T2],
) -> Callable[PS, T2]:
    def chained(*args: PS.args, **kwargs: PS.kwargs) -> T2:
        return func2(func1(*args, **kwargs))

    return chained


def _get_conf_table(
    config: ConfigSource,
    table: Sequence[str],
    _seen_prefixes: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """
    Examples:
        Normal use:

        >>> default_config = ConfigSource.from_default()
        >>> assert _get_conf_table(
        ...     default_config,
        ...     ['tool', 'line_profiler', 'autoprofile'],
        ... ) == default_config.get_subconfig('autoprofile').conf_dict

        If we're beginning from a different subtable, the target
        subtable can be recovered from reloading the config from source:

        >>> subconfig = default_config.get_subconfig('kernprof')
        >>> assert _get_conf_table(
        ...     subconfig,
        ...     ['tool', 'line_profiler', 'autoprofile'],
        ... ) == default_config.get_subconfig('autoprofile').conf_dict

        However, reloading doesn't necessarily help sometimes:

        >>> _get_conf_table(  # doctest: +NORMALIZE_WHITESPACE
        ...     default_config, ['tool', 'something_else'],
        ... )
        Traceback (most recent call last):
          ...
        RuntimeError: Cannot recover table=['tool', 'something_else']
        from config=ConfigSource(...)
    """
    target_keys = tuple(table)
    conf_keys = tuple(config.subtable)
    if _seen_prefixes is None:
        _seen_prefixes = set()

    if conf_keys in _seen_prefixes:
        # We've already tried reloading (see below); didn't work
        raise RuntimeError(f'Cannot recover {table=!r} from {config=!r}')

    if target_keys[:len(conf_keys)] == conf_keys:
        subtable = target_keys[len(conf_keys):]
        return dict(config.get_subconfig(*subtable).conf_dict)

    # Maybe we're in a subtable and have to go up by reloading the
    # config from source...
    _seen_prefixes.add(conf_keys)
    config = config.from_config(config.path)
    return _get_conf_table(config, table, _seen_prefixes)


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
    _toggles_to_node_types: ClassVar[
        dict[CompoundStatement, set[CompoundNodeType]]
    ] = {
        'conditionals': {'If', 'match_case'},
        'try_except': {'Try', 'TryStar', 'ExceptHandler'},
        'contexts': {'AsyncWith', 'With'},
        'loops': {'AsyncFor', 'For', 'While'},
        'func_defs': {'AsyncFunctionDef', 'FunctionDef'},
        'class_defs': {'ClassDef'},
    }

    def __init__(self, allowed: Mapping[CompoundNodeType, bool]) -> None:
        self.allowed = {
            node_type: allowed.get(node_type, False)
            for node_type in self._compound_node_types
        }

    def check(self, node: ast.AST) -> bool:
        node_type = cast(CompoundNodeType, type(node).__name__)
        return self.allowed.get(node_type, True)

    @classmethod
    def _normalize_toggles_to_node_types(
        cls, toggles: Collection[CompoundStatement],
    ) -> set[CompoundNodeType]:
        normalized: set[CompoundNodeType] = {'Module', 'Interactive'}
        toggles = set(toggles)
        for toggle, node_types in cls._toggles_to_node_types.items():
            if toggle in toggles:
                normalized.update(node_types)
        return normalized

    @classmethod
    def _normalize_mixed_toggle_mapping(
        cls, toggles: Mapping[CompoundStatement | CompoundNodeType, bool],
    ) -> dict[CompoundNodeType, bool]:
        result: dict[CompoundNodeType, bool] = dict.fromkeys(
            cls._compound_node_types, False,
        )
        result['Module'] = result['Interactive'] = True
        # Groups of node types
        for toggle, node_types in cls._toggles_to_node_types.items():
            if toggle in toggles:
                result.update(dict.fromkeys(node_types, toggles[toggle]))
        # Specific node types
        for node_type in cls._compound_node_types:
            if node_type in toggles:
                result[node_type] = toggles[node_type]
        return result

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

        allowed = cls._normalize_toggles_to_node_types({
            toggle for toggle, value in toggles.items() if value
        })
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
            _get_conf_table(
                config,
                ['tool', 'line_profiler', 'autoprofile', 'import_discovery'],
            ),
        )


@dataclass
class _ProfModHelper:
    """
    Helper object handling the ``--prof-mod`` supplied by
    :py:mod:`kernprof`.
    """
    script_file: str | os.PathLike[str]
    prof_mod: Sequence[str]

    def to_dotted_paths(
        self, exclude_script_file: bool = False,
    ) -> list[str]:
        """
        Returns:
            dotted_paths (list[str]):
                Dotted paths to the profiling targets.

        Notes:
            Refactored from the old
            `ProfmodExtractor._get_modnames_to_profile_from_prof_mod()`
            contributed by ta946.
        """
        script_dir = os.path.realpath(os.path.dirname(self.script_file))
        script_file_realpath = os.path.realpath(self.script_file)

        dotted_paths: dict[str, None] = {}
        add_dotted_path = dotted_paths.setdefault
        for target in set(self.prof_mod):
            if (
                exclude_script_file
                and script_file_realpath == os.path.realpath(target)
            ):
                continue
            # Try to normalize target names to file paths (if modules),
            # extending `sys.path` to allow `modname_to_modpath()` to
            # resolve "modules" local to the script's directory
            modpath = modname_to_modpath(
                target, sys_path=[script_dir, *sys.path],
            )
            if modpath is None:
                # Can't convert info a file path
                # -> check if it's already one
                if not os.path.exists(target):
                    # *May* be some module that will otherwise become
                    # available, so let it be
                    if all(chunk.isidentifier() for chunk in target.split('.')):
                        add_dotted_path(target)
                    continue
                # The target is probably an installed package
                modpath = target

            # Convert file paths back to dotted paths
            try:
                modname = modpath_to_modname(modpath)
            except ValueError:
                continue
            add_dotted_path(modname)

            # Recursively fetch all subpackages and submodules and also add
            # their dotted paths
            for submod_path in package_modpaths(modpath):
                submod_name = modpath_to_modname(submod_path)
                add_dotted_path(submod_name)

        return list(dotted_paths)

    def script_file_is_included(
        self, match_mode: Literal['filename', 'module'] = 'filename',
    ) -> bool:
        """
        Returns:
            script_file_is_included (bool):
                Whether :py:attr:`.script_file` should be counted as
                being present in :py:attr:`.prof_mod`.

        Notes:
            Refactored from the old
            `AstTreeProfiler._check_profile_full_script()` contributed
            by ta946.
        """
        rp = os.path.realpath
        prof_mod = set(self.prof_mod)

        if match_mode == 'filename':
            script_file_realpath = rp(self.script_file)
            real_paths = {rp(target) for target in prof_mod}
            return script_file_realpath in real_paths

        if match_mode == 'module':
            paths_to_check = {rp(self.script_file)}
            if os.path.basename(self.script_file) == '__main__.py':
                # If the `-m` entry point of a package, check for the
                # package dir itself
                paths_to_check.add(rp(os.path.dirname(self.script_file)))
            paths_to_profile = {rp(target) for target in prof_mod}
            paths_to_profile.update(
                rp(path)
                for target in prof_mod if (path := modname_to_modpath(target))
            )
            return bool(paths_to_check & paths_to_profile)

        raise ValueError(f"{match_mode=!r}: expected 'filename' or 'module'")


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

        config (str | os.PathLike[str] | ConfigSource | None):
            Source from which config options are loaded.

        module (str | None):
            If provided, assume relative imports to be relative to this
            module.

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
'func_defs', 'class_defs',\
'loops', 'conditionals', 'contexts', 'try_except',\
], bool] | None):
            Which compound-statement types to descend into and look for
            imports;
            if :py:const:`None`, it is resolved from the active config's
            ``[tool.line_profiler.autoprofile]::import_discovery``.
            This can be used to control import-statement reachability;
            e.g. to avoid repeatedly calling the aforementioned
            ``add_imported_function_or_module()`` and
            ``add_star_import()`` within a function body.
    """
    def __init__(
        self,
        profiler_name: str = 'profile',
        *,
        config: str | os.PathLike[str] | ConfigSource | None = None,
        module: str | None = None,
        prof_func_defs: bool | None = None,
        prof_explicit_imports: Collection[str] | bool | None = None,
        prof_star_imports: Collection[str] | bool | None = None,
        prof_imports_in: (
            Mapping[CompoundNodeType, bool]
            | Mapping[CompoundStatement, bool]
            | Mapping[CompoundNodeType | CompoundStatement, bool]
            | None
        ) = None,
    ) -> None:
        super().__init__()

        if not isinstance(config, ConfigSource):
            config = ConfigSource.from_config(config)
        conf = _get_conf_table(
            config, ['tool', 'line_profiler', 'autoprofile'],
        )
        if prof_func_defs is None:
            prof_func_defs = conf['prof_func_defs']
        if prof_explicit_imports is None:
            prof_explicit_imports = conf['prof_explicit_imports']
        if prof_star_imports is None:
            prof_star_imports = conf['prof_star_imports']
        if prof_imports_in is None:
            node_type_checker = _CompoundNodeChecker.from_config(config)
        else:
            pii = _CompoundNodeChecker._normalize_mixed_toggle_mapping(cast(
                Mapping[CompoundNodeType | CompoundStatement, bool],
                prof_imports_in,
            ))
            node_type_checker = _CompoundNodeChecker(pii)

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

        self._module = module
        self._profiler_name = profiler_name
        self._prof_func_defs = bool(prof_func_defs)
        self._prof_explicit_imports = prof_explicit_imports
        self._prof_star_imports = prof_star_imports

        self._import_ancestry_checker = node_type_checker
        self._duplicate_import_checker: _DuplicateImportChecker
        self._duplicate_import_checker = _ConcreteDuplicateImportChecker()

        self._dropped_imports: set[ImportTarget] = set()

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
                # In case things go south during on-import profiling,
                # make sure that the errors are attributed to the import
                # line
                result.append(ast.copy_location(maybe_expr, node))
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

        if should_profile(target):
            return create_node(target)
        else:
            self._dropped_imports.add(target)
        return None

    @staticmethod
    def _resolve_importfrom_module(node: ast.ImportFrom, module: str) -> str:
        r"""
        Resolve the full path of a relative import.

        Args:
            node (ast.ImportFrom)
                :py:class:`ast.ImportFrom` node
            module (str)
                Full dotted path relative to which the import is to occur

        Return:
            modname (str)
                Full path of the module from which the names are to be
                imported

        Example:
            >>> import ast
            >>> import functools
            >>> import textwrap
            >>>
            >>>
            >>> abs_import, *rel_imports = ast.parse(textwrap.dedent('''
            ... from a import b
            ... from . import b
            ... from .. import b
            ... from .baz import b
            ... from ..baz import b
            ... '''.strip('\n'))).body
            >>>
            >>>
            >>> get_module = functools.partial(
            ...     SinglePassTransformer._resolve_importfrom_module,
            ...     module='foo.bar.foobar',
            ... )
            >>> assert get_module(abs_import) == 'a'
            >>> assert get_module(rel_imports[0]) == 'foo.bar'
            >>> assert get_module(rel_imports[1]) == 'foo'
            >>> assert get_module(rel_imports[2]) == 'foo.bar.baz'
            >>> assert get_module(rel_imports[3]) == 'foo.baz'
        """
        level = node.level
        if not level:
            return node.module or ''
        chunks = module.split('.')[:-level]
        if node.module:
            chunks.append(node.module)
        return '.'.join(chunks)

    @classmethod
    def _consolidate_relative_import(
        cls, node: ast.ImportFrom, module: str | None = None,
    ) -> ast.ImportFrom:
        if not node.level:  # Absoluteimport
            return node
        if not module:  # nocover
            raise RuntimeError(
                f'{node=!r}, {module=!r}: '
                'cannot resolve relative imports with no ``module`` provided',
            )
        module = cls._resolve_importfrom_module(node, module)
        new_node = ast.ImportFrom(module=module, names=node.names, level=0)
        return ast.copy_location(new_node, node)

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
        self, node: ast.ImportFrom,
    ) -> ast.ImportFrom | list[ast.ImportFrom | ast.Expr]:
        """
        Where appropriate (as determined by ``prof_explicit_imports``,
        ``prof_star_imports``, and ``prof_imports_in``), add nodes that
        profile objects imported using the ``from foo import bar``
        syntax. If ``module`` has been provided, relative imports are
        resolved to absolute imports therewith.

        Args:
            node (ast.ImportFrom):
                import in the AST

        Returns:
            nodes (ast.ImportFrom | list[ast.ImportFrom | ast.Expr]):
                ``node`` with subsequent profiling expressions inserted
                for on-import profiling of the appropriate import
                targets
        """
        node = self._consolidate_relative_import(node, self._module)
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

    def _transform(
        self,
        node: ast.Module,
        filename: os.PathLike[str] | str | None = None,
        warn_dropped_star_imports: (
            bool
            | Callable[[Set[ImportTarget]], Collection[ImportTarget]]
        ) = False,
        *,
        stacklevel: int = 1,
        **warning_kwargs,
    ) -> ast.Module:
        """
        Wrapper around :py:meth:`.visit` with extra bookkeeping.

        Args:
            node (ast.Module):
                AST module node

            filename (os.PathLike[str] | str | None):
                Optional filename to be used in error/warning messages

            warn_dropped_star_imports \
(bool | Callable[[Set[ImportTarget]], Collection[ImportTarget]]):
                If a boolean, whether to report dropped star-imports;
                if a callable taking a set of :py:class:`ImportTarget`s
                and returning a collection thereof, it is used to filter
                the reported dropped star-imports

            stacklevel (int):
            **warning_kwargs:
                Optional keyword arguments to pass to
                :py:func:`warnings.warn` should there be any dropped
                star-imports to be reported

        Returns:
            node (ast.Module):
                Input module node

        Notes:
            The default ``stacklevel`` means that any warning raised
            will be attributed to where this method is called.
        """
        filter_dropped: Callable[[Set[ImportTarget]], Collection[ImportTarget]]
        if callable(warn_dropped_star_imports):
            filter_dropped = warn_dropped_star_imports
        elif warn_dropped_star_imports:
            filter_dropped = set
        else:
            filter_dropped = partial(_return_value, ())

        if filename is None:
            filename = '???'

        try:
            return cast(ast.Module, self.visit(node))
        finally:
            dropped_star_imports = filter_dropped({
                target for target in self._dropped_imports
                if target.resolved_name is None
            })
            ImportTarget._check_and_warn_dropped_imports(
                dropped_star_imports,
                _DROPPED_STAR_IMPORTS_MSG_TEMPLATE.format(
                    action='profiled',
                    argname='profile_star_imports',
                ),
                filename,
                # Warning level relative to this method
                stacklevel=stacklevel + 1,
                **warning_kwargs
            )

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
        Callable[[ImportTarget], ast.Expr], Callable[[ImportTarget], bool],
    ]:
        """
        Returns:
            (create_node, should_profile) \
(tuple[Callable[[ImportTarget], ast.Expr], \
Callable[[ImportTarget], bool]]):
                create_node()
                    Callable taking the import target and returning an
                    AST node responsible for setting up profiling
                    thereof
                should_profile()
                    Callable taking the import target and returning
                    whether it should be profiled
        """
        profiled_imports = self._prof_explicit_imports
        should_profile: Callable[[ImportTarget], bool]
        if profiled_imports in (True, False):
            should_profile = partial(_return_value, bool(profiled_imports))
        else:
            should_profile = _chain_callables(
                cast(Callable[[ImportTarget], str], attrgetter('name')),
                partial(
                    should_profile_regular_import,
                    cast(Collection[str], profiled_imports),
                ),
            )
        create_node: Callable[[ImportTarget], ast.Expr] = _chain_callables(
            cast(Callable[[ImportTarget], str], attrgetter('resolved_name')),
            partial(
                ast_create_profile_node, profiler_name=self._profiler_name,
            ),
        )
        return create_node, should_profile

    @cached_property
    def _star_import_handlers(self) -> tuple[
        Callable[[ImportTarget], ast.Expr], Callable[[ImportTarget], bool],
    ]:
        """
        Returns:
            See :py:attr:`._expl_import_handlers`.
        """
        profiled_imports = self._prof_star_imports
        should_profile: Callable[[ImportTarget], bool]
        if profiled_imports in (True, False):
            star_import_targets: Collection[str] | None = None
            should_profile = partial(_return_value, bool(profiled_imports))
        else:
            star_import_targets = cast(Collection[str], profiled_imports)
            should_profile = _chain_callables(
                cast(Callable[[ImportTarget], str], attrgetter('name')),
                partial(
                    should_profile_star_import,
                    cast(Collection[str], profiled_imports),
                ),
            )
        create_node: Callable[[ImportTarget], ast.Expr] = _chain_callables(
            cast(Callable[[ImportTarget], str], attrgetter('name')),
            partial(
                self._create_star_import_node,
                targets=star_import_targets,
                profiler_name=self._profiler_name,
            ),
        )
        return create_node, should_profile
