from __future__ import annotations

import ast
import os
from collections.abc import Callable, Collection, Sequence
from functools import cached_property
from operator import attrgetter
from typing import TypeVar
from warnings import warn

from ..toml_config import ConfigSource
from .. import _diagnostics as diagnostics
from ._import_targets import _DROPPED_STAR_IMPORTS_MSG_TEMPLATE, ImportTarget
from ._single_pass_transformer import (
    _CompoundNodeChecker,
    CompoundStatement,
    ContextAwareVisitor,
    _ProfModHelper,
    should_profile_regular_import,
    should_profile_star_import,
)


__all__ = ('ProfmodExtractor',)

_Import = TypeVar('_Import', ast.Import, ast.ImportFrom)


class _ImportFinder(ContextAwareVisitor):
    """
    Locate all the imports inside an AST, including those nested inside
    other nodes.
    """
    def __init__(self, checker: _CompoundNodeChecker) -> None:
        super().__init__()
        self._checker = checker
        self.found_imports: dict[
            tuple[str | int, ...], dict[str, ImportTarget]
        ] = {}

    @classmethod
    def find(
        cls,
        node: ast.AST,
        *,
        config: str | os.PathLike[str] | ConfigSource | None = None,
        find_nested_imports: Collection[CompoundStatement] | None = None,
    ) -> dict[tuple[str | int, ...], list[ImportTarget]]:
        """
        Parameters:
            node (ast.AST):
                AST node

            config (str | os.PathLike[str] | ConfigSource | None):
                Config source from which to load import-discovery
                options from

            find_nested_imports (Collection[Literal[\
'func_defs', 'class_defs',\
'loops', 'conditionals', 'contexts', 'try_except',\
]] | None):
                Only collect import statements inside these
                compound-statement nodes (and their children):

                'conditionals'
                    :py:class:`ast.If` and :py:class:`ast.match_case`
                'try_except'
                    :py:class:`ast.ExceptHandler`, :py:class:`ast.Try`,
                    and :py:class:`ast.TryStar`
                'contexts'
                    :py:class:`ast.AsyncWith` and :py:class:`ast.With`
                'loops'
                    :py:class:`ast.AsyncFor`, :py:class:`ast.For`, and
                    :py:class:`ast.While`
                'func_defs'
                    :py:class:`AsyncFunctionDef` and
                    :py:class:`ast.FunctionDef`
                'class_defs'
                    :py:class:`ast.ClassDef`

        Returns:
            found_imports \
(dict[tuple[str | int, ...], list[ImportTarget]]):
                The import targets and their locations in the AST

        Notes:
            If a ``collect_from_*`` option is set to :py:const:`None`,
            the value is taken from the default configs.
        """
        checker = _CompoundNodeChecker.from_config(config, find_nested_imports)
        visitor = cls(checker)
        visitor.visit(node)
        return {
            loc: sorted(targets.values(), key=attrgetter('index'))
            for loc, targets in visitor.found_imports.items()
        }

    def _visit_import(
        self,
        node: _Import,
        get_import_targets: Callable[[int, _Import], Sequence[ImportTarget]],
    ) -> None:
        *loc, index = self._current_loc
        assert isinstance(index, int)
        for target in get_import_targets(index, node):
            imports = self.found_imports.setdefault(tuple(loc), {})
            imports.setdefault(target.name, target)

    def visit_Import(self, node: ast.Import) -> None:
        self._visit_import(node, ImportTarget._from_import_node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._visit_import(node, ImportTarget._from_import_from_node)

    def visit(self, node: ast.AST) -> None:
        if not self._checker.check(node):
            return  # Don't descend into node types we don't care about
        super().visit(node)


class ProfmodExtractor:
    """
    Map ``prof_mod`` to imports in an abstract syntax tree. Takes the
    paths and dotted paths in ``prof_mod`` and finds their respective
    imports in an abstract syntax tree.
    """
    def __init__(
        self,
        tree: ast.Module,
        script_file: str,
        prof_mod: Sequence[str],
        *,
        config: ConfigSource | None = None,
    ) -> None:
        """
        Initializes the AST tree profiler instance with the AST,
        script-file path and ``prof_mod``

        Args:
            tree (_ast.Module):
                abstract syntax tree to fetch imports from.

            script_file (str):
                path to script being profiled.

            prof_mod (Sequence[str]):
                optional list of imports to profile in script.
                passing the path to script will profile the whole
                script.
                the objects can be specified using its dotted path or
                full path (if applicable).

            config (ConfigSource | None):
                optional :py:class:`.ConfigSource` to load additional
                configurations from.
        """
        self._tree = tree
        self._script_file = script_file
        self._prof_mod = prof_mod
        if config is None:
            config = ConfigSource.from_default()
        self._config = config

    @staticmethod
    def _ast_get_imports_from_tree(
        node: ast.AST,
        config: ConfigSource | None = None,
        find_nested_imports: Collection[CompoundStatement] | None = None,
    ) -> dict[tuple[str | int, ...], list[ImportTarget]]:
        return _ImportFinder.find(
            node, config=config, find_nested_imports=find_nested_imports,
        )

    @staticmethod
    def _find_modnames_in_tree_imports(
        modnames_to_profile: Collection[str],
        import_targets: Sequence[ImportTarget],
    ) -> dict[int, list[ImportTarget]]:
        """Map modnames to imports from an abstract sytax tree.

        Find imports in import_targets, created from an abstract syntax tree, that match
        dotted paths in modnames_to_profile.
        When a submodule is imported, both the submodule and the parent module are checked
        whether they are in modnames_to_profile. As the user can ask to profile
        "foo" when only "from foo import bar" is imported, so both foo and bar are checked.
        The real import name of an import is used to map to the dotted paths.
        The import's alias is stored in the output dict.

        Args:
            modnames_to_profile (Collection[str]):
                list of dotted paths to profile.

            import_targets (Sequence[ImportTarget]):
                list of all import targets in the tree

        Returns:
            filtered_imports (dict[int, list[ImportTarget]]):
                dict of imports found
                    key (int):
                        index of the (from-)import statement in AST
                    value (list[ImportTarget]):
                        list of filtered import targets
        """
        filtered_imports: dict[int, list[ImportTarget]] = {}
        modname_added_list = []
        for i, import_target in enumerate(import_targets):
            modname = import_target.name
            if modname in modname_added_list:
                continue
            should_profile: Callable[[Collection[str], str], bool]
            if modname.endswith('.*'):
                should_profile = should_profile_star_import
            else:
                should_profile = should_profile_regular_import
            if not should_profile(modnames_to_profile, modname):
                continue
            modname_added_list.append(modname)
            try:
                filtered_imports[import_target.index].append(import_target)
            except KeyError:  # No imports recorded for the statement
                filtered_imports[import_target.index] = [import_target]
        return filtered_imports

    def extract_all(
        self,
        *,
        filter_star_imports: bool | None = None,
        find_nested_imports: Collection[CompoundStatement] | None = None,
    ) -> dict[tuple[str | int, ...], list[ImportTarget]]:
        """
        Map ``prof_mod`` to imports in an abstract syntax tree.
        Takes the paths and dotted paths in ``prof_mod`` and finds their
        respective imports in an abstract syntax tree, returning their
        aliases and the location they appear in the AST.

        Args:
            filter_star_imports (bool | None):
                If true, filter out star imports
                (``from <module> import *``) with a warning;
                if :py:const:`None`, it is loaded from the ``config``
                (as the negation of ``autoprofile.prof_star_imports``).

            find_nested_imports \
(Collection[Literal['func_defs', 'class_defs', \
'loops', 'conditionals', 'contexts', 'try_except']] | None):
                Which of the compound-statement types to look for nested
                imports in;
                if :py:const:`None`, it is loaded from the ``config``
                (from ``autoprofile.import_discovery``).

        Returns:
            tree_imports_to_profile_dict \
(dict[tuple[str | int, ...], list[ImportTarget]]);
                dict of imports to profile
                    key (tuple[str | int, ...]):
                        Location of the import statement in the AST;
                        e.g. ``('body', 0)`` for the case where it is
                        the first statement in the
                        :py:attr:`ast.Module.body`
                    value (list[ImportTarget]):
                        list of import targets, each with these
                        attributes:

                        name (str):
                            Canonical name of the import
                        index (int):
                            Index where it occurs in e.g. a module body
                        alias (str | None):
                            Optional alias under which the import is
                            inserted into the namespace
                        lineno (int | None):
                            Optional (1-indexed) line number associated
                            with the target
                        resolved_name (str | None):
                            Name under which the import is inserted into
                            the namespace (should never be
                            :py:const`None` for non-star-imports)
        """
        if filter_star_imports is None:
            filter_star_imports = not _should_profile_star_imports(
                self._config,
            )
        import_targets = self._ast_get_imports_from_tree(
            self._tree, self._config, find_nested_imports,
        )
        raw: dict[tuple[str | int, ...], list[ImportTarget]] = {
            (*loc, index): filtered_imports
            for loc, imports in import_targets.items()
            for index, filtered_imports in self._find_modnames_in_tree_imports(
                self._modnames_to_profile, imports,
            ).items()
        }
        filtered: dict[tuple[str | int, ...], list[ImportTarget]] = {}
        star_imports: set[ImportTarget] = set()
        for loc, imports in raw.items():
            # Notes:
            # - We don't issue the warning in
            #   `._find_modnames_in_tree_imports()` because that is a
            #   static method and doesn't have access to the file from
            #   which the AST is generated, which we want to include in
            #   the warning message.
            # - As far normal Python syntax is concerned, each
            #   import-from statement can have at most one `*` target,
            #   which would be the sole target thereof (so
            #   `indices_to_drop` should either be `[]` or `[0]`);
            #   but it doesn't hurt to be cautious
            if filter_star_imports:
                indices_to_drop = [
                    i for i, imp in enumerate(imports)
                    if imp.resolved_name is None  # Star-imports
                ]
                for i in reversed(indices_to_drop):
                    star_imports.add(imports.pop(i))
            if imports:
                filtered[loc] = imports
        ImportTarget._check_and_warn_dropped_imports(
            star_imports,
            _DROPPED_STAR_IMPORTS_MSG_TEMPLATE.format(
                action='extracted',
                argname='filter_star_imports',
            ),
            self._script_file,
            stacklevel=2,  # Attribute warning to the caller
        )
        return filtered

    def run(self) -> dict[int, str]:
        """
        Deprecated, legacy method kept for backward compatibility.

        Returns:
            tree_imports_to_profile_dict (dict[int, str])
                dict of top-level imports to profile
                    key (int):
                        index of import in module AST's body
                    value (str):
                        alias (or name if no alias used) of the LAST
                        target to import in the corresponding
                        :py:class:`ast.Import` or
                        :py:class:`ast.ImportFrom` statement

        Notes:
            - New code should use the :py:meth:`.extract_all` method,
              which handles multi-target import statements (see #434).

            - Calling this method issues a
              :py:class:`DeprecationWarning`.

            - For multi-target import statements, this only preserves
              the last target. If this results in import targets being
              dropped, a :py:class:`UserWarning` is issued.

            - ``from <module> import *`` is not supported, and will
              result in a :py:class:`UserWarning`.

            - Nested imports (e.g. imports in try-except/if blocks) are
              not retrieved.
        """
        msg = (
            '`ProfmodExtractor.run()` is now deprecated, because it cannot '
            'correctly resolve multi-target import statements; '
            'use `ProfmodExtractor.extract_all()` instead.'
        )
        diagnostics.log.warning(f'DeprecationWarning: {msg}')
        warn(msg, DeprecationWarning, stacklevel=2)  # Caller
        result: dict[int, str] = {}
        dropped: set[ImportTarget] = set()
        for index, imports in self.extract_all().items():
            if not (
                len(index) == 2
                and index[0] == 'body'
                and isinstance(index[1], int)
            ):  # We only handle the top-level imports here
                continue
            *remainder, last = imports
            dropped.update(remainder)
            dropped.discard(last)
            name = last.resolved_name
            if name is None:
                # Shouldn't happen with the current `.extract_all()`,
                # but once we fix star-imports...
                continue
            result[index[1]] = name
        ImportTarget._check_and_warn_dropped_imports(
            dropped,
            'the import statement(s) is/are multi-target',
            self._script_file,
            stacklevel=2,  # Attribute warning to caller
        )
        return result

    @cached_property
    def _modnames_to_profile(self) -> frozenset[str]:
        # Skip `script_file` itself, in case it gets normalized to a
        # clashing with another import or function, leading to unwanted
        # profiling
        helper = _ProfModHelper(self._script_file, self._prof_mod)
        return frozenset(helper.to_dotted_paths(exclude_script_file=True))


def _should_profile_star_imports(config: ConfigSource | None) -> bool:
    if config is None:
        config = ConfigSource.from_default()
    kvps = config.get_subconfig('autoprofile').conf_dict
    return bool(kvps['prof_star_imports'])
