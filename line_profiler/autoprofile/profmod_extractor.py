from __future__ import annotations

import ast
import dataclasses
import os
import sys
from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast
from typing_extensions import Self
from warnings import warn

from .util_static import (
    modname_to_modpath,
    modpath_to_modname,
    package_modpaths,
)
from .. import _diagnostics as diagnostics


_ImportTargetField = Literal['name', 'index', 'alias', 'resolved_name']


@dataclasses.dataclass(eq=True, frozen=True)
class _ImportTarget:
    """
    An import target.

    Init attrs:
        name (str):
            The real name of the import. e.g. ``import foo as bar``
            -> ``'foo'``

        index (int):
            The index of the import as found in the AST body

        alias (str | None):
            The alias of an import if applicable. e.g.:

            - ``import foo as bar`` -> ``'bar'``
            - ``import foo`` -> ``None``

        lineno (int | None):
            Optional (1-indexed) line number on which the import occurs.

    Other attrs:
        resolved_name (str):
            Name under which the import can be found, e.g.:

            - ``import foo as bar`` -> ``'bar'``
            - ``import foo`` -> ``'foo'``
    """
    name: str
    index: int
    alias: str | None = None
    lineno: int | None = None

    def __post_init__(self) -> None:
        """
        Type verifications.
        """
        if not isinstance(self.name, str):
            raise TypeError(f'.name = {self.name!r}: expected a str')
        if not isinstance(self.index, int):
            raise TypeError(
                f'.index = {self.index!r}: expected an int',
            )
        if not (self.alias is None or isinstance(self.alias, str)):
            raise TypeError(f'.alias = {self.alias!r}: expected a str or None')
        if not (self.lineno is None or isinstance(self.lineno, int)):
            raise TypeError(
                f'.lineno = {self.lineno!r}: expected an int or None',
            )

    @classmethod
    def from_ast_nodes(cls, nodes: Sequence[ast.AST]) -> list[Self]:
        """
        Get all imports in the body of an AST node.

        Args:
            nodes (Sequence[ast.AST]):
                AST nodes to scan for imports;
                examples: :py:attr:`ast.Module.body`,
                :py:attr:`ast.If.orelse`.

        Returns:
            import_targets (list[Self]):
                List of all imports amond the nodes.

        Notes:
            Imports nested inside the ``nodes`` are not as yet
            handled, e.g. in

            >>> # doctest: +SKIP
            >>> from spam import ham
            >>> try:
            ...     from some_foo import bar
            ... except ImportError:
            ...     from other_foo import ersatz_bar as bar

            Only ``ham`` is extracted but not ``bar``.
        """
        targets: list[Self] = []
        modnames: set[str] = set()
        for index, node in enumerate(nodes):
            if isinstance(node, ast.Import):
                new_targets: list[Self] = cls._from_import_node(index, node)
            elif isinstance(node, ast.ImportFrom):
                new_targets = cls._from_import_from_node(index, node)
            else:  # TODO: descend into other bodied nodes
                new_targets = []
            for target in new_targets:
                if target.name not in modnames:
                    targets.append(target)
                    modnames.add(target.name)
        return targets

    @classmethod
    def _from_import_node(cls, index: int, node: ast.Import) -> list[Self]:
        return [
            cls(name.name, index, name.asname, name.lineno)
            for name in node.names
        ]

    @classmethod
    def _from_import_from_node(
        cls, index: int, node: ast.ImportFrom,
    ) -> list[Self]:
        if node.module is None:  # `from . import ...`
            return []
        return [
            cls(
                f'{node.module}.{name.name}',
                index,
                name.asname or name.name,
                name.lineno,
            )
            for name in node.names
        ]

    @property
    def resolved_name(self) -> str:
        return self.alias or self.name

    @property
    def _star_import_source(self) -> str | None:
        # Note: star-imports are parsed into
        # `_ImportTarget('module.name.*', index, '*', lineno)`
        if self.alias != '*':
            return None
        assert self.name.endswith('.*')
        return self.name[:-2]


class ProfmodExtractor:
    """Map prof_mod to imports in an abstract syntax tree.

    Takes the paths and dotted paths in prod_mod and finds their respective imports in an
    abstract syntax tree.
    """

    def __init__(
        self, tree: ast.Module, script_file: str, prof_mod: list[str]
    ) -> None:
        """Initializes the AST tree profiler instance with the AST, script file path and prof_mod

        Args:
            tree (_ast.Module):
                abstract syntax tree to fetch imports from.

            script_file (str):
                path to script being profiled.

            prof_mod (list[str]):
                list of imports to profile in script.
                passing the path to script will profile the whole script.
                the objects can be specified using its dotted path or full path (if applicable).
        """
        self._tree = tree
        self._script_file = script_file
        self._prof_mod = prof_mod

    @staticmethod
    def _is_path(text: str) -> bool:
        """Check whether a string is a path.

        Checks if a string contains a slash or ends with .py indicating it is a path.

        Args:
            text (str):
                string to check whether it is a path or not

        Returns:
            ret (bool):
                bool indicating whether the string is a path or not
        """
        ret = ('/' in text.replace('\\', '/')) or text.endswith('.py')
        return ret

    @classmethod
    def _get_modnames_to_profile_from_prof_mod(
        cls, script_file: str, prof_mod: list[str]
    ) -> list[str]:
        """Grab the valid paths and all dotted paths in prof_mod and their subpackages
        and submodules, in the form of dotted paths.

        First all items in prof_mod are converted to a valid path. if unable to convert,
        check if the item is an invalid path and skip it, else assume it is an installed package.
        The valid paths are then converted to dotted paths.
        The converted dotted paths along with the items assumed to be installed packages
        are added a list of modnames_to_profile.
        Then all subpackages and submodules under each valid path is fetched, converted to
        dotted path and also added to the list.
        if script_file is in prof_mod it is skipped to avoid name collision with othe imports,
        it will be processed elsewhere in the autoprofile pipeline.

        Args:
            script_file (str):
                path to script being profiled.

            prof_mod (list[str]):
                list of imports to profile in script.
                passing the path to script will profile the whole script.
                the objects can be specified using its dotted path or full path (if applicable).

        Returns:
            modnames_to_profile (list[str]):
                list of dotted paths to profile.
        """
        script_directory = os.path.realpath(os.path.dirname(script_file))
        """add script folder to modname_to_modpath sys_path to allow it to resolve modpaths"""
        new_sys_path = [script_directory] + sys.path
        script_file_realpath = os.path.realpath(script_file)

        modnames_to_profile = []
        for mod in prof_mod:
            if script_file_realpath == os.path.realpath(mod):
                """
                skip script_file as it will add the script's name without its extension which
                could have the same name as another import or function leading to unwanted profiling
                """
                continue
            """
            convert the item in prof_mod into a valid path.
            if it fails, the item may point to an installed module rather than local script
            so we check if the item is path and whether that path exists, else skip the item.
            """
            modpath = modname_to_modpath(
                mod, sys_path=cast('list[str | os.PathLike]', new_sys_path)
            )
            if modpath is None:
                """if cannot convert to modpath, check if already path and if invalid"""
                if not os.path.exists(mod):
                    if cls._is_path(mod):
                        """modpath does not exist, so skip"""
                        continue
                    modnames_to_profile.append(mod)
                    continue
                """assume item is and installed package. modpath_to_modname will have no effect"""
                modpath = mod

            """convert path to dotted path and add it to list to be profiled"""
            try:
                modname = modpath_to_modname(modpath)
            except ValueError:
                continue
            if modname not in modnames_to_profile:
                modnames_to_profile.append(modname)

            """
            recursively fetch all subpackages and submodules, convert them to dotted paths
            and add them to list to be profiled
            """
            for submod_path in package_modpaths(modpath):
                submod_name = modpath_to_modname(submod_path)
                if submod_name not in modnames_to_profile:
                    modnames_to_profile.append(submod_name)

        return modnames_to_profile

    @staticmethod
    def _ast_get_imports_from_tree(tree: ast.Module) -> list[_ImportTarget]:
        """Get all imports in an abstract syntax tree.

        Args:
            tree (_ast.Module):
                abstract syntax tree to fetch imports from.

        Returns:
            import_targets (list[dict[str, str | int]])

        See also:
            :py:meth:`._ImportTarget.from_ast_nodes`
        """
        return _ImportTarget.from_ast_nodes(tree.body)

    @staticmethod
    def _find_modnames_in_tree_imports(
        modnames_to_profile: list[str], import_targets: list[_ImportTarget],
    ) -> dict[int, list[_ImportTarget]]:
        """Map modnames to imports from an abstract sytax tree.

        Find imports in import_targets, created from an abstract syntax tree, that match
        dotted paths in modnames_to_profile.
        When a submodule is imported, both the submodule and the parent module are checked
        whether they are in modnames_to_profile. As the user can ask to profile
        "foo" when only "from foo import bar" is imported, so both foo and bar are checked.
        The real import name of an import is used to map to the dotted paths.
        The import's alias is stored in the output dict.

        Args:
            modnames_to_profile (list[str]):
                list of dotted paths to profile.

            import_targets (list[_ImportTarget]):
                list of dicts of all imports in the tree
                (see return value of
                :py:meth:`._ast_get_imports_from_tree()`)

        Returns:
            filtered_imports (dict[int, list[_ImportTarget]]):
                dict of imports found
                    key (int):
                        index of the (from-)import statement in AST
                    value (list[_ImportTarget]):
                        list of filtered import targets
        """
        filtered_imports: dict[int, list[_ImportTarget]] = {}
        modname_added_list = []
        for i, import_target in enumerate(import_targets):
            modname = import_target.name
            if modname in modname_added_list:
                continue
            # Check if either the parent module or submodule are in
            # `modnames_to_profile`
            if (
                modname not in modnames_to_profile
                and modname.rsplit('.', 1)[0] not in modnames_to_profile
            ):
                continue
            modname_added_list.append(modname)
            try:
                filtered_imports[import_target.index].append(import_target)
            except KeyError:  # No imports recorded for the statement
                filtered_imports[import_target.index] = [import_target]
        return filtered_imports

    def _extract_all(self) -> dict[int, list[_ImportTarget]]:
        """
        Find all the import targets to profile.
        """
        modnames_to_profile = self._get_modnames_to_profile_from_prof_mod(
            self._script_file, self._prof_mod
        )
        import_targets = self._ast_get_imports_from_tree(self._tree)
        raw = self._find_modnames_in_tree_imports(
            modnames_to_profile, import_targets,
        )
        filtered: dict[int, list[_ImportTarget]] = {}
        star_imports: dict[int | None, set[_ImportTarget]] = {}
        imports: Iterable[_ImportTarget]
        for index, imports in raw.items():
            # TODO: runtime introspection of imports to handle
            # star-imports
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
            indices_to_drop = [
                i for i, imp in enumerate(imports)
                if imp._star_import_source is not None
            ]
            for i in reversed(indices_to_drop):
                imp = imports.pop(i)
                try:
                    star_imports[imp.lineno].add(imp)
                except KeyError:
                    star_imports[imp.lineno] = {imp}
            if imports:
                filtered[index] = imports
        if star_imports:
            msg_chunks: list[str] = [
                (
                    '{}: {} would-be profiling target(s) dropped because the '
                    "we don't currently handle `from ... import *` statements:"
                ).format(self._script_file, len(star_imports)),
            ]
            for lineno, imports in sorted(
                star_imports.items(), key=_import_dict_item_sort_key,
            ):
                lineno_repr = '???' if lineno is None else str(lineno)
                targets = ', '.join(sorted(imp.name for imp in imports))
                msg_chunks.append(f'- line {lineno_repr}: {targets}')
            sep = ' ' if len(msg_chunks) < 3 else '\n'
            # Attribute the warning to the caller of `.run()` or
            # `.extract_all()`
            _issue_warning(sep.join(msg_chunks), stacklevel=3)
        return filtered

    def extract_all(self) -> dict[int, list[str]]:
        """Map prof_mod to imports in an abstract syntax tree.

        Takes the paths and dotted paths in prof_mod and finds their respective imports in an
        abstract syntax tree, returning their aliases and the index they appear in the AST.

        Returns:
            tree_imports_to_profile_dict (dict[int, list[str]]);
                dict of imports to profile
                    key (int):
                        index of import in AST
                    value (str | list[str]):
                        list of aliases (or names if no alias used) to
                        import

        Notes:
            As of now, ``from <module> import *`` is not supported, and
            will result in a :py:class:`UserWarning`.
        """
        return {
            index: [imp.resolved_name for imp in imports]
            for index, imports in self._extract_all().items()
        }

    def run(self) -> dict[int, str]:
        """
        Deprecated, legacy method kept for backward compatibility.

        Returns:
            tree_imports_to_profile_dict (dict[int, str])
                dict of imports to profile
                    key (int):
                        index of import in AST
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

            - As of now, ``from <module> import *`` is not supported,
              and will result in a :py:class:`UserWarning`.
        """
        msg = (
            '`ProfmodExtractor.run()` is now deprecated, because it cannot '
            'correctly resolve multi-target import statements; '
            'use `ProfmodExtractor.extract_all()` instead.'
        )
        _issue_warning(msg, DeprecationWarning, stacklevel=2)
        result: dict[int, str] = {}
        dropped: set[_ImportTarget] = set()
        imports: Iterable[_ImportTarget]
        for index, imports in self._extract_all().items():
            *remainder, last = imports
            dropped.update(remainder)
            dropped.discard(last)
            result[index] = last.resolved_name
        if dropped:
            dropped_grouped: dict[int | None, set[_ImportTarget]] = {}
            for imp in dropped:
                try:
                    dropped_grouped[imp.lineno].add(imp)
                except KeyError:
                    dropped_grouped[imp.lineno] = {imp}
            msg_chunks: list[str] = [
                (
                    '{}: {} would-be profiling target(s) dropped because the '
                    'import statement(s) are multi-target:'
                ).format(self._script_file, len(dropped)),
            ]
            for lineno, imports in sorted(
                dropped_grouped.items(), key=_import_dict_item_sort_key,
            ):
                lineno_repr = '???' if lineno is None else str(lineno)
                targets = ', '.join(sorted(
                    imp.name
                    if imp.alias is None else
                    f'{imp.alias} (= {imp.name})'
                    for imp in imports
                ))
                msg_chunks.append(f'- line {lineno_repr}: {targets}')
            sep = ' ' if len(msg_chunks) < 3 else '\n'
            _issue_warning(sep.join(msg_chunks), stacklevel=2)
        return result


def _import_dict_item_sort_key(item: tuple[int | None, Any]) -> float:
    lineno, _ = item
    if lineno is None:
        return float('inf')
    return lineno


def _issue_warning(
    msg: str,
    category: type[Warning] | None = None,
    stacklevel: int = 1,
    *args,
    **kwargs,
) -> None:
    if category is None:
        log_msg = msg
    else:
        log_msg = f'{category.__name__}: {msg}'
    diagnostics.log.warning(log_msg)
    warn(msg, category, stacklevel + 1, *args, **kwargs)
