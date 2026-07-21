from __future__ import annotations

import ast
import os
from collections.abc import Collection, MutableSequence, Sequence
from typing import Any, cast

from ._import_targets import ImportTarget
from ..toml_config import ConfigSource
from .ast_profile_transformer import (  # noqa: F401
    AstProfileTransformer,
    _ast_create_node_from_import_target,
    # Keep import below for compatibility
    ast_create_profile_node,
)
from .profmod_extractor import ProfmodExtractor

__docstubs__ = """
from .ast_profile_transformer import AstProfileTransformer
from .profmod_extractor import ProfmodExtractor
"""


class AstTreeProfiler:
    """Create an abstract syntax tree of a script and add profiling to it.

    Reads a script file and generates an abstract syntax tree, then adds nodes
    and/or decorators to the AST that adds the specified functions/methods,
    classes & modules in prof_mod to the profiler to be profiled.
    """

    def __init__(
        self,
        script_file: str,
        prof_mod: Sequence[str],
        profile_imports: bool,
        ast_transformer_class_handler: (
            type[AstProfileTransformer]
        ) = AstProfileTransformer,
        profmod_extractor_class_handler: (
            type[ProfmodExtractor]
        ) = ProfmodExtractor,
        config: ConfigSource | None = None,
    ) -> None:
        """Initializes the AST tree profiler instance with the script file path

        Args:
            script_file (str):
                path to script being profiled.

            prof_mod (List[str]):
                list of imports to profile in script.
                passing the path to script will profile the whole script.
                the objects can be specified using its dotted path or full path (if applicable).

            profile_imports (bool):
                if True, when auto-profiling whole script, profile all imports aswell.

            ast_transformer_class_handler (type[AstProfileTransformer]):
                the AstProfileTransformer class that handles profiling the whole script.

            profmod_extractor_class_handler (type[ProfmodExtractor]):
                the ProfmodExtractor class that handles mapping prof_mod to objects in the script.

            config (ConfigSource | None):
                optional :py:class:`.ConfigSource` to load additional
                configurations from.
        """
        self._script_file = script_file
        self._prof_mod = prof_mod
        self._profile_imports = profile_imports
        self._ast_transformer_class_handler = ast_transformer_class_handler
        self._profmod_extractor_class_handler = profmod_extractor_class_handler
        self._config = config

    @staticmethod
    def _check_profile_full_script(
        script_file: str, prof_mod: Sequence[str],
    ) -> bool:
        """Check whether whole script should be profiled.

        Checks whether path to script has been passed to prof_mod indicating that
        the whole script should be profiled

        Args:
            script_file (str):
                path to script being profiled.

            prof_mod (Sequence[str]):
                list of imports to profile in script.
                passing the path to script will profile the whole script.
                the objects can be specified using its dotted path or full path (if applicable).

        Returns:
            profile_full_script (bool):
                if True, profile whole script.
        """
        script_file_realpath = os.path.realpath(script_file)
        profile_full_script = script_file_realpath in map(
            os.path.realpath, prof_mod
        )
        return profile_full_script

    @staticmethod
    def _get_script_ast_tree(script_file: str) -> ast.Module:
        """Generate an abstract syntax from a script file.

        Args:
            script_file (str):
                path to script being profiled.

        Returns:
            tree (_ast.Module):
                abstract syntax tree of the script.
        """
        with open(script_file, 'r') as f:
            script_text = f.read()
        tree = ast.parse(script_text, filename=script_file)
        return tree

    def _profile_ast_tree(
        self,
        tree: ast.Module,
        tree_imports_to_profile_dict: dict[
            tuple[str | int, ...], list[ImportTarget]
        ],
        profile_full_script: bool = False,
        profile_imports: bool = False,
        modnames_to_profile: Collection[str] = (),
        profile_star_imports: bool = False,
    ) -> ast.Module:
        """
        Add profiling to an abstract syntax tree by adding nodes to the
        AST that adds the specified objects to the profiler.

        - If ``profile_full_script`` is True, all functions/methods,
          classes & modules in the script have a node added to the AST
          to add them to the profiler.

        - If ``profile_imports ``is True as well as
          ``profile_full_script``, all imports are have a node added to
          the AST to add them to the profiler.

        Args:
            tree (_ast.Module):
                abstract syntax tree to be profiled.

            tree_imports_to_profile_dict (dict[tuple[str | int, ...], \
list[ImportTarget]]):
                dict of imports to profile
                    key (tuple[str | int, ...]):
                        Location of import in AST, e.g. ``('body', 0)``
                        for the case where it is the first statement in
                        the :py:attr:`ast.Module.body`
                    value (list[ImportTarget]):
                        list of import targets (see the documentation of
                        :py:class:`line_profiler.autoprofile\
.profmod_extractor.ImportTarget`)

            profile_full_script (bool):
                if True, profile the entire script.

            profile_imports (bool):
                if True, and ``profile_full_script`` is True, profile
                all imports as well.

            modnames_to_profile (Collection[str]):
                module names to be profiled; needed for processing
                star-imports (``from ... import *``).

            profile_star_imports (bool):
                if True, profile star-imports (those included in
                ``tree_imports_to_profile_dict``, or retrieved by
                :py:class:`.AstProfileTransformer` if both
                ``profile_full_script`` and ``profile_imports`` are
                true)

        Returns:
            (_ast.Module): tree
                abstract syntax tree with profiling.
        """
        profiled_imports = []
        argsort_tree_indexes = sorted(
            list(tree_imports_to_profile_dict), reverse=True
        )
        for tree_loc in argsort_tree_indexes:
            imports = tree_imports_to_profile_dict[tree_loc]
            *loc, tree_index = tree_loc
            assert isinstance(tree_index, int)
            body = cast(MutableSequence[ast.AST], self._descend(tree, loc))
            assert isinstance(body, MutableSequence)
            for imp in reversed(imports):
                # Reversing keeps the order of the inserted nodes
                # consistent with the imports
                expr = _ast_create_node_from_import_target(
                    imp, modnames_to_profile, profile_star_imports,
                )
                if expr is not None:
                    body.insert(tree_index + 1, expr)
                    profiled_imports.append(imp.name)
        if profile_full_script:
            tree = self._ast_transformer_class_handler._transform(
                tree, self._script_file,
                profile_imports=profile_imports,
                profiled_imports=profiled_imports,
                profile_star_imports=profile_star_imports,
            )
        ast.fix_missing_locations(tree)
        return tree

    def profile(self, profile_star_imports: bool = False) -> ast.Module:
        """
        Create an abstract syntax tree of a script and add profiling to
        it:

        - Read a script file and generates an abstract syntax tree.

        - Then matches imports in the script's AST with the names in
          ``prof_mod``.

        - The matched imports are added to the profiler for profiling.

        - If the path to the script is found in ``prof_mod``, all
          functions/methods, classes & modules are added to the
          profiler.

        - If ``profile_imports`` is True as well as path to script in
          ``prof_mod``, all the imports in the script are added to the
          profiler.

        Args:
            profile_star_imports (bool):
                if True, add targets imported by ``from ... import *``
                statements to the profiler.

        Returns:
            (_ast.Module): tree
                abstract syntax tree with profiling.
        """
        profile_full_script = self._check_profile_full_script(
            self._script_file, self._prof_mod
        )

        tree = self._get_script_ast_tree(self._script_file)

        # Note: warnings about dropped star-imports can be issued from 2
        # places:
        # - `ProfmodExtractor.extract_all(filter_star_imports=True)`
        # - `._profile_ast_tree(...), where both
        #   `profile_full_script=True` and `profile_imports=True`
        # So take care to enture that we don't have duplicate warnings
        extractor = self._profmod_extractor_class_handler(
            tree, self._script_file, self._prof_mod, self._config,
        )
        if profile_star_imports:
            # Star imports recovered -> nothing to warn either way
            filter_star_imports_in_extract_all = False
        else:
            filter_star_imports_in_extract_all = not (
                profile_full_script and self._profile_imports
            )
        tree_imports_to_profile_dict = extractor.extract_all(
            filter_star_imports=filter_star_imports_in_extract_all,
        )

        tree_profiled = self._profile_ast_tree(
            tree,
            tree_imports_to_profile_dict,
            profile_full_script=profile_full_script,
            profile_imports=self._profile_imports,
            modnames_to_profile=extractor._modnames_to_profile,
            profile_star_imports=profile_star_imports,
        )
        return tree_profiled

    @staticmethod
    def _descend(obj: Any, loc: Sequence[str | int]) -> Any:
        """
        Follow ``loc``, a sequence of indices (item access) and names
        (attribute access), to descend into an object.

        Examples:
            >>> from types import SimpleNamespace as ns

            >>> my_ns = ns(
            ...     foo={1: 2, 3: ['foo', 'bar']},
            ...     bar=[ns(spam=1), ns(ham=[1, 2], eggs=None)]
            ... )
            >>> AstTreeProfiler._descend(my_ns, ['foo', 3, 1, 2])
            'r'
            >>> AstTreeProfiler._descend(my_ns, ['bar', 1, 'ham', 0])
            1
        """
        for index_or_attr in loc:
            if isinstance(index_or_attr, int):
                obj = obj[index_or_attr]
            else:  # Attribute name
                obj = getattr(obj, index_or_attr)
        return obj
