from __future__ import annotations

import ast
from collections.abc import Sequence
from typing import cast

from ._single_pass_transformer import _ProfModHelper, SinglePassTransformer
from .ast_tree_profiler import AstTreeProfiler
from .util_static import modpath_to_modname


__all__ = (
    'ImportFromTransformer',
    'AstTreeModuleProfiler',
    'get_module_from_importfrom',
)

get_module_from_importfrom = SinglePassTransformer._resolve_importfrom_module
_consolidate = SinglePassTransformer._consolidate_relative_import


class ImportFromTransformer(ast.NodeTransformer):
    """Turn all the relative imports into absolute imports."""

    def __init__(self, module: str) -> None:
        self.module = module

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        node = _consolidate(node, self.module)
        return cast(ast.ImportFrom, self.generic_visit(node))


class AstTreeModuleProfiler(AstTreeProfiler):
    """Create an abstract syntax tree of an executable module and add
    profiling to it.

    Read the module code and generate an abstract syntax tree, then add
    nodes and/or decorators to the AST that add the specified
    functions/methods, classes & modules in ``prof_mod`` to the profiler
    to be profiled.
    """

    @classmethod
    def _get_script_ast_tree(cls, script_file: str) -> ast.Module:
        tree = super()._get_script_ast_tree(script_file)
        # Note: don't drop the `.__init__` or `.__main__` suffix, lest
        # the relative imports fail
        module = modpath_to_modname(
            script_file, hide_main=False, hide_init=False,
        )
        return ImportFromTransformer(module).visit(tree)

    @staticmethod
    def _check_profile_full_script(
        script_file: str, prof_mod: Sequence[str],
    ) -> bool:
        helper = _ProfModHelper(script_file, prof_mod)
        return helper.script_file_is_included(match_mode='module')
