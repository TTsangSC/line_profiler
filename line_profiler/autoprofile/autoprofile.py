"""

AutoProfile Script Demo
=======================

The following demo is end-to-end bash code that writes a demo script and
profiles it with autoprofile.

.. code:: bash

    # Write demo python script to disk
    python -c "if 1:
        import textwrap
        text = textwrap.dedent(
            '''
            def plus(a, b):
                return a + b

            def fib(n):
                a, b = 0, 1
                while a < n:
                    a, b = b, plus(a, b)

            def main():
                import math
                import time
                start = time.time()

                print('start calculating')
                while time.time() - start < 1:
                    fib(10)
                    math.factorial(1000)
                print('done calculating')

            main()
            '''
        ).strip()
        with open('demo.py', 'w') as file:
            file.write(text)
    "

    echo "---"
    echo "## Profile With AutoProfile"
    python -m kernprof -p demo.py -l demo.py
    python -m line_profiler -rmt demo.py.lprof
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
from collections.abc import Collection, MutableMapping
from typing import Any, cast, get_args

from .._diagnostics import USE_LEGACY_AUTOPROF
from ..toml_config import ConfigSource
from ..line_profiler_utils import restore
from ._single_pass_transformer import (
    _ProfModHelper, CompoundStatement, SinglePassTransformer,
)
from .ast_tree_profiler import AstTreeProfiler
from .run_module import AstTreeModuleProfiler
from .line_profiler_utils import (
    add_imported_function_or_module, add_star_import,
)
from .util_static import modpath_to_modname


__all__ = ('run',)

PROFILER_LOCALS_NAME = 'prof'


def _extend_line_profiler_for_profiling_imports(prof: Any) -> None:
    """
    Allow profiler to handle imported functions/methods, classes and
    modules, and also star-import targets, with a single call. This
    adds to a :py:class:`line_profiler.LineProfiler` instance:

    - A method that can identify whether the object is a
      function/method, class, or module, and handle it's profiling
      accordingly; and

    - A method that can retrieve the names imported via a star-import
      and use the above handling to profile them.

    Mainly used for profiling objects that are imported.

    Args:
        prof (LineProfiler):
            instance of :py:class:`line_profiler.LineProfiler`.

    Notes:
        This is a workaround to keep changes needed by autoprofile
        separate from the base :py:class:`line_profiler.LineProfiler`.
    """
    for func in add_imported_function_or_module, add_star_import:
        setattr(prof, func.__name__, types.MethodType(func, prof))


def _rewrite_ast_legacy(
    script_file: str,
    module_name: str | None,
    /,
    prof_mod: list[str],
    profile_imports: bool,
    config: os.PathLike[str] | str | None,
    profile_star_imports: bool | None,
    profile_nested_imports: Collection[CompoundStatement] | None,
) -> ast.Module:
    """
    Use legacy components (``AstTreeProfiler``,
    ``AstProfileTransformer``, ``ProfmodExtractor``) to rewrite the AST
    of ``script_file``.
    """
    Profiler = AstTreeModuleProfiler if module_name else AstTreeProfiler
    profiler = Profiler(
        script_file, prof_mod, profile_imports,
        config=ConfigSource.from_config(config),
    )
    return profiler.profile(
        profile_star_imports=profile_star_imports,
        profile_nested_imports=profile_nested_imports,
    )


def _rewrite_ast_single_pass(
    script_file: str,
    module_name: str | None,
    /,
    prof_mod: list[str],
    profile_imports: bool,
    config: os.PathLike[str] | str | None,
    profile_star_imports: bool | None,
    profile_nested_imports: Collection[CompoundStatement] | None,
) -> ast.Module:
    """
    Use :py:class:`.SinglePassTransformer` to rewrite the AST of
    ``script_file``.
    """
    helper = _ProfModHelper(script_file, prof_mod)
    conf = ConfigSource.from_config(config)
    explicit_targets = helper.to_dotted_paths(exclude_script_file=True)
    if profile_star_imports is None:
        profile_star_imports = bool(
            conf.get_subconfig('autoprofile').conf_dict['prof_star_imports'],
        )

    # If `script_file` is included among the `prof_mod`:
    # - All imports are to be rewritten
    # - All local functions/methods are to be decorated
    profile_entire_script = helper.script_file_is_included(
        match_mode='module' if module_name else 'filename',
    )
    if profile_entire_script and profile_imports:
        prof_explicit_imports: list[str] | bool = True
    else:
        prof_explicit_imports = explicit_targets

    prof_star_imports: list[str] | bool
    prof_star_imports = profile_star_imports and prof_explicit_imports

    if profile_nested_imports is None:
        prof_imports_in: dict[CompoundStatement, bool] | None = None
    else:
        prof_imports_in = {
            stmt: stmt in profile_nested_imports
            for stmt in get_args(CompoundStatement)
        }

    transformer = SinglePassTransformer(
        config=conf,
        module=module_name,
        prof_func_defs=profile_entire_script,
        prof_explicit_imports=prof_explicit_imports,
        prof_star_imports=prof_star_imports,
        prof_imports_in=prof_imports_in,
    )
    with open(script_file) as fobj:
        tree: ast.Module = ast.parse(fobj.read(), filename=script_file)
    tree = transformer._transform(
        tree, script_file,
        # If we are profiling star-imports as indicated by the boolean,
        # the non-profiled star-imports are only a result of their not
        # being selected and therefore should not be reported in
        # warnings
        # FIXME: behavior not entirely consistent with legacy backend
        warn_dropped_star_imports=not profile_star_imports,
    )
    ast.fix_missing_locations(tree)
    return tree


def run(
    script_file: str,
    ns: MutableMapping[str, Any],
    prof_mod: list[str],
    profile_imports: bool = False,
    as_module: bool = False,
    *,
    config: os.PathLike[str] | str | None = None,
    profile_star_imports: bool | None = None,
    profile_nested_imports: Collection[CompoundStatement] | None = None,
) -> None:
    """
    Automatically profile a script and run it, profiling functions,
    classes & modules specified in ``prof_mod`` without needing to add
    ``@profile`` decorators.

    Args:
        script_file (str):
            path to the script being profiled.

        ns (dict):
            local names to injected into the namespace where
            ``script_file``'s code is executed.

        prof_mod (List[str]):
            list of imports to profile in ``script_file``;
            passing the path ``script_file``  will profile the whole
            script via AST rewriting;
            the objects can be specified using its dotted path or
            file-system path (if applicable).

        profile_imports (bool):
            if :py:const:`True`, when rewriting the AST, profile all its
            imports aswell.

        as_module (bool):
            whether we're running ``script_file`` as a module.

        config (os.PathLike[str] | str | None):
            optional path to load the session config from.

        profile_star_imports (bool | None):
            whether to profile star-imports (``from ... import *``);
            if :py:const:`None`, the value is taken from ``config``.

        profile_nested_imports \
(Collection[Literal['func_defs', 'class_defs', \
'loops', 'conditionals', 'contexts', 'try_except']] | None):
            Which of the compound-statement types to look for nested
            imports in;
            if :py:const:`None`, it is loaded from the ``config`` (from
            ``autoprofile.import_discovery``)
    """
    if as_module:
        module_name: str | None = modpath_to_modname(
            script_file, hide_main=False, hide_init=False,
        )
        if not module_name:
            raise ModuleNotFoundError(
                f'script_file = {script_file!r}: '
                'cannot find corresponding module'
            )

        module_obj = types.ModuleType(module_name)
        # Set the `__spec__` correctly
        module_obj.__spec__ = importlib.util.find_spec(module_name)
    else:
        module_name = None
        module_obj = types.ModuleType('__main__')

    namespace: MutableMapping[str, Any] = vars(module_obj)
    namespace.update(ns)

    if USE_LEGACY_AUTOPROF:
        get_rewritten_ast = _rewrite_ast_legacy
    else:
        get_rewritten_ast = _rewrite_ast_single_pass
    tree_profiled = get_rewritten_ast(
        script_file, module_name,
        prof_mod=prof_mod,
        profile_imports=profile_imports,
        config=config,
        profile_star_imports=profile_star_imports,
        profile_nested_imports=profile_nested_imports,
    )

    _extend_line_profiler_for_profiling_imports(ns[PROFILER_LOCALS_NAME])
    code_obj = compile(tree_profiled, script_file, 'exec')
    with restore.mapping(sys.modules, ['__main__']):
        # Always set the module object to `sys.modules['__main__']` and
        # then restore it via the context manager, so that the executed
        # code is run as `__main__`
        sys.modules['__main__'] = module_obj
        exec(
            code_obj,
            cast('dict[str, Any]', namespace),  # type: ignore[ty:redundant-cast]
            namespace,
        )
