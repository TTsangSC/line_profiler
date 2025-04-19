#!/usr/bin/env python
"""
Script to conveniently run profilers on code in a variety of circumstances.

To profile a script, decorate the functions of interest with ``@profile``

.. code:: bash

    echo "if 1:
        @profile
        def main():
            1 + 1
        main()
    " > script_to_profile.py

NOTE:

    New in 4.1.0: Instead of relying on injecting ``profile`` into the builtins
    you can now ``import line_profiler`` and use ``line_profiler.profile`` to
    decorate your functions. This allows the script to remain functional even
    if it is not actively profiled. See :py:mod:`line_profiler` for details.


Then run the script using kernprof:

.. code:: bash

    kernprof -b script_to_profile.py

By default this runs with the default :py:mod:`cProfile` profiler and does not
require compiled modules. Instructions to view the results will be given in the
output. Alternatively, adding ``-v`` to the command line will write results to
stdout.

To enable line-by-line profiling, then :py:mod:`line_profiler` must be
available and compiled. Add the ``-l`` argument to the kernprof invocation.

.. code:: bash

    kernprof -lb script_to_profile.py


For more details and options, refer to the CLI help.
To view kernprof help run:

.. code:: bash

    kernprof --help

which displays:

.. code::

    usage: kernprof [-h] [-V] [-l] [-b] [-o OUTFILE] [-s SETUP] [-v] [-r] [-u UNIT] [-z] [-i [OUTPUT_INTERVAL]] [-p PROF_MOD] [-m] [--prof-imports] {script | -m module} ...

    Run and profile a python script.

    positional arguments:
      {script | -m module}  The python script file or module to run
      args                  Optional script arguments

    options:
      -h, --help            show this help message and exit
      -V, --version         show program's version number and exit
      -l, --line-by-line    Use the line-by-line profiler instead of cProfile. Implies --builtin.
      -b, --builtin         Put 'profile' in the builtins. Use 'profile.enable()'/'.disable()', '@profile' to decorate functions, or 'with profile:' to profile a section of code.
      -o OUTFILE, --outfile OUTFILE
                            Save stats to <outfile> (default: 'scriptname.lprof' with --line-by-line, 'scriptname.prof' without)
      -s SETUP, --setup SETUP
                            Code to execute before the code to profile
      -v, --view            View the results of the profile in addition to saving it
      -r, --rich            Use rich formatting if viewing output
      -u UNIT, --unit UNIT  Output unit (in seconds) in which the timing info is displayed (default: 1e-6)
      -z, --skip-zero       Hide functions which have not been called
      -i [OUTPUT_INTERVAL], --output-interval [OUTPUT_INTERVAL]
                            Enables outputting of cumulative profiling results to file every n seconds. Uses the threading module. Minimum value is 1 (second). Defaults to disabled.
      -p, --prof-mod PROF_MOD
                            List of modules, functions and/or classes to profile specified by their name or path. List is comma separated, adding the current script path profiles
                            the full script. Multiple copies of this flag can be supplied and the.list is extended. Only works with line_profiler -l, --line-by-line
      --prof-imports        If specified, modules specified to `--prof-mod` will also autoprofile modules that they import. Only works with line_profiler -l, --line-by-line
"""
import argparse
import builtins
import contextlib
import functools
import os
import sys
import threading
import asyncio  # NOQA
import concurrent.futures  # NOQA
import time
from runpy import run_module

# NOTE: This version needs to be manually maintained in
# line_profiler/line_profiler.py and line_profiler/__init__.py as well
__version__ = '4.3.0'

# Guard the import of cProfile such that 3.x people
# without lsprof can still use this script.
try:
    from cProfile import Profile
except ImportError:
    from profile import Profile  # type: ignore[assignment,no-redef]

import line_profiler
from line_profiler.profiler_mixin import ByCountProfilerMixin


def execfile(filename, globals=None, locals=None):
    """ Python 3.x doesn't have 'execfile' builtin """
    with open(filename, 'rb') as f:
        exec(compile(f.read(), filename, 'exec'), globals, locals)
# =====================================


class ContextualProfile(ByCountProfilerMixin, Profile):
    """ A subclass of Profile that adds a context manager for Python
    2.5 with: statements and a decorator.
    """
    def __init__(self, *args, **kwds):
        super(ByCountProfilerMixin, self).__init__(*args, **kwds)
        self.enable_count = 0

    def __call__(self, func):
        return self.wrap_callable(func)

    def enable_by_count(self, subcalls=True, builtins=True):
        """ Enable the profiler if it hasn't been enabled before.
        """
        if self.enable_count == 0:
            self.enable(subcalls=subcalls, builtins=builtins)
        self.enable_count += 1

    def disable_by_count(self):
        """ Disable the profiler if the number of disable requests matches the
        number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    # FIXME: `profile.Profile` is fundamentally incompatible with the
    # by-count paradigm we use, as it can't be `.enable()`-ed nor
    # `.disable()`-ed


class RepeatedTimer:
    """
    Background timer for outputting file every n seconds.

    Adapted from [SO474528]_.

    References:
        .. [SO474528] https://stackoverflow.com/questions/474528/execute-function-every-x-seconds/40965385#40965385
    """
    def __init__(self, interval, dump_func, outfile):
        self._timer = None
        self.interval = interval
        self.dump_func = dump_func
        self.outfile = outfile
        self.is_running = False
        self.next_call = time.time()
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.dump_func(self.outfile)

    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = threading.Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def find_module_script(module_name, *, exit_on_error=True):
    """Find the path to the executable script for a module or package."""
    from line_profiler.autoprofile.util_static import modname_to_modpath

    for suffix in '.__main__', '':
        fname = modname_to_modpath(module_name + suffix)
        if fname:
            return fname

    msg = f'Could not find module `{module_name}`'
    if exit_on_error:
        print(msg, file=sys.stderr)
        raise SystemExit(1)
    else:
        raise ModuleNotFoundError(msg)


def find_script(script_name, *, exit_on_error=True):
    """ Find the script.

    If the input is not a file, then $PATH will be searched.
    """
    if os.path.isfile(script_name):
        return script_name
    path = os.getenv('PATH', os.defpath).split(os.pathsep)
    for dir in path:
        if dir == '':
            continue
        fn = os.path.join(dir, script_name)
        if os.path.isfile(fn):
            return fn

    msg = f'Could not find script {script_name!r}'
    if exit_on_error:
        print(msg, file=sys.stderr)
        raise SystemExit(1)
    else:
        raise FileNotFoundError(msg)


def _python_command():
    """
    Return a command that corresponds to :py:obj:`sys.executable`.
    """
    import shutil
    if shutil.which('python') == sys.executable:
        return 'python'
    elif shutil.which('python3') == sys.executable:
        return 'python3'
    else:
        return sys.executable


@contextlib.contextmanager
def _restore_list(lst):
    """
    Restore a list like `sys.path` after running code which potentially
    modifies it.

    Example
    -------
    >>> l = [1, 2, 3]
    >>>
    >>>
    >>> with _restore_list(l):
    ...     print(l)
    ...     l.append(4)
    ...     print(l)
    ...     l[:] = 5, 6
    ...     print(l)
    ...
    [1, 2, 3]
    [1, 2, 3, 4]
    [5, 6]
    >>> l
    [1, 2, 3]
    """
    old = lst.copy()
    yield
    lst[:] = old


def pre_parse_single_arg_directive(args, flag, sep='--'):
    """
    Pre-parse high-priority single-argument directives like `-m module`
    to emulate the behavior of `python [...]`.

    Examples
    --------
    >>> import functools
    >>> pre_parse = functools.partial(pre_parse_single_arg_directive,
    ...                               flag='-m')

    Normal parsing:

    >>> pre_parse(['foo', 'bar', 'baz'])
    (['foo', 'bar', 'baz'], None, [])
    >>> pre_parse(['foo', 'bar', '-m', 'baz'])
    (['foo', 'bar'], 'baz', [])
    >>> pre_parse(['foo', 'bar', '-m', 'baz', 'foobar'])
    (['foo', 'bar'], 'baz', ['foobar'])

    Erroneous case:

    >>> pre_parse(['foo', 'bar', '-m'])
    Traceback (most recent call last):
      ...
    ValueError: argument expected for the -m option

    Prevent erroneous consumption of the flag by passing it `'--'`:

    >>> pre_parse(['foo', '--', 'bar', '-m', 'baz'])
    (['foo', '--'], None, ['bar', '-m', 'baz'])
    >>> pre_parse(['foo', '-m', 'spam',
    ...            'eggs', '--', 'bar', '-m', 'baz'])
    (['foo'], 'spam', ['eggs', '--', 'bar', '-m', 'baz'])
    """
    args = list(args)
    pre = []
    post = []
    try:
        i_sep = args.index(sep)
    except ValueError:  # No such element
        pass
    else:
        pre[:] = args[:i_sep]
        post[:] = args[i_sep + 1:]
        pre_pre, arg, pre_post = pre_parse_single_arg_directive(pre, flag)
        if arg is None:
            assert not pre_post
            return pre_pre + [sep], arg, post
        else:
            return pre_pre, arg, [*pre_post, sep, *post]
    try:
        i_flag = args.index(flag)
    except ValueError:  # No such element
        return args, None, []
    if i_flag == len(args) - 1:  # Last element
        raise ValueError(f'argument expected for the {flag} option')
    return args[:i_flag], args[i_flag + 1], args[i_flag + 2:]


@_restore_list(sys.argv)
@_restore_list(sys.path)
def main(args=None, exit_on_error=True):
    """
    Runs the command line interface
    """
    # Make the absent value a list so that the `append` and `extend`
    # action still works
    def positive_float(value):
        val = float(value)
        if val <= 0:
            raise argparse.ArgumentError
        return val

    def add_argument(parser_like, *args,
                     hide_complementary_flag=True, **kwargs):
        """
        Override the 'store_true' and 'store_false' actions so that they
        are turned into 'store_const' options which don't set the
        default to the opposite boolean, thus allowing us to later
        distinguish between cases where the flag has been passed or not.

        Also automatically generates complementary boolean options for
        `action='store_true'` options. If `hide_complementary_flag` is
        true, the auto-generated option (all the long flags prefixed
        with 'no-', e.g. '--foo' is negated by '--no-foo') is hidden
        from the help text.
        """
        if kwargs.get('action') not in ('store_true', 'store_false'):
            return parser_like.add_argument(*args, **kwargs)
        kwargs['const'] = kwargs['action'] == 'store_true'
        kwargs['action'] = 'store_const'
        kwargs.setdefault('default', None)
        if kwargs['action'] == 'store_false':
            return parser_like.add_argument(*args, **kwargs)
        # Automatically generate a complementary option for a boolean
        # option;
        # for convenience, turn it into a `store_const` action
        # (in Python 3.9+ one can use `argparse.BooleanOptionalAction`,
        # but we want to maintain compatibility with Python 3.8)
        action = parser_like.add_argument(*args, **kwargs)
        long_flags = [arg for arg in args if arg.startswith('--')]
        assert long_flags
        if hide_complementary_flag:
            falsy_help_text = argparse.SUPPRESS
        else:
            falsy_help_text = 'Negate these flags: ' + ', '.join(args)
        parser_like.add_argument(*('--no-' + flag[2:] for flag in long_flags),
                                 **{**kwargs,
                                    'const': False,
                                    'dest': action.dest,
                                    'help': falsy_help_text})
        return action

    create_parser = functools.partial(
        argparse.ArgumentParser,
        description='Run and profile a python script.')

    if args is None:
        args = sys.argv[1:]

    # Special case: `kernprof [...] -m <module>` should terminate the
    # parsing of all subsequent options
    args, module, post_args = pre_parse_single_arg_directive(args, '-m')

    if module is None:  # Normal execution
        real_parser, = parsers = [create_parser()]
        help_parser = None
    else:
        # We've already consumed the `-m <module>`, so we need a dummy
        # parser for generating the help text;
        # but the real parser should not consume the `options.script`
        # positional arg, and it it got the `--help` option, it should
        # hand off the the dummy parser
        real_parser = create_parser(add_help=False)
        real_parser.add_argument('-h', '--help', action='store_true')
        help_parser = create_parser()
        parsers = [real_parser, help_parser]
    for parser in parsers:
        add_argument(parser, '-V', '--version',
                     action='version', version=__version__)
        add_argument(parser, '-l', '--line-by-line', action='store_true',
                     help='Use the line-by-line profiler instead of cProfile. '
                     'Implies --builtin.')
        add_argument(parser, '-b', '--builtin', action='store_true',
                     help="Put 'profile' in the builtins. "
                     "Use 'profile.enable()'/'.disable()', "
                     "'@profile' to decorate functions, "
                     "or 'with profile:' to profile a section of code.")
        add_argument(parser, '-o', '--outfile',
                     help='Save stats to <outfile> '
                     "(default: 'scriptname.lprof' with --line-by-line, "
                     "'scriptname.prof' without)")
        add_argument(parser, '-s', '--setup',
                     help='Code to execute before the code to profile')
        add_argument(parser, '-v', '--view', action='store_true',
                     help='View the results of the profile '
                     'in addition to saving it')
        add_argument(parser, '-r', '--rich', action='store_true',
                     help='Use rich formatting if viewing output')
        add_argument(parser, '-u', '--unit', default='1e-6', type=positive_float,
                     help='Output unit (in seconds) in which '
                     'the timing info is displayed (default: %(default)s)')
        add_argument(parser, '-z', '--skip-zero', action='store_true',
                     help="Hide functions which have not been called")
        add_argument(parser, '-i', '--output-interval',
                     type=int, default=0, const=0, nargs='?',
                     help="Enables outputting of cumulative profiling results "
                     "to file every n seconds. Uses the threading module. "
                     "Minimum value is 1 (second). Defaults to disabled.")
        add_argument(parser, '-p', '--prof-mod', action='append', type=str,
                     help="List of modules, functions and/or classes "
                     "to profile specified by their name or path. "
                     "List is comma separated, adding the current script path "
                     "profiles the full script. "
                     "Multiple copies of this flag can be supplied and "
                     "the list is extended. "
                     "Only works with line_profiler -l, --line-by-line")
        add_argument(parser, '--prof-imports', action='store_true',
                     help="If specified, modules specified to `--prof-mod` "
                     "will also autoprofile modules that they import. "
                     "Only works with line_profiler -l, --line-by-line")

        if parser is help_parser or module is None:
            add_argument(parser, 'script',
                         metavar='{script | -m module}',
                         help='The python script file or module to run')
        add_argument(parser, 'args',
                     nargs='...', help='Optional script arguments')

    # Hand off to the dummy parser if necessary to generate the help
    # text
    try:
        options = real_parser.parse_args(args)
    except SystemExit as e:
        # If `exit_on_error` is true, let `SystemExit` bubble up and
        # kill the interpretor;
        # else, catch and handle it more gracefully
        # (Note: can't use `ArgumentParser(exit_on_error=False)` in
        # Python 3.8)
        if exit_on_error:
            raise
        elif e.code:
            raise RuntimeError from None
        else:
            return
    if help_parser and getattr(options, 'help', False):
        help_parser.print_help()
        if exit_on_error:
            raise SystemExit(0)
        else:
            return

    try:
        del options.help
    except AttributeError:
        pass
    # Add in the pre-partitioned arguments cut off by `-m <module>`
    options.args += post_args
    if module is not None:
        options.script = module

    if not options.outfile:
        extension = 'lprof' if options.line_by_line else 'prof'
        options.outfile = '%s.%s' % (os.path.basename(options.script), extension)

    sys.argv = [options.script] + options.args
    if module:
        # Make sure the current directory is on `sys.path` to emulate
        # `python -m`
        # Note: this NEEDS to happen here, before the setup script (or
        # any other code) has a chance to `os.chdir()`
        sys.path.insert(0, os.path.abspath(os.curdir))
    if options.setup:
        # Run some setup code outside of the profiler. This is good for
        # large imports.
        setup_file = find_script(options.setup, exit_on_error=exit_on_error)
        __file__ = setup_file
        __name__ = '__main__'
        # Make sure the script's directory is on sys.path instead of just
        # kernprof.py's.
        sys.path.insert(0, os.path.dirname(setup_file))
        ns = locals()
        execfile(setup_file, ns, ns)

    if options.line_by_line:
        prof = line_profiler.LineProfiler()
        options.builtin = True
    elif Profile.__module__ == 'profile':
        raise RuntimeError('non-line-by-line profiling depends on cProfile, '
                           'which is not available on this platform')
    else:
        prof = ContextualProfile()

    # Overwrite the explicit decorator
    global_profiler = line_profiler.profile
    install_profiler = global_profiler._kernprof_overwrite
    install_profiler(prof)

    if options.builtin:
        builtins.__dict__['profile'] = prof

    if module:
        script_file = find_module_script(options.script,
                                         exit_on_error=exit_on_error)
    else:
        script_file = find_script(options.script, exit_on_error=exit_on_error)
        # Make sure the script's directory is on sys.path instead of
        # just kernprof.py's.
        sys.path.insert(0, os.path.dirname(script_file))
    __file__ = script_file
    __name__ = '__main__'

    if options.output_interval:
        # XXX: why are we doing this here (5a38626) and again below?
        rt = RepeatedTimer(max(options.output_interval, 1), prof.dump_stats, options.outfile)
    original_stdout = sys.stdout
    if options.output_interval:
        rt = RepeatedTimer(max(options.output_interval, 1), prof.dump_stats, options.outfile)
    try:
        try:
            execfile_ = execfile
            rmod_ = run_module
            ns = locals()
            if options.prof_mod and options.line_by_line:
                from line_profiler.autoprofile import autoprofile
                # Note: `prof_mod` entries can be filenames (which can
                # contain commas), so check against existing filenames
                # before splitting them
                prof_mod = sum(
                    ([spec] if os.path.exists(spec) else spec.split(',')
                     for spec in options.prof_mod),
                    [])
                autoprofile.run(script_file,
                                ns,
                                prof_mod=prof_mod,
                                profile_imports=options.prof_imports,
                                as_module=module is not None)
            elif module and options.builtin:
                run_module(options.script, ns, '__main__')
            elif options.builtin:
                execfile(script_file, ns, ns)
            elif module:
                prof.runctx(f'rmod_({options.script!r}, globals(), "__main__")',
                            ns,
                            ns)
            else:
                prof.runctx('execfile_(%r, globals())' % (script_file,), ns, ns)
        except (KeyboardInterrupt, SystemExit):
            pass
    finally:
        if options.output_interval:
            rt.stop()
        prof.dump_stats(options.outfile)
        print('Wrote profile results to %s' % options.outfile)
        if options.view:
            if isinstance(prof, ContextualProfile):
                prof.print_stats()
            else:
                prof.print_stats(output_unit=options.unit,
                                 stripzeros=options.skip_zero,
                                 rich=options.rich,
                                 stream=original_stdout)
        else:
            print('Inspect results with:')
            py_exe = _python_command()
            if isinstance(prof, ContextualProfile):
                print(f'{py_exe} -m pstats "{options.outfile}"')
            else:
                print(f'{py_exe} -m line_profiler -rmt "{options.outfile}"')
        # Restore the state of the global `@line_profiler.profile`
        install_profiler(None)


if __name__ == '__main__':
    main()
