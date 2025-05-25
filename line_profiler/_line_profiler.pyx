# cython: language_level=3
# cython: infer_types=True
# cython: legacy_implicit_noexcept=True
# distutils: language=c++
# distutils: include_dirs = python25.pxd
r"""
This is the Cython backend used in :py:mod:`line_profiler.line_profiler`.

Ignore:
    # Standalone compile instructions for developers
    # Assuming the cwd is the repo root.
    cythonize --annotate --inplace \
        ./line_profiler/_line_profiler.pyx \
        ./line_profiler/timers.c \
        ./line_profiler/unset_trace.c
"""
from .python25 cimport PyFrameObject, PyObject, PyStringObject
from collections.abc import Callable
from functools import wraps
from sys import byteorder
import sys
cimport cython
from cpython.version cimport PY_VERSION_HEX
from libc.stdint cimport int64_t

from libcpp.unordered_map cimport unordered_map
import threading
import opcode
import types

NOP_VALUE: int = opcode.opmap['NOP']

# The Op code should be 2 bytes as stated in
# https://docs.python.org/3/library/dis.html
# if sys.version_info[0:2] >= (3, 11):
NOP_BYTES: bytes = NOP_VALUE.to_bytes(2, byteorder=byteorder)

# This should be true for Python >=3.11a1
HAS_CO_QUALNAME: bool = hasattr(types.CodeType, 'co_qualname')

# long long int is at least 64 bytes assuming c99
ctypedef unsigned long long int uint64
ctypedef long long int int64

cdef extern from "Python_wrapper.h":
    """
    inline PyObject* get_frame_code(PyFrameObject* frame) {
        PyCodeObject* code = PyFrame_GetCode(frame);
        PyObject* ret = PyCode_GetCode(code);
        Py_DECREF(code);
        return ret;
    }
    """
    ctypedef int (*Py_tracefunc)(object self, PyFrameObject *py_frame,
                                 int what, PyObject *arg)
    cdef object get_frame_code(PyFrameObject* frame)
    ctypedef struct PyFrameObject
    ctypedef struct PyCodeObject
    ctypedef long long PY_LONG_LONG
    cdef bint PyCFunction_Check(object obj)
    cdef int PyCode_Addr2Line(PyCodeObject *co, int byte_offset)

    cdef void PyEval_SetProfile(Py_tracefunc func, object arg)
    cdef void PyEval_SetTrace(Py_tracefunc func, object arg)

    ctypedef object (*PyCFunction)(object self, object args)

    ctypedef struct PyMethodDef:
        char *ml_name
        PyCFunction ml_meth
        int ml_flags
        char *ml_doc

    ctypedef struct PyCFunctionObject:
        PyMethodDef *m_ml
        PyObject *m_self
        PyObject *m_module

    # They're actually #defines, but whatever.
    cdef int PyTrace_CALL
    cdef int PyTrace_EXCEPTION
    cdef int PyTrace_LINE
    cdef int PyTrace_RETURN
    cdef int PyTrace_C_CALL
    cdef int PyTrace_C_EXCEPTION
    cdef int PyTrace_C_RETURN

    cdef int PyFrame_GetLineNumber(PyFrameObject *frame)

cdef extern from "c_trace_callbacks.c":
    ctypedef struct TraceCallback:
        Py_tracefunc c_tracefunc
        PyObject *c_traceobj

    cdef TraceCallback *alloc_callback() except *
    cdef void free_callback(TraceCallback *callback)
    cdef void fetch_callback(TraceCallback *callback)
    cdef void restore_callback(TraceCallback *callback)
    cdef int call_callback(TraceCallback *callback, PyFrameObject *py_frame,
                           int what, PyObject *arg)

cdef extern from "timers.c":
    PY_LONG_LONG hpTimer()
    double hpTimerUnit()

cdef struct LineTime:
    int64 code
    int lineno
    PY_LONG_LONG total_time
    long nhits

cdef struct LastTime:
    int f_lineno
    PY_LONG_LONG time

cdef inline int64 compute_line_hash(uint64 block_hash, uint64 linenum):
    """
    Compute the hash used to store each line timing in an unordered_map.
    This is fairly simple, and could use some improvement since linenum
    isn't technically random, however it seems to be good enough and
    fast enough for any practical purposes.
    """
    # linenum doesn't need to be int64 but it's really a temporary value
    # so it doesn't matter
    return block_hash ^ linenum


if PY_VERSION_HEX < 0x030c00b1:  # 3.12.0b1

    def _sys_monitoring_register() -> None: 
        ...

    def _sys_monitoring_deregister() -> None: 
        ...

else:

    def _is_main_thread() -> bool:
        return threading.current_thread() == threading.main_thread()

    def _sys_monitoring_register() -> None:
        if not _is_main_thread():
            return
        mon = sys.monitoring
        mon.use_tool_id(mon.PROFILER_ID, 'line_profiler')

    def _sys_monitoring_deregister() -> None:
        if not _is_main_thread():
            return
        mon = sys.monitoring
        mon.free_tool_id(mon.PROFILER_ID)

def label(code):
    """
    Return a ``(filename, first_lineno, _name)`` tuple for a given code
    object.

    This is the similar labelling as used by the :py:mod:`cProfile`
    module in Python 2.5.

    Note:
        In Python >= 3.11 we use we return qualname for ``_name``.
        In older versions of Python we just return name.
    """
    if isinstance(code, str):
        return ('~', 0, code)  # built-in functions ('~' sorts at the end)
    else:
        if HAS_CO_QUALNAME:
            return (code.co_filename, code.co_firstlineno, code.co_qualname)
        else:
            return (code.co_filename, code.co_firstlineno, code.co_name)


def disable_line_events(trace_func: Callable) -> Callable:
    """
    Return a thin wrapper around ``trace_func()`` which withholds line
    events.  This is for when a frame-local
    :py:attr:`~types.FrameType.f_trace` disables
    :py:attr:`~types.FrameType.f_trace_lines` -- we would like to keep
    line events enabled (so that line profiling works) while
    "unsubscribing" the trace function from it.
    """
    @wraps(trace_func)
    def wrapper(frame, event, args):
        if event == 'line':
            return
        return trace_func(frame, event, args)

    return wrapper


cpdef _code_replace(func, co_code):
    """
    Implements :py:mod:`types.CodeType.replace` for Python < 3.8
    """
    try:
        code = func.__code__
    except AttributeError:
        code = func.__func__.__code__
    if hasattr(code, 'replace'):
        # python 3.8+
        code = code.replace(co_code=co_code)
    else:
        # python <3.8
        co = code
        code = type(code)(co.co_argcount, co.co_kwonlyargcount,
                        co.co_nlocals, co.co_stacksize, co.co_flags,
                        co_code, co.co_consts, co.co_names,
                        co.co_varnames, co.co_filename, co.co_name,
                        co.co_firstlineno, co.co_lnotab, co.co_freevars,
                        co.co_cellvars)
    return code


# Note: this is a regular Python class to allow easy pickling.
class LineStats(object):
    """
    Object to encapsulate line-profile statistics.

    Attributes:

        timings (dict[tuple[str, int, str], \
list[tuple[int, int, int]]]):
            Mapping from ``(filename, first_lineno, function_name)`` of
            the profiled function to a list of
            ``(lineno, nhits, total_time)`` tuples for each profiled
            line. ``total_time`` is an integer in the native units of
            the timer.

        unit (float):
            The number of seconds per timer unit.
    """
    def __init__(self, timings, unit):
        self.timings = timings
        self.unit = unit


cdef class _ThreadState:
    """
    Helper object for holding the thread-local state; documentations are
    for reference only, and all APIs are to be considered private and
    subject to change.
    """
    cdef TraceCallback *callback
    cdef public object active_instances  # type: set[LineProfiler]
    cdef int _wrap_trace

    def __init__(self, instances=(), wrap_trace=False):
        self.active_instances = set(instances)
        self.callback = NULL
        self.wrap_trace = wrap_trace

    cpdef _handle_enable_event(self, prof):
        instances = self.active_instances
        already_active = bool(instances)
        instances.add(prof)
        if already_active:
            return
        # Register `line_profiler` with `sys.monitoring` in Python 3.12
        # and above;
        # see: https://docs.python.org/3/library/sys.monitoring.html
        _sys_monitoring_register()

        cdef TraceCallback* callback = alloc_callback()
        fetch_callback(callback)
        self.callback = callback
        PyEval_SetTrace(python_trace_callback, self)

    cpdef _handle_disable_event(self, prof):
        instances = self.active_instances
        instances.discard(prof)
        if instances:
            return
        cdef TraceCallback* callback = self.callback
        restore_callback(callback)
        free_callback(callback)
        self.callback = NULL
        # Deregister `line_profiler` with `sys.monitoring` in Python
        # 3.12 and above;
        # see: https://docs.python.org/3/library/sys.monitoring.html
        _sys_monitoring_deregister()

    property wrap_trace:
        def __get__(self):
            return bool(self._wrap_trace)
        def __set__(self, wrap_trace):
            self._wrap_trace = 1 if wrap_trace else 0


cdef class LineProfiler:
    """
    Time the execution of lines of Python code.

    This is the Cython base class for
    :py:class:`line_profiler.line_profiler.LineProfiler`.

    Arguments:
        *functions (types.FunctionType)
            Function objects to be profiled.
        wrap_trace (Optional[bool])
            What to do if there is an existing (non-profiling)
            :py:mod:`sys` trace callback when the profiler is
            :py:meth:`.enable()`-ed:

            :py:const:`True`:
                *Wrap around* said callback: at the end of running our
                trace callback, also run the existing callback.
            :py:const:`False`:
                *Replace* said callback as long as the profiler is
                enabled.
            :py:const:`None` (default):
                For the first instance created, resolves to

                :py:const:`False`
                    If the environment variable
                    :envvar:`LINE_PROFILE_WRAP_TRACE` is undefined, or
                    if it matches any of
                    ``{'', '0', 'off', 'false', 'no'}``
                    (case-insensitive).

                :py:const:`True`
                    Otherwise.

                If there has already been other instances, the value is
                inherited therefrom.

            In any case, when the profiler is :py:meth:`.disable()`-ed,
            it tries to restore the :py:mod:`sys` trace callback (or the
            lack thereof) to the state it was in from when the profiler
            was :py:meth:`.enable()`-ed (but see Notes).

    Example:
        >>> import copy
        >>> import line_profiler
        >>> # Create a LineProfiler instance
        >>> self = line_profiler.LineProfiler()
        >>> # Wrap a function
        >>> copy_fn = self(copy.copy)
        >>> # Call the function
        >>> copy_fn(self)
        >>> # Inspect internal properties
        >>> self.functions
        >>> self.c_last_time
        >>> self.c_code_map
        >>> self.code_map
        >>> self.last_time
        >>> # Print stats
        >>> self.print_stats()

    Notes:
        * ``wrap_trace = True`` helps with using
          :py:class:`LineProfiler` cooperatively with other tools, like
          coverage and debugging tools.
        * However, it should be considered experimental and to be used
          at one's own risk -- because tools generally assume that they
          have sole control over system-wide tracing.
        * When setting ``wrap_trace``, it is set process-wide for all
          instances.
        * In general, Python allows for trace callbacks to unset
          themselves, either intentionally (via ``sys.settrace(None)``)
          or if it errors out.  If the wrapped/cached trace callback
          does so, profiling would continue, but:

          * The cached callback is cleared and is no longer called, and
          * The :py:mod:`sys` trace callback is set to :py:const:`None`

          when the profiler is :py:meth:`.disable()`-ed.
        * It is also allowed for the frame-local trace callable
          (:py:attr:`~types.FrameType.f_trace`) to set
          :py:attr:`~types.FrameType.f_trace_lines` to false in a frame
          to disable line events.  If the wrapped/cached trace callback
          does so, profiling would continue, but
          :py:attr:`~types.FrameType.f_trace` will no longer receive
          line events.
    """
    cdef unordered_map[int64, unordered_map[int64, LineTime]] _c_code_map
    # Mapping between thread-id and map of LastTime
    cdef unordered_map[int64, unordered_map[int64, LastTime]] _c_last_time
    cdef public list functions
    cdef public dict code_hash_map, dupes_map
    cdef public double timer_unit
    cdef public object threaddata

    # This is shared between instances and threads
    _all_thread_states = {}  # type: dict[int, _ThreadState]

    def __init__(self, *functions, wrap_trace=None):
        self.functions = []
        self.code_hash_map = {}
        self.dupes_map = {}
        self.timer_unit = hpTimerUnit()
        # Create a data store for thread-local objects
        # https://docs.python.org/3/library/threading.html#thread-local-data
        self.threaddata = threading.local()
        if wrap_trace is not None:
            self.wrap_trace = wrap_trace

        for func in functions:
            self.add_function(func)

    cpdef add_function(self, func):
        """
        Record line profiling information for the given Python function.

        Note:
            This is a low-level method and is intended for
            :py:class:`types.FunctionType`; users should in general use
            :py:meth:`line_profiler.LineProfiler.add_callable` for
            adding general callables and callable wrappers (e.g.
            :py:class:`property`).
        """
        if hasattr(func, "__wrapped__"):
            import warnings
            warnings.warn(
                "Adding a function with a __wrapped__ attribute. You may want "
                "to profile the wrapped function by adding %s.__wrapped__ "
                "instead." % (func.__name__,)
            )
        try:
            code = func.__code__
        except AttributeError:
            try:
                code = func.__func__.__code__
            except AttributeError:
                import warnings
                warnings.warn(
                    "Could not extract a code object for the object %r"
                    % (func,))
                return

        if code.co_code in self.dupes_map:
            self.dupes_map[code.co_code] += [code]
            # code hash already exists, so there must be a duplicate
            # function. add no-op
            co_padding : bytes = NOP_BYTES * (len(self.dupes_map[code.co_code])
                                              + 1)
            co_code = code.co_code + co_padding
            CodeType = type(code)
            code = _code_replace(func, co_code=co_code)
            try:
                func.__code__ = code
            except AttributeError as e:
                func.__func__.__code__ = code
        else:
            self.dupes_map[code.co_code] = [code]
        # TODO: Since each line can be many bytecodes, this is kinda
        # inefficient
        # See if this can be sped up by not needing to iterate over
        # every byte
        for offset, byte in enumerate(code.co_code):
            code_hash = compute_line_hash(
                hash((code.co_code)),
                PyCode_Addr2Line(<PyCodeObject*>code, offset))
            if not self._c_code_map.count(code_hash):
                try:
                    self.code_hash_map[code].append(code_hash)
                except KeyError:
                    self.code_hash_map[code] = [code_hash]
                self._c_code_map[code_hash]

        self.functions.append(func)

    property enable_count:
        def __get__(self):
            if not hasattr(self.threaddata, 'enable_count'):
                self.threaddata.enable_count = 0
            return self.threaddata.enable_count
        def __set__(self, value):
            self.threaddata.enable_count = value

    # These two are shared between instances, but thread-local
    # (Ideally speaking they could've been class attributes...)

    property wrap_trace:
        def __get__(self):
            return self._thread_state.wrap_trace
        def __set__(self, wrap_trace):
            # Make sure we have a thread state
            state = self._thread_state
            # Sync values between all thread states
            for state in self._all_thread_states.values():
                state.wrap_trace = wrap_trace

    property _thread_state:
        def __get__(self):
            thread_id = threading.get_ident()
            try:
                return self._all_thread_states[thread_id]
            except KeyError:
                pass
            # First profiler instance on the thread, get the correct
            # `wrap_trace` value and set up a `_ThreadState`
            try:
                state, *_ = self._all_thread_states.values()
            except ValueError:
                # First thread in the interpretor: load default
                # `wrap_trace` value from the environment
                # (TODO: migrate to `line_profiler.cli_utils.boolean()`
                # after merging #335)
                from os import environ

                env = environ.get('LINE_PROFILE_WRAP_TRACE', '').lower()
                wrap_trace = env not in {'', '0', 'off', 'false', 'no'}
            else:
                # Fetch the `.wrap_trace` value from an existing state
                wrap_trace = state.wrap_trace
            self._all_thread_states[thread_id] = state = _ThreadState(
                wrap_trace=wrap_trace)
            return state

    def enable_by_count(self):
        """ Enable the profiler if it hasn't been enabled before.
        """
        if self.enable_count == 0:
            self.enable()
        self.enable_count += 1

    def disable_by_count(self):
        """
        Disable the profiler if the number of disable requests matches
        (or exceeds) the number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()

    def enable(self):
        self._thread_state._handle_enable_event(self)

    @property
    def c_code_map(self):
        """
        A Python view of the internal C lookup table.
        """
        return <dict>self._c_code_map

    @property
    def c_last_time(self):
        return (<dict>self._c_last_time)[threading.get_ident()]

    @property
    def code_map(self):
        """
        :py:mod:`line_profiler` 4.0 no longer directly maintains
        :py:attr:`~.code_map`, but this will construct something similar
        for backwards compatibility.
        """
        c_code_map = self.c_code_map
        code_hash_map = self.code_hash_map
        py_code_map = {}
        for code, code_hashes in code_hash_map.items():
            py_code_map.setdefault(code, {})
            for code_hash in code_hashes:
                c_entries = c_code_map[code_hash]
                py_entries = {}
                for key, c_entry in c_entries.items():
                    py_entry = c_entry.copy()
                    py_entry['code'] = code
                    py_entries[key] = py_entry
                py_code_map[code].update(py_entries)
        return py_code_map

    @property
    def last_time(self):
        """
        :py:mod:`line_profiler` 4.0 no longer directly maintains
        :py:attr:`~.last_time`, but this will construct something similar
        for backwards compatibility.
        """
        c_last_time = (<dict>self._c_last_time)[threading.get_ident()]
        code_hash_map = self.code_hash_map
        py_last_time = {}
        for code, code_hashes in code_hash_map.items():
            for code_hash in code_hashes:
                if code_hash in c_last_time:
                    py_last_time[code] = c_last_time[code_hash]
        return py_last_time

    cpdef disable(self):
        self._c_last_time[threading.get_ident()].clear()
        self._thread_state._handle_disable_event(self)

    def get_stats(self):
        """
        Returns:
            :py:class:`LineStats` object containing the timings.
        """
        cdef dict cmap = self._c_code_map

        stats = {}
        for code in self.code_hash_map:
            entries = []
            for entry in self.code_hash_map[code]:
                entries += list(cmap[entry].values())
            key = label(code)

            # Merge duplicate line numbers, which occur for branch
            # entry points like `if`
            nhits_by_lineno = {}
            total_time_by_lineno = {}

            for line_dict in entries:
                _, lineno, total_time, nhits = line_dict.values()
                nhits_by_lineno[lineno] = (
                    nhits_by_lineno.setdefault(lineno, 0) + nhits)
                total_time_by_lineno[lineno] = (
                    total_time_by_lineno.setdefault(lineno, 0) + total_time)

            entries = [(lineno, nhits, total_time_by_lineno[lineno])
                       for lineno, nhits in nhits_by_lineno.items()]
            entries.sort()

            # NOTE: v4.x may produce more than one entry per line. For
            # example:
            #   1:  for x in range(10):
            #   2:      pass
            #  will produce a 1-hit entry on line 1, and 10-hit entries
            #  on lines 1 and 2
            #  This doesn't affect `print_stats`, because it uses the
            #  last entry for a given line (line number is used a dict
            #  key so earlier entries are overwritten), but to keep
            #  compatability with other tools, let's only keep the last
            #  entry for each line
            # Remove all but the last entry for each line
            entries = list({e[0]: e for e in entries}.values())
            stats[key] = entries
        return LineStats(stats, self.timer_unit)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef extern int python_trace_callback(object state_,
                                      PyFrameObject *py_frame,
                                      int what, PyObject *arg):
    """
    The :c:func:`PyEval_SetTrace` callback.

    References:
       https://github.com/python/cpython/blob/de2a4036/Include/cpython/\
pystate.h#L16
    """
    cdef _ThreadState state
    cdef object prof_
    cdef LineProfiler prof
    cdef LastTime old
    cdef int key
    cdef PY_LONG_LONG time
    cdef int has_time = 0
    cdef int64 code_hash
    cdef int64 block_hash
    cdef unordered_map[int64, LineTime] line_entries
    cdef uint64 linenum

    state = <_ThreadState>state_

    if what == PyTrace_LINE or what == PyTrace_RETURN:
        # Normally we'd need to DECREF the return from get_frame_code,
        # but Cython does that for us
        block_hash = hash(get_frame_code(py_frame))

        linenum = PyFrame_GetLineNumber(py_frame)
        code_hash = compute_line_hash(block_hash, linenum)
        
        for prof_ in state.active_instances:
            prof = <LineProfiler>prof_
            if not prof._c_code_map.count(code_hash):
                continue
            if not has_time:
                time = hpTimer()
                has_time = 1
            ident = threading.get_ident()
            if prof._c_last_time[ident].count(block_hash):
                old = prof._c_last_time[ident][block_hash]
                line_entries = prof._c_code_map[code_hash]
                key = old.f_lineno
                if not line_entries.count(key):
                    prof._c_code_map[code_hash][key] = LineTime(
                        code_hash, key, 0, 0)
                prof._c_code_map[code_hash][key].nhits += 1
                prof._c_code_map[code_hash][key].total_time += time - old.time
            if what == PyTrace_LINE:
                # Get the time again. This way, we don't record much
                # time wasted in this function.
                prof._c_last_time[ident][block_hash] = LastTime(
                    linenum, hpTimer())
            elif prof._c_last_time[ident].count(block_hash):
                # We are returning from a function, not executing a
                # line. Delete the last_time record. It may have already
                # been deleted if we are profiling a generator that is
                # being pumped past its end.
                prof._c_last_time[ident].erase(
                    prof._c_last_time[ident].find(block_hash))

    # Call the trace callback that we're wrapping around where
    # appropriate
    if state._wrap_trace:
        return call_callback(state.callback, py_frame, what, arg)
    return 0
