"""
Helper utilities for the bookkeeping required for code-obejct bytecodes.
"""
cimport cython
from cython.operator cimport address
from libcpp.unordered_map cimport unordered_map

from ._common_types_and_ops cimport (
    compute_line_hash, PyObject, PyCodeObject, uint64,
)


cdef extern from "Python_wrapper.h":
    cdef int PyCode_Addr2Line(PyCodeObject *co, int byte_offset)


cdef inline void process_opcodes(
        list code_hashes, dict line_hits, PyCodeObject *code, bytes co_code):
    """
    Process the opcodes in the ``code`` object:

    - Gathering the line hashes, and

    - Log line-number regressions in the code, e.g. multiline calls
      whose line-event counts should be deduplicated (see issue #441).

    Notes:
        The default/previous behavior is already technically correct,
        that we do capture and report two line events on the
        callable-invocation line, when a call has some of its
        arguments on different lines from the callable itself.
        However, this provides a "fix" where the numbers can be made
        to align with user expectations.
    """
    cdef uint64 block_hash = hash(co_code)
    cdef uint64 offset
    cdef uint64 lineno, last_lineno = 0

    for offset, _ in enumerate(co_code):
        lineno = PyCode_Addr2Line(code, offset)
        # Calculate hash for the line
        code_hashes.append(compute_line_hash(block_hash, lineno))
        # Keep track of line-number hits
        if lineno != last_lineno:
            try:
                line_hits[lineno] += 1
            except KeyError:
                line_hits[lineno] = 1
            last_lineno = lineno
