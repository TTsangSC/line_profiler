cdef extern from "Python_wrapper.h":
    ctypedef struct PyObject
    ctypedef struct PyCodeObject
    ctypedef struct PyFrameObject
    ctypedef Py_ssize_t Py_hash_t

ctypedef long long PY_LONG_LONG
# long long int is at least 64 bytes assuming c99
ctypedef unsigned long long int uint64
ctypedef long long int int64


cdef inline int64 compute_line_hash(uint64 block_hash, uint64 linenum) noexcept:
    """
    Compute the hash used to store each line timing in an unordered_map.
    This is fairly simple, and could use some improvement since linenum
    isn't technically random, however it seems to be good enough and
    fast enough for any practical purposes.
    """
    # linenum doesn't need to be int64 but it's really a temporary value
    # so it doesn't matter
    return block_hash ^ linenum
