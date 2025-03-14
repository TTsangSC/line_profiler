"""
Deprecated alias to `~~.ast_profile_transformer`.
"""
import warnings

from . import ast_profile_transformer as _canonical


_this_module = (lambda: None).__module__
_canonical_module = _canonical.__spec__.name


def __getattr__(attr):
    value = getattr(_canonical, attr)
    msg = ('`{0}.{1}` is deprecated; use `{2}.{1}` instead'
           .format(_this_module, attr, _canonical_module))
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return globals().setdefault(attr, value)


def __dir__():
    return list({*globals(), *dir(_canonical)})
