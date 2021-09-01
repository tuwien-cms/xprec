import numpy as np

from . import _raw

DTYPE = np.dtype([("hi", float), ("lo", float)])

_RAW_DTYPE = _raw.dtype

_UFUNC_SUPPORTED = (
    "add", "subtract", "multiply", "true_divide",
    "positive", "negative", "absolute", "floor", "ceil",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    "square", "sqrt", "exp", "expm1", "log",
    "sin", "cos", "sinh", "cosh", "tanh"
    )
_UFUNC_TABLE = {getattr(np, name): getattr(_raw, name)
                for name in _UFUNC_SUPPORTED}

class DDArray(np.ndarray):
    def __new__(cls, shape, buffer=None, offset=0, strides=None, order=None):
        """Create new ndarray instance (needs to be done using __new__)"""
        return super().__new__(cls, shape, DTYPE, buffer, offset, strides,
                               order)

    def __array_ufunc__(self, ufunc, method, *in_, out=None, **kwds):
        """Override what happens when executing numpy ufunc."""
        ufunc = _UFUNC_TABLE[ufunc]
        in_ = map(self._strip, in_)
        if out:
            out = tuple(map(self._strip, out))

        res = super().__array_ufunc__(ufunc, method, *in_, out=out, **kwds)
        if res is NotImplemented:
            return res
        if ufunc.nout == 1:
            return self._dress(res)
        else:
            return tuple(map(self._dress, res))

    def _strip(self, arr):
        arr = np.asarray(arr)
        if arr.dtype == DTYPE:
            return arr.view(_RAW_DTYPE)
        return arr

    def _dress(self, arr):
        if arr.dtype == _RAW_DTYPE:
            return arr.view(DTYPE, self.__class__)
        return arr


def asdoubledouble(arr, copy=False):
    arr = np.array(arr, copy=copy, subok=True)
    if arr.dtype == DTYPE:
        return arr.view(DDArray)

    dd_arr = DDArray(arr.shape)
    dd_arr["hi"] = arr
    dd_arr["lo"] = 0
    return dd_arr