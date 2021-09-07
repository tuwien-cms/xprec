import numpy as np

from . import _raw

DTYPE = np.dtype([("hi", float), ("lo", float)])

_RAW_DTYPE = _raw.dtype

_UFUNC_SUPPORTED = (
    "add", "subtract", "multiply", "true_divide",
    "positive", "negative", "absolute", "floor", "ceil", "rint",
    "copysign", "sign", "signbit", "isfinite", "isinf", "isnan",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    "square", "sqrt", "reciprocal", "exp", "expm1", "log",
    "sin", "cos", "sinh", "cosh", "tanh", "hypot", "fmin", "fmax",
    "matmul"
    )
_UFUNC_TABLE = {getattr(np, name): getattr(_raw, name)
                for name in _UFUNC_SUPPORTED}

_UFUNC_IMPORTED = (_raw.givens, _raw.svd_tri2x2, _raw.svvals_tri2x2)
_UFUNC_TABLE.update({x: x for x in _UFUNC_IMPORTED})

# TODO: not sure where to put those...
givens = _raw.givens
svd_tri2x2 = _raw.svd_tri2x2
svvals_tri2x2 = _raw.svvals_tri2x2


class Array(np.ndarray):
    def __new__(cls, shape, buffer=None, offset=0, strides=None, order=None):
        """Create new ndarray instance (needs to be done using __new__)"""
        return super().__new__(cls, shape, DTYPE, buffer, offset, strides,
                               order)

    def __array_ufunc__(self, ufunc, method, *in_, **kwds):
        """Override what happens when executing numpy ufunc."""
        ufunc = _UFUNC_TABLE[ufunc]
        in_ = map(_strip, in_)
        try:
            out = kwds["out"]
        except KeyError:
            pass
        else:
            kwds["out"] = tuple(map(_strip_or_none, out))

        res = super().__array_ufunc__(ufunc, method, *in_, **kwds)
        if res is NotImplemented:
            return res
        if ufunc.nout == 1:
            return _dress(res)
        else:
            return tuple(map(_dress, res))

    def __array_function__(self, func, types, in_, kwds):
        res = func(*map(_strip, in_), **kwds)
        if isinstance(res, tuple):
            return tuple(map(_dress, res))
        else:
            return _dress(res)

    def __getitem__(self, item):
        arr = super().__getitem__(item)
        if isinstance(arr, np.void):
            return Scalar(arr)
        return arr

    def __setitem__(self, item, value):
        super().__setitem__(item, asddarray(value))

    @property
    def hi(self):
        return self.view(np.ndarray)["hi"]

    @property
    def lo(self):
        return self.view(np.ndarray)["lo"]


class Scalar(np.ndarray):
    def __new__(cls, obj):
        """Create new ndarray instance (needs to be done using __new__)"""
        return super().__new__(cls, (), DTYPE, obj)

    def __array_ufunc__(self, ufunc, method, *in_, **kwds):
        """Override what happens when executing numpy ufunc."""
        ufunc = _UFUNC_TABLE[ufunc]
        in_ = map(_strip, in_)
        try:
            out = kwds["out"]
        except KeyError:
            pass
        else:
            kwds["out"] = tuple(map(_strip_or_none, out))

        res = super().__array_ufunc__(ufunc, method, *in_, **kwds)
        if res is NotImplemented:
            return res
        if ufunc.nout == 1:
            return _dress(res)
        else:
            return tuple(map(_dress, res))

    def __array_function__(self, func, types, in_, kwds):
        res = func(*map(_strip, in_), **kwds)
        if isinstance(res, tuple):
            return tuple(map(_dress, res))
        else:
            return _dress(res)

    @property
    def _data(self):
        return self.view(np.ndarray)[()]

    @property
    def hi(self):
        return self._data[0]

    @property
    def lo(self):
        return self._data[1]

    def __getitem__(self, indx):
        return self._data[indx]

    def __setitem__(self, indx, value):
        self._data[indx] = value

    def __str__(self):
        return self._data.__str__()

    __repr__ = __str__

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return self._data.__len__()


def asddarray(arr, copy=False):
    return ddarray(arr, copy)


def ddarray(arr_like, copy=True, order='K', ndmin=0):
    arr = np.array(arr_like, copy=copy, order=order, ndmin=ndmin)
    if arr.dtype == DTYPE:
        return arr.view(Array)

    dd_arr = np.empty(arr.shape, DTYPE)
    dd_arr["hi"] = arr
    dd_arr["lo"] = 0
    return dd_arr.view(Array)


def _strip_or_none(arr):
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype == DTYPE:
        return arr.view(_RAW_DTYPE)
    return arr


def _strip(arr):
    arr = np.asarray(arr)
    if arr.dtype == DTYPE:
        return arr.view(_RAW_DTYPE)
    return arr


def _dress(arr):
    if arr.dtype == _RAW_DTYPE:
        arr = arr.view(DTYPE, Array)
        if isinstance(arr, np.void):
            arr = Scalar(arr)
    return arr
