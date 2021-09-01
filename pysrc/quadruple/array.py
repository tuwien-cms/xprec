import numpy as np

from . import _raw

DTYPE = np.dtype([("hi", float), ("lo", float)])

_RAW_DTYPE = _raw.dtype

_UFUNC_TABLE = {
    np.add: _raw.add,
    np.subtract: _raw.sub,
    np.multiply: _raw.mul,
    np.true_divide: _raw.div,
    np.floor_divide: _raw.div,
    np.positive: _raw.pos,
    np.negative: _raw.neg,
    np.absolute: _raw.abs,
    #np.round: _raw.round,        # FIXME: Does not yet support decimals arg
    np.floor: _raw.floor,
    np.ceil: _raw.ceil,
    np.equal: _raw.equal,
    np.not_equal: _raw.notequal,
    np.greater: _raw.greater,
    np.greater_equal: _raw.greaterequal,
    np.less: _raw.less,
    np.less_equal: _raw.lessequal,
    np.square: _raw.sqr,
    np.sqrt: _raw.sqrt,
    np.exp: _raw.exp,
    np.expm1: _raw.expm1,
    np.log: _raw.log,
    np.sin: _raw.sin,
    np.cos: _raw.cos,
    np.sinh: _raw.sinh,
    np.cosh: _raw.cosh,
    np.tanh: _raw.tanh,
    }


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
        if ufunc.nout == 1:
            return self._dress(res)
        else:
            return tuple(map(self._dress, res))

    def _strip(self, arr):
        if arr.dtype == self.dtype:
            return arr.view(_RAW_DTYPE, np.ndarray)
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
