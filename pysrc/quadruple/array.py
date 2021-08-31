import numpy as np

from . import raw

DTYPE = np.dtype([("hi", float), ("lo", float)])

_RAW_DTYPE = raw.dtype

_UFUNC_TABLE = {
    np.add: raw.add,
    np.subtract: raw.sub,
    np.multiply: raw.mul,
    np.true_divide: raw.div,
    np.floor_divide: raw.div,
    np.positive: raw.pos,
    np.negative: raw.neg,
    np.absolute: raw.abs,
    #np.round: raw.round,        # FIXME: Does not yet support decimals arg
    np.floor: raw.floor,
    np.ceil: raw.ceil,
    np.equal: raw.equal,
    np.not_equal: raw.notequal,
    np.greater: raw.greater,
    np.greater_equal: raw.greaterequal,
    np.less: raw.less,
    np.less_equal: raw.lessequal,
    np.square: raw.sqr,
    np.sqrt: raw.sqrt,
    np.exp: raw.exp,
    np.expm1: raw.expm1,
    np.log: raw.log,
    np.sin: raw.sin,
    np.cos: raw.cos,
    np.sinh: raw.sinh,
    np.cosh: raw.cosh,
    np.tanh: raw.tanh,
    }


class DDArray(np.ndarray):
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
        if isinstance(arr, DDArray):
            return arr.view(_RAW_DTYPE, np.ndarray)
        return arr

    def _dress(self, arr):
        if arr.dtype == _RAW_DTYPE:
            return arr.view(DTYPE, self.__class__)
        return arr
