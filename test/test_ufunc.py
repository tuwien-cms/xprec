import numpy as np
import quadruple


def _compare_ufunc(ufunc, x):
    fx_d = ufunc(x)
    fx_q = ufunc(quadruple.ddarray(x)).hi

    # Ensure relative accuracy of 2 ulps
    np.testing.assert_array_almost_equal_nulp(fx_d, fx_q, 1)


def test_log():
    x = np.geomspace(1e-300, 1e300, 1953)
    _compare_ufunc(np.log, x)


def test_exp():
    x = np.geomspace(1e-300, 700, 4953)
    x = np.hstack([-x[::-1], 0, x])
    _compare_ufunc(np.exp, x)
    _compare_ufunc(np.expm1, x)
