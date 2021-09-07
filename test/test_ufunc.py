import numpy as np
import quadruple
import quadruple.array


def _compare_ufunc(ufunc, *args, ulps=1):
    fx_d = ufunc(*args)
    fx_q = ufunc(*map(quadruple.ddarray, args)).hi

    # Ensure relative accuracy of 2 ulps
    np.testing.assert_array_almost_equal_nulp(fx_d, fx_q, ulps)


def test_log():
    x = np.geomspace(1e-300, 1e300, 1953)
    _compare_ufunc(np.log, x)


def test_sqrt():
    x = np.geomspace(1e-300, 1e300, 1953)
    _compare_ufunc(np.sqrt, x)


def test_exp():
    x = np.geomspace(1e-300, 700, 4953)
    x = np.hstack([-x[::-1], 0, x])
    _compare_ufunc(np.exp, x)

    # Unfortunately, on Windows expm1 is less precise, so we need to increase
    # the tolerance slightly
    _compare_ufunc(np.expm1, x, ulps=2)


def test_hypot():
    x = np.geomspace(1e-150, 1e150, 47)
    x = np.hstack([-x[::-1], 0, x])
    _compare_ufunc(np.hypot, x, x)


def test_givens():
    a = quadruple.ddarray([2.0, -3.0])
    r, G = quadruple.array.givens(a)
    diff = r - G @ a
    np.testing.assert_allclose(diff.hi, 0, atol=1e-31)
