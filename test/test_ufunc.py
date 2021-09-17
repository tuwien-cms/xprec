# Copyright (C) 2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import xprec


def _compare_ufunc(ufunc, *args, ulps=1):
    fx_d = ufunc(*args)
    fx_q = ufunc(*(a.astype(xprec.ddouble) for a in args)).astype(float)

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


def test_cosh():
    x = np.geomspace(1e-300, 700, 4953)
    x = np.hstack([-x[::-1], 0, x])
    _compare_ufunc(np.cosh, x)
    _compare_ufunc(np.sinh, x)

    thousand = xprec.ddouble.type(1000)
    assert np.isinf(np.cosh(thousand))
    assert np.isinf(np.cosh(-thousand))


def test_hypot():
    x = np.geomspace(1e-300, 1e260, 47)
    x = np.hstack([-x[::-1], 0, x])
    _compare_ufunc(np.hypot, x[:,None], x[None,:])
