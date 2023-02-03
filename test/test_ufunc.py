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

    zeroq = xprec.ddouble.type(0)
    assert np.isinf(np.log(zeroq))


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


def test_modf():
    ulps = 1
    x = np.linspace(-100, 100, 100)
    x_d = x.astype(xprec.ddouble)

    fx_d = np.modf(x)
    fx_q = np.modf(x_d)

    # Ensure relative accuracy of 1 ulp
    np.testing.assert_array_almost_equal_nulp(fx_d[0], fx_q[0].astype(float), ulps)
    np.testing.assert_array_almost_equal_nulp(fx_d[1], fx_q[1].astype(float), ulps)


def test_power():
    x = np.linspace(0, 100, 100)
    _compare_ufunc(np.power, x[:,None], x[None,:])


def test_arctan2():
    x = np.linspace(-100, 100, 100)
    _compare_ufunc(np.arctan2, x[:,None], x[None,:], ulps=2)


def test_arcsin():
    x = np.linspace(-1, 1, 100)
    _compare_ufunc(np.arcsin, x, ulps=2)


def test_arccos():
    x = np.linspace(-1, 1, 100)
    _compare_ufunc(np.arccos, x)


def test_arctan():
    x = np.linspace(-100, 100, 100)
    _compare_ufunc(np.arctan, x)


def test_arccosh():
    x = np.linspace(1, 100, 100)
    _compare_ufunc(np.arccosh, x)


def test_arcsinh():
    x = np.linspace(-100, 100, 100)
    _compare_ufunc(np.arcsinh, x)


def test_arctanh():
    x = np.linspace(-0.99, 0.99, 100)
    _compare_ufunc(np.arctanh, x)
