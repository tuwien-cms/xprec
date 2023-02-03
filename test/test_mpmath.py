# Copyright (C) 2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np
import pytest

import xprec

EPS = xprec.finfo(xprec.ddouble).eps

try:
    import mpmath
except ImportError:
    pytest.skip("No mpmath library avaiable", allow_module_level=True)
else:
    mpmath.mp.prec = 120


def mpf_for_xprec(x):
    """Converts xprec.ddouble array to array of mpmath mpf scalars"""
    x = np.asarray(x)
    if x.dtype != xprec.ddouble:
        raise ValueError("dtype shall be ddouble")

    x_flat = x.ravel()
    x_hi = x_flat.astype(float)
    x_lo = (x_flat - x_hi).astype(float)
    x_mpf = np.array(list(map(mpmath.mpf, x_hi)))
    x_mpf += x_lo
    return x_mpf.reshape(x.shape)


def map_mpmath(fn, x):
    x = np.asarray(x)
    x_flat = x.ravel()
    y_flat = np.array(list(map(fn, x_flat)), dtype=object)
    y = y_flat.reshape(x.shape)
    return y


def check_unary(mpmath_fn, numpy_fn, x, rtol):
    y_ref = map_mpmath(mpmath_fn, x)
    y_our = numpy_fn(x.astype(xprec.ddouble))
    y_float = y_ref.astype(float)

    diff = (y_ref - mpf_for_xprec(y_our)).astype(float)
    ok = np.abs(diff) <= rtol * np.abs(y_float)
    if not ok.all():
        x = x[~ok]
        y_float = y_float[~ok]
        y_our = y_our[~ok]
        diff = diff[~ok]
        reldiff = diff / np.abs(y_float)

        msg = f"{'x':>13s} {'mpmath':>13s} {'xprec':>13s} {'rel diff':>13s}\n"
        msg += "\n".join(f"{xi:13g} {y_refi:13g} {y_ouri:13g} {reldiffi:13g}"
                         for xi, y_refi, y_ouri, reldiffi, _
                         in zip(x, y_float, y_our, reldiff, range(10))
                         )
        raise ValueError(f"not equal to rtol = {rtol:3g}\n" + msg)



def test_sqrt():
    # Once the low part of the ddouble becomes a denormal number, we
    # are in trouble, so we truncate the lower end of the range by
    # another 16 digits
    x = np.geomspace(1e-292, 1e307, 1953)
    check_unary(mpmath.sqrt, np.sqrt, x, 2*EPS)


def test_log():
    x = np.reciprocal(np.geomspace(1e-292, 1e307, 1953))
    check_unary(mpmath.log, np.log, x, 70 * EPS)


def test_exp():
    x = np.geomspace(1e-280, 670, 1511)
    x = np.hstack([-x[::-1], 0, x])
    check_unary(mpmath.exp, np.exp, x, 60 * EPS)
    check_unary(mpmath.expm1, np.expm1, x, 60 * EPS)

    check_unary(mpmath.sinh, np.sinh, x, 60 * EPS)
    check_unary(mpmath.cosh, np.cosh, x, 60 * EPS)
    check_unary(mpmath.tanh, np.tanh, x, 60 * EPS)


def test_sincos():
    x = np.geomspace(1e-280, 4.8 * np.pi, 1511)
    x = np.hstack([-x[::-1], 0, x])
    check_unary(mpmath.sin, np.sin, x, 2 * EPS)
    check_unary(mpmath.cos, np.cos, x, 2 * EPS)
