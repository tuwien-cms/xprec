# Copyright (C) 2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
import numpy as np

import xprec
import xprec.linalg
from xprec import ddouble


def test_householder_vec():
    rng = np.random.RandomState(4711)
    xd = rng.random_sample(20)
    xq = np.array(xd, dtype=ddouble)

    betaq, vq = xprec.linalg.householder(xq)
    eq = xq - betaq * vq * (vq @ xq)
    np.testing.assert_allclose(eq[1:].astype(float), 0, atol=1e-31)


def test_bidiag():
    rng = np.random.RandomState(4711)
    m, n = 7, 5
    A = rng.normal(size=(m,n)).astype(ddouble)
    Q, B, RT = xprec.linalg.bidiag(A)
    diff = Q @ B @ RT - A

    # FIXME: too large precision goals
    np.testing.assert_allclose(diff.astype(float), 0, atol=1e-29)


def test_givens():
    f, g = np.array([3.0, -2.0], dtype=ddouble)
    c, s, r = xprec.linalg.givens_rotation(f, g)

    R = np.reshape([c, s, -s, c], (2,2))
    v = np.hstack([f, g])
    w = np.hstack([r, np.zeros_like(r)])
    res = R @ v - w
    np.testing.assert_allclose(res.astype(float), 0, atol=1e-31)


def test_givens():
    a = np.array([3.0, -2.0], dtype=ddouble)
    r, G = xprec.linalg.givens(a)
    diff = r - G @ a
    np.testing.assert_allclose(diff.astype(float), 0, atol=1e-31)


def test_qr():
    A = np.vander(np.linspace(-1, 1, 60), 80).astype(ddouble)
    Q, R = xprec.linalg.qr(A)
    I_m = np.eye(60)
    D = Q @ Q.T - I_m
    np.testing.assert_allclose(D.astype(float), 0, atol=4e-30)
    D = Q @ R - A
    np.testing.assert_allclose(D.astype(float), 0, atol=4e-30)


def test_qr_pivot():
    A = np.vander(np.linspace(-1, 1, 60), 80).astype(ddouble)
    Q, R, piv = xprec.linalg.rrqr(A)
    I_m = np.eye(60)
    D = Q @ Q.T - I_m
    np.testing.assert_allclose(D.astype(float), 0, atol=4e-30)

    D = Q @ R - A[:,piv]
    np.testing.assert_allclose(D.astype(float), 0, atol=4e-30)

    Rdiag = np.abs(R.diagonal())
    assert (Rdiag[1:] <= Rdiag[:-1]).all()


def test_jacobi():
    A = np.vander(np.linspace(-1, 1, 60), 80).astype(ddouble)
    U, s, VT = xprec.linalg.svd_trunc(A)
    np.testing.assert_allclose((U * s) @ VT - A, 0.0, atol=5e-30)
