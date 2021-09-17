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


def test_svd_tri2x2():
    A = np.array([[2.0, -3.0], [2.0, 4.0]], dtype=ddouble)
    U, s, VH = xprec.linalg.svd2x2(A)
    diff = A - (U * s) @ VH
    np.testing.assert_allclose(diff.astype(float), 0, atol=2e-31)


def test_svd():
    rng = np.random.RandomState(4711)
    Ax = rng.normal(size=(30,20)).astype(ddouble)
    Ux, sx, VTx = xprec.linalg.svd(Ax)
    diff = (Ux[:,:20] * sx) @ VTx - Ax
    np.testing.assert_allclose(diff.astype(float), 0, atol=1e-29)

    # Check singular value ordering
    assert (sx >= 0).all()
    assert (sx[1:] <= sx[:-1]).all()
    np.testing.assert_allclose(Ux @ Ux.T - np.eye(30), 0, atol=1e-29)
    np.testing.assert_allclose(VTx @ VTx.T - np.eye(20), 0, atol=1e-29)


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


def jacobi_sweep_naive(U, VT):
    m, n = U.shape
    if m < n:
        raise ValueError("not possible")
    offd = 0
    U = U.copy()
    VT = VT.copy()
    for i in range(n-1):
        for j in range(i+1, n):
            subset = [i, j]
            usub = U[:, subset]
            ux = usub.T @ usub
            offd += np.square(ux[1,0])
            G, _, GT = xprec.linalg.svd2x2(ux)
            U[:,subset] = U[:,subset] @ G
            VT[subset] = GT @ VT[subset]
    return U, VT, np.sqrt(offd)


def test_jacobi():
    A = np.vander(np.linspace(-1, 1, 60), 80).astype(ddouble)
    _, R, _ = xprec.linalg.rrqr(A)

    U = R.T
    m, n = U.shape
    VT = np.eye(n)
    U2, VT2, od2 = xprec.linalg.jacobi_sweep(R.T.copy(), np.eye(n))
    U1, VT1, od1 = jacobi_sweep_naive(R.T.copy(), np.eye(n))
    np.testing.assert_allclose(od1 - od2, 0.0, atol=1e-29)
    np.testing.assert_allclose(U1 - U2, 0.0, atol=1e-29)
    np.testing.assert_allclose(VT1 - VT2, 0.0, atol=1e-29)
