import numpy as np

import xprec
import xprec.linalg


def test_householder_vec():
    rng = np.random.RandomState(4711)
    xd = rng.random_sample(20)
    xq = xprec.ddarray(xd)

    betaq, vq = xprec.linalg.householder_vector(xq)
    eq = xq - betaq * vq * (vq @ xq)
    np.testing.assert_allclose(eq[1:].hi, 0, atol=1e-31)


def test_bidiag():
    rng = np.random.RandomState(4711)
    m, n = 7, 5
    A = xprec.ddarray(rng.normal(size=(m,n)))
    Q_h, B, R_h = xprec.linalg.householder_bidiag(A)

    Q = xprec.ddarray(np.eye(m))
    Q = xprec.linalg.householder_apply(Q_h, Q)
    R = xprec.ddarray(np.eye(n))
    R = xprec.linalg.householder_apply(R_h, R)
    diff = Q @ B @ R.T - A

    # FIXME: too large precision goals
    np.testing.assert_allclose(diff.hi, 0, atol=1e-29)


def test_givens():
    f, g = xprec.ddarray([3.0, -2.0])
    c, s, r = xprec.linalg.givens_rotation(f, g)

    R = xprec.ddarray([c, s, -s, c]).reshape(2, 2)
    v = np.hstack([f, g])
    w = np.hstack([r, np.zeros_like(r)])
    res = R @ v - w
    np.testing.assert_allclose(res.hi, 0, atol=1e-31)


def test_givens():
    a = xprec.ddarray([2.0, -3.0])
    r, G = xprec.linalg.givens(a)
    diff = r - G @ a
    np.testing.assert_allclose(diff.hi, 0, atol=1e-31)


def test_svd_tri2x2():
    A = xprec.ddarray([[2.0, -3.0], [0.0, 4.0]])
    U, s, VH = xprec.linalg.svd_tri2x2(A)
    diff = A - (U * s) @ VH
    np.testing.assert_allclose(diff.hi, 0, atol=2e-31)


def test_svd():
    rng = np.random.RandomState(4711)
    A = rng.normal(size=(30,20))
    Ax = xprec.ddarray(A)
    Ux, sx, VTx = xprec.linalg.svd(Ax)
    diff = (Ux[:,:20] * sx) @ VTx - Ax
    np.testing.assert_allclose(diff.hi, 0, atol=1e-29)
