import numpy as np

import quadruple
import quadruple.linalg


def test_householder_vec():
    rng = np.random.RandomState(4711)
    xd = rng.random_sample(20)
    xq = quadruple.ddarray(xd)

    betad, vd = quadruple.linalg.householder_vector(xd)
    betaq, vq = quadruple.linalg.householder_vector(xq)
    np.testing.assert_array_almost_equal_nulp(betaq.hi, betad, 4)
    np.testing.assert_array_almost_equal_nulp(vq.hi, vd, 4)

    ed = xd - betad * vd * (vd @ xd)
    np.testing.assert_allclose(ed[1:], 0, atol=5e-16)

    eq = xq - betaq * vq * (vq @ xq)
    np.testing.assert_allclose(eq[1:].hi, 0, atol=1e-31)


def test_bidiag():
    rng = np.random.RandomState(4711)
    m, n = 7, 5
    A = quadruple.ddarray(rng.normal(size=(m,n)))
    Q_h, B, R_h = quadruple.linalg.householder_bidiag(A)

    Q = quadruple.ddarray(np.eye(m))
    Q = quadruple.linalg.householder_apply(Q_h, Q)
    R = quadruple.ddarray(np.eye(n))
    R = quadruple.linalg.householder_apply(R_h, R)
    diff = Q @ B @ R.T - A
    np.testing.assert_allclose(diff.hi, 0, atol=5e-31)


def test_givens():
    f, g = quadruple.ddarray([3.0, -2.0])
    c, s, r = quadruple.linalg.givens_rotation(f, g)

    R = quadruple.ddarray([c, s, -s, c]).reshape(2, 2)
    v = np.hstack([f, g])
    w = np.hstack([r, np.zeros_like(r)])
    res = R @ v - w
    np.testing.assert_allclose(res.hi, 0, atol=1e-31)


def test_symmschur():
    X = quadruple.ddarray([3., -2., -2., 5.]).reshape(2, 2)
    c, s = quadruple.linalg.jacobi_symm2x2(X[0,0], X[0,1], X[1,1])
    R = np.array([c, s, -s, c]).reshape(2, 2)
    D = R.T @ X @ R
    np.testing.assert_allclose(D[1,0].hi, 0, atol=1e-31)
    np.testing.assert_allclose(D[0,1].hi, 0, atol=1e-31)

    eval = D[1,1]
    eval_ref = quadruple.linalg.eigval_symm2x2_closeqq(X[0,0], X[0,1], X[1,1])
    np.testing.assert_allclose(eval - eval_ref, 0, atol=1e-30)


def test_givens():
    a = quadruple.ddarray([2.0, -3.0])
    r, G = quadruple.linalg.givens(a)
    diff = r - G @ a
    np.testing.assert_allclose(diff.hi, 0, atol=1e-31)


def test_svd_tri2x2():
    A = quadruple.ddarray([[2.0, -3.0], [0.0, 4.0]])
    U, s, VH = quadruple.linalg.svd_tri2x2(A)
    diff = A - (U * s) @ VH
    np.testing.assert_allclose(diff.hi, 0, atol=2e-31)
