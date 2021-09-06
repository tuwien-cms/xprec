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


def test_givens():
    f, g = quadruple.ddarray([3.0, -2.0])
    c, s, r = quadruple.linalg.givens_rotation(f, g)

    R = quadruple.ddarray((c, s, -s, c)).reshape(2, 2)
    v = np.hstack((f, g))
    w = np.hstack((r, np.zeros_like(r)))
    res = R @ v - w
    np.testing.assert_allclose(res.hi, 0, atol=1e-31)
