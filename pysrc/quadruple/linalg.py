import numpy as np


def householder_vector(x):
    x = np.asanyarray(x)
    xhead = x[:1]
    xtail = x[1:]
    v = x.copy()
    v[0] = 1

    sigma = xtail @ xtail
    if not sigma:
        beta = 0
    else:
        mu = np.sqrt(np.square(xhead) + sigma)
        if xhead <= 0:
            vhead = xhead - mu
        else:
            vhead = -sigma / (xhead + mu)

        vhead2 = np.square(vhead)
        beta = 2 * vhead2 / (sigma + vhead2)
        v[1:] /= vhead

    return beta, v


def householder_update(A, Q):
    beta, v = householder_vector(A[:,0])
    w = beta * (A.T @ v)
    A -= v[:,None] * w[None,:]
    Q[0,0] = beta
    Q[1:,0] = v[1:]


def householder_bidiag(A):
    A = np.array(A, copy=True, subok=True)

    m, n = A.shape
    if m < n:
        raise NotImplementedError("must be tall matrix")

    rq = n - (m == n)
    Q = np.zeros_like(A)
    R = np.zeros_like(A[:n,:n])

    for j in range(n-2):
        householder_update(A[j:,j:], Q[j:,j:])
        householder_update(A[j:,j+1:].T, R[j+1:,j+1:])
    for j in range(n-2, rq):
        householder_update(A[j:,j:], Q[j:,j:])
    return Q, A, R


def householder_apply(H, Q):
    H = np.asanyarray(H)
    Q = Q.copy()
    m, r = H.shape
    if Q.shape != (m, m):
        raise ValueError("invalid shape")
    for j in range(r-1, -1, -1):
        beta = H[j,j]
        if np.equal(beta, 0):
            continue
        v = np.empty_like(H[j:,0])
        v[0] = 1
        v[1:] = H[j+1:,j]
        Qpart = Q[j:,j:]
        w = beta * (Qpart.T @ v)
        Qpart -= v[:,None] * w[None,:]
    return Q


def givens_rotation(f, g):
    # ACM Trans. Math. Softw. 28(2), 206, Alg 1.
    if np.equal(g, 0):
        return np.ones_like(f), np.zeros_like(g), f
    if np.equal(f, 0):
        return np.zeros_like(f), np.sign(g), np.abs(g)

    r = np.copysign(np.hypot(f, g), f)
    inv_r = np.reciprocal(r)
    return f * inv_r, g * inv_r, r
