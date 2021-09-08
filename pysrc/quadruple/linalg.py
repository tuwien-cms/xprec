import numpy as np

from . import array
from . import _dd_linalg

givens = _dd_linalg.givens
svd_tri2x2 = _dd_linalg.svd_tri2x2
svvals_tri2x2 = _dd_linalg.svvals_tri2x2


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


def givens_apply_left(k, q, G, A):
    """Apply givens rotation `G` to a matrix `A` from the left: `G @ A`"""
    part = A[[k,q],:].copy()
    A[[k,q],:] = G @ part


def givens_apply_right(k, q, G, A):
    """Apply givens rotation `G` to a matrix `A` from the left: `A @ G`"""
    part = A[:,[k,q]].copy()
    A[:,[k,q]] = part @ G


def golub_kahan_svd_step(d, f, U, VH, shift):
    # Alg 8.6.1
    n = d.size

    # First step: generate "unwanted element"
    y = d[0] - np.square(shift) / d[0]
    z = f[0]
    _, G = givens(array.ddarray([y, z]))
    givens_apply_left(0, 1, G, VH)
    c, s = G[0]

    ditmp = d[0]
    fi1 = f[0]
    di = ditmp*c + fi1*s
    fi1 = -ditmp*s + fi1*c
    di1 = d[1]
    bulge = di1*s
    di1 = di1 * c

    for i in range(n, n-2):
        _, G = givens(array.ddarray([di, bulge]))
        givens_apply_right(i, i+1, G.T, U)
        c, s = G[0]
        d[i] = c*di + s*bulge
        fi = c*fi1 + s*di1
        di1 = -s*fi1 + c*di1
        fi1 = f[i+1]
        bulge = s*fi1
        fi1 = fi1 * c

        _, G = givens(array.ddarray([fi, bulge]))
        givens_apply_left(i+1, i+2, G, VH)
        c, s = G[0]
        f[i]  = fi*c + bulge*s
        di = di1*c + fi1*s
        fi1 = -di1*s + fi1*c
        di2 = d[i+2]
        bulge = di2*s
        di1 = di2*c

    _, G = givens(array.ddarray([di, bulge]))
    givens_apply_right(n-2, n-1, G.T, U)
    c, s = G[0]
    d[n-2] = c*di + s*bulge
    f[n-2] = c*fi1 + s*di1
    d[n-1] = -s*fi1 + c*di1


def estimate_sbounds(d, f):
    abs_d = np.abs(d)
    abs_f = np.abs(f)
    n = abs_d.size

    def iter_backward():
        lambda_ = abs_d[n-1]
        yield lambda_
        for j in range(n-2, -1, -1):
            lambda_ = abs_d[j] * (lambda_ / (lambda_ + abs_f[j]))
            yield lambda_

    def iter_forward():
        mu = abs_d[0]
        yield mu
        for j in range(n-1):
            mu = abs_d[j+1] * (mu * (mu + abs_f[j]))
            yield mu

    smin = min(min(iter_backward()), min(iter_forward()))
    smax = max(max(abs_d), max(abs_f))
    return smax, smin


def golub_kahan_svd(d, f, U, VH, max_iter=30):
    n = d.size
    n1 = 0
    n2 = n-1
    count = 0

    # See LAWN3 page 6 and 22
    _, sigma_minus = estimate_sbounds(d, f)
    tol = 100 * 2e-16
    thresh = tol * sigma_minus

    for i_iter in range(max_iter):
        # Search for biggest index for non-zero off diagonal value in e
        for n2i in range(n2, 0, -1):
            if abs(f[n2i-1]) > thresh:
                n2 = n2i
                break
        else:
            return

        # Search for largest sub-bidiagonal matrix ending at n2
        for _n1 in range(n2 - 1, -1, -1):
            if abs(f[_n1]) < thresh:
                n1 = _n1
                break
        else:
            n1 = 0

        print("iter={}, range={}:{}".format(i_iter, n1, n2+1))

        # TODO CHECK THIS!
        if n1 == n2:
            return

        tail = array.ddarray([d[n2-1],   f[n2-1],
                              0 * d[n2], d[n2]]).reshape(2, 2)
        shift = svvals_tri2x2(tail)[1]
        golub_kahan_svd_step(d[n1:n2+1], f[n1:n2],
                             U[:, n1:n2+1], VH[n1:n2+1, :], shift)
