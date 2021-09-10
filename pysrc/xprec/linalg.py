import numpy as np
from warnings import warn

from . import array
from . import _dd_linalg
import sys

norm = _dd_linalg.norm
givens = _dd_linalg.givens
givens_seq = _dd_linalg.givens_seq
svd_tri2x2 = _dd_linalg.svd_tri2x2
svvals_tri2x2 = _dd_linalg.svvals_tri2x2
householder = _dd_linalg.householder
golub_kahan_chase_ufunc = _dd_linalg.golub_kahan_chase


def householder_vector(x):
    return householder(array.asddarray(x))


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


def qr(A, reflectors=False):  # xGEQR2
    R = array.ddarray(A)
    m, n = R.shape
    k = min(m, n)

    Q = array.ddzeros((k, m))
    for i in range(k):
        householder_update(R[i:,i:], Q[i:,i:])
    if not reflectors:
        I = array.ddeye(m)
        Q = householder_apply(Q, I)
    return Q, R


def qr_pivot(A, reflectors=False):   # xGEQPF
    R = array.ddarray(A)
    m, n = R.shape
    k = min(m, n)

    Q = array.ddzeros((k, m))
    jpvt = np.arange(n)
    norms = norm(A.T)
    xnorms = norms.copy()
    TOL3Z = np.finfo(float).eps
    for i in range(k):
        pvt = i + np.argmax(norms[i:])
        if i != pvt:
            R[:,[i, pvt]] = R[:,[pvt, i]]
            jpvt[[i, pvt]] = jpvt[[pvt, i]]
            norms[pvt] = norms[i]
            xnorms[pvt] = xnorms[i]

        householder_update(R[i:,i:], Q[i:,i:])

        j = norms[i+1:].nonzero()[0] + (i+1)
        temp = np.abs(R[i, j]) / norms[j]
        temp = np.maximum(0, (1 + temp)/(1 - temp))
        temp2 = temp * np.square(norms[j] / xnorms[j])

        wheresmall = temp2 < TOL3Z
        jsmall = j[wheresmall]
        if jsmall.size:
            norms[jsmall] = norm(R[i+1:,jsmall].T)
            xnorms[jsmall] = norms[jsmall]

        jbig = j[~wheresmall]
        if jbig.size:
            x = np.multiply(norms[jbig], np.sqrt(temp[~wheresmall]))
            norms[jbig] = x.view(np.complex128, np.ndarray)

    if not reflectors:
        I = array.ddeye(m)
        Q = householder_apply(Q, I)
    return Q, R, jpvt


def golub_kahan_chase(d, e, shift):
    n = d.size
    ex = array.ddempty(d.shape)
    ex[:-1] = e
    Gs = array.ddempty((n, 4))
    golub_kahan_chase_ufunc(d, ex, shift, out=(d, ex, Gs))
    e[:] = ex[:-1]
    Gs[-1] = 0
    return Gs[:,:2], Gs[:,2:]


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
            mu = abs_d[j+1] * (mu / (mu + abs_f[j]))
            yield mu

    smin = min(min(iter_backward()), min(iter_forward()))
    smax = max(max(abs_d), max(abs_f))
    return smax, smin


def golub_kahan_svd(d, f, U, VH, max_iter=30, step=golub_kahan_chase):
    n = d.size
    n1 = 0
    n2 = n-1
    count = 0

    # See LAWN3 page 6 and 22
    _, sigma_minus = estimate_sbounds(d, f)
    tol = 100 * 5e-32
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

        #print("iter={}, range={}:{}".format(i_iter, n1, n2+1))

        # TODO CHECK THIS!
        if n1 == n2:
            return

        tail = array.ddarray([d[n2-1],   f[n2-1],
                              0 * d[n2], d[n2]]).reshape(2, 2)
        shift = svvals_tri2x2(tail)[1]
        G_V, G_U = step(d[n1:n2+1], f[n1:n2], shift)

        VHpart = VH[n1:n2+1, :]
        UHpart = U[:, n1:n2+1].T
        givens_seq(G_V, VHpart, out=VHpart)
        givens_seq(G_U, UHpart, out=UHpart)
    else:
        warn("Did not converge!")


def svd(A):
    m, n = A.shape
    U, B, V = householder_bidiag(A)
    U = householder_apply(U, array.ddarray(np.eye(m)))
    V = householder_apply(V, array.ddarray(np.eye(n)))
    VT = V.T

    d = B.diagonal().copy()
    f = B.diagonal(1).copy()
    golub_kahan_svd(d, f, U, VT, 1000)
    return U, d, VT
