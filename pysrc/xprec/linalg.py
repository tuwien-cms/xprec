# Copyright (C) 2021 Markus Wallerberger and others
# SPDX-License-Identifier: MIT
#
# Some of the code in this module is adapted from the LAPACK reference
# implementation.
import numpy as np
from warnings import warn

from . import ddouble
from . import _dd_linalg

norm = _dd_linalg.norm
givens = _dd_linalg.givens
givens_seq = _dd_linalg.givens_seq
svd_tri2x2 = _dd_linalg.svd_tri2x2
svvals_tri2x2 = _dd_linalg.svvals_tri2x2
householder = _dd_linalg.householder
golub_kahan_chase_ufunc = _dd_linalg.golub_kahan_chase
rank1update = _dd_linalg.rank1update


def qr(A, reflectors=False):
    """QR decomposition without pivoting.

    Decomposes a `(m, n)` matrix `A` into the product:

        A == Q @ R

    where `Q` is an `(m, m)` orthogonal matrix and `R` is a `(m, n)` upper
    triangular matrix.  No pivoting is used.
    """
    R = np.array(A)
    m, n = R.shape
    k = min(m, n)

    Q = np.zeros((k, m), A.dtype)
    for i in range(k):
        householder_update(R[i:,i:], Q[i:,i:])
    if not reflectors:
        I = np.eye(m, dtype=A.dtype)
        Q = householder_apply(Q, I)
    return Q, R


def rrqr(A, tol=5e-32, reflectors=False):
    """Truncated rank-revealing QR decomposition with full column pivoting.

    Decomposes a `(m, n)` matrix `A` into the product:

        A[:,piv] == Q @ R

    where `Q` is an `(m, k)` isometric matrix, `R` is a `(k, n)` upper
    triangular matrix, `piv` is a permutation vector, and `k` is chosen such
    that the relative tolerance `tol` is met in the equality above.
    """
    R = np.array(A)
    m, n = R.shape
    k = min(m, n)

    Q = np.zeros((m, k), A.dtype)
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

        js = (i + 1) + norms[i + 1:].nonzero()[0]
        temp = np.abs(R[i,js]) / norms[js]
        temp = np.fmax(0.0, (1 + temp)*(1 - temp))
        temp2 = temp * np.square(norms[js] / xnorms[js])

        wheresmall = temp2 < TOL3Z
        jsmall = js[wheresmall]
        upd_norms = norm(R[i+1:,jsmall].T)
        norms[jsmall] = upd_norms
        xnorms[jsmall] = upd_norms
        jbig = js[~wheresmall]
        norms[jbig] *= np.sqrt(temp[~wheresmall])

        if tol is not None:
            acc = np.abs(R[i,i] / R[0,0])
            if acc < tol:
                k = i + 1
                Q = Q[:,:k]
                R = R[:k,:]
                break

    if not reflectors:
        I = np.eye(m, k, dtype=A.dtype)
        Q = householder_apply(Q, I)
    return Q, R, jpvt


def bidiag(A, reflectors=False):
    """Biadiagonalizes an arbitray rectangular matrix.

    Decomposes a `(m, n)` matrix `A` into the product:

        A == Q @ B @ RT

    where `Q` is a `(m, m)` orthogonal matrix, `RT` is a `(n, n)` orthogonal
    matrix, and `B` is a bidiagonal matrix, where the upper diagonal is
    nonzero for `m >= n` and the lower diagonal is nonzero for `m < n`.
    """
    A = np.asarray(A)
    m, n = A.shape
    if m < n:
        Q, B, RT = bidiag(A.T, reflectors)
        return RT.T, B.T, Q.T

    rq = n - (m == n)
    B = A.copy()
    Q = np.zeros_like(B)
    R = np.zeros_like(B[:n,:n])

    for j in range(n-2):
        householder_update(B[j:,j:], Q[j:,j:])
        householder_update(B[j:,j+1:].T, R[j+1:,j+1:])
    for j in range(n-2, rq):
        householder_update(B[j:,j:], Q[j:,j:])

    if not reflectors:
        Q = householder_apply(Q, np.eye(m, dtype=B.dtype))
        R = householder_apply(R, np.eye(n, dtype=B.dtype))
    return Q, B, R.T


def svd(A):
    """Singular value decomposition of a rectangular matrix.

    Decomposes a `(m, n)` matrix `A` into the product:

        A == U @ (s[:,None] * VT)    # for m >= n
        A == (U * s) @ VT            # for m <= n

    where `U` is a `(m, m)` orthogonal matrix, `VT` is a `(n, n)` orthogonal
    matrix, and `s` are the singular values, a set of nonnegative numbers
    in descending order.
    """
    A = np.asarray(A)
    m, n = A.shape
    if m < n:
        U, s, VT = svd(A.T)
        return VT.T, s, U.T

    U, B, VT = bidiag(A)
    d = B.diagonal().copy()
    f = B.diagonal(1).copy()
    golub_kahan_svd(d, f, U, VT, 1000)
    return U, d, VT


def svd_trunc(A, tol=5e-32):
    """Truncated singular value decomposition"""
    Q, R, p = rrqr(A, tol)
    U, s, VT = svd(R)
    U = Q @ U[:, :s.size]
    VT = VT[:s.size, p.argsort()]
    return U, s, VT


def golub_kahan_svd(d, f, U, VH, max_iter=30, step=None):
    if step is None:
        step = golub_kahan_chase
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
            break # from iter loop

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
            break # from iter loop

        tail = np.array([d[n2-1],   f[n2-1],
                         0 * d[n2], d[n2]]).reshape(2, 2)
        shift = svvals_tri2x2(tail)[1]
        G_V, G_U = step(d[n1:n2+1], f[n1:n2], shift)

        VHpart = VH[n1:n2+1, :]
        UHpart = U[:, n1:n2+1].T
        givens_seq(G_V, VHpart, out=VHpart)
        givens_seq(G_U, UHpart, out=UHpart)
    else:
        warn("Did not converge!")

    # Invert
    VH[np.signbit(d)] = -VH[np.signbit(d)]
    d[:] = np.abs(d)

    # Sort
    order = np.argsort(d)[::-1]
    d[:] = d[order]
    VH[:] = VH[order]
    U[:,:n] = U[:,order]


def golub_kahan_chase(d, e, shift):
    n = d.size
    ex = np.empty(d.shape, d.dtype)
    ex[:-1] = e
    Gs = np.empty((n, 4), d.dtype)
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


def householder_update(A, Q):
    beta, v = householder(A[:,0])
    w = -beta * (A.T @ v)
    rank1update(A, v, w, out=A)
    Q[0,0] = beta
    Q[1:,0] = v[1:]


def householder_apply(H, Q):
    H = np.asarray(H)
    Q = Q.copy()
    m, r = H.shape
    if Q.shape[0] != m:
        raise ValueError("invalid shape")
    if Q.shape[1] < r:
        raise ValueError("invalid shape")
    for j in range(r-1, -1, -1):
        beta = H[j,j]
        if np.equal(beta, 0):
            continue
        v = np.empty_like(H[j:,0])
        v[0] = 1
        v[1:] = H[j+1:,j]
        Qpart = Q[j:,j:]
        w = -beta * (Qpart.T @ v)
        rank1update(Qpart, v, w, out=Qpart)
    return Q