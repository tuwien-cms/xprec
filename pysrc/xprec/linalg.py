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
householder = _dd_linalg.householder
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


def svd(A, full_matrices=True):
    """Truncated singular value decomposition.

    Decomposes a `(m, n)` matrix `A` into the product:

        A == U @ (s[:,None] * VT)

    where `U` is a `(m, k)` matrix with orthogonal columns, `VT` is a `(k, n)`
    matrix with orthogonal rows and `s` are the singular values, a set of `k`
    nonnegative numbers in non-ascending order and `k = min(m, n)`.
    """
    A = np.asarray(A)
    m, n = A.shape
    if m < n:
        U, s, VT = svd(A.T, full_matrices)
        return VT.T, s, U.T

    Q, B, RT = bidiag(A)
    for _ in range(20 * n):
        if svd_bidiag_step(Q, B, RT):
            break
    else:
        warn("Did not converge")

    U, s, VH = svd_normalize(Q, B.diagonal(), RT)
    if not full_matrices:
        U = U[:,:n]
    return U, s, VH


def svd_trunc(A, tol=5e-32, method='jacobi', max_iter=20):
    """Truncated singular value decomposition.

    Decomposes a `(m, n)` matrix `A` into the product:

        A == U @ (s[:,None] * VT)

    where `U` is a `(m, k)` matrix with orthogonal columns, `VT` is a `(k, n)`
    matrix with orthogonal rows and `s` are the singular values, a set of `k`
    nonnegative numbers in non-ascending order.  The SVD is truncated in the
    sense that singular values below `tol` are discarded.
    """
    # RRQR is an excellent preconditioner for Jacobi.  One should then perform
    # Jacobi on RT
    Q, R, p = rrqr(A, tol)
    if method == 'jacobi':
        U, s, VT = svd_jacobi(R.T, tol, max_iter)
    elif method == 'golub-kahan':
        U, s, VT = svd(R.T, full_matrices=False)
    else:
        raise ValueError("invalid method")

    # Reconstruct A from QRs
    U_A = Q @ VT.T
    VT_B = U.T[:, p.argsort()]
    return U_A, s, VT_B


def bidiag(A, reflectors=False, force_structure=False):
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

    if force_structure:
        d = B.diagonal().copy()
        e = B.diagonal(1).copy()
        B[...] = 0
        i = np.arange(n)
        B[i, i] = d
        B[i[:-1], i[:-1]+1] = e
    if not reflectors:
        Q = householder_apply(Q, np.eye(m, dtype=B.dtype))
        R = householder_apply(R, np.eye(n, dtype=B.dtype))
    return Q, B, R.T


def svd_jacobi(A, tol=5e-32, max_iter=20):
    """Singular value decomposition using Jacobi rotations."""
    U = A.copy()
    m, n = U.shape
    if m < n:
        raise RuntimeError("expecting tall matrix")

    VT = np.eye(n, dtype=U.dtype)
    offd = np.empty((), ddouble)

    limit = tol * np.linalg.norm(U[:n,:n], 'fro')
    for _ in range(max_iter):
        _dd_linalg.jacobi_sweep(U, VT, out=(U, VT, offd))
        if offd <= limit:
            break
    else:
        warn("Did not converge")

    s = norm(U.T)
    U = U / s
    return U, s, VT


def householder_update(A, Q):
    """Reflects the zeroth column onto a multiple of the unit vector"""
    beta, v = householder(A[:,0])
    w = -beta * (A.T @ v)
    rank1update(A, v, w, out=A)
    Q[0,0] = beta
    Q[1:,0] = v[1:]


def householder_apply(H, Q):
    """Applies a set of reflectors to a matrix"""
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


def svd_normalize(U, d, VH):
    """Given a SVD-like decomposition, normalize"""
    # Invert
    n = d.size
    VH[np.signbit(d)] = -VH[np.signbit(d)]
    d = np.abs(d)

    # Sort
    order = np.argsort(d)[::-1]
    d = d[order]
    VH = VH[order]
    U = U.copy()
    U[:,:n] = U[:,order]
    return U, d, VH


def svd_bidiag_step(Q, B, RT):
    """Single SVD step for a bidiagonal matrix"""
    d = B.diagonal().copy()
    e = np.hstack([B.diagonal(1), 0.0])

    p, q = bidiag_partition(d, e)
    if q <= 1:
        return True

    d_part = d[p:q]
    e_part = e[p:q]
    rot = np.empty((d_part.size, 4), d.dtype)
    _dd_linalg.golub_kahan_chase(d_part, e_part, out=(d_part, e_part, rot))

    i = np.arange(p, q)
    B[i, i] = d_part
    B[i[:-1], i[:-1]+1] = e_part[:-1]

    rot_Q = rot[:, 2:]
    rot_R = rot[:, :2]
    QT_part = Q[:, p:q].T
    RT_part = RT[p:q, :]
    _dd_linalg.givens_seq(rot_Q, QT_part, out=QT_part)
    _dd_linalg.givens_seq(rot_R, RT_part, out=RT_part)
    return False


def bidiag_partition(d, e, eps=5e-32):
    """Partition bidiagonal matrix into blocks for implicit QR.

    Return `p,q` which partions a bidiagonal `B` matrix into three blocks:

      - B[0:p, 0:p], an arbitrary bidiaonal matrix
      - B[p:q, p:q], a matrix with all off-diagonal elements nonzero
      - B[q:,  q:],  a diagonal matrix
    """
    abs_e = np.abs(e)
    abs_d = np.abs(d)
    e_zero = abs_e <= eps * (abs_d + abs_e)
    e[e_zero] = 0

    q = _find_last(~e_zero) + 1
    if q <= 0:
        return 0, 0
    p = _find_last(e_zero[:q]) + 1
    return p, q + 1


def _find_last(a, axis=-1):
    a = a.astype(bool)
    maxloc = a.shape[axis] - 1 - a[::-1].argmax(axis)
    return np.where(a[maxloc], maxloc, -1)
