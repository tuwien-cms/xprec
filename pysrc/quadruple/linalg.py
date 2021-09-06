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
    """Compute the Givens rotation.

    For a vector `[f, g]`, determine parameters `c, s, r` such that:

                [  c,  s ]  @  [ f ]  =  [ r ]
                [ -s,  c ]     [ g ]     [ 0 ]
    """
    # ACM Trans. Math. Softw. 28(2), 206, Alg 1.
    if np.equal(g, 0):
        return np.ones_like(f), np.zeros_like(g), f
    if np.equal(f, 0):
        return np.zeros_like(f), np.sign(g), np.abs(g)

    r = np.copysign(np.hypot(f, g), f)
    inv_r = np.reciprocal(r)
    return f * inv_r, g * inv_r, r


def jacobi_symm2x2(a_pp, a_pq, a_qq):
    # 8.4.1
    if np.equal(a_pq, 0):  # already diagonal
        return np.ones_like(a_pp), np.zeros_like(a_pp)

    tau = (a_qq - a_pp) / (2 * a_pq)
    if np.greater_equal(tau, 0):
        t = np.reciprocal(tau + np.hypot(1, tau))
    else:
        t = -np.reciprocal(-tau + np.hypot(1, tau))

    c = np.reciprocal(np.hypot(1, t))
    return c, t * c


def eigval_symm2x2_closeqq(a_pp, a_pq, a_qq):
    # Alg. 8.3.2
    d = .5 * (a_pp - a_qq)
    t = d + np.copysign(np.hypot(d, a_pq), d)
    return a_qq - np.square(a_pq) / t


def square_bidiag(d, f):
    # Suppose we have a bidiagonal matrix with
    #   B[i,i] == d[i],  B[i,i+1] == f[i]
    # Now, note that for T = B^T B, we have:
    #   T[i,i]   == d[i]**2 + f[i-1]**2
    #   T[i,i+1] == d[i] * f[i]
    a = np.square(d)
    a[1:] += np.square(f)
    b = d[:-1] * f
    return a, b


# def bidiag_rotate_row(f, d, k, c, s):
#     fk = f[k]
#     dk = d[k]
#     d[k] = dk * c - fk * s
#     f[k] = dk * s + fk * c


# def golub_kahan_svd_step(d, f):
#     # Alg 8.6.1
#     n = d.size
#     t_d, t_f = square_bidiag(d, f)
#     mu = eigval_symm2x2_closeqq(t_d[-2], t_f[-1], t_d[-1])

#     # First step: generate "unwanted element"
#     y = t_d[0] - mu
#     z = t_f[0]
#     c, s, _ = givens_rotation(y, z)
#     bidiag_rotate_row(f, d, 0, c, s)
#     y = d[0]
#     z = -d[1] * s
#     d[1] = d[1] * c
#     for k in range(n - 1):
#     T_11 = np.square(d[0]) + np.square(f[0])
#     T_12 = d[0] * f[0]
