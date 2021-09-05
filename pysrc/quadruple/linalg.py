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


def householder_update(A):
    beta, v = householder_vector(A[:,0])
    w = beta * (A.T @ v)
    A -= v[:,None] * w[None,:]
    return v


def householder_bidiag(A):
    A = np.array(A, copy=True, subok=True)
    m, n = A.shape
    vs = []
    ws = []
    if m < n:
        raise NotImplementedError("must be tall matrix")
    for j in range(n-1):
        vs.append(householder_update(A[j:,j:]))
        ws.append(householder_update(A[j:,j+1:].T))
    if m > n:
        vs.append(householder_update(A[n-1:,n-1:]))
    return A, vs, ws
