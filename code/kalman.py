# Kalman functions

import numpy as np
from numpy.polynomial.polynomial import polyval
import scipy.linalg
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter as gf
from scipy.sparse.linalg import LinearOperator


# sparse Li = Q^(-1/2), obs error as ystd
def etkf_step(xens, M, y, H, ystd, L=None, inflation=1.0, rotate=False, rho=1, warp=None, return_prior=False):
    """ETKF step."""
    nstate, nens = xens.shape
    if M is None:
        xnew = xens.copy()
    else:
        xnew = M @ xens
    if L is not None:
        rx = sp.linalg.spsolve(L['Li'], np.random.randn(nstate, nens))  # add Gaussian noise
        if L['force']:
            rx = rx - rx.mean(axis=1, keepdims=True)  # force zero mean
        if L['spatial']:
            rx = np.maximum(0.0, xnew) * rx # inflate noise only where xnew has values
        xnew = xnew + rx
    if warp is not None:
        for i in range(nens):
            xnew[:, i] = random_warp(xnew[:, i].reshape(warp['ny'], warp['nx']),
                                     sigu=warp['sigu'],
                                     sigv=warp['sigv']).reshape(-1)
    if rho != 1:
        xnew = rho * xnew + (1-rho) * xnew.mean(axis=1, keepdims=True)
    
    if y.size == 0:
        if return_prior:
            return xnew, xnew
        return xnew

    xmean = xnew.mean(axis=1, keepdims=True)

    X = (xnew - xmean) / np.sqrt(nens-1)
    HX = H @ xnew
    HXmean = HX.mean(axis=1)
    C = (HX - HXmean[:, np.newaxis]) / ystd[:, np.newaxis] / np.sqrt(nens-1) # nobs * nens
    if C.shape[0] < C.shape[1]: 
        _, s, v = np.linalg.svd(C, True)  # v is nens x nens
        s = np.r_[s, np.zeros(C.shape[1] - C.shape[0])]
    else:
        _, s, v = np.linalg.svd(C, False)  # v is nens x nens
    r = (y - HXmean) / ystd  # residuals, nobs * nens

    T = v.T * (1 / (1+s**2) / inflation**2) @ v  #  inv(I + C.T @ C), nens * nens
    T12 = v.T * (1 / np.sqrt(1+s**2) / inflation**2) @ v
    
    w = T @ C.T @ r # nens * 1
    # optional random rotation
    if rotate:
        O, rr = np.linalg.qr(np.random.randn(nens, nens))
        O = O @ np.diag(np.sign(np.diag(rr)))
        W = w[:, np.newaxis] + np.sqrt(nens-1) * T12 @ O  # nens * nens
    else:
        W = w[:, np.newaxis] + np.sqrt(nens-1) * T12  # nens * nens
    xnew_update = xmean + X @ W  # nstate * nens
    
    if return_prior:
        return xnew_update, xnew
    return xnew_update


def qprod(X, C):
    """X*C*X.T for symmetric C."""
    return X.dot(X.dot(C).T)


def solve(A, b, assume_a='gen'):
    """Solve Ax = b."""
    m, n = A.shape
    if m > n:
        return scipy.linalg.lstsq(A, b)[0]
    else:
        return scipy.linalg.solve(A, b, assume_a=assume_a)


def addd(a, d):
    """Add D to the diagonal of a matrix, return A + diag(D)."""    
    np.fill_diagonal(a, a.diagonal() + d)
    return a


def kf_step(x, C, M, Q, y, H, ystd):
    """Linear Kalman filter step.
    ystd is vector of stds
    """
    #x = x.reshape(-1, 1)
    #y = y.reshape(-1, 1)
    #ystd = ystd.reshape(-1, 1)
    #obsind = np.isfinite(y.ravel())
    #if not obsind.all():
    #    y = y[obsind]
    #    H = H[obsind]
    #    ystd = ystd[obsind]
    xp = M.dot(x)
    Cp = qprod(M, C) + Q
    K = solve(addd(qprod(H, Cp), ystd**2), H.dot(Cp), assume_a='sym')
    x = xp + K.T.dot(y - H.dot(xp))
    C = Cp - H.dot(Cp).T.dot(K)
    return x, C


def enkf_step(xens, M, Q, y, H, R):
    """Ensemble Kalman filter step."""
    nstate, nens = xens.shape
    nobs = y.size
    xnew = M @ xens
    # randomization
    xnew = xnew + np.random.multivariate_normal(mean=np.zeros(nstate), cov=Q, size=nens).T
    ry = np.random.multivariate_normal(mean=np.zeros(nobs), cov=R, size=nens).T
    X = (xnew - xnew.mean(axis=1, keepdims=True)) / np.sqrt(nens - 1)
    HX = H.dot(X)
    Cy = HX.dot(HX.T) + R
    yp = H.dot(xnew)
    K = np.linalg.solve(Cy, HX.dot(X.T)).T
    xnew = xnew + K.dot(ry - (yp - y))
    return xnew


def gasparicohn(x, c=1):
    x = np.abs(np.asarray(x) / c)
    y = np.piecewise(x,
                     [(x >= 0) & (x < 1), (x >= 1) & (x < 2)],
                     [lambda x: polyval(x, [1, 0, -5/3, 5/8, 1/2, -1/4]),
                      lambda x: polyval(x, [4, -5, 5/3, 5/8, -1/2, 1/12]) - 2/3/x,
                      0.0])
    return y


def warp_simple(X, u, v, dx=1.0, dy=1.0, dt=1.0, D=0.0):
    ny, nx = X.shape
    xi, yi = np.meshgrid(np.arange(nx), np.arange(ny))
    # copy border
    i = (xi - u*dt/dx).astype(int)
    j = (yi - v*dt/dy).astype(int)
    i = np.maximum(np.minimum(i, nx - 1), 0)
    j = np.maximum(np.minimum(j, ny - 1), 0)
    # diffusion
    if D > 0.0:
        y = gf(X.reshape(ny, nx), sigma=D)[j, i]
    else:
        y = X.reshape(ny, nx)[j, i] #.reshape(X.shape)
    return y


# linear operator from the warp
def genM_warp(u, v, dx=1, dy=1, dt=1, D=0):
    ny, nx = u.shape
    def M(x, u=u, v=v, dx=dx, dy=dy, dt=dt, D=D):
        return warp_simple(x.reshape(ny, nx), u, v, dx=dx, dy=dy, dt=dt, D=D).ravel()
    return LinearOperator(matvec=M, shape=(ny*nx, ny*nx))


def random_warp(x, sigu=1.0, sigv=None):
    sigv = sigu if sigv is None else sigv
    sigu = np.random.rand(1) * sigu
    sigv = np.random.rand(1) * sigv
    ny, nx = x.shape[0:2]
    y = warp_simple(x, np.ones((ny, nx))*sigu, np.ones((ny, nx))*sigv)
    return y

def matern32L(nx, ny, l=1.0, h=1.0, sig=1.0, format='dia', indexing='xy'):
    """
    Matern nu = 3/2 inverse covariance factor L, such that `Q=inv(L*L')`.
    """
    if indexing == 'ij':
        nx, ny = ny, nx
    n = nx * ny
    e = np.ones(n)
    e2 = e.copy()
    e3 = e.copy()
    e2[nx - 1::nx] = 0
    e3[::nx] = 0
    c1 = (l / h) / sig / 2.0 / np.sqrt(np.pi)
    c2 = (h / l) / sig / 2.0 / np.sqrt(np.pi)
    L = sp.spdiags([-e * c1, -e2 * c1, (4.0 * c1 + c2) * e,
                    -e3 * c1, -e * c1], [-nx, -1, 0, 1, nx], n, n,
                   format=format)
    return L

def _get2(x):
    """Get two values from x."""
    x = np.atleast_1d(x)
    return (x.item(0), x.item(1)) if x.size > 1 else (x.item(0), x.item(0))

def _comb2(x):
    """Combine 2 first dimensions."""
    return x.reshape((np.prod(x.shape[0:2]),) + x.shape[2:])

def view_and_ind3(x, nx2, ny2, overlap=[0, 0], combine=True, flatten=False):
    """Sliding views and indeces to a 3d array.

    Given 3d array in x of dimension, (ntime, ny, nx), this function
    returns (n, ntime, ny2, nx2) array of views to sub arrays of the
    original array. The second output is an array of indeces, see Notes.

    Parameters
    ----------
    x : 3d numpy array
        3d array of dimension (ntime, ny, nx).
    nx2 : int
        Output view 'x' dimension.
    ny2 : int
        Output view 'y' dimension.
    overlap : [int, int], optional
        How many pixels to overlap in the views. The default is [0, 0].
    combine : Bool, optional
        Just return one index over the views over 2d array.
        If False, returns selarate row and column ideces.
        The default is True.
    flatten : Bool, optional
        Is True, flatten the output index so that it refers to the flattened
        2d versions. The default is False, which means that the function
        return 2d indeces.

    Returns
    -------
    xb : numpy array.
        Views to the original array. The dimension is (nviews, nt, ny2, nx2).
    xi : numpy array of ints.
        Index to views. Dimesion is (nviews, ny2, nx2) or (nviews, ny2 * nx2).

    Notes
    -----
    `overlap` is the overlap of the sub blocks. Has 1 of 2 values.
    Must be between 0 and nx2 - 1 (ny2 - 1 for the second value).

    In addition, function returns (n, ny2 * nx2) array of indeces of
    the sub arrays, which can be used to index the original x. For example if
    xblocks, inds = view_and_ind3(x, nx2, ny2)
    and
    r, c = np.unravel_index(inds[i], (ny, nx))
    then x[:, r, c] picks the block n:o i from x.
    Or alternatively:
    xblocks, inds = view_and_ind3(x, nx2, ny2, combine=True)
    x.reshape(nt, -1)[:, inds[i]].reshape(nt, ny2, nx2)
    should be same as
    xblocs[i]

    TODO
    ----
    Should add extra block at the and of both x and y directions if the
    given blocking does not fill the full original domain.
    """
    nt, ny, nx = x.shape
    overlap = _get2(overlap)
    sx = nx2 - overlap[0]
    sy = ny2 - overlap[1]
    inds = np.array(range(nx * ny), dtype=int).reshape((ny, nx))
    xb = (np.lib.stride_tricks.
          sliding_window_view(x, (nt, ny2, nx2))[:, ::sy, ::sx, :, :])
    xi = (np.lib.stride_tricks.
          sliding_window_view(inds, (ny2, nx2))[::sy, ::sx, :, :])
    xb = xb.squeeze(axis=0)
    if flatten:  # flatten index
        xi = xi.reshape(xi.shape[0], xi.shape[1], -1)
    if combine:
        xb = _comb2(xb)
        xi = _comb2(xi)
    return xb, xi
