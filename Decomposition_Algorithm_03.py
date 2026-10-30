import numpy as np
from pygments.styles import bw
from pysdkit import MVMD

# region I. Periodogram
def int_period_fun(Y):
    TT = Y.shape[0]
    Y = np.sort(Y, axis=1)
    h_series = np.arange(-(TT-1),TT)
    cov_series = np.zeros(len(h_series))
    for i in range(len(cov_series)):
        h = h_series[i]
        if h==0:
            X_temp = Y
            Y_temp = Y
        else:
            X_temp = Y[abs(h):]
            Y_temp = Y[:-abs(h)]
        cov_series[i] = int_cov_fun(X_temp, Y_temp,5,1,1,1)
    def fft_fun(jj):
        omega_j = jj * 2 * np.pi / TT
        return(1 / (2 * np.pi) * np.sum(cov_series * np.exp(-1j * h_series * omega_j)))
    jj_series = np.arange(1,np.floor((TT-1)/2) +1)
    f_omega = np.zeros(len(jj_series))
    for i in range(len(jj_series)):
        jj = jj_series[i]
        f_omega[i] = fft_fun(jj)
    return np.abs(f_omega)

# endregion

# region II. BEMD
def _find_local_extrema_1d(x):

    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    dx = np.diff(x)

    sign = np.sign(dx)
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    for i in range(len(sign) - 2, -1, -1):
        if sign[i] == 0:
            sign[i] = sign[i + 1]

    max_idx = []
    min_idx = []
    for i in range(1, len(sign)):
        if sign[i - 1] > 0 and sign[i] < 0:
            max_idx.append(i)
        elif sign[i - 1] < 0 and sign[i] > 0:
            min_idx.append(i)

    return np.array(max_idx, dtype=int), np.array(min_idx, dtype=int)

def _augment_boundary(idx, values, T):

    idx = np.asarray(idx, dtype=int)
    values = np.asarray(values, dtype=float)

    if len(idx) == 0:

        idx_new = np.array([0, T - 1], dtype=int)
        val0 = values[0] if values.ndim == 2 and len(values) > 0 else np.zeros(2)
        values_new = np.vstack([val0, val0])
        return idx_new, values_new

    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
        values = np.vstack([values[0], values])

    if idx[-1] != T - 1:
        idx = np.append(idx, T - 1)
        values = np.vstack([values, values[-1]])

    return idx, values

def _interp_envelope(indices, vec_values, T):

    t = np.arange(T)
    indices = np.asarray(indices, dtype=int)
    vec_values = np.asarray(vec_values, dtype=float)

    indices, vec_values = _augment_boundary(indices, vec_values, T)

    env = np.zeros((T, 2), dtype=float)
    for j in range(2):
        env[:, j] = np.interp(t, indices, vec_values[:, j])
    return env

def _mean_envelope_bivariate(h, ndir=3):

    h = np.asarray(h, dtype=float)
    T = h.shape[0]
    thetas = np.linspace(0, np.pi, ndir, endpoint=False)

    mean_env = np.zeros((T, 2), dtype=float)
    used = 0

    for theta in thetas:
        direction = np.array([np.cos(theta), np.sin(theta)])
        proj = h @ direction  # shape (T,)

        max_idx, min_idx = _find_local_extrema_1d(proj)

        if len(max_idx) < 2 or len(min_idx) < 2:
            continue

        upper = _interp_envelope(max_idx, h[max_idx], T)
        lower = _interp_envelope(min_idx, h[min_idx], T)
        mean_env += 0.5 * (upper + lower)
        used += 1

    if used == 0:
        return np.zeros((T, 2), dtype=float)

    return mean_env / used

def _imf_stop_criterion(h, m, tol=1):

    num = np.sum((m ** 2), axis=1)
    den = np.sum((h ** 2), axis=1) + 1e-12
    ratio = np.mean(num / den)
    return ratio < tol

def _has_enough_extrema_bivariate(r, ndir=3):

    r = np.asarray(r, dtype=float)
    thetas = np.linspace(0, np.pi, ndir, endpoint=False)

    valid_dirs = 0
    for theta in thetas:
        direction = np.array([np.cos(theta), np.sin(theta)])
        proj = r @ direction
        max_idx, min_idx = _find_local_extrema_1d(proj)
        if len(max_idx) + len(min_idx) >= 4:
            valid_dirs += 1

    return valid_dirs >= max(2, ndir // 8)

def extract_one_bivariate_imf(r,ndir=3, max_siftings=3,sift_tol=1):

    h = r.copy()

    for _ in range(max_siftings):
        m = _mean_envelope_bivariate(h, ndir=ndir)

        if np.allclose(m, 0.0, atol=1e-12):
            break

        h_new = h - m

        if _imf_stop_criterion(h_new, m, tol=sift_tol):
            h = h_new
            break

        h = h_new

    return h

def BEMD_fun(X, K=3, ndir=3, max_siftings=3, sift_tol=1):

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (T, 2)")

    T = X.shape[0]
    if K < 1:
        raise ValueError("K must >= 1")

    out = np.zeros((2, T, K), dtype=float)

    residual = X.copy()

    for k in range(K - 1):
        if not _has_enough_extrema_bivariate(residual, ndir=ndir):
            break

        imf = extract_one_bivariate_imf(
            residual,
            ndir=ndir,
            max_siftings=max_siftings,
            sift_tol=sift_tol
        )

        out[:, :, k] = imf.T
        residual = residual - imf

    out[:, :, K - 1] = residual.T
    return out

# endregion

# region III. BVMD
def _mirror_extend(X):

    X = np.asarray(X, dtype=float)
    T = X.shape[0]
    left = X[:T][::-1]
    right = X[:T][::-1]
    X_ext = np.vstack([left, right])
    return X_ext

def _analytic_signal_fft(x):
 
    N = len(x)
    x_hat = np.fft.fft(x)
    h = np.zeros(N)

    if N % 2 == 0:
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0

    return x_hat * h

def _init_omegas(K, init_mode, freqs):

    pos_freqs = freqs[freqs >= 0]
    if len(pos_freqs) == 0:
        pos_freqs = np.linspace(0, 0.5, K + 2)[1:-1]

    if init_mode == "uniform":
        omegas = np.linspace(0, 0.5, K + 2)[1:-1]
    elif init_mode == "random":
        rng = np.random.default_rng(1234)
        omegas = np.sort(rng.uniform(0.0, 0.5, size=K))
    elif init_mode == "zero":
        omegas = np.zeros(K)
    else:
        raise ValueError("init_mode must be 'uniform', 'random', or 'zero'")

    return omegas

def BVMD_fun(X,K=3,alpha = 1000.0,tau = 0.0,tol = 1e-7,
    max_iter = 3,init_mode = "uniform",dc = True):
    

    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (T, 2)")
    if K < 1:
        raise ValueError("K must be >= 1")

    T = X.shape[0]

    X_ext = _mirror_extend(X)   # (N,2)
    N = X_ext.shape[0]

    freqs = np.fft.fftfreq(N, d=1.0)

    f_hat_plus = np.zeros((2, N), dtype=complex)
    for c in range(2):
        f_hat_plus[c] = _analytic_signal_fft(X_ext[:, c])

    u_hat = np.zeros((2, K, N), dtype=complex)
    lambda_hat = np.zeros((2, N), dtype=complex)
    omega = _init_omegas(K, init_mode, freqs)

    if dc:
        omega[0] = 0.0

    pos_mask = freqs >= 0

    u_hat_prev = np.copy(u_hat)

    for n in range(max_iter):
        sum_u_hat = np.sum(u_hat, axis=1)  # (2,N)

        for k in range(K):
            sum_others = sum_u_hat - u_hat[:, k, :]  # (2,N)

            denom = 1.0 + 2.0 * alpha * (freqs - omega[k])**2  # (N,)
            denom = denom.astype(float)

            for c in range(2):
                numer = f_hat_plus[c] - sum_others[c] + 0.5 * lambda_hat[c]
                u_hat[c, k, :] = numer / denom
                u_hat[c, k, ~pos_mask] = 0.0
            sum_u_hat = sum_others + u_hat[:, k, :]


            if not (dc and k == 0):
                num = 0.0
                den = 0.0
                for c in range(2):
                    spec_power = np.abs(u_hat[c, k, :])**2
                    num += np.sum(freqs[pos_mask] * spec_power[pos_mask])
                    den += np.sum(spec_power[pos_mask])
                if den > 1e-18:
                    omega[k] = num / den
                else:
                    omega[k] = 0.0

        recon = np.sum(u_hat, axis=1)  # (2,N)
        lambda_hat = lambda_hat + tau * (f_hat_plus - recon)

        diff = np.linalg.norm(u_hat - u_hat_prev) / (np.linalg.norm(u_hat_prev) + 1e-12)
        if diff < tol:
            break

        u_hat_prev = np.copy(u_hat)

    u_time_ext = np.zeros((2, K, N), dtype=float)
    for c in range(2):
        for k in range(K):
            u_time_ext[c, k, :] = np.real(np.fft.ifft(u_hat[c, k, :]))

    start = N // 4
    end = start + T
    U = u_time_ext[:, :, start:end]   # (2,K,T)
    U = np.transpose(U, (0, 2, 1))    # (2,T,K)

    recon = np.sum(U, axis=2)         # (2,T)
    resid = X.T - recon               # (2,T)
    U[:, :, -1] += resid              

    return U

# endregion

# region IV. SVD
def Hankel_mean_fun(X):

    # A 1D array consisting of the averages of the elements on each subdiagonal of the Hankel matrix.

    m, n = X.shape
    i,j = np.indices((m,n))
    k = i + j
    result = np.array([X[k == s].mean() for s in range(m + n - 1)])
    return result

def SVD_single_fun(Y, layer=3):
    info_ratio = 1
    TT = len(Y)
    bw = 15
    col_len = TT - bw + 1
    traj_mat = np.zeros((bw, col_len))
    for i in range(bw):
        traj_mat[i, :] = Y[i:(i + col_len)]
    U, s, Vt = np.linalg.svd(traj_mat)

    Y_layer = np.zeros((TT, layer))
    for i in range(layer):
        match i:
            case 0:
                j_ = [0,1,2,3,4]
            case 1:
                j_ = [5,6,7,8,9]
            case 2:
                j_ = [10,11,12,13,14]
        traj_mat_temp = np.zeros_like(traj_mat)
        for j in j_:
            traj_mat_temp += np.outer((U[:, j] * s[j]), Vt[j, :])
        Y_layer[:, i] = Hankel_mean_fun(traj_mat_temp)

    return Y_layer  # (TT,K)

def SVD_fun(Y,layer=3):

    result = np.zeros((2, Y.shape[0], layer))
    result[0, :, :] = SVD_single_fun(Y[:, 0], layer=3)
    result[1, :, :] = SVD_single_fun(Y[:, 1], layer=3)
    return result# (N,T,K)

# endregion

# region V. ISVD
def int_inprod_fun(X,Y,K11=1,K12=0,K21=0,K22=1):

    K = np.array([K22, -K12, -K21, K11]).reshape(2, 2)
    if X.ndim==1:
        return X @ K @ Y
    else:
        return np.mean(np.sum((X @ K) * Y, axis=1))

def int_cov_fun(X,Y,K11=1,K12=0,K21=0,K22=1):
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    result = (int_inprod_fun(X,Y,K11,K12,K21,K22) -
              int_inprod_fun(X_mean,Y_mean,K11,K12,K21,K22))
    return result

def ISVD_fun(Y,layer = 3,K=np.array([[1,0],[0,1]])):

    TT = Y.shape[0]
    #bw = 12
    bw = 15
    col_len = TT - bw + 1

    traj_l_mat = np.zeros((bw, col_len))
    for i in range(bw):
        traj_l_mat[i, :] = Y[i:(i + col_len), 0]

    traj_r_mat = np.zeros((bw, col_len))
    for i in range(bw):
        traj_r_mat[i, :] = Y[i:(i + col_len), 1]

    int_cov_mat = np.zeros((bw, bw))
    for i in range(bw):
        for j in range(bw):
            int_cov_mat[i, j] = int_cov_fun(Y[i:(i + col_len), :],
                                            Y[j:(j + col_len), :],
                                            K[0, 0], K[0, 1], K[1, 0], K[1, 1])
    U, s, Vt = np.linalg.svd(int_cov_mat)

    Y_layer = np.zeros((2, TT, layer))
    traj_mat_l_temp = np.zeros_like(traj_l_mat)
    traj_mat_r_temp = np.zeros_like(traj_r_mat)
    for j in range(layer):
        match j:
            case 0:
                j_ = [0,1,2,3,4]

            case 1:
                j_ = [5,6,7,8,9]

            case 2:
                j_ = [10,11,12,13,14]

        for i in j_:
            traj_mat_l_temp += np.outer(U[:, i], (U[:, i].T @ traj_l_mat))
            traj_mat_r_temp += np.outer(U[:, i], (U[:, i].T @ traj_r_mat))
        Y_layer[0, :, j] = Hankel_mean_fun(traj_mat_l_temp)
        Y_layer[1, :, j] = Hankel_mean_fun(traj_mat_r_temp)

    return Y_layer# (N,T,K)

# endregion

