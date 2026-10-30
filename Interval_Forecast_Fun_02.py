import numpy as np
from scipy.stats import norm, t
from math import gamma, pi
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import portion as P
from arch.bootstrap import MCS


# region I. Forecast functions
def Order_fun(X):
    return np.array([min(X),max(X)])

def int_dist_fun(X,Y,K11=1,K12=0,K21=0,K22=1):

    X = np.sort(X,axis=1)
    Y = np.sort(Y,axis=1)
    K = np.array([K11, K12, K21, K22]).reshape(2, 2)
    if X.ndim==1:
        return np.array([X[1]-Y[1],-(X[0]-Y[0])]) @ K @ np.array([X[1]-Y[1],-(X[0]-Y[0])])
    else:
        Delta_ = np.zeros_like(X)
        Delta_[:,0] = X[:,1]-Y[:,1]
        Delta_[:, 1] = -(X[:, 0] - Y[:, 0])
        return np.mean(np.sum((Delta_ @ K) * Delta_, axis=1))

def int_distde_fun(X,Y,K11=1,K12=0,K21=0,K22=1):

    X = np.sort(X,axis=1)
    Y = np.sort(Y,axis=1)
    K = np.array([K11, K12, K21, K22]).reshape(2, 2)
    if X.ndim==1:
        return np.sqrt(np.array([X[1]-Y[1],-(X[0]-Y[0])]) @ K @ np.array([X[1]-Y[1],-(X[0]-Y[0])]))
    else:
        Delta_ = np.zeros_like(X)
        Delta_[:,0] = X[:,1]-Y[:,1]
        Delta_[:, 1] = -(X[:, 0] - Y[:, 0])
        return np.sqrt(np.sum((Delta_ @ K) * Delta_, axis=1))

def CRSETAR_fun(Y,p_lag,h_ahead):

    Y_mid = np.mean(Y, axis=1)
    Y_ran = Y[:,1] - Y_mid
    gamma_mid = np.percentile(Y_mid, 50)
    gamma_ran = np.percentile(Y_ran, 50)

    Y_mid_train = Y_mid[p_lag + h_ahead - 1:]
    Train_len = Y_mid_train.shape[0]
    D_mid_1 = (1 * (Y_mid <= gamma_mid))[p_lag - 1:p_lag - 1 + Train_len].reshape(-1, 1)
    D_mid_2 = (1 * (Y_mid > gamma_mid))[p_lag - 1:p_lag - 1 + Train_len].reshape(-1, 1)
    D_ran_1 = (1 * (Y_ran <= gamma_ran))[p_lag - 1:p_lag - 1 + Train_len].reshape(-1, 1)
    D_ran_2 = (1 * (Y_ran > gamma_ran))[p_lag - 1:p_lag - 1 + Train_len].reshape(-1, 1)

    X_mid_train_1 = np.ones(Train_len).reshape(-1, 1)
    for i in range(p_lag):
        X_mid_temp1 = Y_mid[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(-1, 1)
        X_mid_train_1 = np.concatenate((X_mid_train_1,X_mid_temp1),axis=1)
    for i in range(p_lag):
        X_ran_temp1 = Y_ran[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(-1, 1)
        X_mid_train_1 = np.concatenate((X_mid_train_1,X_ran_temp1),axis=1)
    X_mid_train_1 = D_mid_1 * X_mid_train_1

    X_mid_train_2 = np.ones(Train_len).reshape(-1, 1)
    for i in range(p_lag):
        X_mid_temp1 = Y_mid[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(-1, 1)
        X_mid_train_2 = np.concatenate((X_mid_train_2, X_mid_temp1), axis=1)
    for i in range(p_lag):
        X_ran_temp1 = Y_ran[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(-1, 1)
        X_mid_train_2 = np.concatenate((X_mid_train_2, X_ran_temp1), axis=1)
    X_mid_train_2 = D_mid_2 * X_mid_train_2
    X_mid_train = np.c_[X_mid_train_1, X_mid_train_2]

    theta_mid_hat = np.linalg.inv(X_mid_train.T @ X_mid_train) @ X_mid_train.T @ Y_mid_train

    Y_ran_train = Y_ran[p_lag + h_ahead - 1:]
    X_ran_train_1 = np.ones(Train_len).reshape(-1, 1)
    for i in range(p_lag):
        X_mid_temp1 = Y_mid[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(-1, 1)
        X_ran_train_1 = np.concatenate((X_ran_train_1, X_mid_temp1), axis=1)
    for i in range(p_lag):
        X_ran_temp1 = Y_ran[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(-1, 1)
        X_ran_train_1 = np.concatenate((X_ran_train_1, X_ran_temp1), axis=1)
    X_ran_train_1 = D_ran_1 * X_ran_train_1

    X_ran_train_2 = np.ones(Train_len).reshape(-1, 1)
    for i in range(p_lag):
        X_mid_temp1 = Y_mid[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(-1, 1)
        X_ran_train_2 = np.concatenate((X_ran_train_2, X_mid_temp1), axis=1)
    for i in range(p_lag):
        X_ran_temp1 = Y_ran[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(-1, 1)
        X_ran_train_2 = np.concatenate((X_ran_train_2, X_ran_temp1), axis=1)
    X_ran_train_2 = D_ran_2 * X_ran_train_2
    X_ran_train = np.c_[X_ran_train_1, X_ran_train_2]

    theta_ran_hat = np.linalg.inv(X_ran_train.T @ X_ran_train) @ X_ran_train.T @ Y_ran_train

    D_mid_1_test = (1 * (Y_mid <= gamma_mid))[-1]
    D_mid_2_test = (1 * (Y_mid > gamma_mid))[-1]
    D_ran_1_test = (1 * (Y_ran <= gamma_ran))[-1]
    D_ran_2_test = (1 * (Y_ran > gamma_ran))[-1]
    X_mid_test = np.c_[D_mid_1_test * np.c_[1,Y_mid[-1:-1-p_lag:-1].reshape(1,-1), Y_ran[-1:-1-p_lag:-1].reshape(1, -1)],
                       D_mid_2_test * np.c_[1,Y_mid[-1:-1-p_lag:-1].reshape(1,-1), Y_ran[-1:-1-p_lag:-1].reshape(1, -1)]]
    X_ran_test = np.c_[D_ran_1_test * np.c_[1,Y_mid[-1:-1-p_lag:-1].reshape(1,-1), Y_ran[-1:-1-p_lag:-1].reshape(1,-1)],
                       D_ran_2_test * np.c_[1,Y_mid[-1:-1-p_lag:-1].reshape(1,-1), Y_ran[-1:-1-p_lag:-1].reshape(1, -1)]]
    Y_mid_hat = X_mid_test @ (theta_mid_hat)
    Y_ran_hat = X_ran_test @ (theta_ran_hat)
    result = np.array([Y_mid_hat - Y_ran_hat , Y_mid_hat + Y_ran_hat]).reshape(1,-1)
    return result

def CCRM_fun(Y, p_lag, h_ahead):

    Y_c = (Y[:, 0] + Y[:, 1]) / 2
    Y_r = (Y[:, 1] - Y[:, 0]) / 2

    Y_train_c = Y_c[p_lag + h_ahead - 1:]
    Y_train_r = Y_r[p_lag + h_ahead - 1:]

    Train_len = Y_train_c.shape[0]

    X_train_c = np.ones((Train_len, 1))
    X_train_r = np.ones((Train_len, 1))

    for i in range(p_lag):
        X_train_c = np.concatenate((X_train_c,Y_c[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(Train_len, 1)),
                                   axis=1)
        X_train_r = np.concatenate((X_train_r,Y_r[p_lag - i - 1:p_lag - i - 1 + Train_len].reshape(Train_len, 1)),
                                   axis=1)

    X_test_c = np.r_[1, Y_c[-p_lag - h_ahead:-(h_ahead + 2*p_lag):-1]]
    X_test_r = np.r_[1, Y_r[-p_lag - h_ahead:-(h_ahead + 2*p_lag):-1]]

    beta_c_hat = np.linalg.inv(X_train_c.T @ X_train_c) @ X_train_c.T @ Y_train_c
    Y_c_hat = X_test_c @ beta_c_hat

    beta_r_temp = np.zeros(p_lag + 1)
    P_set = np.array([], dtype=int)
    Z_set = np.arange(p_lag+1)

    while True:
        # Step 1
        w = X_train_r.T @ (Y_train_r - X_train_r @ beta_r_temp)
        # Step 2-3
        if len(Z_set) == 0 or np.all(w[Z_set] <= 0):
            beta_r_hat = beta_r_temp.copy()
            break
        # Step 5
        t = Z_set[np.argmax(w[Z_set])]
        # Step 6
        P_set = np.append(P_set, t).astype(int)
        Z_set = Z_set[Z_set != t]
        # Step 7-16
        while True:
            z = np.zeros(p_lag + 1)
            X_p = X_train_r[:, P_set]
            z[P_set] = np.linalg.inv(X_p.T @ X_p) @ X_p.T @ Y_train_r
            # Step 8-10
            if np.all(z[P_set] > 0):
                beta_r_temp = z.copy()
                break
            # Step 12-13
            idx = P_set[z[P_set] <= 0]
            alpha_vec = beta_r_temp[idx] / (beta_r_temp[idx] - z[idx])
            alpha = np.min(alpha_vec)
            # Step 14
            beta_r_temp = beta_r_temp + alpha * (z - beta_r_temp)
            # Step 15
            move_idx = P_set[beta_r_temp[P_set] == 0]
            P_set = np.array([j for j in P_set if j not in move_idx], dtype=int)
            Z_set = np.append(Z_set, move_idx).astype(int)
            # Step 16: back to Step 7

    Y_r_hat = X_test_r @ beta_r_hat
    result = np.array([
        Y_c_hat - Y_r_hat,
        Y_c_hat + Y_r_hat
    ])
    return result


def CIAR_param_initial(Y,p_lag,h_ahead):

    Y_l = Y[:,0]
    Y_u = Y[:,1]
    if np.all(np.isclose(Y_l+Y_u,0,atol=1e-5)):
        Y_l = Y_l + 1e-4
    if np.all(np.isclose(Y_l-Y_u, 0, atol=1e-5)):
        Y_l = Y_l + 1e-4
    Y_train = Y_u[(p_lag + h_ahead - 1):] - Y_l[(p_lag + h_ahead - 1):]
    Train_len = len(Y_train)
    X_train = np.ones((Train_len, 1))
    for i in range(p_lag):
        X_temp = Y_l[p_lag-i-1:p_lag-i-1+Train_len].reshape(Train_len,1)
        X_train = np.concatenate((X_train,X_temp),axis=1)
    for i in range(p_lag):
        X_temp = Y_u[p_lag-i-1:p_lag-i-1+Train_len].reshape(Train_len, 1)
        X_train = np.concatenate((X_train, X_temp), axis=1)

    XtX = X_train.T @ X_train
    epsilon = 1e-10
    XtX_stable = XtX + np.eye(XtX.shape[0]) * epsilon
    delta_beta = np.linalg.pinv(XtX_stable) @ X_train.T @ Y_train
    eps = Y_train - X_train @ delta_beta
    sigma_m = np.std(eps)
    return np.r_[sigma_m, -delta_beta]

def CIAR_step1_mle_fun(theta,Y,p_lag,h_ahead):

    sigma_m = theta[0]
    beta_delta = np.array(theta[1:])

    TT = Y.shape[0]
    Y_l = Y[:, 0]
    Y_u = Y[:, 1]
    Y_delta = Y_u[p_lag + h_ahead - 1:] - Y_l[p_lag + h_ahead - 1:]
    Train_len = len(Y_delta)
    X_train = np.ones((Train_len, 1))
    for i in range(p_lag):
        X_temp = Y_l[p_lag - i -1:p_lag - i -1+Train_len].reshape(Train_len, 1)
        X_train = np.concatenate((X_train, X_temp), axis=1)
    for i in range(p_lag):
        X_temp = Y_u[p_lag - i -1:p_lag - i -1+Train_len].reshape(Train_len, 1)
        X_train = np.concatenate((X_train, X_temp), axis=1)

    Delta_ = X_train @ beta_delta
    log_like = (-np.log(sigma_m) + norm.logpdf(Y_delta / sigma_m + Delta_ / sigma_m) - norm.logsf(Delta_ / sigma_m))
    return np.sum(-log_like)/TT

def CIAR_fun(Y,p_lag,h_ahead):
    Y = np.sort(Y,axis=1)
    Y_l = Y[:, 0]
    Y_u = Y[:, 1]
    Y_l_train = Y_l[p_lag+h_ahead-1:]
    Y_u_train = Y_u[p_lag+h_ahead-1:]
    Train_len = Y_l_train.shape[0]
    X_train = np.ones((Train_len, 1))
    for i in range(p_lag):
        X_temp = Y_l[p_lag-i-1:p_lag-i-1+Train_len].reshape(Train_len, 1)
        X_train = np.concatenate((X_train, X_temp), axis=1)
    for i in range(p_lag):
        X_temp = Y_u[p_lag-i-1:p_lag-i-1+Train_len].reshape(Train_len, 1)
        X_train = np.concatenate((X_train, X_temp), axis=1)

    X_test = np.r_[1,Y_l[-1:-1-p_lag:-1],Y_u[-1:-1-p_lag:-1]]


    init_params = CIAR_param_initial(Y, p_lag, h_ahead)
    bounds_params = [(None, None)] * len(init_params)
    bounds_params[0] = (1e-10, None)
    res = minimize(
        CIAR_step1_mle_fun,
        init_params,
        args=(Y,p_lag,h_ahead),
        bounds=bounds_params,
        method='L-BFGS-B'
    )
    sigma_m_hat = res.x[0]
    beta_delta_hat = res.x[1:]
    Delta_hat = X_train @ beta_delta_hat
    z_hat = Delta_hat / sigma_m_hat
    lambda_hat = np.exp(norm.logpdf(z_hat) - norm.logsf(z_hat))
    H_hat = ( np.concatenate((X_train,lambda_hat.reshape(Train_len, 1)), axis=1))
    HtH=H_hat.T @ H_hat
    epsilon = 1e-10
    HtH_stable = HtH + np.eye(HtH.shape[0]) * epsilon
    gamma_l_hat = (np.linalg.inv(HtH_stable) @ H_hat.T @ Y_l_train)
    gamma_u_hat = (np.linalg.inv(HtH_stable) @ H_hat.T @ Y_u_train)

    Delta_final = X_test @ beta_delta_hat
    z_final = Delta_final / sigma_m_hat
    lambda_final = np.exp(norm.logpdf(z_final) - norm.logsf(z_final))
    X_test_final = np.r_[X_test, lambda_final]

    result = np.r_[X_test_final @ gamma_l_hat,X_test_final @ gamma_u_hat]
    return result

def ACI_fun(Y,p_lag,h_ahead,K11=5,K12=1,K21=1,K22=1):

    K = np.array([K11, K12, K21, K22]).reshape(2, 2)

    Y_int_train = Y[p_lag + h_ahead -1:].reshape(-1, 1)
    Train_len = Y[p_lag + h_ahead -1:].shape[0]
    I0_int = np.ones((Train_len, 2)).reshape(-1, 1)
    I1_int = np.concatenate((
        np.full((Train_len, 1), -0.5),
        np.full((Train_len, 1), 0.5)
    ), axis=1).reshape(-1, 1)
    X_int_train = np.concatenate((I0_int,I1_int), axis=1)
    for i in range(p_lag):
        X_int_temp = Y[p_lag-i-1:p_lag-i-1+Train_len].reshape(-1, 1)
        X_int_train = np.concatenate((X_int_train,X_int_temp),axis=1)

    X_int_test = np.array([[1, -0.5], [1, 0.5]])
    X_int_temp = Y[-1:-1- p_lag:-1].T
    X_int_test = np.concatenate((X_int_test, X_int_temp), axis=1)

    theta_hat = np.linalg.inv(
        (X_int_train.T @ np.kron(np.eye(int(X_int_train.shape[0] / 2)), K) @ X_int_train)
    ) @ (X_int_train.T @ np.kron(np.eye(int(X_int_train.shape[0] / 2)), K) @ Y_int_train)

    result = (X_int_test @ theta_hat).T
    return result

def TARI_fun(Y,p_lag,h_ahead,d_lag=0,K11=5,K12=1,K21=1,K22=1):

    K = np.array([K11, K12, K21, K22]).reshape(2, 2)

    Y_mid = np.mean(Y, axis=1)
    gamma_ = np.percentile(Y_mid, 50)
    Indicator1 = 1 * (Y_mid < gamma_)
    Indicator2 = 1 * (Y_mid >= gamma_)

    Y_int_train = Y[p_lag+h_ahead-1:].reshape(-1, 1)
    Train_len = Y[p_lag+h_ahead-1:].shape[0]

    I0_int = np.ones((Train_len, 2)).reshape(-1, 1)
    I1_int = np.concatenate((
        np.full((Train_len, 1), -0.5),
        np.full((Train_len, 1), 0.5)
    ),axis=1).reshape(-1, 1)
    X_int_train = np.concatenate((I0_int,I1_int), axis=1)

    Indicator1_series = Indicator1[p_lag-d_lag-1:p_lag-d_lag-1+Train_len].reshape(-1, 1)
    Indicator1_series2 = np.concatenate((Indicator1_series, Indicator1_series), axis=1)
    Indicator2_series = Indicator2[p_lag-d_lag-1:p_lag-d_lag-1+Train_len].reshape(-1, 1)
    Indicator2_series2 = np.concatenate((Indicator2_series, Indicator2_series), axis=1)

    for i in range(p_lag):
        X_int_temp1 = Y[p_lag - i - 1:p_lag - i - 1 + Train_len]
        X_int_temp2 = (X_int_temp1 * Indicator1_series2).reshape(-1, 1)
        X_int_train = np.concatenate((X_int_train,X_int_temp2),axis=1)

    for i in range(p_lag):
        X_int_temp1 = Y[p_lag - i - 1:p_lag - i - 1 + Train_len]
        X_int_temp2 = (X_int_temp1 * Indicator2_series2).reshape(-1, 1)
        X_int_train = np.concatenate((X_int_train, X_int_temp2), axis=1)

    X_int_test = np.array([[1, -0.5], [1, 0.5]])
    X_int_temp1 = Y[-1:-1-p_lag:-1].T * Indicator1[-1-h_ahead]
    X_int_temp2 = Y[-1:-1-p_lag:-1].T * Indicator2[-1-h_ahead]
    X_int_test = np.concatenate((X_int_test, X_int_temp1, X_int_temp2), axis=1)

    XtKX = (X_int_train.T @ np.kron(np.eye(int(X_int_train.shape[0] / 2)), K) @ X_int_train)
    epsilon = 1e-10
    XtKX_stable = XtKX + np.eye(XtKX.shape[0]) * epsilon
    theta_hat = np.linalg.inv(
        XtKX_stable
    ) @ (X_int_train.T @ np.kron(np.eye(int(X_int_train.shape[0] / 2)), K) @ Y_int_train)

    result = (X_int_test @ theta_hat).T
    return result

def _prepare_inputs(x, x0, H):

    x = np.asarray(x, dtype=float)
    x0 = np.asarray(x0, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x0.ndim == 1:
        x0 = x0.reshape(-1, 1)

    if x.shape != x0.shape:
        raise ValueError(f"x.shape={x.shape} and x0.shape={x0.shape} must be the same.")

    n, p = x.shape

    if np.isscalar(H):
        H = float(H) * np.eye(p)
    else:
        H = np.asarray(H, dtype=float)
        if H.shape != (p, p):
            raise ValueError(f"H.shape={H.shape} must be ({p}, {p}) for p={p}.")

    return x, x0, H, n, p

def norm_kernel_fun(x, x0, H):

    x, x0, H, n, p = _prepare_inputs(x, x0, H)

    diff = x - x0
    H_inv = np.linalg.inv(H)
    det_H = np.linalg.det(H)

    if det_H <= 0:
        raise ValueError("H must be positive definite, so det(H) > 0.")

    quad = np.einsum("ij,jk,ik->i", diff, H_inv, diff)

    const = (2.0 * np.pi) ** (-p / 2.0) * det_H ** (-0.5)
    return const * np.exp(-0.5 * quad)

def epan_kernel_fun(x, x0, H):

    x, x0, H, n, p = _prepare_inputs(x, x0, H)

    diff = x - x0
    H_inv = np.linalg.inv(H)
    det_H = np.linalg.det(H)

    if det_H <= 0:
        raise ValueError("H must be positive definite, so det(H) > 0.")

    quad = np.einsum("ij,jk,ik->i", diff, H_inv, diff)

    volume_unit_ball = pi ** (p / 2.0) / gamma(p / 2.0 + 1.0)

    c_p = (p + 2.0) / (2.0 * volume_unit_ball)

    out = np.zeros(n, dtype=float)
    mask = quad <= 1.0
    out[mask] = det_H ** (-0.5) * c_p * (1.0 - quad[mask])

    return out

def VC_ACI_2D_fun(Y,p_lag=1,h_ahead=1,K11=5,K12=1,K21=1,K22=1):

    K = np.array([K11, K12, K21, K22]).reshape(2, 2)
    TT = Y.shape[0]
    Y_mid = (Y[:,0] + Y[:,1]) / 2
    Y_mid = (Y_mid - np.min(Y_mid)) / (np.max(Y_mid)-np.min(Y_mid))
    Y_ran = (Y[:,1] - Y[:,0]) / 2
    Y_ran = (Y_ran - np.min(Y_ran)) / (np.max(Y_ran)-np.min(Y_ran))

    Y_int_train = Y[(p_lag + h_ahead - 1):].reshape(-1, 1)

    I0_int = np.ones((TT - p_lag - h_ahead + 1, 2)).reshape(-1, 1)
    I1_int = np.concatenate((
        np.full((TT - p_lag - h_ahead + 1, 1), -0.5),
        np.full((TT - p_lag - h_ahead + 1, 1), 0.5)
    ), axis=1).reshape(-1, 1)
    X_int_train = np.concatenate((I0_int, I1_int), axis=1)

    u_0 = 0.5
    v_0 = 0.5
    up = Y_mid[(p_lag - 1):(TT - h_ahead)].reshape(-1, 1)
    vp = Y_ran[(p_lag - 1):(TT - h_ahead)].reshape(-1, 1)
    up_vp = np.concatenate((up, vp), axis=1)
    up_m_u0 = (Y_mid[(p_lag - 1):(TT - h_ahead)] - u_0).reshape(-1, 1)
    vp_m_v0 = (Y_ran[(p_lag - 1):(TT - h_ahead)] - v_0).reshape(-1, 1)
    up_m_u0_2 = up_m_u0 ** 2
    vp_m_u0_2 = vp_m_v0 ** 2
    up_m_u0_vp_m_u0 = up_m_u0 * vp_m_v0

    for i in range(p_lag):
        X_int_temp1_int = Y[(p_lag - 1 - i):(TT - i - h_ahead)]
        X_int_temp1 = X_int_temp1_int.reshape(-1, 1)
        X_int_temp2 = (X_int_temp1_int * up_m_u0).reshape(-1, 1)
        X_int_temp3 = (X_int_temp1_int * vp_m_v0).reshape(-1, 1)
        X_int_temp4 = (X_int_temp1_int * up_m_u0_2).reshape(-1, 1)
        X_int_temp5 = (X_int_temp1_int * vp_m_u0_2).reshape(-1, 1)
        X_int_temp6 = (X_int_temp1_int * up_m_u0_vp_m_u0).reshape(-1, 1)

        X_int_train = np.concatenate((X_int_train, X_int_temp1, X_int_temp2,
                                      X_int_temp3, X_int_temp4, X_int_temp5, X_int_temp6), axis=1)

    u_range = Y_mid.max() - Y_mid.min()
    v_range = Y_ran.max() - Y_ran.min()
    hu_select = (u_range ** (1 / 5)) ** range(1, 1 + 5)
    hv_select = (v_range ** (1 / 5)) ** range(1, 1 + 5)

    loss = 1e10
    h_u_opt = hu_select[0]
    h_v_opt = hv_select[0]
    K_opt = K
    for h_u in hu_select:
        for h_v in hv_select:
            H = np.diag([h_u, h_v])
            kernel_w = norm_kernel_fun(up_vp, np.full_like(up_vp,0.5), H)
            theta_hat = np.linalg.inv(
                (X_int_train.T @ np.kron(np.diag(kernel_w), K) @ X_int_train)
            ) @ (X_int_train.T @ np.kron(np.diag(kernel_w), K) @ Y_int_train)
            eps_int = (Y_int_train - X_int_train @ theta_hat).reshape(-1, 2)

            K1 = np.zeros((2, 2))
            K1[0, 0] = np.cov(eps_int.T)[1, 1]
            K1[1, 1] = np.cov(eps_int.T)[0, 0]
            K1[1, 0] = -np.cov(eps_int.T)[1, 0]
            K1[0, 1] = -np.cov(eps_int.T)[1, 0]

            if np.linalg.det(K1) > 0:
                K = K1.copy()

            loss1 = np.sum(eps_int[:, 0] ** 2 + eps_int[:, 1] ** 2)
            if loss1 < loss:
                loss = loss1.copy()
                h_u_opt = h_u
                h_v_opt = h_v
                K_opt = K

    H = np.diag([h_u_opt, h_v_opt])
    kernel_w = norm_kernel_fun(up_vp, np.full_like(up_vp,0.5), H)
    XtWKX = (X_int_train.T @ np.kron(np.diag(kernel_w), K_opt) @ X_int_train)
    epsilon = 1e-5
    XtWKX_stable = XtWKX + np.eye(XtWKX.shape[0]) * epsilon
    theta_hat_opt = np.linalg.inv(
        XtWKX_stable
    ) @ (X_int_train.T @ np.kron(np.diag(kernel_w), K_opt) @ Y_int_train)

    up_m_u0_test = (Y_mid[-1] - u_0).reshape(-1, 1)
    vp_m_v0_test = (Y_ran[-1] - v_0).reshape(-1, 1)
    up_m_u0_2_test = up_m_u0_test ** 2
    vp_m_u0_2_test = vp_m_v0_test ** 2
    up_m_u0_vp_m_u0_test = up_m_u0_test * vp_m_v0_test
    I0_int = np.ones(2).reshape(-1, 1)
    I1_int = np.array([-0.5,0.5]).reshape(-1, 1)
    X_int_test = np.concatenate((I0_int, I1_int), axis=1)
    for i in range(p_lag):
        X_int_temp1_int = Y[-(i+1)]
        X_int_temp1 = X_int_temp1_int.reshape(-1, 1)
        X_int_temp2 = (X_int_temp1_int * up_m_u0_test).reshape(-1, 1)
        X_int_temp3 = (X_int_temp1_int * vp_m_v0_test).reshape(-1, 1)
        X_int_temp4 = (X_int_temp1_int * up_m_u0_2_test).reshape(-1, 1)
        X_int_temp5 = (X_int_temp1_int * vp_m_u0_2_test).reshape(-1, 1)
        X_int_temp6 = (X_int_temp1_int * up_m_u0_vp_m_u0_test).reshape(-1, 1)

        X_int_test = np.concatenate((X_int_test, X_int_temp1, X_int_temp2,
                                      X_int_temp3, X_int_temp4, X_int_temp5, X_int_temp6), axis=1)
    result = np.ravel(X_int_test @ theta_hat_opt)
    return result

def MLP_fun(data,p=6,h=1,hidden_layer_sizes=(4, 2),activation="relu",
            solver="adam",alpha=1e-4,learning_rate_init=1e-3,max_iter=2000,random_state=42):

    data = np.asarray(data, dtype=float)

    X, Y = [], []
    T = len(data)
    for t in range(T - p - h + 1):
        X.append(data[t:t + p, :].reshape(-1))
        Y.append(data[t + p + h - 1, :])

    X = np.asarray(X)
    Y = np.asarray(Y)

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state
    )
    model.fit(X_scaled, Y_scaled)

    x_last = data[-p:, :].reshape(1, -1)
    x_last_scaled = scaler_X.transform(x_last)
    forecast_scaled = model.predict(x_last_scaled)
    forecast = scaler_Y.inverse_transform(forecast_scaled)[0]

    return forecast


# endregion

# region II. loss functions

def dm_test_fun(e1, e2, h=1, power=1):

    e1 = np.asarray(e1)
    e2 = np.asarray(e2)

    T = len(e1)

    # loss differential
    d = np.abs(e1) ** power - np.abs(e2) ** power

    d_bar = np.mean(d)

    # HAC variance
    gamma = []

    for lag in range(h):
        cov = np.cov(d[lag:], d[:T-lag], bias=True)[0,1]
        gamma.append(cov)

    var_d = gamma[0] + 2 * np.sum(gamma[1:])

    DM = d_bar / np.sqrt(var_d / T)

    # Harvey et al small sample correction
    adj = np.sqrt((T + 1 - 2*h + h*(h-1)/T)/T)

    DM = DM * adj

    p_value = 2 * t.cdf(-abs(DM), df=T-1)

    return DM, p_value

def int_loss_testsig_fun(loss_mat,h=1):
    model_num = loss_mat.shape[1]
    result_mat = np.zeros((model_num, 3))
    mcs = MCS(loss_mat, size=0.05, reps=50000)
    mcs.compute()
    pv = mcs.pvalues.copy()
    result_mat[:, 0] = loss_mat.mean(axis=0).round(4)
    result_mat[:, 1] = pv.sort_index().to_numpy().reshape(-1).round(3)
    for i in range(model_num):
        result_mat[i, 2] = dm_test_fun(loss_mat[:, -1], loss_mat[:, i],h)[1].round(3)
    return result_mat

def int_loss_single_fun(forecast_int,real_int,model_num,asset_num):

    X = forecast_int[model_num,asset_num]
    TT = X.shape[0]
    Y = real_int[-TT:,(3*asset_num):(3*asset_num+2)]
    X_mid = X.mean(axis=1)
    Y_mid = Y.mean(axis=1)
    X_ran = X[:,1] - X_mid
    Y_ran = Y[:,1] - Y_mid
    Delta_mid = Y_mid - X_mid
    Delta_ran = Y_ran - X_ran
    Delta_left = X[:,0] - Y[:,0]
    Delta_right = X[:,1] - Y[:,1]

    MDE_mr = np.sqrt(Delta_mid ** 2 + Delta_ran ** 2)
    MDE_lr = np.sqrt(Delta_left ** 2 + Delta_right ** 2)

    return MDE_mr,MDE_lr

def int_loss_fun(forecast_int,real_int,asset_num,h=1):
    model_num = forecast_int.shape[0]
    TT = forecast_int.shape[2]
    result = int_loss_single_fun(forecast_int, real_int, 0, asset_num)
    lossfun_num = 2
    loss_single_array = np.zeros([lossfun_num,TT,model_num])
    result_mat = np.zeros([model_num, lossfun_num*3])

    for i in range(lossfun_num):
        loss_single_array[i,:,0] = result[i]

    for i in range(1,model_num):
        result = int_loss_single_fun(forecast_int,real_int,i,asset_num)
        for j in range(lossfun_num):
            loss_single_array[j, :, i] = result[j]

    for i in range(lossfun_num):
        result_mat[:, (3*i):(3*i+3)] = int_loss_testsig_fun(loss_single_array[i],h)

    return result_mat

def int_cumloss_fun(forecast_int,real_int,asset_type,fore_type,h_ahead=1):

    X = forecast_int[fore_type,asset_type]
    TT = X.shape[0]
    Y = real_int[-TT:,(3*asset_type):(3*asset_type+2)]
    Y_1 = real_int[-TT-h_ahead:-h_ahead,(3*asset_type):(3*asset_type+2)]
    Y_M = forecast_int[6,asset_type]

    X_mid = X.mean(axis=1)
    X_ran = X[:,1] - X_mid
    X_l = X[:, 0]
    X_r = X[:, 1]

    Y_mid = Y.mean(axis=1)
    Y_ran = Y[:,1] - Y_mid
    Y_l = Y[:, 0]
    Y_r = Y[:, 1]

    Y_1_mid = Y_1.mean(axis=1)
    Y_1_ran = Y_1[:,1] - Y_1_mid
    Y_1_l = Y_1[:, 0]
    Y_1_r = Y_1[:, 1]

    Y_M_mid = Y_M.mean(axis=1)
    Y_M_ran = Y_M[:,1] - Y_M_mid
    Y_M_l = Y_M[:,0]
    Y_M_r = Y_M[:,1]

    Y_M_mid_ran = (np.sqrt((X_mid - Y_mid)**2+ (X_ran - Y_ran)**2) -
               np.sqrt((Y_M_mid - Y_mid)**2 + (Y_M_ran - Y_ran)**2))
    Y_M_l_r = (np.sqrt((X_l - Y_l)**2+ (X_r - Y_r)**2)-
               np.sqrt((Y_M_l - Y_l)**2+ (Y_M_r - Y_r)**2))
    Y_1_mid_ran = (np.sqrt((X_mid - Y_mid)**2 + (X_ran - Y_ran)**2) -
                   np.sqrt((Y_1_mid - Y_mid)**2 + (Y_1_ran - Y_ran)**2))
    Y_1_l_r = (np.sqrt((X_l - Y_l)**2 + (X_r - Y_r)**2)-
               np.sqrt((Y_1_l - Y_l)**2+ (Y_1_r - Y_r)**2))

    cum_Y_M_mid_ran = np.cumsum(Y_M_mid_ran).reshape(-1,1)
    cum_Y_M_l_r = np.cumsum(Y_M_l_r).reshape(-1,1)
    cum_Y_1_mid_ran = np.cumsum(Y_1_mid_ran).reshape(-1,1)
    cum_Y_1_l_r = np.cumsum(Y_1_l_r).reshape(-1,1)

    return np.concatenate((cum_Y_M_mid_ran,cum_Y_M_l_r,cum_Y_1_mid_ran,cum_Y_1_l_r),axis=1)

def int_coverage_fun(forecast_int,real_int,asset_num):
    X = forecast_int[:,asset_num,:,:]
    TT = X.shape[1]
    Y = real_int[-TT:, (3 * asset_num):(3 * asset_num + 2)]
    model_totalnum = X.shape[0]
    result = np.zeros((model_totalnum,2))
    for model_num in range(model_totalnum):
        X_temp = X[model_num,:,:]
        inter_xy = np.zeros(TT)
        union_xy = np.zeros(TT)
        R_x = X_temp[:, 1] - X_temp[:, 0]
        R_y = Y[:, 1] - Y[:, 0]
        for i in range(TT):
            if P.open(X_temp[i, 0], X_temp[i, 1]).overlaps(P.open(Y[i, 0], Y[i, 1])):
                inter_xy[i] = np.sort(np.r_[X_temp[i], Y[i]])[2] - np.sort(np.r_[X_temp[i], Y[i]])[1]
                union_xy[i] = np.sort(np.r_[X_temp[i], Y[i]])[3] - np.sort(np.r_[X_temp[i], Y[i]])[0]
            else:
                inter_xy[i] = 0
                union_xy[i] = X_temp[i, 1] - X_temp[i, 0] + Y[i, 1] - Y[i, 0]

        numCR = np.mean(inter_xy / R_y).round(3)
        numER = np.mean(inter_xy / R_x).round(3)
        result[model_num,:] = np.array([numCR,numER]).round(3)
    return result.round(3)


# endregion


