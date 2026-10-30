import numpy as np
import pandas as pd
from Decomposition_Algorithm_03 import int_inprod_fun, int_cov_fun
from Interval_Forecast_Fun_02 import dm_test_fun
from arch.bootstrap import MCS
from scipy.optimize import minimize, LinearConstraint

# import data
fore_econapp=fore_f1_p4_w84_l3_ALL[7:,:,:,:]

# region 1. Figure 10 & Table 5-6
fore_len = fore_econapp.shape[2]
model_num = fore_econapp.shape[0]
vfore_f1_p4_w84_l3_ALL = np.zeros((model_num, fore_len, 4))  #  model x forecast x asset
for asset_j in range(4):
    for model_i in range(model_num):
        real_close_price = df_price[-2:-2 - fore_len:-1, 12 + asset_j].reshape(-1, 1)
        est_loglowhigh = np.log(fore_econapp[model_i, asset_j] * real_close_price + real_close_price)
        vol_temp = 0.3606 * (est_loglowhigh[:, 1] - est_loglowhigh[:, 0]) ** 2
        vfore_f1_p4_w84_l3_ALL[model_i, :, asset_j] = pd.Series(vol_temp).ffill().to_numpy()


def vol_loss_testsig_fun(loss_mat):
    model_num = loss_mat.shape[1]
    result_mat = np.zeros((model_num, 3))
    mcs = MCS(loss_mat, size=0.05, reps=50000)
    mcs.compute()
    pv = mcs.pvalues.copy()
    result_mat[:, 0] = loss_mat.mean(axis=0).round(4)
    result_mat[:, 1] = pv.sort_index().to_numpy().reshape(-1).round(3)
    for i in range(model_num):
        result_mat[i, 2] = dm_test_fun(loss_mat[:, -1], loss_mat[:, i])[1].round(3)
    return result_mat

def vol_loss_single_fun(forecast_vol, real_vol, model_num, asset_num):

    X = np.sqrt(forecast_vol[model_num, :, asset_num])
    TT = X.shape[0]
    Y = np.sqrt(real_vol[-TT:, asset_num])
    qlike = Y / X - np.log(Y / X) - 1
    mse = (Y - X) ** 2
    return qlike, mse

def vol_loss_fun(forecast_vol, real_vol, asset_num):
    model_num = forecast_vol.shape[0]
    TT = forecast_vol.shape[1]
    result = vol_loss_single_fun(forecast_vol, real_vol, 0, asset_num)
    lossfun_num = len(result)
    loss_single_array = np.zeros([lossfun_num, TT, model_num])
    result_mat = np.zeros([model_num, lossfun_num * 3])

    for i in range(lossfun_num):
        loss_single_array[i, :, 0] = result[i]

    for i in range(1, model_num):
        result = vol_loss_single_fun(forecast_vol, real_vol, i, asset_num)
        for j in range(lossfun_num):
            loss_single_array[j, :, i] = result[j]

    for i in range(lossfun_num):
        temp = loss_single_array[i]
        temp[np.isinf(temp)] = 1e8
        temp[np.isnan(temp)] = 1e8
        result_mat[:, (3 * i):(3 * i + 3)] = vol_loss_testsig_fun(temp)

    return result_mat

real_vol = rv_dt
forecast_vol = vfore_f1_p4_w84_l3_ALL
vloss_csv_colnames = ['Qlike', 'Qlike_MCS', 'Qlike_DM', 'MSE', 'MSE_MCS', 'MSE_DM']

asset_num = 0
vloss_f1_p4_w84_l3_ALL = vol_loss_fun(vfore_f1_p4_w84_l3_ALL, rv_dt, asset_num)
pd.DataFrame(vloss_f1_p4_w84_l3_ALL,
             columns=vloss_csv_colnames).to_csv(r"figures\figure10\vol_loss_0.csv", index=False)
asset_num = 1
vloss_f1_p4_w84_l3_ALL = vol_loss_fun(vfore_f1_p4_w84_l3_ALL, rv_dt, asset_num)
pd.DataFrame(vloss_f1_p4_w84_l3_ALL,
             columns=vloss_csv_colnames).to_csv(r"figures\figure10\vol_loss_1.csv", index=False)
asset_num = 2
vloss_f1_p4_w84_l3_ALL = vol_loss_fun(vfore_f1_p4_w84_l3_ALL, rv_dt, asset_num)
pd.DataFrame(vloss_f1_p4_w84_l3_ALL,
             columns=vloss_csv_colnames).to_csv(r"figures\figure10\vol_loss_2.csv", index=False)
asset_num = 3
vloss_f1_p4_w84_l3_ALL = vol_loss_fun(vfore_f1_p4_w84_l3_ALL, rv_dt, asset_num)
pd.DataFrame(vloss_f1_p4_w84_l3_ALL,
             columns=vloss_csv_colnames).to_csv(r"figures\figure10\vol_loss_3.csv", index=False)

# endregion

# region 2. Figure 11
def int_cov2_fun(X,Y,X_mean,Y_mean,K11=1,K12=0,K21=0,K22=1):
    result = (int_inprod_fun(X,Y,K11,K12,K21,K22) -
              int_inprod_fun(X_mean,Y_mean,K11,K12,K21,K22))
    return result

def Int_MV_fun(int_forecast, int_real, lam, lam1, info_lag=60,GMV=False,lowerB = 0, upperB = 1):

    TT = int_forecast.shape[2]
    asset_tolnum = int_forecast.shape[1]
    model_tolnum = int_forecast.shape[0]
    weights = np.zeros([model_tolnum, TT, asset_tolnum])

    bounds_params = [(lowerB, upperB)] * asset_tolnum
    w_initial = np.ones(asset_tolnum) / asset_tolnum
    A = np.ones((1, asset_tolnum))
    linear_constraint = LinearConstraint(A, lb=1, ub=1)

    for model_num in range(model_tolnum):
        for tt in range(TT):
            int_real_temp = int_real[-TT + tt - 1:-TT - info_lag + tt - 1:-1]
            int_forecast_temp = int_forecast[model_num, :, tt]

            def obj_fun(w):
                part1 = sum(
                    np.array([lam1, 1 - lam1]) * np.sum(w.reshape(-1, 1) * int_forecast_temp, axis=0)
                )

                var_temp = np.zeros([asset_tolnum, asset_tolnum])
                for i in range(asset_tolnum):
                    for j in range(asset_tolnum):
                        var_temp[i, j] = int_cov2_fun(int_real_temp[:,[2*i,2*i+1]],int_real_temp[:,[2*j,2*j+1]],
                                                      int_forecast_temp[i],int_forecast_temp[j],5,1,1,1)
                part2 = w @ var_temp @ w
                if GMV:
                    result = - part2
                else:
                    result = part1 - lam / 2 * part2
                return - result

            res = minimize(
                obj_fun,
                w_initial,
                bounds=bounds_params,
                method="SLSQP",
                constraints=[linear_constraint]
            )
            if res.success:
                weights[model_num, tt] = res.x
            else:
                weights[model_num, tt] = w_initial

    return weights

def Int_MV_ev_fun(weights, int_real, lam, lam1, info_lag=60,GMV=False):
    TT = weights.shape[1]
    model_tolnum = weights.shape[0]
    evaluations_t = np.zeros([TT, model_tolnum])
    asset_tolnum = int(int_real.shape[1]/2)

    for model_num in range(model_tolnum):
        for tt in range(TT):
            int_real_temp = int_real[-TT + tt - 1:-TT - info_lag + tt - 1:-1,:]
            w = weights[model_num, tt]

            part1 = sum(
                np.array([lam1, 1 - lam1]) * np.sum(w.reshape(-1, 1) * int_real_temp[1].reshape(-1,2), axis=0)
            )

            var_temp = np.zeros([asset_tolnum, asset_tolnum])
            for i in range(asset_tolnum):
                for j in range(asset_tolnum):
                    var_temp[i, j] = int_cov_fun(int_real_temp[:, [2 * i, 2 * i + 1]],
                                                  int_real_temp[:, [2 * j, 2 * j + 1]], 5, 1, 1, 1)
            part2 = w @ var_temp @ w
            if GMV:
                evaluations_t[tt, model_num] = - part2
            else:
                evaluations_t[tt, model_num] = part1 - lam / 2 * part2

    return np.mean(evaluations_t, axis=0)

int_real = np.delete(r_returns, [2, 5, 8, 11], axis=1)

weights_MV = Int_MV_fun(fore_econapp, int_real, 10, 0.5,lowerB = -0.2, upperB = 1.2)
weights_MV_right = Int_MV_fun(fore_econapp, int_real, 10, 0.2,lowerB = -0.2, upperB = 1.2)
weights_MV_left = Int_MV_fun(fore_econapp, int_real, 10, 0.8,lowerB = -0.2, upperB = 1.2)
weights_GMV = Int_MV_fun(fore_econapp, int_real, 0, 0.5,GMV=True,lowerB = -0.2, upperB = 1.2)

weights_MV_S = Int_MV_fun(fore_econapp, int_real, 10, 0.5,lowerB = -0.5, upperB = 1.5)
weights_MV_right_S = Int_MV_fun(fore_econapp, int_real, 10, 0.2,lowerB = -0.5, upperB = 1.5)
weights_MV_left_S = Int_MV_fun(fore_econapp, int_real, 10, 0.8,lowerB = -0.5, upperB = 1.5)
weights_GMV_S = Int_MV_fun(fore_econapp, int_real, 0, 0.5,GMV=True,lowerB = -0.5, upperB = 1.5)

eva_MV = Int_MV_ev_fun(weights_MV, int_real, 10, 0.5)
eva_MV_right = Int_MV_ev_fun(weights_MV_right, int_real, 10, 0.2)
eva_MV_left = Int_MV_ev_fun(weights_MV_left, int_real, 10, 0.8)
eva_GMV = Int_MV_ev_fun(weights_GMV, int_real, 0, 0.5,GMV=True)

eva_MV_S = Int_MV_ev_fun(weights_MV_S, int_real, 10, 0.5)
eva_MV_right_S = Int_MV_ev_fun(weights_MV_right_S, int_real, 10, 0.2)
eva_MV_left_S = Int_MV_ev_fun(weights_MV_left_S, int_real, 10, 0.8)
eva_GMV_S = Int_MV_ev_fun(weights_GMV_S, int_real, 0, 0.5,GMV=True)

pd.DataFrame(eva_MV, columns=['MCS']).to_csv(r"figures\figure11\pf_MV.csv",index=False)
pd.DataFrame(eva_MV_right, columns=['MCS']).to_csv(r"figures\figure11\pf_MV_right.csv", index=False)
pd.DataFrame(eva_MV_left, columns=['MCS']).to_csv(r"figures\figure11\pf_MV_left.csv",index=False)
pd.DataFrame(eva_GMV, columns=['MCS']).to_csv(r"figures\figure11\pf_GMV.csv", index=False)

pd.DataFrame(eva_MV_S, columns=['MCS']).to_csv(r"figures\figure11\pf_MV_S.csv",index=False)
pd.DataFrame(eva_MV_right_S, columns=['MCS']).to_csv(r"figures\figure11\pf_MV_right_S.csv", index=False)
pd.DataFrame(eva_MV_left_S, columns=['MCS']).to_csv(r"figures\figure11\pf_MV_left_S.csv",index=False)
pd.DataFrame(eva_GMV_S, columns=['MCS']).to_csv(r"figures\figure11\pf_GMV_S.csv", index=False)

# endregion





