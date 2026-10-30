import pandas as pd
import numpy as np
import random as rnd
import os
from Interval_Forecast_Fun_02 import MLP_fun, CCRM_fun, CIAR_fun, ACI_fun, TARI_fun, VC_ACI_2D_fun, int_loss_fun, \
    CRSETAR_fun, int_cumloss_fun,int_coverage_fun
from Decomposition_Algorithm_03 import BEMD_fun, SVD_fun, ISVD_fun, BVMD_fun, int_cov_fun, int_period_fun

os.chdir(r"...\Code and Data")

#set random seed
rnd.seed(950818)

# region 1. Import ITS data & Figure 2
df = pd.read_csv(r"data\data.csv")
df = df.dropna().reset_index(drop=True)
dates = pd.to_datetime(df.iloc[1:, 0])
df_price = df.iloc[:, 1:].to_numpy()
assets = df.columns[1:]
r_returns = np.zeros((len(dates), 12))

for tt in range(r_returns.shape[0]):
    r_returns[tt, 0] = ((df.loc[tt + 1, 'SPX.GI.l'] - df.loc[tt, 'SPX.GI.c']) /
                        df.loc[tt, 'SPX.GI.c'])
    r_returns[tt, 1] = ((df.loc[tt + 1, 'SPX.GI.h'] - df.loc[tt, 'SPX.GI.c']) /
                        df.loc[tt, 'SPX.GI.c'])
    r_returns[tt, 2] = ((df.loc[tt + 1, 'SPX.GI.c'] - df.loc[tt, 'SPX.GI.c']) /
                        df.loc[tt, 'SPX.GI.c'])

    r_returns[tt, 3] = ((df.loc[tt + 1, 'CL.NYM.l'] - df.loc[tt, 'CL.NYM.c']) /
                        df.loc[tt, 'CL.NYM.c'])
    r_returns[tt, 4] = ((df.loc[tt + 1, 'CL.NYM.h'] - df.loc[tt, 'CL.NYM.c']) /
                        df.loc[tt, 'CL.NYM.c'])
    r_returns[tt, 5] = ((df.loc[tt + 1, 'CL.NYM.c'] - df.loc[tt, 'CL.NYM.c']) /
                        df.loc[tt, 'CL.NYM.c'])

    r_returns[tt, 6] = ((df.loc[tt + 1, 'TY.CBT.l'] - df.loc[tt, 'TY.CBT.c']) /
                        df.loc[tt, 'TY.CBT.c'])
    r_returns[tt, 7] = ((df.loc[tt + 1, 'TY.CBT.h'] - df.loc[tt, 'TY.CBT.c']) /
                        df.loc[tt, 'TY.CBT.c'])
    r_returns[tt, 8] = ((df.loc[tt + 1, 'TY.CBT.c'] - df.loc[tt, 'TY.CBT.c']) /
                        df.loc[tt, 'TY.CBT.c'])

    r_returns[tt, 9] = ((df.loc[tt + 1, 'USDX.FX.l'] - df.loc[tt, 'USDX.FX.c']) /
                        df.loc[tt, 'USDX.FX.c'])
    r_returns[tt, 10] = ((df.loc[tt + 1, 'USDX.FX.h'] - df.loc[tt, 'USDX.FX.c']) /
                         df.loc[tt, 'USDX.FX.c'])
    r_returns[tt, 11] = ((df.loc[tt + 1, 'USDX.FX.c'] - df.loc[tt, 'USDX.FX.c']) /
                         df.loc[tt, 'USDX.FX.c'])

rv_dt = np.zeros((len(dates), 4))
for tt in range(r_returns.shape[0]):
    rv_dt[tt, 0] = (np.log(df.loc[tt + 1, 'SPX.GI.h'] / df.loc[tt + 1, 'SPX.GI.l'])) ** 2 * 0.3606
    rv_dt[tt, 1] = (np.log(df.loc[tt + 1, 'CL.NYM.h'] / df.loc[tt + 1, 'CL.NYM.l'])) ** 2 * 0.3606
    rv_dt[tt, 2] = (np.log(df.loc[tt + 1, 'TY.CBT.h'] / df.loc[tt + 1, 'TY.CBT.l'])) ** 2 * 0.3606
    rv_dt[tt, 3] = (np.log(df.loc[tt + 1, 'USDX.FX.h'] / df.loc[tt + 1, 'USDX.FX.l'])) ** 2 * 0.3606

int_real = np.delete(r_returns, [2, 5, 8, 11], axis=1)
pd.DataFrame(int_real).to_csv(r"figures\figure2\int_real.csv",index=False)
# endregion

# region 2. Figure 3
asset_ = r_returns[0:0+84, [0, 1]]
BEMD_decom_asset0 = np.c_[asset_,BEMD_fun(asset_)[0],BEMD_fun(asset_)[1]]
BVMD_decom_asset0 = np.c_[asset_,BVMD_fun(asset_)[0],BVMD_fun(asset_)[1]]
SVD_decom_asset0 = np.c_[asset_,SVD_fun(asset_)[0],SVD_fun(asset_)[1]]
ISVD_decom_asset0 = np.c_[asset_,ISVD_fun(asset_)[0],ISVD_fun(asset_)[1]]
pd.DataFrame(BEMD_decom_asset0).to_csv(r"figures\figure3\decom_asset0_BEMD.csv",
                                    index=False)
pd.DataFrame(BVMD_decom_asset0).to_csv(r"figures\figure3\decom_asset0_BVMD.csv",
                                    index=False)
pd.DataFrame(SVD_decom_asset0).to_csv(r"figures\figure3\decom_asset0_SVD.csv",
                                    index=False)
pd.DataFrame(ISVD_decom_asset0).to_csv(r"figures\figure3\decom_asset0_ISVD.csv",
                                    index=False)
# endregion

# region 3. Figure B1-B3
asset_ = r_returns[(-1-84):-1, [3, 4]]
BEMD_decom_asset0 = np.c_[asset_,BEMD_fun(asset_)[0],BEMD_fun(asset_)[1]]
BVMD_decom_asset0 = np.c_[asset_,BVMD_fun(asset_)[0],BVMD_fun(asset_)[1]]
SVD_decom_asset0 = np.c_[asset_,SVD_fun(asset_)[0],SVD_fun(asset_)[1]]
ISVD_decom_asset0 = np.c_[asset_,ISVD_fun(asset_)[0],ISVD_fun(asset_)[1]]
pd.DataFrame(BEMD_decom_asset0).to_csv(r"figures\figureB1\decom_asset1_BEMD.csv",
                                    index=False)
pd.DataFrame(BVMD_decom_asset0).to_csv(r"figures\figureB1\decom_asset1_BVMD.csv",
                                    index=False)
pd.DataFrame(SVD_decom_asset0).to_csv(r"figures\figureB1\decom_asset1_SVD.csv",
                                    index=False)
pd.DataFrame(ISVD_decom_asset0).to_csv(r"figures\figureB1\decom_asset1_ISVD.csv",
                                    index=False)


asset_ = r_returns[(-1-84):-1, [6, 7]]
BEMD_decom_asset0 = np.c_[asset_,BEMD_fun(asset_)[0],BEMD_fun(asset_)[1]]
BVMD_decom_asset0 = np.c_[asset_,BVMD_fun(asset_)[0],BVMD_fun(asset_)[1]]
SVD_decom_asset0 = np.c_[asset_,SVD_fun(asset_)[0],SVD_fun(asset_)[1]]
ISVD_decom_asset0 = np.c_[asset_,ISVD_fun(asset_)[0],ISVD_fun(asset_)[1]]
pd.DataFrame(BEMD_decom_asset0).to_csv(r"figures\figureB2\decom_asset2_BEMD.csv",
                                    index=False)
pd.DataFrame(BVMD_decom_asset0).to_csv(r"figures\figureB2\decom_asset2_BVMD.csv",
                                    index=False)
pd.DataFrame(SVD_decom_asset0).to_csv(r"figures\figureB2\decom_asset2_SVD.csv",
                                    index=False)
pd.DataFrame(ISVD_decom_asset0).to_csv(r"figures\figureB2\decom_asset2_ISVD.csv",
                                    index=False)


asset_ = r_returns[0:0+84, [9, 10]]
BEMD_decom_asset0 = np.c_[asset_,BEMD_fun(asset_)[0],BEMD_fun(asset_)[1]]
BVMD_decom_asset0 = np.c_[asset_,BVMD_fun(asset_)[0],BVMD_fun(asset_)[1]]
SVD_decom_asset0 = np.c_[asset_,SVD_fun(asset_)[0],SVD_fun(asset_)[1]]
ISVD_decom_asset0 = np.c_[asset_,ISVD_fun(asset_)[0],ISVD_fun(asset_)[1]]
pd.DataFrame(BEMD_decom_asset0).to_csv(r"figures\figureB3\decom_asset3_BEMD.csv",
                                    index=False)
pd.DataFrame(BVMD_decom_asset0).to_csv(r"figures\figureB3\decom_asset3_BVMD.csv",
                                    index=False)
pd.DataFrame(SVD_decom_asset0).to_csv(r"figures\figureB3\decom_asset3_SVD.csv",
                                    index=False)
pd.DataFrame(ISVD_decom_asset0).to_csv(r"figures\figureB3\decom_asset3_ISVD.csv",
                                    index=False)

# endregion

# region 4. Figure 4
asset_ = r_returns[:, [3, 4]]
raw_asset0_period = int_period_fun(asset_)

BEMD_asset0_period1 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,2], BEMD_fun(asset_)[1,:,2]])
BEMD_asset0_period2 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,1], BEMD_fun(asset_)[1,:,1]])
BEMD_asset0_period3 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,0], BEMD_fun(asset_)[1,:,0]])
BEMD_asset0_period = np.c_[BEMD_asset0_period1,BEMD_asset0_period2,BEMD_asset0_period3]

BVMD_asset0_period1 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,2], BVMD_fun(asset_)[1,:,2]])
BVMD_asset0_period2 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,1], BVMD_fun(asset_)[1,:,1]])
BVMD_asset0_period3 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,0], BVMD_fun(asset_)[1,:,0]])
BVMD_asset0_period = np.c_[BVMD_asset0_period1,BVMD_asset0_period2,BVMD_asset0_period3]

SVD_asset0_period1 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,0], SVD_fun(asset_)[1,:,0]])
SVD_asset0_period2 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,1], SVD_fun(asset_)[1,:,1]])
SVD_asset0_period3 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,2], SVD_fun(asset_)[1,:,2]])
SVD_asset0_period = np.c_[SVD_asset0_period1,SVD_asset0_period2,SVD_asset0_period3]

ISVD_asset0_period1 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,0], ISVD_fun(asset_)[1,:,0]])
ISVD_asset0_period2 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,1], ISVD_fun(asset_)[1,:,1]])
ISVD_asset0_period3 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,2], ISVD_fun(asset_)[1,:,2]])
ISVD_asset0_period = np.c_[ISVD_asset0_period1,ISVD_asset0_period2,ISVD_asset0_period3]

pd.DataFrame(raw_asset0_period).to_csv(r"figures\figure4\raw_asset0_period.csv",
                                    index=False)
pd.DataFrame(BEMD_asset0_period).to_csv(r"figures\figure4\BEMD_asset0_period.csv",
                                    index=False)
pd.DataFrame(BVMD_asset0_period).to_csv(r"figures\figure4\BVMD_asset0_period.csv",
                                    index=False)
pd.DataFrame(SVD_asset0_period).to_csv(r"figures\figure4\SVD_asset0_period.csv",
                                    index=False)
pd.DataFrame(ISVD_asset0_period).to_csv(r"figures\figure4\ISVD_asset0_period.csv",
                                    index=False)

# endregion

# region 5. Figure B4-B6
asset_ = r_returns[:, [0, 1]]
raw_asset0_period = int_period_fun(asset_)

BEMD_asset0_period1 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,2], BEMD_fun(asset_)[1,:,2]])
BEMD_asset0_period2 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,1], BEMD_fun(asset_)[1,:,1]])
BEMD_asset0_period3 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,0], BEMD_fun(asset_)[1,:,0]])
BEMD_asset0_period = np.c_[BEMD_asset0_period1,BEMD_asset0_period2,BEMD_asset0_period3]

BVMD_asset0_period1 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,2], BVMD_fun(asset_)[1,:,2]])
BVMD_asset0_period2 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,1], BVMD_fun(asset_)[1,:,1]])
BVMD_asset0_period3 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,0], BVMD_fun(asset_)[1,:,0]])
BVMD_asset0_period = np.c_[BVMD_asset0_period1,BVMD_asset0_period2,BVMD_asset0_period3]

SVD_asset0_period1 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,0], SVD_fun(asset_)[1,:,0]])
SVD_asset0_period2 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,1], SVD_fun(asset_)[1,:,1]])
SVD_asset0_period3 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,2], SVD_fun(asset_)[1,:,2]])
SVD_asset0_period = np.c_[SVD_asset0_period1,SVD_asset0_period2,SVD_asset0_period3]

ISVD_asset0_period1 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,0], ISVD_fun(asset_)[1,:,0]])
ISVD_asset0_period2 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,1], ISVD_fun(asset_)[1,:,1]])
ISVD_asset0_period3 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,2], ISVD_fun(asset_)[1,:,2]])
ISVD_asset0_period = np.c_[ISVD_asset0_period1,ISVD_asset0_period2,ISVD_asset0_period3]

pd.DataFrame(raw_asset0_period).to_csv(r"figures\figureB4\raw_asset0_period.csv",
                                    index=False)
pd.DataFrame(BEMD_asset0_period).to_csv(r"figures\figureB4\BEMD_asset0_period.csv",
                                    index=False)
pd.DataFrame(BVMD_asset0_period).to_csv(r"figures\figureB4\BVMD_asset0_period.csv",
                                    index=False)
pd.DataFrame(SVD_asset0_period).to_csv(r"figures\figureB4\SVD_asset0_period.csv",
                                    index=False)
pd.DataFrame(ISVD_asset0_period).to_csv(r"figures\figureB4\ISVD_asset0_period.csv",
                                    index=False)


asset_ = r_returns[:, [6, 7]]
raw_asset0_period = int_period_fun(asset_)

BEMD_asset0_period1 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,2], BEMD_fun(asset_)[1,:,2]])
BEMD_asset0_period2 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,1], BEMD_fun(asset_)[1,:,1]])
BEMD_asset0_period3 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,0], BEMD_fun(asset_)[1,:,0]])
BEMD_asset0_period = np.c_[BEMD_asset0_period1,BEMD_asset0_period2,BEMD_asset0_period3]

BVMD_asset0_period1 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,2], BVMD_fun(asset_)[1,:,2]])
BVMD_asset0_period2 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,1], BVMD_fun(asset_)[1,:,1]])
BVMD_asset0_period3 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,0], BVMD_fun(asset_)[1,:,0]])
BVMD_asset0_period = np.c_[BVMD_asset0_period1,BVMD_asset0_period2,BVMD_asset0_period3]

SVD_asset0_period1 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,0], SVD_fun(asset_)[1,:,0]])
SVD_asset0_period2 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,1], SVD_fun(asset_)[1,:,1]])
SVD_asset0_period3 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,2], SVD_fun(asset_)[1,:,2]])
SVD_asset0_period = np.c_[SVD_asset0_period1,SVD_asset0_period2,SVD_asset0_period3]

ISVD_asset0_period1 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,0], ISVD_fun(asset_)[1,:,0]])
ISVD_asset0_period2 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,1], ISVD_fun(asset_)[1,:,1]])
ISVD_asset0_period3 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,2], ISVD_fun(asset_)[1,:,2]])
ISVD_asset0_period = np.c_[ISVD_asset0_period1,ISVD_asset0_period2,ISVD_asset0_period3]

pd.DataFrame(raw_asset0_period).to_csv(r"figures\figureB5\raw_asset0_period.csv",
                                    index=False)
pd.DataFrame(BEMD_asset0_period).to_csv(r"figures\figureB5\BEMD_asset0_period.csv",
                                    index=False)
pd.DataFrame(BVMD_asset0_period).to_csv(r"figures\figureB5\BVMD_asset0_period.csv",
                                    index=False)
pd.DataFrame(SVD_asset0_period).to_csv(r"figures\figureB5\SVD_asset0_period.csv",
                                    index=False)
pd.DataFrame(ISVD_asset0_period).to_csv(r"figures\figureB5\ISVD_asset0_period.csv",
                                    index=False)


asset_ = r_returns[:, [9, 10]]
raw_asset0_period = int_period_fun(asset_)

BEMD_asset0_period1 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,2], BEMD_fun(asset_)[1,:,2]])
BEMD_asset0_period2 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,1], BEMD_fun(asset_)[1,:,1]])
BEMD_asset0_period3 = int_period_fun(np.c_[BEMD_fun(asset_)[0,:,0], BEMD_fun(asset_)[1,:,0]])
BEMD_asset0_period = np.c_[BEMD_asset0_period1,BEMD_asset0_period2,BEMD_asset0_period3]

BVMD_asset0_period1 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,2], BVMD_fun(asset_)[1,:,2]])
BVMD_asset0_period2 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,1], BVMD_fun(asset_)[1,:,1]])
BVMD_asset0_period3 = int_period_fun(np.c_[BVMD_fun(asset_)[0,:,0], BVMD_fun(asset_)[1,:,0]])
BVMD_asset0_period = np.c_[BVMD_asset0_period1,BVMD_asset0_period2,BVMD_asset0_period3]

SVD_asset0_period1 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,0], SVD_fun(asset_)[1,:,0]])
SVD_asset0_period2 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,1], SVD_fun(asset_)[1,:,1]])
SVD_asset0_period3 = int_period_fun(np.c_[SVD_fun(asset_)[0,:,2], SVD_fun(asset_)[1,:,2]])
SVD_asset0_period = np.c_[SVD_asset0_period1,SVD_asset0_period2,SVD_asset0_period3]

ISVD_asset0_period1 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,0], ISVD_fun(asset_)[1,:,0]])
ISVD_asset0_period2 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,1], ISVD_fun(asset_)[1,:,1]])
ISVD_asset0_period3 = int_period_fun(np.c_[ISVD_fun(asset_)[0,:,2], ISVD_fun(asset_)[1,:,2]])
ISVD_asset0_period = np.c_[ISVD_asset0_period1,ISVD_asset0_period2,ISVD_asset0_period3]

pd.DataFrame(raw_asset0_period).to_csv(r"figures\figureB6\raw_asset0_period.csv",
                                    index=False)
pd.DataFrame(BEMD_asset0_period).to_csv(r"figures\figureB6\BEMD_asset0_period.csv",
                                    index=False)
pd.DataFrame(BVMD_asset0_period).to_csv(r"figures\figureB6\BVMD_asset0_period.csv",
                                    index=False)
pd.DataFrame(SVD_asset0_period).to_csv(r"figures\figureB6\SVD_asset0_period.csv",
                                    index=False)
pd.DataFrame(ISVD_asset0_period).to_csv(r"figures\figureB6\ISVD_asset0_period.csv",
                                    index=False)

# endregion

# region 6. h_ahead=1, win_len=72, p_lag=4, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 72
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f1_p4_w72_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p4_w72_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w72_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w72_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w72_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p4_w72_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 7. h_ahead=1, win_len=84, p_lag=4, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f1_p4_w84_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p4_w84_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w84_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w84_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w84_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p4_w84_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 8. h_ahead=1, win_len=96, p_lag=4, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 96
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f1_p4_w96_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p4_w96_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w96_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w96_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w96_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p4_w96_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 9. h_ahead=2, win_len=84, p_lag=4, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 2
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f2_p4_w84_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f2_p4_w84_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w84_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w84_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w84_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w84_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w84_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w84_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f2_p4_w84_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f2_p4_w84_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f2_p4_w84_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f2_p4_w84_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 10. h_ahead=3, win_len=84, p_lag=4, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 3
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f3_p4_w84_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f3_p4_w84_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w84_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w84_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w84_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w84_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w84_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w84_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f3_p4_w84_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f3_p4_w84_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f3_p4_w84_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f3_p4_w84_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 11. h_ahead=5, win_len=84, p_lag=4, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 5
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f5_p4_w84_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f5_p4_w84_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w84_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w84_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w84_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w84_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w84_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w84_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f5_p4_w84_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f5_p4_w84_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f5_p4_w84_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f5_p4_w84_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 12. h_ahead=2, win_len=96, p_lag=4, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 96
h_ahead = 2
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f2_p4_w96_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f2_p4_w96_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w96_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w96_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w96_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w96_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w96_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f2_p4_w96_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f2_p4_w96_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f2_p4_w96_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f2_p4_w96_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f2_p4_w96_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 13. h_ahead=3, win_len=96, p_lag=4, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 96
h_ahead = 3
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f3_p4_w96_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f3_p4_w96_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w96_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w96_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w96_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w96_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w96_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f3_p4_w96_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f3_p4_w96_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f3_p4_w96_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f3_p4_w96_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f3_p4_w96_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 14. h_ahead=5, win_len=96, p_lag=4, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 96
h_ahead = 5
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f5_p4_w96_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f5_p4_w96_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w96_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w96_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w96_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w96_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w96_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f5_p4_w96_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f5_p4_w96_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f5_p4_w96_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f5_p4_w96_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f5_p4_w96_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 15. h_ahead=1, win_len=84, p_lag=1, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 1
fore_f1_p1_w84_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p1_w84_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p1_w84_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p1_w84_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p1_w84_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p1_w84_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p1_w84_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p1_w84_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p1_w84_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p1_w84_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p1_w84_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p1_w84_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 15. h_ahead=1, win_len=84, p_lag=2, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 2
fore_f1_p2_w84_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p2_w84_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p2_w84_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p2_w84_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p2_w84_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p2_w84_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p2_w84_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p2_w84_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p2_w84_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p2_w84_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p2_w84_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p2_w84_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 16. h_ahead=1, win_len=84, p_lag=3, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 3
fore_f1_p3_w84_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p3_w84_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p3_w84_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p3_w84_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p3_w84_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p3_w84_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p3_w84_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p3_w84_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p3_w84_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p3_w84_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p3_w84_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p3_w84_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 17. h_ahead=1, win_len=84, p_lag=5, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 5
fore_f1_p5_w84_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p5_w84_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p5_w84_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p5_w84_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p5_w84_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p5_w84_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p5_w84_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p5_w84_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p5_w84_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p5_w84_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p5_w84_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p5_w84_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 18. h_ahead=1, win_len=84, p_lag=6, layer=3
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 6
fore_f1_p6_w84_l3 = np.zeros((11, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p6_w84_l3[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p6_w84_l3[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p6_w84_l3[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p6_w84_l3[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p6_w84_l3[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p6_w84_l3[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p6_w84_l3[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p6_w84_l3[7, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p6_w84_l3[8, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p6_w84_l3[9, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                               VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p6_w84_l3[10, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 19. h_ahead=1, win_len=72, p_lag=4, layer=3, ALL
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 72
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f1_p4_w72_l3_ALL = np.zeros((35, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p4_w72_l3_ALL[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3_ALL[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3_ALL[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3_ALL[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3_ALL[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3_ALL[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w72_l3_ALL[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w72_l3_ALL[7, asset_j, i, :] = (MLP_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                   MLP_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                   MLP_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[8, asset_j, i, :] = (CRSETAR_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                   CRSETAR_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                   CRSETAR_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[9, asset_j, i, :] = (CCRM_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                   CCRM_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                   CCRM_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[10, asset_j, i, :] = (CIAR_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    CIAR_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    CIAR_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[11, asset_j, i, :] = (ACI_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    ACI_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    ACI_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[12, asset_j, i, :] = (TARI_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    TARI_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    TARI_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[13, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w72_l3_ALL[14, asset_j, i, :] = (MLP_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    MLP_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    MLP_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[15, asset_j, i, :] = (CRSETAR_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    CRSETAR_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    CRSETAR_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[16, asset_j, i, :] = (CCRM_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    CCRM_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    CCRM_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[17, asset_j, i, :] = (CIAR_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    CIAR_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    CIAR_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[18, asset_j, i, :] = (ACI_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    ACI_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    ACI_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[19, asset_j, i, :] = (TARI_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    TARI_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    TARI_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[20, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w72_l3_ALL[21, asset_j, i, :] = (MLP_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    MLP_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    MLP_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[22, asset_j, i, :] = (CRSETAR_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    CRSETAR_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    CRSETAR_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[23, asset_j, i, :] = (CCRM_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    CCRM_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    CCRM_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[24, asset_j, i, :] = (CIAR_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    CIAR_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    CIAR_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[25, asset_j, i, :] = (ACI_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    ACI_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    ACI_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[26, asset_j, i, :] = (TARI_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    TARI_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    TARI_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w72_l3_ALL[27, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p4_w72_l3_ALL[28, asset_j, i, :] = (MLP_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    MLP_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    MLP_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w72_l3_ALL[29, asset_j, i, :] = (CRSETAR_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    CRSETAR_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    CRSETAR_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w72_l3_ALL[30, asset_j, i, :] = (CCRM_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    CCRM_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    CCRM_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w72_l3_ALL[31, asset_j, i, :] = (CIAR_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    CIAR_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    CIAR_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w72_l3_ALL[32, asset_j, i, :] = (ACI_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    ACI_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    ACI_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w72_l3_ALL[33, asset_j, i, :] = (TARI_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    TARI_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    TARI_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w72_l3_ALL[34, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 20. h_ahead=1, win_len=96, p_lag=4, layer=3, ALL
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 96
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f1_p4_w96_l3_ALL = np.zeros((35, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p4_w96_l3_ALL[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3_ALL[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3_ALL[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3_ALL[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3_ALL[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3_ALL[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w96_l3_ALL[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w96_l3_ALL[7, asset_j, i, :] = (MLP_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                   MLP_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                   MLP_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[8, asset_j, i, :] = (CRSETAR_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                   CRSETAR_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                   CRSETAR_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[9, asset_j, i, :] = (CCRM_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                   CCRM_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                   CCRM_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[10, asset_j, i, :] = (CIAR_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    CIAR_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    CIAR_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[11, asset_j, i, :] = (ACI_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    ACI_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    ACI_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[12, asset_j, i, :] = (TARI_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    TARI_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    TARI_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[13, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w96_l3_ALL[14, asset_j, i, :] = (MLP_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    MLP_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    MLP_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[15, asset_j, i, :] = (CRSETAR_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    CRSETAR_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    CRSETAR_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[16, asset_j, i, :] = (CCRM_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    CCRM_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    CCRM_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[17, asset_j, i, :] = (CIAR_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    CIAR_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    CIAR_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[18, asset_j, i, :] = (ACI_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    ACI_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    ACI_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[19, asset_j, i, :] = (TARI_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    TARI_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    TARI_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[20, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w96_l3_ALL[21, asset_j, i, :] = (MLP_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    MLP_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    MLP_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[22, asset_j, i, :] = (CRSETAR_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    CRSETAR_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    CRSETAR_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[23, asset_j, i, :] = (CCRM_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    CCRM_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    CCRM_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[24, asset_j, i, :] = (CIAR_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    CIAR_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    CIAR_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[25, asset_j, i, :] = (ACI_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    ACI_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    ACI_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[26, asset_j, i, :] = (TARI_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    TARI_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    TARI_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w96_l3_ALL[27, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p4_w96_l3_ALL[28, asset_j, i, :] = (MLP_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    MLP_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    MLP_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w96_l3_ALL[29, asset_j, i, :] = (CRSETAR_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    CRSETAR_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    CRSETAR_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w96_l3_ALL[30, asset_j, i, :] = (CCRM_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    CCRM_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    CCRM_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w96_l3_ALL[31, asset_j, i, :] = (CIAR_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    CIAR_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    CIAR_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w96_l3_ALL[32, asset_j, i, :] = (ACI_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    ACI_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    ACI_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w96_l3_ALL[33, asset_j, i, :] = (TARI_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    TARI_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    TARI_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w96_l3_ALL[34, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 21. h_ahead=1, win_len=84, p_lag=4, layer=3, ALL
Y = r_returns[:, [0, 1]]
TT = Y.shape[0]
win_len = 84
h_ahead = 1
fore_len = TT - win_len - h_ahead + 1
p_lag = 4
fore_f1_p4_w84_l3_ALL = np.zeros((35, 4, fore_len, 2))  # model x asset x forecast x 2

for i in range(0, fore_len):
    for asset_j in range(4):
        match asset_j:
            case 0:
                asset_ = r_returns[i: (i + win_len), [0, 1]]
            case 1:
                asset_ = r_returns[i: (i + win_len), [3, 4]]
            case 2:
                asset_ = r_returns[i: (i + win_len), [6, 7]]
            case 3:
                asset_ = r_returns[i: (i + win_len), [9, 10]]

        # non-decomposition
        fore_f1_p4_w84_l3_ALL[0, asset_j, i, :] = MLP_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3_ALL[1, asset_j, i, :] = CRSETAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3_ALL[2, asset_j, i, :] = CCRM_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3_ALL[3, asset_j, i, :] = CIAR_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3_ALL[4, asset_j, i, :] = ACI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3_ALL[5, asset_j, i, :] = TARI_fun(asset_, p_lag, h_ahead)
        fore_f1_p4_w84_l3_ALL[6, asset_j, i, :] = VC_ACI_2D_fun(asset_, p_lag, h_ahead)

        # BEMD decomposition
        BEMD_decom = BEMD_fun(asset_)
        BEMD_decom_1 = np.sort(np.c_[BEMD_decom[0, :, 0], BEMD_decom[1, :, 0]], axis=1)
        BEMD_decom_2 = np.sort(np.c_[BEMD_decom[0, :, 1], BEMD_decom[1, :, 1]], axis=1)
        BEMD_decom_3 = np.sort(np.c_[BEMD_decom[0, :, 2], BEMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w84_l3_ALL[7, asset_j, i, :] = (MLP_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                   MLP_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                   MLP_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[8, asset_j, i, :] = (CRSETAR_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                   CRSETAR_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                   CRSETAR_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[9, asset_j, i, :] = (CCRM_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                   CCRM_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                   CCRM_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[10, asset_j, i, :] = (CIAR_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    CIAR_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    CIAR_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[11, asset_j, i, :] = (ACI_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    ACI_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    ACI_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[12, asset_j, i, :] = (TARI_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    TARI_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    TARI_fun(BEMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[13, asset_j, i, :] = (VC_ACI_2D_fun(BEMD_decom_1, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BEMD_decom_2, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BEMD_decom_3, p_lag, h_ahead))

        # BVMD decomposition
        BVMD_decom = BVMD_fun(asset_)
        BVMD_decom_1 = np.sort(np.c_[BVMD_decom[0, :, 0], BVMD_decom[1, :, 0]], axis=1)
        BVMD_decom_2 = np.sort(np.c_[BVMD_decom[0, :, 1], BVMD_decom[1, :, 1]], axis=1)
        BVMD_decom_3 = np.sort(np.c_[BVMD_decom[0, :, 2], BVMD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w84_l3_ALL[14, asset_j, i, :] = (MLP_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    MLP_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    MLP_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[15, asset_j, i, :] = (CRSETAR_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    CRSETAR_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    CRSETAR_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[16, asset_j, i, :] = (CCRM_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    CCRM_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    CCRM_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[17, asset_j, i, :] = (CIAR_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    CIAR_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    CIAR_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[18, asset_j, i, :] = (ACI_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    ACI_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    ACI_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[19, asset_j, i, :] = (TARI_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    TARI_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    TARI_fun(BVMD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[20, asset_j, i, :] = (VC_ACI_2D_fun(BVMD_decom_1, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BVMD_decom_2, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(BVMD_decom_3, p_lag, h_ahead))

        # SVD decomposition
        SVD_decom = SVD_fun(asset_)
        SVD_decom_1 = np.sort(np.c_[SVD_decom[0, :, 0], SVD_decom[1, :, 0]], axis=1)
        SVD_decom_2 = np.sort(np.c_[SVD_decom[0, :, 1], SVD_decom[1, :, 1]], axis=1)
        SVD_decom_3 = np.sort(np.c_[SVD_decom[0, :, 2], SVD_decom[1, :, 2]], axis=1)
        fore_f1_p4_w84_l3_ALL[21, asset_j, i, :] = (MLP_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    MLP_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    MLP_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[22, asset_j, i, :] = (CRSETAR_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    CRSETAR_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    CRSETAR_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[23, asset_j, i, :] = (CCRM_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    CCRM_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    CCRM_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[24, asset_j, i, :] = (CIAR_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    CIAR_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    CIAR_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[25, asset_j, i, :] = (ACI_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    ACI_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    ACI_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[26, asset_j, i, :] = (TARI_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    TARI_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    TARI_fun(SVD_decom_3, p_lag, h_ahead))
        fore_f1_p4_w84_l3_ALL[27, asset_j, i, :] = (VC_ACI_2D_fun(SVD_decom_1, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(SVD_decom_2, p_lag, h_ahead) +
                                                    VC_ACI_2D_fun(SVD_decom_3, p_lag, h_ahead))

        # ISVD decomposition
        ISVD_decom = ISVD_fun(asset_, K=np.array([[5, 1], [1, 1]]))
        ISVD_decom_1 = (np.c_[ISVD_decom[0, :, 0], ISVD_decom[1, :, 0]])
        ISVD_decom_2 = (np.c_[ISVD_decom[0, :, 1], ISVD_decom[1, :, 1]])
        ISVD_decom_3 = (np.c_[ISVD_decom[0, :, 2], ISVD_decom[1, :, 2]])

        weight_cov1 = np.array([int_cov_fun(ISVD_decom_1, ISVD_decom_1, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_2, ISVD_decom_2, 5, 1, 1, 1),
                                int_cov_fun(ISVD_decom_3, ISVD_decom_3, 5, 1, 1, 1)])
        weight_cov = weight_cov1 / sum(weight_cov1)

        fore_f1_p4_w84_l3_ALL[28, asset_j, i, :] = (MLP_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    MLP_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    MLP_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w84_l3_ALL[29, asset_j, i, :] = (CRSETAR_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    CRSETAR_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    CRSETAR_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w84_l3_ALL[30, asset_j, i, :] = (CCRM_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    CCRM_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    CCRM_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w84_l3_ALL[31, asset_j, i, :] = (CIAR_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    CIAR_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    CIAR_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w84_l3_ALL[32, asset_j, i, :] = (ACI_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    ACI_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    ACI_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w84_l3_ALL[33, asset_j, i, :] = (TARI_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    TARI_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    TARI_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
        fore_f1_p4_w84_l3_ALL[34, asset_j, i, :] = (VC_ACI_2D_fun(ISVD_decom_1, p_lag, h_ahead) * weight_cov[0] +
                                                    VC_ACI_2D_fun(ISVD_decom_2, p_lag, h_ahead) * weight_cov[1] +
                                                    VC_ACI_2D_fun(ISVD_decom_3, p_lag, h_ahead) * weight_cov[2])
    print(i)
# endregion#

# region 22. Table 1
loss_f1_p4_w84_l3=np.r_[np.c_[int_loss_fun(fore_f1_p4_w84_l3, r_returns, 0),
    int_loss_fun(fore_f1_p4_w84_l3, r_returns, 1)],
    np.c_[int_loss_fun(fore_f1_p4_w84_l3, r_returns, 2),
    int_loss_fun(fore_f1_p4_w84_l3, r_returns, 3)]]
pd.DataFrame(loss_f1_p4_w84_l3).to_csv(r"tables\table1\Table 1.csv",
                                    index=False)
# endregion

# region 23. Figure 5 & Table C1

loss_csv_colnames = ['MDE_mr', 'MDE_mr_MCS', 'MDE_mr_DM',
                     'MDE_lr', 'MDE_lr_MCS', 'MDE_lr_DM']

asset_num = 0
loss_f1_p4_w84_l3_ALL = int_loss_fun(fore_f1_p4_w84_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w84_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"figures\figure5\All_asset0_w84.csv",
                                               index=False)
asset_num = 1
loss_f1_p4_w84_l3_ALL = int_loss_fun(fore_f1_p4_w84_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w84_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"figures\figure5\All_asset1_w84.csv",
                                               index=False)
asset_num = 2
loss_f1_p4_w84_l3_ALL = int_loss_fun(fore_f1_p4_w84_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w84_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"figures\figure5\All_asset2_w84.csv",
                                               index=False)
asset_num = 3
loss_f1_p4_w84_l3_ALL = int_loss_fun(fore_f1_p4_w84_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w84_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"figures\figure5\All_asset3_w84.csv",
                                               index=False)
# endregion

# region 24. Table C2-C3

loss_csv_colnames = ['MDE_mr', 'MDE_mr_MCS', 'MDE_mr_DM',
                     'MDE_lr', 'MDE_lr_MCS', 'MDE_lr_DM']
asset_num = 0
loss_f1_p4_w72_l3_ALL = int_loss_fun(fore_f1_p4_w72_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w72_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"tables\tableC2\All_asset0_w72.csv",
                                               index=False)
asset_num = 1
loss_f1_p4_w72_l3_ALL = int_loss_fun(fore_f1_p4_w72_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w72_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"tables\tableC2\All_asset1_w72.csv",
                                               index=False)
asset_num = 2
loss_f1_p4_w72_l3_ALL = int_loss_fun(fore_f1_p4_w72_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w72_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"tables\tableC2\All_asset2_w72.csv",
                                               index=False)
asset_num = 3
loss_f1_p4_w72_l3_ALL = int_loss_fun(fore_f1_p4_w72_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w72_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"tables\tableC2\All_asset3_w72.csv",
                                               index=False)

asset_num = 0
loss_f1_p4_w96_l3_ALL = int_loss_fun(fore_f1_p4_w96_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w96_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"tables\tableC3\All_asset0_w96.csv",
                                               index=False)
asset_num = 1
loss_f1_p4_w96_l3_ALL = int_loss_fun(fore_f1_p4_w96_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w96_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"tables\tableC3\All_asset1_w96.csv",
                                               index=False)
asset_num = 2
loss_f1_p4_w96_l3_ALL = int_loss_fun(fore_f1_p4_w96_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w96_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"tables\tableC3\All_asset2_w96.csv",
                                               index=False)
asset_num = 3
loss_f1_p4_w96_l3_ALL = int_loss_fun(fore_f1_p4_w96_l3_ALL, r_returns, asset_num)
pd.DataFrame(loss_f1_p4_w96_l3_ALL,
             columns=loss_csv_colnames).to_csv(r"tables\tableC3\All_asset3_w96.csv",
                                               index=False)

# endregion

# region 25. Figure 6
win_len = 84
asset_num = 0
cumloss_asset0 = np.c_[int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 0, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 1, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 2, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 3, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 4, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 5, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 6, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 7, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 8, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 9, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 10, win_len)]
pd.DataFrame(cumloss_asset0).to_csv(r"figures\figure6\cumloss_asset0.csv",
                                    index=False)

asset_num = 1
cumloss_asset1 = np.c_[int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 0, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 1, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 2, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 3, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 4, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 5, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 6, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 7, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 8, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 9, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 10, win_len)]
pd.DataFrame(cumloss_asset1).to_csv(r"figures\figure6\cumloss_asset1.csv",
                                    index=False)

asset_num = 2
cumloss_asset2 = np.c_[int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 0, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 1, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 2, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 3, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 4, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 5, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 6, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 7, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 8, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 9, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 10, win_len)]
pd.DataFrame(cumloss_asset2).to_csv(r"figures\figure6\cumloss_asset2.csv",
                                    index=False)

asset_num = 3
cumloss_asset3 = np.c_[int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 0, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 1, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 2, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 3, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 4, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 5, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 6, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 7, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 8, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 9, win_len),
int_cumloss_fun(fore_f1_p4_w84_l3, r_returns, asset_num, 10, win_len)]
pd.DataFrame(cumloss_asset3).to_csv(r"figures\figure6\cumloss_asset3.csv",
                                    index=False)
# endregion

# region 26. Table 2

cvr_fore_f1_p4_w84_l3=np.c_[int_coverage_fun(fore_f1_p4_w84_l3,r_returns,0)*100,
int_coverage_fun(fore_f1_p4_w84_l3, r_returns, 1) * 100,
int_coverage_fun(fore_f1_p4_w84_l3, r_returns, 2) * 100,
int_coverage_fun(fore_f1_p4_w84_l3, r_returns, 3) * 100]

pd.DataFrame(cvr_fore_f1_p4_w84_l3).to_csv(r"tables\table2\cvr.csv",
                                              index=False)
# endregion

# region 27. Figure 7

cvr_csv_colnames = ['Coverage', 'Efficiency']

asset_num = 0
cvr_f1_p4_w84_l3_ALL = int_coverage_fun(fore_f1_p4_w84_l3_ALL,r_returns,asset_num)
pd.DataFrame(cvr_f1_p4_w84_l3_ALL,
             columns=cvr_csv_colnames).to_csv(r"figures\figure7\cvr_all_asset0.csv",
                                              index=False)

asset_num = 1
cvr_f1_p4_w84_l3_ALL = int_coverage_fun(fore_f1_p4_w84_l3_ALL,r_returns,asset_num)
pd.DataFrame(cvr_f1_p4_w84_l3_ALL,
             columns=cvr_csv_colnames).to_csv(r"figures\figure7\cvr_all_asset1.csv",
                                              index=False)

asset_num = 2
cvr_f1_p4_w84_l3_ALL = int_coverage_fun(fore_f1_p4_w84_l3_ALL,r_returns,asset_num)
pd.DataFrame(cvr_f1_p4_w84_l3_ALL,
             columns=cvr_csv_colnames).to_csv(r"figures\figure7\cvr_all_asset2.csv",
                                              index=False)

asset_num = 3
cvr_f1_p4_w84_l3_ALL = int_coverage_fun(fore_f1_p4_w84_l3_ALL,r_returns,asset_num)
pd.DataFrame(cvr_f1_p4_w84_l3_ALL,
             columns=cvr_csv_colnames).to_csv(r"figures\figure7\cvr_all_asset3.csv",
                                              index=False)
# endregion

# region 28. Table 3 & 4
loss_f1_p4_w72_l3=np.r_[np.c_[int_loss_fun(fore_f1_p4_w72_l3, r_returns, 0),
    int_loss_fun(fore_f1_p4_w72_l3, r_returns, 1)],
    np.c_[int_loss_fun(fore_f1_p4_w72_l3, r_returns, 2),
    int_loss_fun(fore_f1_p4_w72_l3, r_returns, 3)]]
pd.DataFrame(loss_f1_p4_w72_l3).to_csv(r"tables\table3\Table 3.csv",
                                    index=False)

loss_f1_p4_w96_l3=np.r_[np.c_[int_loss_fun(fore_f1_p4_w96_l3, r_returns, 0),
    int_loss_fun(fore_f1_p4_w96_l3, r_returns, 1)],
    np.c_[int_loss_fun(fore_f1_p4_w96_l3, r_returns, 2),
    int_loss_fun(fore_f1_p4_w96_l3, r_returns, 3)]]
pd.DataFrame(loss_f1_p4_w96_l3).to_csv(r"tables\table4\Table 4.csv",
                                    index=False)

# endregion

# region 29. Figure 8
multih_csv_colnames = [f'h{i}' for i in [2,3,5]]
multih_csv_rownames = ['MLP','CRSETAR','CCRM','CIAR','ACI','TARI','VC-ACI',
                       'BEMD-VC-ACI','BVMD-VC-ACI','SVD-VC-ACI','ISVD-VC-ACI']

asset_num = 0
loss_f2_p4_w84_l3 = int_loss_fun(fore_f2_p4_w84_l3, r_returns, asset_num,2)
loss_f3_p4_w84_l3 = int_loss_fun(fore_f3_p4_w84_l3, r_returns, asset_num,3)
loss_f5_p4_w84_l3 = int_loss_fun(fore_f5_p4_w84_l3, r_returns, asset_num,5)

multih_p4_w84_l3_crloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f3_p4_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f5_p4_w84_l3[:,0])) + 1]
pd.DataFrame(multih_p4_w84_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w84_l3_crloss_asset0.csv")

multih_p4_w84_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f3_p4_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f5_p4_w84_l3[:,3])) + 1]
pd.DataFrame(multih_p4_w84_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w84_l3_lrloss_asset0.csv")

asset_num = 1
loss_f2_p4_w84_l3 = int_loss_fun(fore_f2_p4_w84_l3, r_returns, asset_num,2)
loss_f3_p4_w84_l3 = int_loss_fun(fore_f3_p4_w84_l3, r_returns, asset_num,3)
loss_f4_p4_w84_l3 = int_loss_fun(fore_f5_p4_w84_l3, r_returns, asset_num,5)

multih_p4_w84_l3_crloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f3_p4_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f5_p4_w84_l3[:,0])) + 1]
pd.DataFrame(multih_p4_w84_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w84_l3_crloss_asset1.csv")

multih_p4_w84_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f3_p4_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f5_p4_w84_l3[:,3])) + 1]
pd.DataFrame(multih_p4_w84_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w84_l3_lrloss_asset1.csv")

asset_num = 2
loss_f2_p4_w84_l3 = int_loss_fun(fore_f2_p4_w84_l3, r_returns, asset_num,2)
loss_f3_p4_w84_l3 = int_loss_fun(fore_f3_p4_w84_l3, r_returns, asset_num,3)
loss_f5_p4_w84_l3 = int_loss_fun(fore_f5_p4_w84_l3, r_returns, asset_num,5)

multih_p4_w84_l3_crloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f3_p4_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f5_p4_w84_l3[:,0])) + 1]
pd.DataFrame(multih_p4_w84_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w84_l3_crloss_asset2.csv")

multih_p4_w60_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f3_p4_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f5_p4_w84_l3[:,3])) + 1]
pd.DataFrame(multih_p4_w60_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w84_l3_lrloss_asset2.csv")

asset_num = 3
loss_f2_p4_w84_l3 = int_loss_fun(fore_f2_p4_w84_l3, r_returns, asset_num,2)
loss_f3_p4_w84_l3 = int_loss_fun(fore_f3_p4_w84_l3, r_returns, asset_num,3)
loss_f5_p4_w84_l3 = int_loss_fun(fore_f5_p4_w84_l3, r_returns, asset_num,5)

multih_p4_w84_l3_crloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f3_p4_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f5_p4_w84_l3[:,0])) + 1]
pd.DataFrame(multih_p4_w84_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w84_l3_crloss_asset3.csv")

multih_p4_w84_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f3_p4_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f5_p4_w84_l3[:,3])) + 1]
pd.DataFrame(multih_p4_w84_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w84_l3_lrloss_asset3.csv")

multih_csv_colnames = ['h2','h3','h5']
multih_csv_rownames = ['MLP','CRSETAR','CCRM','CIAR','ACI','TARI','VC-ACI',
                       'BEMD-VC-ACI','BVMD-VC-ACI','SVD-VC-ACI','ISVD-VC-ACI']

asset_num = 0
loss_f2_p4_w96_l3 = int_loss_fun(fore_f2_p4_w96_l3, r_returns, asset_num,2)
loss_f3_p4_w96_l3 = int_loss_fun(fore_f3_p4_w96_l3, r_returns, asset_num,3)
loss_f5_p4_w96_l3 = int_loss_fun(fore_f5_p4_w96_l3, r_returns, asset_num,5)

multih_p4_w96_l3_crloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w96_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f3_p4_w96_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f5_p4_w96_l3[:,0])) + 1]
pd.DataFrame(multih_p4_w96_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w96_l3_crloss_asset0.csv")

multih_p4_w96_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w96_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f3_p4_w96_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f5_p4_w96_l3[:,3])) + 1]
pd.DataFrame(multih_p4_w96_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w96_l3_lrloss_asset0.csv")

asset_num = 1
loss_f2_p4_w96_l3 = int_loss_fun(fore_f2_p4_w96_l3, r_returns, asset_num,2)
loss_f3_p4_w96_l3 = int_loss_fun(fore_f3_p4_w96_l3, r_returns, asset_num,3)
loss_f5_p4_w96_l3 = int_loss_fun(fore_f5_p4_w96_l3, r_returns, asset_num,5)

multih_p4_w96_l3_crloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w96_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f3_p4_w96_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f5_p4_w96_l3[:,0])) + 1]
pd.DataFrame(multih_p4_w96_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w96_l3_crloss_asset1.csv")

multih_p4_w96_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w96_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f3_p4_w96_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f5_p4_w96_l3[:,3])) + 1]
pd.DataFrame(multih_p4_w96_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w96_l3_lrloss_asset1.csv")

asset_num = 2
loss_f2_p4_w96_l3 = int_loss_fun(fore_f2_p4_w96_l3, r_returns, asset_num,2)
loss_f3_p4_w96_l3 = int_loss_fun(fore_f3_p4_w96_l3, r_returns, asset_num,3)
loss_f5_p4_w96_l3 = int_loss_fun(fore_f5_p4_w96_l3, r_returns, asset_num,5)

multih_p4_w96_l3_crloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w96_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f3_p4_w96_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f5_p4_w96_l3[:,0])) + 1]
pd.DataFrame(multih_p4_w96_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w96_l3_crloss_asset2.csv")

multih_p4_w96_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w96_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f3_p4_w96_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f5_p4_w96_l3[:,3])) + 1]
pd.DataFrame(multih_p4_w96_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w96_l3_lrloss_asset2.csv")

asset_num = 3
loss_f2_p4_w96_l3 = int_loss_fun(fore_f2_p4_w96_l3, r_returns, asset_num,2)
loss_f3_p4_w96_l3 = int_loss_fun(fore_f3_p4_w96_l3, r_returns, asset_num,3)
loss_f5_p4_w96_l3 = int_loss_fun(fore_f5_p4_w96_l3, r_returns, asset_num,5)

multih_p4_w96_l3_crloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w96_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f3_p4_w96_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f5_p4_w96_l3[:,0])) + 1]
pd.DataFrame(multih_p4_w96_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w96_l3_crloss_asset3.csv")

multih_p4_w96_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f2_p4_w96_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f3_p4_w96_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f5_p4_w96_l3[:,3])) + 1]
pd.DataFrame(multih_p4_w96_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure8\multih_p4_w96_l3_lrloss_asset3.csv")

# endregion

# region 30. Figure 9
multih_csv_colnames = [f'p{i}' for i in [1,2,3,5]]
multih_csv_rownames = ['MLP','CRSETAR','CCRM','CIAR','ACI','TARI','VC-ACI',
                       'BEMD-VC-ACI','BVMD-VC-ACI','SVD-VC-ACI','ISVD-VC-ACI']

asset_num = 0
loss_f1_p1_w84_l3 = int_loss_fun(fore_f1_p1_w84_l3, r_returns, asset_num)
loss_f1_p5_w84_l3 = int_loss_fun(fore_f1_p5_w84_l3, r_returns, asset_num)
loss_f1_p6_w84_l3 = int_loss_fun(fore_f1_p6_w84_l3, r_returns, asset_num)
loss_f1_p2_w84_l3 = int_loss_fun(fore_f1_p2_w84_l3, r_returns, asset_num)
loss_f1_p3_w84_l3 = int_loss_fun(fore_f1_p3_w84_l3, r_returns, asset_num)

multip_f1_w84_l3_crloss=np.c_[np.argsort(np.argsort(loss_f1_p1_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p2_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p3_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p5_w84_l3[:,0])) + 1]
pd.DataFrame(multip_f1_w84_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure9\multip_f1_w84_l3_crloss_asset0.csv")

multip_f1_w84_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f1_p1_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p2_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p3_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p5_w84_l3[:,3])) + 1]
pd.DataFrame(multip_f1_w84_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure9\multip_f1_w84_l3_lrloss_asset0.csv")


asset_num = 1
loss_f1_p1_w84_l3 = int_loss_fun(fore_f1_p1_w84_l3, r_returns, asset_num)
loss_f1_p5_w84_l3 = int_loss_fun(fore_f1_p5_w84_l3, r_returns, asset_num)
loss_f1_p6_w84_l3 = int_loss_fun(fore_f1_p6_w84_l3, r_returns, asset_num)
loss_f1_p2_w84_l3 = int_loss_fun(fore_f1_p2_w84_l3, r_returns, asset_num)
loss_f1_p3_w84_l3 = int_loss_fun(fore_f1_p3_w84_l3, r_returns, asset_num)

multip_f1_w84_l3_crloss=np.c_[np.argsort(np.argsort(loss_f1_p1_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p2_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p3_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p5_w84_l3[:,0])) + 1]
pd.DataFrame(multip_f1_w84_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure9\multip_f1_w84_l3_crloss_asset1.csv")

multip_f1_w84_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f1_p1_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p2_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p3_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p5_w84_l3[:,3])) + 1]
pd.DataFrame(multip_f1_w84_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure9\multip_f1_w84_l3_lrloss_asset1.csv")

asset_num = 2
loss_f1_p1_w84_l3 = int_loss_fun(fore_f1_p1_w84_l3, r_returns, asset_num)
loss_f1_p5_w84_l3 = int_loss_fun(fore_f1_p5_w84_l3, r_returns, asset_num)
loss_f1_p6_w84_l3 = int_loss_fun(fore_f1_p6_w84_l3, r_returns, asset_num)
loss_f1_p2_w84_l3 = int_loss_fun(fore_f1_p2_w84_l3, r_returns, asset_num)
loss_f1_p3_w84_l3 = int_loss_fun(fore_f1_p3_w84_l3, r_returns, asset_num)

multip_f1_w84_l3_crloss=np.c_[np.argsort(np.argsort(loss_f1_p1_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p2_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p3_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p5_w84_l3[:,0])) + 1]
pd.DataFrame(multip_f1_w84_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure9\multip_f1_w84_l3_crloss_asset2.csv")

multip_f1_w84_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f1_p1_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p2_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p3_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p5_w84_l3[:,3])) + 1]
pd.DataFrame(multip_f1_w84_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure9\multip_f1_w84_l3_lrloss_asset2.csv")

asset_num = 3
loss_f1_p1_w84_l3 = int_loss_fun(fore_f1_p1_w84_l3, r_returns, asset_num)
loss_f1_p5_w84_l3 = int_loss_fun(fore_f1_p5_w84_l3, r_returns, asset_num)
loss_f1_p6_w84_l3 = int_loss_fun(fore_f1_p6_w84_l3, r_returns, asset_num)
loss_f1_p2_w84_l3 = int_loss_fun(fore_f1_p2_w84_l3, r_returns, asset_num)
loss_f1_p3_w84_l3 = int_loss_fun(fore_f1_p3_w84_l3, r_returns, asset_num)

multip_f1_w84_l3_crloss=np.c_[np.argsort(np.argsort(loss_f1_p1_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p2_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p3_w84_l3[:,0])) + 1,
np.argsort(np.argsort(loss_f1_p5_w84_l3[:,0])) + 1]
pd.DataFrame(multip_f1_w84_l3_crloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure9\multip_f1_w84_l3_crloss_asset3.csv")

multip_f1_w84_l3_lrloss=np.c_[np.argsort(np.argsort(loss_f1_p1_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p2_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p3_w84_l3[:,3])) + 1,
np.argsort(np.argsort(loss_f1_p5_w84_l3[:,3])) + 1]
pd.DataFrame(multip_f1_w84_l3_lrloss,
             columns=multih_csv_colnames,
             index=multih_csv_rownames).to_csv(r"figures\figure9\multip_f1_w84_l3_lrloss_asset3.csv")

# endregion


