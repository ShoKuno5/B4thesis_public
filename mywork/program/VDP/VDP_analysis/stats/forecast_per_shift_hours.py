# %% [markdown]
# #### 各パッケージのインストール，データ，hyperparametersの読み込み

# %%
#必要なパッケージのインポート

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import reservoirpy as rpy

from scipy.integrate import solve_ivp
import pandas as pd
from reservoirpy.observables import nrmse, rsquare

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os

rpy.verbosity(0)

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass

# just a little tweak to center the plots, nothing to worry about
from IPython.core.display import HTML
HTML("""
<style>
.img-center {
    display: block;
    margin-left: auto;
    margin-right: auto;
    }
.output_png {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
    }
</style>
""")

rpy.set_seed(42)


# %% [markdown]
# ### 7. Generative Modelのうち，外力のデータのみ実データで更新し続ける．
# 
# 期待としては，X, Yの精度も上がるということである．

# %%
from reservoirpy.datasets import to_forecasting


# %%
def reset_esn():
    from reservoirpy.nodes import Reservoir, Ridge

    reservoir = Reservoir(N, 
                      sr=sr, 
                      lr=lr, 
                      input_scaling=iss, 
                      seed=seed)
    readout = Ridge(ridge=ridge)

    return reservoir >> readout


# %%
opt_shift_hour = -7

N = 10000
iss = 0.27485355587624805
lr = 0.5345512189408888
ridge = 0.000655264507023661
seed = 3
sr = 0.33549492278694165
forecast = 1

train_len = 40000
start_time = 0
test_length = 20000
nb_generations = 10000

seed_timesteps = test_length 


# %%
dir_name_1 = f"opt_{opt_shift_hour}_VDP_val"
os.makedirs(dir_name_1, exist_ok=True)

dir_name_2 = f"opt_{opt_shift_hour}_VDP_gen"
os.makedirs(dir_name_2, exist_ok=True)

# %%
for shift_hour in range(-12, 13):

    # CSVファイルにデータを保存
    filename_with_force = f'/home/kuno/my_project/VDP/VDP_analysis/generate_data/data/VDP_{shift_hour}.csv'

    # CSVファイルを読み込む
    data_loaded_with_force = pd.read_csv(filename_with_force)

    # CSVから値を抽出してNumpy配列に格納
    X = data_loaded_with_force[['x', 'y', 'P_shifted']].values

    n,m = X.shape

    esn = reset_esn()

    X_train = X[start_time:start_time+train_len]
    y_train = X[start_time+1 :start_time+train_len + 1]

    X_test = X[start_time+train_len : start_time+train_len + seed_timesteps]
    y_test = X[start_time+train_len + 1: start_time+train_len + seed_timesteps + 1]

    X_evolve = X[start_time+train_len + seed_timesteps:]

    esn = esn.fit(X_train, y_train)

    warming_inputs = X_test

    warming_out = esn.run(warming_inputs, reset=True)  # warmup
    #warming_outはX_test[seed_timesteps]を近似する．

    X_gen = np.zeros((nb_generations, m))
    y = warming_out[-1] 
    print(y.shape) 
    y = y.reshape(1, -1) 
    print(y.shape) #配列の形式は(n, m)の二次元配列にする必要があるので調整した

    for t in range(nb_generations):  
        y[:, 2:3] = X_evolve[t, 2:3] #外力にあたる[:, 2:3]に実測値を代入する．
        y = esn(y) #ESNで1回=0.1ステップ先を予測する．
        X_gen[t, :] = y #配列に記録していく
        
    X_t = X_evolve[: nb_generations]

    # X_tを適当なファイル名で保存する場合
    file_name_1 = f"{dir_name_1}/VDP_{shift_hour}.csv"

    # X_tをCSVファイルに書き出す
    np.savetxt(file_name_1, X_t, delimiter=',')


    # X_genを適当なファイル名で保存する場合
    file_name_2 = f"{dir_name_2}/VDP_{shift_hour}.csv"

    # X_genをCSVファイルに書き出す
    np.savetxt(file_name_2, X_gen, delimiter=',')



