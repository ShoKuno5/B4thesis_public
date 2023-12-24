# %% [markdown]
# #### 各パッケージのインストール，データ，hyperparametersの読み込み

# %%

#必要なパッケージのインポート

import numpy as np

import sys

import matplotlib
import matplotlib.pyplot as plt

import reservoirpy as rpy

from scipy.integrate import solve_ivp
import pandas as pd
from reservoirpy.observables import nrmse, rsquare

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from reservoirpy.datasets import to_forecasting

rpy.verbosity(0)

import os

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.datasets import mackey_glass

import joblib


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
def plot_generation(X_gen, X_t, nb_generations, warming_out=None, warming_inputs=None, seed_timesteps=0):
    plt.figure(figsize=(15, 5))
    if warming_out is not None:
        plt.plot(np.vstack([warming_out, X_gen]), label="Generated timeseries")
    else:
        plt.plot(X_gen, label="Generated timeseries")

    plt.plot(np.arange(nb_generations) + seed_timesteps, X_t, linestyle="--", label="Real timeseries")

    # `warming_inputs`のサイズを検証して調整します。
    if warming_inputs is not None and len(warming_inputs) > seed_timesteps:
        # `seed_timesteps`に合わせてサイズを調整
        warming_inputs = warming_inputs[:seed_timesteps]
        plt.plot(warming_inputs, linestyle="--", label="Warmup")

    plt.plot(np.arange(nb_generations) + seed_timesteps, np.abs(X_t - X_gen), label="Absolute deviation")

    if seed_timesteps > 0:
        plt.fill_between([0, seed_timesteps], *plt.ylim(), facecolor='lightgray', alpha=0.5, label="Warmup period")

    plt.legend()
    plt.show()


# %%
N = 10000
iss = 0.05149642735978288
lr = 0.270051165245735
ridge = 1.5838543746634222e-05
seed = 3
sr = 0.4650867210512849
forecast = 1

train_len = 60000
start_time = 0
test_length = 30000
nb_generations = 1000

seed_timesteps = test_length 

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

# コマンドラインからファイル名とstudy名を受け取る
if len(sys.argv) > 1:
    i = sys.argv[1]
else:
    print("位相シフトの時間数を24時間表記でコマンドライン引数として入力してください。")
    sys.exit(1)


dir_name_1 = f"SE_result_random_det"
os.makedirs(dir_name_1, exist_ok=True)

dir_name_2 = f"SE_val_random_det"
os.makedirs(dir_name_2, exist_ok=True)


# CSVファイルにデータを保存
filename_with_force = f"data/rossler_data_with_shifted_force_{i}.csv"

# CSVファイルを読み込む
data_loaded_with_force = pd.read_csv(filename_with_force)

# CSVから値を抽出してNumpy配列に格納
X = data_loaded_with_force[['x', 'y', 'P_shifted']].values

X.shape

n,m = X.shape

N = 10000
iss = 0.05149642735978288
lr = 0.270051165245735
ridge = 1.5838543746634222e-05
seed = 3
sr = 0.4650867210512849
forecast = 1

train_len = 60000
start_time = 0
test_length = 30000
nb_generations = 1000

seed_timesteps = test_length 

nb_generations = len(X) - (train_len + test_length)

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
y = y.reshape(1, -1) 

for t in range(nb_generations):  
    y[:, 2:3] = X_evolve[t, 2:3] #外力にあたる[:, 2:3]に実測値を代入する．
    y = esn(y) #ESNで1回=0.1ステップ先を予測する．
    X_gen[t, :] = y #配列に記録していく
            
X_t = X_evolve[: nb_generations]


# X_genを適当なファイル名で保存する場合
file_name_1 = f"{dir_name_1}/CmaEs_shift{i}.csv"

# X_tをCSVファイルに書き出す
np.savetxt(file_name_1, X_t, delimiter=',')

# X_tを適当なファイル名で保存する場合
file_name_2 = f"{dir_name_2}/CmaEs_shift{i}.csv"

# X_tをCSVファイルに書き出す
np.savetxt(file_name_2, X_gen, delimiter=',')


