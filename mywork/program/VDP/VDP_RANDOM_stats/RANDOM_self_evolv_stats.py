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

import sys

# コマンドラインからファイル名とstudy名を受け取る
if len(sys.argv) > 1:
    shift_hour = sys.argv[1]
else:
    print("ファイル名とstudy名をコマンドライン引数として入力してください。")
    sys.exit(1)

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

print(f"これから{shift_hour}に対して予測を行います．\n")

dir_name_1 = f"VDP_stats_result"
os.makedirs(dir_name_1, exist_ok=True)

dir_name_2 = f"VDP_stats_val"
os.makedirs(dir_name_2, exist_ok=True)

# %%
# CSVファイルにデータを保存
filename_with_force = f"data/VDP_{shift_hour}.csv"

# CSVファイルを読み込む
data_loaded_with_force = pd.read_csv(filename_with_force)

# CSVから値を抽出してNumpy配列に格納
X = data_loaded_with_force[['x', 'y', 'P_shifted']].values

X.shape

n,m = X.shape

N = 10000
iss = 0.5768591838727728
lr = 0.4304317721067772
ridge = 0.00041298826582042474
seed = 3
sr = 0.866615511675912
forecast = 1

train_len = 40000
start_time = 0
test_length = 20000
nb_generations = 10000

seed_timesteps = test_length 

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
file_name_1 = f"{dir_name_1}/CmaEs_shift{shift_hour}.csv"

# X_tをCSVファイルに書き出す
np.savetxt(file_name_1, X_gen, delimiter=',')

# X_tを適当なファイル名で保存する場合
file_name_2 = f"{dir_name_2}/CmaEs_shift{shift_hour}.csv"

# X_tをCSVファイルに書き出す
np.savetxt(file_name_2, X_t, delimiter=',')


print(f"{shift_hour}に対して予測が完了しました．\n")