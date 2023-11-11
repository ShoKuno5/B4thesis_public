# %% [markdown]
# # レスラー方程式(外力のある状態)
# 
# レスラー方程式の外力のある場合に関して，$sin$波に位相のシフトがある場合を考える．

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


import os

import optuna
import neptune
import neptune.integrations.optuna as optuna_utils


# %%
filename_with_force = 'rossler_data_with_shifted_force2.1.1.csv'

# CSVファイルを読み込む
data_loaded_with_force = pd.read_csv(filename_with_force)

# CSVから値を抽出してNumpy配列に格納
X = data_loaded_with_force[['x', 'y', 'P_shifted']].values

from reservoirpy.datasets import to_forecasting

train_len = 10000
test_len = 10000

x, y = to_forecasting(X, forecast=1)
X_train, y_train = x[:train_len], y[:train_len]
X_test, y_test = x[train_len:train_len+test_len], y[train_len:train_len+test_len]

dataset = ((X_train, y_train), (X_test, y_test))


# Neptuneの初期化と実験の開始
api_token = os.getenv('NEPTUNE_API_TOKEN')
project_name = 'shokuno55/B4thesis'  # Neptuneプロジェクト名

# This step may vary depending on what you put inside 'dataset'
train_data, validation_data = dataset
X_train, y_train = train_data
X_val, y_val = validation_data

# Optunaの目的関数
def objective(trial):
    # パラメータの提案
    N_value = 500  # Nの値は固定
    sr = trial.suggest_float('sr', 1e-2, 10, log = True)
    lr = trial.suggest_float('lr', 1e-3, 1, log = True)
    iss = trial.suggest_float('iss', 0, 1)
    ridge = trial.suggest_float('ridge', 1e-9, 1e-2, log = True)
    
    losses = []; r2s = [];
    for n in range(3):  # 例としてインスタンスごとに3回試行
        # モデルの構築
        reservoir = Reservoir(N_value, sr=sr, lr=lr, input_scaling=iss, seed=n)
        readout = Ridge(ridge=ridge)
        model = reservoir >> readout

        # モデルの訓練とテスト
        # Train your model and test your model.
        prediction = model.fit(X_train, y_train) \
                           .run(X_test)
        
        loss = nrmse(y_test, prediction, norm_value=np.ptp(X_train))
        r2 = rsquare(y_test, prediction)

        # 評価指標の計算
        loss = nrmse(y_test, prediction, norm_value=np.ptp(X_train))
        r2 = rsquare(y_test, prediction)
        losses.append(loss)
        r2s.append(r2)

    return np.mean(losses)

# %%

# 並列処理のための関数
from multiprocessing import Pool

def optimize(n_trials):
    # Neptuneの実験を開始
    run = neptune.init_run(project=project_name, api_token=api_token)

    # Optunaのスタディを作成し、最適化を実行
    study = optuna.create_study(study_name='example_study', direction='minimize', storage='sqlite:///example.db', load_if_exists=True)
    study.optimize(objective, n_trials=n_trials)

    # 最適なパラメータの取得と保存
    best_params = study.best_params
    run['best_params'] = best_params

    # Neptuneでの実験の終了
    run.stop()


if __name__ == '__main__':
    n_processes = 4  # 同時に実行するプロセスの数
    with Pool(n_processes) as pool:
        pool.map(optimize, range(n_processes))

# %%
