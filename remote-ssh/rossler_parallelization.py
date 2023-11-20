# %% [markdown]
# # Local parallelization of Hyper Parameter Search
# 

# %% [markdown]
# In this notebook, we will tackle the same problem as before but with a focus on parallelization using multiple CPU cores.
# 
# Thanks to the joblib library, we will define a new `optimize_study` function and implement the necessary code for parallel execution. This parallelization can significantly speed up the hyperparameter search process.
# 
# Additionally, we will provide an example to determine the optimal number of processes to use based on your local computer's capabilities.

# %% [markdown]
# ### Step 1 : Prepare your data 

# %% [markdown]
# The first 3 steps are the same than in the 1st tutorial that explains how to conduct an hyperparameter search with optuna. You can directly jump to the 4th step if you are already familiar with it.

# %%
import numpy as np
import reservoirpy as rpy
import pandas as pd

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare

# CSVファイルにデータを保存
filename_with_force = 'rossler_data_with_force1.3.1.csv'

# CSVファイルを読み込む
data_loaded_with_force = pd.read_csv(filename_with_force)

# CSVから値を抽出してNumpy配列に格納
X = data_loaded_with_force[['x', 'y', 'P']].values

train_len = 10000

X_train = X[:train_len]
y_train = X[1 : train_len + 1]

X_test = X[train_len : -1]
y_test = X[train_len + 1:]

dataset = ((X_train, y_train), (X_test, y_test))


# %% [markdown]
# ### Step 2: Define fixed parameters for the hyper parameter search

# %%
import os
import time
import joblib
import optuna
import datetime
import matplotlib.pyplot as plt

from optuna.storages import JournalStorage, JournalFileStorage
from optuna.visualization import plot_optimization_history, plot_param_importances


optuna.logging.set_verbosity(optuna.logging.ERROR)
rpy.verbosity(0)


# %%
# Trial Fixed hyper-parameters
nb_seeds = 3

# %%
def objective(trial):
    # Record objective values for each trial
    losses = []

    # Trial generated parameters (with log scale)
    N = 5000  # Nの値は固定
    sr = trial.suggest_float('sr', 1e-2, 10, log = True)
    lr = trial.suggest_float('lr', 1e-3, 1, log = True)
    iss = trial.suggest_float('iss', 0, 1)
    ridge = trial.suggest_float('ridge', 1e-9, 1e-2, log = True)

    for seed in range(nb_seeds):
        reservoir = Reservoir(N,
                              sr=sr,
                              lr=lr,
                              input_scaling=iss,
                              seed=seed)
        
        readout = Ridge(ridge=ridge)

        model = reservoir >> readout

        # Train and test your model
        predictions = model.fit(X_train, y_train).run(X_test)

        # Compute the desired metrics
        loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))

        losses.append(loss)

    return np.mean(losses)


# %% [markdown]
# ### Step 4: Create a Study Optimization function
import datetime

# 現在の日時を取得
current_time = datetime.datetime.now()

# 日時を文字列フォーマットに変換 (例: '2023-11-20_12-30-00')
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

# ファイル名に日時を含める
study_name = f"optuna_tutorial_{formatted_time}.txt"
print(study_name)

nb_trials = 100

sampler = optuna.samplers.RandomSampler() 

log_name = f"optuna-journal_{study_name}.log"

storage = JournalStorage(JournalFileStorage(log_name))

def optimize_study(n_trials):
    study = optuna.create_study(
        study_name=study_name, #ここを毎回変える必要があるみたい
        direction='minimize',
        storage=storage,
        sampler=optuna.samplers.RandomSampler(),
        load_if_exists=True
    )

    for i in range(n_trials):
        trial = study.ask()
        study.tell(trial, objective(trial))
        
nb_cpus = os.cpu_count()
print(f"Number of available CPUs : {nb_cpus}")

n_process = 60
times = []

print("")
print(f"Optization with n_process = {n_process}")
start = time.time()

n_trials_per_process = nb_trials // n_process
args_list = [n_trials_per_process for i in range(n_process)]

joblib.Parallel(n_jobs=n_process)(joblib.delayed(optimize_study)(args) for args in args_list)

end = time.time()
times.append(end - start)
print(f"Done in {str(datetime.timedelta(seconds=end-start))}")

study = optuna.load_study(study_name=study_name, storage=storage)

from optuna.visualization import plot_optimization_history, plot_param_importances

# 最適化履歴のプロット
plot_optimization_history(study)

# パラメータの重要度のプロット
plot_param_importances(study)

# 試行結果をDataFrameに変換
df = study.trials_dataframe()

# DataFrameをCSVファイルに保存
df.to_csv("study_results.csv")
