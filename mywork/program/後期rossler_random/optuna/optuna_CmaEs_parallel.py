import numpy as np
import reservoirpy as rpy
import pandas as pd
from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
import os
import sys
import uuid
import optuna
from optuna.samplers import CmaEsSampler, TPESampler

# コマンドラインからファイル名とstudy名を受け取る
if len(sys.argv) > 2:
    filename = sys.argv[1]
    study_name = sys.argv[2]
else:
    print("ファイル名とstudy名をコマンドライン引数として入力してください。")
    sys.exit(1)


# CSVファイルを読み込む
data_loaded_with_force = pd.read_csv(filename)

# CSVから値を抽出してNumpy配列に格納
X = data_loaded_with_force[['x', 'y', 'P_shifted']].values

print(study_name)

# ディレクトリの作成
results_dir = f"result_{study_name}"
os.makedirs(results_dir, exist_ok=True)
# ディレクトリを移動
os.chdir(results_dir)

train_len = 60000
start_time = 0
test_length = 30000

X_train = X[start_time:start_time+train_len]
y_train = X[start_time+1 :start_time+train_len + 1]

X_test = X[start_time+train_len : start_time+train_len + test_length]
y_test = X[start_time+train_len + 1: start_time+train_len + test_length + 1]

dataset = ((X_train, y_train), (X_test, y_test))


# %% [markdown]
# ### Step 2: Define fixed parameters for the hyper parameter search

# %%
import time
import joblib
import optuna
import datetime
import matplotlib.pyplot as plt

from optuna.storages import JournalStorage, JournalFileStorage
from optuna.visualization import plot_optimization_history, plot_param_importances


optuna.logging.set_verbosity(optuna.logging.ERROR)
rpy.verbosity(0)

import json

# %%
# Trial Fixed hyper-parameters
nb_seeds = 3


# %%
def objective(trial):
    # Record objective values for each trial
    losses = []

    # Trial generated parameters (with log scale)
    N = 10000  # Nの値は固定
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
        
        readout = Ridge(ridge=ridge, name=f"Ridge-{uuid.uuid4()}")

        model = reservoir >> readout
        
        # Train and test your model
        predictions = model.fit(X_train, y_train).run(X_test)

        # Compute the desired metrics
        loss = nrmse(y_test, predictions, norm_value=np.ptp(X_train))

        losses.append(loss)

    return np.mean(losses)




nb_trials = 3000

log_name = f"optuna-journal_{study_name}.log"

storage = JournalStorage(JournalFileStorage(log_name))

print(f"url:{storage}")

def optimize_study(n_trials):
    study = optuna.create_study(
        study_name=study_name, #ここを毎回変える必要があるみたい
        direction='minimize',
        storage=storage,
        sampler=optuna.samplers.CmaEsSampler(),
        pruner= optuna.pruners.SuccessiveHalvingPruner,
        load_if_exists=True
    )

    for i in range(n_trials):
        trial = study.ask()
        study.tell(trial, objective(trial))
        
        intermediate_value = objective(trial)
        trial.report(intermediate_value, i)
        
        # プルーニングのチェック
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
nb_cpus = os.cpu_count()
print(f"Number of available CPUs : {nb_cpus}")

n_process = 20
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

study_results = {
    "study_name": study_name,
    "best_params": study.best_params,
    "best_value": study.best_value,
    "trials": len(study.trials)
    # 他に必要な情報があればここに追加
}

# JSONファイルとして出力
with open(f"{results_dir}/study_results.json", "w") as f:
    json.dump(study_results, f, indent=4)

from optuna.visualization import plot_optimization_history, plot_param_importances

