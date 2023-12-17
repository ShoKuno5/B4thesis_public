import numpy as np
import reservoirpy as rpy
import pandas as pd

from reservoirpy.nodes import Reservoir, Ridge
from reservoirpy.observables import nrmse, rsquare
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

import datetime

study_name = f"RS_N10000_6030"

log_name = f"optuna-journal_{study_name}.log"

storage = JournalStorage(JournalFileStorage(log_name))

study = optuna.load_study(study_name=study_name, storage=storage)

from optuna.visualization import plot_optimization_history, plot_param_importances
import plotly


# 新しいディレクトリを作成
dir_name = f"study_results_{study_name}"
os.makedirs(dir_name, exist_ok=True)

# 試行結果をDataFrameに変換してCSVファイルに保存
df = study.trials_dataframe()
df.to_csv(os.path.join(dir_name, "study_results.csv"))

# 最適化履歴のプロットを生成してPNGファイルに保存
fig1 = plot_optimization_history(study)
plotly.io.write_image(fig1, os.path.join(dir_name, "optimization_history.png"))

# パラメータの重要度のプロットを生成してPNGファイルに保存
fig2 = plot_param_importances(study)
plotly.io.write_image(fig2, os.path.join(dir_name, "param_importances.png"))

best_trial = study.best_trial

# 最良の目的関数の値
best_value = best_trial.value

# 最良のパラメータ
best_params = best_trial.params

print(f"Best trial ID: {best_trial.number}")
print(f"Best value: {best_value}")
print(f"Best parameters: {best_params}")
