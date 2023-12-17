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

shifts = {}  # 4日ごとの位相シフトを格納する辞書

def phase_shift(t):
    # 4日ごとの区間を識別
    period = int(t / (4*2*np.pi))

    # 新しい位相シフトが必要かどうかをチェック
    if period not in shifts:
        random_shift_hour = np.random.choice(np.arange(-12, 12, 1)) #シフト範囲は -12 から +12
        shift_value = (random_shift_hour / 24) * 2 * np.pi
        shifts[period] = shift_value

    # 現在の位相シフトを返す
    return shifts[period]


def rossler_system_with_shifted_force(t, state, a, b, c, A):
    x, y, z = state
    dxdt = -y - z + A * np.sin(t + phase_shift(t))  # X項に外力P(t)を加える（位相シフト付き）
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]




A = 2.0  # 外力の振幅を設定します。この値を変更することで、外力の強さを変えられます。

# パラメータ
a = 0.2
b = 0.2
c = 5.7

# 初期条件
initial_state = [1.0, 1.0, 1.0]

# 時間の設定
t_span = [0, 10000]  # 開始時刻と終了時刻
t_eval = np.linspace(t_span[0], t_span[1], 100000)  # 評価する時間点


# 微分方程式の数値解を求める
solution_with_shifted_force = solve_ivp(
    rossler_system_with_shifted_force, t_span, initial_state,
    args=(a, b, c, A), t_eval=t_eval, max_step=0.01
)

# 外力P(t)の計算（位相シフト付き）
P_shifted = A * np.sin(solution_with_shifted_force.t + np.array([phase_shift(ti) for ti in solution_with_shifted_force.t]))

# DataFrameの作成
data_frame_with_shifted_force = pd.DataFrame({
    'time': solution_with_shifted_force.t,
    'x': solution_with_shifted_force.y[0],
    'y': solution_with_shifted_force.y[1],
    'z': solution_with_shifted_force.y[2],
    'P_shifted': P_shifted  # 位相シフトされた外力P(t)の列を追加
})



dir_name = f"data"
os.makedirs(dir_name, exist_ok=True)

filename_with_force = f"{dir_name}/rossler_data_with_random_shifted_force.csv"

# CSVファイルにデータを保存
data_frame_with_shifted_force.to_csv(filename_with_force, index=False)

# CSVファイルを読み込む
data_loaded_with_force = pd.read_csv(filename_with_force)

# CSVから値を抽出してNumpy配列に格納
X = data_loaded_with_force[['x', 'y', 'P_shifted']].values



sample = 0
plot_length = 4000

# XYZの三次元グラフをプロット
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(data_loaded_with_force['x'][sample: sample+plot_length], data_loaded_with_force['y'][sample: sample+plot_length], data_loaded_with_force['z'][sample: sample+plot_length])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Rössler Attractor')
plt.show()

# 時間ごとのx, y, zそれぞれのグラフ
fig, axs = plt.subplots(4, 1, figsize=(15, 8), sharex=True)
axs[0].plot(data_loaded_with_force['time'][sample: sample+plot_length], data_loaded_with_force['x'][sample: sample+plot_length], label='x')
axs[0].set_ylabel('x')
axs[0].legend(loc='upper right')

axs[1].plot(data_loaded_with_force['time'][sample: sample+plot_length], data_loaded_with_force['y'][sample: sample+plot_length], label='y', color='orange')
axs[1].set_ylabel('y')
axs[1].legend(loc='upper right')

axs[2].plot(data_loaded_with_force['time'][sample: sample+plot_length], data_loaded_with_force['z'][sample: sample+plot_length], label='z', color='green')
axs[2].set_ylabel('z')
axs[2].set_xlabel('Time')
axs[2].legend(loc='upper right')

axs[3].plot(data_loaded_with_force['time'][sample: sample+plot_length], data_loaded_with_force['P_shifted'][sample: sample+plot_length], label='P_shifted', color='green')
axs[3].set_ylabel('P_shifted')
axs[3].set_xlabel('Time')
axs[3].legend(loc='upper right')

plt.suptitle('Time Evolution of the Rössler System')
plt.show()



print(shifts)



import numpy as np
import matplotlib.pyplot as plt

# shifts 辞書からキーと値を取得し、ラジアンから時間（時）に変換
periods = list(shifts.keys())
phase_shifts = [shift / (2 * np.pi) * 24 for shift in shifts.values()]

# グラフの作成
plt.figure(figsize=(20, 6))
plt.plot(periods, phase_shifts, marker='o')
plt.title('Phase Shifts Over Periods (in hours)')
plt.xlabel('Period (4-day intervals)')
plt.ylabel('Phase Shift (hours)')
plt.grid(True)
plt.show()



