import neptune
import optuna
import os

# Neptuneの初期化と実験の開始
api_token = os.getenv('NEPTUNE_API_TOKEN')
project_name = 'shokuno55/B4thesis'  # Neptuneプロジェクト名
run = neptune.init_run(
    project=project_name,
    api_token=api_token,
)

# 最適化する目的関数を定義（Neptuneとの統合を含む）
def objective(trial):
    x = trial.suggest_float('x', -10, 10)
    result = (x - 2) ** 2

    # Neptuneに試行のパラメータと結果を記録（時間の経過に沿って）
    run["parameters/x"].log(x)
    run["results/result"].log(result)

    return result

# Optunaのスタディを作成し、最適化を実行
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 最適なパラメータと結果を出力し、Neptuneでの実験を終了
best_params = study.best_params  # 最適なパラメータ
best_value = study.best_value    # 最適な値
print(f"Best parameters: {best_params}")
print(f"Best value: {best_value}")

# Neptuneに最適なパラメータと結果を記録
run["study/best_params"] = best_params
run["study/best_value"] = best_value

run.stop()
