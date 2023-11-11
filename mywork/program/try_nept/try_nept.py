# %%
import neptune
import os


# %%
api_token = os.getenv('NEPTUNE_API_TOKEN')


# %%
if api_token is not None:
    print("APIトークンは正しく設定されています。")
else:
    print("APIトークンが設定されていません。")


# %%
import neptune

run = neptune.init_run(
    project="shokuno55/B4thesis",
    api_token=api_token,
)  # your credentials

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].append(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()



