from task_model import TaskDurationModel
from task_input import encode_task
import numpy as np
import pandas as pd
from config import Config
import json
# === Load trained model ===
model = TaskDurationModel()
Config.load()

dataset_size=int(Config.dataset_rows)

for i in range(1):
    for j in range(1):
        with open(f"models/size_{i+1}/model_{j+1}/config.json", "r") as file:
            text = json.load(file)

            Config.current_loss=text["current_loss"]
            Config.current_layers=text["current_layers"]

            Config.save()
        Config.save()
        Config.load()

        model = TaskDurationModel()

        model.load(f"models/size_{i+1}/model_{j+1}/model_weights.npz")

        # config_of_model=
        # === Load test dataset ===
        test_df = pd.read_csv(f"dataset.csv").sample(dataset_size)

        # === Predict and Evaluate ===
        predictions = []
        actuals = []

        for _, row in test_df.iterrows():
            task_data = row.to_dict()
            actual_duration = task_data.pop("predicted_duration_days")
            pred_duration = model.predict(task_data, scaler_path=f"processed/scaler.json")
            predictions.append(pred_duration)
            actuals.append(actual_duration)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # === Calculate precision (Mean Absolute Percentage Error) ===
        mape = np.mean(np.abs((actuals - predictions) / actuals))
        precision_percentage = (1 - mape) * 100


        print(f"Model Precision: {round(precision_percentage,8)}%")

        # print(f"Predicted Duration for example task for size {dataset_size}: {days} days and {hours} hours")
