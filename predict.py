from task_model import TaskDurationModel
from task_input import encode_task
import numpy as np
import pandas as pd
from config import Config
# === Load trained model ===
model = TaskDurationModel()
Config.load()

dataset_size=100
for i in range(Config.iteration_number-1):

    for j in range(3):
        model.load(f"models/size_{i+1}/model_{j+1}/model_weights.npz")
        # === Load test dataset ===
        test_df = pd.read_csv(f"models/size_{i+1}/dataset.csv").sample(dataset_size)

        # === Predict and Evaluate ===
        predictions = []
        actuals = []

        for _, row in test_df.iterrows():
            task_data = row.to_dict()
            actual_duration = task_data.pop("predicted_duration_days")
            pred_duration = model.predict(task_data, scaler_path=f"models/size_{i+1}/processed/scaler.json")

            predictions.append(pred_duration)
            actuals.append(actual_duration)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # === Calculate precision (Mean Absolute Percentage Error) ===
        mape = np.mean(np.abs((actuals - predictions) / actuals))
        precision_percentage = (1 - mape) * 100



        # === Example task prediction ===
        test_task = {
            "task_type": "Feature Dev",
            "complexity": "High",
            "assignee_level": "Mid",
            "tech_stack": "Python",
            "task_priority": "Critical",
            "story_points": 8,
            "team_size": 4,
            "num_dependencies": 2,
            "estimated_hours": 24,
            "sprint_day": 4,
            "created_hour": 13,
            "remote_work": True,
            "meetings_today": 1,
            "blocker_flag": False,
            "avg_experience": 3.5,
            "juniors": 4,
            "mediors": 0,
            "seniors": 0,
            "tech_leads": 0
        }

        prediction = model.predict(test_task, scaler_path=f"models/size_{i+1}/processed/scaler.json")
        days = int(prediction)
        hours = round((prediction - days) * 24,8)
        while(hours >=24):
            days+=1
            hours-=24

        print(f"Model Precision: {round(precision_percentage,8)}%")
        print(f"Predicted Duration for example task for size {dataset_size}: {days} days and {hours} hours")
    dataset_size*=10
