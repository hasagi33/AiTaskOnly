from itertools import combinations_with_replacement
import numpy as np
from task_model import TaskDurationModel

# === Load trained model ===
model = TaskDurationModel()
model.load("models/model_weights.npz")

levels = ["Junior", "Mid", "Senior", "Tech Lead"]
experience_map = {
    "Junior": 1.5,
    "Mid": 3.25,
    "Senior": 5.75,
    "Tech Lead": 7.75
}

# === Priority impact factor ===
priority_factor_map = {
    "Low": 1.1,
    "Medium": 1.0,
    "High": 0.9,
    "Critical": 0.85
}

# === Fixed task configuration ===
priorities_to_test = ["Low", "Medium", "High", "Critical"]

for priority in priorities_to_test:
    print(f"\n--- Testing Priority: {priority} ---")

    task_base = {
        "task_type": "Feature Dev",
        "complexity": "High",
        "assignee_level": "Mid",
        "tech_stack": "Python",
        "task_priority": priority,
        "story_points": 8,
        "num_dependencies": 2,
        "estimated_hours": 24,
        "sprint_day": 4,
        "created_hour": 13,
        "remote_work": True,
        "meetings_today": 1,
        "blocker_flag": False
    }

    best_team = None
    best_duration = float("inf")

    for team_size in range(2, 6):
        for combo in combinations_with_replacement(levels, team_size):
            counts = {lvl: combo.count(lvl) for lvl in levels}

            # === Hierarchy rules ===
            if counts["Tech Lead"] > 1:
                continue
            if counts["Tech Lead"] > 0 and team_size < 3:
                continue
            if counts["Senior"] > 2:
                continue
            if counts["Junior"] > 2 and task_base["complexity"] == "High":
                continue
            if counts["Senior"] > 0 and task_base["complexity"] == "Low":
                continue

            avg_exp = np.mean([experience_map[lvl] for lvl in combo])

            task_input = task_base.copy()
            task_input.update({
                "team_size": team_size,
                "avg_experience": round(avg_exp, 8),
                "juniors": counts["Junior"],
                "mediors": counts["Mid"],
                "seniors": counts["Senior"],
                "tech_leads": counts["Tech Lead"]
            })

            prediction = model.predict(task_input)

            if prediction < best_duration:
                best_duration = prediction
                best_team = (counts.copy(), team_size, prediction)

    # === Output best configuration per priority ===
    days = int(best_team[2])
    hours = round((best_team[2] - days) * 24,8)

    print("Optimal Team Configuration:")
    print(f"Team Composition: {best_team[0]}")
    print(f"Team Size: {best_team[1]}")
    print(f"Predicted Duration: {days} days and {hours} hours")
