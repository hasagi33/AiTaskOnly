import pandas as pd
import random
import numpy as np
import config
from config import dataset_rows

TASK_TYPES = [
    "Code Review", "Bug Fix", "Feature Dev", "Testing", "Documentation",
    "Design Review", "Client Meeting", "Deployment", "Refactoring"
]
COMPLEXITIES = ["Low", "Medium", "High"]
ASSIGNEE_LEVELS = ["Junior", "Mid", "Senior", "Tech Lead"]
TECH_STACKS = ["Python", "Java", "JavaScript", "C++", "Go", "Other"]
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]

def random_task():
    task_type = random.choice(TASK_TYPES)
    complexity = random.choice(COMPLEXITIES)
    assignee_level = random.choice(ASSIGNEE_LEVELS)
    tech_stack = random.choice(TECH_STACKS)
    task_priority = random.choice(PRIORITY_LEVELS)

    story_points = random.randint(1, 13)
    blockers = random.choices([True, False], weights=[0.25, 0.75])[0]
    meetings = random.randint(0, 3)
    deps = random.randint(0, 3)
    remote = random.choices([True, False], weights=[0.4, 0.6])[0]

    team_size = random.randint(2, 5)
    team_roles = random.choices(ASSIGNEE_LEVELS, weights=[4, 3, 2, 1], k=team_size)
    counts = {role: team_roles.count(role) for role in ASSIGNEE_LEVELS}

    # === Hierarchy rules ===
    if counts["Tech Lead"] > 1:
        return None
    if counts["Tech Lead"] > 0 and team_size < 3:
        return None
    if counts["Senior"] > 2:
        return None
    if counts["Junior"] > 2 and complexity == "High":
        return None
    if counts["Senior"] > 0 and complexity == "Low":
        return None

    exp_map = {
        "Junior": 1.5,
        "Mid": 3.25,
        "Senior": 5.75,
        "Tech Lead": 7.75
    }
    avg_exp = round(np.mean([exp_map[role] for role in team_roles]), 8)

    # Base duration and modifiers
    story_point_factor = 0.4
    base = story_points * story_point_factor
    complexity_multiplier = {"Low": 0.9, "Medium": 1.0, "High": 1.15}[complexity]
    priority_factor = {"Low": 1.1, "Medium": 1.0, "High": 0.9, "Critical": 0.85}[task_priority]

    duration = base * complexity_multiplier
    duration += deps * 0.15
    duration += meetings * 0.1

    team_factor = 1 - (0.03 * counts.get("Senior", 0) + 0.05 * counts.get("Tech Lead", 0))
    duration *= team_factor

    exp_factor = np.clip(1.1 - avg_exp * 0.05, 0.85, 1.1)
    duration *= exp_factor

    if blockers:
        duration *= 1.1
    if remote:
        duration *= 1.05

    duration *= priority_factor
    # duration += random.uniform(-0.1, 0.1)     no random
    duration = round(np.clip(duration, 0.5, 8.0), 8)

    return {
        "task_type": task_type,
        "complexity": complexity,
        "assignee_level": assignee_level,
        "tech_stack": tech_stack,
        "task_priority": task_priority,
        "story_points": story_points,
        "team_size": team_size,
        "num_dependencies": deps,
        "estimated_hours": story_points * 2,
        "sprint_day": random.randint(1, 10),
        "created_hour": random.randint(9, 18),
        "remote_work": remote,
        "meetings_today": meetings,
        "blocker_flag": blockers,
        "avg_experience": avg_exp,
        "juniors": counts.get("Junior", 0),
        "mediors": counts.get("Mid", 0),
        "seniors": counts.get("Senior", 0),
        "tech_leads": counts.get("Tech Lead", 0),
        "predicted_duration_days": duration
    }

# Generate data with hierarchy-respecting logic
data = []

while len(data) < config.dataset_rows:
    task = random_task()
    if task:
        data.append(task)

# Save dataset
df = pd.DataFrame(data)
df.to_csv("realistic_tasks_large.csv", index=False)
print("Saved realistic_tasks_large.csv with", len(data), "rows")