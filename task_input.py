import numpy as np

# Categorical fields
TASK_TYPES = [
    "Code Review", "Bug Fix", "Feature Dev", "Testing", "Documentation",
    "Design Review", "Client Meeting", "Deployment", "Refactoring"
]

COMPLEXITIES = ["Low", "Medium", "High"]
ASSIGNEE_LEVELS = ["Junior", "Mid", "Senior", "Tech Lead"]
TECH_STACKS = ["Python", "Java", "JavaScript", "C++", "Go", "Other"]
PRIORITY_LEVELS = ["Low", "Medium", "High", "Critical"]

def encode_task(task):
    """
    Converts a detailed task dictionary into a numerical feature vector.
    Includes role breakdown and experience for team optimization.
    Total vector length = 45.
    """
    input_vector = []

    # One-hot encoding
    input_vector += [1 if task["task_type"] == t else 0 for t in TASK_TYPES]
    input_vector += [1 if task["complexity"] == c else 0 for c in COMPLEXITIES]
    input_vector += [1 if task["assignee_level"] == a else 0 for a in ASSIGNEE_LEVELS]
    input_vector += [1 if task["tech_stack"] == tech else 0 for tech in TECH_STACKS]
    input_vector += [1 if task["task_priority"] == p else 0 for p in PRIORITY_LEVELS]

    # Numeric input fields
    input_vector.append(task.get("story_points", 0))
    input_vector.append(task.get("team_size", 1))
    input_vector.append(task.get("num_dependencies", 0))
    input_vector.append(task.get("estimated_hours", 0))
    input_vector.append(task.get("sprint_day", 1))
    input_vector.append(task.get("created_hour", 9))
    input_vector.append(1 if task.get("remote_work", False) else 0)
    input_vector.append(task.get("meetings_today", 0))
    input_vector.append(1 if task.get("blocker_flag", False) else 0)

        # Required team capability fields (5)
    input_vector.append(task.get("avg_experience", 0))
    input_vector.append(task.get("juniors", 0))
    input_vector.append(task.get("mediors", 0))
    input_vector.append(task.get("seniors", 0))
    input_vector.append(task.get("tech_leads", 0))

    for key in ["avg_experience", "juniors", "mediors", "seniors", "tech_leads"]:
        try:
            input_vector.append(float(task.get(key, 0)))
        except:
            input_vector.append(0.0)

    return np.array([input_vector])

