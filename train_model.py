import numpy as np
from matplotlib.ticker import ScalarFormatter
import pickle

import stopping
from config import Config
from task_model import TaskDurationModel
import matplotlib.pyplot as plt
import os

# === Load preprocessed data ===
Config.load()
X = np.load("processed/X_scaled.npy")
y = np.load("processed/y.npy")

# === Initialize model ===
model = TaskDurationModel(input_size=X.shape[1])
print(f"Training on {X.shape[0]} samples, input dim = {X.shape[1]}")

Config.load()  # Load json file with variables

epochs = Config.epochs
batch_size = Config.batch_size
learning_rate = Config.LEARNING_RATE

early_stopper = stopping.stopping(patience=50, min_delta=0.0001)        #placeholder of 50 while testing

precision_history = []
loss_history = []

# === Training loop ===
for epoch in range(epochs):
    total_loss = 0
    predictions = []
    actuals = []

    # Shuffle data each epoch
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Train in batches
    for i in range(0, len(X), batch_size):
        xb = X_shuffled[i:i + batch_size]
        yb = y_shuffled[i:i + batch_size]

        for xi, yi in zip(xb, yb):
            loss = model.train_on_vector(X_input=xi.reshape(1, -1), true_duration=yi, learning_rate=learning_rate)
            total_loss += loss

            # === Predict for precision ===
            y_pred = model.forward(xi.reshape(1, -1))[0][0]
            predictions.append(y_pred)
            actuals.append(yi)

    # === Calculate metrics after the epoch ===
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    avg_loss = total_loss / len(X)
    loss_history.append(avg_loss)

    mape = np.mean(np.abs((actuals - predictions) / actuals))
    precision_percentage = (1 - mape) * 100
    precision_history.append(precision_percentage)

    early_stopper(avg_loss)
    if early_stopper.early_stop:
        print(f"Early stopping at epoch {epoch}. No improvement in {early_stopper.patience} epochs.")
        break

    # === Print loss and precision
    print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Precision = {precision_percentage:.2f}%")


# === Save trained model ===
model.save(f"models/size_{Config.iteration_number}/model_{Config.model_number}/model_weights.npz")
print("Model training complete and saved to models/model_weights.npz")

output_dir = f"models/size_{Config.iteration_number}/pickle_loss"
os.makedirs(output_dir, exist_ok=True)
# === Plot precision curve ===
fig = plt.figure(figsize=(10, 5))
plt.plot(precision_history, label="Precision (%)", color='green')
plt.title("Model Precision Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Precision (%)")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()

# Save figure and viewer like before
fig_filename = os.path.join(output_dir, f"precision_plot_{Config.model_number}.fig.pickle")
with open(fig_filename, "wb") as f:
    pickle.dump(fig, f)

# Replace 'loss_plot' with 'precision_plot' in viewer and batch script generation...

viewer_code = f"""
import pickle
import matplotlib.pyplot as plt
import os
import matplotlib

backends = ['TkAgg', 'Qt5Agg', 'WXAgg']
for backend in backends:
    try:
        matplotlib.use(backend)
        break
    except:
        continue

script_dir = os.path.dirname(os.path.abspath(__file__))
fig_path = os.path.join(script_dir, "loss_plot_{Config.model_number}.fig.pickle")

if os.path.exists(fig_path):
    with open(fig_path, "rb") as f:
        fig = pickle.load(f)
    plt.show()
else:
    print(f"File not found: {{fig_path}}")
"""

viewer_py_path = os.path.join(output_dir, f"view_loss_plot_{Config.model_number}.py")
with open(viewer_py_path, "w") as f:
    f.write(viewer_code.strip())

batch_code = f"""
@echo off
start "" "pythonw.exe" "%~dp0../pickle_loss/view_loss_plot_{Config.model_number}.py
exit
"""
os.makedirs(f"models/size_{Config.iteration_number}/batches", exist_ok=True)
batch_path = os.path.join(f"models/size_{Config.iteration_number}/batches",
                          f"loss_plot_batch_{Config.model_number}.bat")
with open(batch_path, "w") as f:
    f.write(batch_code)

# viewer_py_path = os.path.join(output_dir, f"view_loss_plot_shell_{Config.model_number}.py")
# with open(viewer_py_path, "w") as f:
#     f.write(viewer_code.strip())

# === Create Ubuntu-compatible shell script to show the precision plot
bash_script_path = os.path.join(
    f"models/size_{Config.iteration_number}/batches",
    f"show_precision_plot_{Config.model_number}.sh"
)

bash_script_code = f"""#!/bin/bash
cd "$(dirname "$0")/../pickle_loss"
python3 view_loss_plot_{Config.model_number}.py
"""

with open(bash_script_path, "w") as f:
    f.write(bash_script_code.strip())

# Make script executable
os.chmod(bash_script_path, 0o755)
