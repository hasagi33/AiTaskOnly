import numpy as np
from matplotlib.ticker import ScalarFormatter
import pickle

import config
from task_model import TaskDurationModel
import matplotlib.pyplot as plt
import os

# === Load preprocessed data ===
X = np.load("processed/X_scaled.npy")
y = np.load("processed/y.npy")

# === Initialize model ===
model = TaskDurationModel(input_size=X.shape[1])
print(f"Training on {X.shape[0]} samples, input dim = {X.shape[1]}")

# === Training parameters ===
batch_size = config.batch_size
epochs = config.epochs                #epohiranje
learning_rate = config.LEARNING_RATE
loss_history = []

# === Training loop ===
for epoch in range(epochs):
    total_loss = 0

    # Shuffle data each epoch
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Train in batches
    for i in range(0, len(X), batch_size):
        xb = X_shuffled[i:i+batch_size]
        yb = y_shuffled[i:i+batch_size]

        for xi, yi in zip(xb, yb):
            loss = model.train_on_vector(X_input=xi.reshape(1, -1), true_duration=yi, learning_rate=learning_rate)
            total_loss += loss

    avg_loss = total_loss / len(X)
    loss_history.append(avg_loss)

    if epoch % 1 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")

# === Save trained model ===
os.makedirs("models", exist_ok=True)
model.save("models/model_weights.npz")
print("Model training complete and saved to models/model_weights.npz")

# === Plot loss curve with log Y axis and normal tick format ===
fig=plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Avg Loss")
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale('log')

# Set Y-axis to show plain numbers (no scientific notation)
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
plt.ticklabel_format(style='plain', axis='y')

plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

