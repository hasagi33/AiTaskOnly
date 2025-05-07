from activations import Activation_ReLU
from dense_layer import Layer_Dense
from losses import HuberLoss
from task_input import encode_task
import numpy as np
import os
import json
import sys


class TaskDurationModel:
    def __init__(self, input_size=45):
        # Define layers
        self.layer1 = Layer_Dense(input_size, 64)
        self.activation1 = Activation_ReLU()

        self.layer2 = Layer_Dense(64, 32)
        self.activation2 = Activation_ReLU()

        self.layer3 = Layer_Dense(32, 16)
        self.activation3 = Activation_ReLU()

        self.layer4= Layer_Dense(16,1)

        self.loss_function = HuberLoss(delta=1.0)
    def forward(self, X):
        self.layer1.forward(X)
        self.activation1.forward(self.layer1.output)

        self.layer2.forward(self.activation1.output)
        self.activation2.forward(self.layer2.output)

        self.layer3.forward(self.activation2.output)
        self.activation3.forward(self.layer3.output)

        self.layer4.forward(self.activation3.output)
        return self.layer4.output

    def backward(self, y_pred, y_true):
        self.loss_function.backward(y_pred, y_true)
        self.layer4.backward(self.loss_function.dinputs)

        self.activation3.backward(self.layer4.dinputs)
        self.layer3.backward(self.activation3.dinputs)

        self.activation2.backward(self.layer3.dinputs)
        self.layer2.backward(self.activation2.dinputs)

        self.activation1.backward(self.layer2.dinputs)
        self.layer1.backward(self.activation1.dinputs)

    def train_on_vector(self, X_input, true_duration, epochs=1, learning_rate=0.0001):
        y_true = np.array([[true_duration]])
        for epoch in range(epochs):
            y_pred = self.forward(X_input)
            loss = self.loss_function.forward(y_pred, y_true)
            self.loss_function.backward(y_pred, y_true)
            self.backward(y_pred, y_true)

            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                np.clip(layer.dweights, -1, 1, out=layer.dweights)
                np.clip(layer.dbiases, -1, 1, out=layer.dbiases)
                layer.weights -= learning_rate * layer.dweights
                layer.biases -= learning_rate * layer.dbiases

        return loss


    def predict(self, task_dict, scaler_path="processed/scaler.json"):

        if hasattr(sys, "_MEIPASS"):
            scaler_path = os.path.join(sys._MEIPASS, scaler_path)

        X_input = encode_task(task_dict)

        # === Load training-time scaler
        with open(scaler_path, "r") as f:
            scaler = json.load(f)
        min_vals = np.array(scaler["min"])
        max_vals = np.array(scaler["max"])
        denom = max_vals - min_vals
        denom[denom == 0] = 1e-6

        X_scaled = (X_input - min_vals) / denom

        y_pred = self.forward(X_scaled)
        return y_pred[0][0]

    def save(self, path="models/model_weights.npz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
            w1=self.layer1.weights, b1=self.layer1.biases,
            w2=self.layer2.weights, b2=self.layer2.biases,
            w3=self.layer3.weights, b3=self.layer3.biases,
            w4=self.layer4.weights, b4=self.layer4.biases
        )
        print(f"Model saved to {path}")

    def load(self, path="models/model_weights.npz"):
        if hasattr(sys, "_MEIPASS"):
            path = os.path.join(sys._MEIPASS, path)

        data = np.load(path)
        self.layer1.weights = data['w1']
        self.layer1.biases = data['b1']
        self.layer2.weights = data['w2']
        self.layer2.biases = data['b2']
        self.layer3.weights = data['w3']
        self.layer3.biases = data['b3']
        self.layer4.weights = data['w4']
        self.layer4.biases = data['b4']
        print(f"Model loaded from {path}")