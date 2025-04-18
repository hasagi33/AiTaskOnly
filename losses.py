import numpy as np

class HuberLoss:
    def __init__(self, delta=1.0):
        self.delta = delta
        self.dinputs = None

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * error**2
        #sqrt_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        sqrt_loss=np.sqrt(np.abs(error))          #not linear, sqrt loss
        # root4=np.power(np.abs(error),(1/4))
        #sqrt_loss=np.power(np.abs(error),(1/2))
        return np.mean(np.where(is_small_error, squared_loss, sqrt_loss))

    def backward(self, y_pred, y_true):
        error = y_pred - y_true
        is_small_error = (np.abs(error) <= self.delta)
        dloss = np.where(is_small_error, error, self.delta * np.sign(error))
        self.dinputs = dloss / y_true.shape[0]  # Average over batch


