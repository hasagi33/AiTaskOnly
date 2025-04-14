import numpy as np

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs 
        self.output = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0