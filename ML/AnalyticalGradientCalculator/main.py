import numpy as np
from src.neuron import neuron
from utils.gradientChecker import checkGradient

# Set fixed seed for reproducibility across test runs
np.random.seed(42)

x = np.array([
    [1.5],
    [-0.8],
    [2.0]
])
w = np.array([
    [0.4],
    [-0.5],
    [0.1]
])
b = 0.25
y = 1.0

singleNeuron = neuron()
loss, yhat = singleNeuron.forwardPass(x,w,b,y)
dL_dw, dL_db = singleNeuron.computeGradients()

print("Calculated Weight Gradient dL_dw:\n", dL_dw)
print("dL_dw Shape:", dL_dw.shape)  # Output: (3, 1) - matches w!

print("Calculated Bias Gradient dL_db:\n", dL_db)

withinRange = checkGradient(x, w, b, y, dL_dw,dL_db)
print("Gradient Check Passed:", withinRange)  # Output: True