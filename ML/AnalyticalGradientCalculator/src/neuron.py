import numpy as np
from numpy import ndarray
from src import activations
from src import losses

class neuron:

    def __init__(self) -> None:
        self.x = np.empty([3,1])
        self.dyhat_dz = 0
        self.dL_dyhat = 0

    def forwardPass(self, x: ndarray, weights, bias, y):
        # Calculate a neuron output from input, weight, and bias
        # Calculate how much the weights need to change
        z = np.dot(weights.T, x) + bias
        self.x = x;

        yhat, self.dyhat_dz = activations.reLU(z)

        loss, self.dL_dyhat = losses.squaredLoss(yhat,y)

        return (loss, yhat)

    def computeGradients(self):
        # Calculate the gradient used to adjust weights during backpropagation
        scalar_derivs = self.dL_dyhat * self.dyhat_dz

        # (3,1) = (1,1) * (3*1)
        dL_dw = scalar_derivs * self.x
        dL_db = np.squeeze(scalar_derivs * 1.0)

        return (dL_dw, dL_db)

    

